#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crush radar: paste your chat log with someone you are pursuing, get an honest
read on their interest level and whether your next move is 冲/稳/缓/停.

编写时间: 2026-09-30 15:24:32
脚本功能: parse a two-person "name: message" chat log, score every message
          with the local masked-softmax engine (interest 0-3 score, warmth
          noul, perfunctory noul), aggregate per speaker, compute the
          interest trend across the conversation, then print one of four
          move recommendations (冲 / 稳 / 缓 / 停) derived from those
          machine-computed numbers only.
参数: --file PATH (chat log) | stdin; --model; --you NAME; --them NAME
输入格式: lines like "我: 周末要一起去看展吗" ("name: message");
          # comments ignored; speaker names default to heaviest first two.
输出格式: stdout per-message scores, per-speaker radar, trend split,
          and a single move recommendation with its numeric justification.
依赖: openjev; model weights (default models/Qwen3-0.6B or HF Qwen/Qwen3-0.6B).
注意事项: entertainment first; the 0.6B model reads Chinese messages at
          toy accuracy, treat output as a party trick, not relationship
          advice. All numbers are raw softmax probabilities, no parsing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openjev.core import LocalJev
from openjev.types import Noul, Score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOCAL_MODEL = os.path.join(REPO, "models", "Qwen3-0.6B")

LINE_RE = re.compile(r"^([^:：]{1,24})[:：]\s*(.+)$")

INTEREST = Score(
    instructions="说这句话的人对当前聊天话题的兴趣和投入程度",
    criteria=[
        "完全不想聊, 想结束对话",
        "礼貌应付, 能少说就少说",
        "正常聊天, 有问有答",
        "热情投入, 主动接话题",
    ],
)
WARMTH = Noul(instructions="这句话有没有传递好感和亲近的信号?")
PERFUNCTORY = Noul(instructions="这句话是不是在敷衍对方?")

DEFAULT_DEMO = (
    "# default demo chat (classic low-interest pattern)\n"
    "我: 周末要不要一起去看展? 听说挺不错的\n"
    "她: 我看看有没有时间吧\n"
    "我: 那我先把票买好, 到时候叫你\n"
    "她: 嗯\n"
    "我: 最近有部电影评分很高, 要不要一起看?\n"
    "她: 哦, 我对电影不太感兴趣\n"
    "我: 那你周末一般喜欢干嘛呀\n"
    "她: 没干嘛, 就是挺忙的\n"
)


def bar(p: float, width: int = 20) -> str:
    """ASCII bar from a real probability."""
    filled = max(1, round(p * width))
    return "#" * filled + "." * (width - filled)


def parse_log(text: str):
    """Parse 'name: message' lines; return [(name, message), ...]."""
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if m:
            entries.append((m.group(1).strip(), m.group(2).strip()))
    return entries


def score_log(engine, entries):
    """Score every message; return list of dicts with all metrics."""
    rows = []
    print("scanning %d messages..." % len(entries))
    for name, msg in entries:
        result = engine.system_one(
            msg, {"interest": INTEREST, "warmth": WARMTH, "perfunctory": PERFUNCTORY}
        )
        a = result["answers"]
        probs = a["interest"]["probabilities"]
        i_val = sum(int(lvl) * p for lvl, p in probs.items())
        w = a["warmth"]["noul"]
        f = a["perfunctory"]["noul"]
        rows.append(
            {"name": name, "msg": msg, "interest": i_val, "warmth": w, "perfunctory": f}
        )
        print('  %s: "%s"' % (name, msg))
        print(
            "    interest %.3f  warmth %s %.3f  perfunctory %s %.3f"
            % (i_val, bar(w), w, bar(f), f)
        )
    return rows


def aggregate(rows, top_n=2):
    """Per-speaker means; return ordered stats list (most messages first)."""
    stats = {}
    for r in rows:
        s = stats.setdefault(
            r["name"], {"n": 0, "interest": 0.0, "warmth": 0.0, "perfunctory": 0.0}
        )
        s["n"] += 1
        s["interest"] += r["interest"]
        s["warmth"] += r["warmth"]
        s["perfunctory"] += r["perfunctory"]
    for s in stats.values():
        for k in ("interest", "warmth", "perfunctory"):
            s[k] /= s["n"]
    return sorted(stats.items(), key=lambda kv: kv[1]["n"], reverse=True)[:top_n]


def trend(rows, them):
    """Split their messages into halves; return (first_mean, second_mean)."""
    theirs = [r["interest"] for r in rows if r["name"] == them]
    if len(theirs) < 2:
        return theirs[0] if theirs else 0.0, theirs[0] if theirs else 0.0
    mid = len(theirs) // 2
    first = sum(theirs[:mid]) / max(1, mid)
    second = sum(theirs[mid:]) / max(1, len(theirs) - mid)
    return first, second


def recommend(mine, them, first, second):
    """Derive one move from RELATIVE machine-computed numbers.

    The 0.6B model compresses absolute Chinese-message scores (it rates
    almost everything perfunctory ~0.9, like the yin-yang detector rates
    everything yin-yang). So the verdict uses the gap between speakers and
    the trend direction, never absolute thresholds.
    """
    gap = them["interest"] - mine["interest"]
    falling = second < first - 0.05
    rising = second > first + 0.05
    if falling and gap < 0:
        return "停", (
            "对方兴趣趋势下降 (%.2f -> %.2f) 且已低于你 (%.2f vs %.2f): "
            "这段对话正在失去能量, 不要再主动追加"
            % (first, second, them["interest"], mine["interest"])
        )
    if rising and gap > 0:
        return "冲", (
            "对方兴趣上升 (%.2f -> %.2f) 且比你更投入 (差值 +%.2f): "
            "热度在对方那边, 直接约线下, 别只停在文字"
            % (first, second, gap)
        )
    if gap < -0.1:
        return "缓", (
            "对方投入明显低于你 (%.2f vs %.2f, 差值 %.2f): "
            "降低主动频率, 把话题主导权交回去观察"
            % (them["interest"], mine["interest"], gap)
        )
    if rising:
        return "稳", (
            "趋势上升 (%.2f -> %.2f) 但对方投入未超过你: 正常推进, 下一段再看"
            % (first, second)
        )
    return "稳", (
        "趋势平稳 (%.2f -> %.2f), 双方投入接近 (%.2f vs %.2f): "
        "正常聊, 用下一段对话的趋势决定"
        % (first, second, them["interest"], mine["interest"])
    )


def main() -> None:
    """Run the crush radar."""
    parser = argparse.ArgumentParser(description="OpenJev crush radar")
    parser.add_argument("--file", help="chat log file (name: message per line)")
    parser.add_argument(
        "--model",
        default=LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else "Qwen/Qwen3-0.6B",
    )
    parser.add_argument("--you", default=None, help="your name in the log")
    parser.add_argument("--them", default=None, help="their name in the log")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = sys.stdin.read()
    if not text.strip():
        text = DEFAULT_DEMO

    entries = parse_log(text)
    if not entries:
        raise SystemExit('no "name: message" lines found')

    engine = LocalJev(model_id=args.model, dtype="float32", device="cpu")
    rows = score_log(engine, entries)

    ranked = aggregate(rows)
    you_name = args.you or ranked[0][0]
    them_name = args.them or (ranked[1][0] if len(ranked) > 1 else ranked[0][0])
    them_stats = dict(ranked)[them_name]

    first, second = trend(rows, them_name)

    print("")
    print("===== crush radar (%s -> %s) =====" % (you_name, them_name))
    for name, s in ranked:
        tag = " <- target" if name == them_name else ""
        print(
            "  %-8s msgs=%-3d interest=%.2f/3 warmth=%.3f perfunctory=%.3f%s"
            % (name, s["n"], s["interest"], s["warmth"], s["perfunctory"], tag)
        )
    print(
        "  interest trend (%s): %.3f -> %.3f (%s)"
        % (them_name, first, second,
           "falling" if second < first - 0.1 else ("rising" if second > first + 0.1 else "flat"))
    )

    move, why = recommend(dict(ranked)[you_name], them_stats, first, second)
    print("")
    print("===== 建议: %s =====" % move)
    print("  %s" % why)
    print("  (numbers above are raw masked-softmax outputs; toy model, toy verdict)")


if __name__ == "__main__":
    main()
