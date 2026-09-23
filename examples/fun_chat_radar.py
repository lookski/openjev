#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat radar: paste a group chat log, get everyone's hostility profile.

编写时间: 2026-09-23 21:17:38
脚本功能: parse a "name: message" chat log, score every message with the
          local masked-softmax engine (hostility 0-3, about-to-explode noul,
          yin-yang noul), then aggregate per speaker and predict who blows
          up next.
参数: --file PATH (chat log) | stdin; --model; --top N
输入格式: lines like "张三: 你说的对" ("name: message"); # comments ignored.
输出格式: stdout per-message scores + per-speaker radar + explosion ranking.
依赖: openjev; model weights.
注意事项: entertainment first; 0.6B Chinese ability is limited, which is
          part of the fun. All numbers are raw softmax probabilities.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from openjev.core import LocalJev
from openjev.types import Noul, Score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOCAL_MODEL = os.path.join(REPO, "models", "Qwen3-0.6B")

LINE_RE = re.compile(r"^([^:：]{1,24})[:：]\s*(.+)$")

HOSTILITY = Score(
    instructions="说话人的敌意程度",
    criteria=[
        "没有敌意, 正常交流",
        "有点不爽, 但还在忍",
        "明显不满, 讽刺意味浓",
        "敌意拉满, 就差骂人了",
    ],
)
EXPLODE = Noul(instructions="说这话的人是不是马上就要发火了?")
YINYANG = Noul(instructions="这句话是不是在阴阳怪气?")


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


def main() -> None:
    """Run the radar."""
    parser = argparse.ArgumentParser(description="OpenJev chat radar")
    parser.add_argument("--file", help="chat log file (name: message per line)")
    parser.add_argument(
        "--model",
        default=LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else "Qwen/Qwen3-0.6B",
    )
    parser.add_argument("--top", type=int, default=3, help="show N explosion suspects")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = sys.stdin.read()
    if not text.strip():
        text = (
            "# default demo chat\n"
            "老板: 方案我看完了, 再优化一下\n"
            "小王: 好的老板, 我今晚改完发您\n"
            "老板: 不急, 明天就行\n"
            "小王: 嗯\n"
            "老板: 你这个态度有问题啊\n"
            "小王: 行, 都是我的错\n"
        )

    entries = parse_log(text)
    if not entries:
        raise SystemExit('no "name: message" lines found')

    engine = LocalJev(model_id=args.model, dtype="float32", device="cpu")

    stats = {}
    print("scanning %d messages..." % len(entries))
    for name, msg in entries:
        result = engine.system_one(
            msg, {"hostility": HOSTILITY, "explode": EXPLODE, "yinyang": YINYANG}
        )
        a = result["answers"]
        probs = a["hostility"]["probabilities"]
        h = sum(int(lvl) * p for lvl, p in probs.items())
        ex = a["explode"]["noul"]
        yy = a["yinyang"]["noul"]
        s = stats.setdefault(
            name, {"n": 0, "hostility": 0.0, "explode": 0.0, "yinyang": 0.0,
                   "peak": 0.0, "peak_msg": ""}
        )
        s["n"] += 1
        s["hostility"] += h
        s["explode"] += ex
        s["yinyang"] += yy
        if ex - s["peak"] > 1e-9:
            s["peak"] = ex
            s["peak_msg"] = msg
        print('  %s: "%s"' % (name, msg))
        print("    hostility %.3f  explode %s %.3f  yinyang %s %.3f"
              % (h, bar(ex), ex, bar(yy), yy))

    print("")
    print("===== per-speaker radar =====")
    rows = []
    for name, s in stats.items():
        rows.append((name, s["hostility"] / s["n"], s["explode"] / s["n"],
                     s["yinyang"] / s["n"], s["peak"], s["peak_msg"], s["n"]))
    rows.sort(key=lambda r: r[2], reverse=True)
    for name, h, ex, yy, peak, peak_msg, n in rows:
        print("  %-8s msgs=%-3d hostility=%.3f explode=%.3f yinyang=%.3f"
              % (name, n, h, ex, yy))

    print("")
    print("===== next to explode (top %d) =====" % args.top)
    for i, (name, h, ex, yy, peak, peak_msg, n) in enumerate(rows[:args.top], 1):
        print('  %d. %s  peak-explode %.3f on: "%s"' % (i, name, peak, peak_msg))
    print("(all numbers are raw softmax probabilities; entertainment first)")


if __name__ == "__main__":
    main()
