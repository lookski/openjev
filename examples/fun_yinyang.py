#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yin-yang detector: a fun demo of OpenJev on Chinese internet sarcasm.

编写时间: 2026-09-23 20:57:29
脚本功能: score Chinese messages for sarcasm ("yin-yang"), hostility level,
          and the sender's real vibe, using the local masked-softmax engine;
          print probability bars. The number IS the joke.
参数: --model PATH_OR_HF_ID (default models/Qwen3-0.6B or HF id), --text TEXT
输入格式: Chinese text messages (hardcoded corpus unless --text).
输出格式: stdout probability bars per message.
依赖: openjev; model weights.
注意事项: Qwen3-0.6B is a small model; treat the output as entertainment
          first, benchmark second.
"""

from __future__ import annotations

import argparse
import os

from openjev.core import LocalJev
from openjev.types import Choice, Noul, Score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOCAL_MODEL = os.path.join(REPO, "models", "Qwen3-0.6B")

CORPUS = [
    "哦",
    "好的，都可以，你决定就好",
    "那可真是太厉害了呢",
    "行，是我错了行了吧",
    "我不是针对你，我是说在座的各位",
    "你这个方案很有意思，我们再聊聊?",
    "6",
]

QUESTIONS = {
    "yin_yang": Noul(
        instructions="这条消息是不是在阴阳怪气 (表面客气实际讽刺)?",
    ),
    "hostility": Score(
        instructions="说话人的敌意程度",
        criteria=[
            "没有敌意, 正常交流",
            "有点不爽, 但还在忍",
            "明显不满, 讽刺意味浓",
            "敌意拉满, 就差骂人了",
        ],
    ),
    "vibe": Choice(
        instructions="这条消息的真实语气更像",
        criteria={
            "sincere": "真诚的, 表面和实际一致",
            "passive_aggressive": "被动攻击, 用客气包裹不满",
            "pure_sarcasm": "纯讽刺, 明摆着阴阳怪气",
        },
    ),
}


def bar(p: float, width: int = 22) -> str:
    """ASCII bar computed from the real probability."""
    filled = max(1, round(p * width))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    """Run the detector over the corpus."""
    parser = argparse.ArgumentParser(description="OpenJev yin-yang detector")
    parser.add_argument(
        "--model",
        default=LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else "Qwen/Qwen3-0.6B",
    )
    parser.add_argument("--text", help="score one custom message instead of the corpus")
    args = parser.parse_args()

    engine = LocalJev(model_id=args.model, dtype="float32", device="cpu")

    messages = [args.text] if args.text else CORPUS
    for msg in messages:
        result = engine.system_one(msg, QUESTIONS)
        a = result["answers"]
        print('"%s"' % msg)
        print("  yin-yang  %s %.4f" % (bar(a["yin_yang"]["noul"]), a["yin_yang"]["noul"]))
        for lvl, p in a["hostility"]["probabilities"].items():
            print("  敌意 L%s    %s %.4f" % (lvl, bar(p), p))
        for name, p in a["vibe"]["probabilities"].items():
            mark = "*" if name == a["vibe"]["choice"] else " "
            print("  vibe %s %-20s %s %.4f" % (mark, name, bar(p), p))
        print()


if __name__ == "__main__":
    main()
