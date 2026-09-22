#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library usage example: confidence-gated routing with a local decision engine.

编写时间: 2026-09-22 18:41:07
脚本功能: demonstrate the official Jev "Confidence-Gated Routing" pattern:
          low confidence -> human review, safe action -> auto, risky action
          -> high-confidence gate. Uses the local engine, no API key.
参数: --model PATH_OR_HF_ID (default models/Qwen3-0.6B if present else HF id)
输入格式: hardcoded example states.
输出格式: stdout routing decisions with probabilities.
依赖: openjev; model weights.
注意事项: run from the repo root: python examples/example_api.py
"""

from __future__ import annotations

import argparse
import os

from openjev.core import LocalJev
from openjev.types import Choice

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODEL = os.path.join(HERE, "models", "Qwen3-0.6B")


def route(user_message: str, engine: LocalJev) -> None:
    """Confidence-gated routing on one user message."""
    result = engine.system_one(
        user_message,
        {
            "action": Choice(
                instructions="What does the user want to do",
                criteria={
                    "check_balance": "View account balance",
                    "approve_transfer": "Approve a pending withdrawal",
                    "support": "Get help with something else",
                },
            ),
        },
    )
    ans = result["answers"]["action"]
    print("message : %s" % user_message)
    print("answer  : %s  p=%.4f  confidence=%.4f" % (
        ans["choice"],
        ans["probabilities"][ans["choice"]],
        ans["confidence"],
    ))
    if ans["confidence"] < 0.5:
        print("route   : -> route_to_human()  (low confidence)")
    elif ans["choice"] == "check_balance":
        print("route   : -> show_balance()    (safe action)")
    elif ans["choice"] == "approve_transfer":
        if ans["confidence"] > 0.9:
            print("route   : -> confirm_then_execute()  (high confidence)")
        else:
            print("route   : -> ask_user_to_confirm()")
    print()


def main() -> None:
    """Run the routing demo."""
    parser = argparse.ArgumentParser(description="OpenJev routing example")
    parser.add_argument(
        "--model",
        default=LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else "Qwen/Qwen3-0.6B",
    )
    args = parser.parse_args()

    engine = LocalJev(model_id=args.model, dtype="float32", device="cpu")
    print("model: %s\n" % engine.model_id)

    route("What's my current balance?", engine)
    route("I approve the $500 withdrawal to my savings, please proceed.", engine)
    route("Hey so I was wondering maybe about that thing from yesterday?", engine)


if __name__ == "__main__":
    main()
