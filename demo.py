#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end demo: local LLM as a Jev-style System One decision engine.

Runs the canonical TypeSafe Jev quickstart example (Stripe support ticket)
fully offline on a small local model, printing raw masked-softmax
probabilities for Choice / Score / Noul questions.

编写时间: 2026-09-22 18:38:54
脚本功能: load Qwen3-0.6B locally, ask 3 typed questions about the ticket,
          print the answer dicts (choice / score / noul with probabilities).
参数: --model PATH_OR_HF_ID (default local models/Qwen3-0.6B, fallback HF id)
输入格式: none (hardcoded demo state and questions).
输出格式: stdout, JSON answer dicts per question + usage line.
依赖: openjev (torch, transformers); model weights in models/ or HF cache.
注意事项: first run downloads ~1.5 GB if the local snapshot is missing;
          CPU-only float32 latency is around 1-3 s per question on desktop CPUs.
"""

from __future__ import annotations

import argparse
import json
import os

from openjev.core import LocalJev
from openjev.types import Choice, Noul, Score

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL = os.path.join(HERE, "models", "Qwen3-0.6B")
HF_MODEL = "Qwen/Qwen3-0.6B"

DEMO_STATE = (
    "Hi, I have been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing. I am losing sales. Please help ASAP."
)


def main() -> None:
    """Run the demo."""
    parser = argparse.ArgumentParser(description="OpenJev end-to-end demo")
    parser.add_argument(
        "--model",
        default=LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else HF_MODEL,
        help="local model dir or HF id",
    )
    parser.add_argument("--json", action="store_true", help="print raw JSON only")
    args = parser.parse_args()

    engine = LocalJev(model_id=args.model, dtype="float32", device="cpu")

    questions = {
        "department": Choice(
            instructions="Which team should handle this",
            criteria={
                "billing": "Payments, invoices, refunds",
                "technical": "Integration errors, API failures, bugs",
                "sales": "Pricing questions, upgrades, new purchases",
            },
        ),
        "frustration": Score(
            instructions="How frustrated is the customer",
            criteria=[
                "Neutral or polite, no complaint",
                "Annoyed, mentions a problem but stays civil",
                "Angry, threatens to leave or uses hostile language",
            ],
        ),
        "is_urgent": Noul(instructions="Does this message express urgency?"),
    }

    result = engine.system_one(DEMO_STATE, questions)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("state: %s" % DEMO_STATE)
    print("model: %s" % result["model"])
    for qid, ans in result["answers"].items():
        print("\n[%s] (%s)" % (qid, ans["type"]))
        if ans["type"] == "choice":
            for name, p in ans["probabilities"].items():
                print("  %-10s %.4f" % (name, p))
            print("  -> choice=%s confidence=%.4f" % (ans["choice"], ans["confidence"]))
        elif ans["type"] == "score":
            for lvl, p in ans["probabilities"].items():
                print("  level %s: %.4f" % (lvl, p))
            print("  -> score=%.4f confidence=%.4f" % (ans["score"], ans["confidence"]))
        elif ans["type"] == "noul":
            print("  Yes: %.4f" % ans["noul"])
            print("  No:  %.4f" % (1.0 - ans["noul"]))

    u = result["usage"]
    print(
        "\nusage: forward_passes=%d input_tokens=%d latency_ms=%.1f"
        % (u["forward_passes"], u["input_tokens"], u["latency_ms_total"])
    )


if __name__ == "__main__":
    main()
