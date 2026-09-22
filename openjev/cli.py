#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line interface: openjev ask / openjev serve / openjev models.

编写时间: 2026-09-22 18:29:43
脚本功能: single entry `python -m openjev.cli`:
          - ask    : one state + typed questions answered locally;
          - serve  : start the local systemone HTTP server;
          - models : print recommended fast-TTFT local models.
参数:
  ask --state TEXT | --state-file PATH --question JSON [--question JSON ...]
        [--model ID] [--dtype D] [--device D] [--json]
        --question spec is a JSON object {id:{type,instructions,criteria}}.
  serve [--host H] [--port P] [--model ID] [--dtype D] [--device D] [--preload]
  models
输入格式: command line arguments; JSON question specs.
输出格式: human-readable answer table, or raw JSON with --json.
依赖: openjev.core, openjev.server, json, argparse.
注意事项: HF_ENDPOINT=https://hf-mirror.com speeds up model downloads in CN;
          the first run downloads the model weights to the HF cache.
"""

from __future__ import annotations

import argparse
import json
import sys

from openjev.core import DEFAULT_MODEL_ID, LocalJev
from openjev.server import main as serve_main

RECOMMENDED_MODELS = [
    ("Qwen/Qwen3-0.6B", "0.6B", "fastest CPU TTFT, default; 32k context"),
    ("Qwen/Qwen3-1.7B", "1.7B", "better accuracy, still low TTFT"),
    ("Qwen/Qwen3-4B-Instruct-2507", "4B", "best accuracy, needs GPU or patience"),
    ("Qwen/Qwen3-0.6B", "0.6B", "alias row kept for quick copy-paste"),
]


def _print_answer(answers: dict) -> None:
    """Render answers as a compact human-readable block."""
    for qid, ans in answers.items():
        line = "%s: " % qid
        if ans["type"] == "noul":
            line += "noul=%.4f" % ans["noul"]
        elif ans["type"] == "choice":
            line += "%s (p=%.4f, confidence=%.4f)" % (
                ans["choice"],
                ans["probabilities"][ans["choice"]],
                ans["confidence"],
            )
        elif ans["type"] == "score":
            line += "%.4f (confidence=%.4f)" % (ans["score"], ans["confidence"])
        print(line)


def cmd_ask(args: argparse.Namespace) -> int:
    """Handle `openjev ask`."""
    state = args.state
    if state is None and args.state_file:
        with open(args.state_file, "r", encoding="utf-8") as handle:
            state = handle.read()
    if state is None:
        state = sys.stdin.read()

    questions = {}
    for spec in args.question:
        one = json.loads(spec)
        questions.update(one)

    engine = LocalJev(
        model_id=args.model,
        dtype=args.dtype,
        device=args.device,
    )
    result = engine.system_one(state, questions)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("model: %s" % result["model"])
        _print_answer(result["answers"])
        u = result["usage"]
        print(
            "usage: forward_passes=%d input_tokens=%d latency_ms=%.1f"
            % (u["forward_passes"], u["input_tokens"], u["latency_ms_total"])
        )
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    """Handle `openjev models`."""
    print("recommended local models (fast first-token, decision-grade):")
    for mid, size, note in RECOMMENDED_MODELS:
        print("  %-34s %-5s %s" % (mid, size, note))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Handle `openjev serve`; forward flags to the server entry."""
    argv = ["--host", args.host, "--port", str(args.port), "--model", args.model,
            "--dtype", args.dtype]
    if args.device:
        argv += ["--device", args.device]
    if args.preload:
        argv.append("--preload")
    serve_main(argv)
    return 0


def main() -> None:
    """Argparse entry point."""
    parser = argparse.ArgumentParser(prog="openjev", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="answer typed questions locally")
    p_ask.add_argument("--state", help="state text (or use --state-file / stdin)")
    p_ask.add_argument("--state-file")
    p_ask.add_argument(
        "--question",
        action="append",
        required=True,
        metavar="JSON",
        help='question spec, e.g. \'{"urgent":{"type":"noul","instructions":"..."}}\'',
    )
    p_ask.add_argument("--model", default=DEFAULT_MODEL_ID)
    p_ask.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p_ask.add_argument("--device", default=None)
    p_ask.add_argument("--json", action="store_true", help="print raw JSON")
    p_ask.set_defaults(func=cmd_ask)

    p_serve = sub.add_parser("serve", help="start the local systemone server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8771)
    p_serve.add_argument("--model", default=DEFAULT_MODEL_ID)
    p_serve.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p_serve.add_argument("--device", default=None)
    p_serve.add_argument("--preload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_models = sub.add_parser("models", help="list recommended models")
    p_models.set_defaults(func=cmd_models)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
