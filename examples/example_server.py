#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server usage example: start the local systemone server and hit it with HTTP.

编写时间: 2026-09-22 18:41:07
脚本功能: spawn `openjev.server` in a subprocess on a free port, wait for
          /health, POST one systemone request (noul + score + choice),
          print the JSON response, then shut down.
参数: --model PATH_OR_HF_ID
输入格式: none.
输出格式: stdout, the raw JSON response of the local server.
依赖: openjev; model weights; stdlib urllib.
注意事项: python examples/example_server.py; the server binds 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    """Start server, call it once, stop it."""
    parser = argparse.ArgumentParser(description="OpenJev server example")
    parser.add_argument(
        "--model",
        default=os.path.join(HERE, "models", "Qwen3-0.6B")
        if os.path.isdir(os.path.join(HERE, "models", "Qwen3-0.6B"))
        else "Qwen/Qwen3-0.6B",
    )
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "openjev.server",
            "--host", "127.0.0.1",
            "--port", str(args.port),
            "--model", args.model,
        ],
        cwd=HERE,
    )
    base = "http://127.0.0.1:%d" % args.port
    try:
        # wait for the socket
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not come up")

        payload = {
            "state": (
                "Subject: cannot login since this morning. "
                "I typed the password 10 times, still invalid. "
                "I need to file taxes today!"
            ),
            "questions": {
                "urgent": {
                    "type": "noul",
                    "instructions": "Does this message express urgency?",
                },
                "category": {
                    "type": "choice",
                    "instructions": "Which category fits best",
                    "criteria": {
                        "auth": "Login, password, 2FA problems",
                        "billing": "Payments and invoices",
                        "feature": "Feature requests and feedback",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is this report",
                    "criteria": ["cosmetic", "workaround exists", "blocked"],
                },
            },
        }
        req = urllib.request.Request(
            base + "/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            print(json.dumps(json.loads(resp.read()), indent=2, ensure_ascii=False))
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
