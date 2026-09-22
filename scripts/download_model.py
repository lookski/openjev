#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the recommended decision-engine model snapshot into models/.

编写时间: 2026-09-22 19:02:41
脚本功能: fetch a HF model snapshot (default Qwen/Qwen3-0.6B) into
          models/<name> so OpenJev runs fully offline.
参数: --model HF_ID (default Qwen/Qwen3-0.6B), --out DIR (default models/<basename>)
输入格式: command line arguments.
输出格式: prints the local snapshot path when done.
依赖: huggingface_hub.
注意事项: for mainland China the script defaults HF_ENDPOINT to
          https://hf-mirror.com and disables the Xet backend; set the real
          HF_ENDPOINT env var to override.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    """Download the snapshot."""
    parser = argparse.ArgumentParser(description="Download an OpenJev model")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = args.out or os.path.join("models", args.model.split("/")[-1])
    # plain-HTTP download backend (Xet 401s against mirrors) + CN mirror
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from huggingface_hub import snapshot_download

    path = snapshot_download(args.model, local_dir=out_dir)
    print("DOWNLOAD_DONE", path)


if __name__ == "__main__":
    main()
