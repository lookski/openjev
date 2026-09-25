#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编写时间: 2026-09-25 09:58:00
脚本功能: 把 tmp_jscheck/space (浏览器版 OpenJev, 613MB: q4f16 权重+代码+运行时)
          上传到 HF Static Space LinRin0306/openjev-detector.
          注: 首次上传 1.8GB 超 Space 1GB 存储限额 (403), 改为仅含 q4f16;
          fp16 高精度档在 app 内从 onnx-community/Qwen3-0.6B-ONNX 在线加载.
参数: 无
输入格式: 环境变量 HF_TOKEN (上传前 export, 不落盘)
输出格式: stdout: 上传进度与完成信息
依赖: huggingface_hub
注意事项: 需代理 127.0.0.1:7892; upload_folder 走多线程, 1.8GB 约 5-15 分钟;
          commit_message 标注浏览器版首推.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
assert os.environ["HF_TOKEN"], "export HF_TOKEN first"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7892"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7892"

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
REPO = "LinRin0306/openjev-detector"
LOCAL = r"D:\starPlan\1.jev\tmp_jscheck\space"

print("uploading", LOCAL, "->", REPO, "...", flush=True)
res = api.upload_folder(
    folder_path=LOCAL,
    repo_id=REPO,
    repo_type="space",
    commit_message="browser-only detector: transformers.js fp16 (webgpu) + q4f16 (wasm fallback), self-hosted weights",
    commit_description="OpenJev masked-softmax port verified against Python golden data (max abs err 0.0032 fp16); self-hosted ort jsep runtime; static space no backend",
)
print("upload done:", res, flush=True)
print("DONE")
