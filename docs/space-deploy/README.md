---
title: OpenJev 阴阳怪气检测器
emoji: 🙃
colorFrom: gray
colorTo: green
sdk: static
pinned: false
license: mit
short_description: 浏览器本地推理的阴阳怪气/敌意/语气三合一检测器 (Qwen3-0.6B fp16)
---

# OpenJev 阴阳怪气检测器 (浏览器版)

纯前端实现的 OpenJev 掩码 logit 决策引擎, 全部推理在访客浏览器内完成 (WebGPU 不可用时回退 WASM), 不依赖任何后端。

- 模型: q4f16 量化档 (0.5GB, 本仓库自托管, WASM 快速模式) / fp16 档 (1.2GB, WebGPU 高精度模式, 从 onnx-community/Qwen3-0.6B-ONNX 加载)
- 引擎: transformers.js 4.3 (onnxruntime-web)
- 决策逻辑: 与 Python 版 `openjev/core.py` 的 masked softmax 一致 (移植验证最大绝对误差 0.0032)
- 题目: 与 `openjev/webapp.py` QUESTIONS 一致 (阴阳怪气判定 / 敌意 0-3 打分 / 语气三分类)

打开页面首次需下载约 1.2GB 权重, 之后走浏览器 Cache API。

源码: https://github.com/lookski/openjev
