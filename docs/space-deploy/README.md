---
title: OpenJev 阴阳怪气检测器
emoji: 🙃
colorFrom: gray
colorTo: green
sdk: static
pinned: false
license: mit
short_description: 浏览器本地推理的阴阳怪气/敌意/语气三合一检测器 (Qwen3-0.6B)
---

# OpenJev 阴阳怪气检测器 (浏览器版)

粘贴一条消息, 三秒钟告诉你: **是不是阴阳怪气 · 敌意几级 · 真实语气是什么**。

纯前端实现, 推理全程在**你的设备**上跑 (WebGPU 或 WASM), 消息**不上传任何服务器**。

## 它是怎么工作的

不用生成文本, 不用关键词匹配 —— 直接读 Qwen3-0.6B 的**原始 logit 概率** (masked-softmax): 一次前向传播, 掩码答案 token, softmax 得到校准概率。类型错误在数学上不可能发生。

| 检测项 | 输出 |
|---|---|
| 阴阳怪气判定 | Yes/No + 概率 |
| 敌意程度 | 0-3 级连续评分 + 分布 |
| 真实语气 | 真诚 / 被动攻击 / 纯讽刺 + 概率 |

## 技术细节

- 引擎: [OpenJev](https://github.com/lookski/openjev) masked-softmax 决策核心的 JS 移植
- 模型: [onnx-community/Qwen3-0.6B-ONNX](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX)
- 双模式: 有 WebGPU → fp16 高精度档 (1.2GB, 与 Python 参考实现最大误差 0.0032); 无 → q4f16 快速档 (0.5GB, 本仓库自托管)
- 运行时: transformers.js 4.3 + onnxruntime-web (jsep, 自托管无 CDN 依赖)
- 移植验证: prompt 逐字节一致 + 概率 ALL-PASS (见 GitHub 仓库 docs/)

## 使用提示

- 首次打开需下载模型权重 (快速模式约 500MB / 高精度约 1.2GB), 之后走浏览器缓存秒开
- 快速模式下概率有量化偏差, 追求准确请用最新 Chrome/Edge (WebGPU)
- 结果为模型原生概率输出, 仅供娱乐参考 — 但它看「嗯」的眼神确实很毒

## 相关链接

- 源码与 Python 版: https://github.com/lookski/openjev
- 本地网页版 (`openjev-web`, 无需下载权重): `pip install -e . && openjev-web`
