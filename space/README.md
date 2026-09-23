---
title: OpenJev 阴阳怪气鉴定器
emoji: 🕵️
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Paste a message, get raw-softmax sarcasm probabilities. 100% local weights.
---

# OpenJev yin-yang detector (Space)

A Hugging Face Space wrapper around `openjev.webapp`. The Space builds the
CPU-torch image, downloads Qwen3-0.6B at first start into a persistent
volume, and serves the single-page detector on the Space's public URL.

Create the Space with `sdk: docker` and push this repo's `space/` folder as
the Space repository root (see README "Deploy your own" section).
