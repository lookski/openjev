# OpenJev

[English](README.md) | 简体中文

**把任意本地 LLM 变成 [Jev](https://jevai.net) 式的 "System One" 决策模型, 100% 跑在你自己的机器上.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/lookski/openjev/actions/workflows/ci.yml/badge.svg)](https://github.com/lookski/openjev/actions/workflows/ci.yml)
[![No API Key](https://img.shields.io/badge/API%20key-none-success)](#快速开始)

Jev (TypeSafe AI, 2026 年 9 月) 让 "决策模型" 一词刷屏: 发送一段状态 (state) + 若干类型化问题, 返回**带校准概率的类型安全答案** —— 不生成文本, 没有幻觉, 70–500 ms 延迟.

**OpenJev 用纯本地方案复刻了同一个思路.** 任意小参数 LLM 通过**掩码 logit softmax** 变成决策引擎: 一次前向传播, 掩掉答案 token 之外的 logits, softmax 得到原始概率. 零 API 费用, 数据不出机器, 类型错误在数学上不可能发生.

```
$ python demo.py

state: Hi, I have been trying to connect my Stripe account for 3 days ...
model: openjev/models/Qwen3-0.6B

[department] (choice)
  billing    0.0010
  technical  0.9990
  sales      0.0000
  -> choice=technical confidence=0.9985

[frustration] (score)
  level 0: 0.9041
  level 1: 0.0619
  level 2: 0.0339
  -> score=0.1298 confidence=0.8562

[is_urgent] (noul)
  Yes: 0.8733
  No:  0.1267

usage: forward_passes=3 input_tokens=338 latency_ms=1234.4
```

*(实测输出, CPU fp32, Qwen3-0.6B, 你的硬件与版本下数字可能略有差异)*

![OpenJev 演示](assets/demo.svg)

## 原理

LLM 其实已经 "知道" 答案, 问题出在**解码方式**: 生成文本又慢又不稳定, 还没有类型. OpenJev 完全不解码, 它直接读取第一个答案位置的**原始 logits**:

| | Jev (云端) | OpenJev (本地) |
|---|---|---|
| 机制 | 并行采样器 (闭源) | 掩码 logit softmax (开源) |
| 输出 | 类型安全 + 概率 | 类型安全 + 概率 |
| 延迟 | 70–500 ms | **约 0.1–3 s** (单次前向, 无解码) |
| 成本 | $0.042 / 百万输入 token | **$0** |
| 隐私 | 数据出机器 | **100% 本地** |
| 权重 | 闭源 | 任意开源 LLM (Qwen / Llama / Phi / ...) |
| 上下文 | 64K | 取决于模型 (Qwen3: 32K) |
| 幻觉 | 设计上不可能 | 设计上不可能 |

这些概率是掩码 logits 的**原始 softmax 值** —— 不是采样出来的, 不是提示词逼出来的, 也不是从文本里解析出来的. 这就是模型原生的, 诚实的置信度.

## 快速开始

### 傻瓜模式 (零代码, 推荐先跑这个)

```bash
pip install -e .
openjev-easy          # 或者: python -m openjev.easy_cli
```

向导自动帮你选大脑: 自动探测本机正在运行的 **Ollama / LM Studio / vLLM / llama.cpp** 服务, 或者输入 **OpenAI / OpenRouter / 官方 Jev** 的 API key (隐藏输入), 冒烟测试通过后进入交互界面, 粘贴任意文本就返回类型化概率:

```
You> The server is down, we are losing money, fix it NOW.

  intent       #....................... 0.0001
  intent     * ######################## 0.9999
  intent       #....................... 0.0000
  intent       #....................... 0.0000
  -> choice=complaint (confidence 0.9998)

  urgent       Yes #######################. 0.9520
               No  #....................... 0.0480

  sentiment    level 0 #....................... 0.0016
  sentiment    level 1 ##################...... 0.7488
  sentiment    level 2 ######.................. 0.2496
  -> score=1.2480 (confidence 0.6233)
```

*(实测输出, 内置引擎, Qwen3-0.6B)*

非交互单次调用: `openjev-easy --backend ollama --model qwen3:0.6b --once "some text"`. 所有后端 —— 包括官方 Jev 云 API (`--backend jev`) —— 接口完全一致, 一个参数就能 A/B 对比.

### 完整引擎 (掩码 softmax, 默认深度路径)

```bash
pip install -e .
python demo.py
```

首次运行会从 Hugging Face 下载 Qwen3-0.6B (约 1.5 GB). 国内环境: `export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`, 或直接用 `python scripts/download_model.py` (已内置镜像与 Xet 开关).

### 一行代码 (库 API)

```python
from openjev import Choice, LocalJev, Noul, Score

engine = LocalJev("Qwen/Qwen3-0.6B")           # 或本地模型路径
result = engine.system_one(
    "Hi, I have been trying to connect my Stripe account for 3 days ...",
    {
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
            criteria=["Neutral", "Annoyed", "Angry"],
        ),
        "is_urgent": Noul(instructions="Does this message express urgency?"),
    },
)
print(result["answers"]["department"])   # {'type': 'choice', 'choice': 'technical', ...}
```

### 本地 HTTP 服务 (与 Jev 官方接口同构)

```bash
openjev serve --port 8771
# POST http://127.0.0.1:8771/v1/systemone
curl -s http://127.0.0.1:8771/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "The server is down, we are losing money, fix it NOW.",
    "questions": {
      "urgent":   {"type": "noul",   "instructions": "Does this message express urgency?"},
      "severity": {"type": "score",  "instructions": "How severe is this",
                   "criteria": ["cosmetic", "degraded", "outage"]},
      "route":    {"type": "choice", "instructions": "Which team",
                   "criteria": {"billing": "invoices", "ops": "infrastructure"}}
    }
  }'
```

响应结构与官方 API 对齐 (answers 按问题 id 键控, noul 没有 confidence 字段, score 是等级下标的加权均值, `confidence = (K·max_p − 1)/(K − 1)`).

### 双后端, 同一接口 (本地 softmax ⇄ 官方 Jev API)

```bash
export TYPESAFE_API_KEY=sk-...                       # 可选的云端后端
openjev ask --backend jev --state "..." \
  --question '{"urgent":{"type":"noul","instructions":"urgent?"}}'
openjev serve --backend jev                          # 成为 Jev 云 API 的同构代理
```

```python
from openjev import Choice, LocalJev, RemoteJev

local = LocalJev("Qwen/Qwen3-0.6B")   # 免费离线
cloud = RemoteJev()                    # 官方 Jev API, 自动读 $TYPESAFE_API_KEY
# 同一个 .system_one(state, questions), A/B 准确率与延迟对比一步到位
```

Remote 客户端: 从 `$TYPESAFE_API_KEY` 读密钥, 429/5xx 按 `retry-after` 自动退避, 透传官方 `usage`.

### 部署

Docker / systemd / Nginx 反代 / GPU 加速: 见 [docs/DEPLOY.md](docs/DEPLOY.md). Docker 快速启动:

```bash
docker compose up -d openjev     # 本地后端, 端口 8771, 模型自动下载
```

## 三种问题原语

| 原语 | 问什么 | 返回什么 |
|---|---|---|
| `Choice` | "哪个团队处理?" + 最多 255 个选项 | `choice`, `probabilities`, `confidence` |
| `Score` | "用户多愤怒?" + 2–10 个有序等级 | `score` (加权均值), `probabilities`, `confidence`, `legend` |
| `Noul` | "紧急吗?" | `noul` ∈ [0,1] ("是" 的概率) |

## 推荐模型 (快 TTFT)

决策类负载的感知延迟由 TTFT 主导. 决策引擎从不解码, 所以 TTFT **就是全部延迟**, 优先选训练充分的小模型:

| 模型 | 参数量 | 内存 (fp32) | 说明 |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 0.6B | 约 3 GB | **默认**, CPU 上 TTFT 最快, 指令跟随好 |
| `Qwen/Qwen3-1.7B` | 1.7B | 约 7 GB | 难例上校准更好 |
| `Qwen/Qwen3-4B-Instruct-2507` | 4B | 约 16 GB | 准确率最好, 建议 GPU |
| `microsoft/Phi-4-mini-instruct` | 3.8B | 约 15 GB | 强力替代, MIT 许可 |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | 约 5 GB | Llama 生态之选 |
| `openai/gpt-oss-20b` | 20B MoE | 约 13 GB (MXO) | 原生 MXFP4, 激活参数约 3.8B, 建议 GPU |

**怎么选**: 只有 CPU → 不超过 2B; 一张消费级 GPU (8 GB+) → 4B 档; 多卡 → 20B MoE.

自己动手测:

```bash
python -m openjev.cli ask --state "server down, losing money" \
  --question '{"urgent":{"type":"noul","instructions":"Is this urgent?"}}' \
  --model Qwen/Qwen3-0.6B --json
```

## 官方模式 (来自 Jev 文档)

- **Confidence-Gated Routing (置信度门控路由)**, 低置信度转人工, 高置信度自动执行:
  ```python
  ans = result["answers"]["action"]
  if ans["confidence"] < 0.5:
      route_to_human()
  elif ans["choice"] == "approve_transfer" and ans["confidence"] > 0.9:
      execute()
  ```
- **Speculative Fan-Out (投机扇出)**, 一次请求问 20 个问题, 代码只读需要的 3 个.
- **Composite Scoring (复合打分)**, 多个原子 Score 加权合成一个综合指数.

## 项目结构

```
openjev/
  types.py    # Choice / Score / Noul, 接口 schema, confidence 与 score 数学
  core.py     # LocalJev 引擎: 答案 token 上的掩码 logit softmax
  server.py   # 标准库 HTTP 服务, POST /v1/systemone (与 Jev 接口同构)
  cli.py      # openjev ask / serve / models
tests/        # pytest; 无模型权重时引擎冒烟测试自动跳过
demo.py       # 用 Jev 官方工单示例做端到端演示
```

## 局限 (诚实声明)

- **提示词敏感**: 绝对数值随措辞变化. 分布是模型原生的, 但没有像 Jev 的 RLCD 那样做校准训练.
- **英语优先**: 与 Jev 一样, 英文提问效果最好 (小模型的多语言能力更弱).
- **无推理链**: 问题应能在第一个 token 处回答; 多步推理请用 LLM Agent, 而不是 System One 引擎.
- **只读第一次前向**: OpenJev 读取答案位置的 logits, 不先生成思维链 —— 这既是特性也是边界.

## FAQ

**问: 这是真的 Jev 吗?** 不是. Jev 是闭源模型, 用 RLCD 专门训练过校准. OpenJev 是**接口与机制的本地复现**: 同样的接口格式, 同样的原语, 概率用同样诚实的方式 (答案 token 上的掩码 softmax) 计算, 底座可以是任意开源 LLM.

**问: 能换更大的模型吗?** 能. `--model` 接任意 HF causal LM 的 id 或本地路径. 更大的模型改变概率数值, 不改变机制.

**问: 数据会出机器吗?** 永远不会. 模型本地运行, 服务默认只绑定 `127.0.0.1`.

**问: Windows / macOS / Linux?** 都可以. 纯 CPU 可跑; 有 CUDA / MPS 会自动使用.

## License

MIT
