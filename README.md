# OpenJev

English | [简体中文](README.zh-CN.md)

**Turn any local LLM into a [Jev](https://jevai.net) — the viral "System One" decision model — running 100% on your machine.**

> **30-second version.** A *decision model* doesn't chat — you send a state plus typed questions ("which team?", "how urgent?"), and it returns **type-safe answers with calibrated probabilities**: no free-text generation, no hallucination, one forward pass. [Jev](https://jevai.net) made this viral as a closed cloud API. **OpenJev is the open version:** point it at a small local model (masked-logit softmax, $0, private), *or* at any OpenAI-compatible API you already have a key for (OpenAI, OpenRouter, DeepSeek, Groq, a self-hosted vLLM...), *or* at the official Jev cloud — same interface, swap with one flag. Ship a router, a ticket triage, a confidence gate — without paying per decision or leaking data.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/lookski/openjev/actions/workflows/ci.yml/badge.svg)](https://github.com/lookski/openjev/actions/workflows/ci.yml)
[![No API Key](https://img.shields.io/badge/API%20key-none-success)](#quick-start)

Jev (TypeSafe AI, Sept 2026) made "decision models" viral: send a state + typed questions, get a **type-safe answer with a calibrated probability** — no text generation, no hallucination, 70–500 ms latency.

**OpenJev is the same idea, fully local.** Any small LLM becomes a decision engine via **masked-logit softmax**: one forward pass, mask the answer-token logits, softmax → raw probabilities. Zero API cost. Zero data leaves your machine. Type errors are mathematically impossible.

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

*(measured output, CPU-only fp32, Qwen3-0.6B — your numbers may differ slightly by version/hardware)*

![OpenJev demo](assets/demo.svg)

## Why it works

LLMs already "know" the answer. The problem is the **decoding**: generating text is slow, sampled, and untyped. OpenJev never decodes. It reads the **raw logits** at the first answer position:

| | Jev (cloud) | OpenJev (local) |
|---|---|---|
| Mechanism | parallel samplers (closed) | masked-logit softmax (open) |
| Output | type-safe + probability | type-safe + probability |
| Latency | 70–500 ms | **~0.1–3 s** (one forward pass, no decoding) |
| Cost | $0.042 / M input tokens | **$0** |
| Privacy | data leaves the machine | **100% local** |
| Weights | closed | any open LLM (Qwen, Llama, Phi, ...) |
| Context | 64K | model-dependent (Qwen3: 32K) |
| Hallucination | impossible by design | impossible by design |

The probabilities are **raw softmax values** of the masked logits — not sampled, not prompted for, not parsed from text. That is the honest, model-native confidence.

## Run it on three kinds of brains

| Brain | Engine | Needs | Cost | Privacy |
|---|---|---|---|---|
| **Local model** (default) | `LocalJev` — masked-logit softmax | any HF causal LM or local path (0.6B ≈ 3 GB RAM) | $0 | never leaves the machine |
| **OpenAI-compatible API** | `OpenAICompatJev` — top-logprobs of the first token | a key or a local server: OpenAI, OpenRouter, DeepSeek, Groq, Together, Ollama, LM Studio, vLLM, llama.cpp | per-token (or free self-hosted) | per provider; self-hosted stays local |
| **Official Jev cloud** | `RemoteJev` — wire-compatible client | `$TYPESAFE_API_KEY` | Jev pricing | data leaves the machine |

All three expose the same `.system_one(state, questions)` / `POST /v1/systemone` interface — swap with one flag, A/B accuracy and latency with zero code changes.

> **Honest boundary:** API backends read probabilities from the response's `logprobs`, so they need a provider that returns them. OpenAI-compatible endpoints do; **Anthropic's API does not expose logprobs, so it can't power an OpenJev backend** — that's a property of their API, not a limitation of this design.

## Quick start

### Easy mode (zero code, recommended first)

```bash
pip install -e .
openjev-easy          # or: python -m openjev.easy_cli
```

A wizard picks the brain for you: it auto-detects a running **Ollama / LM Studio / vLLM / llama.cpp** server, or takes an **OpenAI / OpenRouter / official Jev** API key (hidden input), runs a smoke test, then drops you into a REPL — paste any text, get typed probabilities back:

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

*(measured output, built-in engine, Qwen3-0.6B)*

Non-interactive one-shot: `openjev-easy --backend ollama --model qwen3:0.6b --once "some text"`.

**All eight backends** (menu is skipped when `--backend` is given):

| `--backend` | Connects to | Key / notes |
|---|---|---|
| `local` | in-process masked-logit softmax | downloads a model (default) |
| `ollama` | `localhost:11434/v1` | your already-running Ollama |
| `lmstudio` | `localhost:1234/v1` | your already-running LM Studio |
| `llamacpp` | `localhost:8080/v1` | llama.cpp server |
| `vllm` | `localhost:8000/v1` | self-hosted vLLM |
| `openai` | `api.openai.com/v1` | `$OPENAI_API_KEY` |
| `openrouter` | `openrouter.ai/api/v1` | `$OPENROUTER_API_KEY`; hundreds of models incl. free tiers |
| `jev` | official TypeSafe Jev cloud | `$TYPESAFE_API_KEY` |

Any other OpenAI-compatible server (DeepSeek, Groq, Together, your own vLLM behind a domain) works via `--base-url`:

```bash
openjev-easy --backend openai --base-url https://api.deepseek.com/v1 \
  --model deepseek-chat --api-key sk-... --once "text"
```

Library equivalent — one factory, every brain:

```python
from openjev.easy import make_engine

engine = make_engine("vllm", model="Qwen/Qwen3-0.6B", base_url="http://gpu-box:8000/v1")
# same .system_one(state, questions) as LocalJev / RemoteJev — swap brains freely
```

All backends expose the same interface, so you can A/B local vs cloud with one flag.

> API backends read probabilities from `logprobs` of the first generated token — that's why they need an OpenAI-compatible endpoint (Anthropic's API doesn't expose logprobs). No logprobs, no honest probabilities: the backend refuses instead of making numbers up.

### Full engine (masked softmax, the default deep path)

```bash
pip install -e .
python demo.py
```

First run downloads Qwen3-0.6B (~1.5 GB) from Hugging Face. In China: `export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`.

### One-liner (library API)

```python
from openjev import Choice, LocalJev, Noul, Score

engine = LocalJev("Qwen/Qwen3-0.6B")           # or a local path
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

### Local server (Jev wire-compatible)

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

Response shape matches the official API (`answers` keyed by question id, `noul` has no `confidence`, `score` is the index-weighted mean, `confidence = (K·max_p − 1)/(K − 1)`).

### Two backends, one interface (local softmax ⇄ official Jev API)

> For the full backend menu (OpenAI-compatible APIs too) see Easy mode above and `make_engine()` in [`openjev/easy.py`](openjev/easy.py). This section is the library-level local ⇄ official-Jev pair.

```bash
export TYPESAFE_API_KEY=sk-...                      # optional cloud backend
openjev ask --backend jev --state "..." \
  --question '{"urgent":{"type":"noul","instructions":"urgent?"}}'
openjev serve --backend jev                          # wire-compatible Jev proxy
```

```python
from openjev import Choice, LocalJev, RemoteJev

local = LocalJev("Qwen/Qwen3-0.6B")   # free, offline
cloud = RemoteJev()                    # official Jev API, reads $TYPESAFE_API_KEY
# same .system_one(state, questions) — perfect for A/B accuracy & latency tests
```

The remote client retries on 429/5xx with `retry-after` backoff and passes the official `usage` through.

### Deployment

Docker, systemd, Nginx reverse proxy and GPU notes: see [docs/DEPLOY.md](docs/DEPLOY.md). Quick Docker start:

```bash
docker compose up -d openjev     # local backend on :8771, model auto-downloaded
```

## The three primitives

| Primitive | Ask | Answer |
|---|---|---|
| `Choice` | "Which team?" + up to 255 options | `choice`, `probabilities`, `confidence` |
| `Score` | "How frustrated?" + 2–10 ordered levels | `score` (weighted mean), `probabilities`, `confidence`, `legend` |
| `Noul` | "Is this urgent?" | `noul` ∈ [0,1] ("yes" probability) |

## Recommended models (fast TTFT)

TTFT dominates perceived latency of decision workloads. On a decision engine you never decode, so TTFT **is** the whole latency — pick a small, well-trained model:

| Model | Size | RAM (fp32) | Notes |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 0.6B | ~3 GB | **default** — fastest CPU TTFT, strong English instruction following |
| `Qwen/Qwen3-1.7B` | 1.7B | ~7 GB | better calibration on hard cases |
| `Qwen/Qwen3-4B-Instruct-2507` | 4B | ~16 GB | best accuracy; GPU recommended |
| `microsoft/Phi-4-mini-instruct` | 3.8B | ~15 GB | strong alternative, permissive MIT |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | ~5 GB | good Llama ecosystem choice |
| `openai/gpt-oss-20b` | 20B MoE | ~13 GB (MXO) | native MXFP4, ~3.8B active params, GPU recommended |

**How to choose:** CPU-only → stay ≤ 2B. One consumer GPU (8 GB+) → 4B class. Multi-GPU → 20B MoE.

Measure it yourself:

```bash
python -m openjev.cli ask --state "server down, losing money" \
  --question '{"urgent":{"type":"noul","instructions":"Is this urgent?"}}' \
  --model Qwen/Qwen3-0.6B --json
```

## Patterns (from the official Jev docs)

- **Confidence-Gated Routing** — low confidence → human, high confidence → act:
  ```python
  ans = result["answers"]["action"]
  if ans["confidence"] < 0.5:
      route_to_human()
  elif ans["choice"] == "approve_transfer" and ans["confidence"] > 0.9:
      execute()
  ```
- **Speculative Fan-Out** — ask 20 questions in one request, read the 3 you need.
- **Composite Scoring** — sum weighted `Score` answers into one composite index.

## Project layout

```
openjev/
  types.py    # Choice / Score / Noul, wire schema, confidence + score math
  core.py     # LocalJev engine: masked-logit softmax over the answer tokens
  server.py   # stdlib HTTP server, POST /v1/systemone (Jev wire-compatible)
  cli.py      # openjev ask / serve / models
tests/        # pytest; engine tests auto-skip without weights
demo.py       # end-to-end demo on the canonical Jev ticket example
```

## Party trick: the yin-yang detector

**🚀 Try it in your browser, no install:** [**openjev-detector online demo**](https://linrin0306-openjev-detector.static.hf.space) — pure-frontend Hugging Face Space (transformers.js, WebGPU fp16 / WASM q4f16 fallback, inference runs on your device, nothing is uploaded).

```bash
python examples/fun_yinyang.py        # --text "your message" for custom input
```

Measured with the default 0.6B brain (real run output):

| message | yin-yang | vibe |
|---|---|---|
| `哦` | 0.8424 | sincere 0.9802 |
| `好的，都可以，你决定就好` | 0.7837 | sincere 0.9904 |
| `6` | 0.8751 | sincere 0.9790 |
| `今天天气真好，一起去吃火锅吧` | 0.7155 | sincere 0.9951 |

Yes, it rates everything at ~70-88% sarcasm while simultaneously insisting the sender is 98% sincere. **The model itself is the most yin-yang thing here** - and that is exactly why raw probabilities beat hard labels: you can see the contradiction instead of trusting a single verdict.

### Chat radar: who blows up next?

```bash
python examples/fun_chat_radar.py          # default boss-vs-intern demo
# or: python examples/fun_chat_radar.py --file chat.txt   ("name: message" lines)
```

Measured on the default demo chat (real output):

```
===== per-speaker radar =====
  小王    msgs=3  hostility=0.192  explode=0.951  yinyang=0.961
  老板    msgs=3  hostility=0.132  explode=0.904  yinyang=0.894

===== next to explode (top 3) =====
  1. 小王  peak-explode 0.982 on: "嗯"
```

In the model's eyes **everyone is 90%+ about to explode**, and the most dangerous message in the whole chat is... `嗯`. Honestly? Accurate.

### Crush radar: should you text back?

```bash
python examples/fun_crush_radar.py --file chat.txt   # two speakers, "name: message" lines
```

Paste a chat log with someone you are pursuing; every message gets an interest score (0-3), a warmth and a perfunctory probability, then the tool prints one move: **冲 / 稳 / 缓 / 停**. Two demo chats, measured with the default 0.6B brain (real output):

| their replies | their interest | yours | verdict |
|---|---|---|---|
| `我看看有没有时间吧` / `嗯` / `没干嘛, 就是挺忙的` | 0.06/3 | 0.73/3 | **缓** — their investment is far below yours (gap −0.67), hand the topic back |
| `去呀! 那我们一起呗` / `太好了, 那说定了` | 0.62/3 | 0.40/3 | **稳** — trend flat, investments close; decide on the next round |

The verdict is derived from **relative** signals only (interest gap + trend direction) — the toy model compresses absolute Chinese-message scores into a narrow band (it rates almost everything perfunctory at ~0.9, exactly like the yin-yang detector rates everything yin-yang), so absolute thresholds would be meaningless. Honest numbers in, honest hedging out.

No WeChat/QQ integration by design: auto-reading chat apps risks account bans. OpenJev reads what **you paste**, nothing more.

### Crush radar, serious edition: full-context LLM engine + bot hub

Token probabilities from a 0.6B model are a party trick. For a verdict you might actually act on, point the radar at a **big LLM through any OpenAI-compatible API** — the engine is provider-agnostic and ships **no built-in endpoint and no default model**.

**One-time setup, then it just works** (CLI wizard or the hub's web panel; the config is saved to `~/.openjev/llm.json`, mode 0600):

```bash
python -m openjev.llm_config            # wizard: paste base URL, fetch /models, pick one, smoke test, save
# or non-interactive:
python -m openjev.llm_config --base-url https://api.deepseek.com/v1 --model deepseek-chat --api-key sk-...
```

The wizard fetches the endpoint's `/models` list so you pick from what actually exists, runs a one-word smoke test, and saves. Afterwards every command and the hub read that config — no env vars needed (though `OPENJEV_LLM_BASE_URL/MODEL/API_KEY` and CLI flags still override it). Inspect with `--status` (key masked).

The whole conversation goes in **with context** and comes back as structured JSON: verdict, interest scores, trend, warmth, **reply direction** (a one-phrase strategy like "nail down time+place, offer a binary choice"), a paste-ready next message, and the reason (measured outputs):

```text
#   low-interest chat  -> their 0.4/3, trend falling  -> 停: "她连续用敷衍、拒绝和'挺忙的'收尾, 没有一次反问或主动"
#   high-interest chat -> their 2.9/3, trend rising   -> 冲: "她主动提议同行、敲定时间地点还回请吃饭"
#   borderline chat    -> their 1.3/3, trend flat     -> 稳: "邀约时用'看情况+可能加班'打太极"
#   warming chat       -> their 2.7/3, trend rising   -> 冲: "应趁热把时间地点钉死防止鸽掉" + next_advice
```

The verdict includes `next_advice` — a concrete suggestion for your next message, not just a number. Hardened against chatty providers: `response_format: json_object` with fallback, plus a per-key salvage parser that recovers verdicts from truncated/garbled completions.

### WeChat radar: screenshot in, verdict out (floating widget)

The paste UI works, but switching windows to paste is friction. `openjev/wechat_radar.py` is a small **always-on-top floating widget** with an auto-watch clipboard: you screenshot in WeChat, the verdict just pops up.

```bash
python -m openjev.wechat_radar      # floating widget appears, always on top, watching
```

**Auto mode (default, zero keystrokes):** screenshot the chat area (`Alt+A` / `Win+Shift+S`) or multi-select messages → 复制 — the widget notices the new clipboard content and runs the whole chain automatically (local OCR → transcript → LLM verdict → big colored 冲/稳/缓/停 on the widget). Non-chat content is ignored silently (a 2-line + CJK heuristic gates text; images below 200×80 are skipped), so normal copying never triggers a false analysis.

Manual hotkeys still force-run anything: **Ctrl+F2** (clipboard text) / **Ctrl+F3** (clipboard screenshot). Click the result to copy the verdict; drag the title to move; double-click for compact mode; right-click to close. `--no-auto` disables the watcher (hotkeys only), `--poll-ms` tunes the check interval.

The widget never touches WeChat itself — no hooks, no automation, no reading of WeChat memory. The watcher checks only a sequence number (zero cost) and reads clipboard content when it changes, exactly as if you pasted it yourself. Screenshots stay in RAM (nothing written to disk). OCR is local; only the recognized transcript goes to your configured LLM endpoint (same privacy note as above).

Known OCR floor: single-character bubbles like `嗯` fall below the OCR detection threshold and may be missed — an acceptable loss, since a one-char reply rarely flips a verdict.

### WeChat assistant: incoming message in, verdict + draft out

The radar still needs you to screenshot. `openjev/wechat_assistant.py` goes further: it **watches a conversation for new incoming messages** and runs the whole loop by itself.

```bash
python -m openjev.wechat_assistant --who 她的备注名     # assistant mode, one chat
python -m openjev.wechat_assistant --who 甲,乙          # several chats
python -m openjev.wechat_assistant --no-draft            # verdict only, no draft
```

When the watched contact sends a message, the widget pops the 冲/稳/缓/停 verdict automatically, and the **suggested reply is pasted into that chat's input box as a draft** — you review it and press Enter (or clear it). Nothing is ever sent without your keystroke; that is the hard line.

How it works, and what it does not do:

- Built on [wxauto4](https://docs.wxauto.org/) (Windows UI Automation): OpenJev reads the on-screen chat window of your own logged-in client and writes into the input box, the same way you would with mouse and keyboard. **No DLL injection, no memory reading, no protocol reverse-engineering** — the techniques that got accounts banned in the 2025 crackdown. UI automation of your own session is the lowest-risk integration that exists for personal WeChat; it is still third-party, not official, so use a chat you own (start with 文件传输助手) and keep frequencies human.
- New-message detection is dual-channel: the wxauto callback plus an independent poll-and-diff fallback, so a silent callback cannot blind the assistant.
- Your own sent messages are never judged (only incoming/friend messages trigger analysis).
- The clipboard is saved and restored around every draft paste — your copied stuff survives.
- Same privacy shape as the radar: everything local except the transcript sent to your configured LLM.
- Requires Windows + WeChat PC 4.x logged in; `pip install wxauto4` (free edition works on Python 3.13 via cp313 wheel ≥41.1.7). If attach fails once with "未找到已登录的客户端主窗口", it retries automatically.

### QQ assistant: OneBot 11 real-time verdicts (easier than WeChat)

QQ is structurally the easy case: NapCat speaks the OneBot 11 protocol, so messages arrive as structured JSON events - no OCR, no UI automation, and even your **own outgoing messages** come through, giving full two-sided context.

```bash
python -m openjev.qq_assistant --ws ws://127.0.0.1:3001   # NapCat forward WS
python -m openjev.qq_assistant --watch 10086              # one QQ uin only (default: all private chats)
python -m openjev.qq_assistant --selftest                 # full-chain selftest without NapCat
```

Prerequisite: NapCat (or LLOneBot) with forward WebSocket enabled (default port 3001). Message bursts are debounced - analysis fires only after `--quiet-secs` (default 6) of silence, over the last `--context` (default 30) lines. `--send-reply` posts the suggestion back into the chat; off by default (suggest, never auto-send - same hard boundary as the WeChat assistant).

Honest note: NapCat is a third-party injection into the NTQQ client; theoretical ban risk exists (historically QQ has been far more tolerant than WeChat) - use a spare account if worried. The official q.qq.com bot platform cannot read friend chats, so it can't serve this use case. The module sends nothing to anyone unless you pass `--send-reply`.

### Telegram assistant: the cleanest path of all (official API + history backfill)

Telegram officially opens the MTProto user API - log in **as yourself** via a legal third-party client (same class as Telegram Desktop; no injection, no hooking). Messages flow in real time both ways, and startup can **backfill recent history** straight into context (something neither QQ nor WeChat can do):

```bash
python -m openjev.tg_assistant --login     # one-time login (api_id/api_hash from my.telegram.org)
python -m openjev.tg_assistant             # live verdicts on all private chats
python -m openjev.tg_assistant --watch 123456789 --backfill 50 --push http://127.0.0.1:8793/api/verdict
python -m openjev.tg_assistant --selftest  # full-chain selftest without logging in
```

The event core reuses the QQ assistant (OneBot-shaped mapping); quiet-debounce / verdict / --push semantics are identical. Groups are ignored - private chats only.

Other platforms, honestly: **Discord** - official bots can't read your DMs, self-bots violate ToS; **WhatsApp** - no personal API, only WhatsApp-Web automation (WeChat-UIA risk class) or protocol reimplementations (hook risk class); **LINE/KakaoTalk** - official APIs are service-accounts only, no path; **Signal** - official "linked device" exists but with heavy restrictions; **iMessage** - macOS AppleScript only. Verdict: Telegram > QQ > WeChat > WhatsApp > the rest.

**In-chat overlay card (`qq_overlay`)**: the verdict is layered right at the end of the message area in a specified chat window - visually "analysis right after their messages":

```bash
# first pop the chat out into its own window (double-click her in the session list)
python -m openjev.qq_overlay --who her-remark-name
```

The card is a translucent always-on-top overlay that follows the chat window; click it to copy the suggested reply; click **[+ details]** or double-click to expand the full report (big font + copy/collapse buttons). **No hooking, no injection - the QQ process never knows it exists** (visual inlining via overlay beats risking injection). Events still come from NapCat; `--selftest` runs the full chain without it. Rendering the analysis into the actual message-flow DOM is LiteLoaderQQNT-plugin territory (injection; not done here).

**Bot hub** (paste UI + chat-platform bridge, one process):

```bash
python -m openjev.crush_bot --port 8792       # then open http://127.0.0.1:8792
```

- `POST /api/analyze` `{chat, you, them}` — used by the built-in paste UI
- `POST /qq` / `POST /wechat` — same contract for a chat-platform relay you operate: the platform side (QQ official-bot webhook, WeChat bridge) posts the conversation, the hub replies with a formatted verdict text. On the WeChat side OpenJev deliberately ships **no auto-receiving** (hooking risks bans); on the QQ side real-time receiving is built in via OneBot 11 - see `qq_assistant` above.
- Privacy note: full-context analysis means the conversation is sent to the LLM provider. Use a self-hosted endpoint (`--base-url http://your-vllm/v1`) for sensitive chats.

**Use it from your phone**: the hub page is mobile-ready (viewport + responsive). On the same WiFi:

```bash
python -m openjev.crush_bot --host 0.0.0.0 --port 8792   # open http://<pc-ip>:8792 in the phone browser
```

Phones can paste and analyze, but **never see the endpoint or key** (the remote view returns only "configured + model name"); config writes and model listing are loopback-only - LAN requests get 403 unless you explicitly set `OPENJEV_ALLOW_REMOTE_CONFIG=1`.

The phone page also has a **live verdict feed**: pass `--push` to the local QQ/WeChat processes and verdicts stream onto the page in real time (SSE):

```bash
python -m openjev.qq_overlay --who her-remark-name --push http://127.0.0.1:8793/api/verdict
```

Note: with the hub bound to 0.0.0.0 the feed is visible to the whole LAN (including chat-derived reason text) - use it only on a WiFi you trust.

### Browser UI (paste and judge)

```bash
openjev-web                     # then open http://127.0.0.1:8791
# friends on the same Wi-Fi:  openjev-web --host 0.0.0.0
```

Single-file stdlib web app (no JS framework, no CDN): paste any message, get animated probability bars for yin-yang / hostility / vibe. 100% local, `127.0.0.1` only by default.

Public hosted instance (same questions, same masked-softmax math, running fully in the browser via transformers.js + onnx-community/Qwen3-0.6B-ONNX): **https://linrin0306-openjev-detector.static.hf.space** — source in [`docs/space-deploy/`](docs/space-deploy/).

### Deploy your own public instance (free)

Two routes:

1. **Static Space (browser-only, zero backend)** — what the demo above runs on. Weights: q4f16 (570 MB, WASM fallback) self-hosted in the Space; fp16 (WebGPU high-precision) streams from [`onnx-community/Qwen3-0.6B-ONNX`](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX). Push `docs/space-deploy/` + the quantized weights to a Static Space and you are live. Note: Static Spaces have a **1 GB storage limit**.
2. **Docker Space (server-side engine)** — the `space/` folder is a ready-to-push [Hugging Face Space](https://huggingface.co/spaces) (Docker SDK): create a Space → **Docker** → empty, copy this repo's `space/` contents (plus `openjev/`, `scripts/`, `pyproject.toml`), push, and the Space builds, downloads Qwen3-0.6B, and serves the detector on a public URL.

Zero server cost, your own public link for sharing.

## Limitations (honest section)

- **Prompt sensitivity**: the absolute numbers depend on prompt phrasing. The distribution is model-native, but calibration is not RL-trained like Jev's RLCD.
- **English-first**: like Jev, decision quality is best on English prompts (small-model multilingual ability is weaker).
- **No reasoning**: questions should be answerable at first token; for multi-step problems use an LLM agent, not a System One engine.
- **First forward pass only**: OpenJev reads the logits at the answer position. It does not generate a chain of thought first — that is the point, and also the limitation.

## FAQ

**Q: Is this the real Jev?** No. Jev is a closed model trained with RLCD specifically for calibrated decisions. OpenJev is a **local reimplementation of the interface and mechanism** — same wire format, same primitives, probabilities computed the same honest way (masked softmax over answer tokens), using whatever open LLM you point it at.

**Q: Can I use bigger models?** Yes — pass `--model` any HF causal LM id or local path. Bigger models shift the probabilities, not the mechanism.

**Q: Does my data leave the machine?** Never. The model runs locally; the server binds to `127.0.0.1` by default.

**Q: Windows / macOS / Linux?** All fine. CPU-only works; CUDA/MPS auto-detected if present.

## Star history note

If OpenJev saved you from paying for decision-model APIs, drop a ⭐ — it keeps a broke grad student in GPUs.

## License

MIT
