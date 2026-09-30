#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crush analysis engine: full-context LLM verdict for pursuit-phase chat logs.

编写时间: 2026-09-30 16:18:39
脚本功能: send a whole "name: message" conversation (with context) to a
          large LLM via an OpenAI-compatible endpoint, get back compact JSON:
          interest_their / interest_mine / trend / warmth_signals / move
          (chong|wen|huan|ting) / next_advice / reason. Provider-agnostic:
          no built-in endpoint; configure via $OPENJEV_LLM_BASE_URL,
          $OPENJEV_LLM_MODEL, $OPENJEV_LLM_API_KEY or CLI flags.
参数: analyze(text, you=None, them=None) -> dict; also CLI:
          python openjev/crush_llm.py --file chat.txt [--you 我] [--them 她]
输入格式: "name: message" lines, # comments ignored (same as fun_chat_radar).
输出格式: dict with the fields above (move mapped to Chinese label outside);
          CLI prints a human-readable report. Raises RuntimeError on
          unavailable endpoint or unparseable model output.
依赖: stdlib only (urllib, ssl, json, re); $OPENJEV_LLM_API_KEY or --api-key.
注意事项: privacy: the whole conversation is sent to the LLM provider --
          tell users to use a self-hosted endpoint (e.g. local vLLM) if the
          chat is sensitive. This is advice, not a promise; toy verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request

# No default endpoint on purpose: the engine is provider-agnostic and reads
# $OPENJEV_LLM_BASE_URL / $OPENJEV_LLM_MODEL / $OPENJEV_LLM_API_KEY, or
# --base-url/--model/--api-key. Point it at ANY OpenAI-compatible server
# (OpenAI, DeepSeek, Moonshot, SiliconFlow, OpenRouter, your own vLLM...).
DEFAULT_TIMEOUT = 120

LINE_RE = re.compile(r"^([^:：]{1,24})[:：]\s*(.+)$")

SYSTEM_PROMPT = (
    "You are a candid relationship chat analyzer for someone in the pursuit phase "
    "(追求期). Read the FULL conversation with context, distinguish the two speakers "
    "(one is pursuing, the other is pursued), and judge the pursued person's real "
    "attitude. Be honest, not polite: reluctant short replies, topic changes and "
    "excuses mean low interest; questions back, initiative, time/place proposals mean "
    "high interest. Return ONLY compact JSON, no markdown fence, with keys: "
    '{"interest_their": <0.0-3.0 number>, "interest_mine": <0.0-3.0 number>, '
    '"trend": "<rising|flat|falling>", "warmth_signals": <0.0-1.0 number>, '
    '"move": "<chong|wen|huan|ting>", '
    '"next_advice": "<what concrete message to send next, one short Chinese sentence>", '
    '"reason": "<one short Chinese sentence justifying the move>"} '
    "Scale guide: 0.0-1.0 = reluctant/cold, 1.0-2.0 = polite but passive, "
    "2.0-3.0 = engaged/warm. move meanings: chong = escalate, ask for a date; "
    "wen = keep normal pace, observe; huan = slow down, hand back the topic; "
    "ting = stop pursuing this thread entirely."
)

MOVE_LABELS = {"chong": "冲", "wen": "稳", "huan": "缓", "ting": "停"}
VALID_MOVES = set(MOVE_LABELS)
VALID_TRENDS = {"rising", "flat", "falling"}


def parse_log(text):
    """Parse 'name: message' lines; return [(name, message), ...]."""
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if m:
            entries.append((m.group(1).strip(), m.group(2).strip()))
    return entries


def guess_roles(entries, you=None, them=None):
    """Resolve speaker roles: explicit args > heaviest-two heuristic."""
    counts = {}
    for name, _ in entries:
        counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    you_name = you or (ranked[0][0] if ranked else "我")
    them_name = them or (ranked[1][0] if len(ranked) > 1 else you_name)
    return you_name, them_name


def _repair_json(raw):
    """Best-effort repair of near-miss JSON from the model.

    Common failure: unescaped ASCII double quotes inside a Chinese string
    value ("行，\"你先忙\"吧"). Strategy: strip code fences, then quote-wise
    escape any inner ASCII quotes that break the object structure.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    return text


def _salvage_fields(raw):
    """Pull known keys out of a garbled/truncated model reply.

    Some models in json_object mode sometimes emit a broken first attempt and
    then recovers with a full object (duplicate keys, unicode escapes).
    Lenient per-key regex; LAST occurrence wins (recovery-write pattern).
    Lenient per-key regex; LAST occurrence wins (recovery-write pattern).
    Returns a dict for _validate, or raises ValueError if move/reason absent.
    """
    keys = "interest_their|interest_mine|trend|warmth_signals|move|next_advice|reason"
    pat = re.compile(
        r'"(%s)"\s*:\s*(' r'"(?:[^"\\]|\\.)*"' r'|[-0-9.eE]+' r'|true|false)' % keys,
        re.S,
    )
    found = {}
    for k, v in pat.findall(raw):
        found[k] = v  # last occurrence wins
    if "move" not in found or "reason" not in found:
        raise ValueError("salvage found no move/reason in %r" % raw[:120])
    out = {}
    for k, v in found.items():
        if v.startswith('"'):
            out[k] = json.loads(v)  # decodes \uXXXX and escapes properly
        elif v in ("true", "false"):
            out[k] = v == "true"
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _extract_json(raw):
    """Pull the first JSON object out of a model reply (fence-tolerant)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in model reply: %r" % raw[:120])
    return json.loads(m.group(0))


def _validate(d):
    """Check required keys and enum values; normalize numbers to float."""
    for k in ("interest_their", "interest_mine", "trend", "move", "reason"):
        if k not in d:
            raise ValueError("missing key %r in %r" % (k, list(d.keys())))
    d["interest_their"] = float(d["interest_their"])
    d["interest_mine"] = float(d["interest_mine"])
    if d["move"] not in VALID_MOVES:
        raise ValueError("bad move: %r" % d["move"])
    if d["trend"] not in VALID_TRENDS:
        d["trend"] = "flat"
    d.setdefault("warmth_signals", None)
    d.setdefault("next_advice", "")
    return d


def analyze(
    chat_text,
    you=None,
    them=None,
    base_url=None,
    model=None,
    api_key=None,
    timeout=DEFAULT_TIMEOUT,
):
    """Analyze a full conversation; return the validated verdict dict.

    chat_text: raw log text ("name: message" lines) - sent with full context.
    Raises RuntimeError on HTTP failure or unparseable output.
    """
    entries = parse_log(chat_text)
    if not entries:
        raise ValueError('no "name: message" lines found')
    you_name, them_name = guess_roles(entries, you, them)
    transcript = "\n".join("%s: %s" % (n, m) for n, m in entries)
    user_msg = (
        "The person pursuing is %r, the person being pursued is %r.\n"
        "Conversation:\n%s" % (you_name, them_name, transcript)
    )
    base = (base_url or os.environ.get("OPENJEV_LLM_BASE_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError(
            "no LLM endpoint configured: set $OPENJEV_LLM_BASE_URL "
            "(OpenAI-compatible, e.g. https://api.deepseek.com/v1) "
            "or pass --base-url"
        )
    mdl = model or os.environ.get("OPENJEV_LLM_MODEL") or ""
    if not mdl:
        raise RuntimeError(
            "no model configured: set $OPENJEV_LLM_MODEL or pass --model"
        )
    key = api_key or os.environ.get("OPENJEV_LLM_API_KEY") or ""
    body = {
        "model": mdl,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(
            req, timeout=timeout, context=ssl.create_default_context()
        )
        payload = json.loads(resp.read())
        raw = payload["choices"][0]["message"].get("content") or ""
    except urllib.error.HTTPError as exc:
        # json_object is a hard requirement on some gateways; retry once without it
        if exc.code == 400:
            body.pop("response_format", None)
            req = urllib.request.Request(
                base + "/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
            )
            try:
                resp = urllib.request.urlopen(
                    req, timeout=timeout, context=ssl.create_default_context()
                )
                payload = json.loads(resp.read())
                raw = payload["choices"][0]["message"].get("content") or ""
            except Exception as exc2:
                raise RuntimeError("crush LLM endpoint failed: %s" % exc2) from exc2
        else:
            raise RuntimeError("crush LLM endpoint failed: %s" % exc) from exc
    except Exception as exc:  # network errors -> one loud failure
        raise RuntimeError("crush LLM endpoint failed: %s" % exc) from exc
    if not raw.strip():
        raise RuntimeError("crush LLM returned empty content")
    verdict = None
    last_err = None
    for attempt in range(3):
        try:
            if attempt == 0:
                verdict = _validate(_extract_json(raw))
            elif attempt == 1:
                verdict = _validate(_extract_json(_repair_json(raw)))
            else:
                verdict = _validate(_salvage_fields(raw))
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
    if verdict is None:
        raise RuntimeError("crush LLM output unparseable: %s" % last_err)
    verdict["_you"] = you_name
    verdict["_them"] = them_name
    verdict["_model"] = mdl
    return verdict


def main():
    """CLI: analyze a chat file (or stdin) and print a report."""
    ap = argparse.ArgumentParser(description="OpenJev crush radar (LLM engine)")
    ap.add_argument("--file", help="chat log file; stdin if omitted")
    ap.add_argument("--you", help="name of the pursuing speaker")
    ap.add_argument("--them", help="name of the pursued speaker")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint")
    ap.add_argument("--model", help="model id")
    ap.add_argument("--json", action="store_true", help="print raw JSON verdict")
    args = ap.parse_args()
    text = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    verdict = analyze(text, you=args.you, them=args.them,
                      base_url=args.base_url, model=args.model)
    if args.json:
        print(json.dumps(verdict, ensure_ascii=False, indent=1))
        return
    print("===== crush radar (%s -> %s, engine %s) ====="
          % (verdict["_you"], verdict["_them"], verdict["_model"]))
    print("  interest: theirs %.1f/3, yours %.1f/3, trend %s, warmth %s"
          % (verdict["interest_their"], verdict["interest_mine"], verdict["trend"],
             ("%.2f" % verdict["warmth_signals"]) if verdict["warmth_signals"] is not None else "n/a"))
    print("  move: %s" % MOVE_LABELS[verdict["move"]])
    print("  reason: %s" % verdict["reason"])
    if verdict["next_advice"]:
        print("  next: %s" % verdict["next_advice"])


if __name__ == "__main__":
    main()
