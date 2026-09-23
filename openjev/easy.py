#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Easy mode: plug any OpenAI-compatible chat API into the decision engine.

Reads raw answer probabilities from the top logprobs of the first generated
token (logprobs=True + top_logprobs), so Ollama / LM Studio / vLLM /
OpenRouter / OpenAI all work with the same code path. This is the
"download and just use it" backend for people who already run a local
inference server or hold an API key.

编写时间: 2026-09-23 11:49:40
脚本功能: implement OpenAICompatJev (logprobs-based engine) and
          make_engine() factory resolving backend names to engines.
参数: see OpenAICompatJev.__init__ and make_engine docstrings.
输入格式: openjev.types question objects; state as str / dict / list.
输出格式: same answer dicts as LocalJev / RemoteJev.
依赖: stdlib urllib / json / base64; openjev.core, openjev.types.
注意事项: logprobs come back as bytes-encoder entries for non-ASCII tokens;
          this module decodes them by utf-8 first and falls back to latin-1
          (matching the OpenAI logprobs bytes convention).
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from math import exp, inf
from typing import Any, Dict, List, Optional, Tuple

from openjev.core import (
    DEFAULT_MODEL_ID,
    DEFAULT_SYSTEM_PROMPT,
    LETTERS,
    LocalJev,
    build_user_text,
)
from openjev.types import Choice, Noul, Question, Score, SystemOneRequest, render_answer

OPENAI_TIMEOUT = 120.0


def _decode_token(token: Any) -> str:
    """Decode one logprobs token entry (str, or {'bytes': [...]} dict)."""
    if isinstance(token, str):
        return token
    if isinstance(token, dict) and "bytes" in token:
        raw = bytes(token["bytes"])
        for enc in ("utf-8", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
    return str(token)


def _top_logprobs(entries: List[dict]) -> List[Tuple[str, float]]:
    """Normalize one content.top_logprobs list to [(token, prob), ...]."""
    out: List[Tuple[str, float]] = []
    for entry in entries:
        tok = _decode_token(entry.get("token"))
        lp = float(entry.get("logprob")) if entry.get("logprob") is not None else -inf
        out.append((tok, lp))
    return out


def _probs_from_top(top: List[Tuple[str, float]], labels: List[str]) -> Optional[List[float]]:
    """
    Extract label probabilities from top-logprobs entries.

    Returns None when any label is missing from the top-k window: in that
    case the honest fallback is generation mode, not made-up numbers.
    """
    table = dict(top)
    vals: List[float] = []
    for label in labels:
        if label not in table:
            return None
        vals.append(exp(table[label]))
    total = sum(vals)
    return [v / total for v in vals]


class OpenAICompatJev:
    """
    Decision engine over any OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    base_url : e.g. "http://localhost:11434/v1" (Ollama),
               "http://localhost:1234/v1" (LM Studio),
               "https://openrouter.ai/api/v1" (OpenRouter),
               "https://api.openai.com/v1" (OpenAI).
    model : model name the server exposes (e.g. "qwen3:0.6b").
    api_key : bearer token; defaults to $OPENAI_API_KEY; Ollama ignores it.
    system_prompt : override the shared decision-engine system prompt.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.system_prompt = system_prompt
        self.usage = {"forward_passes": 0, "input_tokens": 0, "latency_ms_total": 0.0}

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------

    def _chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One POST to /chat/completions with retry on 429/5xx."""
        url = self.base_url + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        last_error: Optional[Exception] = None
        for attempt in range(3):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=OPENAI_TIMEOUT) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    retry_after = exc.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else 2.0 * (attempt + 1)
                    time.sleep(delay)
                    last_error = exc
                    continue
                raise
            except urllib.error.URLError as exc:
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    last_error = exc
                    continue
                raise
        raise last_error  # pragma: no cover

    # ------------------------------------------------------------------
    # decision engine surface
    # ------------------------------------------------------------------

    def _labels_for(self, question: Question) -> List[str]:
        """Answer labels for one question type."""
        if question.type_name == "choice":
            return LETTERS[: len(question.criteria)]
        if question.type_name == "score":
            from openjev.core import DIGITS

            return DIGITS[: len(question.criteria)]
        if question.type_name == "noul":
            from openjev.core import YESNO

            return list(YESNO)
        raise ValueError("unknown question type: %s" % question.type_name)

    def answer_one(self, question: Question, state: Any) -> Dict[str, Any]:
        """Answer one typed question via top-logprobs of the first token."""
        question.validate()
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        labels = self._labels_for(question)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_text(state_text, question)},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
        }
        t0 = time.perf_counter()
        data = self._chat(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        choice0 = data["choices"][0]
        content = choice0.get("logprobs", {}).get("content") or []
        if not content or not content[0].get("top_logprobs"):
            raise RuntimeError(
                "server returned no logprobs; "
                "this backend cannot produce honest probabilities"
            )
        top = [(t, lp) for t, lp in _top_logprobs(content[0]["top_logprobs"])]

        # try exact label match first
        probs = _probs_from_top(top, labels)
        if probs is None and question.type_name == "noul":
            # fuzzy case-insensitive match for Yes/No variants ("yes", "YES")
            lowered = {}
            for tok, lp in top:
                lowered.setdefault(tok.strip().lower(), lp)
            vals = []
            for label in labels:
                if label.lower() not in lowered:
                    raise RuntimeError(
                        "answer token %r not in top-logprobs window; "
                        "probabilities would be dishonest" % label
                    )
                vals.append(exp(lowered[label.lower()]))
            total = sum(vals)
            probs = [v / total for v in vals]
        if probs is None:
            raise RuntimeError(
                "answer tokens not fully covered by top-logprobs window; "
                "try a bigger top_logprobs or another backend"
            )

        self.usage["forward_passes"] += 1
        self.usage["input_tokens"] += int(data.get("usage", {}).get("prompt_tokens", 0))
        self.usage["latency_ms_total"] += elapsed_ms
        return render_answer(question, probs)

    def system_one(self, state: Any, questions: Dict[str, Question]) -> Dict[str, Any]:
        """Answer many typed questions; same surface as LocalJev/RemoteJev."""
        request = SystemOneRequest(state=state, questions=questions)
        request.validate()
        answers: Dict[str, Any] = {}
        for qid, question in request.questions.items():
            answers[qid] = self.answer_one(question, state)
        return {
            "model": "openai-compat/%s" % self.model,
            "answers": answers,
            "usage": dict(self.usage),
        }


# ---------------------------------------------------------------------------
# engine factory
# ---------------------------------------------------------------------------

KNOWN_LOCAL_ENDPOINTS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "llamacpp": "http://localhost:8080/v1",
    "vllm": "http://localhost:8000/v1",
}


def _resolve_local_model(model: Optional[str]) -> str:
    """
    Pick the local backend model id: explicit arg > $OPENJEV_MODEL >
    an existing models/<basename> snapshot > the HF default id.
    The snapshot check keeps the wizard usable fully offline.
    """
    if model:
        return model
    env = os.environ.get("OPENJEV_MODEL")
    if env:
        return env
    basename = DEFAULT_MODEL_ID.split("/")[-1]
    for candidate in (
        os.path.join("models", basename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", basename),
    ):
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return DEFAULT_MODEL_ID


def make_engine(
    backend: str = "local",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dtype: str = "float32",
    device: Optional[str] = None,
    jev_model: str = "jev-latest",
):
    """
    Resolve a backend name to an engine instance.

    backends
    --------
    local    : in-process masked-logit softmax (default, downloads a model)
    ollama   : OpenAI-compat API at localhost:11434/v1
    lmstudio : OpenAI-compat API at localhost:1234/v1
    llamacpp : OpenAI-compat API at localhost:8080/v1
    vllm     : OpenAI-compat API at localhost:8000/v1
    openai   : https://api.openai.com/v1 (needs $OPENAI_API_KEY)
    openrouter : https://openrouter.ai/api/v1 (needs $OPENROUTER_API_KEY)
    jev      : official TypeSafe Jev cloud API (needs $TYPESAFE_API_KEY)
    """
    if backend == "local":
        return LocalJev(
            model_id=_resolve_local_model(model),
            dtype=dtype,
            device=device,
        )
    if backend == "jev":
        from openjev.remote import RemoteJev

        return RemoteJev(api_key=api_key, model=jev_model)
    if backend in KNOWN_LOCAL_ENDPOINTS:
        url = base_url or KNOWN_LOCAL_ENDPOINTS[backend]
        return OpenAICompatJev(
            base_url=url,
            model=model or os.environ.get("OPENJEV_EASY_MODEL", "qwen3:0.6b"),
            api_key=api_key,
        )
    if backend == "openai":
        return OpenAICompatJev(
            base_url=base_url or "https://api.openai.com/v1",
            model=model or os.environ.get("OPENJEV_EASY_MODEL", "gpt-4o-mini"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        )
    if backend == "openrouter":
        return OpenAICompatJev(
            base_url=base_url or "https://openrouter.ai/api/v1",
            model=model or os.environ.get("OPENJEV_EASY_MODEL", "qwen/qwen3-0.6b-04-28:free"),
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
        )
    raise ValueError("unknown backend: %s" % backend)
