#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteJev: client for the real TypeSafe Jev cloud API (api.typesafe.ai).

Same system_one() interface as LocalJev, so benchmarks and routing code can
switch between local masked-softmax and the official cloud model by changing
one constructor argument. Useful for A/B accuracy, latency and cost tests.

编写时间: 2026-09-23 11:28:08
脚本功能: implement RemoteJev (requests-free, stdlib urllib POST client) and
          question-spec serialization shared with the wire format.
参数: see RemoteJev.__init__ docstring.
输入格式: openjev.types question objects; state as str / dict / list.
输出格式: same answer dicts as LocalJev.system_one, plus usage passthrough.
依赖: stdlib urllib / json / os; openjev.types.
注意事项: the API key is read from the TYPESAFE_API_KEY environment variable
          (or passed explicitly); never hardcode keys. Jev is English-first.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from openjev.types import Choice, Noul, Question, Score, SystemOneRequest

DEFAULT_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"


def question_to_spec(question: Question) -> Dict[str, Any]:
    """Serialize one question object into the Jev wire-format spec dict."""
    spec: Dict[str, Any] = {"type": question.type_name,
                            "instructions": question.instructions}
    if question.type_name == "choice" and question.criteria:
        spec["criteria"] = dict(question.criteria)
    elif question.type_name == "score" and question.criteria:
        spec["criteria"] = list(question.criteria)
    elif question.type_name == "noul" and question.criteria:
        spec["criteria"] = str(question.criteria)
    return spec


def serialize_questions(questions: Dict[str, Question]) -> Dict[str, Dict[str, Any]]:
    """Serialize the full questions dict for the wire."""
    return {qid: question_to_spec(q) for qid, q in questions.items()}


class RemoteJev:
    """
    Jev cloud client with the same surface as LocalJev.

    Parameters
    ----------
    api_key : TypeSafe API key; defaults to $TYPESAFE_API_KEY.
    endpoint : full systemone URL; override for gateways / proxies.
    model : "jev-latest" (default) or a pinned version like "jev-1.13.0".
    timeout : request timeout in seconds (default 60).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_JEV_ENDPOINT,
        model: str = DEFAULT_JEV_MODEL,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "no API key: pass api_key= or set the TYPESAFE_API_KEY env var"
            )
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.usage = {"forward_passes": 0, "input_tokens": 0, "latency_ms_total": 0.0}

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One POST to the systemone endpoint with retry on 429/5xx."""
        body = json.dumps(payload).encode("utf-8")
        last_error: Optional[Exception] = None
        for attempt in range(3):
            req = urllib.request.Request(
                self.endpoint,
                data=body,
                headers={
                    "Authorization": "Bearer %s" % self.api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # 429 rate limit / transient 5xx: honor retry-after, back off
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

    def system_one(self, state: Any, questions: Dict[str, Question]) -> Dict[str, Any]:
        """
        Answer many typed questions against one state via the Jev API.

        Returns {"model", "answers", "usage"} shaped like LocalJev output;
        answers are keyed by question ids exactly like the official wire.
        """
        request = SystemOneRequest(state=state, questions=questions)
        request.validate()
        payload = {
            "state": state,
            "model": self.model,
            "questions": serialize_questions(request.questions),
        }
        t0 = time.perf_counter()
        raw = self._post(payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_usage = raw.get("usage", {})
        self.usage["forward_passes"] += len(questions)
        self.usage["input_tokens"] += int(raw_usage.get("input_tokens", 0))
        self.usage["latency_ms_total"] += elapsed_ms
        return {
            "model": raw.get("model", self.model),
            "answers": raw.get("answers", {}),
            "usage": dict(self.usage),
        }
