#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for openjev: schema, confidence math, question parsing, engine smoke.

编写时间: 2026-09-22 18:38:54
脚本功能: verify the type layer (Choice/Score/Noul validation, confidence and
          score math, answer rendering), the server question parser, and
          (if model weights are available) the real masked-softmax engine.
参数: run with `pytest tests/ -v` or `python -m pytest tests/ -v`.
输入格式: none.
输出格式: pytest results on stdout.
依赖: pytest; torch + transformers + model weights for the smoke tests.
注意事项: engine smoke tests skip automatically when models/Qwen3-0.6B is
          absent, so CI and fresh clones stay green.
"""

import os

import pytest

from openjev.core import _pick_first_token_ids
from openjev.types import (
    Choice,
    Noul,
    Score,
    confidence_from_probs,
    entropy_nats,
    render_answer,
    weighted_score,
)
from openjev.server import parse_question, parse_questions

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODEL = os.path.join(HERE, "models", "Qwen3-0.6B")


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Question contract tests."""

    def test_choice_rejects_single_option(self):
        with pytest.raises(ValueError):
            Choice(instructions="pick", criteria={"only": "one"}).validate()

    def test_choice_rejects_empty(self):
        with pytest.raises(ValueError):
            Choice(instructions="pick", criteria={}).validate()

    def test_choice_accepts_two_options(self):
        Choice(instructions="pick", criteria={"a": "x", "b": "y"}).validate()

    def test_score_rejects_one_level(self):
        with pytest.raises(ValueError):
            Score(instructions="rate", criteria=["only"]).validate()

    def test_score_rejects_eleven_levels(self):
        with pytest.raises(ValueError):
            Score(instructions="rate", criteria=["l"] * 11).validate()

    def test_score_accepts_two_levels(self):
        Score(instructions="rate", criteria=["low", "high"]).validate()

    def test_empty_instructions_rejected(self):
        with pytest.raises(ValueError):
            Noul(instructions="   ").validate()

    def test_request_rejects_empty_questions(self):
        from openjev.types import SystemOneRequest

        with pytest.raises(ValueError):
            SystemOneRequest(state="s", questions={}).validate()


# ---------------------------------------------------------------------------
# math helpers
# ---------------------------------------------------------------------------

class TestMath:
    """Confidence and score formulas."""

    def test_confidence_peaked_three_options(self):
        # official example: max 1.0 with 3 options -> (3 - 1) / 2 = 1.0
        assert confidence_from_probs([1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_confidence_uniform_three_options(self):
        # (1/3 * 3 - 1) / 2 = 0.0
        assert confidence_from_probs([1 / 3] * 3) == pytest.approx(0.0, abs=1e-9)

    def test_confidence_binary_range(self):
        assert confidence_from_probs([0.5, 0.5]) == pytest.approx(0.0)
        assert confidence_from_probs([1.0, 0.0]) == pytest.approx(1.0)

    def test_score_weighted_mean(self):
        # official example: level1 0.57 + level2 0.43 -> 1.43
        assert weighted_score([0.0, 0.57, 0.43]) == pytest.approx(1.43)

    def test_entropy_zero_when_peaked(self):
        assert entropy_nats([1.0, 0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# answer rendering
# ---------------------------------------------------------------------------

class TestRender:
    """Answer dict shape tests (Jev wire compatibility)."""

    def test_noul_answer_shape(self):
        ans = render_answer(Noul(instructions="u?"), [0.99, 0.01])
        assert ans == {"type": "noul", "noul": 0.99}
        assert "confidence" not in ans  # Jev: noul has no confidence field

    def test_choice_answer_shape(self):
        q = Choice(instructions="d", criteria={"a": "x", "b": "y"})
        ans = render_answer(q, [0.7, 0.3])
        assert ans["choice"] == "a"
        assert ans["probabilities"] == {"a": 0.7, "b": 0.3}
        assert 0.0 <= ans["confidence"] <= 1.0

    def test_score_answer_shape(self):
        q = Score(instructions="r", criteria=["low", "mid", "high"])
        ans = render_answer(q, [0.1, 0.5, 0.4])
        assert ans["score"] == pytest.approx(1.3)
        assert ans["legend"] == ["low", "mid", "high"]
        assert ans["probabilities"]["2"] == 0.4


# ---------------------------------------------------------------------------
# server parsing
# ---------------------------------------------------------------------------

class TestParse:
    """Server-side question dict parsing."""

    def test_parse_choice(self):
        q = parse_question(
            "dept",
            {
                "type": "choice",
                "instructions": "which",
                "criteria": {"a": "x", "b": "y"},
            },
        )
        assert isinstance(q, Choice)

    def test_parse_noul(self):
        q = parse_question("u", {"type": "noul", "instructions": "urgent?"})
        assert isinstance(q, Noul)

    def test_parse_unknown_type(self):
        with pytest.raises(ValueError):
            parse_question("x", {"type": "magic", "instructions": "?"})

    def test_parse_questions_rejects_empty(self):
        with pytest.raises(ValueError):
            parse_questions({})

    def test_parse_questions_multi(self):
        qs = parse_questions(
            {
                "a": {"type": "noul", "instructions": "n?"},
                "b": {"type": "score", "instructions": "s?", "criteria": ["l", "h"]},
            }
        )
        assert set(qs) == {"a", "b"}


# ---------------------------------------------------------------------------
# engine (needs weights; skips gracefully)
# ---------------------------------------------------------------------------

def _model_available() -> bool:
    return os.path.isdir(LOCAL_MODEL) and os.path.isfile(
        os.path.join(LOCAL_MODEL, "config.json")
    )


@pytest.fixture(scope="module")
def engine():
    from openjev.core import LocalJev

    return LocalJev(model_id=LOCAL_MODEL, dtype="float32", device="cpu")


@pytest.mark.skipif(not _model_available(), reason="model weights not downloaded")
class TestEngineSmoke:
    """Real masked-softmax engine tests against the local snapshot."""

    def test_letter_token_ids_unique(self, engine):
        ids = engine._letter_ids[3]
        assert len(ids) == 3 and len(set(ids)) == 3

    def test_yesno_token_ids(self, engine):
        assert len(engine._yesno_ids) == 2

    def test_noul_probs_sum_to_one(self, engine):
        ans = engine.answer_one(
            Noul(instructions="Does this message express urgency?"),
            "The server is down, fix it now.",
        )
        assert 0.0 <= ans["noul"] <= 1.0

    def test_choice_probs_sum_to_one(self, engine):
        q = Choice(
            instructions="Which team should handle this",
            criteria={"billing": "invoices", "technical": "API bugs"},
        )
        ans = engine.answer_one(q, "My API returns 500 errors all day.")
        total = sum(ans["probabilities"].values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_system_one_multi(self, engine):
        result = engine.system_one(
            "I want a refund for order #123.",
            {
                "dept": Choice(
                    instructions="Which team",
                    criteria={"billing": "refunds", "tech": "bugs"},
                ),
                "urgent": Noul(instructions="Is this urgent?"),
            },
        )
        assert set(result["answers"]) == {"dept", "urgent"}
        assert result["model"].startswith("openjev/")

    def test_pick_first_token_ids_rejects_multitoken(self):
        class FakeTok:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [1, 2] if len(text) > 1 else [7]

        with pytest.raises(ValueError):
            _pick_first_token_ids(FakeTok(), ["ab", "c"])


# ---------------------------------------------------------------------------
# easy mode: OpenAI-compatible logprobs backend (fake server, no network)
# ---------------------------------------------------------------------------

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from openjev.easy import (
    OpenAICompatJev,
    _decode_token,
    _probs_from_top,
    make_engine,
)


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    """Configurable fake OpenAI-compatible server."""

    reply_for = None  # set per test

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        body = json.dumps({"data": [{"id": "fake-mini"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        body = json.dumps(type(self).reply_for(payload)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeServer:
    """Context manager wrapper around _FakeOpenAIHandler."""

    def __init__(self, reply_for):
        handler = type("H", (_FakeOpenAIHandler,), {"reply_for": reply_for})
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()


class TestEasyHelpers:
    """Offline helpers of the OpenAI-compat backend."""

    def test_decode_bytes_token(self):
        # "Yes" as utf-8 bytes, the OpenAI logprobs convention
        assert _decode_token({"bytes": [89, 101, 115]}) == "Yes"

    def test_decode_plain_token(self):
        assert _decode_token("No") == "No"

    def test_probs_from_top_normalizes(self):
        probs = _probs_from_top([("Yes", -0.01), ("No", -5.2)], ["Yes", "No"])
        assert abs(sum(probs) - 1.0) < 1e-9
        assert probs[0] > 0.9

    def test_probs_missing_label_returns_none(self):
        # honest fallback: never invent probabilities for missing labels
        assert _probs_from_top([("The", -0.1)], ["Yes", "No"]) is None

    def test_make_engine_openai_compat(self):
        engine = make_engine("ollama", model="qwen3:0.6b")
        assert isinstance(engine, OpenAICompatJev)
        assert engine.base_url == "http://localhost:11434/v1"

    def test_make_engine_unknown(self):
        import pytest

        with pytest.raises(ValueError):
            make_engine("magic-brain")


class TestEasyEndToEnd:
    """Full wizard path against a fake OpenAI-compatible server."""

    def test_e2e_full(self):
        # content-aware fake server: noul + choice + score all answered
        def reply_for(body):
            text = body["messages"][1]["content"]
            if "Yes or No" in text:
                toks = [{"token": {"bytes": [89, 101, 115]}, "logprob": -0.02},
                        {"token": {"bytes": [78, 111]}, "logprob": -4.5}]
            elif "one letter" in text:
                toks = [{"token": "A", "logprob": -8.0},
                        {"token": "B", "logprob": -7.5},
                        {"token": "C", "logprob": -0.05},
                        {"token": "D", "logprob": -6.0}]
            else:
                toks = [{"token": "0", "logprob": -3.0},
                        {"token": "1", "logprob": -0.3},
                        {"token": "2", "logprob": -1.6}]
            return {"choices": [{"logprobs": {"content": [{"top_logprobs": toks}]}}],
                    "usage": {"prompt_tokens": 42}}

        with _FakeServer(reply_for) as server:
            engine = OpenAICompatJev(base_url=server.url, model="fake-mini", api_key="test")
            result = engine.system_one(
                "The server is down, fix it NOW.",
                {
                    "u": Noul(instructions="Is this urgent?"),
                    "c": Choice(instructions="Which", criteria={
                        "a": "one", "b": "two", "c": "three", "d": "four"}),
                    "s": Score(instructions="Rate", criteria=["low", "mid", "high"]),
                },
            )
        answers = result["answers"]
        assert answers["u"]["noul"] > 0.9
        assert answers["c"]["choice"] == "c"
        assert abs(sum(answers["s"]["probabilities"].values()) - 1.0) < 1e-6
        assert result["model"].startswith("openai-compat/")

    def test_e2e_missing_label_raises(self):
        # a server that always answers Yes/No: choice questions must fail loudly
        def reply_for(_body):
            toks = [{"token": {"bytes": [89, 101, 115]}, "logprob": -0.02},
                    {"token": {"bytes": [78, 111]}, "logprob": -4.5}]
            return {"choices": [{"logprobs": {"content": [{"top_logprobs": toks}]}}]}

        with _FakeServer(reply_for) as server:
            engine = OpenAICompatJev(base_url=server.url, model="fake-mini")
            import pytest

            with pytest.raises(RuntimeError):
                engine.answer_one(
                    Choice(instructions="Which", criteria={"a": "x", "b": "y"}),
                    "state",
                )
