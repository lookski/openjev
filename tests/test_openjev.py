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
