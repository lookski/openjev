#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type-safe question primitives and response schema for OpenJev.

The dataclasses mirror the TypeSafe Jev systemone wire format:
Choice / Score / Noul questions answered from a masked softmax over a
local LLM's next-token logits.

编写时间: 2026-09-22 18:29:43
脚本功能: define Choice, Score, Noul question types, the SystemOneRequest /
          SystemOneResponse envelopes, plus helpers (confidence formula,
          weighted score, answer rendering) shared by engine, CLI and server.
参数: not a script, import-only module.
输入格式: python import.
输出格式: python objects; answers render as plain dicts.
依赖: dataclasses, json, math, typing.
注意事项: question ids live only in the answers dict (never sent to the model),
          matching Jev's official behavior.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# confidence and score helpers
# ---------------------------------------------------------------------------

def confidence_from_probs(probs: List[float]) -> float:
    """
    Jev-style confidence: how peaked the distribution is, in [0, 1].

    Generalization of the official 3-option formula (max_p * K - 1) / (K - 1)
    to any K >= 2. For K = 2 this reduces to (max_p - 0.5) * 2, i.e. how far
    the binary distribution is from uniform.
    """
    k = len(probs)
    if k < 2:
        return 1.0
    return max(0.0, min(1.0, (max(probs) * k - 1.0) / (k - 1.0)))


def weighted_score(probs: List[float]) -> float:
    """
    Jev-style score: the index-weighted mean of the probability vector.

    Example: [0.57, 0.43] -> 0.43, [0.0, 0.57, 0.43] -> 1.43.
    """
    return sum(i * p for i, p in enumerate(probs))


# ---------------------------------------------------------------------------
# question primitives
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """Base class for all typed questions. Do not instantiate directly."""

    type_name = "question"
    instructions: str
    criteria: Optional[Any] = None
    # max option letters (Choice) / levels (Score); mirrors Jev limits
    MAX_OPTIONS: int = 255
    MAX_SCORE_LEVELS: int = 10

    def validate(self) -> None:
        """Validate the question; raise ValueError on contract violations."""
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("instructions must be a non-empty string")

    def answer(self) -> Dict[str, Any]:
        """Empty answer skeleton, filled by the engine."""
        return {"type": self.type_name}


@dataclass
class Choice(Question):
    """Pick one option from a fixed set; criteria maps name -> description."""

    type_name = "choice"
    criteria: Dict[str, str] = None  # type: ignore[assignment]

    def validate(self) -> None:
        super().validate()
        if not self.criteria or not isinstance(self.criteria, dict):
            raise ValueError("Choice.criteria must be a non-empty dict")
        if len(self.criteria) < 2:
            raise ValueError(
                "Choice needs at least 2 options, got %d" % len(self.criteria)
            )
        if len(self.criteria) > self.MAX_OPTIONS:
            raise ValueError(
                "Choice supports at most %d options, got %d"
                % (self.MAX_OPTIONS, len(self.criteria))
            )


@dataclass
class Score(Question):
    """Rate on ordered levels; criteria is a low-to-high list (2..10)."""

    type_name = "score"
    criteria: List[str] = None  # type: ignore[assignment]

    def validate(self) -> None:
        super().validate()
        if not self.criteria or not isinstance(self.criteria, list):
            raise ValueError("Score.criteria must be a non-empty list")
        if len(self.criteria) < 2:
            raise ValueError("Score needs at least 2 levels")
        if len(self.criteria) > self.MAX_SCORE_LEVELS:
            raise ValueError(
                "Score supports at most %d levels, got %d"
                % (self.MAX_SCORE_LEVELS, len(self.criteria))
            )


@dataclass
class Noul(Question):
    """Yes/no judgement; criteria optionally explains true/false meaning."""

    type_name = "noul"


QUESTION_TYPES = {"choice": Choice, "score": Score, "noul": Noul}


# ---------------------------------------------------------------------------
# wire format
# ---------------------------------------------------------------------------

@dataclass
class SystemOneRequest:
    """A systemone request: one state, many typed questions."""

    state: Any
    questions: Dict[str, Question]

    def validate(self) -> None:
        if not self.questions:
            raise ValueError("questions must not be empty")
        for qid, q in self.questions.items():
            try:
                q.validate()
            except ValueError as exc:
                raise ValueError("question '%s': %s" % (qid, exc)) from exc

    def state_text(self) -> str:
        """Render the state as text; dicts/lists are dumped as compact JSON."""
        if isinstance(self.state, str):
            return self.state
        return json.dumps(self.state, ensure_ascii=False)


def render_answer(question: Question, probs: List[float]) -> Dict[str, Any]:
    """
    Build a Jev-shaped answer dict from the masked probability vector.

    choice -> {type, choice, probabilities, confidence}
    score  -> {type, score, probabilities, confidence, legend}
    noul   -> {type, noul}  (binary, no confidence, like Jev)
    """
    conf = round(confidence_from_probs(probs), 4)
    if question.type_name == "noul":
        return {"type": "noul", "noul": round(probs[0], 4)}
    if question.type_name == "choice":
        names = list(question.criteria.keys())
        best = names[probs.index(max(probs))]
        return {
            "type": "choice",
            "choice": best,
            "probabilities": {n: round(p, 4) for n, p in zip(names, probs)},
            "confidence": conf,
        }
    if question.type_name == "score":
        levels = list(question.criteria)
        return {
            "type": "score",
            "score": round(weighted_score(probs), 4),
            "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
            "confidence": conf,
            "legend": levels,
        }
    raise ValueError("unknown question type: %s" % question.type_name)


def probs_to_logprob_str(probs: List[float]) -> str:
    """Render the masked probability vector as a compact logprob string."""
    return " ".join("%.4f" % p for p in probs)


def entropy_nats(probs: List[float]) -> float:
    """Shannon entropy of the masked distribution in nats (QC / diagnostics)."""
    return float(-sum(p * math.log(p) for p in probs if p > 0.0))
