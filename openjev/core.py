#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Masked-logit softmax decision engine: the core of OpenJev.

Loads a local causal LM once, then answers Jev-style typed questions
(Choice / Score / Noul) with a single forward pass per question. The answer
probabilities are raw softmax values of the masked next-token logits:
no text generation, no sampling, type errors are impossible.

编写时间: 2026-09-22 18:29:43
脚本功能: implement LocalJev (model loader + masked softmax answering) and the
          answer_question convenience helper.
参数: see LocalJev.__init__ docstring; answer_question(question, state, model).
输入格式: openjev.types question objects; state as str / dict / list.
输出格式: Jev-shaped answer dicts (see openjev.types.render_answer).
依赖: torch, transformers (imported lazily); openjev.types.
注意事项:
- reasoning / thinking chat templates are disabled so the first generated
  position is the answer token itself;
- logits_to_keep=1 is used when supported to save memory, with a fallback;
- on CPU keep dtype float32; pass dtype="float16" only on GPU.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from openjev.types import (
    Choice,
    Noul,
    Question,
    Score,
    SystemOneRequest,
    render_answer,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise decision engine. "
    "You do not generate text. "
    "You answer only with the exact label requested."
)

# label pools; choice supports up to 26 letters in one forward pass
LETTERS = [chr(ord("A") + i) for i in range(26)]
YESNO = ["Yes", "No"]
DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def _pick_first_token_ids(tokenizer, labels: List[str]) -> List[int]:
    """
    Map each label string to its first token id under the tokenizer.

    Single characters like "A" or "0" always encode to one token in
    sentencepiece/BPE vocabularies; assert to fail loudly otherwise.
    """
    ids: List[int] = []
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                "label %r encodes to %d tokens, expected 1"
                % (label, len(token_ids))
            )
        ids.append(token_ids[0])
    if len(set(ids)) != len(ids):
        raise ValueError("label token ids collide: %r" % (labels,))
    return ids


def _build_prompt(
    tokenizer,
    state_text: str,
    question: Question,
    system_prompt: str,
) -> str:
    """Build the chat prompt for one question via the tokenizer template."""
    parts = ["<state>", state_text, "</state>", "", "Question:", question.instructions]
    if question.type_name == "choice":
        names = list(question.criteria.keys())
        lines = []
        for i, name in enumerate(names):
            lines.append("%s. %s" % (LETTERS[i], question.criteria[name]))
        parts += ["", "\n".join(lines), "", "Answer with exactly one letter."]
    elif question.type_name == "score":
        lines = []
        for i, desc in enumerate(question.criteria):
            lines.append("%s = %s" % (DIGITS[i], desc))
        parts += [
            "",
            "\n".join(lines),
            "",
            "Answer with exactly one digit for the best level.",
        ]
    elif question.type_name == "noul":
        if question.criteria:
            parts += ["", str(question.criteria)]
        parts += ["", "Answer with exactly one word, Yes or No."]
    else:
        raise ValueError("unknown question type: %s" % question.type_name)

    user_text = "\n".join(parts)
    try:
        # thinking disabled: the first generated position must be the answer
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # template without enable_thinking support
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )


def _forward_last_logits(model, inputs) -> Any:
    """Run one forward pass and return logits at the last position."""
    try:
        out = model(**inputs, logits_to_keep=1)
        return out.logits[0, -1, :]
    except TypeError:
        out = model(**inputs)
        return out.logits[0, -1, :]


class LocalJev:
    """
    A Jev-style decision engine backed by a local causal LM.

    Parameters
    ----------
    model_id : HF model id or local path; default Qwen/Qwen3-0.6B.
    device : "cpu" / "cuda" / "mps" / None (auto).
    dtype : "float32" (default, CPU-safe) / "float16" / "bfloat16".
    system_prompt : override the default decision-engine system prompt.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        dtype: str = "float32",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.system_prompt = system_prompt

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        torch_dtype = getattr(torch, dtype)
        load_kwargs: Dict[str, Any] = {"dtype": torch_dtype}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, **load_kwargs
            )
        except TypeError:
            # transformers < 5 uses torch_dtype
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype
            )
        self.model.to(device)
        self.model.eval()

        # precompute label token ids for every supported arity
        self._letter_ids = {
            k: _pick_first_token_ids(self.tokenizer, LETTERS[:k]) for k in range(2, 27)
        }
        self._yesno_ids = _pick_first_token_ids(self.tokenizer, YESNO)
        self._digit_ids = _pick_first_token_ids(self.tokenizer, DIGITS)
        self.usage = {"forward_passes": 0, "input_tokens": 0, "latency_ms_total": 0.0}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _masked_probs(self, prompt: str, label_token_ids: List[int]) -> List[float]:
        """One forward pass; softmax over the masked label logits only."""
        torch = self.torch
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        n_tokens = int(inputs["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.no_grad():
            last_logits = _forward_last_logits(self.model, inputs)
        logits = last_logits.to(torch.float32)
        mask = torch.full_like(logits, float("-inf"))
        for tid in label_token_ids:
            mask[tid] = 0.0
        masked = logits + mask
        probs = torch.softmax(masked, dim=-1)
        picked = probs[label_token_ids]
        picked = picked / picked.sum()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.usage["forward_passes"] += 1
        self.usage["input_tokens"] += n_tokens
        self.usage["latency_ms_total"] += elapsed_ms
        return [float(p) for p in picked.tolist()]

    def answer_one(self, question: Question, state: Any) -> Dict[str, Any]:
        """Answer a single typed question against the given state."""
        question.validate()
        if isinstance(state, str):
            state_text = state
        else:
            import json

            state_text = json.dumps(state, ensure_ascii=False)

        if question.type_name == "choice":
            k = len(question.criteria)
            prompt = _build_prompt(self.tokenizer, state_text, question, self.system_prompt)
            probs = self._masked_probs(prompt, self._letter_ids[k])
        elif question.type_name == "score":
            k = len(question.criteria)
            prompt = _build_prompt(self.tokenizer, state_text, question, self.system_prompt)
            probs = self._masked_probs(prompt, self._digit_ids[:k])
        elif question.type_name == "noul":
            prompt = _build_prompt(self.tokenizer, state_text, question, self.system_prompt)
            probs = self._masked_probs(prompt, self._yesno_ids)
        else:
            raise ValueError("unknown question type: %s" % question.type_name)
        return render_answer(question, probs)

    def system_one(self, state: Any, questions: Dict[str, Question]) -> Dict[str, Any]:
        """
        Answer many typed questions against one state (sequentially).

        Returns {"answers": {...}, "usage": {...}}; answers are keyed by the
        question ids exactly like the Jev wire format.
        """
        request = SystemOneRequest(state=state, questions=questions)
        request.validate()
        answers: Dict[str, Any] = {}
        for qid, question in request.questions.items():
            answers[qid] = self.answer_one(question, state)
        return {
            "model": "openjev/%s" % self.model_id,
            "answers": answers,
            "usage": dict(self.usage),
        }


# ---------------------------------------------------------------------------
# module-level convenience
# ---------------------------------------------------------------------------

_DEFAULT_ENGINE: Optional[LocalJev] = None


def get_default_engine(model_id: str = DEFAULT_MODEL_ID, **kwargs) -> LocalJev:
    """Lazily create (and reuse) a process-wide default engine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None or _DEFAULT_ENGINE.model_id != model_id:
        _DEFAULT_ENGINE = LocalJev(model_id=model_id, **kwargs)
    return _DEFAULT_ENGINE


def answer_question(
    question: Question,
    state: Any,
    model: Optional[LocalJev] = None,
) -> Dict[str, Any]:
    """
    One-shot helper: answer a single question, loading a default engine
    on first use if none is passed.
    """
    engine = model if model is not None else get_default_engine()
    return engine.answer_one(question, state)
