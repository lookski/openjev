#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenJev: turn any local LLM into a Jev-style "System One" decision engine.

This module re-exports the public API of the openjev package.

编写时间: 2026-09-22 18:29:43
脚本功能: provide the single-line import surface (`from openjev import ...`)
          for the decision engine built on masked-logit softmax of local LLMs.
参数: not a script, import-only module.
输入格式: python import.
输出格式: public names listed in __all__.
依赖: openjev.core, openjev.types.
注意事项: the heavy transformers import happens lazily inside openjev.core,
          so `import openjev` stays cheap for schema-only usage.
"""

from openjev.types import (
    Choice,
    Noul,
    Question,
    Score,
    SystemOneRequest,
)
from openjev.core import LocalJev, answer_question
from openjev.remote import RemoteJev

__version__ = "0.1.0"

__all__ = [
    "Choice",
    "Noul",
    "Score",
    "Question",
    "LocalJev",
    "RemoteJev",
    "SystemOneRequest",
    "answer_question",
    "__version__",
]
