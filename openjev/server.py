#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local HTTP server exposing the Jev-style /v1/systemone endpoint.

Wire format is drop-in compatible with the TypeSafe Jev API shape:
POST /v1/systemone  {"state": ..., "questions": {...}} -> {"answers": {...}}.
Questions are parsed from plain dicts, so any HTTP client works without the
python dataclass layer.

编写时间: 2026-09-22 18:29:43
脚本功能: serve the masked-softmax engine over HTTP; also provide
          parse_question / parse_questions helpers shared by tests.
参数: --host (default 127.0.0.1), --port (default 8771),
      --model (default Qwen/Qwen3-0.6B or $OPENJEV_MODEL),
      --dtype (default float32), --device (default auto).
输入格式: JSON body {"state": str|obj, "questions": {id: {type, instructions, criteria}}}.
输出格式: JSON {"model", "answers", "usage"}; errors as {"error": "..."}.
依赖: stdlib http.server / json; openjev.core (torch, transformers lazy).
注意事项: GET /health returns model load state; the engine loads lazily on
          first request (or eagerly with --preload).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openjev.core import DEFAULT_MODEL_ID, LocalJev
from openjev.types import QUESTION_TYPES, Noul, Choice, Score

_ENGINE: LocalJev = None  # type: ignore[assignment]
_ENGINE_LOCK = threading.Lock()
_ENGINE_ARGS = {
    "backend": "local",
    "model_id": DEFAULT_MODEL_ID,
    "dtype": "float32",
    "device": None,
    "jev_model": "jev-latest",
}


def get_engine():
    """Create the process-wide engine on first use, thread-safely."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            if _ENGINE_ARGS["backend"] == "jev":
                from openjev.remote import RemoteJev

                _ENGINE = RemoteJev(model=_ENGINE_ARGS["jev_model"])
            else:
                _ENGINE = LocalJev(
                    model_id=_ENGINE_ARGS["model_id"],
                    dtype=_ENGINE_ARGS["dtype"],
                    device=_ENGINE_ARGS["device"],
                )
        return _ENGINE


def parse_question(qid: str, spec: dict) -> Choice | Score | Noul:
    """Parse one question spec dict into a typed question object."""
    if not isinstance(spec, dict):
        raise ValueError("question '%s' must be an object" % qid)
    qtype = spec.get("type")
    if qtype not in QUESTION_TYPES:
        raise ValueError(
            "question '%s': unknown type %r (expected choice|score|noul)"
            % (qid, qtype)
        )
    cls = QUESTION_TYPES[qtype]
    q = cls(
        instructions=str(spec.get("instructions", "")).strip(),
        criteria=spec.get("criteria"),
    )
    try:
        q.validate()
    except ValueError as exc:
        raise ValueError("question '%s': %s" % (qid, exc)) from exc
    return q


def parse_questions(raw: dict) -> dict:
    """Parse the whole questions dict; ids are keys, never sent to the model."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError("'questions' must be a non-empty object")
    return {qid: parse_question(qid, spec) for qid, spec in raw.items()}


class SystemOneHandler(BaseHTTPRequestHandler):
    """HTTP handler for /v1/systemone and /health."""

    server_version = "openjev/0.1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "engine_loaded": _ENGINE is not None,
                    "model_id": _ENGINE_ARGS["model_id"],
                },
            )
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path != "/v1/systemone":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            questions = parse_questions(payload.get("questions", {}))
            if "state" not in payload:
                raise ValueError("'state' is required")
            engine = get_engine()
            result = engine.system_one(payload["state"], questions)
            self._send_json(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # keep the server alive on engine errors
            self._send_json(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        print("[openjev-server] %s" % (fmt % args))


def main(argv=None) -> None:
    """CLI entry: python -m openjev.server [--host H] [--port P] [--model M]."""
    parser = argparse.ArgumentParser(description="OpenJev local systemone server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "jev"],
        help="local masked-softmax engine, or proxy to the Jev cloud API",
    )
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--model", default=os.environ.get("OPENJEV_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default=None, help="cpu | cuda | mps, default auto")
    parser.add_argument("--preload", action="store_true", help="load the model before serving")
    args = parser.parse_args(argv)

    _ENGINE_ARGS["backend"] = args.backend
    _ENGINE_ARGS["jev_model"] = args.jev_model
    _ENGINE_ARGS["model_id"] = args.model
    _ENGINE_ARGS["dtype"] = args.dtype
    _ENGINE_ARGS["device"] = args.device

    if args.preload:
        t0 = time.perf_counter()
        get_engine()
        print("[openjev-server] model loaded in %.1fs" % (time.perf_counter() - t0))

    httpd = ThreadingHTTPServer((args.host, args.port), SystemOneHandler)
    print("[openjev-server] listening on http://%s:%d" % (args.host, args.port))
    print("[openjev-server] POST /v1/systemone  |  GET /health")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
