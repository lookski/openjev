#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Easy wizard: the "download and just use it" entry point for OpenJev.

Zero-code flow: detect local inference servers (Ollama / LM Studio / vLLM /
llama.cpp), let the user pick a brain, paste an API key if needed, run a
smoke test, then drop into a paste-anything REPL that answers typed
questions with raw probabilities.

编写时间: 2026-09-23 11:49:40
脚本功能: interactive backend selection, key prompt (getpass), smoke test,
          and an interactive REPL (plain text -> default questions, JSON ->
          full spec, :help :demo :backend :logout :quit).
参数: python -m openjev.easy_cli [--once TEXT] [--backend NAME] [--model NAME]
          [--base-url URL] [--yes]
输入格式: terminal interaction; --once for non-interactive one-shot.
输出格式: stdout answers with probability bars; config saved to ~/.openjev/config.json.
依赖: openjev.easy (make_engine, OpenAICompatJev), openjev.core, openjev.types.
注意事项: ASCII-only output (Windows GBK consoles); API keys are stored
          plaintext in ~/.openjev/config.json with a warning, :logout erases.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from openjev.core import DEFAULT_MODEL_ID
from openjev.easy import KNOWN_LOCAL_ENDPOINTS, make_engine
from openjev.types import Choice, Noul, Score

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".openjev")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

BANNER = r"""
  ___             _     ___
 / _ \ _ __  ___ (_) __| _ \_____ __ _
| (_) | '_ \/ _ \| |/ _` \ \ \ / _` |
 \__, |_| .__/\__/|_|\__,_/_\_\ \__,_|
 |___/|_|   v0.1 - local decision engine
"""

MENU = """
Pick a brain (the thing that answers):

  [1] Ollama          (local server at localhost:11434)
  [2] LM Studio       (local server at localhost:1234)
  [3] llama.cpp       (local server at localhost:8080)
  [4] vLLM            (local server at localhost:8000)
  [5] OpenAI          (api.openai.com, needs sk-... key)
  [6] OpenRouter      (openrouter.ai, free models available)
  [7] Jev cloud API   (official TypeSafe Jev, needs TYPESAFE_API_KEY)
  [8] Built-in engine (downloads ~1.5 GB once, 100%% offline, no key)
"""

DEFAULT_QUESTIONS = {
    "intent": Choice(
        instructions="What does the author of this message want",
        criteria={
            "question": "Asks for information or help",
            "complaint": "Reports a problem or expresses dissatisfaction",
            "request": "Asks us to do something or approve something",
            "chat": "Small talk or anything else",
        },
    ),
    "urgent": Noul(instructions="Does this message express urgency?"),
    "sentiment": Score(
        instructions="How positive is the message",
        criteria=[
            "Negative, frustrated or angry",
            "Neutral, informational",
            "Positive, happy or grateful",
        ],
    ),
}

ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "jev": "TYPESAFE_API_KEY",
}


# ---------------------------------------------------------------------------
# tiny helpers (ASCII-only for GBK consoles)
# ---------------------------------------------------------------------------

def say(msg: str = "") -> None:
    """Print one line, GBK-safe."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def probe(base_url: str, timeout: float = 0.8) -> bool:
    """True when an OpenAI-compatible server answers /models."""
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/models")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def list_models(base_url: str) -> List[str]:
    """Model ids from an OpenAI-compatible /models endpoint."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("id", "?") for m in data.get("data", [])]
    except Exception:
        return []


def bar(p: float, width: int = 24) -> str:
    """ASCII probability bar computed from the real probability."""
    filled = max(1, round(p * width))
    return "#" * filled + "." * (width - filled)


def show_answers(answers: Dict[str, Any]) -> None:
    """Pretty-print answers with probability bars."""
    for qid, ans in answers.items():
        say("")
        if ans["type"] == "noul":
            p = ans["noul"]
            say("  %-12s Yes %s %.4f" % (qid, bar(p), p))
            say("  %-12s No  %s %.4f" % ("", bar(1.0 - p), 1.0 - p))
        elif ans["type"] == "choice":
            for name, p in ans["probabilities"].items():
                mark = "*" if name == ans["choice"] else " "
                say("  %-12s %s %s %.4f" % (qid, mark, bar(p), p))
            say("  -> choice=%s (confidence %.4f)" % (ans["choice"], ans["confidence"]))
        elif ans["type"] == "score":
            for lvl, p in ans["probabilities"].items():
                say("  %-12s level %s %s %.4f" % (qid, lvl, bar(p), p))
            say("  -> score=%.4f (confidence %.4f)" % (ans["score"], ans["confidence"]))


def save_config(cfg: Dict[str, Any]) -> None:
    """Persist wizard choices (backend/model/base_url/key) for next time."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, ensure_ascii=False, indent=2)
    try:
        os.chmod(CONFIG_PATH, 0o600)  # POSIX; Windows ignores silently
    except OSError:
        pass
    say("[saved] %s" % CONFIG_PATH)


def load_config() -> Dict[str, Any]:
    """Load saved wizard choices, if any."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def ask_key(env_name: str, hint: str) -> str:
    """Take the key from env first, else hidden input."""
    env_key = os.environ.get(env_name, "")
    if env_key:
        say("[ok] using %s from environment" % env_name)
        return env_key
    say(hint)
    key = getpass.getpass("Paste your API key (input hidden): ").strip()
    if not key:
        raise SystemExit("no key given, bye")
    return key


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------

def detect_running() -> Dict[str, List[str]]:
    """Probe known local endpoints; return {backend: [model ids]}."""
    found: Dict[str, List[str]] = {}
    for backend, url in KNOWN_LOCAL_ENDPOINTS.items():
        if probe(url):
            found[backend] = list_models(url)
    return found


def pick_backend(args: argparse.Namespace) -> Dict[str, Any]:
    """Interactive backend + model + key selection. Returns a config dict."""
    saved = load_config()
    if getattr(args, "backend", None):
        cfg = {"backend": args.backend, "model": args.model,
               "base_url": args.base_url, "api_key": None}
        return _complete_cfg(cfg, args)

    running = detect_running()
    say(BANNER)
    if running:
        for backend, models in running.items():
            say("[detected] %s running at %s (%d model(s))"
                % (backend, KNOWN_LOCAL_ENDPOINTS[backend], len(models)))
    if saved:
        say("[saved] last time you used %s (%s)" % (saved.get("backend"), saved.get("model")))

    say(MENU)
    default_choice = "1" if "ollama" in running else ("8" if not running else "")
    raw = input("Choose [1-8]%s: " % (
        " (default %s)" % default_choice if default_choice else "")).strip() or default_choice
    if raw not in {"1", "2", "3", "4", "5", "6", "7", "8"}:
        raise SystemExit("invalid choice: %r" % raw)

    mapping = {"1": "ollama", "2": "lmstudio", "3": "llamacpp", "4": "vllm",
               "5": "openai", "6": "openrouter", "7": "jev", "8": "local"}
    backend = mapping[raw]
    cfg: Dict[str, Any] = {"backend": backend, "model": None,
                           "base_url": None, "api_key": None}

    if backend in KNOWN_LOCAL_ENDPOINTS:
        models = running.get(backend) or list_models(KNOWN_LOCAL_ENDPOINTS[backend])
        if models:
            say("")
            for i, mid in enumerate(models, 1):
                say("  [%d] %s" % (i, mid))
            pick = input("Which model? [1-%d] (default 1): " % len(models)).strip() or "1"
            cfg["model"] = models[int(pick) - 1]
        else:
            cfg["model"] = input("Model name (e.g. qwen3:0.6b): ").strip()
    elif backend in ("openai", "openrouter", "jev"):
        env_name = ENV_KEYS[backend]
        cfg["api_key"] = ask_key(
            env_name,
            "[info] no %s in environment. Get a key from the provider dashboard." % env_name,
        )
        if backend == "jev":
            cfg["model"] = (input("Jev model [jev-latest]: ").strip() or "jev-latest")
    else:  # local built-in
        say("[info] will download %s on first use (~1.5 GB, cached)." % DEFAULT_MODEL_ID)

    return _complete_cfg(cfg, args)


def _complete_cfg(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides and defaults onto a partial config."""
    if args.model and not cfg.get("model"):
        cfg["model"] = args.model
    if args.base_url and not cfg.get("base_url"):
        cfg["base_url"] = args.base_url
    if args.api_key and not cfg.get("api_key"):
        cfg["api_key"] = args.api_key
    if cfg["backend"] in ("openai", "openrouter", "jev") and not cfg.get("api_key"):
        # custom endpoints (self-hosted vLLM etc.) may not need any key
        if cfg["backend"] == "jev" or not cfg.get("base_url"):
            cfg["api_key"] = ask_key(
                ENV_KEYS[cfg["backend"]], "[info] key required."
            )
        else:
            say("[info] custom endpoint without API key (no Authorization header).")
    return cfg


def build_engine(cfg: Dict[str, Any]):
    """Instantiate the engine from a wizard config dict."""
    if cfg["backend"] == "jev":
        return make_engine("jev", api_key=cfg.get("api_key"),
                           jev_model=cfg.get("model") or "jev-latest")
    if cfg["backend"] == "local":
        return make_engine("local", model=cfg.get("model"))
    if cfg["backend"] in ("openai", "openrouter"):
        return make_engine(cfg["backend"], model=cfg.get("model"),
                           base_url=cfg.get("base_url"),
                           api_key=cfg.get("api_key"))
    # ollama / lmstudio / llamacpp / vllm and any custom OpenAI-compatible URL
    return make_engine(cfg["backend"], model=cfg.get("model"),
                       base_url=cfg.get("base_url"))


# ---------------------------------------------------------------------------
# smoke test + REPL
# ---------------------------------------------------------------------------

def smoke_test(engine, label: str) -> bool:
    """One real question; returns True when the backend works end to end."""
    say("")
    say("[test] asking %s: 'The server is down and we are losing money.' -> urgent?" % label)
    try:
        ans = engine.answer_one(
            Noul(instructions="Does this message express urgency?"),
            "The server is down and we are losing money.",
        )
        say("[ok]  noul = %.4f  (works!)" % ans["noul"])
        return True
    except Exception as exc:
        say("[FAIL] %s: %s" % (type(exc).__name__, exc))
        say("> 中文解释: 后端没有返回可用概率; 检查服务是否在线 / 模型名是否正确 / key 是否有效.")
        return False


HELP = """
Just paste any English text and I will judge it:
  intent    -> question / complaint / request / chat   (choice)
  urgent    -> yes/no urgency                          (noul)
  sentiment -> negative / neutral / positive           (score)

Or paste full JSON:
  {"state": "...", "questions": {"id": {"type": "noul|choice|score",
                                        "instructions": "...",
                                        "criteria": ...}}}

Commands:
  :demo      run the built-in ticket example
  :backend   show current backend
  :logout    erase saved config (API key)
  :help      this help
  :quit      leave
"""


def run_spec(engine, raw_json: str) -> None:
    """Answer a pasted JSON spec (state+questions, or bare questions)."""
    data = json.loads(raw_json)
    if isinstance(data, dict) and "questions" in data:
        state = data.get("state", "")
        qspecs = data["questions"]
    elif isinstance(data, dict) and data and all(
        isinstance(v, dict) and "type" in v for v in data.values()
    ):
        state, qspecs = "", data
    else:
        raise ValueError("JSON must contain a 'questions' object")
    questions = {}
    for qid, spec in qspecs.items():
        qtype = spec.get("type")
        if qtype == "noul":
            questions[qid] = Noul(instructions=spec.get("instructions", ""),
                                  criteria=spec.get("criteria"))
        elif qtype == "choice":
            questions[qid] = Choice(instructions=spec.get("instructions", ""),
                                    criteria=spec.get("criteria"))
        elif qtype == "score":
            questions[qid] = Score(instructions=spec.get("instructions", ""),
                                   criteria=spec.get("criteria"))
        else:
            raise ValueError("question '%s': unknown type %r" % (qid, qtype))
    result = engine.system_one(state, questions)
    show_answers(result["answers"])


def repl(engine, backend_label: str) -> None:
    """Interactive loop: paste text or JSON, get typed probabilities."""
    say("")
    say("=" * 64)
    say("OpenJev ready. backend: %s" % backend_label)
    say(HELP)
    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            say("")
            break
        if not line:
            continue
        if line.startswith(":"):
            cmd = line.lower()
            if cmd in (":q", ":quit", ":exit"):
                break
            if cmd == ":help":
                say(HELP)
            elif cmd == ":demo":
                run_demo(engine)
            elif cmd == ":backend":
                say("backend: %s" % backend_label)
            elif cmd == ":logout":
                if os.path.exists(CONFIG_PATH):
                    os.remove(CONFIG_PATH)
                say("[ok] saved config erased.")
            else:
                say("unknown command %r, try :help" % line)
            continue
        try:
            if line.startswith("{"):
                run_spec(engine, line)
            else:
                result = engine.system_one(line, DEFAULT_QUESTIONS)
                show_answers(result["answers"])
        except KeyboardInterrupt:
            say("")
        except Exception as exc:
            say("[error] %s: %s" % (type(exc).__name__, exc))
            say("> 中文解释: 输入没有解析成功或后端报错; JSON 需要含 questions 对象.")
    say("bye.")


def run_demo(engine) -> None:
    """Canonical Jev ticket example."""
    state = (
        "Hi, I have been trying to connect my Stripe account for 3 days and "
        "the integration keeps failing. I am losing sales. Please help ASAP."
    )
    questions = {
        "department": Choice(
            instructions="Which team should handle this",
            criteria={
                "billing": "Payments, invoices, refunds",
                "technical": "Integration errors, API failures, bugs",
                "sales": "Pricing questions, upgrades, new purchases",
            },
        ),
        "urgent": Noul(instructions="Does this message express urgency?"),
    }
    say("state: %s" % state)
    result = engine.system_one(state, questions)
    show_answers(result["answers"])


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def main() -> None:
    """Wizard entry: python -m openjev.easy_cli."""
    parser = argparse.ArgumentParser(description="OpenJev easy wizard")
    parser.add_argument("--once", metavar="TEXT", help="non-interactive: answer TEXT once")
    parser.add_argument("--backend", help="skip the menu: ollama|lmstudio|llamacpp|vllm|openai|openrouter|jev|local")
    parser.add_argument("--model", help="model name override")
    parser.add_argument("--base-url", help="custom OpenAI-compatible base URL")
    parser.add_argument("--api-key", help="API key (otherwise $ENV or hidden prompt)")
    parser.add_argument("--json-spec", help="with --once: full JSON spec instead of plain text")
    args = parser.parse_args()

    cfg = pick_backend(args)
    label = cfg["backend"] + (("/" + cfg["model"]) if cfg.get("model") else "")
    engine = build_engine(cfg)

    if not smoke_test(engine, label):
        raise SystemExit(1)

    interactive = sys.stdin.isatty() and not args.once
    if interactive and cfg["backend"] != "local" and \
            input("Save this setup for next time? [Y/n]: ").strip().lower() != "n":
        save_config(cfg)

    if args.once:
        if args.json_spec:
            run_spec(engine, args.json_spec)
        else:
            result = engine.system_one(args.once, DEFAULT_QUESTIONS)
            show_answers(result["answers"])
        return
    repl(engine, label)


if __name__ == "__main__":
    main()
