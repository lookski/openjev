#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM config center: configure ANY OpenAI-compatible endpoint ONCE, use it
everywhere (crush_llm CLI, crush_bot hub, future engines).

编写时间: 2026-09-30 17:40:38
脚本功能: persistent config at ~/.openjev/llm.json storing base_url, model,
          api_key (0600 file). resolve() applies the priority chain
          CLI args > environment variables > saved config. interactive
          wizard: ask for base URL (+key if the server demands one), fetch
          GET /models to list available models, let the user pick one, run
          a one-word smoke test, save. configure() does the same
          non-interactively. Wizard also reachable from the hub web UI.
参数: resolve(base_url=None, model=None, api_key=None) -> dict or raises
          ConfigError; wizard(base_url=None, model=None, api_key=None,
          stdin=sys.stdin, stdout=sys.stdout) -> dict; configure(...)=>
          non-interactive save (no probes); CLI:
          python -m openjev.llm_config            # interactive wizard
          python -m openjev.llm_config --status   # show saved config (masked)
输入格式: interactive terminal input; non-interactive keyword args.
输出格式: dict {base_url, model, api_key}; CLI prints masked status.
依赖: stdlib only (urllib, ssl, json, os, getpass).
注意事项: api_key is stored PLAINTEXT in ~/.openjev/llm.json with 0600
          permissions (Windows: user-profile ACL) - same trust level as
          ~/.netrc / gh CLI auth. Never commit or echo the file. Smoke test
          sends one word to the configured endpoint.
"""

from __future__ import annotations

import json
import os
import ssl
import stat
import sys
import urllib.error
import urllib.request

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".openjev", "llm.json")
SMOKE_PROMPT = "Reply with exactly one word: ready"
SMOKE_EXPECT_TIMEOUT = 60


class ConfigError(RuntimeError):
    """Raised when no usable LLM endpoint can be resolved."""


def _read_file():
    """Load saved config; return {} when absent/corrupt."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_file(cfg):
    """Persist config with 0600-style permissions; create ~/.openjev first."""
    d = os.path.dirname(CONFIG_PATH)
    os.makedirs(d, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, ensure_ascii=False, indent=1)
    try:
        os.chmod(CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 0600 on posix
    except OSError:
        pass  # Windows: profile-dir ACL already user-only by default
    return cfg


def load():
    """Return the saved config dict (possibly empty)."""
    return _read_file()


def mask(cfg):
    """Human-safe view: show base_url/model, mask the key."""
    key = cfg.get("api_key") or ""
    shown = (key[:3] + "***" + key[-4:]) if len(key) > 9 else ("***" if key else "")
    return {"base_url": cfg.get("base_url", ""), "model": cfg.get("model", ""),
            "api_key": shown, "config_path": CONFIG_PATH}


def resolve(base_url=None, model=None, api_key=None):
    """Merge CLI args > env > saved file into an effective config.

    Raises ConfigError when base_url or model is still missing; api_key may
    legitimately be empty (self-hosted servers often need none).
    """
    saved = _read_file()
    eff = {
        "base_url": (base_url
                     or os.environ.get("OPENJEV_LLM_BASE_URL")
                     or saved.get("base_url") or "").rstrip("/"),
        "model": (model
                  or os.environ.get("OPENJEV_LLM_MODEL")
                  or saved.get("model") or ""),
        "api_key": (api_key
                    or os.environ.get("OPENJEV_LLM_API_KEY")
                    or saved.get("api_key") or ""),
    }
    if not eff["base_url"]:
        raise ConfigError(
            "no LLM endpoint configured. Run: python -m openjev.llm_config "
            "(one-time wizard), or set $OPENJEV_LLM_BASE_URL / pass --base-url"
        )
    if not eff["model"]:
        raise ConfigError(
            "no LLM model configured. Re-run: python -m openjev.llm_config, "
            "or set $OPENJEV_LLM_MODEL / pass --model"
        )
    return eff


def _http(url, payload=None, api_key="", timeout=SMOKE_EXPECT_TIMEOUT):
    """GET/POST JSON against an OpenAI-compatible endpoint; return parsed."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    resp = urllib.request.urlopen(
        req, timeout=timeout, context=ssl.create_default_context()
    )
    return json.loads(resp.read())


def fetch_models(base_url, api_key="", timeout=30):
    """GET {base}/models; return sorted model-id list (OpenAI schema)."""
    data = _http(base_url.rstrip("/") + "/models", api_key=api_key, timeout=timeout)
    ids = []
    for item in data.get("data", []):
        mid = item.get("id") if isinstance(item, dict) else None
        if mid:
            ids.append(mid)
    return sorted(ids)


def smoke_test(base_url, model, api_key="", timeout=SMOKE_EXPECT_TIMEOUT):
    """One-word chat completion; return (ok, detail)."""
    try:
        data = _http(
            base_url.rstrip("/") + "/chat/completions",
            payload={"model": model,
                     "messages": [{"role": "user", "content": SMOKE_PROMPT}],
                     "max_tokens": 20},
            api_key=api_key,
            timeout=timeout,
        )
        text = (data["choices"][0]["message"].get("content") or "").strip()
        return True, text[:40] or "(empty reply but endpoint works)"
    except Exception as exc:  # noqa: BLE001 - report any failure verbatim
        return False, str(exc)[:200]


def configure(base_url, model, api_key="", run_smoke=True):
    """Non-interactive save with optional smoke test; returns (cfg, ok, detail)."""
    cfg = {"base_url": base_url.rstrip("/"), "model": model, "api_key": api_key or ""}
    ok, detail = (True, "smoke test skipped") if not run_smoke else smoke_test(
        cfg["base_url"], cfg["model"], cfg["api_key"]
    )
    if ok:
        _write_file(cfg)
    return cfg, ok, detail


def wizard(base_url=None, model=None, api_key=None, stdin=sys.stdin, stdout=sys.stdout):
    """Interactive one-time setup: URL -> (key) -> /models list -> pick -> smoke -> save."""
    def ask(prompt, default=""):
        suffix = " [%s]" % default if default else ""
        stdout.write(prompt + suffix + ": ")
        stdout.flush()
        val = (stdin.readline() or "").strip()
        return val or default

    saved = _read_file()
    base_url = base_url or ask(
        "OpenAI-compatible base URL (e.g. https://api.deepseek.com/v1)",
        saved.get("base_url", ""),
    ).rstrip("/")
    if not base_url:
        raise ConfigError("base URL is required")
    api_key = api_key
    if api_key is None:
        need_key = ask("Does this endpoint need an API key? (y/N)", "n").lower() == "y"
        if need_key:
            import getpass
            api_key = getpass.getpass("API key (hidden input): ")
        else:
            api_key = saved.get("api_key", "")
    # fetch model list; on failure fall back to manual entry
    models = []
    try:
        models = fetch_models(base_url, api_key or "")
        if models:
            stdout.write("available models (%d), first 30:\n" % len(models))
            for i, mid in enumerate(models[:30], 1):
                stdout.write("  %2d. %s\n" % (i, mid))
    except Exception as exc:  # noqa: BLE001 - wizard must survive bad endpoints
        stdout.write("could not list models (%s)\n" % str(exc)[:100])
    if model is None:
        if models:
            pick = ask("pick model number or type a name", "1")
            if pick.isdigit() and 1 <= int(pick) <= min(30, len(models)):
                model = models[int(pick) - 1]
            else:
                model = pick
        else:
            model = ask("model name", saved.get("model", ""))
    if not model:
        raise ConfigError("model is required")
    stdout.write("smoke testing %s @ %s ...\n" % (model, base_url))
    ok, detail = smoke_test(base_url, model, api_key or "")
    if not ok:
        raise ConfigError("smoke test FAILED, nothing saved: %s" % detail)
    stdout.write("smoke OK (%s). saved -> %s\n" % (detail, CONFIG_PATH))
    cfg = _write_file({"base_url": base_url, "model": model, "api_key": api_key or ""})
    return cfg


def main():
    """CLI: wizard by default, --status to inspect, --show to include key."""
    import argparse
    ap = argparse.ArgumentParser(description="OpenJev LLM endpoint setup (one-time)")
    ap.add_argument("--status", action="store_true", help="show saved config (key masked)")
    ap.add_argument("--show", action="store_true", help="status with full key visible")
    ap.add_argument("--base-url", help="non-interactive base URL")
    ap.add_argument("--model", help="non-interactive model")
    ap.add_argument("--api-key", help="non-interactive API key")
    args = ap.parse_args()
    if args.status or args.show:
        cfg = load()
        if not cfg:
            print("no config saved yet at %s" % CONFIG_PATH)
            return
        view = mask(cfg) if not args.show else dict(cfg, config_path=CONFIG_PATH)
        print(json.dumps(view, ensure_ascii=False, indent=1))
        return
    if args.base_url and args.model:
        cfg, ok, detail = configure(args.base_url, args.model, args.api_key or "")
        if not ok:
            raise SystemExit("smoke test FAILED, nothing saved: %s" % detail)
        print("saved -> %s (smoke: %s)" % (CONFIG_PATH, detail))
        return
    wizard()


if __name__ == "__main__":
    main()
