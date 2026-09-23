#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webapp: single-file yin-yang detector web UI, stdlib only.

编写时间: 2026-09-23 21:17:38
脚本功能: serve a dark-themed single-page UI at http://127.0.0.1:8790 :
          paste any message, get yin-yang / hostility / vibe probabilities
          rendered as animated bars. Engine loads lazily on first request.
参数: --host (default 127.0.0.1) --port (default 8790) --model
输入格式: POST /api/judge {"text": "..."}; GET / for the page; GET /health.
输出格式: HTML page; JSON answers for the API route.
依赖: stdlib http.server / json; openjev (torch, transformers lazy).
注意事项: pure-local entertainment tool; binds 127.0.0.1 by default.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openjev.core import LocalJev
from openjev.types import Choice, Noul, Score

_HERE = __file__
_REPO_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
_DEFAULT_MODEL = __import__("os").path.join(_REPO_DIR, "models", "Qwen3-0.6B")

_ENGINE = None
_LOCK = threading.Lock()
_ENGINE_KWARGS = {"model_id": None, "dtype": "float32", "device": "cpu"}

QUESTIONS = {
    "yin_yang": Noul(instructions="这条消息是不是在阴阳怪气 (表面客气实际讽刺)?"),
    "hostility": Score(
        instructions="说话人的敌意程度",
        criteria=[
            "没有敌意, 正常交流",
            "有点不爽, 但还在忍",
            "明显不满, 讽刺意味浓",
            "敌意拉满, 就差骂人了",
        ],
    ),
    "vibe": Choice(
        instructions="这条消息的真实语气更像",
        criteria={
            "sincere": "真诚的, 表面和实际一致",
            "passive_aggressive": "被动攻击, 用客气包裹不满",
            "pure_sarcasm": "纯讽刺, 明摆着阴阳怪气",
        },
    ),
}

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenJev - yin-yang detector</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; display:flex; align-items:center;
    justify-content:center; background:#0d1117; color:#e6edf3;
    font-family:"Segoe UI",system-ui,sans-serif; }
  .card { width:min(680px,94vw); background:#161b22; border:1px solid #30363d;
    border-radius:14px; padding:28px; }
  h1 { margin:0 0 4px; font-size:26px; }
  h1 span { color:#3fb950; }
  .sub { color:#8b949e; font-size:14px; margin-bottom:18px; }
  textarea { width:100%; min-height:88px; background:#0d1117; color:#e6edf3;
    border:1px solid #30363d; border-radius:10px; padding:12px; font-size:15px;
    resize:vertical; }
  button { margin-top:12px; width:100%; padding:11px; font-size:15px;
    background:#238636; color:#fff; border:0; border-radius:10px; cursor:pointer; }
  button:disabled { background:#2ea04366; cursor:wait; }
  .row { margin-top:14px; }
  .label { font-size:13px; color:#8b949e; margin-bottom:4px; }
  .track { background:#21262d; border-radius:6px; height:16px; overflow:hidden; }
  .fill { height:100%; border-radius:6px; width:0;
    transition:width .7s cubic-bezier(.2,.8,.2,1); }
  .green { background:#3fb950; } .orange { background:#d29922; }
  .blue { background:#58a6ff; } .grey { background:#30363d; }
  .val { font-family:Consolas,monospace; font-size:13px; color:#e6edf3; margin-top:2px; }
  .verdict { margin-top:16px; padding:12px; border-radius:10px; display:none;
    background:#0d1117; border:1px solid #30363d; font-size:14px; color:#e6edf3; }
  .err { color:#f85149; font-size:13px; margin-top:10px; display:none; }
  .foot { margin-top:16px; font-size:12px; color:#8b949e; text-align:center; }
  .foot a { color:#58a6ff; text-decoration:none; }
</style>
</head>
<body>
<div class="card">
  <h1>OpenJev <span>阴阳怪气鉴定器</span></h1>
  <div class="sub">本地小模型 · 原始 softmax 概率 · 数据不出你的电脑</div>
  <textarea id="text" placeholder="粘贴你想鉴定的消息..."></textarea>
  <button id="go" onclick="judge()">开 鉴</button>
  <div class="err" id="err"></div>
  <div id="out">
    <div class="row"><div class="label">阴阳怪气概率</div>
      <div class="track"><div id="b-yy" class="fill green"></div></div>
      <div class="val" id="v-yy"></div></div>
    <div class="row"><div class="label">敌意指数 (0-3 加权)</div>
      <div class="track"><div id="b-h" class="fill orange"></div></div>
      <div class="val" id="v-h"></div></div>
    <div class="row"><div class="label">语气判定</div>
      <div class="track"><div id="b-v" class="fill blue"></div></div>
      <div class="val" id="v-v"></div></div>
  </div>
  <div class="verdict" id="verdict"></div>
  <div class="foot">powered by <a href="https://github.com/lookski/openjev">openjev</a>
    · masked-logit softmax · 0.6B brain · numbers may be funnier than expected</div>
</div>
<script>
function bar(id, vid, p, pct) {
  document.getElementById(id).style.width = pct + "%";
  document.getElementById(vid).textContent = p.toFixed(4);
}
async function judge() {
  const t = document.getElementById("text").value.trim();
  const err = document.getElementById("err");
  const btn = document.getElementById("go");
  if (!t) { err.textContent = "先输入一句话"; err.style.display = "block"; return; }
  err.style.display = "none"; btn.disabled = true; btn.textContent = "鉴定中...";
  try {
    const r = await fetch("/api/judge", {method:"POST",
      headers:{"Content-Type":"application/json"}, body: JSON.stringify({text:t})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.status);
    const a = d.answers;
    const yy = a.yin_yang.noul;
    const hv = a.hostility.probabilities;
    const h = Object.keys(hv).reduce((s,k)=>s+k*hv[k],0);
    const vb = a.vibe;
    bar("b-yy","v-yy",yy, Math.max(2, yy*100));
    bar("b-h","v-h", h/3, Math.max(2, h/3*100));
    const best = vb.probabilities[vb.choice];
    bar("b-v","v-v", best, Math.max(2, best*100));
    document.getElementById("v-v").textContent =
      vb.choice + " " + best.toFixed(4) + "  (" +
      Object.entries(vb.probabilities).map(([k,p])=>k+" "+p.toFixed(3)).join(" | ") + ")";
    const v = document.getElementById("verdict");
    let line;
    if (yy > 0.75) line = "鉴定结论: 阴气重, 建议迂回周旋.";
    else if (yy > 0.45) line = "鉴定结论: 半阴半阳, 概率说了不算, 你自己掂量.";
    else line = "鉴定结论: 阳气充足, 可以放心接话.";
    v.textContent = line + "  (conflicts? that is the point - raw probabilities, no black box)";
    v.style.display = "block";
  } catch(e) {
    err.textContent = "error: " + e.message + " (first request loads the model, wait a moment)";
    err.style.display = "block";
  } finally {
    btn.disabled = false; btn.textContent = "开 鉴";
  }
}
</script>
</body>
</html>"""


def get_engine():
    """Lazy engine creation, thread-safe."""
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            kwargs = dict(_ENGINE_KWARGS)
            if not kwargs["model_id"]:
                kwargs["model_id"] = (
                    _DEFAULT_MODEL
                    if __import__("os").path.isdir(_DEFAULT_MODEL)
                    else "Qwen/Qwen3-0.6B"
                )
            _ENGINE = LocalJev(**kwargs)
        return _ENGINE


class Handler(BaseHTTPRequestHandler):
    """Routes: GET / , GET /health , POST /api/judge."""

    server_version = "openjev-web/0.1.0"

    def log_message(self, fmt, *args):  # quieter
        print("[webapp] %s" % (fmt % args))

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self._json(200, {"status": "ok", "loaded": _ENGINE is not None})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/api/judge":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(data.get("text", "")).strip()
            if not text:
                raise ValueError("'text' is required")
            if len(text) > 4000:
                raise ValueError("text too long (max 4000 chars)")
            engine = get_engine()
            result = engine.system_one(text, QUESTIONS)
            self._json(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # engine errors keep the server alive
            self._json(500, {"error": "%s: %s" % (type(exc).__name__, exc)})


def main() -> None:
    """Entry: python webapp.py [--host H] [--port P] [--model M]."""
    parser = argparse.ArgumentParser(description="OpenJev yin-yang web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    _ENGINE_KWARGS["model_id"] = args.model

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("[webapp] yin-yang detector at http://%s:%d" % (args.host, args.port))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
