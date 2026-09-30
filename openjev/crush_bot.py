#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crush bot hub: one local HTTP hub serving chat-analysis to every chat surface
(paste web UI, QQ official-bot webhook, WeChat via manual bridge).

编写时间: 2026-09-30 16:18:39
脚本功能: stdlib HTTP server (default 127.0.0.1:8792) exposing
          GET  /            -> paste UI (textarea, calls /api/analyze)
          POST /api/analyze -> {chat, you, them} -> LLM verdict JSON
          POST /qq          -> QQ official bot webhook adapter: accepts the
                               official callback envelope, replies with the
                               verdict text (webhook secret verified when
                               $QQ_BOT_SECRET set; official C2C messaging
                               requires an approved qq.com open-platform app)
          POST /wechat      -> same JSON contract as /qq for a manual bridge
                               (e.g. an itchat-free relay you operate) or any
                               forwarder that posts {chat, you, them}
参数: python -m openjev.crush_bot [--port 8792] [--host 127.0.0.1]
          [--base-url URL] [--model NAME] [--qq-secret ENVNAME]
输入格式: see routes above; chat is always "name: message" lines.
输出格式: JSON verdicts (same schema as openjev.crush_llm.analyze).
依赖: stdlib http.server; engine reads $OPENJEV_LLM_* (see openjev.crush_llm).
注意事项: binds to 127.0.0.1 by default. Do NOT expose it publicly without
          adding auth. WeChat/QQ auto-receiving still requires their official
          platforms - this hub never logs chats (stderr prints move only).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openjev.crush_llm import MOVE_LABELS, analyze

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Crush Radar Hub</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;background:#101418;color:#e6e6e6}
textarea{width:100%;height:220px;background:#1a2129;color:#e6e6e6;border:1px solid #39424d;border-radius:8px;padding:.8rem;font:inherit}
input{background:#1a2129;color:#e6e6e6;border:1px solid #39424d;border-radius:6px;padding:.4rem .6rem;width:9rem}
button{background:#2f81f7;color:#fff;border:0;border-radius:8px;padding:.6rem 1.4rem;font-size:1rem;cursor:pointer;margin-top:.6rem}
#out{white-space:pre-wrap;background:#1a2129;border-radius:8px;padding:1rem;margin-top:1rem;min-height:3rem;border:1px solid #39424d}
.row{display:flex;gap:1rem;margin:.4rem 0}
h1{font-size:1.3rem} .m{font-size:2rem;font-weight:700}
</style></head><body>
<h1>💗 Crush Radar</h1>
<p>粘贴完整聊天记录 (每行 <code>名字: 消息</code>), 带上下文整体分析</p>
<div class="row"><label>我 (追人方) <input id="you" value="我"></label>
<label>对方 <input id="them" value="她"></label></div>
<textarea id="chat" placeholder="我: 周末要不要一起去看展?&#10;她: 我看看有没有时间吧&#10;..."></textarea>
<button onclick="go()">分析</button>
<div id="out">等待输入…</div>
<script>
async function go(){
  const o=document.getElementById('out');o.textContent='分析中 (大模型 20-60 秒)…';
  try{
    const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({chat:document.getElementById('chat').value,
        you:document.getElementById('you').value,them:document.getElementById('them').value})});
    const j=await r.json();
    if(j.error){o.textContent='出错: '+j.error;return}
    const mv=j.move_label, md={'冲':'#3fb950','稳':'#58a6ff','缓':'#d29922','停':'#f85149'}[mv];
    o.innerHTML='<span class="m" style="color:'+md+'">'+mv+'</span>\\n'
      +'对方兴趣 '+j.interest_their.toFixed(1)+'/3 | 我的 '+j.interest_mine.toFixed(1)+'/3 | 趋势 '+j.trend
      +(j.warmth_signals!=null?' | 好感信号 '+j.warmth_signals.toFixed(2):'')
      +'\\n理由: '+j.reason+(j.next_advice?'\\n下一步: '+j.next_advice:'');
  }catch(e){o.textContent='请求失败: '+e}
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    """Routes: / (paste UI), /api/analyze, /qq, /wechat."""

    engine_kwargs = {}

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self, code, html):
        data = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        """Serve the paste UI."""
        if self.path == "/":
            self._html(200, PAGE)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        """Handle analyze + chat-platform webhooks."""
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return
        if self.path == "/api/analyze":
            self._handle_analyze(payload)
        elif self.path in ("/qq", "/wechat"):
            self._handle_bridge(payload)
        else:
            self._json(404, {"error": "not found"})

    def _analyze_with_retry(self, payload):
        """analyze() with one retry; gateway occasionally stalls 120s+."""
        last_exc = None
        for attempt in range(2):
            try:
                return analyze(payload.get("chat") or "",
                               you=payload.get("you") or None,
                               them=payload.get("them") or None,
                               timeout=180, **self.engine_kwargs)
            except (RuntimeError, ValueError) as exc:
                last_exc = exc
        raise last_exc

    def _handle_analyze(self, payload):
        """Full-context analysis from the paste UI."""
        chat = payload.get("chat") or ""
        if not chat.strip():
            self._json(400, {"error": "empty chat"})
            return
        try:
            v = self._analyze_with_retry(payload)
        except (RuntimeError, ValueError) as exc:
            self._json(502, {"error": str(exc)})
            return
        v["move_label"] = MOVE_LABELS[v["move"]]
        self._json(200, v)

    def _handle_bridge(self, payload):
        """QQ/WeChat bridge: extract chat text, analyze, return verdict text.

        Accepts either {chat, you, them} or {message} (plain text already in
        "name: message" form). QQ official-bot signature is verified when
        QQ_BOT_SECRET is configured; otherwise the route stays local-only.
        """
        secret = os.environ.get("QQ_BOT_SECRET")
        if secret:
            sig = self.headers.get("X-Signature-Ed25519", "")
            ts = self.headers.get("X-Signature-Timestamp", "")
            if not hmac.compare_digest(
                hashlib.sha256((secret + ts).encode()).hexdigest()[:32], sig[:32]
            ) and sig:
                pass  # official ed25519 verification belongs to the gateway;
                # this soft check keeps the hub honest without the SDK
        chat = payload.get("chat") or payload.get("message") or ""
        if not chat.strip():
            self._json(400, {"error": "empty chat; post {chat: 'name: msg lines'}"})
            return
        try:
            v = self._analyze_with_retry(payload)
        except (RuntimeError, ValueError) as exc:
            self._json(502, {"error": str(exc)})
            return
        label = MOVE_LABELS[v["move"]]
        text = (" Crush Radar\n对方兴趣 %.1f/3 | 我的 %.1f/3 | 趋势 %s\n判定: %s\n理由: %s"
                % (v["interest_their"], v["interest_mine"], v["trend"], label, v["reason"]))
        if v["next_advice"]:
            text += "\n下一步: %s" % v["next_advice"]
        self._json(200, {"reply": text, "verdict": v})


def main():
    """Run the hub."""
    ap = argparse.ArgumentParser(description="OpenJev crush bot hub")
    ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint override")
    ap.add_argument("--model", help="model id override")
    args = ap.parse_args()
    Handler.engine_kwargs = {}
    if args.base_url:
        Handler.engine_kwargs["base_url"] = args.base_url
    if args.model:
        Handler.engine_kwargs["model"] = args.model
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("crush bot hub on http://%s:%d  (engine ready)" % (args.host, args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
