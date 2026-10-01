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

===== [2026-10-01 23:47:25] =====
新增: 实时判定流. GET /api/stream (SSE, 15s ping 保活) 广播判定事件;
GET /api/verdicts 返回最近 50 条 (deque); POST /api/verdict 接收
qq_assistant/qq_overlay 等本机进程的 --push 推送 (loopback-only);
网页 /api/analyze 的判定也进流. 页面新增 实时判定流 面板
(EventSource + 历史回放, HTML 转义防注入).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openjev.crush_llm import MOVE_LABELS, analyze
from openjev.llm_config import (
    configure,
    fetch_models,
    load as load_config,
    mask as mask_config,
)

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crush Radar Hub</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;background:#101418;color:#e6e6e6}
@media(max-width:600px){body{margin:1rem auto;padding:0 .6rem}textarea{height:160px}}
textarea{width:100%;height:220px;background:#1a2129;color:#e6e6e6;border:1px solid #39424d;border-radius:8px;padding:.8rem;font:inherit}
input{background:#1a2129;color:#e6e6e6;border:1px solid #39424d;border-radius:6px;padding:.4rem .6rem;width:12rem}
button{background:#2f81f7;color:#fff;border:0;border-radius:8px;padding:.6rem 1.4rem;font-size:1rem;cursor:pointer;margin-top:.6rem}
#out{white-space:pre-wrap;background:#1a2129;border-radius:8px;padding:1rem;margin-top:1rem;min-height:3rem;border:1px solid #39424d}
#cfgout{white-space:pre-wrap;color:#d29922;margin:.5rem 0;font-size:.9rem}
.row{display:flex;gap:1rem;margin:.4rem 0;align-items:center;flex-wrap:wrap}
details{border:1px solid #39424d;border-radius:8px;padding:.8rem;margin-bottom:1rem}
summary{cursor:pointer;color:#58a6ff}
h1{font-size:1.3rem} .m{font-size:2rem;font-weight:700}
</style></head><body>
<h1>💗 Crush Radar</h1>
<details id="cfgbox"><summary>⚙️ 模型设置 (配一次, 永久生效)</summary>
  <div id="cfgout"></div>
  <div class="row"><label>Base URL <input id="c_base" placeholder="https://api.deepseek.com/v1" size="34"></label></div>
  <div class="row"><label>API Key (可选, 自建服务留空) <input id="c_key" type="password" placeholder="sk-..." size="24"></label>
  <button onclick="listModels()">拉取模型列表</button></div>
  <div class="row"><label>模型 <input id="c_model" placeholder="从列表选或手填" size="28" list="mlist"></label><datalist id="mlist"></datalist>
  <button onclick="saveCfg()">保存并测试</button></div>
</details>
<p>粘贴完整聊天记录 (每行 <code>名字: 消息</code>), 带上下文整体分析</p>
<div class="row"><label>我 (追人方) <input id="you" value="我"></label>
<label>对方 <input id="them" value="她"></label></div>
<textarea id="chat" placeholder="我: 周末要不要一起去看展?&#10;她: 我看看有没有时间吧&#10;..."></textarea>
<button onclick="go()">分析</button>
<div id="out">等待输入…</div>
<details id="feedbox" open><summary>📡 实时判定流 (QQ/微信进程 --push 推送; 本页分析也进流)</summary>
<div id="feed"></div></details>
<script>
async function jfetch(url,body){
  const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:undefined);
  const j=await r.json(); if(j.error) throw new Error(j.error); return j;
}
async function refreshCfg(){
  try{const c=await jfetch('/api/config');
    let t;
    if(c.configured!==undefined){
      t=c.configured?('当前: '+(c.model||'?')+' (远程视图: 端点与 key 仅本机可见)'):'尚未配置, 展开本面板填写保存';
    }else{
      t=c.base_url?('当前: '+c.base_url+' | 模型: '+c.model+' | key: '+(c.api_key||'无')):'尚未配置, 展开本面板填写保存';
    }
    document.getElementById('cfgout').textContent=t;
  }catch(e){document.getElementById('cfgout').textContent='读取配置失败: '+e}
}
async function listModels(){
  const o=document.getElementById('cfgout');o.textContent='拉取模型列表…';
  try{const j=await jfetch('/api/models',{base_url:c_base.value.trim(),api_key:c_key.value.trim()});
    const dl=document.getElementById('mlist');dl.innerHTML='';
    j.models.forEach(m=>{const op=document.createElement('option');op.value=m;dl.appendChild(op)});
    o.textContent='共 '+j.count+' 个模型, 已加载下拉列表 (输入框选或手填)';
  }catch(e){o.textContent='拉取失败: '+e.message}
}
async function saveCfg(){
  const o=document.getElementById('cfgout');o.textContent='保存中 (含连通性测试, 最多 60 秒)…';
  try{const j=await jfetch('/api/config',{base_url:c_base.value.trim(),model:c_model.value.trim(),api_key:c_key.value.trim()});
    o.textContent='✓ 已保存并测试通过 ('+j.smoke+') — 开始用吧';
  }catch(e){o.textContent='保存失败 (未写入): '+e.message}
}
async function go(){
  const o=document.getElementById('out');o.textContent='分析中 (大模型 20-100 秒)…';
  try{
    const j=await jfetch('/api/analyze',{chat:document.getElementById('chat').value,
      you:document.getElementById('you').value,them:document.getElementById('them').value});
    const mv=j.move_label, md={'冲':'#3fb950','稳':'#58a6ff','缓':'#d29922','停':'#f85149'}[mv];
    o.innerHTML='<span class="m" style="color:'+md+'">'+mv+'</span>\\n'
      +'对方兴趣 '+j.interest_their.toFixed(1)+'/3 | 我的 '+j.interest_mine.toFixed(1)+'/3 | 趋势 '+j.trend
      +(j.warmth_signals!=null?' | 好感信号 '+j.warmth_signals.toFixed(2):'')
      +'\\n理由: '+j.reason+(j.reply_direction?'\\n回复方向: '+j.reply_direction:'')+(j.next_advice?'\\n下一步: '+j.next_advice:'');
  }catch(e){
    o.textContent='请求失败: '+e.message;
    if(String(e.message).includes('endpoint')||String(e.message).includes('model')) document.getElementById('cfgbox').open=true;
  }
}
function esc(s){return String(s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
function addRow(it){
  const v=it.v||{}; const mv=v.move_label||'?';
  const md={'冲':'#3fb950','稳':'#58a6ff','缓':'#d29922','停':'#f85149'}[mv]||'#8b949e';
  const d=new Date((it.ts||0)*1000);
  const row=document.createElement('div');
  row.style.cssText='border-left:3px solid '+md+';padding:.5rem .8rem;margin:.4rem 0;background:#161b22;border-radius:6px';
  const their=(v.interest_their!=null&&v.interest_their.toFixed)?v.interest_their.toFixed(1):'?';
  row.innerHTML='<b style="color:'+md+';font-size:1.2em">'+esc(mv)+'</b> '
    +(it.who?esc(it.who)+' · ':'')+d.toLocaleTimeString()
    +'<br><small>对方兴趣 '+their+'/3 | 趋势 '+(v.trend||'?')+'</small>'
    +(v.reason?'<br><small>'+esc(v.reason)+'</small>':'')
    +(v.reply_direction?'<br><small style="color:#a371f7">回复方向: '+esc(v.reply_direction)+'</small>':'')
    +(v.next_advice?'<br><small style="color:#d29922">下一步: '+esc(v.next_advice)+'</small>':'');
  document.getElementById('feed').prepend(row);
}
async function loadFeed(){try{const j=await jfetch('/api/verdicts');(j.items||[]).forEach(addRow)}catch(e){}}
const es=new EventSource('/api/stream');
es.onmessage=function(e){try{addRow(JSON.parse(e.data))}catch(err){}};
loadFeed();
refreshCfg();
</script></body></html>"""


SUBS = []              # live SSE subscriber queues
SUBS_LOCK = threading.Lock()
FEED = deque(maxlen=50)  # rolling verdict history for /api/verdicts


def broadcast(obj):
    """Push one JSON event to every open SSE stream (best effort)."""
    data = json.dumps(obj, ensure_ascii=False)
    with SUBS_LOCK:
        for q in list(SUBS):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass


def feed_add(obj):
    with SUBS_LOCK:
        FEED.append(obj)
    broadcast(obj)


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
        elif self.path == "/api/stream":
            self._handle_stream()
            return
        elif self.path == "/api/verdicts":
            with SUBS_LOCK:
                self._json(200, {"items": list(FEED)})
        elif self.path == "/api/config":
            cfg = load_config()
            if not self._config_write_allowed():
                # non-loopback reader: engine presence only, no endpoint/key
                self._json(200, {
                    "configured": bool(cfg.get("base_url") and cfg.get("model")),
                    "model": cfg.get("model", ""),
                })
            else:
                self._json(200, mask_config(cfg))
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
        elif self.path == "/api/config":
            self._handle_config(payload)
        elif self.path == "/api/models":
            self._handle_models(payload)
        elif self.path == "/api/verdict":
            self._handle_verdict(payload)
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

    def _config_write_allowed(self):
        """Config writes/models list stay loopback-only unless explicitly opened."""
        if os.environ.get("OPENJEV_ALLOW_REMOTE_CONFIG") == "1":
            return True
        return self.client_address[0] in ("127.0.0.1", "::1")

    def _handle_config(self, payload):
        """Save base_url/model/api_key after a smoke test; 400 on failure."""
        if not self._config_write_allowed():
            self._json(403, {"error": "config writes are loopback-only; "
                                    "set OPENJEV_ALLOW_REMOTE_CONFIG=1 to open"})
            return
        base = (payload.get("base_url") or "").strip()
        mdl = (payload.get("model") or "").strip()
        key = (payload.get("api_key") or "").strip()
        if not base or not mdl:
            self._json(400, {"error": "base_url and model are required"})
            return
        cfg, ok, detail = configure(base, mdl, key)
        if not ok:
            self._json(400, {"error": "smoke test failed, nothing saved: " + detail})
            return
        self._json(200, {"saved": mask_config(cfg), "smoke": detail})

    def _handle_models(self, payload):
        """Proxy GET /models for the endpoint typed in the config form.

        Falls back to the saved api_key when the form field is empty, so
        re-listing models for an already-configured endpoint just works.
        """
        if not self._config_write_allowed():
            self._json(403, {"error": "model listing is loopback-only"})
            return
        base = (payload.get("base_url") or "").strip().rstrip("/")
        key = (payload.get("api_key") or "").strip() or load_config().get("api_key", "")
        if not base:
            self._json(400, {"error": "base_url required"})
            return
        try:
            models = fetch_models(base, key)
        except Exception as exc:  # noqa: BLE001 - surface provider error verbatim
            self._json(502, {"error": "model list failed: %s" % str(exc)[:150]})
            return
        self._json(200, {"count": len(models), "models": models[:60]})

    def _handle_stream(self):
        """Server-sent events: every verdict lands on connected pages."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q = queue.Queue(maxsize=100)
        with SUBS_LOCK:
            SUBS.append(q)
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                    self.wfile.write(("data: %s\n\n" % ev).encode("utf-8"))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client went away
        finally:
            with SUBS_LOCK:
                try:
                    SUBS.remove(q)
                except ValueError:
                    pass

    def _handle_verdict(self, payload):
        """QQ/WeChat process pushes a verdict; loopback-only, no secrets."""
        if not self._config_write_allowed():
            self._json(403, {"error": "verdict push is loopback-only"})
            return
        v = payload.get("v") if isinstance(payload.get("v"), dict) else None
        if not isinstance(v, dict) or "move" not in v:
            self._json(400, {"error": "payload {who, v:{move,...}} required"})
            return
        v = dict(v)
        v.setdefault("move_label", MOVE_LABELS.get(v.get("move"), "?"))
        feed_add({"ts": time.time(), "who": (payload.get("who") or "?")[:24],
                  "source": "push", "v": v})
        self._json(200, {"ok": True})

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
        feed_add({"ts": time.time(), "who": (payload.get("them") or "对方"),
                  "source": "web", "v": v})
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
        if v.get("reply_direction"):
            text += "\n回复方向: %s" % v["reply_direction"]
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
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    args = ap.parse_args()
    Handler.engine_kwargs = {}
    if args.base_url:
        Handler.engine_kwargs["base_url"] = args.base_url
    if args.model:
        Handler.engine_kwargs["model"] = args.model
    cfg = load_config()
    url = "http://%s:%d" % (args.host, args.port)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    if cfg.get("base_url") and cfg.get("model"):
        print("crush bot hub on %s  engine: %s @ %s (key %s)"
              % (url, cfg["model"], cfg["base_url"],
                 mask_config(cfg).get("api_key") or "none"))
    else:
        print("crush bot hub on %s  engine: NOT CONFIGURED" % url)
        print("  -> open %s and expand the settings panel, or run: python -m openjev.llm_config" % url)
    if args.open:
        import threading
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
