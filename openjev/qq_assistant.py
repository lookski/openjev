#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编写时间: 2026-10-01 23:34:46
脚本功能: OpenJev QQ 助手 - OneBot 11 协议客户端 (NapCat / LLOneBot 的
          forward WebSocket). 实时读取 QQ 私聊消息, 双向积累上下文 (对方的
          消息 post_type=message, 你自己发出的 post_type=message_sent),
          每个会话静默 N 秒后自动调用 openjev 判定引擎 (crush_llm.analyze)
          输出 冲/稳/缓/停 + 理由 + 下一步建议. 无 OCR, 无 UI 自动化, 无
          截图 - 消息本身就是结构化 JSON, 这是 QQ 比 WeChat 好做实时
          的根本原因. --send-reply 可把建议作为私聊消息发回去 (默认关,
          发送权原则同 wechat_assistant: 默认只提示, 不代发).
参数: python -m openjev.qq_assistant [--ws ws://127.0.0.1:3001]
          [--watch 10086,10010]   只监听这些 QQ 号 (默认全部私聊)
          [--quiet-secs 6]        静默多少秒后触发判定 (聊天连发等待)
          [--context 30]          送入判定的最近消息条数
          [--send-reply]          判定后把建议发回该会话 (默认关)
          [--selftest]            内置 mock OneBot server 全链路自测
输入格式: OneBot 11 事件 JSON (post_type=message / message_sent,
          message_type=private); message 为字符串 (含 CQ 码) 或分段数组.
输出格式: stdout 判定行 + 日志; 无落盘.
依赖: websocket-client; openjev.crush_llm / openjev.llm_config.
注意事项: NapCat 属于第三方注入式实现, 存在理论封号风险 (历史上 QQ 对
          此宽容得多); 担心就用小号. 官方 q.qq.com 机器人读不了好友私聊,
          这条路做不了 crush 场景. 本模块不主动给对方发任何消息, 除非
          显式加 --send-reply.

===== [2026-10-01 23:47:25] =====
新增 --push URL: 每次判定后把 {who, v} POST 到 hub 的 /api/verdict
(crush_bot), 手机页经 SSE 实时收到判定. 默认关. 推送失败不影响
本地判定.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import socket
import sys
import threading
import time
import urllib.request

from openjev.crush_llm import MOVE_LABELS, analyze
from openjev.llm_config import load as load_config, mask as mask_config

DEFAULT_WS = "ws://127.0.0.1:3001"
CQ_RE = re.compile(r"\[CQ:[^\]]+\]")
MY_NAME = "我"

try:
    import websocket  # websocket-client
except ImportError:  # pragma: no cover
    websocket = None


def _seg_text(ev):
    """Extract plain text from an OneBot message (string or segment array)."""
    msg = ev.get("message")
    if isinstance(msg, str):
        return CQ_RE.sub("", msg).strip()
    if isinstance(msg, list):
        parts = []
        for seg in msg:
            st = seg.get("type")
            data = seg.get("data") or {}
            if st == "text":
                parts.append(data.get("text", ""))
            elif st == "image":
                parts.append("[图片]")
            elif st in ("face", "mface"):
                parts.append("[表情]")
            elif st == "record":
                parts.append("[语音]")
            elif st == "reply":
                parts.append("[回复]")
        # marker segments read better with a space boundary
        buf = ""
        for p in parts:
            if buf and not buf.endswith(" ") and p.startswith("["):
                buf += " "
            buf += p
        return buf.strip()
    return ""


class QqLink:
    """OneBot 11 forward-WS client with per-session history + quiet debounce."""

    def __init__(self, ws_url, watch, quiet_secs=6.0, context_lines=30,
                 send_reply=False, analyze_fn=None, on_event_log=None,
                 push_url=None):
        self.ws_url = ws_url
        self.watch = set(x for x in watch if x)
        self.quiet_secs = quiet_secs
        self.context_lines = context_lines
        self.send_reply = send_reply
        self._analyze = analyze_fn or analyze
        self._log = on_event_log or (lambda s: print(s, flush=True))
        self.sessions = {}   # uin -> list[str] "名字: 消息"
        self.names = {}      # uin -> display name
        self._timers = {}    # uin -> threading.Timer
        self._sock = None    # underlying ws for --send-reply
        self.push_url = push_url  # e.g. http://127.0.0.1:8792/api/verdict

    # ---- connection ----
    def start(self):
        if websocket is None:
            raise RuntimeError("websocket-client not installed: pip install websocket-client")
        while True:
            try:
                ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=lambda w, m: self._on_event(json.loads(m)),
                    on_error=lambda w, e: self._log("ws error: %s" % e),
                    on_close=lambda w, c, r: self._log("ws closed (will retry in 5s)"),
                )
                ws.run_forever(ping_interval=15, ping_timeout=10)
            except Exception as exc:  # noqa: BLE001
                self._log("ws connect failed: %s (retry in 5s)" % exc)
            time.sleep(5)

    def _on_open(self, ws):
        self._sock = ws
        self._log("connected to %s, watching %s"
                  % (self.ws_url, ",".join(sorted(self.watch)) or "all private chats"))

    def send_private(self, uin, text):
        """Send a private message (only used with --send-reply)."""
        if not self._sock:
            return
        self._sock.send(json.dumps({
            "action": "send_private_msg",
            "params": {"user_id": int(uin), "message": text},
        }))

    # ---- event handling ----
    def _on_event(self, ev):
        pt = ev.get("post_type")
        if pt not in ("message", "message_sent"):
            return
        if ev.get("message_type") != "private":
            return
        uin = str(ev.get("user_id") or "")
        if not uin:
            return
        text = _seg_text(ev)
        if not text:
            return
        sender = ev.get("sender") or {}
        if pt == "message_sent":
            # self outgoing: user_id = recipient, sender = me
            who = self.names.get(uin) or uin
            self._add_line(uin, "%s: %s" % (MY_NAME, text))
            self.names.setdefault(uin, uin)
        else:
            if self.watch and uin not in self.watch:
                return
            who = sender.get("card") or sender.get("nickname") or uin
            self.names[uin] = who
            self._add_line(uin, "%s: %s" % (who, text))
        self._log("  <%s> %s" % (MY_NAME if pt == "message_sent" else self.names.get(uin, uin), text[:30]))

    def _add_line(self, uin, line):
        buf = self.sessions.setdefault(uin, [])
        buf.append(line)
        old = self._timers.pop(uin, None)
        if old:
            old.cancel()
        t = threading.Timer(self.quiet_secs, self._fire, args=(uin,))
        self._timers[uin] = t
        t.daemon = True
        t.start()

    def _fire(self, uin):
        self._timers.pop(uin, None)
        lines = self.sessions.get(uin, [])[-self.context_lines:]
        if len(lines) < 2:
            return
        transcript = "\n".join(lines)
        threading.Thread(target=self._judge, args=(uin, transcript), daemon=True).start()

    def _judge(self, uin, transcript):
        who = self.names.get(uin, uin)
        try:
            v = self._analyze(transcript, timeout=180)
        except Exception as exc:  # noqa: BLE001
            try:
                v = self._analyze(transcript, timeout=180)
            except Exception as exc2:  # noqa: BLE001
                self._log("[QQ %s] analyze failed twice: %s / %s" % (uin, exc, exc2))
                return
        warm = ("%.2f" % v["warmth_signals"]) if v.get("warmth_signals") is not None else "n/a"
        self._log("======== [QQ %s %s] %s  对方兴趣 %.1f/3 | 我 %.1f/3 | %s | 好感 %s"
                  % (uin, who, MOVE_LABELS[v["move"]],
                     v["interest_their"], v["interest_mine"], v["trend"], warm))
        self._log("  理由: %s" % v.get("reason", ""))
        if v.get("next_advice"):
            self._log("  下一步: %s" % v["next_advice"])
            if self.send_reply:
                self.send_private(uin, "[OpenJev 建议, 可直接改] " + v["next_advice"])
                self._log("  (已发送建议到会话, --send-reply 模式)")
        if self.push_url:
            self._push(uin, who, v)


    def _push(self, uin, who, v):
        """Best-effort POST of the verdict to the hub (live feed on phones)."""
        try:
            data = json.dumps({"who": who or uin, "v": v},
                              ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(self.push_url, data=data,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as exc:  # noqa: BLE001 - never break the verdict loop
            self._log("push to %s failed: %s" % (self.push_url, exc))


# ---------------- mock OneBot server (selftest only) ----------------

WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _accept_key(key):
    return base64.b64encode(
        hashlib.sha1((key + WS_MAGIC).encode()).digest()).decode()


def _send_frame(conn, payload):
    n = len(payload)
    if n < 126:
        conn.sendall(bytes([0x81, n]) + payload)
    else:
        conn.sendall(bytes([0x81, 126]) + n.to_bytes(2, "big") + payload)


def _mock_server(port, events):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    req = conn.recv(4096).decode("utf-8", "replace")
    m = re.search(r"Sec-WebSocket-Key: (.+?)\r\n", req)
    conn.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                  "Connection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n"
                  % _accept_key(m.group(1).strip())).encode())
    for ev in events:
        _send_frame(conn, json.dumps(ev, ensure_ascii=False).encode("utf-8"))
        time.sleep(0.2)
    time.sleep(2.5)  # let the quiet timer fire
    try:
        conn.close()
    except OSError:
        pass
    srv.close()


def selftest():
    """In-process e2e: mock OneBot WS server feeds events, stub engine, assert."""
    seen = {"transcripts": [], "calls": 0}

    def stub_analyze(transcript, timeout=0):
        seen["calls"] += 1
        seen["transcripts"].append(transcript)
        return {"interest_their": 2.4, "interest_mine": 2.0, "trend": "rising",
                "warmth_signals": 0.66, "move": "chong",
                "reason": "对方连续秒回且主动抛新话题",
                "next_advice": "趁热定周末时间"}

    port = 3099
    ev_in = {"post_type": "message", "message_type": "private", "user_id": 10086,
             "sender": {"nickname": "她", "card": ""}, "message": "哈哈你也好会讲"}
    ev_in2 = {"post_type": "message", "message_type": "private", "user_id": 10086,
              "sender": {"nickname": "她", "card": ""},
              "message": [{"type": "text", "data": {"text": "周六那个展一起去?"}},
                          {"type": "image", "data": {"url": "x"}}]}
    ev_out = {"post_type": "message_sent", "message_type": "private", "user_id": 10086,
              "sender": {"nickname": "我"}, "message": "好啊我去买票"}
    threading.Thread(target=_mock_server, args=(port, [ev_in, ev_in2, ev_out]),
                     daemon=True).start()
    logs = []
    link = QqLink("ws://127.0.0.1:%d" % port, watch=set(), quiet_secs=0.8,
                  context_lines=30, analyze_fn=stub_analyze,
                  on_event_log=logs.append)
    t = threading.Thread(target=link.start, daemon=True)
    t.daemon = True
    t.start()
    deadline = time.time() + 8
    while time.time() < deadline and seen["calls"] < 1:
        time.sleep(0.1)
    tr = seen["transcripts"][0] if seen["transcripts"] else ""
    ok_hist = ("她: 哈哈你也好会讲" in tr and "她: 周六那个展一起去? [图片]" in tr
               and "我: 好啊我去买票" in tr)
    ok_debounce = seen["calls"] == 1  # 3 messages in one burst -> ONE analyze
    print("selftest: calls=%d both-sides-history=%s one-analyze-per-burst=%s"
          % (seen["calls"], ok_hist, ok_debounce))
    print("transcript:\n%s" % tr)
    assert ok_hist and ok_debounce and seen["calls"] == 1, "selftest FAILED"
    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser(description="OpenJev QQ assistant (OneBot 11)")
    ap.add_argument("--ws", default=DEFAULT_WS, help="NapCat forward WS url")
    ap.add_argument("--watch", default="", help="comma-separated QQ uins (default all)")
    ap.add_argument("--quiet-secs", type=float, default=6.0)
    ap.add_argument("--context", type=int, default=30, help="recent lines per analyze")
    ap.add_argument("--send-reply", action="store_true",
                    help="send the suggestion back into the chat (default off)")
    ap.add_argument("--push", default="",
                    help="POST each verdict to this hub url (e.g. http://127.0.0.1:8792/api/verdict)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    cfg = load_config()
    if cfg.get("base_url") and cfg.get("model"):
        print("engine: %s @ %s (key %s)" % (cfg["model"], cfg["base_url"],
                                            mask_config(cfg).get("api_key") or "none"))
    else:
        print("engine: NOT CONFIGURED -> run: python -m openjev.llm_config")
        sys.exit(1)
    watch = [x.strip() for x in args.watch.split(",") if x.strip()]
    print("QQ assistant on %s  watch=%s quiet=%ss context=%s send-reply=%s"
          % (args.ws, watch or "ALL", args.quiet_secs, args.context, args.send_reply))
    QqLink(args.ws, set(watch), args.quiet_secs, args.context,
           args.send_reply, push_url=(args.push or None)).start()


if __name__ == "__main__":
    main()
