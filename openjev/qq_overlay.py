#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编写时间: 2026-10-01 23:43:09
脚本功能: OpenJev QQ 聊天窗内嵌分析悬浮层 - 在指定 QQ 聊天窗口的右下角
          (消息区末尾) 直接叠一块半透明判定卡片, 视觉效果等同于 "对方消息
          后面直接跟出分析". 实现方式: 用 Win32 找到标题匹配的聊天窗口,
          一个 overrideredirect 透明置顶 tk 窗口跟随其位置, 卡片锚定在
          消息区底部右侧. 事件源复用 qq_assistant.QqLink (OneBot 11 /
          NapCat forward WS), 判定复用 crush_llm.analyze, 静默去抖语义同
          qq_assistant. 点击卡片 = 复制下一步建议到剪贴板.
          全程不注入不 hook QQ 进程 - 只是屏幕空间的覆盖窗口, 对 QQ 完全
          不可见; 这也是 "不考虑安全因素也未必需要注入" 的原因: 视觉内嵌
          用覆盖层即可实现, 真正改消息流 DOM 需要走 LiteLoaderQQNT 插件
          (另一条路, 本模块不做).
参数: python -m openjev.qq_overlay [--who 窗口标题关键字]
          [--ws ws://127.0.0.1:3001] [--watch 10086]
          [--quiet-secs 6] [--context 30] [--fade-secs 0]
          [--selftest]   用内置假聊天窗 + mock OneBot server 全链路自测
输入格式: OneBot 11 私聊事件 (经 QqLink); 无 NapCat 时 --selftest 走 mock.
输出格式: 聊天窗内叠放的判定卡片 (冲/稳/缓/停 + 兴趣分 + 理由 + 下一步);
          stdout 状态行; 无落盘.
依赖: tkinter; pywin32 不需要 (用 ctypes 调 user32); websocket-client;
          openjev.crush_llm / openjev.llm_config / openjev.qq_assistant.
注意事项: 卡片不会拦截 QQ 的鼠标操作区域之外的内容, 但卡片本身可点击;
          --fade-secs 0 表示判定常驻直到下一条判定覆盖. 窗口跟随以 400ms
          轮询 GetWindowRect 实现, 拖动/缩放聊天窗时卡片跟着走.

===== [2026-10-01 23:47:25] =====
新增: 卡片 [+ 详情] / 双击 -> 展开完整报告 (470px 宽, 大字号, 复制建议
与收起按钮, 不自动淡出); --push URL 把判定 POST 给 hub (手机页 SSE
实时收到). _follow 改为按卡片实际尺寸定位.

===== [2026-10-02 23:42:53] =====
卡片美化: 彩色头带 (大号判定徽章 + 深色文字 + 分数双行 + 趋势中文化 +
时间戳), 正文/方向/建议行内边距统一; selftest 文本断言改为递归收集.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import queue
import sys
import threading
import time
import tkinter as tk

from openjev.crush_llm import MOVE_LABELS, analyze
from openjev.llm_config import load as load_config, mask as mask_config
from openjev.qq_assistant import QqLink, _mock_server

user32 = ctypes.windll.user32

MOVE_COLORS = {"冲": "#3fb950", "稳": "#58a6ff", "缓": "#d29922", "停": "#f85149"}

CARD_W = 300
CARD_W2 = 470
CARD_H = 168


def find_window_by_title(keyword):
    """Return (hwnd, rect) of the first visible top-level window whose
    title contains keyword, or (None, None)."""
    result = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if keyword in buf.value:
            result.append(hwnd)
            return False  # stop enumeration
        return True

    user32.EnumWindows(cb, 0)
    if not result:
        return None, None
    hwnd = result[0]
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return hwnd, (r.left, r.top, r.right, r.bottom)


class QqOverlayApp:
    """Transparent overlay card pinned inside a chat window's message area."""

    def __init__(self, keyword, ws_url, watch, quiet_secs=6.0, context_lines=30,
                 fade_secs=0, analyze_fn=None, push_url=None):
        self.keyword = keyword
        self.ws_url = ws_url
        self.watch = watch
        self.quiet_secs = quiet_secs
        self.context_lines = context_lines
        self.fade_secs = fade_secs
        self._analyze = analyze_fn or analyze
        self.verdicts = queue.Queue()   # (uin, verdict_dict) from link threads
        self.root = tk.Tk()
        self.root.withdraw()            # only the card is visible
        self.card = None
        self._link = None
        self._hwnd = None
        self._rect = None
        self._current_v = None
        self._expanded = False
        self.push_url = push_url

    # ---- lifecycle ----
    def start(self):
        self._locate()
        if not self._hwnd:
            print("target chat window not found: %r  (open the chat, pop it "
                  "out, then retry)" % self.keyword)
            sys.exit(1)
        print("overlay target: hwnd=%s rect=%s keyword=%r"
              % (self._hwnd, self._rect, self.keyword))
        self._link = QqLink(self.ws_url, set(self.watch),
                            quiet_secs=self.quiet_secs,
                            context_lines=self.context_lines,
                            analyze_fn=self._piped_analyze,
                            push_url=self.push_url)
        threading.Thread(target=self._link.start, daemon=True).start()
        self.root.after(400, self._follow)
        self.root.after(200, self._pump_verdicts)
        self.root.mainloop()

    def _piped_analyze(self, transcript, timeout=180):
        v = self._analyze(transcript, timeout=timeout)
        self.verdicts.put(v)
        return v

    # ---- window tracking ----
    def _locate(self):
        self._hwnd, self._rect = find_window_by_title(self.keyword)

    def _follow(self):
        self._locate()
        if self._rect and self.card is not None and self.card.winfo_exists():
            l, t, r, b = self._rect
            w = self.card.winfo_width()
            h = self.card.winfo_height()
            if w <= 1:
                w = CARD_W2 if self._expanded else CARD_W
            if h <= 1:
                h = 260 if self._expanded else CARD_H
            x = r - w - 24
            y = b - h - 150   # above the input box, end of message area
            self.card.geometry("+%d+%d" % (x, y))
        self.root.after(400, self._follow)

    # ---- verdict card ----
    def _pump_verdicts(self):
        try:
            while True:
                v = self.verdicts.get_nowait()
                self._show(v)
        except queue.Empty:
            pass
        self.root.after(200, self._pump_verdicts)

    def _show(self, v):
        self._current_v = v
        self._build_card(expanded=False)

    def _expand(self):
        self._build_card(expanded=True)

    def _shrink(self):
        self._build_card(expanded=False)

    def _build_card(self, expanded=False):
        """Small card (click=copy) or expanded report (buttons, no fade)."""
        if self.card is not None and self.card.winfo_exists():
            self.card.destroy()
        v = self._current_v
        if v is None:
            return
        import time as _time
        mv = MOVE_LABELS[v["move"]]
        warm = ("%.2f" % v["warmth_signals"]) if v.get("warmth_signals") is not None else "n/a"
        wrap = (CARD_W2 if expanded else CARD_W) - 26
        fnt = 10 if expanded else 9
        stamp = _time.strftime("%H:%M")
        card = tk.Toplevel(self.root)
        card.overrideredirect(True)
        card.attributes("-topmost", True)
        card.attributes("-transparentcolor", "#010104")
        card.configure(bg="#010104")
        box = tk.Frame(card, bg="#14181f", highlightthickness=1,
                       highlightbackground=MOVE_COLORS[mv])
        box.pack(fill="both", expand=True, padx=0, pady=0)
        # invisible spacer forces the card's minimum width to the wrap width
        tk.Frame(box, bg="#14181f", width=wrap + 2, height=0).pack(fill="x")
        # colored header strip: big verdict badge + scores, dark-on-color
        strip = tk.Frame(box, bg=MOVE_COLORS[mv])
        strip.pack(fill="x")
        tk.Label(strip, text=mv, bg=MOVE_COLORS[mv], fg="#101418",
                 font=("Microsoft YaHei", 26 if expanded else 17, "bold")
                 ).pack(side="left", padx=(12, 2), pady=(6, 6))
        tk.Label(strip,
                 text="对方 %.1f/3 · 我 %.1f/3\n%s · 好感 %s"
                 % (v["interest_their"], v["interest_mine"],
                    {"rising": "升温", "flat": "持平", "falling": "降温"}[v["trend"]], warm),
                 bg=MOVE_COLORS[mv], fg="#101418", justify="left",
                 font=("Microsoft YaHei", 9 if expanded else 8)
                 ).pack(side="left", pady=(6, 6))
        tk.Label(strip, text=stamp, bg=MOVE_COLORS[mv], fg="#101418",
                 font=("Microsoft YaHei", 8)).pack(side="right", padx=10, pady=(6, 6))
        body = tk.Label(box, text="理由: %s" % v.get("reason", ""),
                        bg="#14181f", fg="#c9d1d9", wraplength=wrap,
                        justify="left", font=("Microsoft YaHei", fnt))
        body.pack(fill="x", padx=12, pady=(8, 0))
        adv = v.get("next_advice", "")
        if v.get("reply_direction"):
            tk.Label(box, text="方向: %s" % v["reply_direction"], bg="#14181f",
                     fg="#a371f7", wraplength=wrap, justify="left",
                     font=("Microsoft YaHei", fnt)).pack(fill="x", padx=12)
        if adv:
            tk.Label(box, text="下一步: %s" % adv, bg="#14181f", fg="#d29922",
                     wraplength=wrap, justify="left",
                     font=("Microsoft YaHei", fnt)).pack(fill="x", padx=12)
        foot = tk.Frame(box, bg="#14181f")
        foot.pack(fill="x", padx=10, pady=(0, 6))
        if expanded:
            tk.Button(foot, text="复制建议", command=lambda: self._copy(adv),
                      bg="#21262d", fg="#e6e6e6", relief="flat",
                      font=("Microsoft YaHei", 9)).pack(side="left")
            tk.Button(foot, text="收起", command=self._shrink,
                      bg="#21262d", fg="#8b949e", relief="flat",
                      font=("Microsoft YaHei", 9)).pack(side="left", padx=6)
            if adv:
                tk.Label(foot, text="(复制后粘贴到输入框)", bg="#14181f",
                         fg="#59636e", font=("Microsoft YaHei", 8)).pack(side="right")
        else:
            hint = tk.Label(foot, text="点击卡片复制建议", bg="#14181f",
                            fg="#59636e", font=("Microsoft YaHei", 8))
            hint.pack(side="right")
            det = tk.Label(foot, text="[+ 详情]", bg="#14181f",
                           fg="#58a6ff", font=("Microsoft YaHei", 8, "bold"))
            det.pack(side="left")
            det.bind("<Button-1>", lambda e: self._expand())
            card.bind("<Button-1>", lambda e: self._copy(adv))
            for w in (body, box, hint):
                w.bind("<Button-1>", lambda e: self._copy(adv))
            card.bind("<Double-1>", lambda e: self._expand())
        self.card = card
        self._expanded = expanded
        self.root.update_idletasks()
        self._follow()
        if self.fade_secs > 0 and not expanded:
            card.after(self.fade_secs * 1000, lambda: card.destroy())

    def _copy(self, adv):
        if not adv:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(adv)
        print("copied advice to clipboard: %s" % adv[:40])


# ---------------- selftest ----------------

def selftest():
    """Fake chat window + mock OneBot WS server + stub engine, assert card."""
    seen = {"transcripts": []}

    def stub_analyze(transcript, timeout=0):
        seen["transcripts"].append(transcript)
        return {"interest_their": 2.4, "interest_mine": 2.0, "trend": "rising",
                "warmth_signals": 0.66, "move": "chong", "reply_direction": "接梗+抛二选一",
                "reason": "对方秒回并主动抛新话题",
                "next_advice": "趁热定周末时间"}

    app = QqOverlayApp("OPENJEV-TEST-CHAT", "ws://127.0.0.1:3098", set(),
                       quiet_secs=0.8, analyze_fn=stub_analyze)

    # fake target chat window: a plain tk window with the keyword as title
    target = tk.Toplevel(app.root)
    target.title("OPENJEV-TEST-CHAT")
    target.geometry("500x620+120+120")
    tk.Label(target, text="simulated QQ chat window").pack()
    # realize the tk window before enumerating Win32 windows
    for _ in range(10):
        app.root.update()
        time.sleep(0.05)

    def run_mock():
        ev = {"post_type": "message", "message_type": "private", "user_id": 10086,
              "sender": {"nickname": "她", "card": ""},
              "message": "周六那个展一起去?"}
        _mock_server(3098, [ev, dict(ev, message="就等你这句话了")])

    threading.Thread(target=run_mock, daemon=True).start()
    app._locate()
    assert app._hwnd, "fake chat window not found"
    # drive without blocking main(): pump like mainloop would
    app._link = QqLink(app.ws_url, set(), quiet_secs=0.8,
                       analyze_fn=app._piped_analyze)
    threading.Thread(target=app._link.start, daemon=True).start()
    deadline = time.time() + 8
    while time.time() < deadline:
        app.root.update()
        try:
            app._show(app.verdicts.get_nowait())
        except queue.Empty:
            pass
        if app.card is not None:
            break
        time.sleep(0.05)
    ok_card = app.card is not None
    geo = app.card.geometry() if ok_card else ""
    ok_pos = False
    if ok_card:
        l, t, r, b = app._rect
        x, y = map(int, geo.split("+")[1:])
        w = app.card.winfo_width() or app.card.winfo_reqwidth()
        h = app.card.winfo_height() or app.card.winfo_reqheight()
        ok_pos = l < x and x + w < r and t < y and y + h < b
        def all_text(w, acc):
            for c in w.winfo_children():
                t = c.cget("text") if isinstance(c, (tk.Label, tk.Button)) else ""
                if t:
                    acc.append(str(t))
                all_text(c, acc)
            return acc
        card_text = " | ".join(all_text(app.card, []))
        ok_text = "冲" in card_text and "2.4" in card_text and "方向" in card_text
    # expand interaction: small card -> full report
    ok_expand = False
    if ok_card:
        app._expand()
        app.root.update()
        app.root.update_idletasks()
        app.root.update()
        def find_btn(w, text):
            for c in w.winfo_children():
                t = c.cget("text") if isinstance(c, (tk.Button, tk.Label)) else ""
                if text in str(t):
                    return True
                if find_btn(c, text):
                    return True
            return False
        app.root.update_idletasks()
        ew = app.card.winfo_reqwidth()
        ok_expand = (app.card.winfo_exists() and find_btn(app.card, "收起")
                     and ew > CARD_W + 50)
        app._shrink()
        app.root.update()
    print("selftest: card-shown=%s inside-chat-rect=%s verdict-text=%s expand=%s geometry=%s"
          % (ok_card, ok_pos, ok_text, ok_expand, geo))
    print("history: %s" % seen["transcripts"][0].replace("\n", " / ") if seen["transcripts"] else "none")
    assert ok_card and ok_pos and ok_text and ok_expand, "selftest FAILED"
    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser(description="OpenJev QQ in-chat overlay")
    ap.add_argument("--who", default="", help="chat window title keyword")
    ap.add_argument("--ws", default="ws://127.0.0.1:3001")
    ap.add_argument("--watch", default="", help="comma QQ uins (default all)")
    ap.add_argument("--quiet-secs", type=float, default=6.0)
    ap.add_argument("--context", type=int, default=30)
    ap.add_argument("--fade-secs", type=int, default=0, help="0 = keep until next")
    ap.add_argument("--push", default="",
                    help="POST each verdict to this hub url (live feed on phones)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.who:
        ap.error("--who is required (chat window title keyword)")
    cfg = load_config()
    if not (cfg.get("base_url") and cfg.get("model")):
        print("engine: NOT CONFIGURED -> run: python -m openjev.llm_config")
        sys.exit(1)
    print("engine: %s @ %s (key %s)" % (cfg["model"], cfg["base_url"],
                                        mask_config(cfg).get("api_key") or "none"))
    watch = [x.strip() for x in args.watch.split(",") if x.strip()]
    QqOverlayApp(args.who, args.ws, watch, args.quiet_secs, args.context,
                 args.fade_secs, push_url=(args.push or None)).start()


if __name__ == "__main__":
    main()
