#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编写时间: 2026-10-02 23:42:53
脚本功能: OpenJev WhatsApp Web 助手 (**实验性**) - Playwright 驱动浏览器中的
          WhatsApp Web, 持久 profile (~/.openjev/wa_profile) 首次扫码后免登录.
          轮询当前打开聊天的消息 DOM (data-testid=msg-container), 用
          copyable-text 元素的 data-pre-plain-text="[时间] 名字: " 还原说话人,
          按 message-out 祖先类识别自己发出的消息 -> 双向上下文; 新消息按
          (方向, 名字, 文本, 出现次数) 去重 diff 后喂给 QqLink 内核
          (OneBot 形状, 与 QQ/TG 同一套), 静默去抖 -> crush_llm 判定 -> --push.
          风险等级同 wxauto4 思路: 纯 UI 自动化 (浏览器层), 不碰协议不注入;
          但 WhatsApp 官方 ToS 不支持第三方自动化, 理论封号风险存在, 自担.
参数: python -m openjev.wa_assistant [--who "备注名"] [--poll-secs 2]
          [--quiet-secs 6] [--context 30]
          [--push http://127.0.0.1:8793/api/verdict] [--selftest]
          --who: 启动时用搜索框打开该会话; 不给则盯当前已打开的聊天.
输入格式: WhatsApp Web 消息 DOM 行 {out: bool, pre: "[..] Name: ", text: str}.
输出格式: stdout 判定 (内核 _log); --push 推 hub 实时流; 无落盘.
依赖: playwright (pip install playwright; playwright install chromium);
          openjev.crush_llm / openjev.qq_assistant.
注意事项: 实验性质 - DOM 结构随 WA 更新可能变化 (选择器已留多级回退);
          首次运行需手机扫码; 群组未特殊处理 (按发言人名进历史);
          相同 (名字, 文本) 消息靠出现次数区分, WA 只保留可见的最近消息.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter

from openjev.crush_llm import MOVE_LABELS
from openjev.llm_config import load as load_config, mask as mask_config
from openjev.qq_assistant import QqLink

PROFILE_DIR = __import__("os").path.join(
    __import__("os").path.expanduser("~"), ".openjev", "wa_profile")

PRE_RE = re.compile(r"^\[[^\]]*\]\s*(.*?):\s*$")

# JS executed inside the page: pull every message container as a plain row.
# Selectors try the stable data-testid hooks first, class fallbacks second.
ROWS_JS = """
() => Array.from(
  document.querySelectorAll('[data-testid="msg-container"], div[class*="message-in"], div[class*="message-out"]')
).map(el => {
  const ct = el.querySelector('[data-pre-plain-text], .copyable-text');
  return {
    out: !!el.closest('[class*="message-out"]') || (el.className || '').includes('message-out'),
    pre: ct ? (ct.getAttribute('data-pre-plain-text') || '') : '',
    text: ((ct ? ct.innerText : el.innerText) || '').trim()
  };
}).filter(r => r.text)
"""

SEARCH_SEL = '[data-testid="chat-list-search"]'
HEADER_SEL = '[data-testid="conversation-info-header"], header [title]'


def extract_messages(rows):
    """Rows from ROWS_JS -> [(name, text, out)]; my own lines get out=True."""
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        name = ""
        m = PRE_RE.match((r.get("pre") or "").strip())
        if m:
            name = m.group(1).strip()
        if r.get("out"):
            out.append((name or "我", text, True))
        else:
            out.append((name or "对方", text, False))
    return out


class WaLink:
    """Playwright loop -> QqLink core (history/debounce/verdict/--push)."""

    def __init__(self, who="", poll_secs=2.0, quiet_secs=6.0, context_lines=30,
                 push_url=None, analyze_fn=None, on_event_log=None, core=None):
        self.who = who
        self.poll_secs = poll_secs
        self.core = core or QqLink("whatsapp://unused", set(),
                                   quiet_secs=quiet_secs,
                                   context_lines=context_lines,
                                   push_url=push_url, analyze_fn=analyze_fn,
                                   on_event_log=on_event_log)
        self._counts = Counter()   # (out, name, text) -> times seen in DOM
        self.chat_name = who or "chat"

    # ---- diff handling (unit-testable) ----
    def diff_new(self, rows):
        """Return the rows not yet fed (occurrence-count based dedupe)."""
        fresh = []
        for name, text, out in extract_messages(rows):
            key = (out, name, text)
            self._counts[key] += 1
            if self._counts[key] == 1:
                fresh.append((name, text, out))
        return fresh

    def feed(self, name, text, out):
        self.core._on_event({
            "post_type": "message_sent" if out else "message",
            "message_type": "private",
            "user_id": self.chat_name,
            "sender": {"nickname": name, "card": ""},
            "message": text,
        })

    # ---- real run ----
    def run(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(
                PROFILE_DIR, headless=False,
                viewport={"width": 1280, "height": 860})
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto("https://web.whatsapp.com")
            self.core._log("waiting for WhatsApp Web (scan the QR on first run)...")
            page.wait_for_selector('[data-testid="chat-list-search"], div[contenteditable="true"]',
                                   timeout=180000)
            if self.who:
                try:
                    box = page.locator(SEARCH_SEL).first
                    box.click(timeout=5000)
                    box.fill(self.who)
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(1200)
                    self.core._log("searched chat: %s" % self.who)
                except Exception as exc:  # noqa: BLE001
                    self.core._log("search failed (%s); watching current chat" % exc)
            try:
                h = page.locator(HEADER_SEL).first
                if h.count():
                    self.chat_name = (h.inner_text() or self.who or "chat").strip().split("\n")[0]
            except Exception:  # noqa: BLE001
                pass
            self.core._log("watching: %s (poll %.1fs)" % (self.chat_name, self.poll_secs))
            while True:
                try:
                    rows = page.evaluate(ROWS_JS)
                except Exception as exc:  # noqa: BLE001
                    self.core._log("dom read failed: %s" % str(exc)[:80])
                    time.sleep(self.poll_secs)
                    continue
                for name, text, out in self.diff_new(rows):
                    self.feed(name, text, out)
                time.sleep(self.poll_secs)


def selftest():
    """DOM-fixture parse + diff + core integration, no browser needed."""
    rows = [
        {"out": False, "pre": "[23:10, 02/10/2026] Alice: ", "text": "are you free this weekend?"},
        {"out": True, "pre": "", "text": "yes! was thinking of that new cafe"},
        {"out": False, "pre": "[23:12, 02/10/2026] Alice: ", "text": "perfect, you pick the time"},
        {"out": False, "pre": "[23:12, 02/10/2026] Alice: ", "text": "perfect, you pick the time"},
    ]
    parsed = extract_messages(rows)
    ok_parse = (parsed[0] == ("Alice", "are you free this weekend?", False)
                and parsed[1] == ("我", "yes! was thinking of that new cafe", True))

    seen = {"transcripts": [], "calls": 0}

    def stub_analyze(transcript, timeout=0):
        seen["calls"] += 1
        seen["transcripts"].append(transcript)
        return {"interest_their": 2.2, "interest_mine": 2.0, "trend": "rising",
                "warmth_signals": 0.6, "move": "chong",
                "reply_direction": "给二选一时间",
                "reason": "对方主动约且让你定时间", "next_advice": "周六下午或晚上都行, 你挑"}

    logs = []
    link = WaLink(who="Alice", quiet_secs=0.8, analyze_fn=stub_analyze,
                  on_event_log=logs.append)
    fresh = link.diff_new(rows)
    ok_dedupe = len(fresh) == 3  # duplicate "perfect..." counted once
    for name, text, out in fresh:
        link.feed(name, text, out)
    deadline = time.time() + 6
    while time.time() < deadline and seen["calls"] < 1:
        time.sleep(0.1)
    tr = seen["transcripts"][0] if seen["transcripts"] else ""
    ok_hist = ("Alice: are you free this weekend?" in tr
               and "我: yes! was thinking of that new cafe" in tr
               and "Alice: perfect, you pick the time" in tr)
    ok_once = seen["calls"] == 1
    print("wa selftest: parse=%s dedupe=%s two-sided-history=%s one-analyze=%s"
          % (ok_parse, ok_dedupe, ok_hist, ok_once))
    print("transcript: %s" % tr.replace("\n", " / "))
    assert ok_parse and ok_dedupe and ok_hist and ok_once, "wa selftest FAILED"
    print("WA-SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser(description="OpenJev WhatsApp Web assistant (experimental)")
    ap.add_argument("--who", default="", help="open this chat via search; default: current chat")
    ap.add_argument("--poll-secs", type=float, default=2.0)
    ap.add_argument("--quiet-secs", type=float, default=6.0)
    ap.add_argument("--context", type=int, default=30)
    ap.add_argument("--push", default="", help="hub url for live-feed push")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    cfg = load_config()
    if not (cfg.get("base_url") and cfg.get("model")):
        print("engine: NOT CONFIGURED -> run: python -m openjev.llm_config")
        sys.exit(1)
    print("engine: %s @ %s (key %s)" % (cfg["model"], cfg["base_url"],
                                        mask_config(cfg).get("api_key") or "none"))
    WaLink(args.who, args.poll_secs, args.quiet_secs, args.context,
           push_url=(args.push or None)).run()


if __name__ == "__main__":
    main()
