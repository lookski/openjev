#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编写时间: 2026-10-01 23:54:10
脚本功能: OpenJev Telegram 助手 - 用 Telethon 以**用户自己的账号**登录
          (MTProto 官方 API, 合法的第三方客户端, 无注入无 hook, Telegram
          官方允许, 只有垃圾轰炸才会封号). 这是所有聊天平台里接入最干净
          的一条路: 双向消息实时收 (对方来 + 自己发), 还能直接**回填会话
          历史** (--backfill 条), 上下文质量最高. 事件内核复用
          qq_assistant.QqLink (把 Telegram 事件映射成 OneBot 形状),
          静默去抖 -> crush_llm.analyze -> 冲/稳/缓/停 + 下一步;
          --push 把判定推给 hub (手机页 SSE 实时流).
参数: python -m openjev.tg_assistant --login            # 一次性登录 (建 session)
          python -m openjev.tg_assistant [--watch user_id|@username,...]
              [--backfill 30]   启动时回填最近 N 条历史 (0 = 不回填)
              [--quiet-secs 6]  [--context 30]
              [--push http://127.0.0.1:8793/api/verdict]
              [--selftest]      不登录, 用 stub 事件全链路自测
          api_id/api_hash 来源: my.telegram.org -> API development tools
          ($TG_API_ID / $TG_API_H 环境变量, 或登录时交互输入; 只用于登录,
          不入库; session 存 ~/.openjev/tg.session)
输入格式: Telegram 私聊消息事件 (Telethon events.NewMessage, 含 outgoing).
输出格式: stdout 判定行; --push 时同时 POST {who, v} 给 hub; 无落盘.
依赖: telethon (pip install telethon); openjev.crush_llm /
          openjev.llm_config / openjev.qq_assistant.
注意事项: 官方 API 路线, 风险等级 = 正常第三方客户端 (Telegram Desktop
          同类). 频率像人即可; 判定调用与消息读取解耦 (去抖节流).
          只处理私聊 (message_type private), 群组忽略.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from openjev.crush_llm import MOVE_LABELS
from openjev.llm_config import load as load_config, mask as mask_config
from openjev.qq_assistant import QqLink

SESSION_PATH = os.path.join(os.path.expanduser("~"), ".openjev", "tg.session")


def ev_private(uin, name, text, out=False):
    """Map a Telegram private message to the OneBot-shaped dict QqLink eats."""
    return {
        "post_type": "message_sent" if out else "message",
        "message_type": "private",
        "user_id": uin,
        "sender": {"nickname": name, "card": ""},
        "message": text,
    }


class TgLink:
    """Telethon -> QqLink adapter: login, backfill, live events, verdicts."""

    def __init__(self, watch, quiet_secs=6.0, context_lines=30, backfill=30,
                 push_url=None, analyze_fn=None, on_event_log=None,
                 core=None):
        # QqLink provides: history, watch filter, quiet debounce, analyze,
        # verdict formatting, --push. Its ws start() is never called.
        self.core = core or QqLink(
            "telegram://unused", set(watch), quiet_secs=quiet_secs,
            context_lines=context_lines, push_url=push_url,
            analyze_fn=analyze_fn, on_event_log=on_event_log)
        self.backfill = backfill

    def feed(self, ev):
        self.core._on_event(ev)

    # ---- real run (needs telethon + logged-in session) ----
    async def run(self, api_id, api_hash):
        from telethon import TelegramClient, events

        os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
        client = TelegramClient(SESSION_PATH, api_id, api_hash)
        await client.start()  # uses saved session; prompts only if absent
        me = await client.get_me()
        self.core._log("logged in as %s (id %s)" % (me.username or me.first_name, me.id))

        async def name_of(peer):
            try:
                ent = await client.get_entity(peer)
                return getattr(ent, "username", None) or getattr(ent, "first_name", None) or str(peer)
            except Exception:  # noqa: BLE001
                return str(peer)

        async def backfill_chats():
            if not self.backfill:
                return
            dialogs = await client.get_dialogs(limit=50)
            done = 0
            for d in dialogs:
                if not d.is_user or d.message is None:
                    continue
                if self.core.watch and str(d.id) not in self.core.watch:
                    continue
                msgs = await client.get_messages(d.entity, limit=self.backfill)
                name = await name_of(d.entity)
                for m in reversed(msgs):  # oldest first
                    if not m.message:
                        continue
                    self.feed(ev_private(m.chat_id or d.id, name, m.message, out=m.out))
                done += 1
                if done >= 10:
                    break
            if done:
                self.core._log("backfilled %d chat(s), %d lines each" % (done, self.backfill))

        @client.on(events.NewMessage(incoming=True))
        async def _incoming(event):
            if not event.is_private:
                return
            who = await name_of(event.chat_id)
            self.feed(ev_private(event.chat_id, who, event.message.message or ""))

        @client.on(events.NewMessage(outgoing=True))
        async def _outgoing(event):
            if not event.is_private:
                return
            who = await name_of(event.chat_id)
            self.feed(ev_private(event.chat_id, who, event.message.message or "", out=True))

        await backfill_chats()
        self.core._log("watching: %s (private chats only)" %
                       (",".join(sorted(self.core.watch)) or "ALL"))
        await client.run_until_disconnected()


def selftest():
    """Feed stub Telegram events through the QqLink core; assert behavior."""
    seen = {"transcripts": [], "calls": 0}

    def stub_analyze(transcript, timeout=0):
        seen["calls"] += 1
        seen["transcripts"].append(transcript)
        return {"interest_their": 2.5, "interest_mine": 2.0, "trend": "rising",
                "warmth_signals": 0.7, "move": "chong",
                "reason": "对方秒回并反问", "next_advice": "趁热定周末"}

    logs = []
    link = TgLink(watch={"777"}, quiet_secs=0.8, backfill=3,
                  analyze_fn=stub_analyze, on_event_log=logs.append)
    # backfill: oldest first, two-sided (out=True = my own messages)
    link.feed(ev_private(777, "alice", "上次说的地方我想去"))
    link.feed(ev_private(777, "alice", "好啊你去买票", out=True))
    link.feed(ev_private(777, "alice", "周六那个展一起去?"))
    # live burst: one more line, debounce should coalesce into ONE analyze
    link.feed(ev_private(777, "alice", "就等你这句话了"))
    import time as _t
    deadline = _t.time() + 6
    while _t.time() < deadline and seen["calls"] < 1:
        _t.sleep(0.1)
    tr = seen["transcripts"][0] if seen["transcripts"] else ""
    ok_hist = ("alice: 上次说的地方我想去" in tr
               and "我: 好啊你去买票" in tr
               and "alice: 就等你这句话了" in tr)
    ok_once = seen["calls"] == 1
    # watch filter: unwatched uin must not enter history
    n0 = len(link.core.sessions.get("777", []))
    link.feed(ev_private(999, "stranger", "hi"))
    _t.sleep(0.3)
    ok_watch = (999 not in link.core.sessions) and n0 >= 4
    print("tg selftest: calls=%d two-sided-history=%s one-analyze=%s watch-filter=%s"
          % (seen["calls"], ok_hist, ok_once, ok_watch))
    print("transcript: %s" % tr.replace("\n", " / "))
    assert ok_hist and ok_once and ok_watch, "tg selftest FAILED"
    print("TG-SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser(description="OpenJev Telegram assistant (Telethon)")
    ap.add_argument("--login", action="store_true",
                    help="one-time interactive login (creates ~/.openjev/tg.session)")
    ap.add_argument("--watch", default="", help="comma uids/@usernames (default all private)")
    ap.add_argument("--backfill", type=int, default=30,
                    help="replay N history lines per chat at startup (0 = off)")
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
    watch = [x.strip() for x in args.watch.split(",") if x.strip()]
    link = TgLink(watch, args.quiet_secs, args.context, args.backfill,
                  push_url=(args.push or None))
    api_id = os.environ.get("TG_API_ID")
    api_hash = os.environ.get("TG_API_H")
    if args.login:
        if not api_id:
            api_id = input("api_id (from my.telegram.org): ").strip()
        if not api_hash:
            api_hash = input("api_hash: ").strip()
    if not api_id or not api_hash:
        print("need $TG_API_ID / $TG_API_H (my.telegram.org -> API development tools)")
        sys.exit(1)
    asyncio.run(link.run(int(api_id), api_hash))


if __name__ == "__main__":
    main()
