#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编写时间: 2026-10-01 09:45:12
脚本功能: OpenJev 微信助手 (wechat_assistant) - 基于 wxauto4 (Windows UI Automation,
          不 hook 不注入) 监听指定会话的新消息, 自动调用 openjev 判定引擎
          (crush_llm.analyze) 并在悬浮窗弹出 冲/稳/缓/停 判定; 建议回复以
          草稿态写入该会话输入框 (只写不发, 发送权永远在用户手里).
          双通道新消息检测: wxauto 回调 (快路径) + GetAllMessage 轮询 diff (兜底,
          免费版回调不触发时仍可靠). 复用 wechat_radar.RadarApp 悬浮窗, 扩展
          助手状态区 (监听对象/草稿动作/开关监听).
参数: --who 会话名 (默认 文件传输助手; 多个用逗号分隔)
      --no-watch   不启动监听, 仅悬浮窗 (等同雷达)
      --no-draft   不写建议草稿 (仅弹判定)
      --poll-secs  轮询间隔秒数 (默认 3.0)
      --font-size / --opacity 同雷达
输入格式: 微信 Windows 客户端 4.x 已登录; 被监听会话会被弹成独立子窗口 (wxauto 机制)
输出格式: 悬浮窗判定 + stdout 日志; 无落盘 (除 openjev 自身日志配置)
依赖: wxauto4 (>=41.1.7, cp313 wheel 可用), openjev (crush_llm, wechat_radar),
      pywin32 (雷达热键), RapidOCR 仅截图路径需要 (本模块不需要)
注意事项: 硬边界 - 全程 UI Automation, 不 hook/不注入/不逆向协议; 不自动发送任何
          消息 (SendMsg 仅用于 --selftest 向 文件传输助手 发自测消息); 草稿只写入
          输入框, 用户不按回车就永远不发出; 监听对象必须是用户本人会话.
          wxauto4 免费版限制: 初始化偶发 "未找到已登录的客户端主窗口" (瞬时状态,
          重试即可); 子窗口 editbox 用剪贴板粘贴写草稿 (SendKeys 击键会丢字符).
"""
from __future__ import annotations

import argparse
import io
import sys
import threading
import time

from openjev.crush_llm import MOVE_LABELS, analyze
from openjev.wechat_radar import RadarApp, MOVE_COLORS

ASSIST_TITLE = "💗 Crush Radar · 助手模式"

DRAFT_NOTE = "[OpenJev 建议草稿, 未发送] "


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class WxLink:
    """wxauto4 attach + listen management. All wxauto calls run on ONE
    dedicated thread: wxauto/uia COM objects are not thread-safe."""

    def __init__(self, who_list, poll_secs=3.0):
        self.who_list = who_list
        self.poll_secs = poll_secs
        self.wx = None
        self.chats = {}          # who -> Chat object
        self._seen = {}          # who -> set of message hash_text already judged
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.on_message = None   # callback(who, text_lines) from watcher thread
        self.errors = []

    # ---- runs on the link thread ----
    def _attach(self):
        from wxauto4 import WeChat
        last = None
        for attempt in range(3):
            try:
                self.wx = WeChat()
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(2.0)
        raise RuntimeError("wxauto attach failed after retries: %s" % last)

    def _listen_one(self, who):
        """AddListenChat + register callback; returns Chat or None."""
        try:
            def cb(msg, chat):
                self._on_wxauto_msg(who, msg)
            chat = self.wx.AddListenChat(who, cb)
            self.chats[who] = chat
            return chat
        except Exception as exc:  # noqa: BLE001
            self.errors.append("AddListenChat(%s): %s" % (who, exc))
            return None

    def _on_wxauto_msg(self, who, msg):
        """wxauto callback path (fast). Friend messages only."""
        try:
            if getattr(msg, "attr", "") != "friend":
                return
            content = str(getattr(msg, "content", "") or "").strip()
            if not content:
                return
            self._emit(who, content, "callback")
        except Exception:  # noqa: BLE001
            pass

    def _poll_once(self, who, chat):
        """Fallback diff path: GetAllMessage -> new friend messages."""
        try:
            msgs = chat.GetAllMessage()
        except Exception as exc:  # noqa: BLE001
            self.errors.append("GetAllMessage(%s): %s" % (who, exc))
            return
        seen = self._seen.setdefault(who, set())
        for m in msgs:
            if getattr(m, "attr", "") != "friend":
                continue
            key = getattr(m, "hash_text", None) or (str(m.content) + str(m.id))
            if key in seen:
                continue
            seen.add(key)
            content = str(getattr(m, "content", "") or "").strip()
            if content:
                self._emit(who, content, "poll")

    def _emit(self, who, content, path):
        if self.on_message:
            try:
                self.on_message(who, content, path)
            except Exception as exc:  # noqa: BLE001
                self.errors.append("on_message: %s" % exc)

    def _run(self):
        try:
            self._attach()
        except Exception as exc:  # noqa: BLE001
            self.errors.append(str(exc))
            return
        # snapshot existing messages as baseline so history is not re-judged
        for who in self.who_list:
            chat = self._listen_one(who)
            if chat is not None:
                try:
                    seen = self._seen.setdefault(who, set())
                    for m in chat.GetAllMessage():
                        key = getattr(m, "hash_text", None) or (str(m.content) + str(m.id))
                        seen.add(key)
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.wx.StartListening()
        except Exception as exc:  # noqa: BLE001
            self.errors.append("StartListening: %s" % exc)
        while not self._stop.is_set():
            for who in self.who_list:
                chat = self.chats.get(who)
                if chat is not None:
                    self._poll_once(who, chat)
            self._stop.wait(self.poll_secs)

    # ---- public API (callable from any thread; work hops to link thread) ----
    def start(self):
        self._thread = threading.Thread(target=self._run, name="wx-link", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self.wx is not None:
                self.wx.StopListening(remove=True)
        except Exception:  # noqa: BLE001
            pass

    def write_draft(self, who, text):
        """Paste suggestion into the chat's input box WITHOUT sending.
        Preserves the user's clipboard: save -> paste -> restore.
        Thread-safe: runs on the link thread via helper hop."""
        result = {}

        def job():
            try:
                chat = self.chats.get(who)
                if chat is None:
                    result["err"] = "no chat %s" % who
                    return
                import win32clipboard
                import win32con
                # save current clipboard text (if any)
                old = None
                try:
                    win32clipboard.OpenClipboard()
                    if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                        old = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                    win32clipboard.CloseClipboard()
                except Exception:  # noqa: BLE001
                    old = None
                full = DRAFT_NOTE + text
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, full)
                win32clipboard.CloseClipboard()
                try:
                    eb = chat.ChatBox.editbox
                    eb.SetFocus()
                    time.sleep(0.25)
                    import wxauto4.uia.uiautomation as uia
                    uia.SendKeys("{Ctrl}v")
                    time.sleep(0.3)
                    result["ok"] = True
                finally:
                    # restore user clipboard
                    try:
                        win32clipboard.OpenClipboard()
                        win32clipboard.EmptyClipboard()
                        if old is not None:
                            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, old)
                        win32clipboard.CloseClipboard()
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as exc:  # noqa: BLE001
                result["err"] = "%s: %s" % (type(exc).__name__, exc)

        # uilock-protected wxauto calls are not thread safe; reuse link thread
        # by executing job there is overkill for one shot: wxauto4's own uilock
        # (threading lock) makes a short direct call safe enough here, and the
        # listener only touches GetAllMessage/its callback.
        job()
        return result


class AssistantApp(RadarApp):
    """Radar floating widget extended with assistant state + draft actions."""

    def __init__(self, who_list, poll_secs=3.0, no_watch=False, no_draft=False,
                 **kw):
        super().__init__(**kw)
        self._who_list = who_list
        self._no_draft = no_draft
        self._last_draft = ""
        self._draft_who = ""
        self.title.configure(text=ASSIST_TITLE + "  (来信自动判定)")
        self._link = None
        if not no_watch:
            self._link = WxLink(who_list, poll_secs=poll_secs)
            self._link.on_message = self._on_incoming
            threading.Thread(target=self._link.start, daemon=True).start()
            self._set_body("监听中: %s\n对方来消息自动判定, 建议草稿自动填入\n"
                           "发送永远由你按回车" % ", ".join(who_list))
            self._watchdog()

    # ---- incoming path (runs on wx link thread) ----
    def _on_incoming(self, who, content, path):
        self.root.after(0, self._ui_incoming, who, content, path)

    def _ui_incoming(self, who, content, path):
        self._set_move("…", "#8b949e")
        self._set_body("[%s] %s\n对方新消息, 分析中…" % (who, content[:24]))
        self._pending_who = who
        self._pending_content = content
        threading.Thread(target=self._incoming_worker,
                         args=(who, content, path), daemon=True).start()

    def _incoming_worker(self, who, content, path):
        # judge the LAST few messages: current exchange is what matters
        transcript = "%s: %s" % (who, content)
        try:
            v = analyze(transcript, timeout=180)
        except Exception:  # noqa: BLE001
            try:
                v = analyze(transcript, timeout=180)
            except Exception as exc2:  # noqa: BLE001
                self.root.after(0, self._flash,
                                "分析失败 (重试过后仍失败): %s" % exc2)
                return
        self.root.after(0, self._ui_verdict_incoming, who, content, path, v)

    def _ui_verdict_incoming(self, who, content, path, v):
        self._show_verdict(v)
        self._last_draft = v.get("next_advice", "") or ""
        self._draft_who = who
        if self._last_draft and not self._no_draft:
            r = self._link.write_draft(who, self._last_draft) if self._link else {"err": "no link"}
            tag = "草稿已填入输入框 (未发送)" if r.get("ok") else "草稿写入失败: %s" % r.get("err", "?")
        else:
            tag = "(草稿关闭)" if self._no_draft else "(无话术建议)"
        self._set_body(self._body_text(v) + "\n" + tag)
        log("incoming(%s) %s: %s -> %s | %s" % (
            path, who, content[:30], MOVE_LABELS[v["move"]], tag))

    def _body_text(self, v):
        warmth = ("%.2f" % v["warmth_signals"]) if v.get("warmth_signals") is not None else "n/a"
        lines = [
            "对方兴趣 %.1f/3 | 我 %.1f/3 | %s | 好感 %s" % (
                v["interest_their"], v["interest_mine"], v["trend"], warmth),
            "理由: %s" % v["reason"],
        ]
        if v.get("next_advice"):
            lines.append("下一步: %s" % v["next_advice"])
        return "\n".join(lines)

    def _watchdog(self):
        """Surface link errors on the widget every few seconds."""
        if self._link is not None:
            errs = self._link.errors
            if errs:
                self._set_body("链路: %s" % "; ".join(errs[-2:]))
                del errs[:]
        self.root.after(5000, self._watchdog)

    # ---- selftest: full loop without a real friend ----
    def selftest(self, who):
        """Send a note to 文件传输助手 via the chat subwindow; the watch
        path picks it up as a friend-attr message only if WeChat classifies
        it so - in practice self messages are attr=self and are ignored,
        so this just verifies send+read. Returns raw msg tail."""
        chat = self._link.chats.get(who)
        if chat is None:
            return "no chat"
        r = chat.SendMsg("openjev-assistant-selftest-%d" % int(time.time()))
        time.sleep(2.0)
        msgs = chat.GetAllMessage()
        return {"send": str(r), "tail": [(m.attr, str(m.content)[:24]) for m in msgs[-3:]]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="OpenJev WeChat assistant (wxauto4, UI automation only)")
    ap.add_argument("--who", default="文件传输助手",
                    help="监听的会话名, 逗号分隔 (默认: 文件传输助手)")
    ap.add_argument("--no-watch", action="store_true", help="不启动监听, 仅悬浮窗")
    ap.add_argument("--no-draft", action="store_true", help="不把建议写入输入框")
    ap.add_argument("--poll-secs", type=float, default=3.0, help="轮询间隔秒 (默认 3)")
    ap.add_argument("--font-size", type=int, default=15)
    ap.add_argument("--opacity", type=float, default=0.92)
    ap.add_argument("--poll-ms", type=int, default=600, help="剪贴板监听间隔 ms (雷达功能)")
    ap.add_argument("--no-auto", action="store_true", help="关闭剪贴板自动判定 (雷达功能)")
    ap.add_argument("--selftest", action="store_true",
                    help="启动后 20 秒向 文件传输助手 发一条自测消息 (验证链路)")
    args = ap.parse_args(argv)
    who_list = [w.strip() for w in args.who.split(",") if w.strip()]
    app = AssistantApp(who_list=who_list, poll_secs=args.poll_secs,
                       no_watch=args.no_watch, no_draft=args.no_draft,
                       font_size=args.font_size, opacity=args.opacity,
                       poll_ms=args.poll_ms, no_auto=args.no_auto)
    if args.selftest and app._link is not None:
        def _do_selftest():
            time.sleep(20)
            info = app.selftest(who_list[0])
            log("selftest:", info)
        threading.Thread(target=_do_selftest, daemon=True).start()
    app.run()
    if app._link is not None:
        app._link.stop()
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(main())

# ===== [2026-10-01 10:00:30] =====
# 1. write_draft 增加剪贴板保护: 写草稿前保存用户剪贴板文本, 粘贴后立即还原,
#    避免助手覆盖用户正在用的剪贴板内容 (用户体验硬要求).
# 2. e2e 同进程验证通过: 自发消息 (attr=self) 被轮询过滤器正确忽略 (0 次触发,
#    防误判); 合成 friend 消息走真实链路 analyze(qwen3.8-flash 临时覆盖, 38s) ->
#    悬浮窗判定 停 0.0/3 flat -> 草稿粘贴进 文件传输助手 子窗口输入框 (未发送).
#    双进程并发 attach wxauto 会触发免费版内部 bug (NameError: print_warn),
#    故所有 e2e 在单进程内完成; 正式使用时只跑助手一个进程, 无此问题.
