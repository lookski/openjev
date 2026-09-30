#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeChat radar launcher: a small always-on-top floating widget that turns
"WeChat screenshot" or "WeChat multi-select copy" into a crush verdict with
two clicks. No WeChat hooking, no automation of WeChat itself: it only reads
the Windows clipboard and the screen region the USER captured - identical
trust level to the user pressing Ctrl+C themselves.

编写时间: 2026-09-30 23:07:53
脚本功能: floating tkinter widget (always-on-top, draggable, 320px) that
          listens for two hotkeys while running:
          - F2 = analyze the current clipboard text as a chat log
          - F3 = analyze the screenshot stored on the clipboard as an image
            (user presses Win+Shift+S / WeChat screenshot Alt+A first, box
            the chat area; the tool OCRs it locally with RapidOCR and sends
            the recognized transcript to the LLM engine)
          Verdict pops up on the widget (big colored move char + numbers +
          reason + next_advice). Optional --launch-minimized. The LLM call,
          config and parsing are all reused from openjev.crush_llm /
          openjev.llm_config. OCR is local-only; recognized text goes to the
          configured LLM endpoint exactly like a pasted log.
参数: python -m openjev.wechat_radar [--port 8792] [--no-hotkeys]
          [--font-size N] [--opacity 0.9]
          Hotkeys: F2 clipboard-text analyze, F3 clipboard-image OCR analyze.
          Escape inside the result area copies the advice; double-click the
          title bar toggles compact mode; right-click closes.
输入格式: clipboard CF_UNICODETEXT ("name: message" lines, WeChat
          multi-select 复制 output parses fine) or clipboard CF_DIB/PNG
          bitmap from any screenshot tool; OCR output is grouped into
          speaker turns by geometry (left column = them, right = me, name
          lines are dropped by heuristics).
输出格式: verdict on the widget; stderr one-line status; nothing is written
          to disk (screenshots stay in RAM).
依赖: stdlib tkinter; pywin32 (clipboard); Pillow; rapidocr-onnxruntime
          (local OCR, first run downloads/loads ~15MB models);
          openjev.crush_llm + openjev.llm_config for the verdict.
注意事项: NOT a WeChat hook - nothing injects or reads WeChat memory; the
          user manually puts content on the clipboard. Do not run on a
          shared/desktop-remoted machine you do not trust (it reads the
          clipboard only on F2/F3, never in the background). The OCR path
          needs the chat window in light contrast; dark-mode WeChat works
          but confidence drops. F2/F3 are global hotkeys (RegisterHotKey);
          they are unregistered on exit and do not swallow the keys (other
          apps still see them).

===== [2026-10-01 00:11:15] =====
1. 实测修正轮: 修复 _dib_to_pil 的 struct.unpack 长度错误 (切片 20 字节但
   '<IiiHH' 恰需 16, 实测报 unpack requires a buffer of 16 bytes -> 整个
   截图通道静默返回 None); 修复 OCR 坐标系混用 (rows 来自 2x 放大图而
   img_w 用原图宽 -> 左侧气泡全部误判为 我); OCR 前 2x LANCZOS 放大,
   修正了 周未 误识 (原字形下 1x 识别为 周未, 2x 为 周末).
2. log_to_transcript 增加 WeChat 多选复制无冒号格式的严格交替解析
   (偶数行名字/奇数行消息), 比名字启发式可靠; 名字启发式保留给混排文本.
3. _name_like 增加 名字重复出现>=2次 信号 (同一名学多条消息上方重复),
   同列下方跟随行做次级信号; 原版把 周末一起吃饭吗 误判成名字.
4. 实测记录: RapidOCR 对单字气泡 (嗯) 低于检测下限, 1x/2x/4x 均无法
   识别, 属引擎检测下限, 已在 docstring 与 README 标注为已知损失.
5. e2e 实测: 剪贴板文字 -> 判定 (stub + 真端点 qwen3.8-flash 各一);
   截图 -> OCR -> transcript -> 判定; 热键注册/注销 OK; 悬浮窗 4 秒
   完整启停 OK. 网关今晚严重拥堵 (glm-5.3-flash 300s 超时, gemini 503),
   换 qwen3.8-flash 完成 e2e (配置未改, 临时参数覆盖).

===== [2026-10-01 01:00:33] =====
1. 自动监听模式上线 (用户需求 "读取不能更简单一点吗"): 悬浮窗内置
   GetClipboardSequenceNumber 零开销轮询 (默认 600ms, --poll-ms 可调),
   剪贴板新内容落地即自动分析; 新图片 (>=200x80) 直接触发, 新文字需过
   聊天相似度门槛 (>=2 行发言 + >=6 个中文字符) 才触发, 非聊天内容
   静默忽略; 同文本去重 (剪贴板连按两次复制不重复分析); --no-auto 关闭.
2. 关键坑: 首次轮询把启动时剪贴板上已有内容当作基线, 若用户在启动后
   1.5 秒内截图会被静默吞掉 -> 首次轮询延迟到 1500ms, 且写入早于基线
   的场景在实测中复现后确认; 自动模式下 OCR 出空 transcript 显示
   "截图不含聊天内容, 已忽略" 而非报错.
3. _worker 增加一次静默重试 (网关单请求偶发卡死, 与 hub 的重试策略对齐).
4. 实测 (stub 引擎): 图片变更自动触发 PASS; 聊天文本自动触发 PASS;
   非聊天文本 (URL+普通段落) 静默 PASS; 同文本重复去重 PASS; 手动热键
   强制分析不受去重限制 PASS; 真引擎 OCR 链路 2.9s 识别 3 行 ->
   qwen3.8-flash 判定 缓 1.6/3 falling PASS; 悬浮窗 4 秒启停 PASS.
   注: 网关拥堵时段 analyze 主模型请求 >330s 无响应, 但小请求 17s OK;
   悬浮窗 worker 重试后会显示明确失败信息, 不悬挂.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import io
import re
import struct
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont

from openjev.crush_llm import LINE_RE, MOVE_LABELS, analyze

try:
    import win32clipboard
    import win32con
    import win32gui
except ImportError:  # pragma: no cover - Windows-only tool
    win32clipboard = None

APP_TITLE = "💗 Crush Radar"
HOTKEY_TEXT_ID = 1  # RegisterHotKey id for F2
HOTKEY_IMG_ID = 2   # RegisterHotKey id for F3
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000

MOVE_COLORS = {"冲": "#3fb950", "稳": "#58a6ff", "缓": "#d29922", "停": "#f85149"}


def clipboard_text():
    """Return clipboard CF_UNICODETEXT or '' (never raises)."""
    if win32clipboard is None:
        return ""
    try:
        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT) or ""
        finally:
            win32clipboard.CloseClipboard()
    except Exception:
        return ""
    return ""


def clipboard_image():
    """Return the clipboard bitmap as a PIL Image, or None.

    Handles CF_DIB (Win+Shift+S, WeChat Alt+A, PrtSc all land here) and the
    PNG format some apps prefer. Image is fully detached from the clipboard
    before return; no temp files.
    """
    if win32clipboard is None:
        return None
    try:
        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_DIB):
                data = win32clipboard.GetClipboardData(win32con.CF_DIB)
                return _dib_to_pil(data)
            png_fmt = win32clipboard.RegisterClipboardFormat("PNG")
            if win32clipboard.IsClipboardFormatAvailable(png_fmt):
                data = win32clipboard.GetClipboardData(png_fmt)
                from PIL import Image

                img = Image.open(io.BytesIO(data))
                img.load()
                return img.convert("RGB")
        finally:
            win32clipboard.CloseClipboard()
    except Exception:
        return None
    return None


def _dib_to_pil(data):
    """Convert a CF_DIB blob to a PIL Image (handles top-down DIBs).

    Wrap the DIB in a minimal BITMAPFILEHEADER so PIL parses it; a negative
    biHeight (top-down row order, emitted by some capture tools) is flipped
    back after load.
    """
    (_hsize, _w, h, _planes, _bpp) = struct.unpack("<IiiHH", data[:16])
    top_down = h < 0
    bmp = (b"BM" + struct.pack("<IHHI", 14 + len(data), 0, 0, 14) + data)
    from PIL import Image

    img = Image.open(io.BytesIO(bmp))
    img.load()
    if top_down:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img.convert("RGB")


def log_to_transcript(raw):
    """Clipboard text -> cleaned 'name: message' transcript.

    WeChat multi-select 复制 emits 'name\\nmessage\\n' (no colon) or
    'name: message'; both are normalized. Pure numbers/dates/timestamps and
    [图片]/[语音] system tags are dropped.
    """
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    # WeChat multi-select 复制 with no colons is a STRICT name/message
    # alternation (name, msg, name, msg...). Position parity is the most
    # reliable signal there; the name-like heuristic alone eats short
    # messages like 在吗/忙.
    colon_free = not any(LINE_RE.match(l) or _SYSTEM_TAG.match(l) for l in lines)
    if colon_free and len(lines) >= 2 and len(lines) % 2 == 0:
        return "\n".join(
            "%s: %s" % (lines[i], lines[i + 1])
            for i in range(0, len(lines), 2)
        )
    entries = []
    pending_name = None
    for idx, line in enumerate(lines):
        m = LINE_RE.match(line)
        if m and not _SYSTEM_TAG.match(line):
            entries.append((m.group(1).strip(), m.group(2).strip()))
            pending_name = None
            continue
        if _SYSTEM_TAG.match(line):
            continue
        # a short standalone line followed by longer text = WeChat name line
        # (names in WeChat multi-select copy always precede their message;
        # require the next line to be LONGER so short messages like 在吗/忙
        # are not eaten as names)
        nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
        nxt_is_message = (
            bool(nxt)
            and (len(nxt) > len(line) or _looks_like_message(nxt))
            and not _SYSTEM_TAG.match(nxt)
            and not LINE_RE.match(nxt)
        )
        if len(line) <= 24 and not _looks_like_message(line) and nxt_is_message:
            pending_name = line
            continue
        name = pending_name or "对方"
        entries.append((name, line))
        pending_name = None
    return "\n".join("%s: %s" % (n, m) for n, m in entries)


_SYSTEM_TAG = re.compile(
    r"^\[(图片|语音|视频|表情|文件|链接|红包|转账|音乐|卡片|动画表情)\]$|"
    r"^\d{1,2}:\d{2}(:\d{2})?$|"  # pure time like 12:30
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{1,2}月\d{1,2}日|"
    r"^星期[一二三四五六天日]|^周[一二三四五六天日]"
)


def _looks_like_message(line):
    """Heuristic: names are short, no punctuation tails, no system tags.

    A bare pronoun/name-ish short line (周末一起吃饭吗 has no punctuation
    yet is clearly a message) cannot be distinguished by punctuation alone,
    so also treat lines with verbs-question particles or length >= 6 as
    message candidates ONLY when they are CJK-heavy; pure ASCII short
    tokens stay name-like.
    """
    if _SYSTEM_TAG.match(line):
        return False
    if line.endswith(("：", ":", "。", "，", "？", "！")):
        return False
    cjk = sum(1 for c in line if "\u4e00" <= c <= "\u9fff")
    if len(line) > 24:
        return True
    if cjk >= 4:  # multi-CJK short line: likely a short message, not a name
        return True
    return any(c in line for c in "。，？！,?!")


def ocr_image_to_transcript(img):
    """OCR a WeChat chat screenshot into a 'name: message' transcript.

    Geometry heuristics for WeChat 4.0 light theme:
    - name labels sit ABOVE their bubble, small gray text; message lines
      inside bubbles are larger/darker;
    - left-aligned bubbles = the other person, right-aligned = me;
    - each text line gets (x_center, y_center); lines are sorted by y then
      grouped; alignment decides speaker, tiny gray lines are names.
    Confidence < 0.35 lines are dropped. Returns '' when OCR finds nothing.
    The image is upscaled 2x with LANCZOS first (sharper small bubbles);
    single-character bubbles (嗯 / 哦) fall below the OCR detection floor
    and are accepted losses - a one-char reply rarely flips a verdict.
    """
    import numpy as np
    from PIL import Image

    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    scale = 2 if img.width < 1600 else 1
    big = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    result, _ = engine(np.array(big))
    if not result:
        return ""
    rows = []
    for box, text, conf in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        h = max(ys) - min(ys)
        rows.append({
            "x": sum(xs) / len(xs),
            "y": sum(ys) / len(ys),
            "h": h,
            "text": text.strip(),
            "conf": float(conf),
        })
    rows.sort(key=lambda r: r["y"])
    img_w = img.width * scale  # rows were measured on the scaled image
    repeat_counts = {}
    for r in rows:
        repeat_counts[r["text"]] = repeat_counts.get(r["text"], 0) + 1
    entries = []
    pending_name = None
    for r in rows:
        t = r["text"]
        if not t or r["conf"] < 0.35:
            continue
        if _SYSTEM_TAG.match(t):
            continue
        is_name = _name_like(t, r, rows, repeat_counts)
        if is_name:
            pending_name = t
            continue
        right_side = r["x"] > img_w * 0.55
        name = "我" if right_side else (pending_name or "对方")
        entries.append((name, t))
        pending_name = None
    return "\n".join("%s: %s" % (n, m) for n, m in entries)


def _name_like(text, row, rows, repeat_counts=None):
    """WeChat name labels: repeated short gray lines above bubbles.

    Signals, in order of reliability:
    1. the same text appears at 2+ different y positions (people send many
       messages; their name label repeats above each bubble group);
    2. a small line with a message row starting at nearly the same x
       within 4*h below (name-above-bubble layout);
    3. never: rows containing terminal punctuation or system tags.
    """
    if len(text) > 12 or len(text) < 2:
        return False
    if text.endswith(("：", ":", "。", "，", "？", "！")):
        return False
    if repeat_counts and repeat_counts.get(text, 0) >= 2:
        return True
    below = [r for r in rows if 0 < r["y"] - row["y"] < row["h"] * 4]
    if not below:
        return False
    nearest = min(below, key=lambda r: r["y"] - row["y"])
    same_column = abs(nearest["x"] - row["x"]) < max(row["h"], 24)
    return same_column and len(nearest["text"]) > len(text)


class RadarApp:
    """The floating widget: tkinter window + global hotkeys + worker thread.

    Auto mode (default): a clipboard poller (GetClipboardSequenceNumber,
    zero cost) watches for new content. A NEW IMAGE is always analyzed
    (you just took a screenshot); a NEW TEXT blob is analyzed only when it
    looks like a chat log (name/message lines). Manual hotkeys still work
    and force analysis regardless of the chat-likeness check.
    """

    def __init__(self, port_host=None, font_size=15, opacity=0.92,
                 poll_ms=600, no_auto=False):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", opacity)
        self.root.configure(bg="#101418")
        self.root.geometry("330x170-40-140")
        self.root.minsize(300, 150)
        self.font_size = font_size
        self._busy = False
        self._compact = False
        self._poll_ms = poll_ms
        self._last_seq = 0
        self._last_text = ""
        self._auto = not no_auto
        self._build_ui()
        self._install_hotkeys()
        if self._auto:
            # first poll delayed: whatever is on the clipboard at startup
            # becomes the baseline, so an image dropped in the first
            # seconds isn't silently swallowed as "initial state"
            self.root.after(1500, self._watch_clipboard)
        self.root.after(1500, self._hint_fade)

    # ---------- UI ----------
    def _build_ui(self):
        bg = "#101418"
        f_title = tkfont.Font(size=self.font_size - 2, weight="bold")
        f_move = tkfont.Font(size=self.font_size + 26, weight="bold")
        f_body = tkfont.Font(size=self.font_size)
        f_small = tkfont.Font(size=self.font_size - 3)
        self.title = tk.Label(self.root, text=APP_TITLE + "  (自动监听: 截图即判定)",
                              bg=bg, fg="#8b949e", font=f_title, cursor="fleur")
        self.title.pack(fill="x", pady=(4, 0))
        self.title.bind("<Button-1>", self._start_drag)
        self.title.bind("<B1-Motion>", self._do_drag)
        self.title.bind("<Double-Button-1>", lambda e: self._toggle_compact())
        self.title.bind("<Button-3>", lambda e: self.root.destroy())
        self.move = tk.Label(self.root, text="…", bg=bg, fg="#58a6ff", font=f_move)
        self.move.pack(pady=2)
        hint = "自动模式: 微信截图/多选复制后判定自动弹出\n热键仍在: Ctrl+F2 文字 | Ctrl+F3 截图"
        self.body = tk.Label(self.root, text=hint,
                             bg=bg, fg="#e6e6e6", font=f_body,
                             wraplength=300, justify="left")
        self.body.pack(fill="both", expand=True, padx=8)
        self.body.bind("<Button-1>", self._copy_advice)

    def _toggle_compact(self):
        self._compact = not self._compact
        if self._compact:
            self.body.pack_forget()
            self.root.geometry("")
        else:
            self.body.pack(fill="both", expand=True, padx=8)

    def _start_drag(self, event):
        self._dx = event.x
        self._dy = event.y

    def _do_drag(self, event):
        x = self.root.winfo_pointerx() - self._dx
        y = self.root.winfo_pointery() - self._dy
        self.root.geometry("+%d+%d" % (x, y))

    def _hint_fade(self):
        if str(self.move["text"]) == "…":
            self.move.configure(text="待命")

    # ---------- hotkeys ----------
    def _install_hotkeys(self):
        if win32gui is None:
            self._set_body("非 Windows 或缺 pywin32: 热键不可用, 仍可用 CLI")
            return
        user32 = ctypes.windll.user32
        ok1 = user32.RegisterHotKey(None, HOTKEY_TEXT_ID,
                                    win32con.MOD_CONTROL | MOD_NOREPEAT,
                                    win32con.VK_F2)
        ok2 = user32.RegisterHotKey(None, HOTKEY_IMG_ID,
                                    win32con.MOD_CONTROL | MOD_NOREPEAT,
                                    win32con.VK_F3)
        self._hot_ok = bool(ok1 and ok2)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _pump_hotkeys(self):
        """Poll the Windows message queue so WM_HOTKEY reaches us."""
        msg = ctypes.wintypes.MSG() if hasattr(ctypes, "wintypes") else None
        if msg is None:
            return
        while ctypes.windll.user32.PeekMessageW(
                ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE
            if msg.message == WM_HOTKEY:
                if msg.wParam == HOTKEY_TEXT_ID:
                    self.analyze_clipboard_text()
                elif msg.wParam == HOTKEY_IMG_ID:
                    self.analyze_clipboard_image()
        self.root.after(80, self._pump_hotkeys)

    def _on_close(self):
        try:
            ctypes.windll.user32.UnregisterHotKey(None, HOTKEY_TEXT_ID)
            ctypes.windll.user32.UnregisterHotKey(None, HOTKEY_IMG_ID)
        except Exception:
            pass
        self.root.destroy()

    # ---------- clipboard watch (auto mode) ----------
    def _watch_clipboard(self):
        """Zero-cost poll: GetClipboardSequenceNumber changes on every
        clipboard write. Images -> auto analyze; text -> only when it
        parses as a chat. Hotkeys still force-analysis anything."""
        try:
            seq = ctypes.windll.user32.GetClipboardSequenceNumber()
        except Exception:
            seq = 0
        if seq and self._last_seq and seq != self._last_seq:
            self._last_seq = seq
            self._on_clipboard_changed()
        elif seq and not self._last_seq:
            self._last_seq = seq
        self.root.after(self._poll_ms, self._watch_clipboard)

    def _on_clipboard_changed(self):
        """Dispatch a clipboard change: image wins, then chat-like text."""
        img = clipboard_image()
        if img is not None and (img.width >= 200 and img.height >= 80):
            self.analyze_clipboard_image()
            return
        raw = clipboard_text()
        if not raw.strip():
            return
        # dedupe: same text as last auto-run -> skip (multi-copy of one log)
        if raw == self._last_text:
            return
        transcript = log_to_transcript(raw)
        entries = transcript.count(": ")
        # chat-likeness gate for AUTOMATIC runs only (hotkey path bypasses):
        # need >= 2 speaker lines and some CJK/latin message mass
        cjk_mass = sum(1 for c in raw if "\u4e00" <= c <= "\u9fff")
        if entries >= 2 and cjk_mass >= 6:
            self._last_text = raw
            self._run_async(transcript, source="auto")

    # ---------- actions ----------
    def analyze_clipboard_text(self):
        """Ctrl+F2: clipboard text -> transcript -> verdict (forced)."""
        raw = clipboard_text()
        if not raw.strip():
            self._flash("剪贴板没有文字. 微信多选消息 -> 复制, 或直接截图按 Ctrl+F3")
            return
        self._last_text = raw  # manual run marks dedupe watermark
        transcript = log_to_transcript(raw)
        if not transcript.strip():
            self._flash("剪贴板文字不是聊天记录 (没有可识别的发言行)")
            return
        self._run_async(transcript)

    def analyze_clipboard_image(self):
        """Ctrl+F3 or auto: clipboard screenshot -> OCR -> verdict."""
        img = clipboard_image()
        if img is None:
            self._flash("剪贴板没有截图. 微信 Alt+A 或 Win+Shift+S 截聊天区即可")
            return
        self._set_body("OCR 识别中 (本地, 1-3 秒)…")
        self.root.update_idletasks()
        try:
            transcript = ocr_image_to_transcript(img)
        except Exception as exc:  # noqa: BLE001
            self._flash("OCR 失败: %s" % exc)
            return
        if not transcript.strip():
            if getattr(self, "_auto", False):
                self._set_move("…", "#8b949e")
                self._set_body("截图不含聊天内容, 已忽略 (截聊天区再试)")
            else:
                self._flash("截图里没认出聊天文字 (太暗/太小?). 放大窗口再截一次")
            return
        self._set_body("已识别 %d 行, 大模型分析中 (20-100 秒)…" % transcript.count("\n"))
        self._run_async(transcript)

    def _run_async(self, transcript, source="manual"):
        if self._busy:
            if source == "manual":
                self._flash("上一单还在跑, 稍等")
            return
        self._busy = True
        self._set_move("…", "#8b949e")
        self._set_body("大模型分析中 (20-100 秒)…")
        threading.Thread(target=self._worker, args=(transcript,), daemon=True).start()

    def _worker(self, transcript):
        try:
            v = analyze(transcript, timeout=180)
            self.root.after(0, self._show_verdict, v)
        except Exception as exc:  # noqa: BLE001
            # one silent retry: the gateway sometimes stalls a single request
            try:
                v = analyze(transcript, timeout=180)
                self.root.after(0, self._show_verdict, v)
            except Exception as exc2:  # noqa: BLE001
                self.root.after(0, self._flash,
                                "分析失败 (重试过后仍失败): %s" % exc2)
        finally:
            self._busy = False

    # ---------- display ----------
    def _show_verdict(self, v):
        label = MOVE_LABELS[v["move"]]
        self._set_move(label, MOVE_COLORS[label])
        warmth = ("%.2f" % v["warmth_signals"]) if v.get("warmth_signals") is not None else "n/a"
        lines = [
            "对方兴趣 %.1f/3 | 我 %.1f/3 | %s | 好感 %s" % (
                v["interest_their"], v["interest_mine"], v["trend"], warmth),
            "理由: %s" % v["reason"],
        ]
        if v.get("next_advice"):
            lines.append("下一步: %s" % v["next_advice"])
        self._set_body("\n".join(lines))
        self._verdict = v

    def _copy_advice(self, event):
        v = getattr(self, "_verdict", None)
        if not v:
            return
        text = "判定: %s\n理由: %s\n下一步: %s" % (
            MOVE_LABELS[v["move"]], v["reason"], v.get("next_advice", ""))
        if win32clipboard is not None:
            try:
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(
                    win32con.CF_UNICODETEXT, text)
                win32clipboard.CloseClipboard()
                self._flash("已复制判定 (点结果区复制, 右键标题关闭)")
            except Exception:
                pass

    def _set_move(self, text, color):
        self.move.configure(text=text, fg=color)

    def _set_body(self, text):
        self.body.configure(text=text)

    def _flash(self, text):
        self._set_body(text)
        self._set_move("!", "#f85149")

    def run(self):
        if getattr(self, "_hot_ok", False):
            self._pump_hotkeys()
        self.root.mainloop()


def main():
    """CLI entry."""
    ap = argparse.ArgumentParser(description="OpenJev WeChat radar launcher")
    ap.add_argument("--font-size", type=int, default=15)
    ap.add_argument("--opacity", type=float, default=0.92)
    ap.add_argument("--no-auto", action="store_true",
                    help="disable clipboard auto-watch (hotkeys only)")
    ap.add_argument("--poll-ms", type=int, default=600,
                    help="clipboard poll interval, ms (default 600)")
    args = ap.parse_args()
    RadarApp(font_size=args.font_size, opacity=args.opacity,
             poll_ms=args.poll_ms, no_auto=args.no_auto).run()


if __name__ == "__main__":
    main()
