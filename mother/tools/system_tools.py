"""Screen awareness + self-monitoring tools.

look_at_screen — capture the active window (or full primary screen),
send the single screenshot to Claude vision, and return what it shows
/ the answer to the user's question about it. Local-only: the image
goes to the Anthropic API for the one vision call and is never written
to disk.

system_status — live telemetry on Ultron's own body: the PM2 process
table, service health probes, and host CPU/RAM, so "how are you
feeling" returns real state instead of persona filler. Also exposes
degraded_summary() for the world-state brief, which lets Ultron
NOTICE a dead service and mention it instead of silently degrading.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger("mother.system_tools")

# Anthropic vision guidance: long edge <= ~1568px keeps token cost sane
# with no meaningful loss for UI text.
_MAX_EDGE = 1568
_VISION_MODEL = "anthropic/claude-haiku-4-5-20251001"
_VISION_MAX_TOKENS = 500

_dpi_aware = False


def _ensure_dpi_aware() -> None:
    """Without per-monitor DPI awareness, GetWindowRect returns virtual
    coords that don't match ImageGrab pixels on scaled displays and the
    crop lands on the wrong region."""
    global _dpi_aware
    if _dpi_aware:
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    _dpi_aware = True


def _active_window() -> tuple:
    """(bbox or None, window title)."""
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None, ""
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None, buf.value
        bbox = (rect.left, rect.top, rect.right, rect.bottom)
        if bbox[2] - bbox[0] < 50 or bbox[3] - bbox[1] < 50:
            return None, buf.value
        return bbox, buf.value
    except Exception as e:
        logger.debug("[screen] active window lookup failed: %s", e)
        return None, ""


def look_at_screen(args: dict) -> str:
    """Tool: screenshot the active window (or full screen) and answer
    a question about it via Claude vision.

    Args:
        question (optional): what the user wants to know about the screen
        scope (optional): 'active_window' (default) or 'full_screen'
    """
    question = (args.get("question") or "").strip()
    scope = (args.get("scope") or "active_window").strip().lower()
    _ensure_dpi_aware()

    try:
        from PIL import ImageGrab
    except ImportError:
        return "Screen capture unavailable (Pillow not installed)."

    bbox, title = (None, "")
    if scope != "full_screen":
        bbox, title = _active_window()
    try:
        img = ImageGrab.grab(bbox=bbox, all_screens=bbox is not None)
    except Exception as e:
        return f"Screen capture failed: {e}"

    # Downscale + JPEG keeps the vision call fast and cheap.
    w, h = img.size
    if max(w, h) > _MAX_EDGE:
        scale = _MAX_EDGE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()

    ask = question or "Describe what is on screen, concisely but concretely."
    prompt = (
        f"This is a screenshot of the user's screen"
        + (f' (active window: "{title}")' if title else "")
        + f". {ask} Read any text that matters for the answer. "
        "Be specific and brief — this will be spoken aloud."
    )
    try:
        import litellm
        resp = litellm.completion(
            model=_VISION_MODEL,
            max_tokens=_VISION_MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        out = (resp.choices[0].message.content or "").strip()
        if not out:
            return "Vision model returned nothing for the screenshot."
        return (f"[Active window: {title}] " if title else "") + out
    except Exception as e:
        return f"Screen capture worked but the vision call failed: {e}"


# ────────────────────── self-monitoring ─────────────────────────────

_RAG_HEALTH_URL = os.environ.get(
    "ULTRON_RAG_BASE", "http://127.0.0.1:8123"
)


def _pm2_table() -> list:
    """Parsed `pm2 jlist` (name, status, cpu, mem MB, restarts, uptime s).
    Empty list when pm2 is missing or errors."""
    pm2 = shutil.which("pm2") or shutil.which("pm2.cmd")
    if not pm2:
        return []
    try:
        r = subprocess.run(
            [pm2, "jlist"], capture_output=True, text=True, timeout=8,
        )
        data = json.loads(r.stdout or "[]")
    except Exception as e:
        logger.debug("[status] pm2 jlist failed: %s", e)
        return []
    out = []
    now_ms = time.time() * 1000
    for p in data:
        env = p.get("pm2_env") or {}
        monit = p.get("monit") or {}
        out.append({
            "name": p.get("name", "?"),
            "status": env.get("status", "?"),
            "cpu": monit.get("cpu", 0),
            "mem_mb": round((monit.get("memory") or 0) / 1e6),
            "restarts": env.get("restart_time", 0),
            "uptime_s": max(0, int((now_ms - (env.get("pm_uptime") or now_ms)) / 1000)),
        })
    return out


def _rag_up() -> bool:
    try:
        import httpx
        r = httpx.get(_RAG_HEALTH_URL + "/health", timeout=2.0)
        # Any HTTP answer (even 404) means the process is serving.
        return r.status_code < 500
    except Exception:
        return False


def system_status(args: dict) -> str:
    """Tool: Ultron's own runtime state — processes, services, host."""
    lines = []
    procs = _pm2_table()
    if procs:
        for p in procs:
            up_h = p["uptime_s"] / 3600
            lines.append(
                f"{p['name']}: {p['status']}, {p['mem_mb']}MB, "
                f"{p['restarts']} restarts, up {up_h:.1f}h"
            )
    else:
        lines.append("pm2 table unavailable (not running under pm2?)")
    lines.append(f"rag service: {'up' if _rag_up() else 'DOWN'}")
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.2)
        mem = psutil.virtual_memory()
        lines.append(
            f"host: {cpu:.0f}% CPU, {mem.percent:.0f}% RAM used "
            f"({mem.available / 1e9:.1f}GB free)"
        )
    except Exception:
        pass
    return "\n".join(lines)


def search_conversations(args: dict, *, current_user=None) -> str:
    """Tool: keyword search over the persisted conversation history.

    Answers "what did we talk about / decide about X" from the actual
    transcript instead of the model guessing from its 20-exchange
    window. Simple case-insensitive substring match over the saved
    messages, newest first.
    """
    query = (args.get("query") or "").strip().lower()
    if not query:
        return "No search query provided."
    user_id = getattr(current_user, "user_id", None) or "win"
    from pathlib import Path
    path = (
        Path(__file__).resolve().parents[2]
        / "assistant" / "memory" / "users" / user_id / "conv_history.json"
    )
    if not path.exists():
        return "No conversation history on file."
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        msgs = payload.get("messages") if isinstance(payload, dict) else payload
        summary = payload.get("summary", "") if isinstance(payload, dict) else ""
    except Exception as e:
        return f"History unreadable: {e}"

    terms = [t for t in query.split() if len(t) > 2] or [query]
    hits = []
    for i, m in enumerate(msgs or []):
        content = str(m.get("content", ""))
        if any(t in content.lower() for t in terms):
            who = "user" if m.get("role") == "user" else "you"
            hits.append(f"{who}: {content[:200]}")
    out_lines = []
    if summary and any(t in summary.lower() for t in terms):
        out_lines.append(f"From the compacted summary: {summary[:400]}")
    if hits:
        out_lines.append("Matching exchanges (oldest to newest):")
        out_lines.extend(hits[-8:])
    if not out_lines:
        return f"Nothing in stored conversation history matches {query!r}."
    return "\n".join(out_lines)


def degraded_summary() -> str:
    """One short line naming anything wrong, or '' when all healthy.
    Used by the world-state brief so Ultron notices problems unprompted;
    empty when nominal to avoid injecting noise every turn."""
    problems = []
    for p in _pm2_table():
        if p["status"] != "online":
            problems.append(f"{p['name']} is {p['status']}")
        elif p["restarts"] >= 15:
            problems.append(f"{p['name']} has restarted {p['restarts']}x")
    if not _rag_up():
        problems.append("rag/notes service unreachable (memory of notes+code degraded)")
    return "; ".join(problems)
