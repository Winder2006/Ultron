"""Persistent Python REPL for the execute_python tool.

Each WS session gets its own long-lived Python subprocess so variables
persist across turns:

    Turn 1:  "Load that CSV at C:\\data\\sales.csv into a DataFrame"
             → exec: import pandas as pd; df = pd.read_csv(r"C:\\data\\sales.csv")
             → "Loaded 12,043 rows, 8 columns."

    Turn 2:  "What's the median revenue?"
             → exec: df['revenue'].median()
             → "$4,820"

    Turn 3:  "Plot the distribution"
             → exec: df['revenue'].hist(); plt.savefig(...)
             → returns the image base64

The subprocess runs in the project's venv so pandas/numpy/matplotlib are
available without setup. Each exec is wrapped in stdout/stderr capture
markers so we can cleanly delimit one call's output from the next.

Honesty: this is NOT a security sandbox. The subprocess runs with the
same filesystem and network access as the parent server. The user is
sending voice queries to their own LLM running on their own machine; if
they ask for "rm -rf /", that's their problem. The subprocess gives:
  • Resource isolation — runaway loops get killed via timeout
  • Crash isolation — a subprocess SEGV doesn't take down the API
  • State persistence — variables survive between calls
  • Clean shutdown — kill the subprocess, leak nothing

For real isolation later: wrap the subprocess in Docker / WSL2 / a
restricted user account. That's a deployment concern, not a code one.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("mother.tools.code_exec")

# Default soft timeout per exec call. The model can pass a higher value
# up to MAX_TIMEOUT.
DEFAULT_TIMEOUT_S = 8.0
MAX_TIMEOUT_S = 60.0

# Output cap — keep the LLM context manageable. Anything past this is
# truncated; the dashboard /exec view receives the full output (it
# isn't bound by token budgets).
LLM_OUTPUT_CAP = 3000


# Path to the Python that runs the REPL — same venv the backend uses
# so packages match.
def _repl_python() -> str:
    return sys.executable


# Bridge script we exec inside the subprocess. Reads { "code": str }
# JSON lines from stdin, evals each, returns { "stdout": str,
# "stderr": str, "result": str, "duration_s": float, "images": [...] }
# JSON lines on stdout. Markers delimit so the parent doesn't have
# to parse Python's native stream output.
#
# matplotlib is forced to the Agg backend BEFORE pyplot is imported,
# so any plot the user creates won't try to open a Tk window (which
# blocks the subprocess) and savefig works headlessly. After each
# exec, every open figure is captured as a base64 PNG and the
# figures are closed so they don't accumulate across calls.
_BRIDGE_SCRIPT = r"""
import sys, io, json, traceback, time, contextlib, os, base64, pickle, threading

# Force matplotlib to a non-interactive backend BEFORE any user code
# can import pyplot. Without this, plt.show() hangs the subprocess.
os.environ.setdefault("MPLBACKEND", "Agg")

_GLOBALS = {"__name__": "__ultron_exec__"}

# Pre-import matplotlib in a daemon thread at startup so the first
# user-driven plot doesn't pay the 3-5s font-cache scan + backend
# init cost. The thread runs in parallel with the bridge's input
# loop, so REPL readiness is unblocked. If matplotlib isn't
# installed, the import simply fails and user plots will see the
# normal ImportError when they try.
def _prewarm_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        # Touch a figure to force backend init too — no need to save it.
        try:
            f = plt.figure()
            plt.close(f)
        except Exception:
            pass
    except Exception:
        pass
threading.Thread(target=_prewarm_matplotlib, daemon=True, name="mpl-prewarm").start()

def _save_state(path):
    # Pickle the user's REPL namespace so it survives a backend
    # restart. Skip names starting with "_", modules, and anything
    # that doesn't pickle (open file handles, sockets, locks). We
    # try each entry individually so one bad value doesn't lose the
    # whole namespace.
    keep = {}
    skipped = []
    for k, v in list(_GLOBALS.items()):
        if k.startswith("_") or k == "__name__":
            continue
        # Modules, classes from C extensions, etc. usually don't pickle.
        try:
            pickle.dumps(v)
        except Exception:
            skipped.append(k)
            continue
        keep[k] = v
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(keep, f, protocol=pickle.HIGHEST_PROTOCOL)
        return {"saved": len(keep), "skipped": skipped}
    except Exception as e:
        return {"saved": 0, "skipped": skipped, "error": str(e)}

def _load_state(path):
    if not path or not os.path.exists(path):
        return {"loaded": 0, "error": "no_state_file"}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            return {"loaded": 0, "error": "bad_format"}
        for k, v in data.items():
            _GLOBALS[k] = v
        return {"loaded": len(data)}
    except Exception as e:
        return {"loaded": 0, "error": str(e)}

def _capture_figures():
    # Snapshot any open matplotlib figures as base64 PNGs, then close
    # them so the next call starts clean. Returns a list of dicts:
    # [{"png_b64": str, "fig_num": int}, ...]. Empty if matplotlib
    # isn't loaded or no figures are open (the fast path).
    images = []
    try:
        import sys as _s
        plt_mod = _s.modules.get("matplotlib.pyplot")
        if plt_mod is None:
            return images
        for fnum in plt_mod.get_fignums():
            try:
                fig = plt_mod.figure(fnum)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                images.append({
                    "png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "fig_num": int(fnum),
                })
            except Exception:
                # Skip figures that can't render — keep going for others.
                continue
        # Close ALL figures so they don't leak across exec calls.
        try:
            plt_mod.close("all")
        except Exception:
            pass
    except Exception:
        pass
    return images

def _run(code):
    out = io.StringIO()
    err = io.StringIO()
    t0 = time.monotonic()
    result_repr = ""
    try:
        # Try eval first — gives a printable result for expressions.
        try:
            compiled = compile(code, "<voice>", "eval")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                value = eval(compiled, _GLOBALS)
            if value is not None:
                try:
                    result_repr = repr(value)
                except Exception:
                    result_repr = "<unrepresentable>"
        except SyntaxError:
            # Not a single expression — exec it as statements.
            compiled = compile(code, "<voice>", "exec")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compiled, _GLOBALS)
    except Exception:
        err.write(traceback.format_exc())
    return {
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "result": result_repr,
        "duration_s": round(time.monotonic() - t0, 4),
        "images": _capture_figures(),
    }

# Greeting so the parent knows we're alive
sys.stdout.write(json.dumps({"_ready": True}) + "\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        cmd = json.loads(line)
    except Exception as _e:
        sys.stdout.write(json.dumps({"_error": "bad json: " + str(_e)}) + "\n")
        sys.stdout.flush()
        continue
    # Control commands take priority over plain code.
    if isinstance(cmd, dict) and cmd.get("_op") == "save_state":
        sys.stdout.write(json.dumps(_save_state(cmd.get("path") or "")) + "\n")
        sys.stdout.flush()
        continue
    if isinstance(cmd, dict) and cmd.get("_op") == "load_state":
        sys.stdout.write(json.dumps(_load_state(cmd.get("path") or "")) + "\n")
        sys.stdout.flush()
        continue
    code = cmd.get("code", "") if isinstance(cmd, dict) else ""
    out = _run(code)
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
"""


class PersistentRepl:
    """One long-lived Python subprocess. Methods are thread-safe via
    a single mutex — only one execution runs at a time per session."""

    def __init__(self, scratch_dir: Optional[Path] = None):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._scratch = scratch_dir or Path(os.environ.get("TEMP", "/tmp")) / f"ultron_repl_{uuid.uuid4().hex[:8]}"
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._dead = False

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        # Spawn fresh
        self._proc = subprocess.Popen(
            [_repl_python(), "-u", "-c", _BRIDGE_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self._scratch),
            text=True,
            encoding="utf-8",
            bufsize=1,  # line-buffered
        )
        # Wait for the bridge's "_ready" handshake — bounded so a
        # busted bridge doesn't hang us forever.
        ready_q: "queue.Queue[Optional[str]]" = queue.Queue()

        def _wait_ready():
            try:
                line = self._proc.stdout.readline()
                ready_q.put(line)
            except Exception:
                ready_q.put(None)

        t = threading.Thread(target=_wait_ready, daemon=True)
        t.start()
        try:
            line = ready_q.get(timeout=5.0)
        except queue.Empty:
            self.shutdown()
            raise RuntimeError("REPL subprocess didn't signal ready in 5s")
        if not line or "_ready" not in line:
            self.shutdown()
            raise RuntimeError(f"REPL subprocess bad handshake: {line!r}")
        logger.info("[code_exec] REPL ready (cwd=%s)", self._scratch)

    def warmup(self) -> None:
        """Start the subprocess if it isn't running, and pre-import the
        heavy data stack. Called from the WS connect path (off the
        loop) so the first execute_python of a session pays neither
        spawn + interpreter boot (~300-800ms) nor the numpy/pandas/
        matplotlib import tax (~1-3s) on the hot path. np/pd land in
        the namespace under their conventional aliases — models write
        `pd.read_csv` unprompted anyway."""
        self.execute(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n",
            timeout_s=20.0,
        )

    def execute(self, code: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> Dict:
        """Run code, return {stdout, stderr, result, duration_s, timed_out}.

        Timeout kills+respawns the subprocess (variables lost) — this
        is rare and only happens on runaway loops the user explicitly
        asked for. Better than hanging the WS task.
        """
        timeout_s = max(0.5, min(MAX_TIMEOUT_S, float(timeout_s)))
        with self._lock:
            try:
                self._ensure_started()
            except Exception as e:
                return {
                    "stdout": "",
                    "stderr": f"REPL startup failed: {e}",
                    "result": "",
                    "duration_s": 0.0,
                    "timed_out": False,
                }

            import json as _json
            request = _json.dumps({"code": code}) + "\n"
            try:
                assert self._proc and self._proc.stdin
                self._proc.stdin.write(request)
                self._proc.stdin.flush()
            except Exception as e:
                # Pipe broken — kill and report. Next call will respawn.
                self.shutdown()
                return {
                    "stdout": "",
                    "stderr": f"REPL stdin write failed: {e}",
                    "result": "",
                    "duration_s": 0.0,
                    "timed_out": False,
                }

            # Read response with timeout via a reader thread.
            line_q: "queue.Queue[Optional[str]]" = queue.Queue()

            def _read_one():
                try:
                    line = self._proc.stdout.readline()
                    line_q.put(line)
                except Exception:
                    line_q.put(None)

            reader = threading.Thread(target=_read_one, daemon=True)
            reader.start()
            try:
                response = line_q.get(timeout=timeout_s)
            except queue.Empty:
                # Runaway code. Kill the subprocess; state is lost
                # but it's the only safe move.
                self.shutdown()
                return {
                    "stdout": "",
                    "stderr": f"Execution exceeded {timeout_s}s — killed.",
                    "result": "",
                    "duration_s": timeout_s,
                    "timed_out": True,
                }

            if not response:
                # Subprocess died unexpectedly. Capture stderr for the
                # user to see why.
                self.shutdown()
                return {
                    "stdout": "",
                    "stderr": "REPL subprocess died unexpectedly.",
                    "result": "",
                    "duration_s": 0.0,
                    "timed_out": False,
                }

            try:
                payload = _json.loads(response.strip())
            except Exception as e:
                return {
                    "stdout": "",
                    "stderr": f"REPL response parse error: {e} (raw: {response!r})",
                    "result": "",
                    "duration_s": 0.0,
                    "timed_out": False,
                }

            payload.setdefault("timed_out", False)
            payload.setdefault("stdout", "")
            payload.setdefault("stderr", "")
            payload.setdefault("result", "")
            payload.setdefault("duration_s", 0.0)
            payload.setdefault("images", [])
            return payload

    def _send_op(self, op_payload: Dict, timeout_s: float = 10.0) -> Dict:
        """Send a control op (e.g. save_state, load_state) and parse
        the JSON reply. Internal helper for save/load."""
        import json as _json
        with self._lock:
            try:
                self._ensure_started()
            except Exception as e:
                return {"error": f"REPL startup failed: {e}"}
            try:
                assert self._proc and self._proc.stdin
                self._proc.stdin.write(_json.dumps(op_payload) + "\n")
                self._proc.stdin.flush()
            except Exception as e:
                return {"error": f"stdin write failed: {e}"}
            line_q: "queue.Queue[Optional[str]]" = queue.Queue()

            def _read_one():
                try:
                    line = self._proc.stdout.readline()
                    line_q.put(line)
                except Exception:
                    line_q.put(None)
            threading.Thread(target=_read_one, daemon=True).start()
            try:
                response = line_q.get(timeout=timeout_s)
            except queue.Empty:
                return {"error": "op response timed out"}
            if not response:
                return {"error": "op had no response"}
            try:
                return _json.loads(response.strip())
            except Exception as e:
                return {"error": f"op response parse: {e}"}

    def save_state(self, path: str, timeout_s: float = 30.0) -> Dict:
        """Pickle the user's namespace to *path*. Returns
        {"saved": int, "skipped": [str], "error"?: str}."""
        return self._send_op(
            {"_op": "save_state", "path": path}, timeout_s=timeout_s,
        )

    def load_state(self, path: str, timeout_s: float = 30.0) -> Dict:
        """Restore from *path* if it exists. Returns
        {"loaded": int, "error"?: str}."""
        return self._send_op(
            {"_op": "load_state", "path": path}, timeout_s=timeout_s,
        )

    def shutdown(self) -> None:
        with self._lock if self._lock.locked() is False else _NullCM():
            try:
                if self._proc is not None:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                    try:
                        self._proc.wait(timeout=2.0)
                    except Exception:
                        pass
                    self._proc = None
            except Exception:
                pass


class _NullCM:
    """Used so shutdown() doesn't deadlock when called from inside
    a method that already holds _lock."""
    def __enter__(self): pass
    def __exit__(self, *a): return False


# ────────────────────── per-WS session registry ──────────────────────
# The voice route opens one REPL per WebSocket session and tears it
# down when the WS closes. We DON'T have a global REPL because:
#   • Two open browser tabs would share state, which is confusing
#   • Cleaning up at WS-close is the natural boundary

_SESSION_REPLS: Dict[int, PersistentRepl] = {}
_SESSION_LOCK = threading.Lock()


def get_or_create_repl(session_key: int) -> PersistentRepl:
    """Get the REPL for *session_key* (typically id(ws)), creating if needed."""
    with _SESSION_LOCK:
        repl = _SESSION_REPLS.get(session_key)
        if repl is None:
            repl = PersistentRepl()
            _SESSION_REPLS[session_key] = repl
        return repl


def shutdown_repl(session_key: int) -> None:
    """Tear down the REPL for *session_key*. Called when its WS closes."""
    with _SESSION_LOCK:
        repl = _SESSION_REPLS.pop(session_key, None)
    if repl is not None:
        try:
            repl.shutdown()
        except Exception:
            pass


def _state_path_for_user(user_id: str) -> Path:
    """Where the per-user pickled REPL namespace lives. Anchored to repo
    root so it survives whatever cwd the server was launched from."""
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root / "assistant" / "memory" / "users" / user_id / "repl_namespace.pkl"
    )


def restore_session_state(session_key: int, user_id: str) -> Dict:
    """Restore the user's pickled namespace into this session's REPL.

    Called once per WS connect. Idempotent — if no pickle exists, the
    REPL just starts empty. The REPL is started here as a side-effect
    of _ensure_started(), so subsequent code calls don't pay startup."""
    path = _state_path_for_user(user_id)
    if not path.exists():
        return {"loaded": 0, "skipped": "no_state_file"}
    repl = get_or_create_repl(session_key)
    return repl.load_state(str(path))


def persist_session_state(session_key: int, user_id: str) -> Dict:
    """Pickle the current namespace to disk. Called before WS shutdown
    so variables survive the next reconnect."""
    with _SESSION_LOCK:
        repl = _SESSION_REPLS.get(session_key)
    if repl is None:
        return {"saved": 0, "skipped": "no_active_repl"}
    path = _state_path_for_user(user_id)
    return repl.save_state(str(path))


# ────────────────────── tool entrypoint ──────────────────────────────

def execute_python(args: dict, *, session_key: Optional[int] = None) -> str:
    """Tool handler shape: dict in, str out.

    Args:
        code (required): Python code to execute. Single expression
                          returns its repr; multiple statements run via exec.
        timeout_s (optional): hard timeout in seconds, default 8, max 60.

    Returns a formatted result string suitable for the LLM. The
    parallel SSE event (emitted by the dispatcher) carries the full
    structured payload for the dashboard /exec view.
    """
    code = (args.get("code") or "").strip()
    if not code:
        return "No code provided."
    timeout_s = float(args.get("timeout_s", DEFAULT_TIMEOUT_S) or DEFAULT_TIMEOUT_S)

    if session_key is None:
        # Fall back to a process-global REPL — fine for single-user
        # local deploys; a shared backend should always pass session_key.
        session_key = 0

    repl = get_or_create_repl(session_key)
    payload = repl.execute(code, timeout_s=timeout_s)

    # Format for LLM consumption — keep it tight.
    parts: list[str] = []
    if payload.get("result"):
        parts.append(f"=> {payload['result']}")
    if payload.get("stdout"):
        out = payload["stdout"].rstrip()
        if out:
            parts.append(f"stdout:\n{out}")
    if payload.get("stderr"):
        err = payload["stderr"].rstrip()
        if err:
            parts.append(f"stderr:\n{err}")
    if payload.get("timed_out"):
        parts.append("(timed out — subprocess restarted, variables lost)")
    if not parts:
        parts.append("(no output)")
    parts.append(f"[{payload.get('duration_s', 0):.2f}s]")

    text = "\n".join(parts)
    if len(text) > LLM_OUTPUT_CAP:
        text = text[:LLM_OUTPUT_CAP] + f"\n... (truncated; full output in /exec view)"
    return text


# Expose the structured payload for the dispatcher so it can emit a
# rich SSE event without re-running the code.
def execute_python_full(args: dict, *, session_key: Optional[int] = None) -> Dict:
    """Like execute_python but returns the full structured dict."""
    code = (args.get("code") or "").strip()
    if not code:
        return {"stdout": "", "stderr": "no code provided", "result": "",
                "duration_s": 0.0, "timed_out": False, "code": code}
    timeout_s = float(args.get("timeout_s", DEFAULT_TIMEOUT_S) or DEFAULT_TIMEOUT_S)
    if session_key is None:
        session_key = 0
    repl = get_or_create_repl(session_key)
    payload = repl.execute(code, timeout_s=timeout_s)
    payload["code"] = code
    return payload
