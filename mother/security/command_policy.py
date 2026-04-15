from __future__ import annotations

import collections
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, Any

_SANDBOX_ROOT = Path(os.path.expanduser("~/AI_Workspace")).resolve()
_AUDIT_LOG = Path("logs/commands.log").resolve()

_LOCK = threading.Lock()
_COMMANDS: Dict[str, Tuple[Callable[..., Any], bool]] = {}
_WINDOW: collections.deque = collections.deque()  # timestamps for rate limiting
_MAX_PER_MINUTE = 10
_FAILURES = 0
_FAILURE_COOL_OFF_S = 5.0
_FAILURE_COOLOFF_UNTIL: float = 0.0  # monotonic time after which cooloff expires


def _ensure_dirs() -> None:
    _SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)


def is_path_allowed(path: str | Path) -> bool:
    p = Path(path).expanduser().resolve()
    try:
        return _SANDBOX_ROOT in p.parents or p == _SANDBOX_ROOT
    except Exception:
        return False


def _audit(user: str, name: str, args: tuple[Any, ...], kwargs: dict[str, Any], ok: bool, error: str | None) -> None:
    _ensure_dirs()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{ts}\tuser={user}\tcmd={name}\tok={str(ok).lower()}\targs={repr(args)}\tkwargs={repr(kwargs)}\terror={error or ''}\n"
    with _LOCK:
        with _AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(line)


def register_command(name: str, func: Callable[..., Any], *, risky: bool = False) -> None:
    with _LOCK:
        _COMMANDS[name] = (func, risky)


def _rate_limit_ok() -> bool:
    now = time.monotonic()
    with _LOCK:
        while _WINDOW and now - _WINDOW[0] > 60.0:
            _WINDOW.popleft()
        if len(_WINDOW) >= _MAX_PER_MINUTE:
            return False
        if _FAILURES >= 3 and now < _FAILURE_COOLOFF_UNTIL:
            return False
        _WINDOW.append(now)
        return True


def _increment_failures() -> None:
    global _FAILURES, _FAILURE_COOLOFF_UNTIL
    with _LOCK:
        _FAILURES += 1
        if _FAILURES >= 3:
            _FAILURE_COOLOFF_UNTIL = time.monotonic() + _FAILURE_COOL_OFF_S


def _reset_failures() -> None:
    global _FAILURES, _FAILURE_COOLOFF_UNTIL
    with _LOCK:
        _FAILURES = 0
        _FAILURE_COOLOFF_UNTIL = 0.0


def run_command(name: str, *args: Any, user_confirm: bool | None = None, user: str = "local", **kwargs: Any) -> dict:
    _ensure_dirs()
    func_risky = _COMMANDS.get(name)
    if func_risky is None:
        _audit(user, name, args, kwargs, False, "not registered")
        _increment_failures()
        return {"ok": False, "error": "command not allowed"}
    func, risky = func_risky
    if risky and not user_confirm:
        _audit(user, name, args, kwargs, False, "confirmation required")
        _increment_failures()
        return {"ok": False, "error": "confirmation required"}
    if not _rate_limit_ok():
        _audit(user, name, args, kwargs, False, "rate limited")
        return {"ok": False, "error": "rate limited"}
    # Enforce sandbox on any path-like args
    for k, v in list(kwargs.items()):
        if isinstance(v, (str, Path)) and ("path" in k or k in {"file", "dest", "target"}):
            if not is_path_allowed(v):
                _audit(user, name, args, kwargs, False, "path outside sandbox")
                _increment_failures()
                return {"ok": False, "error": "path outside sandbox"}
    try:
        result = func(*args, **kwargs)
        _audit(user, name, args, kwargs, True, None)
        _reset_failures()
        return {"ok": True, "data": result}
    except Exception as e:
        _audit(user, name, args, kwargs, False, str(e))
        _increment_failures()
        return {"ok": False, "error": str(e)}
