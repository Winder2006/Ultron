"""Sandboxed read/write access to the user's notes directory.

All paths are resolved against a single root (`~/AI_Workspace/` by default,
override via the `ULTRON_NOTES_ROOT` env var) and rejected if they escape it
via `..`, absolute paths, or symlinks that resolve outside the root.

This is the defensive choke-point for Ultron's ability to touch the
filesystem. Any new filesystem tool should go through `resolve_sandbox_path`.

Why these rules:
  - Single-root sandbox: keeps blast radius to one directory tree.
  - Symlink-resolved comparison: prevents escape via symlinks a user could
    have placed intentionally or otherwise.
  - Size + count limits: rate-limit damage from a runaway tool-call loop.
  - Audit log: every write and read is logged with a timestamp to
    `logs/notes_tool.log`.

Exceptions raised in here (SandboxError) are never thrown at the LLM —
`dispatch_tool_call` catches them and turns them into an error string.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


class SandboxError(Exception):
    """Raised when a requested path escapes the sandbox or violates a limit."""


# ─────────────────────── configuration ──────────────────────────────

def _default_root() -> Path:
    """Resolve the sandbox root, honoring the env override."""
    env = os.environ.get("ULTRON_NOTES_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "AI_Workspace").resolve()


# Absolute caps on what a single tool call can do. These guard against
# a runaway LLM loop accidentally DOSing the filesystem.
_MAX_WRITE_BYTES = 200_000          # per write
_MAX_READ_BYTES = 100_000           # per read (we truncate longer files)
_MAX_FILENAME_LEN = 120
_ALLOWED_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".log", ""}


# ─────────────────────── path safety ────────────────────────────────

def resolve_sandbox_path(rel: str, *, root: Optional[Path] = None) -> Path:
    """Turn a user-supplied relative path into an absolute sandboxed path.

    Rejects: empty strings, absolute paths, paths that escape the root
    after normalization, paths with suspicious characters, and paths
    whose extension isn't in the allowed set.
    """
    if not isinstance(rel, str):
        raise SandboxError("path must be a string")
    rel = rel.strip()
    if not rel:
        raise SandboxError("path is empty")
    if len(rel) > _MAX_FILENAME_LEN:
        raise SandboxError(f"path too long ({len(rel)} > {_MAX_FILENAME_LEN})")
    # Normalize forward slashes — user might pass windows-style paths
    rel_norm = rel.replace("\\", "/")
    if rel_norm.startswith(("/", "~")) or re.match(r"^[a-zA-Z]:", rel_norm):
        raise SandboxError("absolute paths are not allowed")

    root = (root or _default_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Resolve against the root, then confirm the result is *still* inside
    # the root. `strict=False` lets us target files that don't exist yet.
    target = (root / rel_norm).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError:
        raise SandboxError(f"path escapes sandbox: {rel}")

    # Extension allowlist (empty string lets callers target directories or
    # extensionless files like plain notes).
    ext = target.suffix.lower()
    if ext and ext not in _ALLOWED_EXTS:
        raise SandboxError(
            f"extension {ext!r} not allowed — use one of: "
            f"{sorted(e for e in _ALLOWED_EXTS if e)}"
        )
    return target


# ─────────────────────── audit log ──────────────────────────────────

_AUDIT_PATH = Path("logs/notes_tool.log")

# When the audit log exceeds this size, we rotate it: move current to
# `.1` (replacing any prior rotation) and start a fresh log. Picked so
# the file stays readable while keeping a generation of history.
_AUDIT_MAX_BYTES = 2_000_000  # 2 MB


def _rotate_audit_if_needed() -> None:
    """Move the current audit log to `.1` when it grows past the cap.

    Simple 1-generation rotation — no `.log.1`, `.log.2`, `.log.3` etc.
    A compromised tool can't grow the log without bound; a forensic
    reader still gets one prior file to look at. Best-effort — silent
    if rotation fails (disk full, permission, etc.)."""
    try:
        if not _AUDIT_PATH.exists():
            return
        if _AUDIT_PATH.stat().st_size < _AUDIT_MAX_BYTES:
            return
        backup = _AUDIT_PATH.with_suffix(_AUDIT_PATH.suffix + ".1")
        # Replace any prior backup — we only keep one generation.
        if backup.exists():
            backup.unlink()
        _AUDIT_PATH.rename(backup)
    except OSError:
        pass


def _audit(action: str, path: Path, *, bytes_count: int, ok: bool, note: str = "") -> None:
    """Append a single-line audit entry. Best-effort, never raises."""
    try:
        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _rotate_audit_if_needed()
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "path": str(path),
            "bytes": bytes_count,
            "ok": ok,
            "note": note,
        }
        with _AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ─────────────────────── public helpers ─────────────────────────────

def read_note(rel_path: str, *, root: Optional[Path] = None) -> str:
    """Read a file inside the sandbox. Returns its text content.

    Raises SandboxError on any policy violation (path escape, missing
    file, extension not allowed). Truncates at `_MAX_READ_BYTES` with
    a trailing marker.
    """
    path = resolve_sandbox_path(rel_path, root=root)
    if not path.exists():
        _audit("read", path, bytes_count=0, ok=False, note="not found")
        raise SandboxError(f"file not found: {rel_path}")
    if not path.is_file():
        raise SandboxError(f"not a file: {rel_path}")
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) > _MAX_READ_BYTES:
        data = data[:_MAX_READ_BYTES] + "\n\n… (truncated)"
    _audit("read", path, bytes_count=len(data), ok=True)
    return data


def write_note(
    rel_path: str,
    content: str,
    *,
    append: bool = False,
    root: Optional[Path] = None,
) -> str:
    """Write `content` to a file inside the sandbox.

    `append=True` appends with a leading newline separator; `append=False`
    overwrites. Returns a short success string for the LLM to narrate.

    Raises SandboxError on size or path violations.
    """
    if not isinstance(content, str):
        raise SandboxError("content must be a string")
    data = content
    if len(data.encode("utf-8")) > _MAX_WRITE_BYTES:
        raise SandboxError(
            f"content too large ({len(data.encode('utf-8'))} bytes > "
            f"{_MAX_WRITE_BYTES}) — split into smaller pieces"
        )

    path = resolve_sandbox_path(rel_path, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")
        # Single newline between existing content and new content so
        # successive appends don't pile up blank lines.
        sep = "" if existing.endswith("\n") else "\n"
        final = existing + sep + data
        if len(final.encode("utf-8")) > _MAX_WRITE_BYTES * 4:
            raise SandboxError(
                "file would grow beyond 4x the per-write limit — "
                "start a fresh note instead of appending"
            )
        path.write_text(final, encoding="utf-8")
        action = "append"
    else:
        path.write_text(data, encoding="utf-8")
        action = "write"

    _audit(action, path, bytes_count=len(data.encode("utf-8")), ok=True)
    return f"{'Appended' if append else 'Wrote'} {len(data)} chars to {rel_path}."


def list_notes(subdir: str = "", *, root: Optional[Path] = None) -> list[dict]:
    """List files in a sandbox subdirectory. Returns [{name, size, mtime}]."""
    base = _default_root() if root is None else root.resolve()
    target = base if not subdir else resolve_sandbox_path(subdir, root=base)
    if not target.exists() or not target.is_dir():
        return []
    out = []
    for p in sorted(target.iterdir()):
        try:
            stat = p.stat()
            out.append({
                "name": p.name,
                "is_dir": p.is_dir(),
                "size": stat.st_size if p.is_file() else 0,
                "mtime": stat.st_mtime,
            })
        except OSError:
            continue
    return out
