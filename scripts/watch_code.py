"""Watch the project tree and re-index the codebase RAG on every save.

Runs alongside `uvicorn assistant.app:app --port 8123`. When a Python,
TypeScript, JS, YAML, JSON, or Markdown file under the indexed roots
changes, we debounce and then rebuild the code index.

Full rebuild takes ~5s for the whole project, which is fast enough that
per-file incremental updates aren't worth the complexity. We just
re-run `collect_code_records` + `build_code_index`, then POST to
`/reindex-code` so the live RAG service picks up the new index without
a server restart.

Usage:
    python scripts/watch_code.py
    python scripts/watch_code.py --no-hot-reload   # skip the HTTP ping
    python scripts/watch_code.py --debounce 2.0    # wait 2s of quiet
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Set

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from assistant.nlp.code_indexer import (  # noqa: E402
    DEFAULT_ROOTS,
    DEFAULT_TOP_LEVEL_FILES,
    collect_code_records,
    summarize,
)
from assistant.nlp.rag import build_code_index  # noqa: E402


# File extensions we care about. Anything else gets ignored on save.
_WATCHED_EXT = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".yaml", ".yml", ".json", ".toml", ".md",
}

# Directory fragments we never rebuild for. This avoids thrashing when
# `pytest` writes caches or `uvicorn`'s auto-reload writes .pyc files.
_IGNORE_DIR_FRAGMENTS = {
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "node_modules", "dist", "build",
    "logs", "assistant/memory",
    # Editor/temp noise on Windows:
    "~",  # VS Code recovery files often contain this
}


def _relevant(path: str) -> bool:
    """True if a save at this path should trigger a rebuild."""
    p = Path(path).as_posix().lower()
    if any(frag in p for frag in _IGNORE_DIR_FRAGMENTS):
        return False
    return Path(path).suffix.lower() in _WATCHED_EXT


class _DebouncedRebuild:
    """Coalesce a burst of save events into a single rebuild."""

    def __init__(self, index_path: str, meta_path: str, debounce_s: float,
                 hot_reload_url: str | None) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.debounce_s = debounce_s
        self.hot_reload_url = hot_reload_url
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._pending: Set[str] = set()

    def trigger(self, path: str) -> None:
        """Called from watchdog's thread — schedule a rebuild."""
        with self._lock:
            self._pending.add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_s, self._rebuild)
            self._timer.daemon = True
            self._timer.start()

    def _rebuild(self) -> None:
        with self._lock:
            files = sorted(self._pending)
            self._pending.clear()
            self._timer = None

        if not files:
            return

        print(f"\n[watch] {len(files)} file(s) changed — rebuilding code index")
        for f in files[:5]:
            print(f"  • {Path(f).relative_to(_PROJECT_ROOT) if Path(f).is_relative_to(_PROJECT_ROOT) else f}")
        if len(files) > 5:
            print(f"  • …and {len(files) - 5} more")

        t0 = time.perf_counter()
        try:
            records = collect_code_records(_PROJECT_ROOT)
            n = build_code_index(records, self.index_path, self.meta_path)
            dt = time.perf_counter() - t0
            print(f"[watch] indexed {n} chunks in {dt:.1f}s")
            stats = summarize(records)
            print(
                f"[watch]   code/py={stats.get('code/python', 0)}  "
                f"code/ts={stats.get('code/typescript', 0)}  "
                f"code/js={stats.get('code/javascript', 0)}  "
                f"config={stats.get('config/yaml', 0) + stats.get('config/json', 0) + stats.get('config/toml', 0)}  "
                f"docs={stats.get('docs/markdown', 0)}"
            )
        except Exception as e:
            print(f"[watch] ERROR during rebuild: {e}")
            return

        # Ping the RAG service so it warms the new index. The service's
        # _maybe_load_cache already mtime-checks, so a reindex + warm
        # call is enough; no restart required.
        if self.hot_reload_url:
            try:
                import httpx
                r = httpx.get(self.hot_reload_url, timeout=2.0)
                if r.status_code == 200:
                    print(f"[watch] hot-reloaded: {self.hot_reload_url}")
                else:
                    print(f"[watch] hot-reload HTTP {r.status_code}")
            except Exception as e:
                print(f"[watch] hot-reload skipped: {e}")


def _build_watched_paths(roots: Iterable[str]) -> list[Path]:
    """Resolve the directory roots we want watchdog to observe."""
    out: list[Path] = []
    for r in roots:
        p = (_PROJECT_ROOT / r).resolve()
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-reindex the codebase RAG on save.")
    ap.add_argument("--debounce", type=float, default=1.0,
                    help="Seconds of quiet before rebuilding (default 1.0).")
    ap.add_argument("--out", default="assistant/memory",
                    help="Directory containing code.index + code_meta.json.")
    ap.add_argument("--no-hot-reload", action="store_true",
                    help="Skip pinging the running RAG service after rebuild.")
    ap.add_argument("--rag-base", default="http://127.0.0.1:8123",
                    help="Base URL for the RAG service's warm endpoint.")
    args = ap.parse_args()

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print(
            "[watch] watchdog is not installed. "
            "Install it with: pip install 'watchdog>=4.0.0'",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    index_path = str(out / "code.index")
    meta_path = str(out / "code_meta.json")
    hot_reload = None if args.no_hot_reload else f"{args.rag_base}/warmup"

    rebuilder = _DebouncedRebuild(index_path, meta_path, args.debounce, hot_reload)

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):  # type: ignore[override]
            if not event.is_directory and _relevant(event.src_path):
                rebuilder.trigger(event.src_path)

        def on_created(self, event):  # type: ignore[override]
            if not event.is_directory and _relevant(event.src_path):
                rebuilder.trigger(event.src_path)

        def on_moved(self, event):  # type: ignore[override]
            dest = getattr(event, "dest_path", None)
            if dest and _relevant(dest):
                rebuilder.trigger(dest)

    observer = Observer()
    handler = _Handler()
    roots = _build_watched_paths(DEFAULT_ROOTS)
    for root in roots:
        observer.schedule(handler, str(root), recursive=True)
        print(f"[watch] watching {root.relative_to(_PROJECT_ROOT)}")
    # Top-level docs like README.md sit in the repo root; watch the root
    # non-recursively so editing them triggers a rebuild.
    observer.schedule(handler, str(_PROJECT_ROOT), recursive=False)
    print(f"[watch] watching {_PROJECT_ROOT.name}/ (non-recursive)")

    observer.start()
    print(f"[watch] debounce={args.debounce}s  hot_reload={hot_reload or 'off'}")
    print("[watch] ctrl-c to exit")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[watch] stopping")
    finally:
        observer.stop()
        observer.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
