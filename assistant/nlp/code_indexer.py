"""Walk the project tree and emit semantic chunks of source code.

Produces records shaped identically to the notes RAG records
(`{id, text, path, meta}`) so they can be embedded by the same pipeline
and stored in a dedicated code FAISS index.

Python files are split with the `ast` module into per-function / per-class
chunks. TypeScript / TSX files are split with a forgiving regex — good
enough to locate exported functions, React components, and hooks without
pulling in a full TS parser.

YAML / JSON config files are indexed whole (small, high-signal).
Markdown (README, SECURITY, *.md) is indexed as a fallback single chunk.

Each record's `meta` carries:
    kind:    "code" | "config" | "docs"
    lang:    "python" | "typescript" | "yaml" | "json" | "markdown"
    symbol:  dotted path like "mother.tts.engine.DeepgramTTSEngine.speak"
             (absent for whole-file docs/configs)
    lines:   "start-end" in the source file

The `text` field is what actually gets embedded. It concatenates:
    - a header line describing the symbol
    - the docstring (if any)
    - the source excerpt (truncated to a budget)

Headers matter more than you'd think — asking Ultron "how does TTS
streaming work" matches the header "mother.tts.engine.DeepgramTTSEngine
— streams PCM from Deepgram" far better than matching raw Python tokens.
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Directories we never descend into — they either aren't our code or are
# too noisy to index (generated output, caches, deps).
_SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "node_modules", "dist", "build",
    ".next", ".turbo", "out", "coverage", ".vscode", ".idea",
    "logs", "ultron_clips", "tts", "bin", ".cache", ".wdm",
    "assistant/memory",  # the FAISS index itself lives here — don't self-ingest
}


# File extensions we index, grouped by how we treat them.
_PY_EXT = {".py"}
_TS_EXT = {".ts", ".tsx", ".js", ".jsx"}
_CONFIG_EXT = {".yaml", ".yml", ".json", ".toml"}
_DOC_EXT = {".md"}


# Default roots. Everything lives under the project root, but we pass
# these explicitly so we never accidentally walk the whole filesystem.
DEFAULT_ROOTS = (
    "mother",
    "dashboard/src",
    "dashboard/public",  # AudioWorklet lives here as static .js
    "scripts",
    "configs",
)
DEFAULT_TOP_LEVEL_FILES = ("README.md", "SECURITY.md")


# Max characters of source excerpt we keep in any chunk. Chunks are what
# gets fed to a 384-dim MiniLM — very long chunks dilute signal. 1200
# chars is ~ 300 tokens, comfortably inside the model's 128-token window
# after the embedder truncates (truncation is fine — the header + docstring
# up front are what drives retrieval).
_MAX_EXCERPT_CHARS = 1200


# ───────────────────────── path helpers ─────────────────────────────

def _should_skip_dir(abs_path: Path, project_root: Path) -> bool:
    """True if this directory matches a skip name or skip path fragment."""
    name = abs_path.name
    if name in _SKIP_DIR_NAMES:
        return True
    try:
        rel = abs_path.relative_to(project_root).as_posix()
    except ValueError:
        return True  # outside the project — don't touch
    for skip in _SKIP_DIR_NAMES:
        if "/" in skip and skip in rel:
            return True
    return False


def _iter_source_files(roots: Iterable[str], project_root: Path) -> List[Path]:
    """Walk `roots` under `project_root`; return files we know how to index."""
    out: List[Path] = []
    for root in roots:
        base = (project_root / root).resolve()
        if not base.exists():
            continue
        if base.is_file():
            out.append(base)
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            # Prune skipped subdirs in-place so os.walk doesn't descend
            dp = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if not _should_skip_dir(dp / d, project_root)
            ]
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext in _PY_EXT | _TS_EXT | _CONFIG_EXT | _DOC_EXT:
                    out.append(dp / fn)
    return out


def _rel(path: Path, project_root: Path) -> str:
    """Relative POSIX path (forward slashes) for stable meta/dotted names."""
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _module_dotted(rel_path: str) -> str:
    """'mother/tts/engine.py' -> 'mother.tts.engine'."""
    no_ext = rel_path.rsplit(".", 1)[0]
    return no_ext.replace("/", ".")


# ───────────────────────── Python extraction ────────────────────────

def _py_signature(node: ast.AST) -> str:
    """Render a readable signature for a function or class def."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = []
        a = node.args
        for arg in a.args:
            args.append(arg.arg)
        if a.vararg:
            args.append("*" + a.vararg.arg)
        for arg in a.kwonlyargs:
            args.append(arg.arg)
        if a.kwarg:
            args.append("**" + a.kwarg.arg)
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        return f"{prefix}{node.name}({', '.join(args)})"
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) if hasattr(ast, "unparse") else "" for b in node.bases]
        bases = [b for b in bases if b]
        suffix = f"({', '.join(bases)})" if bases else ""
        return f"class {node.name}{suffix}"
    return ""


def _extract_excerpt(source: str, node: ast.AST) -> str:
    """Pull the source text for `node` from `source`, bounded by _MAX_EXCERPT_CHARS."""
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    lines = source.splitlines()
    excerpt = "\n".join(lines[start:end])
    if len(excerpt) > _MAX_EXCERPT_CHARS:
        excerpt = excerpt[:_MAX_EXCERPT_CHARS] + "\n# ... (truncated)"
    return excerpt


def _py_records(path: Path, project_root: Path) -> List[Dict]:
    """Turn a .py file into one record per top-level function/class
    (plus methods nested inside classes), plus a module-level summary."""
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # skip syntactically invalid files silently

    rel = _rel(path, project_root)
    module = _module_dotted(rel)
    records: List[Dict] = []

    # Module-level docstring as its own chunk so "what is mother.tts.engine"
    # resolves cleanly even before hitting a specific symbol.
    mod_doc = ast.get_docstring(tree) or ""
    if mod_doc.strip():
        header = f"Module {module} — module-level overview"
        text = f"{header}\n\n{mod_doc.strip()}"
        records.append({
            "text": text,
            "path": rel,
            "meta": {
                "kind": "code",
                "lang": "python",
                "symbol": module,
                "lines": "module",
            },
        })

    def _walk(nodes: List[ast.AST], parent_dotted: str) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                dotted = f"{parent_dotted}.{node.name}"
                sig = _py_signature(node)
                doc = ast.get_docstring(node) or ""
                excerpt = _extract_excerpt(source, node)
                # Header first — drives retrieval more than the raw body.
                header = f"{dotted} — {sig}"
                parts = [header]
                if doc.strip():
                    parts.append(doc.strip())
                parts.append(excerpt)
                text = "\n\n".join(parts)
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                records.append({
                    "text": text,
                    "path": rel,
                    "meta": {
                        "kind": "code",
                        "lang": "python",
                        "symbol": dotted,
                        "signature": sig,
                        "lines": f"{start}-{end}",
                    },
                })
                # Descend into classes to pick up methods as their own chunks.
                if isinstance(node, ast.ClassDef):
                    _walk(node.body, dotted)

    _walk(tree.body, module)
    return records


# ───────────────────────── TS/TSX extraction ────────────────────────

# Matches common export/declaration forms — deliberately forgiving. We
# don't try to parse TS properly; we just want anchor symbols that a
# user might ask about by name.
_TS_SYMBOL_RE = re.compile(
    r"""^\s*
        (?:export\s+(?:default\s+)?)?
        (?:
            (?:async\s+)?function\s+(?P<fn>[A-Za-z_$][\w$]*)
          | const\s+(?P<const>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:\(|async|<|function)
          | class\s+(?P<cls>[A-Za-z_$][\w$]*)
          | interface\s+(?P<iface>[A-Za-z_$][\w$]*)
          | type\s+(?P<type>[A-Za-z_$][\w$]*)\s*=
        )
    """,
    re.MULTILINE | re.VERBOSE,
)


def _ts_records(path: Path, project_root: Path) -> List[Dict]:
    """Extract exported symbols from a TS/TSX file with regex.

    We give each symbol a chunk containing the declaration line plus a
    bounded window of following source. Good enough for RAG; no parser.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    rel = _rel(path, project_root)
    module = _module_dotted(rel)
    records: List[Dict] = []
    lines = source.splitlines()

    # Always emit a file-level overview chunk. For short files this is
    # the whole file; for long files it's the first 60 lines (usually
    # imports + top-level exports) which is where intent lives.
    head = "\n".join(lines[:60])
    if head.strip():
        records.append({
            "text": f"File {rel} — top of file\n\n{head}",
            "path": rel,
            "meta": {
                "kind": "code",
                "lang": _ts_lang(path),
                "symbol": module,
                "lines": "1-60",
            },
        })

    for m in _TS_SYMBOL_RE.finditer(source):
        name = m.group("fn") or m.group("const") or m.group("cls") or m.group("iface") or m.group("type")
        if not name:
            continue
        start_line = source.count("\n", 0, m.start()) + 1
        # Window of ~40 lines after the symbol start — most components
        # and functions fit comfortably in that.
        window_end = min(len(lines), start_line + 40)
        excerpt = "\n".join(lines[start_line - 1:window_end])
        if len(excerpt) > _MAX_EXCERPT_CHARS:
            excerpt = excerpt[:_MAX_EXCERPT_CHARS] + "\n// ... (truncated)"
        dotted = f"{module}.{name}"
        header = f"{dotted} — {rel}:{start_line}"
        text = f"{header}\n\n{excerpt}"
        records.append({
            "text": text,
            "path": rel,
            "meta": {
                "kind": "code",
                "lang": _ts_lang(path),
                "symbol": dotted,
                "lines": f"{start_line}-{window_end}",
            },
        })
    return records


def _ts_lang(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".ts", ".tsx"}:
        return "typescript"
    return "javascript"


# ───────────────────────── config + docs ────────────────────────────

def _config_record(path: Path, project_root: Path) -> Optional[Dict]:
    """Whole-file chunk for small config files."""
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    if not source.strip():
        return None
    rel = _rel(path, project_root)
    # Keep config files bounded — for very large ones, keep head only.
    if len(source) > _MAX_EXCERPT_CHARS * 2:
        source = source[:_MAX_EXCERPT_CHARS * 2] + "\n# ... (truncated)"
    lang = "yaml" if path.suffix.lower() in {".yaml", ".yml"} else (
        "json" if path.suffix.lower() == ".json" else "toml"
    )
    header = f"Config {rel} — {lang} configuration file"
    return {
        "text": f"{header}\n\n{source}",
        "path": rel,
        "meta": {
            "kind": "config",
            "lang": lang,
            "symbol": rel,
        },
    }


def _doc_record(path: Path, project_root: Path) -> Optional[Dict]:
    """Whole-file chunk for markdown docs (README etc.)."""
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    if not source.strip():
        return None
    rel = _rel(path, project_root)
    if len(source) > _MAX_EXCERPT_CHARS * 3:
        source = source[:_MAX_EXCERPT_CHARS * 3] + "\n... (truncated)"
    header = f"Docs {rel}"
    return {
        "text": f"{header}\n\n{source}",
        "path": rel,
        "meta": {
            "kind": "docs",
            "lang": "markdown",
            "symbol": rel,
        },
    }


# ───────────────────────── public API ───────────────────────────────

def collect_code_records(
    project_root: str | os.PathLike[str],
    roots: Iterable[str] = DEFAULT_ROOTS,
    top_level_files: Iterable[str] = DEFAULT_TOP_LEVEL_FILES,
) -> List[Dict]:
    """Walk the project and return one list of RAG-shaped records.

    Records are not yet embedded — this is the pure extraction step so
    it's cheap to call in tests or CLI scripts.
    """
    root = Path(project_root).resolve()
    files = _iter_source_files(roots, root)

    # Top-level docs live directly in the repo root, not under `roots`.
    for fname in top_level_files:
        p = (root / fname).resolve()
        if p.exists() and p.is_file():
            files.append(p)

    records: List[Dict] = []
    for path in files:
        ext = path.suffix.lower()
        if ext in _PY_EXT:
            records.extend(_py_records(path, root))
        elif ext in _TS_EXT:
            records.extend(_ts_records(path, root))
        elif ext in _CONFIG_EXT:
            rec = _config_record(path, root)
            if rec:
                records.append(rec)
        elif ext in _DOC_EXT:
            rec = _doc_record(path, root)
            if rec:
                records.append(rec)

    # Assign stable ids (position in list). FAISS doesn't care but the
    # records file is human-readable and stable ids make diffs nicer.
    for i, rec in enumerate(records):
        rec["id"] = i
    return records


def summarize(records: List[Dict]) -> Dict[str, int]:
    """Quick stats — useful for CLI output."""
    counts: Dict[str, int] = {}
    for r in records:
        kind = r.get("meta", {}).get("kind", "unknown")
        lang = r.get("meta", {}).get("lang", "unknown")
        key = f"{kind}/{lang}"
        counts[key] = counts.get(key, 0) + 1
    counts["_total"] = len(records)
    return counts


if __name__ == "__main__":
    # Run as a module for a quick dry-run — no indexing, just extraction.
    # python -m assistant.nlp.code_indexer
    here = Path(__file__).resolve().parents[2]
    recs = collect_code_records(here)
    stats = summarize(recs)
    print(json.dumps(stats, indent=2))
