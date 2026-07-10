from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import faiss  # type: ignore
import numpy as np

from .embeddings import embed_texts, load_encoder
from .chunker import chunk_text
from .frontmatter_util import parse_frontmatter


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def _iter_note_files(notes_dir: str) -> List[Path]:
    paths: List[Path] = []
    for root, _, files in os.walk(notes_dir):
        for f in files:
            if f.lower().endswith((".md", ".txt")):
                paths.append(Path(root) / f)
    return paths


def build_index(notes_dir: str, index_path: str, meta_path: str) -> None:
    notes = _iter_note_files(notes_dir)
    records: List[Dict] = []
    texts: List[str] = []
    for p in notes:
        content = p.read_text(encoding="utf-8", errors="ignore")
        meta, body = parse_frontmatter(content)
        # Fold selected meta/front-matter into indexable text to aid QA-style queries
        meta_bits: List[str] = []
        try:
            for k, v in (meta or {}).items():
                if isinstance(v, (str, int, float)):
                    meta_bits.append(f"{k}: {v}")
                elif isinstance(v, list):
                    meta_bits.append(f"{k}: {', '.join(str(x) for x in v)}")
                elif isinstance(v, dict):
                    # flatten one level
                    for sk, sv in v.items():
                        if isinstance(sv, (str, int, float)):
                            meta_bits.append(f"{k}.{sk}: {sv}")
                        elif isinstance(sv, list):
                            meta_bits.append(f"{k}.{sk}: {', '.join(str(x) for x in sv)}")
        except Exception:
            pass
        # Add filename as possible alias (e.g., people/oliver_meredith)
        alias = p.stem.replace("_", " ")
        if alias:
            meta_bits.append(f"name: {alias}")
        meta_text = "; ".join(meta_bits)
        for chunk in chunk_text(body):
            combined = (meta_text + "\n" + chunk).strip() if meta_text else chunk
            rec = {
                "id": len(records),
                "text": combined,
                "path": str(p),
                "meta": meta,
            }
            records.append(rec)
            texts.append(combined)
    if not records:
        # create empty index
        dim = 384
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, index_path)
        Path(meta_path).write_text("[]", encoding="utf-8")
        return

    X = embed_texts(texts)
    X = _l2_normalize(X)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    Path(meta_path).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def build_code_index(records: List[Dict], index_path: str, meta_path: str) -> int:
    """Embed pre-chunked code records and write a dedicated FAISS index.

    Unlike `build_index`, this takes records directly instead of walking
    a notes directory — the `code_indexer` module does the walking and
    AST-aware chunking. Kept separate from the notes index so rebuilding
    code doesn't disturb notes (and vice versa).

    Returns the number of records indexed.
    """
    if not records:
        # Still write an empty index so the app can start up.
        dim = 384
        index = faiss.IndexFlatIP(dim)
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, index_path)
        Path(meta_path).write_text("[]", encoding="utf-8")
        return 0

    texts = [r["text"] for r in records]
    X = embed_texts(texts)
    X = _l2_normalize(X)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    Path(meta_path).write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(records)


def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, topk: int, lam: float = 0.5) -> List[int]:
    selected: List[int] = []
    candidates = list(range(cand_vecs.shape[0]))
    sim_q = (cand_vecs @ query_vec.reshape(-1, 1)).ravel()
    while candidates and len(selected) < topk:
        best = None
        best_score = -1e9
        for i in candidates:
            div = 0.0
            if selected:
                div = max((cand_vecs[i] @ cand_vecs[j] for j in selected), default=0.0)
            score = lam * sim_q[i] - (1 - lam) * div
            if score > best_score:
                best_score = score
                best = i
        selected.append(best)  # type: ignore[arg-type]
        candidates.remove(best)  # type: ignore[arg-type]
    return selected


# Cache keyed by (index_path, meta_path) so the notes index and the
# code index can both stay loaded simultaneously. The previous single-
# slot dict thrashed on every alternation between /search and
# /code-search, reloading FAISS from disk (~50ms) on every call.
_CACHE: Dict[Tuple[str, str], Dict] = {}


def _maybe_load_cache(index_path: str, meta_path: str) -> tuple[faiss.Index | None, List[Dict] | None]:  # type: ignore[name-defined]
    ip = Path(index_path); mp = Path(meta_path)
    if not ip.exists() or not mp.exists():
        return None, None
    idx_m = ip.stat().st_mtime
    mt_m = mp.stat().st_mtime
    key = (index_path, meta_path)
    entry = _CACHE.get(key)
    if entry is None or entry["idx_mtime"] != idx_m or entry["meta_mtime"] != mt_m:
        idx = faiss.read_index(index_path)
        with mp.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        entry = {"index": idx, "meta": meta, "idx_mtime": idx_m, "meta_mtime": mt_m}
        _CACHE[key] = entry
    return entry["index"], entry["meta"]


def warm_cache(index_path: str, meta_path: str) -> bool:
    try:
        _ = load_encoder()
        idx, meta = _maybe_load_cache(index_path, meta_path)
        return bool(idx is not None and meta is not None)
    except Exception:
        return False


def search_index(query: str, index_path: str, meta_path: str, k: int = 4, mmr: bool = True, path_contains: str | None = None) -> List[Dict]:
    # Warm encoder session once per process
    _ = load_encoder()
    index, meta = _maybe_load_cache(index_path, meta_path)
    if index is None or meta is None:
        return []
    q = embed_texts([query])
    q = _l2_normalize(q)
    # Search a larger candidate pool to improve recall on small notes
    cand = min(32, index.ntotal if index.ntotal > 0 else 32)
    D, I = index.search(q, cand)  # type: ignore[union-attr]
    if index.ntotal == 0:
        return []
    idxs = I[0].tolist()
    if mmr:
        vecs = _l2_normalize(embed_texts([meta[i]["text"] for i in idxs]))  # type: ignore[index]
        pick = _mmr(q[0], vecs, k)
        idxs = [idxs[i] for i in pick]
    else:
        idxs = idxs[:k]
    out: List[Dict] = []
    for i in idxs:
        rec = meta[i]  # type: ignore[index]
        if path_contains and path_contains.lower() not in rec["path"].lower():
            continue
        out.append({"text": rec["text"], "path": rec["path"], "meta": rec.get("meta", {})})
    # If a restrictive filter produced no results, return top-k without filter
    if path_contains and not out:
        for i in idxs[:k]:
            rec = meta[i]
            out.append({"text": rec["text"], "path": rec["path"], "meta": rec.get("meta", {})})
    return out


