from __future__ import annotations

from typing import List


def chunk_text(s: str, size: int = 700, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Paragraphs are split on double newlines. Then we window across paragraphs
    to build chunks with an overlap of `overlap` characters.
    """
    if not s:
        return []
    paragraphs = [p.strip() for p in s.split("\n\n") if p.strip()]
    joined = "\n\n".join(paragraphs)
    chunks: List[str] = []
    i = 0
    while i < len(joined):
        end = min(len(joined), i + size)
        chunk = joined[i:end]
        chunks.append(chunk)
        if end == len(joined):
            break
        i = max(0, end - overlap)
    return chunks


