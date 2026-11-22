"""
chunking.py
-------------
Simple, deterministic text chunking: splits on paragraph boundaries first,
then packs paragraphs into ~target_chars-sized chunks with a small
overlap, falling back to a hard character split for any single paragraph
longer than target_chars. Good enough for the structured document types
this service ingests (manuals, incident reports, logs) without pulling in
a heavier NLP sentence-segmentation dependency.
"""

from __future__ import annotations


def chunk_text(text: str, target_chars: int = 500, overlap_chars: int = 80) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > target_chars:
            # Hard-split an oversized paragraph on its own
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), target_chars - overlap_chars):
                chunks.append(para[i:i + target_chars])
            continue

        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) <= target_chars:
            current = candidate
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks or [text.strip()]
