"""
ingestion.py
--------------
Extracts plain text from each supported content type so the rest of the
pipeline (chunking, embedding, storage) only ever deals with strings.

  text      — used as-is
  pdf       — extracted page-by-page with pypdf
  json_log  — structured log records flattened into readable text lines
              (so a semantic search can actually match against them —
              searching raw JSON text matches poorly)
  csv       — each row flattened into a "column: value" text line

Images are a documented extension point, not implemented here: this
sandbox has no route to a captioning/OCR model host, so wiring one in is
left as the natural next file to add alongside this one (an `image`
branch calling out to an OCR or vision-captioning service to produce the
text this module already knows what to do with).
"""

from __future__ import annotations

import csv
import io
import json

from pypdf import PdfReader


def extract_text(content_type: str, raw_bytes: bytes | None = None, text: str | None = None) -> str:
    if content_type == "text":
        if text is None:
            raise ValueError("text content_type requires `text`, not raw bytes")
        return text

    if content_type == "pdf":
        if raw_bytes is None:
            raise ValueError("pdf content_type requires raw_bytes")
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())

    if content_type == "json_log":
        raw_text = text if text is not None else raw_bytes.decode("utf-8")
        records = json.loads(raw_text)
        if isinstance(records, dict):
            records = [records]
        lines = []
        for i, rec in enumerate(records):
            fields = "; ".join(f"{k}={v}" for k, v in rec.items())
            lines.append(f"Log entry {i}: {fields}")
        return "\n\n".join(lines)

    if content_type == "csv":
        raw_text = text if text is not None else raw_bytes.decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw_text))
        lines = []
        for i, row in enumerate(reader):
            fields = "; ".join(f"{k}: {v}" for k, v in row.items())
            lines.append(f"Row {i}: {fields}")
        return "\n\n".join(lines)

    raise ValueError(f"Unsupported content_type: {content_type}")
