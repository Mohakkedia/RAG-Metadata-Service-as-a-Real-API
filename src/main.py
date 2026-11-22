"""
main.py
---------
FastAPI service combining a RAG retrieval pipeline with an explicit
metadata layer — designed to be called by both a human-facing UI and
another AI agent (see `response_format` on /search and the machine-
readable /schema contract).

Run:
    uvicorn src.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import chunk_text
from ingestion import extract_text
from schema import (
    ContentType, DocumentMetadata, IngestResponse, IngestTextRequest,
    SCHEMA_VERSION, SearchFilters, SearchRequest, SearchResponse, SearchResultItem,
    schema_definition,
)
from store import MetadataStore

DB_PATH = os.environ.get("METADATA_DB_PATH", "data/metadata_service.db")

app = FastAPI(
    title="RAG + Metadata Service",
    version=SCHEMA_VERSION,
    description="Ingests mixed documents, embeds and stores them with schema-versioned "
                "metadata, and exposes /search and /ingest for both human UIs and AI agents.",
)

store = MetadataStore(DB_PATH)


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/schema")
def get_schema():
    """Machine-readable metadata contract — an agent should call this once
    (or cache it, keyed by schema_version) before parsing /search or
    /documents responses, so a future schema bump doesn't silently break it."""
    return schema_definition()


@app.get("/stats")
def stats():
    return {
        "n_documents": store.count_documents(),
        "n_chunks": store.count_chunks(),
        "schema_version": SCHEMA_VERSION,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest):
    """Ingest a plain-text or json_log/csv document supplied inline as a string."""
    try:
        extracted = extract_text(req.content_type, text=req.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Extraction failed: {exc}")

    chunks = chunk_text(extracted)
    doc_meta = store.add_document(
        source=req.source, content_type=req.content_type, title=req.title,
        chunks_text=chunks, tags=req.tags,
    )
    return IngestResponse(doc_id=doc_meta.doc_id, n_chunks=len(chunks), metadata=doc_meta)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source: str = Form(...),
    content_type: ContentType = Form(...),
    title: str = Form(...),
    tags: str = Form(""),  # comma-separated
):
    """Ingest an uploaded file (PDF, JSON log export, or CSV)."""
    raw_bytes = await file.read()
    try:
        extracted = extract_text(content_type, raw_bytes=raw_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Extraction failed: {exc}")

    if not extracted.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in uploaded file")

    chunks = chunk_text(extracted)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    doc_meta = store.add_document(
        source=source, content_type=content_type, title=title,
        chunks_text=chunks, tags=tag_list, extra={"original_filename": file.filename},
    )
    return IngestResponse(doc_id=doc_meta.doc_id, n_chunks=len(chunks), metadata=doc_meta)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    filters = req.filters or SearchFilters()
    raw_results = store.search(
        req.query, top_k=req.top_k, source=filters.source,
        content_type=filters.content_type, tags=filters.tags,
    )

    items = []
    for r in raw_results:
        citation = f"[{r['doc_title']}, {r['source']}] chunk {r['chunk_index']}"
        text = r["text"]
        if req.response_format == "human" and len(text) > 280:
            text = text[:277] + "..."
        items.append(SearchResultItem(
            chunk_id=r["chunk_id"], doc_id=r["doc_id"], doc_title=r["doc_title"],
            source=r["source"], content_type=r["content_type"],
            relevance_score=r["relevance_score"], text=text, citation=citation,
        ))

    return SearchResponse(query=req.query, n_results=len(items), results=items)


@app.get("/documents")
def list_documents(source: str | None = None, limit: int = 50, offset: int = 0):
    return {"documents": store.list_documents(source=source, limit=limit, offset=offset)}


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    deleted = store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"status": "deleted", "doc_id": doc_id}
