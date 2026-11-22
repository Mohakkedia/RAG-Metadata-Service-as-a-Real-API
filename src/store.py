"""
store.py
----------
SQLite-backed storage for documents, chunks, and their embeddings. Real
persistence (not an in-memory demo dict) so /documents, /search, and
DELETE all reflect actual durable state across process restarts —
exactly what a metadata service needs to be trustworthy.

Embeddings are stored as raw float32 bytes in a BLOB column; search does
a brute-force cosine similarity scan in numpy, which is the right choice
at this scale (hundreds to low thousands of chunks) and is swapped for an
ANN index (FAISS, pgvector, etc.) at the point a real deployment's corpus
outgrows brute force — a change confined to `search()` below.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import numpy as np

from embeddings import EMBEDDING_DIM, cosine_similarity, embed
from schema import ChunkMetadata, DocumentMetadata, SCHEMA_VERSION

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    content_type TEXT NOT NULL,
    title TEXT NOT NULL,
    tags TEXT NOT NULL,           -- JSON-encoded list
    ingested_at TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    n_chunks INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    extra TEXT NOT NULL            -- JSON-encoded dict
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
"""


class MetadataStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # ---- Ingestion ----------------------------------------------------

    def add_document(self, source: str, content_type: str, title: str,
                      chunks_text: list[str], tags: list[str] | None = None,
                      extra: dict | None = None) -> DocumentMetadata:
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        tags = tags or []
        extra = extra or {}
        char_count = sum(len(c) for c in chunks_text)

        doc_meta = DocumentMetadata(
            doc_id=doc_id, source=source, content_type=content_type, title=title,
            tags=tags, n_chunks=len(chunks_text), char_count=char_count, extra=extra,
        )

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, source, content_type, title, tags, "
                "ingested_at, schema_version, n_chunks, char_count, extra) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, source, content_type, title, json.dumps(tags), doc_meta.ingested_at,
                 SCHEMA_VERSION, len(chunks_text), char_count, json.dumps(extra)),
            )
            for i, chunk_text in enumerate(chunks_text):
                chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
                vec = embed(chunk_text)
                conn.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, char_count, "
                    "schema_version, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, i, chunk_text, len(chunk_text), SCHEMA_VERSION, vec.tobytes()),
                )
        return doc_meta

    # ---- Retrieval ------------------------------------------------------

    def get_document(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if row is None:
                return None
            doc = dict(row)
            doc["tags"] = json.loads(doc["tags"])
            doc["extra"] = json.loads(doc["extra"])
            chunk_rows = conn.execute(
                "SELECT chunk_id, chunk_index, text, char_count FROM chunks "
                "WHERE doc_id = ? ORDER BY chunk_index", (doc_id,),
            ).fetchall()
            doc["chunks"] = [dict(r) for r in chunk_rows]
            return doc

    def list_documents(self, source: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._connect() as conn:
            query = "SELECT * FROM documents"
            params: list = []
            if source:
                query += " WHERE source = ?"
                params.append(source)
            query += " ORDER BY ingested_at DESC LIMIT ? OFFSET ?"
            params += [limit, offset]
            rows = conn.execute(query, params).fetchall()
            docs = []
            for row in rows:
                d = dict(row)
                d["tags"] = json.loads(d["tags"])
                d["extra"] = json.loads(d["extra"])
                docs.append(d)
            return docs

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            return cur.rowcount > 0

    def count_documents(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def count_chunks(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ---- Search -----------------------------------------------------------

    def search(self, query: str, top_k: int = 5, source: str | None = None,
               content_type: str | None = None, tags: list[str] | None = None) -> list[dict]:
        with self._connect() as conn:
            sql = (
                "SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.embedding, "
                "d.title as doc_title, d.source, d.content_type, d.tags "
                "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id WHERE 1=1"
            )
            params: list = []
            if source:
                sql += " AND d.source = ?"
                params.append(source)
            if content_type:
                sql += " AND d.content_type = ?"
                params.append(content_type)
            rows = conn.execute(sql, params).fetchall()

        if tags:
            rows = [r for r in rows if set(tags) & set(json.loads(r["tags"]))]

        if not rows:
            return []

        doc_vecs = np.stack([
            np.frombuffer(r["embedding"], dtype=np.float32) for r in rows
        ])
        query_vec = embed(query)
        sims = cosine_similarity(query_vec, doc_vecs)

        ranked_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            row = rows[idx]
            results.append({
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "doc_title": row["doc_title"],
                "source": row["source"],
                "content_type": row["content_type"],
                "relevance_score": round(float(sims[idx]), 4),
            })
        return results
