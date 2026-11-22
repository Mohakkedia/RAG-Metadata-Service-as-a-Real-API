"""
schema.py
-----------
Explicit, versioned metadata schema for ingested documents and chunks —
the "metadata service" half of this project, as distinct from just "a
RAG pipeline." Every record stored carries a `schema_version` field, and
GET /schema lets a caller (human or another AI agent) introspect the
current contract programmatically before it ingests or parses a response,
rather than needing to read source code or documentation to know what
fields exist.

Bumping SCHEMA_VERSION is how a breaking metadata change gets communicated:
old records keep their original schema_version forever (never silently
rewritten), and callers can branch on it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

ContentType = Literal["text", "pdf", "json_log", "csv"]


class DocumentMetadata(BaseModel):
    """Metadata stored per source document (not per chunk)."""
    doc_id: str
    source: str = Field(..., description="Where this document came from, e.g. 'equipment-manuals', 'maintenance-logs'")
    content_type: ContentType
    title: str
    tags: list[str] = Field(default_factory=list)
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = SCHEMA_VERSION
    n_chunks: int = 0
    char_count: int = 0
    extra: dict = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Metadata stored per chunk — the actual unit that gets embedded and searched."""
    chunk_id: str
    doc_id: str  # lineage: which document this chunk came from
    chunk_index: int
    text: str
    char_count: int
    schema_version: str = SCHEMA_VERSION


class IngestTextRequest(BaseModel):
    source: str
    content_type: ContentType = "text"
    title: str
    text: str
    tags: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    doc_id: str
    n_chunks: int
    metadata: DocumentMetadata


class SearchFilters(BaseModel):
    source: str | None = None
    content_type: ContentType | None = None
    tags: list[str] | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: SearchFilters | None = None
    response_format: Literal["human", "agent"] = "agent"


class SearchResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    source: str
    content_type: ContentType
    relevance_score: float
    text: str
    citation: str  # a stable, human-and-agent-readable reference string


class SearchResponse(BaseModel):
    query: str
    n_results: int
    results: list[SearchResultItem]
    schema_version: str = SCHEMA_VERSION


def schema_definition() -> dict:
    """Machine-readable description of the current metadata contract —
    served at GET /schema so a calling agent can introspect field names,
    types, and the current version before parsing responses."""
    return {
        "schema_version": SCHEMA_VERSION,
        "document_metadata_fields": {
            "doc_id": "string, stable unique identifier",
            "source": "string, logical source/collection name",
            "content_type": "enum: text | pdf | json_log | csv",
            "title": "string",
            "tags": "list of strings",
            "ingested_at": "ISO-8601 UTC timestamp",
            "schema_version": "string, semver — the version THIS record was written under",
            "n_chunks": "integer",
            "char_count": "integer, total characters across all chunks",
        },
        "chunk_metadata_fields": {
            "chunk_id": "string, stable unique identifier",
            "doc_id": "string, foreign key to the parent document (lineage)",
            "chunk_index": "integer, 0-based position within the parent document",
            "text": "string, the chunk's raw text",
            "char_count": "integer",
        },
        "search_result_fields": {
            "chunk_id": "string",
            "doc_id": "string",
            "doc_title": "string",
            "source": "string",
            "content_type": "enum",
            "relevance_score": "float, 0-1, cosine similarity",
            "text": "string",
            "citation": "string, e.g. '[doc_title, source] chunk 2'",
        },
    }
