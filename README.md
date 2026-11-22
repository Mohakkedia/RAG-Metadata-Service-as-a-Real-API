# 🔎 RAG + Metadata Service — a Real API, Not a Notebook Demo

A FastAPI service that ingests mixed document types (PDF manuals,
structured JSON logs, CSV work orders, plain text), embeds and stores them
with **explicit, schema-versioned metadata**, and exposes `/search` and
`/ingest` designed to be called by both a human UI and another AI agent —
the "metadata services" and "API design for AI systems" pieces most RAG
tutorials skip because they only build the retrieval half.

## Why this is a service, not a retrieval script

| What most RAG demos have | What this has |
|---|---|
| A notebook cell that embeds and searches | A running FastAPI app with `/ingest`, `/search`, `/documents`, `/schema`, `/stats`, `DELETE` |
| An implicit, undocumented data shape | `GET /schema` — a machine-readable contract an agent can introspect *before* parsing responses, versioned so a future breaking change doesn't silently corrupt a caller |
| One document type (usually plain text) | Four: `text`, `pdf` (pypdf), `json_log` (flattened to searchable text), `csv` (row-flattened) |
| An in-memory vector list that resets on restart | A real SQLite-backed store — documents, chunks, and embeddings persist across restarts, with lineage (`chunk.doc_id` links back to its parent) |
| A `response_format` nobody thought about | `search_format: "human"` (truncated snippets) vs. `"agent"` (full text + a stable `citation` string) — the same endpoint serving two different callers correctly |

## Architecture

```
src/schema.py       Pydantic models + the versioned metadata contract (GET /schema)
src/chunking.py      Deterministic paragraph-aware text chunking
src/ingestion.py      Extracts plain text from text/pdf/json_log/csv inputs
src/embeddings.py      Stateless HashingVectorizer embeddings (see below)
src/store.py            SQLite-backed documents/chunks/embeddings storage + brute-force cosine search
src/main.py               FastAPI app wiring it all together
data/sample_docs/          A real PDF manual, JSON log, and CSV work-order file
scripts/demo_ingest_and_search.py   Ingests all 4 types + runs 5 real searches -> outputs/demo_results.json
tests/test_api.py            9 tests covering ingest/search/delete/filters/formats
```

## Why a stateless hashing embedding, not a downloaded model

`src/embeddings.py` uses scikit-learn's `HashingVectorizer` rather than a
downloaded dense-embedding model (no route to huggingface.co in this
sandbox, and OpenAI embeddings would cost API calls) — but the choice is
also the *architecturally correct* one for this specific requirement: a
hashing vectorizer needs no corpus-wide fit, so `/ingest` can genuinely add
one document at a time, incrementally, forever, with no periodic
reindexing job. A TF-IDF vectorizer (fit once on a corpus) cannot support
that without a background re-fit job. Swapping in a real dense embedding
model later is a ~10-line change confined to this one file.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_sample_pdf.py      # -> data/sample_docs/hydraulic_press_manual.pdf
python scripts/demo_ingest_and_search.py   # ingests all 4 doc types, runs 5 searches, saves outputs/demo_results.json

uvicorn src.main:app --host 0.0.0.0 --port 8001   # run the live service
# then: curl http://localhost:8001/schema
#       curl http://localhost:8001/stats

pytest tests/ -v
```

## Real results from a live run

Ingesting a text manual, a generated PDF manual, a JSON maintenance log,
and a CSV work-order export (13 chunks across 5 documents), then
searching **"bearing running hot with rising vibration"** returns, ranked
correctly by cosine similarity, the actual bearing-wear manual first
(0.204) — ahead of a work order that just happens to mention a different
motor (0.083) and the unrelated hydraulic manual (0.080):

```json
[0.2039] [Bearing Wear and Overheating — Diagnostic Guide, equipment-manuals] chunk 0
[0.0825] [Open/Closed Work Orders, work-orders] chunk 0
[0.0803] [HP-900 Maintenance Manual, equipment-manuals] chunk 0
```

Searching **"hydraulic press losing pressure erratic cycling"** correctly
surfaces the PDF-extracted HP-900 manual text (0.315) ahead of everything
else — proving the PDF text-extraction path works, not just the plain-text
path. Metadata filters work too: scoping a search to `source: "work-orders"`
returns only work-order chunks. Full output: `outputs/demo_results.json`.

## Example API calls

```bash
# Ingest inline text
curl -X POST localhost:8001/ingest -H "Content-Type: application/json" -d '{
  "source": "equipment-manuals", "content_type": "text",
  "title": "Bearing Guide", "text": "Bearing wear causes rising temperature and vibration.",
  "tags": ["bearing"]
}'

# Ingest a file (PDF/JSON/CSV)
curl -X POST localhost:8001/ingest/file \
  -F "file=@manual.pdf" -F "source=equipment-manuals" -F "content_type=pdf" -F "title=My Manual"

# Search, agent-formatted (full text + citation)
curl -X POST localhost:8001/search -H "Content-Type: application/json" -d '{
  "query": "bearing running hot", "top_k": 3, "response_format": "agent"
}'

# Introspect the metadata contract
curl localhost:8001/schema
```

## Key Technologies
FastAPI, SQLite (persistent metadata + vector storage), scikit-learn (HashingVectorizer), pypdf, Pydantic (schema/contract design)

## Disclaimer
Educational/portfolio project. Sample documents are entirely synthetic.
Image ingestion is a documented extension point (see `src/ingestion.py`
docstring) rather than implemented, since this sandbox has no route to an
OCR/captioning model host.
