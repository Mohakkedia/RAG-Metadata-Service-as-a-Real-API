import os
import sys
from pathlib import Path

TEST_DB = Path(__file__).resolve().parents[1] / "data" / "test_metadata_service.db"
if TEST_DB.exists():
    TEST_DB.unlink()
os.environ["METADATA_DB_PATH"] = str(TEST_DB)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_schema_contract_has_version():
    resp = client.get("/schema")
    body = resp.json()
    assert "schema_version" in body
    assert "document_metadata_fields" in body
    assert "chunk_metadata_fields" in body


def test_ingest_text_and_search_finds_it():
    ingest_resp = client.post("/ingest", json={
        "source": "test-source", "content_type": "text", "title": "Test Doc About Bearings",
        "text": "Bearing wear causes elevated temperature and vibration in conveyor motors.",
        "tags": ["test"],
    })
    assert ingest_resp.status_code == 200
    doc_id = ingest_resp.json()["doc_id"]
    assert doc_id.startswith("doc_")

    search_resp = client.post("/search", json={"query": "bearing temperature vibration", "top_k": 3})
    assert search_resp.status_code == 200
    results = search_resp.json()["results"]
    assert len(results) > 0
    assert any(r["doc_id"] == doc_id for r in results)


def test_search_filter_by_source():
    client.post("/ingest", json={
        "source": "source-a", "content_type": "text", "title": "Doc A", "text": "Alpha content about widgets.",
    })
    client.post("/ingest", json={
        "source": "source-b", "content_type": "text", "title": "Doc B", "text": "Beta content about widgets.",
    })
    resp = client.post("/search", json={
        "query": "widgets", "top_k": 10, "filters": {"source": "source-a"},
    })
    results = resp.json()["results"]
    assert len(results) > 0
    assert all(r["source"] == "source-a" for r in results)


def test_get_and_delete_document():
    ingest_resp = client.post("/ingest", json={
        "source": "test-source", "content_type": "text", "title": "Deletable Doc",
        "text": "This document will be deleted in this test.",
    })
    doc_id = ingest_resp.json()["doc_id"]

    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["title"] == "Deletable Doc"

    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200

    get_after_delete = client.get(f"/documents/{doc_id}")
    assert get_after_delete.status_code == 404


def test_ingest_csv_file():
    import io
    csv_content = "name,value\nwidget,10\ngadget,20\n"
    resp = client.post(
        "/ingest/file",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        data={"source": "csv-test", "content_type": "csv", "title": "Test CSV"},
    )
    assert resp.status_code == 200
    assert resp.json()["n_chunks"] >= 1


def test_ingest_json_log_file():
    import io
    json_content = '[{"event": "test_event", "value": 42}]'
    resp = client.post(
        "/ingest/file",
        files={"file": ("test.json", io.BytesIO(json_content.encode()), "application/json")},
        data={"source": "json-test", "content_type": "json_log", "title": "Test Log"},
    )
    assert resp.status_code == 200


def test_search_response_format_human_truncates_long_text():
    long_text = "word " * 200
    client.post("/ingest", json={
        "source": "truncate-test", "content_type": "text", "title": "Long Doc", "text": long_text,
    })
    resp = client.post("/search", json={
        "query": "word", "top_k": 1, "filters": {"source": "truncate-test"}, "response_format": "human",
    })
    result = resp.json()["results"][0]
    assert len(result["text"]) <= 280


def test_stats_endpoint_reflects_ingestion():
    before = client.get("/stats").json()
    client.post("/ingest", json={
        "source": "stats-test", "content_type": "text", "title": "Stats Doc", "text": "Some content for stats.",
    })
    after = client.get("/stats").json()
    assert after["n_documents"] == before["n_documents"] + 1


if __name__ == "__main__":
    test_health()
    test_schema_contract_has_version()
    test_ingest_text_and_search_finds_it()
    test_search_filter_by_source()
    test_get_and_delete_document()
    test_ingest_csv_file()
    test_ingest_json_log_file()
    test_search_response_format_human_truncates_long_text()
    test_stats_endpoint_reflects_ingestion()
    print("All tests passed.")
