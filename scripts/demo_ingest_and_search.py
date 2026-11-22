"""
demo_ingest_and_search.py
-----------------------------
End-to-end demo: ingests all four supported content types (text, pdf,
json_log, csv) through the actual FastAPI app (via TestClient, so this
exercises the real HTTP routes, not the store directly), runs a handful
of representative searches, and saves everything to
outputs/demo_results.json as a real, inspectable artifact.

Run:
    python scripts/demo_ingest_and_search.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "demo_metadata_service.db"

import os
os.environ["METADATA_DB_PATH"] = str(DB_PATH)
if DB_PATH.exists():
    DB_PATH.unlink()

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)

TEXT_DOCS = [
    {
        "source": "equipment-manuals", "content_type": "text", "title": "Bearing Wear and Overheating — Diagnostic Guide",
        "tags": ["bearing", "manual", "CB-4400"],
        "text": "Elevated bearing temperature combined with increased vibration amplitude is the classic "
                "signature of bearing wear or insufficient lubrication. Normal operating temperature for "
                "CB-4400 bearings is 60-75C with vibration under 1.0 mm/s RMS. Sustained readings above 85C "
                "accompanied by vibration exceeding 1.2 mm/s RMS indicate advanced bearing degradation. Root "
                "causes include grease starvation, contaminant ingress, misalignment, or end-of-life wear. "
                "Recommended action: schedule immediate inspection; LOTO is required before any physical work.",
    },
    {
        "source": "equipment-manuals", "content_type": "text", "title": "Distinguishing Sensor Fault from Real Equipment Fault",
        "tags": ["sensor", "manual", "VS-100"],
        "text": "Before acting on an anomalous reading, rule out sensor fault: a genuine equipment problem "
                "produces a gradual trend over minutes to hours, while a sensor fault often produces a sudden "
                "step-change or a reading stuck at a fixed value. If a suspected fault reading has not shown "
                "a gradual precursor trend, sensor malfunction should be considered alongside equipment failure.",
    },
]


def main():
    results_log = {"ingestions": [], "searches": []}

    # 1. Ingest inline text documents
    for doc in TEXT_DOCS:
        resp = client.post("/ingest", json=doc)
        resp.raise_for_status()
        results_log["ingestions"].append({"type": "text", "title": doc["title"], "response": resp.json()})

    # 2. Ingest the PDF manual
    pdf_path = ROOT / "data" / "sample_docs" / "hydraulic_press_manual.pdf"
    with open(pdf_path, "rb") as f:
        resp = client.post(
            "/ingest/file",
            files={"file": ("hydraulic_press_manual.pdf", f, "application/pdf")},
            data={"source": "equipment-manuals", "content_type": "pdf", "title": "HP-900 Maintenance Manual", "tags": "hydraulic,manual,HP-900"},
        )
    resp.raise_for_status()
    results_log["ingestions"].append({"type": "pdf", "title": "HP-900 Maintenance Manual", "response": resp.json()})

    # 3. Ingest the JSON maintenance log
    json_path = ROOT / "data" / "sample_docs" / "maintenance_logs.json"
    with open(json_path, "rb") as f:
        resp = client.post(
            "/ingest/file",
            files={"file": ("maintenance_logs.json", f, "application/json")},
            data={"source": "maintenance-logs", "content_type": "json_log", "title": "August 2026 Maintenance Log", "tags": "log,august-2026"},
        )
    resp.raise_for_status()
    results_log["ingestions"].append({"type": "json_log", "title": "August 2026 Maintenance Log", "response": resp.json()})

    # 4. Ingest the CSV work orders
    csv_path = ROOT / "data" / "sample_docs" / "work_orders.csv"
    with open(csv_path, "rb") as f:
        resp = client.post(
            "/ingest/file",
            files={"file": ("work_orders.csv", f, "text/csv")},
            data={"source": "work-orders", "content_type": "csv", "title": "Open/Closed Work Orders", "tags": "work-order"},
        )
    resp.raise_for_status()
    results_log["ingestions"].append({"type": "csv", "title": "Open/Closed Work Orders", "response": resp.json()})

    # ---- Searches ----
    queries = [
        {"query": "bearing running hot with rising vibration", "top_k": 3, "response_format": "agent"},
        {"query": "sudden temperature spike no gradual trend", "top_k": 3, "response_format": "agent"},
        {"query": "hydraulic press losing pressure erratic cycling", "top_k": 3, "response_format": "human"},
        {"query": "work orders for LINE-03", "top_k": 3, "response_format": "agent",
         "filters": {"source": "work-orders"}},
        {"query": "sensor thermocouple failure false alarm", "top_k": 3, "response_format": "agent",
         "filters": {"source": "maintenance-logs"}},
    ]
    for q in queries:
        resp = client.post("/search", json=q)
        resp.raise_for_status()
        results_log["searches"].append({"request": q, "response": resp.json()})

    stats = client.get("/stats").json()
    results_log["final_stats"] = stats

    out_path = ROOT / "outputs" / "demo_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results_log, indent=2))

    print(f"Ingested {len(results_log['ingestions'])} documents, ran {len(results_log['searches'])} searches.")
    print(f"Final stats: {stats}")
    print(f"Wrote {out_path}")

    print("\n--- Sample search: 'bearing running hot with rising vibration' ---")
    for r in results_log["searches"][0]["response"]["results"]:
        print(f"  [{r['relevance_score']}] {r['citation']}: {r['text'][:100]}...")


if __name__ == "__main__":
    main()
