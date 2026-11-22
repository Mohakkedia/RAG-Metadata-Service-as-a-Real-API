"""
generate_sample_pdf.py
-------------------------
Generates a small, realistic-looking PDF equipment manual so
/ingest/file's PDF path has something real to extract text from and be
tested against (rather than only ever exercising the text/json/csv
branches of ingestion.py).

Output: data/sample_docs/hydraulic_press_manual.pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT_PATH = Path(__file__).parent.parent / "data" / "sample_docs" / "hydraulic_press_manual.pdf"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SECTIONS = [
    ("Hydraulic Press HP-900 — Maintenance Manual", None),
    ("1. Overview", "The HP-900 hydraulic press is used for sheet metal forming operations across "
                     "Lines 5 through 8. This manual covers routine maintenance, troubleshooting of "
                     "pressure loss and erratic cycling, and safety procedures required before any "
                     "physical inspection or repair."),
    ("2. Pressure Loss Diagnosis", "Erratic cycle times or inability to reach set pressure on the HP-900 "
                     "is most commonly caused by internal seal wear in the main cylinder. Look for slow "
                     "pressure bleed-down when the press is idle as the primary diagnostic signal. A "
                     "secondary cause is air entrainment in the hydraulic fluid, visible as foaming in "
                     "the reservoir sight glass. A partially clogged filter increasing pump strain and "
                     "heat is a third, less common cause."),
    ("3. Fluid Temperature", "Hydraulic fluid temperature above 60 degrees Celsius accelerates seal "
                     "degradation and should be investigated as a contributing root cause, not merely "
                     "treated as a symptom. Fluid temperature should be checked at the reservoir monthly "
                     "and logged."),
    ("4. Safety Requirements", "Lockout-tagout (LOTO) and full pressure bleed-down are required before "
                     "opening any hydraulic line or accessing the main cylinder. Visual inspection of "
                     "the reservoir sight glass from outside the guarding does NOT require LOTO."),
    ("5. Scheduled Maintenance", "Replace the hydraulic filter every 2,000 operating hours. Inspect main "
                     "cylinder seals every 5,000 operating hours or immediately if pressure loss symptoms "
                     "are observed. Fluid should be fully replaced every 10,000 operating hours."),
]


def main():
    doc = SimpleDocTemplate(str(OUT_PATH), pagesize=LETTER)
    styles = getSampleStyleSheet()
    flow = []
    title, _ = SECTIONS[0]
    flow.append(Paragraph(title, styles["Title"]))
    flow.append(Spacer(1, 16))
    for heading, body in SECTIONS[1:]:
        flow.append(Paragraph(heading, styles["Heading2"]))
        flow.append(Paragraph(body, styles["BodyText"]))
        flow.append(Spacer(1, 10))
    doc.build(flow)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
