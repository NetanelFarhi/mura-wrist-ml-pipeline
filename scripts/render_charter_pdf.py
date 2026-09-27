"""Render the submission Charter as a compact, review-ready PDF.

The source of truth remains CHARTER.md. This renderer intentionally keeps
unassigned roles visible rather than inventing team ownership.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "MURA_XR_WRIST_CAPSTONE_CHARTER.pdf"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def footer(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2F0"))
    canvas.line(document.leftMargin, 13 * mm, A4[0] - document.rightMargin, 13 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#53657A"))
    canvas.drawString(document.leftMargin, 8.8 * mm, "MURA XR_WRIST | Capstone Charter | Team draft")
    canvas.drawRightString(A4[0] - document.rightMargin, 8.8 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=13 * mm,
        bottomMargin=18 * mm,
        title="MURA XR_WRIST Capstone Charter",
        author="Artium, Hagit, Hodaya, Netanel",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "CharterTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        textColor=colors.HexColor("#17365D"),
        alignment=TA_CENTER,
        spaceAfter=3,
    )
    metadata = ParagraphStyle(
        "Metadata",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.3,
        leading=10,
        textColor=colors.HexColor("#53657A"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.1,
        leading=12,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=3,
        spaceAfter=1.5,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.05,
        leading=10.15,
        textColor=colors.HexColor("#202A35"),
        spaceAfter=2.2,
    )
    note = ParagraphStyle(
        "Note",
        parent=body,
        fontName="Helvetica-Oblique",
        fontSize=7.4,
        leading=9.0,
        textColor=colors.HexColor("#7A4B00"),
        backColor=colors.HexColor("#FFF4D6"),
        borderColor=colors.HexColor("#E6BE6A"),
        borderWidth=0.5,
        borderPadding=4,
        spaceBefore=2,
        spaceAfter=4,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=8.6,
        textColor=colors.white,
    )
    table_body = ParagraphStyle(
        "TableBody",
        parent=body,
        fontSize=7.0,
        leading=8.2,
        spaceAfter=0,
    )

    story = [
        Paragraph("MURA XR_WRIST Capstone Charter", title),
        paragraph("Submission: 4.10.2026 | Team: Artium, Hagit, Hodaya, Netanel", metadata),
        paragraph(
            "Draft note: Team names are confirmed. Assign the Data, Model, Agent, and Product hats before final submission; ownership is intentionally not invented here.",
            note,
        ),
        Paragraph("1. Problem + user", heading),
        paragraph(
            "User: Emergency Department physician reviewing wrist radiographs before expert review. The MVP flags MURA-defined musculoskeletal abnormality per study (NORMAL/ABNORMAL), not acute fracture, lesion location, treatment, or autonomous diagnosis. In the historical Phase-1 reference, selected operating points missed 17 or 26 of 97 abnormal studies; this makes the sensitivity/specificity trade-off measurable. Delayed review is clinical motivation only, because this dataset cannot measure ED revisits/outcomes.",
            body,
        ),
        Paragraph("2. Success metric", heading),
        paragraph(
            "Pre-specified target: study-level sensitivity >=90%. During development, choose the highest threshold meeting that floor on internal patient-level validation, freeze it before the defined evaluation, then report specificity, precision, F1, AUROC, AUPRC, FN rate (=1-sensitivity among abnormal studies), and 95% patient/study bootstrap CIs. NPV is not primary because prevalence-dependent. This is a target, not an untouched advanced-model claim; the future advanced-model evaluation protocol remains an open methodological decision.",
            body,
        ),
        Paragraph("3. Data + Plan B", heading),
        paragraph(
            "Plan A: Stanford MURA v1.1, research/non-commercial, XR_WRIST: local access was verified and indexed as 3,474 patients, 3,697 studies and 10,411 images; patient -> study -> 1-8 views, one NORMAL/ABNORMAL study label. Official train -> patient-level internal train/validation; official valid was used once as frozen Phase-1 reference after validation-only choices, and is now observed/not untouched for later selection. No raw images in Git.",
            body,
        ),
        paragraph(
            "Plan B: FracAtlas only if access fails; it changes the task to fracture detection plus anatomy/QC/evaluation assumptions and requires documented adaptation.",
            body,
        ),
        Paragraph("4. Architecture sketch", heading),
        paragraph(
            "Data -> Patient/Study/View -> Model -> declared study aggregation -> probability/evidence -> deterministic safety/routing -> optional ambiguity investigation -> bounded adaptive agent -> workflow escalation -> human clinician.",
            body,
        ),
        paragraph(
            "The deterministic system works without agent: supported anatomy, corruption/QC rejection, mandatory review rules, safety thresholds. AI may raise priority; never lower a case below standard review. Historical aggregation: Model A mean embeddings; Model B mean probability; learned aggregation remains experimental.",
            body,
        ),
        Paragraph("5. Agent seat", heading),
        paragraph(
            "Known rules remain deterministic. The bounded agent only investigates ambiguity: conflicting view/model evidence, borderline QC, whether permitted uncertainty/TTA is worthwhile, missing evidence, and grounded summary. It cannot diagnose/treat, override safety, or alter thresholds; it may return NEEDS_HUMAN_REVIEW or UNSUPPORTED.",
            body,
        ),
        paragraph(
            "Tools: untrusted metadata, QC/support checker, primary model, per-view disagreement, approved uncertainty/TTA, Grad-CAM diagnostic, model card, calculator. It is evaluated against rule and fixed-workflow baselines on the same curated cases for escalation, groundedness, evidence, tool use/cost, unsupported handling and violations. Metadata instructions never alter policy.",
            body,
        ),
        Paragraph("6. Risks + cut line", heading),
        paragraph(
            "1. MURA abnormality != acute fracture: honest scope; exploratory errors only. 2. Shortcuts (markers/text/borders/hardware): Grad-CAM diagnostic + targeted review; no localization claims. 3. Agent wrapper/hallucination: non-overridable rules, hostile-input evaluation, bounded tools/failure-to-review. 4. Small evaluation: bootstrap CIs/no over-precision.",
            body,
        ),
        paragraph(
            "Cut line: reproducible wrist-abnormality classifier, cheap baseline comparison, pre-specified operating point, error/shortcut analysis, deterministic safety policy, separately evaluated bounded agent. Optional advanced ML/UI cut first.",
            body,
        ),
        Paragraph("7. Milestones + hats", heading),
    ]

    table_rows = [
        [paragraph("Checkpoint", table_header), paragraph("Deliverable", table_header), paragraph("Ownership", table_header)],
        [paragraph("18.10.2026", table_body), paragraph("Thin end-to-end baseline without agent", table_body), paragraph("Data / Model / Product: [TEAM ASSIGNMENT REQUIRED]", table_body)],
        [paragraph("30.10.2026", table_body), paragraph("Guarded agent integrated + evaluated", table_body), paragraph("Agent / presenters: [TEAM ASSIGNMENT REQUIRED]", table_body)],
        [paragraph("12.11.2026", table_body), paragraph("ML/error analysis, rule-vs-agent evaluation, demo/docs/cost", table_body), paragraph("Data / Model / Agent / Product: [TEAM ASSIGNMENT REQUIRED]", table_body)],
    ]
    milestone_table = Table(table_rows, colWidths=[26 * mm, 73 * mm, 67 * mm], repeatRows=1)
    milestone_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C9DD")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F8FC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.extend(
        [
            milestone_table,
            Spacer(1, 3),
            paragraph(
                "Mandatory cheap baseline: frozen ImageNet ResNet-18 embeddings + study aggregation + Logistic Regression, fairly compared with fine-tuned ResNet-18 on same population/metrics. Preliminary results are historical, not future untouched-test evidence.",
                body,
            ),
        ]
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
