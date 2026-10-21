"""Build the visual PDF version of the concise DS23 MURA XR_WRIST Charter."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


OUTPUT = Path("docs/submissions/MURA_Agentic_AI_Capstone_Charter_final.pdf")


def section(number: int, heading: str, text: str, styles) -> list:
    return [
        Paragraph(f"{number}. {heading}", styles["section"]),
        Paragraph(text, styles["body"]),
        Spacer(1, 3),
    ]


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawCentredString(letter[0] / 2, 0.26 * inch, "DS23 | MURA XR WRIST | Research prototype only")
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter,
        rightMargin=0.52 * inch, leftMargin=0.52 * inch,
        topMargin=0.46 * inch, bottomMargin=0.42 * inch,
        title="DS23 Project Charter MURA XR WRIST Decision Support",
        author="Team 1",
    )
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("CharterTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=15, leading=17, alignment=TA_CENTER, textColor=colors.black, spaceAfter=2),
        "team": ParagraphStyle("Team", parent=base["Normal"], fontName="Helvetica", fontSize=9.3, leading=11, alignment=TA_CENTER, textColor=colors.black, spaceAfter=5),
        "section": ParagraphStyle("Section", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=9.7, leading=11.2, textColor=colors.black, spaceBefore=2.6, spaceAfter=1.2),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=8.65, leading=10.15, textColor=colors.black, spaceAfter=0),
        "cell": ParagraphStyle("Cell", parent=base["BodyText"], fontName="Helvetica", fontSize=7.9, leading=9.0, textColor=colors.black),
        "cellhead": ParagraphStyle("CellHead", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8.0, leading=9.0, textColor=colors.white, alignment=TA_CENTER),
    }
    story = [
        Paragraph("DS23 Project Charter  MURA XR WRIST Decision Support", styles["title"]),
        Paragraph("Team 1  Netanel  Hagit  Hodaya  Artium", styles["team"]),
    ]
    story += section(1, "Problem and user", "The intended user is an Emergency Department physician reviewing wrist radiographs before specialist radiology review is available. The project studies decision support for the MURA study-level NORMAL or ABNORMAL label. It does not claim acute-fracture diagnosis, lesion localization, autonomous diagnosis, treatment advice, or a measurable reduction in return visits.", styles)
    story += section(2, "Success metric", "<b>Pre-specified operating target:</b> study-level sensitivity &gt;=95%. For each model, select the highest threshold that attains the target on aggregated internal-validation study probabilities, then freeze it. Report specificity, false-negative rate (1 - sensitivity), precision, F1, AUROC, AUPRC, confusion matrix, and 95% patient-cluster bootstrap intervals. This is a project target, not an untouched-test claim.", styles)
    story += section(3, "Data and Plan B", "<b>Plan A:</b> Stanford MURA v1.1 XR_WRIST, organised patient -&gt; study -&gt; one or more views. Official MURA train is split at patient level into internal train and validation. Logistic Regression C is selected by patient-grouped CV within train; thresholds are selected only on internal validation. Official MURA valid was opened once for a frozen Phase-1 reference and is not untouched for future selection. Raw images and protected local data never enter Git. <b>Plan B:</b> FracAtlas changes the task to fracture detection and requires revised labels, anatomy/QC rules, and evaluation.", styles)
    story += section(4, "Architecture sketch", "Data -&gt; patient / study / views -&gt; image model -&gt; mean study-level aggregation -&gt; versioned probability and evidence -&gt; deterministic safety and routing -&gt; routine human workflow or ambiguity investigation -&gt; bounded adaptive agent -&gt; structured human handoff. The mandatory baseline is frozen ImageNet ResNet-18 embeddings -&gt; mean study embedding -&gt; Logistic Regression, compared fairly with fine-tuned ResNet-18 on the same development cohort and metrics. The deterministic path works without an agent.", styles)
    story += section(5, "Agent seat", "Hard rules are deterministic and non-overridable: supported anatomy, corrupt/missing input, mandatory human review, and the frozen operating policy. AI may raise priority but never reduce standard review. A bounded agent is proposed only for genuine ambiguity: conflicting view evidence; borderline score with borderline QC; or missing/conflicting tool output. It may call approved QC, support, model, disagreement, uncertainty, diagnostic Grad-CAM, model-card, and calculator tools, then return NEEDS_HUMAN_REVIEW or UNSUPPORTED. It may not diagnose, treat, alter thresholds, bypass guardrails, or claim localization. Rule routing, fixed workflow, and the adaptive agent will be compared on independently authored safety cases.", styles)
    story.append(Paragraph("6. Risks and cut line", styles["section"]))
    risks = [
        [Paragraph("Risk", styles["cellhead"]), Paragraph("Mitigation", styles["cellhead"])],
        [Paragraph("MURA abnormality is not acute fracture", styles["cell"]), Paragraph("Maintain scoped wording; no fracture claims; defensible error review only.", styles["cell"])],
        [Paragraph("Markers, text, casts, hardware, borders, artifacts", styles["cell"]), Paragraph("Targeted shortcut/error checks; Grad-CAM is diagnostic, not localization ground truth.", styles["cell"])],
        [Paragraph("Observed official-valid reference and limited sample size", styles["cell"]), Paragraph("No later selection on official valid; bootstrap intervals and transparent reuse limitation.", styles["cell"])],
        [Paragraph("Agent as if/else wrapper or hallucination source", styles["cell"]), Paragraph("Non-overridable guardrails, typed tools, hostile-input tests, independent evaluation.", styles["cell"])],
    ]
    risk_table = Table(risks, colWidths=[2.05 * inch, 5.12 * inch], hAlign="LEFT")
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF2F8")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D9D9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [risk_table, Spacer(1, 3)]
    story.append(Paragraph("<b>Cut line:</b> a reproducible study-level wrist-abnormality classifier with the mandatory baseline, patient-safe protocol, high-sensitivity policy, bootstrap/error/shortcut analysis, deterministic safety, and a separately evaluated bounded agent design. Stronger models, learned aggregation, calibration, Grad-CAM, and UI are optional.", styles["body"]))
    story += [Spacer(1, 3), Paragraph("7. Milestones and hats", styles["section"])]
    milestones = [
        [Paragraph("Stage", styles["cellhead"]), Paragraph("Deliverable", styles["cellhead"]), Paragraph("Owner / hat", styles["cellhead"])],
        [Paragraph("1", styles["cell"]), Paragraph("Thin end-to-end baseline: patient-safe split, aggregation, validation-only threshold, baseline comparison.", styles["cell"]), Paragraph("Hagit - Data<br/>Hodaya - Model", styles["cell"])],
        [Paragraph("2", styles["cell"]), Paragraph("Agent integration plus safety/evaluation suite; deterministic workflow remains operational.", styles["cell"]), Paragraph("Netanel - Agent", styles["cell"])],
        [Paragraph("Final", styles["cell"]), Paragraph("Error/shortcut analysis, governance, rule-vs-agent comparison, demo/docs, future cost estimate.", styles["cell"]), Paragraph("Artium - Product", styles["cell"])],
    ]
    milestone_table = Table(milestones, colWidths=[0.72 * inch, 4.86 * inch, 1.59 * inch], hAlign="LEFT")
    milestone_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF2F8")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D9D9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(milestone_table)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
