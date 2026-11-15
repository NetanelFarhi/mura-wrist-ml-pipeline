"""In-memory PDF export for a completed Wrist Review CrewAI explanation.

The browser can request this export only after the optional research agent has
returned a schema-validated explanation.  The exporter never receives or
persists image bytes, local paths, view file names, labels, or patient IDs.
It is a presentation layer: it cannot run inference, invoke tools, or change a
deterministic route.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import io
import re
from typing import Any


MAX_TEXT_LENGTH = 500
MAX_LIST_ITEMS = 6
_SAFE_ALIAS = re.compile(r"[^A-Za-z0-9._ -]+")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any, *, maximum: int = MAX_LIST_ITEMS) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = _clean_text(item)
        if text:
            items.append(text)
        if len(items) >= maximum:
            break
    return items


def _clean_text(value: Any, *, maximum: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\x00", " ").split())[:maximum]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if 0.0 <= numeric <= 1.0 else None


def _display_action(outcome: str) -> str:
    return {
        "RAISE_PRIORITY": "Priority human review",
        "STANDARD_HUMAN_REVIEW": "Standard human review",
    }.get(outcome, "Human review")


def _completed_tools(agent_report: Mapping[str, Any]) -> list[str]:
    labels = {
        "uncertainty_tta": "TTA stability summary",
        "gradcam_explainer": "Grad-CAM summary",
    }
    result: list[str] = []
    for item in agent_report.get("allowed_tools_called", []):
        tool = _mapping(item)
        if tool.get("status") == "OK" and tool.get("tool") in labels:
            result.append(labels[str(tool["tool"])])
    return result


def report_filename(review: Mapping[str, Any]) -> str:
    """Return a conservative, non-identifying filename for one local export."""

    alias = _clean_text(review.get("case_id"), maximum=64) or "LOCAL_CASE"
    alias = _SAFE_ALIAS.sub("_", alias).strip("._ ") or "LOCAL_CASE"
    return f"{alias}_research_explanation.pdf"


def build_research_report_pdf(review: Mapping[str, Any]) -> bytes:
    """Render a one-page research PDF from an already completed LLM report.

    Raises ``ValueError`` unless the input represents a valid supported,
    deterministic study with a completed, research-only CrewAI explanation.
    """

    research = _mapping(review.get("research_agent_report"))
    if research.get("agent_status") != "completed_research_explanation":
        raise ValueError("A completed CrewAI research explanation is required before exporting a PDF.")
    if research.get("research_use_only") is not True or research.get("human_review_required") is not True:
        raise ValueError("The research explanation does not contain the required safety declarations.")

    outcome = str(review.get("deterministic_outcome") or "")
    if outcome not in {"RAISE_PRIORITY", "STANDARD_HUMAN_REVIEW"}:
        raise ValueError("Only a completed supported study can be exported as a research PDF.")

    evidence = _mapping(review.get("view_evidence"))
    score = _number(evidence.get("mean_probability"))
    threshold = _number(evidence.get("threshold"))
    if score is None or threshold is None:
        raise ValueError("The completed study is missing its frozen score or threshold.")

    summary = _clean_text(research.get("summary"))
    evidence_used = _strings(research.get("evidence_used"))
    limitations = _strings(research.get("limitations"))
    if not summary or not evidence_used or not limitations:
        raise ValueError("The completed research explanation is incomplete.")

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - depends on local optional extras
        raise RuntimeError(
            "PDF export requires ReportLab. Install the optional agent dependencies first."
        ) from exc

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Wrist Review - Research Explanation",
        author="Wrist Review local research prototype",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "WristReviewTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#12335B"),
        alignment=TA_LEFT,
        spaceAfter=3 * mm,
    )
    eyebrow = ParagraphStyle(
        "WristReviewEyebrow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0A8174"),
        spaceAfter=2 * mm,
    )
    section = ParagraphStyle(
        "WristReviewSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#12335B"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    body = ParagraphStyle(
        "WristReviewBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#28374A"),
        spaceAfter=1.5 * mm,
    )
    note = ParagraphStyle(
        "WristReviewNote",
        parent=body,
        fontSize=8.1,
        leading=11,
        textColor=colors.HexColor("#536577"),
    )

    def bullet_rows(items: list[str]) -> list[list[Paragraph]]:
        return [[Paragraph("-", body), Paragraph(item, body)] for item in items]

    def footer(canvas, _document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B9D8D4"))
        canvas.line(18 * mm, 12 * mm, A4[0] - 18 * mm, 12 * mm)
        canvas.setFillColor(colors.HexColor("#536577"))
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(18 * mm, 8 * mm, "Research prototype - mandatory human review - no diagnostic recommendation")
        canvas.drawRightString(A4[0] - 18 * mm, 8 * mm, f"Page {_document.page}")
        canvas.restoreState()

    alias = _clean_text(review.get("case_id"), maximum=80) or "LOCAL_CASE"
    predicted_abnormal = bool(review.get("model_predicted_abnormal"))
    prediction = "Abnormality flagged" if predicted_abnormal else "No abnormality flagged"
    agent_report = _mapping(review.get("agent_report"))
    tools = _completed_tools(agent_report)
    tool_text = ", ".join(tools) if tools else "No optional local tools were used"
    disagreement = "Yes" if evidence.get("threshold_disagreement") is True else "No"

    table_data = [
        ["Study alias", alias],
        ["Model output", prediction],
        ["Mean study score", f"{score * 100:.1f}%"],
        ["Frozen threshold", f"{threshold * 100:.1f}%"],
        ["Workflow route", _display_action(outcome)],
        ["View disagreement", disagreement],
        ["Local evidence", tool_text],
    ]
    overview = Table(table_data, colWidths=[43 * mm, 121 * mm], hAlign="LEFT")
    overview.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E9F3F4")),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F8FBFC")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#213247")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8DCDF")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    evidence_table = Table(bullet_rows(evidence_used), colWidths=[6 * mm, 158 * mm], hAlign="LEFT")
    evidence_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    limits_table = Table(bullet_rows(limitations), colWidths=[6 * mm, 158 * mm], hAlign="LEFT")
    limits_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))

    story = [
        Paragraph("WRIST REVIEW - LOCAL RESEARCH TOOL", eyebrow),
        Paragraph("LLM research explanation", title),
        Paragraph(
            "This is a de-identified research summary of an already-completed deterministic review. It does not diagnose, alter the model output, or replace human review.",
            note,
        ),
        Spacer(1, 4 * mm),
        overview,
        Paragraph("LLM explanation", section),
        Paragraph(summary, body),
        Paragraph("Evidence provided to the LLM", section),
        evidence_table,
        Paragraph("Limits of this report", section),
        limits_table,
        Spacer(1, 3 * mm),
        KeepTogether(
            [
                Paragraph("Safety boundary", section),
                Paragraph(
                    "The LLM received structured, de-identified evidence only. It did not receive radiographs, image paths, file names, labels, or patient identifiers. It cannot change the score, frozen threshold, queue route, or mandatory human-review requirement.",
                    note,
                ),
            ]
        ),
    ]
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
