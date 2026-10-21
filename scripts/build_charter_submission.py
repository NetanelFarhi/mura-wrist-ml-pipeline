"""Build the concise DS23 MURA XR_WRIST Charter submission DOCX.

The source-of-truth prose lives in ``CHARTER.md``.  This script produces the
two-page submission format without embedding any MURA images or protected data.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUTPUT = Path("docs/submissions/MURA_Agentic_AI_Capstone_Charter_final.docx")
BLACK = RGBColor(0, 0, 0)
WHITE = "FFFFFF"
DARK_BLUE = "1F4E79"
PALE_BLUE = "EAF2F8"
PALE_GRAY = "F5F7F9"


def shade(cell, fill: str) -> None:
    cell_properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    cell_properties.append(shading)


def set_cell_border(cell, color: str = "D9D9D9") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top: int = 75, start: int = 95, bottom: int = 75, end: int = 95) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_run(paragraph, text: str, *, bold: bool = False, size: float = 9.4) -> None:
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK


def add_field(document: Document, number: str, heading: str, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(3.5)
    paragraph.paragraph_format.space_after = Pt(1.2)
    paragraph.paragraph_format.line_spacing = 1.0
    add_run(paragraph, f"{number}. {heading}", bold=True, size=10.0)
    body = document.add_paragraph()
    body.paragraph_format.space_after = Pt(1.6)
    body.paragraph_format.line_spacing = 1.02
    add_run(body, text, size=9.35)


def add_compact_table(document: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = False
    table.style = "Table Grid"
    header = table.rows[0].cells
    for index, label in enumerate(headers):
        header[index].width = Inches(widths[index])
        shade(header[index], DARK_BLUE)
        set_cell_border(header[index])
        set_cell_margins(header[index])
        header[index].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        paragraph = header[index].paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(label)
        run.bold = True
        run.font.name = "Arial"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor.from_string(WHITE)
    for row_number, values in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].width = Inches(widths[index])
            shade(cells[index], PALE_BLUE if row_number % 2 else "FFFFFF")
            set_cell_border(cells[index])
            set_cell_margins(cells[index])
            cells[index].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cells[index].paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            if index == 0:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_run(paragraph, value, size=8.55)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.48)
    section.bottom_margin = Inches(0.48)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    section.header_distance = Inches(0.18)
    section.footer_distance = Inches(0.18)

    normal = document.styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    normal.font.size = Pt(9.35)
    normal.font.color.rgb = BLACK

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(1)
    title_run = title.add_run("DS23 Project Charter  MURA XR WRIST Decision Support")
    title_run.font.name = "Arial"
    title_run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    title_run.font.size = Pt(15.2)
    title_run.font.bold = True
    title_run.font.color.rgb = BLACK
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(5)
    add_run(subtitle, "Team 1  Netanel  Hagit  Hodaya  Artium", size=9.4)

    add_field(
        document,
        "1",
        "Problem and user",
        "The intended user is an Emergency Department physician reviewing wrist radiographs before specialist radiology review is available. The project studies decision support for the MURA study-level NORMAL or ABNORMAL label. It does not claim acute-fracture diagnosis, lesion localization, autonomous diagnosis, treatment advice, or a measurable reduction in return visits.",
    )
    add_field(
        document,
        "2",
        "Success metric",
        "The pre-specified operating target is study-level sensitivity >=95%. For each model, the highest threshold that attains the target is selected on aggregated internal-validation study probabilities only, then frozen. We report specificity, false-negative rate (1 - sensitivity), precision, F1, AUROC, AUPRC, confusion matrix, and 95% patient-cluster bootstrap intervals. This is a project target, not a claim of final untouched-test performance.",
    )
    add_field(
        document,
        "3",
        "Data and Plan B",
        "Plan A uses Stanford MURA v1.1 XR_WRIST, organised as patient -> study -> one or more views. Official MURA train is split at patient level into internal train and validation. Logistic Regression C is selected by patient-grouped CV within train; thresholds are selected only on internal validation. Official MURA valid was already opened once for a frozen Phase-1 reference and cannot be called untouched for future selection. Raw images and protected local data never enter Git. Plan B, FracAtlas, changes the task to fracture detection and requires revised labels, anatomy/QC rules, and evaluation.",
    )
    add_field(
        document,
        "4",
        "Architecture sketch",
        "Data -> patient / study / views -> image model -> mean study-level aggregation -> versioned probability and evidence -> deterministic safety and routing -> routine human workflow or ambiguity investigation -> bounded adaptive agent -> structured human handoff. The mandatory baseline is frozen ImageNet ResNet-18 embeddings -> mean study embedding -> Logistic Regression, compared fairly with fine-tuned ResNet-18 on the same development cohort and metrics. The deterministic path works without an agent.",
    )
    add_field(
        document,
        "5",
        "Agent seat",
        "Hard rules remain deterministic and non-overridable: supported anatomy, corrupt or missing input, mandatory human review, and the frozen operating policy. AI may raise priority but never reduce standard review. A bounded agent is proposed only for genuine ambiguity: conflicting view evidence, borderline score with borderline QC, or missing/conflicting tool output. It may call approved QC, support, model, disagreement, uncertainty, diagnostic Grad-CAM, model-card, and calculator tools, then return NEEDS_HUMAN_REVIEW or UNSUPPORTED. It may not diagnose, treat, alter thresholds, bypass guardrails, or claim localization. Rule routing, fixed workflow, and the adaptive agent will be compared on the same independently authored safety cases.",
    )

    risk_heading = document.add_paragraph()
    risk_heading.paragraph_format.space_before = Pt(3.5)
    risk_heading.paragraph_format.space_after = Pt(2)
    add_run(risk_heading, "6. Risks and cut line", bold=True, size=10.0)
    add_compact_table(
        document,
        ["Risk", "Mitigation"],
        [
            ["MURA abnormality is not acute fracture", "Maintain scoped wording; no fracture claims; only defensible error review."],
            ["Markers, text, casts, hardware, borders, or artifacts", "Targeted shortcut/error checks; Grad-CAM is diagnostic, not localization ground truth."],
            ["Observed official-valid reference and limited sample size", "No later selection on official valid; bootstrap intervals and transparent reuse limitation."],
            ["Agent as an if/else wrapper or hallucination source", "Non-overridable deterministic guardrails, typed tools, hostile-input tests, independent evaluation."],
        ],
        [2.05, 5.25],
    )
    cut = document.add_paragraph()
    cut.paragraph_format.space_after = Pt(1.5)
    cut.paragraph_format.line_spacing = 1.0
    add_run(cut, "Cut line: ", bold=True, size=9.35)
    add_run(cut, "a reproducible study-level wrist-abnormality classifier with the mandatory baseline, patient-safe protocol, high-sensitivity policy, bootstrap/error/shortcut analysis, deterministic safety, and a separately evaluated bounded agent design. Stronger models, learned aggregation, calibration, Grad-CAM, and UI are optional.", size=9.35)

    milestone_heading = document.add_paragraph()
    milestone_heading.paragraph_format.space_before = Pt(3.5)
    milestone_heading.paragraph_format.space_after = Pt(2)
    add_run(milestone_heading, "7. Milestones and hats", bold=True, size=10.0)
    add_compact_table(
        document,
        ["Stage", "Deliverable", "Owner / hat"],
        [
            ["1", "Thin end-to-end baseline: patient-safe split, aggregation, validation-only threshold, baseline comparison.", "Hagit - Data\nHodaya - Model"],
            ["2", "Agent integration plus safety/evaluation suite; deterministic workflow remains operational.", "Netanel - Agent"],
            ["Final", "Error/shortcut analysis, governance, rule-vs-agent comparison, demo/docs, future cost estimate.", "Artium - Product"],
        ],
        [0.75, 4.95, 1.6],
    )

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.space_before = Pt(0)
    add_run(footer, "DS23  |  MURA XR WRIST  |  Research prototype only", size=7.7)

    properties = document.core_properties
    properties.title = "DS23 Project Charter MURA XR WRIST Decision Support"
    properties.author = "Team 1"
    properties.subject = "DS23 project charter"
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
