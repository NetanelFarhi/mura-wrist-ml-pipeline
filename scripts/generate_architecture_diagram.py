"""Generate a safe, diagram-only MURA wrist system architecture figure.

The figure intentionally uses no MURA radiographs, patient data, or image
thumbnails so it can be committed to the repository under the data-governance
policy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT = Path("docs/assets/MURA_System_Architecture.png")


def add_box(ax, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.3,
        edgecolor="#334155",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.4, color="#0f172a", wrap=True)


def arrow(ax, start: tuple[float, float], end: tuple[float, float], text: str = "") -> None:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#475569"},
    )
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, text, ha="center", va="bottom", fontsize=7, color="#475569")


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 7.1), dpi=180)
    figure.patch.set_facecolor("white")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")

    axis.text(0.5, 0.95, "MURA Wrist Decision-Support Architecture", ha="center", va="center", fontsize=16, fontweight="bold", color="#0f172a")
    axis.text(0.5, 0.915, "Study-level abnormality support with deterministic safety rules and bounded ambiguity investigation", ha="center", va="center", fontsize=9.5, color="#475569")

    add_box(axis, 0.05, 0.67, 0.17, 0.14, "Input study\n1-n wrist views\n+ metadata", "#dbeafe")
    add_box(axis, 0.28, 0.67, 0.17, 0.14, "Study organization\npatient -> study -> views", "#e0f2fe")
    add_box(axis, 0.51, 0.67, 0.17, 0.14, "Model layer\nview evidence ->\nstudy aggregation", "#dcfce7")
    add_box(axis, 0.74, 0.67, 0.20, 0.14, "Validated probability\n+ evidence / uncertainty\nwhen available", "#fef3c7")
    arrow(axis, (0.22, 0.74), (0.28, 0.74))
    arrow(axis, (0.45, 0.74), (0.51, 0.74))
    arrow(axis, (0.68, 0.74), (0.74, 0.74))

    add_box(axis, 0.33, 0.41, 0.34, 0.15, "Deterministic safety and routing policy\nSupported anatomy | input QC | mandatory human review\nAI may raise priority; it cannot reduce standard review", "#fee2e2")
    arrow(axis, (0.84, 0.67), (0.58, 0.56), "hard rules")

    add_box(axis, 0.05, 0.14, 0.24, 0.16, "Straightforward / known case\nDeterministic workflow recommendation\n-> human clinician", "#e2e8f0")
    add_box(axis, 0.38, 0.14, 0.24, 0.16, "Ambiguous case only\nBounded adaptive agent\nselects permitted evidence tools", "#ede9fe")
    add_box(axis, 0.71, 0.14, 0.24, 0.16, "Structured evidence summary\nNEEDS_HUMAN_REVIEW or UNSUPPORTED\n-> human clinician", "#e2e8f0")
    arrow(axis, (0.47, 0.41), (0.17, 0.30), "known path")
    arrow(axis, (0.55, 0.41), (0.50, 0.30), "conflict / missing evidence")
    arrow(axis, (0.62, 0.22), (0.71, 0.22), "grounded summary")

    axis.text(0.5, 0.055, "The deterministic system works without the agent. The agent cannot override safety policy or make an autonomous diagnosis.", ha="center", va="center", fontsize=9, color="#334155", fontweight="bold")
    figure.tight_layout(pad=0.7)
    figure.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
