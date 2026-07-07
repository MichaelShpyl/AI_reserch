"""Concept figures for the graded intermediate presentation (general audience).

Three plain-language diagrams so someone who knows nothing about the project can follow it:
  fig_pipeline_overview.png     - what the whole system does, end to end
  fig_discrimination_explainer  - how we test a question without real students
  fig_roadmap.png               - the plan, showing the project is about half done

    python dissertation/presentation/make_concept_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGS = Path(__file__).resolve().parent.parent / "figures"
TEAL = "#2b6777"
ORANGE = "#d98e3b"
INK = "#222831"
GREY = "#52616b"
LIGHT = "#e8eef0"
RUST = "#a63d2e"


def box(ax, x, y, w, h, title, body, fc="white", ec=TEAL, tc=TEAL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=11.5,
            fontweight="bold", color=tc, zorder=3)
    ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=9.2,
            color=INK, zorder=3, wrap=True)


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2.0, color=color, zorder=1))


def pipeline_overview():
    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 5.0); ax.axis("off")

    # top row: submission -> detect -> explain
    box(ax, 0.2, 3.2, 2.5, 1.5, "A student essay",
        "The work handed in", fc=LIGHT, ec=GREY, tc=INK)
    box(ax, 3.3, 3.2, 2.8, 1.5, "1. Detect",
        "Is this likely\nwritten by AI?")
    box(ax, 6.7, 3.2, 2.8, 1.5, "2. Explain",
        "In plain words,\nwhy it was flagged")
    box(ax, 10.1, 3.2, 2.3, 1.5, "and test it",
        "Prove the reason\nis the real one", fc=LIGHT, ec=GREY, tc=INK)
    arrow(ax, 2.7, 3.95, 3.3, 3.95)
    arrow(ax, 6.1, 3.95, 6.7, 3.95)
    arrow(ax, 9.5, 3.95, 10.1, 3.95)

    # down to bottom row
    arrow(ax, 11.25, 3.2, 11.25, 2.7)
    arrow(ax, 11.25, 2.55, 9.7, 2.05)

    # bottom row (right to left): find claims -> ask questions -> guide
    box(ax, 6.9, 0.6, 2.8, 1.5, "3. Find the claims",
        "What the student\nargued, their words", ec=ORANGE, tc=ORANGE)
    box(ax, 3.5, 0.6, 2.8, 1.5, "4. Ask questions",
        "Only the real author\ncould answer well", ec=ORANGE, tc=ORANGE)
    box(ax, 0.2, 0.6, 2.9, 1.5, "Interview guide",
        "For the lecturer,\nto talk it through", fc="#eaf3ef", ec=TEAL, tc=TEAL)
    arrow(ax, 6.9, 1.35, 6.3, 1.35, color=ORANGE)
    arrow(ax, 3.5, 1.35, 3.1, 1.35, color=ORANGE)

    ax.text(6.3, 2.55, "A flag opens a conversation, never an accusation.",
            ha="center", va="center", fontsize=11.5, fontstyle="italic", color=RUST)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pipeline_overview.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_pipeline_overview.png")


def discrimination_explainer():
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.text(6, 4.9, "How do we test a question without real students?",
            ha="center", fontsize=14, fontweight="bold", color=INK)

    box(ax, 4.1, 3.4, 3.8, 1.1,
        "The question", "\"Why did you choose that\nexample as evidence?\"", ec=INK, tc=INK)

    # left: AI that read the essay
    box(ax, 0.5, 1.0, 3.6, 1.5, "AI that read the essay",
        "Answers well:\nit knows the specifics", ec=TEAL, tc=TEAL, fc="#eaf3ef")
    # right: AI that did not
    box(ax, 7.9, 1.0, 3.6, 1.5, "AI that did NOT read it",
        "Can only guess\nfrom general knowledge", ec=GREY, tc=GREY, fc=LIGHT)

    arrow(ax, 5.0, 3.4, 2.3, 2.5, color=TEAL)
    arrow(ax, 7.0, 3.4, 9.7, 2.5, color=GREY)

    ax.text(6, 0.45, "Big gap between the two answers  =  a good verification question "
            "(you really needed the essay).",
            ha="center", fontsize=11.5, fontweight="bold", color=RUST)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_discrimination_explainer.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_discrimination_explainer.png")


def roadmap():
    fig, ax = plt.subplots(figsize=(12.4, 5.0))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 5.0); ax.axis("off")

    # timeline bar
    ax.add_patch(FancyBboxPatch((0.5, 3.7), 6.0, 0.5, boxstyle="round,pad=0.0,rounding_size=0.1",
                                facecolor=TEAL, edgecolor="none", zorder=2))
    ax.add_patch(FancyBboxPatch((6.5, 3.7), 5.4, 0.5, boxstyle="round,pad=0.0,rounding_size=0.1",
                                facecolor=LIGHT, edgecolor=GREY, linewidth=1, zorder=2))
    ax.text(3.5, 3.95, "BUILT SO FAR  (June to mid-July)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white", zorder=3)
    ax.text(9.2, 3.95, "SECOND HALF  (mid-July to end of August)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK, zorder=3)
    ax.plot([6.5, 6.5], [3.4, 4.5], color=RUST, lw=2.5, zorder=4)
    ax.text(6.5, 4.62, "we are here", ha="center", fontsize=10, color=RUST, fontweight="bold")

    done = ["Matched essay dataset (1,280)", "Hybrid AI-text detector", "Explainability + honesty test",
            "Claim extractor + Bloom classifier", "Question generation, both backends",
            "The evaluation and the fine-tune story", "The lecturer's guide, end to end"]
    todo = ["Make the explanations clear and visual", "Validate with real student answers",
            "Widen the data (more AI models, domains)", "Push the local model further",
            "Fix the weak spots (see next slide)", "Discussion, polish, submit"]

    ax.text(0.5, 3.25, "Done", fontsize=11, fontweight="bold", color=TEAL)
    for i, d in enumerate(done):
        ax.text(0.7, 2.85 - i * 0.36, f"✓  {d}", fontsize=9.6, color=INK)
    ax.text(6.7, 3.25, "Still to do", fontsize=11, fontweight="bold", color=RUST)
    for i, d in enumerate(todo):
        ax.text(6.9, 2.85 - i * 0.36, f"•  {d}", fontsize=9.6, color=INK)

    fig.tight_layout()
    fig.savefig(FIGS / "fig_roadmap.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_roadmap.png")


if __name__ == "__main__":
    pipeline_overview()
    discrimination_explainer()
    roadmap()
