"""Three self-explanatory figures for the graded deck, built from real project output.

  fig_problem.png          what lecturers get today: an indefensible number, and biased
  fig_guide_pages.png      page one of the two REAL guides side by side (AI twin vs human twin)
  fig_degenerate_demo.png  the score said / the text said, with the real broken and real fixed question

    python dissertation/presentation/make_demo_figures.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIGS = HERE.parent / "figures"
GUIDES = REPO / "outputs" / "verification_guides"

TEAL, INK, GREY, RUST, ORANGE = "#2b6777", "#222831", "#52616b", "#a63d2e", "#d98e3b"
BAND = "#e3e7ea"


def card(ax, x, y, w, h, color, lw=2.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc="white", ec=color, lw=lw, zorder=2))


def fig_problem():
    fig, (l, r) = plt.subplots(1, 2, figsize=(12.4, 5.2), width_ratios=[1.15, 1])
    for ax in (l, r):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    l.text(0.5, 0.96, "What a lecturer gets today", ha="center", fontsize=15,
           fontweight="bold", color=INK)
    card(l, 0.16, 0.52, 0.68, 0.30, INK)
    l.text(0.5, 0.72, "82% AI", ha="center", fontsize=34, fontweight="bold", color=INK)
    l.text(0.5, 0.585, "(no reasons given)", ha="center", fontsize=11, color=GREY)
    for i, q in enumerate(["Why was it flagged?", "Which parts?",
                           "What do I say to the student?"]):
        y = 0.40 - i * 0.115
        l.text(0.13, y, q, fontsize=12.5, color=INK)
        l.text(0.87, y, "no answer", fontsize=12.5, color=RUST, ha="right",
               fontweight="bold", style="italic")

    r.text(0.5, 0.96, "And the tools are biased", ha="center", fontsize=15,
           fontweight="bold", color=INK)
    bars = [("Students writing in\na second language", 0.61, RUST),
            ("Native\nwriters", 0.05, TEAL)]
    for i, (lab, v, c) in enumerate(bars):
        x = 0.22 + i * 0.38
        r.add_patch(plt.Rectangle((x, 0.22), 0.18, v * 0.85, fc=c, zorder=2))
        r.text(x + 0.09, 0.22 + v * 0.85 + 0.03, f"{v:.0%}", ha="center", fontsize=16,
               fontweight="bold", color=c)
        r.text(x + 0.09, 0.145, lab, ha="center", fontsize=10.5, color=INK)
    r.text(0.5, 0.045, "essays falsely flagged as AI (Liang et al., 2023)", ha="center",
           fontsize=10.5, color=GREY)

    fig.text(0.5, -0.03, "A number nobody can defend, biased against the wrong students, "
             "answering how the text was made instead of whether the student understands it.",
             ha="center", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_problem.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig_problem.png")


def fig_guide_pages():
    import fitz
    panels = [("3108a_ai_guide.pdf", "The AI-written version", "FLAGGED  (score 0.96)", RUST),
              ("3108a_human_guide.pdf", "The real student's essay", "NOT FLAGGED  (score 0.02)", TEAL)]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4))
    for ax, (pdf, title, verdict, colr) in zip(axes, panels):
        doc = fitz.open(str(GUIDES / pdf))
        pix = doc[0].get_pixmap(dpi=110)
        import numpy as np
        img = np.frombuffer(pix.samples, dtype="uint8").reshape(pix.height, pix.width, pix.n)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(colr); s.set_linewidth(3)
        ax.set_title(f"{title}\n{verdict}", fontsize=13.5, color=colr, fontweight="bold", pad=10)
        doc.close()
    fig.text(0.5, -0.015, "Both pages are real output from the system, generated for the same essay "
             "topic: one guide opens a verification conversation, the other says no interview is needed.",
             ha="center", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_guide_pages.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig_guide_pages.png")


def fig_degenerate_demo():
    four = json.loads((REPO / "outputs" / "likeforlike_4way.json").read_text(encoding="utf-8"))
    v1 = four["result"]["arms"]["ft3b"]["pooled_mean"]
    base = four["result"]["arms"]["base3b"]["pooled_mean"]

    fig, (l, r) = plt.subplots(1, 2, figsize=(12.4, 5.0), width_ratios=[0.75, 1.25])
    l.axis("off"); l.set_xlim(0, 1); l.set_ylim(0, 1)
    r.axis("off"); r.set_xlim(0, 1); r.set_ylim(0, 1)

    l.text(0.5, 0.96, "The score said", ha="center", fontsize=15, fontweight="bold", color=INK)
    for i, (lab, v, c) in enumerate([("fine-tune v1", v1, RUST), ("its base model", base, GREY)]):
        x = 0.18 + i * 0.38
        l.add_patch(plt.Rectangle((x, 0.24), 0.2, v * 3.6, fc=c, zorder=2))
        l.text(x + 0.10, 0.24 + v * 3.6 + 0.03, f"{v:.3f}", ha="center", fontsize=15,
               fontweight="bold", color=c)
        l.text(x + 0.10, 0.16, lab, ha="center", fontsize=11, color=INK)
    l.text(0.5, 0.02, '"the fine-tune is 4x better!"', ha="center", fontsize=12,
           color=RUST, style="italic")

    r.text(0.5, 0.96, "The text said", ha="center", fontsize=15, fontweight="bold", color=INK)
    card(r, 0.05, 0.56, 0.9, 0.3, RUST)
    r.text(0.5, 0.76, '"Which of the following is correct?"', ha="center", fontsize=14,
           color=INK, style="italic")
    r.text(0.5, 0.645, "95% of the fine-tune's output: no options, nothing to answer.\n"
           "This empty stem alone scores 0.44, above every real question writer.",
           ha="center", fontsize=10, color=RUST)
    card(r, 0.05, 0.10, 0.9, 0.34, TEAL)
    real_q = ("How did you decide to use the phrase 'challenging decisions made by "
              "public authorities' rather than just 'challenging decisions'?")
    r.text(0.5, 0.30, textwrap.fill(f'"{real_q}"', 62), ha="center", fontsize=11,
           color=INK, style="italic")
    r.text(0.5, 0.135, "after retraining on the right data: a real question about the "
           "student's own choice", ha="center", fontsize=10, color=TEAL)

    fig.text(0.5, -0.03, "The high score was measuring emptiness. Reading the questions caught it; "
             "no number in this project is trusted until the text behind it has been read.",
             ha="center", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_degenerate_demo.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig_degenerate_demo.png")


if __name__ == "__main__":
    fig_problem()
    fig_guide_pages()
    fig_degenerate_demo()


def fig_dataset_pipeline():
    """How the dataset was built: five stages, each one removing a shortcut."""
    fig = plt.figure(figsize=(12.6, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    stages = [
        ("Real essays", ["BAWE corpus", "2,761 assignments by", "UK students, four", "discipline groups"], GREY),
        ("1. Balanced sample", ["640 essays: 80 from", "each of 8 cells", "(4 disciplines x native", "/ non-native), capped", "per student"], TEAL),
        ("2. AI twin for each", ["Local Llama 3.1 gets", "the topic keywords", "and target length,", "writes a matched twin", "(640 AI essays)"], ORANGE),
        ("3. Clean both sides", ["Same normalisation", "for both classes;", "hidden format tags", "removed; 1,280", "essays total"], TEAL),
        ("4. Split by student", ["394 students; no", "writer appears in two", "splits, so the model", "cannot memorise", "a person"], GREY),
    ]
    w, gap = 0.165, 0.028
    x0 = (1 - (len(stages) * w + (len(stages) - 1) * gap)) / 2
    for i, (head, body, c) in enumerate(stages):
        x = x0 + i * (w + gap)
        card(ax, x, 0.28, w, 0.42, c)
        ax.text(x + w / 2, 0.78, head, ha="center", fontsize=12, fontweight="bold", color=c)
        ax.text(x + w / 2, 0.49, chr(10).join(body), ha="center", va="center", fontsize=9.2, color=INK)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.004, 0.49), xytext=(x + w + 0.004, 0.49),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
    ax.text(0.5, 0.115, "Each step removes a shortcut the detector could cheat with:",
            ha="center", fontsize=12.5, color=INK, fontweight="bold")
    ax.text(0.5, 0.045, "balancing removes topic bias, matching removes length, cleaning removes "
            "formatting tells, student-level splits remove memorising a writer.",
            ha="center", fontsize=11.5, color=INK)
    fig.savefig(FIGS / "fig_dataset_pipeline.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("fig_dataset_pipeline.png")
