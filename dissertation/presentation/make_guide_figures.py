"""Render the Verification Interview Guide figures from the guide PDFs themselves.

Two figures come out of here:
  fig_guide_page1.png   page one of the flagged guide, the dissertation's Figure 7.5
  fig_guide_pages.png   the flagged guide beside the not-flagged one, for the contrast

Both are rasterised from the real generated PDFs rather than drawn, so they cannot drift from
what the pipeline actually produces. They did drift once: the guide was restyled on 4 August and
these figures kept showing the old, hard-to-read typography for a month.

    python dissertation/presentation/make_guide_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
GUIDES = REPO / "outputs" / "verification_guides"
FIGS = REPO / "dissertation" / "figures"

INK = "#222831"
TEAL = "#2b6777"
RUST = "#a63d2e"
GREY = "#52616b"


def page_image(pdf_name: str, page: int = 0, dpi: int = 200):
    import fitz
    doc = fitz.open(str(GUIDES / pdf_name))
    pix = doc[page].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype="uint8").reshape(pix.height, pix.width, pix.n)
    doc.close()
    return img


def crop_to_content(img, pad: int = 40):
    """Trim the empty part of the page. The guide's first page is little over half full, and the
    blank half was costing the figure most of its readable size on the printed page."""
    grey = img[:, :, :3].mean(axis=2)
    ink = grey < 245
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return img
    r0, r1 = max(0, rows[0] - pad), min(img.shape[0], rows[-1] + pad)
    c0, c1 = max(0, cols[0] - pad), min(img.shape[1], cols[-1] + pad)
    return img[r0:r1, c0:c1]


def fig_guide_page1():
    """Page one on its own, large enough to actually read in the printed dissertation."""
    img = crop_to_content(page_image("3108a_ai_guide.pdf", 0, dpi=220))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * h / w))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#d8dcdf"); s.set_linewidth(1.2)
    fig.tight_layout(pad=0.3)
    out = FIGS / "fig_guide_page1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out.name)


def fig_guide_pages():
    """The flagged guide beside the not-flagged one: same topic, two outcomes."""
    panels = [("3108a_ai_guide.pdf", "The AI-written version", "FLAGGED  (score 0.96)", RUST),
              ("3108a_human_guide.pdf", "The real student's essay", "NOT FLAGGED  (score 0.02)", TEAL)]
    imgs = [crop_to_content(page_image(pdf, 0, dpi=150)) for pdf, *_ in panels]
    # Pad both crops to a common shape, otherwise the two panels sit at different heights and the
    # titles no longer line up.
    H = max(i.shape[0] for i in imgs)
    W = max(i.shape[1] for i in imgs)
    padded = []
    for i in imgs:
        canvas = np.full((H, W, i.shape[2]), 255, dtype="uint8")
        canvas[:i.shape[0], :i.shape[1]] = i
        padded.append(canvas)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    for ax, img, (pdf, title, verdict, colr) in zip(axes, padded, panels):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(colr); s.set_linewidth(3)
        ax.set_title(f"{title}\n{verdict}", fontsize=13.5, color=colr, fontweight="bold", pad=10)
    fig.text(0.5, -0.015,
             "Both pages are real output from the system, generated for the same essay topic: one "
             "guide opens a verification conversation, the other says no interview is needed.",
             ha="center", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    out = FIGS / "fig_guide_pages.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out.name)


if __name__ == "__main__":
    fig_guide_page1()
    fig_guide_pages()
