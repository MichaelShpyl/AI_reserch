"""Build the reproducibility figure for the presentation.

Two panels side by side. On the left, the page of the dissertation where a result is discussed,
with the file paths showing as links. On the right, the file one of those links opens, on GitHub.
The point of the slide is that the two are one click apart, so the figure has to show both halves
rather than assert the connection.

Inputs are produced by other scripts, so run those first:
  - dissertation/Dissertation_Shpyl_progress_draft.pdf   (dissertation/docgen/build_dissertation.js, then export)
  - a GitHub screenshot of outputs/detector_metrics_clean.json

    python dissertation/presentation/make_repro_figure.py <github_screenshot.png>

Writes dissertation/figures/fig_repro_link.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "dissertation" / "figures"
PDF = ROOT / "dissertation" / "Dissertation_Shpyl_progress_draft.pdf"

PAPER = (250, 249, 246)
INK = (34, 40, 49)
GREY = (110, 122, 132)
TEAL = (31, 78, 95)
LINE = (214, 214, 208)

PANEL_H = 1500          # both panels are scaled to this height, so they sit level
GAP = 90
PAD = 36
LABEL_H = 58


def font(size, bold=False):
    for name in (("calibrib.ttf", "calibri.ttf") if bold else ("calibri.ttf", "calibrib.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def find_page(needle: str) -> int:
    """The section moves as the document grows, so locate it by its text rather than by number."""
    doc = fitz.open(PDF)
    for i in range(doc.page_count):
        if needle in (doc[i].get_text() or ""):
            return i
    raise SystemExit(f"could not find {needle!r} in {PDF.name}")


def dissertation_panel() -> Image.Image:
    """The section itself, not the whole page. On a projector the difference is whether the reader
    can see that the paths are links or only that there is grey text somewhere."""
    heading = "The code, and how to read this document beside it"
    page_no = find_page("how to read this document beside it")
    doc = fitz.open(PDF)
    page = doc[page_no]
    hits = page.search_for(heading)
    dpi = 260
    scale = dpi / 72.0
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    w, h = img.size
    # Start a little above the heading if it is on this page, otherwise at the top of the text block.
    top = int(hits[0].y0 * scale) - 30 if hits else int(h * 0.07)
    return img.crop((int(w * 0.09), max(0, top), int(w * 0.93), int(h * 0.925)))


def crop_github(path: Path) -> Image.Image:
    """Keep the repository header and the file itself, drop the empty space underneath."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # The file panel ends well before the bottom of a full-page capture. Find the last row that is
    # not the page background, so the crop follows the content rather than a guessed fraction.
    px = img.load()
    bg = px[w - 5, h - 5]
    bottom = h
    for y in range(h - 1, 0, -8):
        row = [px[x, y] for x in range(0, w, 40)]
        if any(sum(abs(a - b) for a, b in zip(c, bg)) > 40 for c in row):
            bottom = min(h, y + 40)
            break
    # Drop the file-tree sidebar and GitHub's own navigation. The sidebar repeats what the tree
    # already showed and costs half the width, which on a projector is half the readable size of
    # the file. The repository name goes in the panel label instead, where it can be set at a size
    # that actually reads from the back of a room.
    return img.crop((int(w * 0.275), int(h * 0.019), w, bottom))


def scaled(img: Image.Image, height: int) -> Image.Image:
    w = round(img.width * height / img.height)
    return img.resize((w, height), Image.LANCZOS)


def panel(img: Image.Image, label: str, height: int) -> Image.Image:
    body = scaled(img, height)
    out = Image.new("RGB", (body.width + 2 * 4, height + LABEL_H + 8), PAPER)
    d = ImageDraw.Draw(out)
    d.text((4, 6), label.upper(), font=font(30, bold=True), fill=TEAL)
    out.paste(body, (4, LABEL_H))
    d.rectangle([4, LABEL_H, 4 + body.width - 1, LABEL_H + height - 1], outline=LINE, width=3)
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    left = panel(dissertation_panel(), "In the dissertation, section 1.10", PANEL_H)
    right = panel(crop_github(Path(sys.argv[1])), "github.com/MichaelShpyl/AI_reserch, public", PANEL_H)

    W = PAD * 2 + left.width + GAP + right.width
    H = PAD * 2 + max(left.height, right.height)
    out = Image.new("RGB", (W, H), PAPER)
    out.paste(left, (PAD, PAD))
    out.paste(right, (PAD + left.width + GAP, PAD))

    # An arrow between the panels, so the relationship reads without the caption.
    d = ImageDraw.Draw(out)
    ax = PAD + left.width + GAP // 2
    ay = PAD + LABEL_H + PANEL_H // 2
    d.line([ax - 26, ay, ax + 18, ay], fill=TEAL, width=9)
    d.polygon([(ax + 34, ay), (ax + 12, ay - 17), (ax + 12, ay + 17)], fill=TEAL)

    FIGS.mkdir(parents=True, exist_ok=True)
    dest = FIGS / "fig_repro_link.png"
    out.save(dest, optimize=True)
    print(f"wrote {dest}  {out.width}x{out.height}  ratio {out.height / out.width:.2f}")


if __name__ == "__main__":
    main()
