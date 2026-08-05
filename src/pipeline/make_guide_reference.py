"""Build the reference .docx that styles the Verification Interview Guide.

Pandoc copies its styles from a reference document. Without one it falls back to its defaults,
which set the body in a serif and render bold as a heavy slab face that is genuinely hard to read
on screen. Since a lecturer reads this document in a meeting, often on a laptop between classes,
legibility is a product requirement rather than a nicety.

This writes src/pipeline/guide_reference.docx. assemble_guide.py passes it to pandoc with
--reference-doc. Regenerate it only if the house style changes:

    python src/pipeline/make_guide_reference.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.shared import Pt, RGBColor

OUT = Path(__file__).resolve().parent / "guide_reference.docx"
PANDOC = Path.home() / "AppData" / "Local" / "Pandoc" / "pandoc.exe"

BODY_FONT = "Calibri"
INK = RGBColor(0x22, 0x28, 0x31)
TEAL = RGBColor(0x1F, 0x4E, 0x5F)
MUTED = RGBColor(0x52, 0x61, 0x6B)


def set_font(style, *, name=BODY_FONT, size=11, bold=False, italic=False, color=INK):
    f = style.font
    f.name = name
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    # Word keeps a separate east-asian font slot; without this it can substitute a different face.
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for attr in ("ascii", "hAnsi", "cs", "eastAsia"):
        rfonts.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}" + attr, name
        )


def space(style, *, before=0, after=8, line=1.15):
    p = style.paragraph_format
    p.space_before = Pt(before)
    p.space_after = Pt(after)
    p.line_spacing = line
    p.line_spacing_rule = WD_LINE_SPACING.MULTIPLE


def main() -> int:
    # Start from pandoc's OWN default reference document rather than a blank one. A blank document
    # has no numbering definitions, so restyling it silently stripped the bullets off every list.
    base = OUT.with_name("_pandoc_default.docx")
    if PANDOC.exists():
        with open(base, "wb") as fh:
            subprocess.run([str(PANDOC), "--print-default-data-file", "reference.docx"],
                           stdout=fh, check=True, timeout=60)
        d = Document(str(base))
    else:
        print("pandoc not found, falling back to a blank template (lists may lose their bullets)")
        d = Document()
    st = d.styles

    set_font(st["Normal"], size=11)
    space(st["Normal"], after=8, line=1.2)

    # Headings: same family as the body, distinguished by weight, size and colour rather than by
    # switching to a different typeface, which is what made the old output look mismatched.
    for name, size in (("Heading 1", 17), ("Heading 2", 13.5), ("Heading 3", 11.5)):
        if name in [s.name for s in st]:
            set_font(st[name], size=size, bold=True, color=TEAL)
            space(st[name], before=14 if name != "Heading 1" else 18, after=6, line=1.1)

    if "Title" in [s.name for s in st]:
        set_font(st["Title"], size=24, bold=True, color=TEAL)
        space(st["Title"], before=0, after=10, line=1.05)

    # Pandoc puts the opening framing paragraph in a block quote. It carries the sentence that says
    # this document is not a verdict, so it should read as a calm aside, not as fine print.
    for q in ("Quote", "Block Text", "Intense Quote"):
        if q in [s.name for s in st]:
            set_font(st[q], size=11, italic=True, color=MUTED)
            space(st[q], before=6, after=10, line=1.2)

    for lst in ("List Bullet", "List Number", "List Paragraph"):
        if lst in [s.name for s in st]:
            set_font(st[lst], size=11)
            space(st[lst], after=4, line=1.15)

    # Bold inside a paragraph is the main offender in the old output: pandoc's default Strong
    # resolves to a slab serif. Pin it to the body face.
    if "Strong" in [s.name for s in st]:
        set_font(st["Strong"], size=11, bold=True)
    if "Emphasis" in [s.name for s in st]:
        set_font(st["Emphasis"], size=11, italic=True, color=MUTED)

    # Captions sit under the explanation card image.
    if "Caption" in [s.name for s in st]:
        set_font(st["Caption"], size=9, italic=True, color=MUTED)
        space(st["Caption"], before=2, after=12, line=1.1)

    for sec in d.sections:
        sec.left_margin = sec.right_margin = Pt(54)   # 0.75 inch, more text per line
        sec.top_margin = sec.bottom_margin = Pt(54)

    d.save(OUT)
    if base.exists():
        base.unlink()
    print("Saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
