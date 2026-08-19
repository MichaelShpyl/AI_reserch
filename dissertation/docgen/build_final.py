"""Build the dissertation to PDF with a correct, populated table of contents.

A Word TableOfContents field only fills in when a person opens the file and presses F9. Nobody
presses F9 on a PDF, so the submitted document had a contents page whose entire content was the
instruction to generate a contents page. The build now writes a literal contents list instead, and
that list needs real page numbers, which only exist once the document has been rendered.

So the build runs twice. The first pass writes the contents with placeholder numbers, renders, and
records where each heading actually landed. The second pass rewrites the contents with those
numbers. The two passes stay in step because the entry text is identical in both and the number
sits on a right-aligned tab at the margin, so changing its width cannot re-wrap a line and move the
pagination underneath it. The script checks that assumption rather than trusting it: it compares
every printed page number against the rendered document's own outline and fails if any disagree.

    python dissertation/docgen/build_final.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DISS = HERE.parent
REPO = DISS.parent
DOCX = DISS / "Dissertation_Shpyl_progress_draft.docx"
PDF = DISS / "Dissertation_Shpyl_progress_draft.pdf"
PAGES = HERE / "toc_pages.json"
SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")


def build_docx() -> None:
    r = subprocess.run(["node", str(HERE / "build_dissertation.js")],
                       cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"docx build failed:\n{r.stdout}\n{r.stderr}")
    for line in r.stdout.strip().split("\n"):
        if "Repository links" in line or "Not linked" in line:
            print("   " + line.strip())


def to_pdf() -> None:
    if not SOFFICE.exists():
        sys.exit(f"LibreOffice not found at {SOFFICE}")
    r = subprocess.run([str(SOFFICE), "--headless", "--convert-to", "pdf",
                        "--outdir", str(DISS), str(DOCX)], capture_output=True, text=True)
    if not PDF.exists():
        sys.exit(f"pdf conversion failed:\n{r.stdout}\n{r.stderr}")


def outline_pages() -> dict[str, int]:
    import fitz
    d = fitz.open(str(PDF))
    pages: dict[str, int] = {}
    for level, title, page in d.get_toc():
        if level <= 2:
            pages.setdefault(title.strip(), page)
    d.close()
    return pages


def printed_contents() -> dict[str, int]:
    """What the contents pages actually claim, read back off the rendered PDF."""
    import fitz
    d = fitz.open(str(PDF))
    claimed: dict[str, int] = {}
    for i in range(min(14, d.page_count)):
        for line in d[i].get_text().split("\n"):
            # The dot leader can end with a space before the number, depending on where the tab
            # lands, so the gap is optional. An earlier version of this regex required the
            # digits to touch the dots and reported three perfectly good entries as missing.
            m = re.match(r"(.+?)\.{3,}\s*(\d+)$", line.strip())
            if m:
                claimed.setdefault(m.group(1).strip(), int(m.group(2)))
    d.close()
    return claimed


def main() -> None:
    print("pass 1: build with placeholder page numbers")
    PAGES.unlink(missing_ok=True)
    build_docx()
    to_pdf()
    pages = outline_pages()
    PAGES.write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"   captured {len(pages)} heading positions")

    print("pass 2: rebuild with the real page numbers")
    build_docx()
    to_pdf()

    import fitz
    d = fitz.open(str(PDF))
    n_pages = d.page_count
    d.close()

    actual, claimed = outline_pages(), printed_contents()
    wrong = {k: (v, actual.get(k)) for k, v in claimed.items() if actual.get(k) != v}
    print(f"\n{n_pages} pages, {len(claimed)} contents entries")
    if wrong:
        print(f"   {len(wrong)} WRONG page numbers:")
        for k, (c, a) in list(wrong.items())[:10]:
            print(f"      '{k[:60]}' says {c}, actually {a}")
        sys.exit(1)
    print("   every printed page number matches the document. Contents is correct.")

    missing = [k for k in actual if k not in claimed]
    if missing:
        print(f"   headings not listed in the contents: {missing}")


if __name__ == "__main__":
    main()
