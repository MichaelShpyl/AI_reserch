"""Normalise essay text so the detector learns writing style, not corpus markup.

This exists because of an artifact found during the detector audit (2026-06-16):
the human BAWE plain-text export keeps structural tags such as <heading>, <fnote>,
<list>, <figure> and <quote>. About 88% of the sampled human essays carry at least
one, and no AI essay does. Left in, a detector can separate the classes on the tag
string alone, which is corpus formatting and not writing style. The AI side has the
mirror problem: Llama sometimes emits markdown (**bold**, ## headings, "* " bullets,
numbered lists) that human plain text never contains.

`normalize_text` strips BOTH families of markup and flattens layout (newlines and
runs of whitespace collapse to single spaces). After it runs, the two classes differ
only in language, which is what we actually want to measure. We keep the inner text of
tags (so <heading>1. Introduction</heading> becomes "1. Introduction"): headings and
footnotes are part of how a person wrote, only the tag markers are an export artifact.

    from text_normalize import normalize_text
    clean = normalize_text(raw)
"""

from __future__ import annotations

import re

# A real BAWE/HTML-style tag: "<" or "</" then a letter, then word chars, then
# anything up to ">". Requiring a letter right after the bracket means prose like
# "a < b" or a statistic "p < 0.05" is left untouched (no letter follows the "<").
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*[^>]*>")
_MD_BOLD = re.compile(r"\*\*|__|`+")
_MD_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MD_LIST = re.compile(r"(?m)^\s{0,3}(?:[-*+]|\d{1,3}[.)])\s+")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return the essay with corpus/markdown markup removed and whitespace flattened."""
    if not text:
        return ""
    t = _TAG.sub(" ", text)          # strip BAWE export tags, keep their inner text
    t = _MD_HEADING.sub("", t)       # drop "## " / "### " markdown heading markers
    t = _MD_LIST.sub("", t)          # drop leading "- ", "* ", "1. " list markers
    t = _MD_BOLD.sub("", t)          # drop ** __ ` emphasis characters
    t = _WS.sub(" ", t).strip()      # collapse all whitespace and newlines to spaces
    return t


def has_markup(text: str) -> bool:
    """True if the text still contains a BAWE/HTML-style tag (audit helper)."""
    return bool(_TAG.search(text or ""))


if __name__ == "__main__":  # tiny smoke test
    human = "<heading>1. Introduction</heading>For a new business, a plan is an outline."
    ai = "**Business Description**\n\n* Basic: $0.99 per track\n1. **Track Sales**: revenue."
    print("HUMAN raw :", repr(human))
    print("HUMAN norm:", repr(normalize_text(human)))
    print("AI    raw :", repr(ai))
    print("AI    norm:", repr(normalize_text(ai)))
    print("math kept :", repr(normalize_text("the result held when p < 0.05 and x < y here")))
