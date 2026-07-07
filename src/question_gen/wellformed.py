"""Well-formedness gate for generated verification questions.

The quality audit (src/evaluation/qg_quality_audit.py) showed the discrimination simulation can be
gamed by degenerate questions: a contentless multiple-choice stem ("Which of the following is
correct?") out-scores every real question because it gives the source-aware and source-blind
answerers nothing to anchor on. The lesson written into Chapter 8 is that no question should reach
either the lecturer's guide or a trusted score without a well-formedness check. This module is that
check, one transparent rule shared by the audit, the production pipeline, and the evaluations.

A question is degenerate if it is a multiple-choice stem (no options are ever supplied in this
pipeline), leaked JSON, or a contentless "which is correct" fragment. Production (build_guide)
filters these and records how many were dropped; evaluations keep the raw output and report the
degeneracy rate alongside any score, so a degenerate generator is caught, not hidden.
"""

from __future__ import annotations

import re

DEGEN_PATTERNS = [
    r"which of the following",
    r"^\s*\{?\"?questions\"?\s*:",              # raw JSON leaked into the text
    r"^which (statement|option|answer|case)\b",
    r"^what is the (correct|right) answer",
    r"^(choose|select) the\b",
    r"^true or false",
]


def is_degenerate(q: str) -> bool:
    ql = (q or "").lower().strip()
    if any(re.search(p, ql) for p in DEGEN_PATTERNS):
        return True
    words = re.findall(r"[a-z]+", ql)
    if len(words) <= 6 and any(w in ql for w in ("following", "correct", "statement")):
        return True
    return False


def well_formed(q: str) -> bool:
    """A question the pipeline is willing to show a lecturer: non-degenerate, a real sentence,
    ends with a question mark."""
    q = (q or "").strip()
    return bool(q) and q.endswith("?") and len(q.split()) >= 4 and not is_degenerate(q)
