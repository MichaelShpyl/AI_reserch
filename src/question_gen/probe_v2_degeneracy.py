"""Probe the v2 fine-tune for degeneracy BEFORE trusting any discrimination score.

The v1 lesson: a high metric score meant nothing because the questions were degenerate multiple-choice
stems. So the first thing to check for v2 is not its discrimination but whether its questions are
well-formed. This loads the v2 adapter, generates questions for a fixed set of real claims, and runs
the same transparent degeneracy rule used in the audit. It makes no quality claim; it only reports how
many questions are usable, with every question printed for inspection.

    python src/question_gen/probe_v2_degeneracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ADAPTER_V1 = REPO / "models" / "qg_finetune_qwen3b"
ADAPTER_V2 = REPO / "models" / "qg_finetune_qwen3b_v2"
CLAIMS_SRC = REPO / "outputs" / "likeforlike_4way.json"
OUT = REPO / "outputs" / "qg_v2_probe.json"
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "evaluation"))


def fixed_claims(n=12):
    es = json.loads(CLAIMS_SRC.read_text(encoding="utf-8"))["essays"]
    out = []
    for eid, e in es.items():
        for c in e.get("claims", []):
            out.append({"essay": eid, "claim": c["claim"], "source": c["source"]})
            if len(out) >= n:
                return out
    return out


def probe(adapter, claims, label):
    from hf_backend import HFBackend
    from generate_questions import questions_for_claim
    from qg_quality_audit import is_degenerate
    be = HFBackend(adapter=str(adapter))
    rows = []
    for c in claims:
        try:
            qs = questions_for_claim(c["claim"], c["source"], be, k=3)
        except Exception as ex:
            print(f"  gen failed: {type(ex).__name__}", flush=True)
            qs = []
        for q in qs:
            rows.append({"q": q, "degenerate": bool(is_degenerate(q))})
    be.close()
    n = len(rows)
    deg = sum(r["degenerate"] for r in rows)
    print(f"\n=== {label}: {deg}/{n} degenerate ({100*deg/max(n,1):.0f}%) ===", flush=True)
    for r in rows:
        print(f"  [{'DEGEN' if r['degenerate'] else '  ok '}] {r['q'][:110]}", flush=True)
    return {"label": label, "n": n, "n_degenerate": deg,
            "pct_degenerate": round(100 * deg / max(n, 1), 1), "questions": rows}


def main() -> int:
    if not ADAPTER_V2.exists():
        raise SystemExit(f"No v2 adapter at {ADAPTER_V2}; run finetune_qg_v2.py first.")
    claims = fixed_claims(12)
    print(f"Probing on {len(claims)} fixed claims.", flush=True)
    result = {"n_claims": len(claims), "v2": probe(ADAPTER_V2, claims, "v2 (SQuAD)")}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT.relative_to(REPO)}")
    v2 = result["v2"]
    print(f"\nVERDICT: v2 degeneracy {v2['pct_degenerate']}% "
          f"(v1 was 95%). {'FIXED' if v2['pct_degenerate'] < 20 else 'STILL DEGENERATE'} "
          "on this transparent rule; inspect the questions above for quality before any score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
