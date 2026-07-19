"""v4 evaluation: the same fixed 18 claims and scorer as v1, v2 and v3, plus a uniqueness check.

Thin wrapper over eval_qg_v3 with the v4 adapter and output patched in. Afterwards it renames
the result key, computes question uniqueness (the failure mode this prompt invites is mode
collapse into stock phrasings, which the well-formedness gate cannot see), and prints the
comparison against v3 and the generic baseline.

    python src/question_gen/eval_qg_v4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "question_gen"))

import eval_qg_v3 as base  # noqa: E402

base.ADAPTER = REPO / "models" / "qg_finetune_qwen3b_v4"
base.OUT = REPO / "outputs" / "qg_v4_eval.json"


def postprocess() -> None:
    out = REPO / "outputs" / "qg_v4_eval.json"
    d = json.loads(out.read_text(encoding="utf-8"))
    if "v3" in d and "v4" not in d:
        d["v4"] = d.pop("v3")
    qs = []
    for c in d.get("per_claim", []):
        qs.extend(c.get("questions", []))
    if not qs:
        for c in d["v4"].get("per_claim", []):
            qs.extend(c.get("questions", []) if isinstance(c, dict) else [])
    if qs:
        norm = [" ".join(q.lower().split()) for q in qs]
        d["uniqueness"] = {"n_questions": len(qs),
                           "n_unique": len(set(norm)),
                           "unique_ratio": round(len(set(norm)) / len(qs), 3)}
    v3 = json.loads((REPO / "outputs" / "qg_v3_eval.json").read_text(encoding="utf-8"))
    d["reference"] = {"v3_mean": v3["v3"]["mean_discrimination"],
                      "generic_baseline": 0.30}
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print("v4:", d["v4"].get("mean_discrimination"),
          "| v3:", d["reference"]["v3_mean"],
          "| generic baseline: 0.30",
          "| uniqueness:", d.get("uniqueness"))


if __name__ == "__main__":
    rc = base.main()
    postprocess()
    raise SystemExit(rc)
