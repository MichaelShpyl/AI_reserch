"""Evaluate the v2 (SQuAD) fine-tune properly: is it both well-formed AND useful?

The v1 lesson was that a discrimination score is meaningless if the questions are degenerate, so this
does the two checks in order. On the SAME 18 claims the v1 isolated eval used (reused from
`outputs/finetune_eval.json`), it generates questions with the base Qwen 3B and the v2 adapter, using
the cleaned extractor, then (1) audits each question for degeneracy with the same transparent rule as
`qg_quality_audit.py`, and (2) scores discrimination with the same simulation. Reporting both together
is the point: v1 scored high but was 95% degenerate; v2 must be judged on being usable first.

VRAM is 8 GB, so models never co-reside: base and v2 generate in turn (freed between), then Ollama
scores.

    python src/question_gen/eval_qg_v2.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE_ADAPTER = None
V2_ADAPTER = REPO / "models" / "qg_finetune_qwen3b_v2"
CLAIMS = REPO / "outputs" / "finetune_eval.json"
OUT = REPO / "outputs" / "qg_v2_eval.json"
OLLAMA_EXE = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "evaluation"))
sys.path.insert(0, str(REPO / "src" / "detection"))


def free_ollama():
    for m in ("llama3.1:8b", "nomic-embed-text"):
        subprocess.run([str(OLLAMA_EXE), "stop", m], capture_output=True)
    time.sleep(3)


def generate(adapter, claims):
    from hf_backend import HFBackend
    from generate_questions import questions_for_claim
    be = HFBackend(adapter=adapter)
    per_claim = []
    for i, c in enumerate(claims, 1):
        try:
            qs = questions_for_claim(c["claim"], c["source"], be, k=3)
        except Exception as ex:
            print(f"    claim {i}: gen failed {type(ex).__name__}", flush=True)
            qs = []
        per_claim.append(qs)
        print(f"    [{i}/{len(claims)}] {len(qs)} q: {qs[0][:70] if qs else '(none)'}", flush=True)
    be.close()
    return per_claim


def score(claims, per_claim):
    from discrimination_sim import discrimination
    out = []
    for c, qs in zip(claims, per_claim):
        ds = []
        for q in qs:
            try:
                ds.append(discrimination(q, c["source"])["discrimination"])
            except Exception as ex:
                print(f"    score failed: {type(ex).__name__}", flush=True)
        out.append(ds)
    return out


def boot_ci(vals, n=5000):
    vals = np.array([v for ds in vals for v in ds]) if vals and isinstance(vals[0], list) else np.array(vals)
    if len(vals) == 0:
        return None, [None, None]
    rng = np.random.default_rng(42)
    ms = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return round(float(vals.mean()), 4), [round(float(np.percentile(ms, 2.5)), 4),
                                          round(float(np.percentile(ms, 97.5)), 4)]


def summarise(label, per_claim_qs, per_claim_disc):
    from qg_quality_audit import is_degenerate
    qs = [q for c in per_claim_qs for q in c]
    ds = [d for c in per_claim_disc for d in c]
    deg = sum(1 for q in qs if is_degenerate(q))
    m, ci = boot_ci([ds]) if ds else (None, [None, None])
    return {"label": label, "n_questions": len(qs), "n_degenerate": deg,
            "pct_degenerate": round(100 * deg / max(len(qs), 1), 1),
            "mean_discrimination": m, "ci95": ci,
            "questions": qs}


def main() -> int:
    if not V2_ADAPTER.exists():
        raise SystemExit(f"No v2 adapter at {V2_ADAPTER}")
    claims = json.loads(CLAIMS.read_text(encoding="utf-8"))["claim_set"]
    print(f"== {len(claims)} fixed claims (reused from finetune_eval.json) ==", flush=True)
    free_ollama()

    print("== base Qwen 3B ==", flush=True)
    base_qs = generate(BASE_ADAPTER, claims)
    print("== v2 (SQuAD) Qwen 3B ==", flush=True)
    v2_qs = generate(str(V2_ADAPTER), claims)

    print("== score both (Ollama) ==", flush=True)
    base_d = score(claims, base_qs)
    v2_d = score(claims, v2_qs)
    free_ollama()

    from scipy import stats
    base = summarise("base 3B", base_qs, base_d)
    v2 = summarise("v2 (SQuAD) 3B", v2_qs, v2_d)
    flat_b = [d for c in base_d for d in c]
    flat_v = [d for c in v2_d for d in c]
    result = {"n_claims": len(claims), "base": base, "v2": v2,
              "v1_for_reference": {"note": "from finetune_eval.json / qg_quality_audit.json",
                                   "mean_discrimination": 0.154, "pct_degenerate": 95.2}}
    if flat_b and flat_v:
        result["mannwhitney_v2_vs_base_p"] = round(
            float(stats.mannwhitneyu(flat_v, flat_b, alternative="two-sided").pvalue), 4)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== v2 EVALUATION ===")
    for r in (base, v2):
        print(f"{r['label']:15s} n={r['n_questions']:3d}  degenerate {r['pct_degenerate']:5.1f}%  "
              f"disc {r['mean_discrimination']} CI {r['ci95']}")
    print(f"v1 (reference): 0.154 discrimination but 95.2% degenerate (artifact)")
    print(f"Mann-Whitney v2 vs base p = {result.get('mannwhitney_v2_vs_base_p')}")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
