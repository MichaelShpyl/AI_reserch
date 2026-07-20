"""v4 with the content gate at inference: the one engineering step Section 8.13 names, measured.

Ungated v4 leaks claim content in 6 of 54 questions because the gate only shaped its training
data. This run adds the same gate at generation time: a question that names claim content is
rejected and regenerated, first at the default temperature, then at a higher one for variety.
Everything else matches the ungated evaluation (same adapter, same production prompt, same fixed
18 claims, same scorer), so the difference measures the gate alone: uniformity, score, and the
retry cost a deployment would pay.

    python src/question_gen/eval_qg_v4_gated.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ADAPTER = REPO / "models" / "qg_finetune_qwen3b_v4"
CLAIMS = REPO / "outputs" / "finetune_eval.json"
OUT = REPO / "outputs" / "qg_v4_gated_eval.json"
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "evaluation"))
sys.path.insert(0, str(REPO / "src" / "detection"))

MAX_ROUNDS = 4
RETRY_TEMP = 0.7


def main() -> int:
    from eval_qg_v3 import boot_ci, free_ollama
    from build_v4_dataset import content_free
    from wellformed import is_degenerate, well_formed
    from generate_questions import questions_for_claim
    from hf_backend import HFBackend

    claims = json.loads(CLAIMS.read_text(encoding="utf-8"))["claim_set"]
    print(f"== gated v4 on {len(claims)} fixed claims ==", flush=True)
    free_ollama()

    be = HFBackend(adapter=str(ADAPTER))
    per_claim, gate_stats = [], {"attempt_rounds": 0, "rejected": 0, "fallback_leaky": 0}
    for i, c in enumerate(claims, 1):
        clean: list[str] = []
        leaky: list[str] = []
        for rnd in range(MAX_ROUNDS):
            if len(clean) >= 3:
                break
            be.temperature = 0.2 if rnd == 0 else RETRY_TEMP
            gate_stats["attempt_rounds"] += 1
            try:
                cands = questions_for_claim(c["claim"], c["source"], be, k=3)
            except Exception as ex:
                print(f"    claim {i} round {rnd}: gen failed {type(ex).__name__}", flush=True)
                continue
            for q in cands:
                if not well_formed(q) or q in clean:
                    continue
                if content_free(q, c["claim"], c["source"]):
                    clean.append(q)
                else:
                    gate_stats["rejected"] += 1
                    leaky.append(q)
        qs = clean[:3]
        while len(qs) < 3 and leaky:
            qs.append(leaky.pop(0))
            gate_stats["fallback_leaky"] += 1
        per_claim.append(qs)
        print(f"    [{i}/{len(claims)}] {len(qs)} q "
              f"({len([q for q in qs if content_free(q, c['claim'], c['source'])])} clean)",
              flush=True)
    be.close()

    flat_q = [q for qs in per_claim for q in qs]
    deg = sum(1 for q in flat_q if is_degenerate(q))
    clean_n = sum(1 for c, qs in zip(claims, per_claim)
                  for q in qs if content_free(q, c["claim"], c["source"]))
    norm = [" ".join(q.lower().split()) for q in flat_q]
    print(f"audit: {deg} degenerate, {clean_n}/{len(flat_q)} content-free, "
          f"{len(set(norm))}/{len(flat_q)} unique", flush=True)

    print("== scoring (Ollama) ==", flush=True)
    from discrimination_sim import discrimination
    discs = []
    for c, qs in zip(claims, per_claim):
        for q in qs:
            try:
                discs.append(discrimination(q, c["source"])["discrimination"])
            except Exception as ex:
                print(f"    score failed: {type(ex).__name__}", flush=True)
    free_ollama()
    m, ci = boot_ci(discs)

    ungated = json.loads((REPO / "outputs" / "qg_v4_eval.json").read_text(encoding="utf-8"))
    result = {
        "label": "v4 with content gate at inference",
        "n_questions": len(discs), "n_degenerate": deg,
        "pct_degenerate": round(100 * deg / max(len(flat_q), 1), 1),
        "content_free": {"n_clean": clean_n, "n_total": len(flat_q),
                         "clean_ratio": round(clean_n / max(len(flat_q), 1), 3)},
        "uniqueness": {"n_unique": len(set(norm)), "n_total": len(flat_q)},
        "gate_stats": gate_stats,
        "mean_discrimination": m, "ci95": ci,
        "questions": flat_q,
        "ungated_reference": {"mean": ungated["v4"]["mean_discrimination"],
                              "content_free_ratio":
                                  ungated["v4"]["content_free_at_inference"]["clean_ratio"]},
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== GATED v4 ===")
    print(f"gated {m} CI {ci} | ungated {result['ungated_reference']['mean']} | "
          f"content-free {result['content_free']['clean_ratio']:.0%} "
          f"(ungated {result['ungated_reference']['content_free_ratio']:.0%}) | "
          f"gate: {gate_stats}")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
