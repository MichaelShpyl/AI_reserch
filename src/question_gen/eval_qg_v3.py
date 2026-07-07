"""Evaluate the v3 (self-distilled verification) fine-tune: well-formed AND discriminative?

Same protocol as eval_qg_v2.py so the three fine-tunes are directly comparable: the fixed 18 claims,
base against adapter, degeneracy audited before any score, discrimination scored by the same
simulation. Base results are reused from the v2 evaluation (same claims, same models, same seed
policy) so this run only generates with v3 and scores it, then assembles the full base/v1/v2/v3
picture in one figure.

VRAM: the v3 model generates first (HF), then Ollama scores; never co-resident.

    python src/question_gen/eval_qg_v3.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ADAPTER = REPO / "models" / "qg_finetune_qwen3b_v3"
CLAIMS = REPO / "outputs" / "finetune_eval.json"
V2_EVAL = REPO / "outputs" / "qg_v2_eval.json"
OUT = REPO / "outputs" / "qg_v3_eval.json"
FIGS = REPO / "dissertation" / "figures"
OLLAMA_EXE = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "evaluation"))
sys.path.insert(0, str(REPO / "src" / "detection"))


def free_ollama():
    for m in ("llama3.1:8b", "nomic-embed-text"):
        subprocess.run([str(OLLAMA_EXE), "stop", m], capture_output=True)
    time.sleep(3)


def boot_ci(vals, n=5000):
    vals = np.array(vals)
    if len(vals) == 0:
        return None, [None, None]
    rng = np.random.default_rng(42)
    ms = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return round(float(vals.mean()), 4), [round(float(np.percentile(ms, 2.5)), 4),
                                          round(float(np.percentile(ms, 97.5)), 4)]


def main() -> int:
    from wellformed import is_degenerate
    if not ADAPTER.exists():
        raise SystemExit(f"No v3 adapter at {ADAPTER}; run finetune_qg_v3.py first.")
    claims = json.loads(CLAIMS.read_text(encoding="utf-8"))["claim_set"]
    print(f"== {len(claims)} fixed claims (same set as v1/v2 evals) ==", flush=True)
    free_ollama()

    # Generate with v3 (HF), then free it before Ollama loads.
    from hf_backend import HFBackend
    from generate_questions import questions_for_claim
    be = HFBackend(adapter=str(ADAPTER))
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

    # Audit BEFORE scoring (the standing rule since the v1 artifact).
    flat_q = [q for qs in per_claim for q in qs]
    deg = sum(1 for q in flat_q if is_degenerate(q))
    pct_deg = round(100 * deg / max(len(flat_q), 1), 1)
    print(f"degeneracy audit: {deg}/{len(flat_q)} ({pct_deg}%)", flush=True)

    # Score with the discrimination simulation.
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

    # Assemble the full picture: base and v2 from the saved v2 eval, v1 from its audit numbers.
    v2 = json.loads(V2_EVAL.read_text(encoding="utf-8"))
    from scipy import stats
    base_flat = None
    result = {
        "n_claims": len(claims),
        "v3": {"label": "v3 (verification) 3B", "n_questions": len(discs),
               "n_degenerate": deg, "pct_degenerate": pct_deg,
               "mean_discrimination": m, "ci95": ci, "questions": flat_q},
        "base": {k: v2["base"][k] for k in ("label", "n_questions", "pct_degenerate",
                                            "mean_discrimination", "ci95")},
        "v2": {k: v2["v2"][k] for k in ("label", "n_questions", "pct_degenerate",
                                        "mean_discrimination", "ci95")},
        "v1_for_reference": {"mean_discrimination": 0.154, "pct_degenerate": 95.2,
                             "note": "artifact; see qg_quality_audit.json"},
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(result)
    print("\n=== v3 EVALUATION ===")
    print(f"base 0.027 | v1 0.154 (95% degenerate, artifact) | v2 0.102 | "
          f"v3 {m} CI {ci} ({pct_deg}% degenerate, n={len(discs)})")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["base 3B", "v1: EduQG", "v2: SQuAD", "v3: verification"]
    means = [result["base"]["mean_discrimination"], 0.154,
             result["v2"]["mean_discrimination"], result["v3"]["mean_discrimination"]]
    cis = [result["base"]["ci95"], [0.085, 0.227], result["v2"]["ci95"], result["v3"]["ci95"]]
    deg = [result["base"]["pct_degenerate"], 95.2,
           result["v2"]["pct_degenerate"], result["v3"]["pct_degenerate"]]
    colors = ["#8a8a8a", "#c0503a", "#9bb7bd", "#2b6777"]
    err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    bars = ax.bar(labels, means, yerr=err, capsize=6, width=0.6, color=colors,
                  error_kw={"ecolor": "#222831"})
    bars[1].set_hatch("////"); bars[1].set_edgecolor("#7a2718"); bars[1].set_linewidth(1.2)
    gb = REPO / "outputs" / "generic_baseline_batch.json"
    if gb.exists():
        g = json.loads(gb.read_text(encoding="utf-8")).get("pooled", {}).get("mean")
        if g is not None:
            ax.axhline(g, color="#a63d2e", ls="--", lw=1.1, label=f"generic baseline ({g:.2f})")
            ax.legend(fontsize=9, frameon=False, loc="upper left")
    for b, c, d in zip(bars, cis, deg):
        tag = f"{d:.0f}% degenerate" + ("\n(artifact)" if d > 50 else "")
        ax.text(b.get_x() + b.get_width() / 2, c[1] + 0.008, tag, ha="center", va="bottom",
                fontsize=8.5, color=("#7a2718" if d > 50 else "#52616B"))
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylim(0, max(0.32, max(c[1] for c in cis) + 0.05))
    ax.set_ylabel("mean discrimination (aware - blind)")
    ax.set_title("The data-format experiment, complete: multiple-choice, factual,\n"
                 "and verification-style training data on the same 18 claims",
                 fontsize=11.5, fontweight="bold", color="#222831")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_qg_v3_eval.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_qg_v3_eval.png")


if __name__ == "__main__":
    raise SystemExit(main())
