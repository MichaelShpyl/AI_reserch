"""Evaluate the QLoRA fine-tune: does fine-tuning the local model improve its verification questions?

Isolates the fine-tuning effect. On a FIXED set of claims (extracted once with the local model), the
base Qwen 3B and the fine-tuned Qwen 3B each generate verification questions, and both sets are scored
by the same discrimination simulation. Only the question-writing model differs, so any gap is the
fine-tune.

VRAM is 8 GB, so the phases are sequenced and never co-resident: extract claims with Ollama, free it,
generate with each HF model in turn (freeing between), then score with Ollama.

    python src/question_gen/eval_finetune.py --essays 6 --claims 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
BATCH = REPO / "outputs" / "backend_comparison_batch.json"
ADAPTER = REPO / "models" / "qg_finetune_qwen3b"
OUT = REPO / "outputs" / "finetune_eval.json"
FIGS = REPO / "dissertation" / "figures"
OLLAMA_EXE = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "evaluation"))
sys.path.insert(0, str(REPO / "src" / "detection"))


def free_ollama():
    for m in ("llama3.1:8b", "nomic-embed-text"):
        subprocess.run([str(OLLAMA_EXE), "stop", m], capture_output=True)
    time.sleep(3)


def build_claim_set(essay_ids, n_claims):
    from generate_questions import OllamaBackend, sentences, extract_claims
    from text_normalize import normalize_text
    be = OllamaBackend("llama3.1:8b")
    claim_set = []
    for eid in essay_ids:
        text = (AI_DIR / f"{eid}.txt").read_text(encoding="utf-8", errors="ignore")
        sents = sentences(normalize_text(text))
        for c in extract_claims(sents, be, n_claims):
            src = " ".join(s["text"] for s in c["source_sentences"])
            claim_set.append({"essay": eid, "claim": c["claim"], "source": src})
        print(f"  {eid}: cumulative claims {len(claim_set)}", flush=True)
    return claim_set


def generate_with(backend, claim_set):
    from generate_questions import questions_for_claim
    out = []
    for i, c in enumerate(claim_set, 1):
        try:
            qs = questions_for_claim(c["claim"], c["source"], backend, k=3)
        except Exception as e:
            print(f"    claim {i}: gen failed {type(e).__name__}", flush=True)
            qs = []
        out.append(qs)
        print(f"    [{i}/{len(claim_set)}] {len(qs)} questions", flush=True)
    return out


def score(claim_set, qs_per_claim):
    from discrimination_sim import discrimination
    discs = []
    for c, qs in zip(claim_set, qs_per_claim):
        for q in qs:
            discs.append(discrimination(q, c["source"])["discrimination"])
    return discs


def boot_ci(vals, n=5000):
    vals = np.array(vals)
    rng = np.random.default_rng(42)
    ms = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return round(float(vals.mean()), 4), [round(float(np.percentile(ms, 2.5)), 4),
                                          round(float(np.percentile(ms, 97.5)), 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--essays", type=int, default=6)
    ap.add_argument("--claims", type=int, default=3)
    args = ap.parse_args()
    if not ADAPTER.exists():
        raise SystemExit(f"No fine-tuned adapter at {ADAPTER}; run finetune_qg.py first.")

    ids = (json.loads(BATCH.read_text(encoding="utf-8")).get("balanced", {}).get("essays")
           or ["3108a"])[:args.essays]

    # Phase A: claims with Ollama, then free it.
    print("== Phase A: extract claim set (Ollama) ==", flush=True)
    claim_set = build_claim_set(ids, args.claims)
    OUT.write_text(json.dumps({"claim_set": claim_set}, indent=2), encoding="utf-8")
    free_ollama()

    from hf_backend import HFBackend
    # Phase B: base model generation.
    print("== Phase B: base Qwen 3B generation ==", flush=True)
    base = HFBackend(adapter=None)
    base_qs = generate_with(base, claim_set)
    base.close()

    # Phase C: fine-tuned model generation.
    print("== Phase C: fine-tuned Qwen 3B generation ==", flush=True)
    ft = HFBackend(adapter=str(ADAPTER))
    ft_qs = generate_with(ft, claim_set)
    ft.close()

    # Phase D: score both with Ollama.
    print("== Phase D: score both (Ollama discrimination sim) ==", flush=True)
    base_d = score(claim_set, base_qs)
    ft_d = score(claim_set, ft_qs)
    bm, bci = boot_ci(base_d)
    fm, fci = boot_ci(ft_d)

    from scipy import stats
    result = {
        "adapter": str(ADAPTER.relative_to(REPO)),
        "n_essays": len(ids), "n_claims": len(claim_set),
        "base_qwen3b": {"n_questions": len(base_d), "mean_discrimination": bm, "ci95": bci},
        "finetuned_qwen3b": {"n_questions": len(ft_d), "mean_discrimination": fm, "ci95": fci},
        "reading": "Same claims, same discrimination scorer; only the question-writing model differs "
                   "(base vs QLoRA-fine-tuned on EduQG). Preliminary, pending 8B-to-3B scope sign-off.",
    }
    # Question-level paired test where both produced a question at the same slot is not well defined
    # (counts differ), so report an unpaired Mann-Whitney on the pooled per-question scores too.
    if base_d and ft_d:
        mw = stats.mannwhitneyu(ft_d, base_d, alternative="two-sided")
        result["mannwhitney_ft_vs_base_p"] = round(float(mw.pvalue), 4)
    OUT.write_text(json.dumps({"claim_set": claim_set, **result}, indent=2), encoding="utf-8")
    make_figure(result)
    print("\n=== FINE-TUNE EVALUATION ===")
    print(f"base       mean {bm} CI {bci} (n={len(base_d)})")
    print(f"fine-tuned mean {fm} CI {fci} (n={len(ft_d)})")
    print(f"Mann-Whitney p = {result.get('mannwhitney_ft_vs_base_p')}")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["base 3B", "fine-tuned 3B"]
    means = [result["base_qwen3b"]["mean_discrimination"],
             result["finetuned_qwen3b"]["mean_discrimination"]]
    cis = [result["base_qwen3b"]["ci95"], result["finetuned_qwen3b"]["ci95"]]
    err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.bar(labels, means, yerr=err, capsize=6, width=0.5,
           color=["#8a8a8a", "#2b6777"], error_kw={"ecolor": "#222831"})
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylabel("mean discrimination (aware - blind)")
    ax.set_title("Does QLoRA fine-tuning improve the local model's questions?",
                 fontsize=11.5, fontweight="bold", color="#222831")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_finetune_eval.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_finetune_eval.png")


if __name__ == "__main__":
    raise SystemExit(main())
