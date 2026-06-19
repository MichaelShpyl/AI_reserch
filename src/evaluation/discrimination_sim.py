"""Discrimination simulation: the project's primary, judge-free evaluation of question quality.

A good verification question is one that someone who understands the source can answer and someone
who does not cannot. We approximate that without humans or an LLM judge: for each question we have a
context-aware model answer it WITH the source passage, and a context-blind model answer it with the
question ONLY. We then measure how close each answer is to the source (embedding cosine similarity).
The discrimination score is the gap, aware minus blind: a high gap means the question genuinely needs
the source, a near-zero gap means it can be answered without understanding the specific text.

We compare our claim-grounded questions against a set of generic essay questions; the grounded
questions should discriminate more. Everything runs on the local model and embeddings via Ollama, so
no API key or judge is needed (CLAUDE.md: this is the main empirical evidence).

    python src/evaluation/discrimination_sim.py --guide 3108a_ai
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

REPO = Path(__file__).resolve().parents[2]
GUIDES = REPO / "outputs" / "verification_guides"
FIGS = REPO / "dissertation" / "figures"
OUT = REPO / "outputs" / "discrimination_sim.json"
OLLAMA = "http://localhost:11434"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

GENERIC = [
    "What is the main argument of your essay?",
    "Why did you choose to approach the topic this way?",
    "Explain your reasoning in your own words.",
    "What evidence most strongly supports your conclusion?",
    "Which part of your essay are you least sure about, and why?",
    "How would you summarise your essay to someone who has not read it?",
]


def answer(question: str, source: str | None) -> str:
    if source:
        system = ("Answer the lecturer's question about the student's essay using the provided source "
                  "passage. Be specific and concise, two or three sentences.")
        user = f"Source passage:\n{source}\n\nQuestion: {question}\n\nAnswer:"
    else:
        system = "Answer the question concisely, two or three sentences."
        user = f"Question: {question}\n\nAnswer:"
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def embed(text: str) -> np.ndarray:
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text}, timeout=120)
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def discrimination(question: str, source: str) -> dict:
    a_aware = answer(question, source)
    a_blind = answer(question, None)
    es, ea, eb = embed(source), embed(a_aware), embed(a_blind)
    sim_aware = float(ea @ es)
    sim_blind = float(eb @ es)
    return {"question": question, "sim_aware": round(sim_aware, 4),
            "sim_blind": round(sim_blind, 4), "discrimination": round(sim_aware - sim_blind, 4)}


def boot_ci(vals, n=5000):
    vals = np.array(vals)
    rng = np.random.default_rng(42)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return round(float(vals.mean()), 4), [round(float(np.percentile(means, 2.5)), 4),
                                          round(float(np.percentile(means, 97.5)), 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guide", default="3108a_ai", help="guide file stem in outputs/verification_guides")
    args = ap.parse_args()
    guide = json.loads((GUIDES / f"{args.guide}.json").read_text(encoding="utf-8"))

    whole_source = " ".join(s["text"] for c in guide["claims"] for s in c["source_sentences"])

    print("Scoring claim-grounded questions ...", flush=True)
    grounded = []
    for c in guide["claims"]:
        src = " ".join(s["text"] for s in c["source_sentences"])
        for q in c["questions"]:
            d = discrimination(q["question"], src)
            d["bloom"] = q["bloom_level"]
            grounded.append(d)
            print(f"  disc {d['discrimination']:+.3f}  {q['question'][:70]}", flush=True)

    print("Scoring generic baseline questions ...", flush=True)
    generic = []
    for q in GENERIC:
        d = discrimination(q, whole_source)
        generic.append(d)
        print(f"  disc {d['discrimination']:+.3f}  {q[:70]}", flush=True)

    g_mean, g_ci = boot_ci([d["discrimination"] for d in grounded])
    b_mean, b_ci = boot_ci([d["discrimination"] for d in generic])
    report = {
        "guide": args.guide, "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL,
        "n_grounded": len(grounded), "n_generic": len(generic),
        "grounded": {"mean_discrimination": g_mean, "ci95": g_ci,
                     "mean_sim_aware": round(float(np.mean([d["sim_aware"] for d in grounded])), 4),
                     "mean_sim_blind": round(float(np.mean([d["sim_blind"] for d in grounded])), 4)},
        "generic": {"mean_discrimination": b_mean, "ci95": b_ci},
        "questions": {"grounded": grounded, "generic": generic},
        "reading": "Discrimination = similarity(answer-with-source, source) minus "
                   "similarity(answer-without-source, source). Higher means the question needs the "
                   "source. Claim-grounded questions should discriminate more than generic ones.",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_figure(grounded, generic, g_mean, g_ci, b_mean, b_ci)

    print("\n=== DISCRIMINATION SIMULATION ===")
    print(f"grounded questions (n={len(grounded)}): mean {g_mean}  CI {g_ci}")
    print(f"generic questions  (n={len(generic)}): mean {b_mean}  CI {b_ci}")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(grounded, generic, g_mean, g_ci, b_mean, b_ci) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    # Left: distribution of per-question discrimination.
    gd = [d["discrimination"] for d in grounded]
    bd = [d["discrimination"] for d in generic]
    ax1.scatter(np.random.default_rng(1).normal(0, 0.04, len(gd)), gd, color="#2b6777",
                alpha=0.7, label="claim-grounded")
    ax1.scatter(np.random.default_rng(2).normal(1, 0.04, len(bd)), bd, color="#d98e3b",
                alpha=0.7, label="generic")
    ax1.axhline(0, color="#888", lw=1, ls="--")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["claim-grounded", "generic"])
    ax1.set_ylabel("discrimination (aware - blind)")
    ax1.set_title("Per-question discrimination", fontsize=11, fontweight="bold", color="#222831")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    # Right: mean with CI.
    ax2.bar(["claim-\ngrounded", "generic"], [g_mean, b_mean], color=["#2b6777", "#d98e3b"],
            width=0.55, yerr=[[g_mean - g_ci[0], b_mean - b_ci[0]], [g_ci[1] - g_mean, b_ci[1] - b_mean]],
            capsize=5, error_kw={"ecolor": "#222831"})
    ax2.axhline(0, color="#888", lw=1, ls="--")
    ax2.set_ylabel("mean discrimination")
    ax2.set_title("Mean (95% CI)", fontsize=11, fontweight="bold", color="#222831")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig.suptitle("Discrimination simulation: claim-grounded vs generic verification questions",
                 fontsize=12.5, fontweight="bold", color="#222831")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "fig_discrimination_sim.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_discrimination_sim.png")


if __name__ == "__main__":
    raise SystemExit(main())
