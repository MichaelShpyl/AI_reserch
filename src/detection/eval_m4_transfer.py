"""Zero-shot robustness test: apply the in-domain detector to the M4 benchmark.

The detector was trained on one domain (BAWE student essays) and one generator (Llama 3.1).
We test transfer with no adaptation, in two honest steps:

  A. Cross-generator, same domain type. M4's OUTFOX test split is essays written by humans
     versus six unseen generators (GPT-4, ChatGPT, Cohere, BLOOMz, Dolly, davinci). This asks:
     does an AI-style detector trained only on Llama generalise to other models on essays?

  B. Cross-domain. M4's monolingual data spans reddit, wikihow, arxiv, wikipedia and peerread,
     where the human text is nothing like a student essay. This is the real domain-shift test
     and is where a single-domain detector is expected to struggle.

Reporting both avoids the trap of calling the easy case (A) "robust". Compare each F1 to the
in-domain F1 (about 0.99).

    python src/detection/eval_m4_transfer.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from text_normalize import normalize_text

REPO = Path(__file__).resolve().parents[2]
M4 = REPO / "data" / "raw" / "m4"
MODELS = REPO / "models" / "detector"
FIGS = REPO / "dissertation" / "figures"
OUT = REPO / "outputs" / "m4_transfer.json"
IN_DOMAIN_F1 = 0.99
PER_GEN = 400        # machine essays per generator (cross-generator test)
PER_DOMAIN = 300     # per class, per domain (cross-domain test)
MAXLEN = 512
BATCH = 16
SEED = 42


def latest_checkpoint() -> Path:
    cps = glob.glob(str(MODELS / "checkpoint-*"))
    if not cps:
        raise SystemExit(f"No checkpoint under {MODELS}. Train the detector first.")
    return Path(max(cps, key=lambda p: int(p.split("-")[-1])))


def metrics(y, yp) -> dict:
    return {"accuracy": round(accuracy_score(y, yp), 4),
            "precision": round(precision_score(y, yp, zero_division=0), 4),
            "recall": round(recall_score(y, yp, zero_division=0), 4),
            "f1": round(f1_score(y, yp, zero_division=0), 4),
            "confusion_matrix_[hu,ai]x[pred_hu,pred_ai]": confusion_matrix(y, yp).tolist()}


def main() -> int:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = latest_checkpoint()
    print(f"Loading detector from {ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()

    def predict(texts):
        preds, probs = [], []
        texts = [normalize_text(t) for t in texts]
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAXLEN,
                          padding=True, return_tensors="pt").to(device)
                logits = model(**enc).logits
                probs.extend(torch.softmax(logits, -1)[:, 1].tolist())
                preds.extend(logits.argmax(-1).tolist())
        return np.array(preds), np.array(probs)

    # ---- Pass A: cross-generator on essays (OUTFOX test split) ----
    test = pd.read_parquet(M4 / "test.parquet")
    gens = [g for g in test["model"].unique() if g != "human"]
    parts = [test[test["model"] == g].sample(n=min(PER_GEN, (test["model"] == g).sum()),
                                              random_state=SEED) for g in gens]
    machine = pd.concat(parts)
    humans = test[test["label"] == 0].sample(n=len(machine), random_state=SEED)
    A = pd.concat([machine, humans]).reset_index(drop=True)
    print(f"[A] cross-generator (essays): {len(A)} essays, {len(gens)} generators", flush=True)
    A["pred"], _ = predict(A["text"].tolist())
    a_overall = metrics(A["label"].values, A["pred"].values)
    a_fpr = round(float((A[A.label == 0]["pred"] == 1).mean()), 4)
    a_pergen = {g: round(float((A[A.model == g]["pred"] == 1).mean()), 4) for g in gens}

    # ---- Pass B: cross-domain (reddit/wikihow/arxiv/wikipedia/peerread) ----
    train = pd.read_parquet(M4 / "train.parquet")
    domains = sorted(train["source"].dropna().unique())
    parts = []
    for d in domains:
        sd = train[train["source"] == d]
        for lab in (0, 1):
            s = sd[sd["label"] == lab]
            if len(s):
                parts.append(s.sample(n=min(PER_DOMAIN, len(s)), random_state=SEED))
    B = pd.concat(parts).reset_index(drop=True)
    print(f"[B] cross-domain: {len(B)} texts across {len(domains)} domains", flush=True)
    B["pred"], _ = predict(B["text"].tolist())
    b_overall = metrics(B["label"].values, B["pred"].values)
    per_domain = {}
    for d in domains:
        sd = B[B["source"] == d]
        hu, ai = sd[sd.label == 0], sd[sd.label == 1]
        per_domain[d] = {
            "accuracy": round(float((sd["pred"] == sd["label"]).mean()), 4),
            "human_false_positive_rate": round(float((hu["pred"] == 1).mean()), 4) if len(hu) else None,
            "machine_detection_rate": round(float((ai["pred"] == 1).mean()), 4) if len(ai) else None,
        }

    report = {
        "in_domain_f1": IN_DOMAIN_F1,
        "A_cross_generator_essays": {
            "n": int(len(A)), "overall": a_overall, "human_false_positive_rate": a_fpr,
            "per_generator_detection_rate": a_pergen,
            "note": "OUTFOX essays: human essays vs six unseen generators. Same domain type as "
                    "training (essays), so this isolates cross-generator transfer."},
        "B_cross_domain": {
            "n": int(len(B)), "overall": b_overall, "per_domain": per_domain,
            "note": "reddit/wikihow/arxiv/wikipedia/peerread: human web and academic text vs four "
                    "generators. The real domain-shift test."},
        "reading": "Compare each F1 to the in-domain F1 (~0.99). Strong cross-generator transfer "
                   "but weaker cross-domain transfer would mean the detector generalises across "
                   "models yet is sensitive to the kind of text, which is the honest limitation.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_figures(report)

    print("\n=== M4 ZERO-SHOT TRANSFER ===")
    print(f"in-domain F1 {IN_DOMAIN_F1}")
    print("A cross-generator (essays): F1", a_overall["f1"], "| human FPR", a_fpr)
    print("   per-generator detection:", a_pergen)
    print("B cross-domain: F1", b_overall["f1"])
    for d, m in per_domain.items():
        print(f"   {d:10s} acc {m['accuracy']}  humanFPR {m['human_false_positive_rate']}  "
              f"machineDet {m['machine_detection_rate']}")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figures(r: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    INK = "#222831"
    # Figure 1: per-generator detection rate (cross-generator essays).
    pg = r["A_cross_generator_essays"]["per_generator_detection_rate"]
    gens = sorted(pg, key=pg.get)
    vals = [pg[g] for g in gens]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.bar(gens, vals, color="#d98e3b", width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v*100:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.1); ax.set_ylabel("detected as AI (recall)")
    ax.set_title("Cross-generator transfer on essays: each unseen model's detection rate",
                 fontsize=12, fontweight="bold", color=INK)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "fig_m4_per_generator.png", dpi=200, facecolor="white")
    plt.close(fig); print("Saved fig_m4_per_generator.png")

    # Figure 2: the gap, in-domain vs cross-generator vs cross-domain.
    labels = ["In-domain\n(BAWE vs Llama)", "Cross-generator\n(M4 essays)", "Cross-domain\n(M4 web/academic)"]
    vals = [r["in_domain_f1"], r["A_cross_generator_essays"]["overall"]["f1"],
            r["B_cross_domain"]["overall"]["f1"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(labels, vals, color=["#2b6777", "#3a7d44", "#b0483b"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
    ax.axhline(0.5, ls="--", lw=1, color="#888"); ax.text(2.4, 0.515, "chance", fontsize=9, color="#888", ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("F1")
    ax.set_title("Robustness: the in-domain score does not transfer everywhere",
                 fontsize=12, fontweight="bold", color=INK)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "fig_m4_transfer_gap.png", dpi=200, facecolor="white")
    plt.close(fig); print("Saved fig_m4_transfer_gap.png")

    # Figure 3: per-domain failure modes (machine caught vs human wrongly flagged).
    pd_ = r["B_cross_domain"]["per_domain"]
    doms = list(pd_.keys())
    det = [pd_[d]["machine_detection_rate"] for d in doms]
    fpr = [pd_[d]["human_false_positive_rate"] for d in doms]
    x = np.arange(len(doms)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, det, w, color="#d98e3b", label="AI text caught (recall)")
    ax.bar(x + w / 2, fpr, w, color="#2b6777", label="human text wrongly flagged (false-positive rate)")
    ax.set_xticks(x); ax.set_xticklabels(doms)
    ax.set_ylim(0, 1.1); ax.set_ylabel("rate")
    ax.set_title("Cross-domain failure modes by domain (human text is not essays here)",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "fig_m4_per_domain.png", dpi=200, facecolor="white")
    plt.close(fig); print("Saved fig_m4_per_domain.png")


if __name__ == "__main__":
    raise SystemExit(main())
