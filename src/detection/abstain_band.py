"""The abstain band, measured: what does "the detector may refuse to answer" actually buy?

Chapters 6 and 9 propose an abstain band as a deployment safeguard: when the hybrid's probability
sits in a middle band, the system says "uncertain" instead of flagging. This turns that proposal into
numbers. It re-scores the same cross-domain M4 sample as Sections 6.3 and 6.7 with the saved hybrid,
keeps the per-text probabilities this time (outputs/crossdomain_probs.parquet), then sweeps abstain
bands and reports, for each: how many texts the system declines to judge, and what happens to
accuracy and the human false-positive rate among the texts it still judges.

    python src/detection/abstain_band.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
M4 = REPO / "data" / "raw" / "m4"
HYBRID = REPO / "models" / "hybrid"
PROBS = REPO / "outputs" / "crossdomain_probs.parquet"
OUT = REPO / "outputs" / "abstain_band.json"
FIGS = REPO / "dissertation" / "figures"
PER_DOMAIN = 300
SEED = 42

import sys
sys.path.insert(0, str(REPO / "src" / "detection"))
from text_normalize import normalize_text  # noqa: E402
from hybrid_fusion import gpt2_perplexity, deberta_probs  # noqa: E402
from stylometric import load_nlp, stylometric_features  # noqa: E402

BANDS = [(0.5, 0.5), (0.4, 0.6), (0.35, 0.65), (0.3, 0.7), (0.25, 0.75), (0.2, 0.8)]


def compute_probs() -> pd.DataFrame:
    if PROBS.exists():
        print("Loading cached cross-domain probabilities", flush=True)
        return pd.read_parquet(PROBS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = pd.read_parquet(M4 / "train.parquet")
    parts = []
    for d in sorted(train["source"].dropna().unique()):
        sd = train[train["source"] == d]
        for lab in (0, 1):
            s = sd[sd["label"] == lab]
            if len(s):
                parts.append(s.sample(n=min(PER_DOMAIN, len(s)), random_state=SEED))
    B = pd.concat(parts).reset_index(drop=True)
    texts = [normalize_text(t) for t in B["text"].tolist()]
    print(f"{len(B)} texts; transformer probabilities ...", flush=True)
    p_deb = deberta_probs(texts, device)
    print("style features ...", flush=True)
    nlp = load_nlp()
    rows = []
    for i, t in enumerate(texts):
        rows.append(stylometric_features(t, nlp))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(texts)}", flush=True)
    F = pd.DataFrame(rows)
    print("perplexity ...", flush=True)
    F["gpt2_ppl"] = gpt2_perplexity(texts, device)
    feat_cols = json.loads((HYBRID / "feat_cols.json").read_text(encoding="utf-8"))
    gbm = pickle.load(open(HYBRID / "gbm_ppl.pkl", "rb"))
    fuser = pickle.load(open(HYBRID / "fuser.pkl", "rb"))
    p_style = gbm.predict_proba(F[feat_cols + ["gpt2_ppl"]])[:, 1]
    p_hy = fuser.predict_proba(np.column_stack([p_deb, p_style]))[:, 1]
    dfp = pd.DataFrame({"domain": B["source"], "label": B["label"],
                        "p_transformer": p_deb, "p_style": p_style, "p_hybrid": p_hy})
    dfp.to_parquet(PROBS, index=False)
    print(f"Saved {PROBS.relative_to(REPO)}", flush=True)
    return dfp


def main() -> int:
    df = compute_probs()
    y = df["label"].values
    p = df["p_hybrid"].values
    sweep = []
    for lo, hi in BANDS:
        judged = (p < lo) | (p > hi)
        pj, yj = p[judged], y[judged]
        preds = (pj >= 0.5).astype(int)
        hu = yj == 0
        entry = {
            "band": [lo, hi],
            "abstain_rate": round(float(1 - judged.mean()), 4),
            "accuracy_on_judged": round(float((preds == yj).mean()), 4) if judged.sum() else None,
            "human_FPR_on_judged": round(float((preds[hu] == 1).mean()), 4) if hu.sum() else None,
        }
        per_dom = {}
        for d in sorted(df["domain"].unique()):
            m = (df["domain"] == d).values & judged
            hu_d = m & (y == 0)
            per_dom[d] = {
                "abstain_rate": round(float(1 - ((df["domain"] == d).values & judged).sum()
                                            / (df["domain"] == d).sum()), 3),
                "human_FPR": round(float(((p[hu_d] >= 0.5)).mean()), 3) if hu_d.sum() else None,
            }
        entry["per_domain"] = per_dom
        sweep.append(entry)
        print(f"band {lo}-{hi}: abstain {entry['abstain_rate']:.1%}, "
              f"acc {entry['accuracy_on_judged']}, human FPR {entry['human_FPR_on_judged']}", flush=True)

    result = {"n": int(len(df)), "sweep": sweep,
              "note": "same cross-domain sample and hybrid as Section 6.7; the band is applied to "
                      "the hybrid probability; 'judged' texts are those outside the band"}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(result)
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep = result["sweep"]
    x = [s["abstain_rate"] * 100 for s in sweep]
    fpr = [s["human_FPR_on_judged"] * 100 for s in sweep]
    acc = [s["accuracy_on_judged"] * 100 for s in sweep]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(x, fpr, "-o", color="#a63d2e", lw=2, label="human false-positive rate (on judged texts)")
    ax.plot(x, acc, "-o", color="#2b6777", lw=2, label="accuracy (on judged texts)")
    for s in sweep:
        ax.annotate(f"{s['band'][0]}-{s['band'][1]}", (s["abstain_rate"] * 100,
                    s["human_FPR_on_judged"] * 100), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color="#52616B")
    ax.set_xlabel("texts the system declines to judge (%)")
    ax.set_ylabel("percent")
    ax.set_title("What abstention buys, measured: accuracy rises, but the remaining\n"
                 "false accusations are confident ones the band does not catch",
                 fontsize=11.5, fontweight="bold", color="#222831")
    ax.legend(fontsize=9.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_abstain_band.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_abstain_band.png")


if __name__ == "__main__":
    raise SystemExit(main())
