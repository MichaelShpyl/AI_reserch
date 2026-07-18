"""The training-distribution control, re-run where the task has resolution.

Section 6.6 trained two same-size detectors that differ only in the human-writer distribution
(balanced across the eight discipline-by-language cells versus the corpus's natural skew) and
found the home task too easy to separate them: both sit at the ceiling. The conclusions chapter
left the obvious next step as future work: evaluate the same two models where detection is hard.
This script does that, on the exact cross-domain sample from eval_m4_transfer.py (same seed, same
caps), where the main detector drops to 0.79 and the failure mode is false accusation.

The fairness-relevant question: the balanced training set holds non-native writers at one half
against roughly three in ten naturally, so if writer balance changes out-of-domain behaviour at
all, the human false-positive rate is where it should show.

    python src/detection/dist_crossdomain.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from text_normalize import normalize_text

REPO = Path(__file__).resolve().parents[2]
M4 = REPO / "data" / "raw" / "m4"
DIST = REPO / "models" / "detector_dist"
FIGS = REPO / "dissertation" / "figures"
OUT = REPO / "outputs" / "dist_crossdomain.json"
PER_DOMAIN = 300
MAXLEN = 512
BATCH = 16
SEED = 42


def build_sample() -> pd.DataFrame:
    train = pd.read_parquet(M4 / "train.parquet")
    domains = sorted(train["source"].dropna().unique())
    parts = []
    for d in domains:
        sd = train[train["source"] == d]
        for lab in (0, 1):
            s = sd[sd["label"] == lab]
            if len(s):
                parts.append(s.sample(n=min(PER_DOMAIN, len(s)), random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)


def evaluate(name: str, sample: pd.DataFrame, device) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ckpt = sorted((DIST / name).glob("checkpoint-*"))[-1]
    print(f"[{name}] loading {ckpt.name}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device).eval()
    texts = [normalize_text(t) for t in sample["text"].tolist()]
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to(device)
            preds.extend(model(**enc).logits.argmax(-1).tolist())
    del model
    torch.cuda.empty_cache()
    sample = sample.assign(pred=preds)
    per_domain = {}
    for d in sorted(sample["source"].unique()):
        sd = sample[sample["source"] == d]
        hu, ai = sd[sd.label == 0], sd[sd.label == 1]
        per_domain[d] = {
            "accuracy": round(float((sd["pred"] == sd["label"]).mean()), 4),
            "human_false_positive_rate": round(float((hu["pred"] == 1).mean()), 4),
            "machine_detection_rate": round(float((ai["pred"] == 1).mean()), 4),
        }
    hu_all = sample[sample.label == 0]
    return {
        "overall_accuracy": round(float(accuracy_score(sample["label"], sample["pred"])), 4),
        "overall_f1": round(float(f1_score(sample["label"], sample["pred"])), 4),
        "human_fpr_overall": round(float((hu_all["pred"] == 1).mean()), 4),
        "per_domain": per_domain,
    }, np.array(preds)


def make_figure(r: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    domains = sorted(r["balanced"]["per_domain"])
    x = np.arange(len(domains))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for off, (name, color) in zip((-w / 2, w / 2),
                                  [("balanced", "#2b6777"), ("natural", "#d98e3b")]):
        vals = [r[name]["per_domain"][d]["human_false_positive_rate"] for d in domains]
        ax.bar(x + off, vals, w, label=f"{name} training mix", color=color)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("human false-positive rate")
    ax.set_title("The training-distribution control on the hard task: false accusations by domain",
                 fontsize=12, fontweight="bold", color="#222831")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_dist_crossdomain.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_dist_crossdomain.png")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = build_sample()
    print(f"cross-domain sample: {len(sample)} texts, "
          f"{sample['source'].nunique()} domains (seed {SEED}, same as eval_m4_transfer)")
    results = {"note": "identical cross-domain sample as m4_transfer Pass B; the two models "
                       "differ only in the human-writer distribution of their training data",
               "n_texts": int(len(sample)), "seed": SEED}
    all_preds = {}
    for name in ("balanced", "natural"):
        results[name], all_preds[name] = evaluate(name, sample, device)
        print(f"[{name}] acc {results[name]['overall_accuracy']}  "
              f"f1 {results[name]['overall_f1']}  "
              f"human FPR {results[name]['human_fpr_overall']}", flush=True)

    # Paired significance on the same texts: exact McNemar via the binomial on discordant pairs.
    from scipy.stats import binomtest
    y = sample["label"].values
    bal, nat = all_preds["balanced"], all_preds["natural"]
    hum = y == 0
    b_wrong_n_right = int(((bal[hum] == 1) & (nat[hum] == 0)).sum())
    n_wrong_b_right = int(((nat[hum] == 1) & (bal[hum] == 0)).sum())
    n_disc = b_wrong_n_right + n_wrong_b_right
    p_fpr = binomtest(b_wrong_n_right, n_disc, 0.5).pvalue if n_disc else 1.0
    acc_b_only = int(((bal == y) & (nat != y)).sum())
    acc_n_only = int(((nat == y) & (bal != y)).sum())
    n_disc_a = acc_b_only + acc_n_only
    p_acc = binomtest(acc_b_only, n_disc_a, 0.5).pvalue if n_disc_a else 1.0
    results["paired"] = {
        "human_fpr_mcnemar": {"balanced_only_wrong": b_wrong_n_right,
                              "natural_only_wrong": n_wrong_b_right,
                              "p": round(float(p_fpr), 4)},
        "accuracy_mcnemar": {"balanced_only_right": acc_b_only,
                             "natural_only_right": acc_n_only,
                             "p": round(float(p_acc), 4)},
    }
    print("paired:", json.dumps(results["paired"]))
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    make_figure(results)
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
