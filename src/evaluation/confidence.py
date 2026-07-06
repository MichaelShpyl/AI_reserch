"""Uncertainty for the detector results: bootstrap confidence intervals and seed stability.

The review (16 June 2026) flagged that every headline number is a single-seed point estimate on
a small test set (n=200), with no interval, while the locked evaluation plan asks
for "stability across seeds". This script fixes that two ways:

  1. Bootstrap 95% CIs on the held-out test metrics for each detector, by resampling the 200
     test outcomes (reconstructed from each saved confusion matrix). With n=200 a single
     misclassification moves F1 by about 0.005, so the interval matters.
  2. Seed stability for the (fast, CPU) stylometric detector: refit across seeds 41 to 45 and
     report mean and standard deviation of the test F1. The transformer multi-seed run is the
     heavier confirmation and is noted as a follow-up.

    python src/evaluation/confidence.py

Writes outputs/confidence_intervals.json and dissertation/figures/fig_confidence_intervals.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "outputs"
FIGS = REPO / "dissertation" / "figures"
FEATS = REPO / "data" / "processed" / "stylometric_features.parquet"
OUT = OUTDIR / "confidence_intervals.json"
N_BOOT = 5000
SEEDS = [41, 42, 43, 44, 45]
DROP = ["n_words", "n_sents", "id", "label", "split", "native"]
RNG = np.random.default_rng(42)


def outcomes_from_cm(cm):
    """cm = [[TN, FP], [FN, TP]] for [human, ai] x [pred_human, pred_ai]."""
    (tn, fp), (fn, tp) = cm
    y = np.array([0] * (tn + fp) + [1] * (fn + tp))
    p = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    return y, p


def boot_ci(y, p, fn, n=N_BOOT):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = RNG.choice(idx, size=len(idx), replace=True)
        vals.append(fn(y[b], p[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(np.mean(vals)), 4), [round(float(lo), 4), round(float(hi), 4)]


def metrics_with_ci(cm) -> dict:
    y, p = outcomes_from_cm(cm)
    def acc(a, b): return float((a == b).mean())
    def f1(a, b): return f1_score(a, b, zero_division=0)
    def prec(a, b):
        d = (b == 1).sum(); return float(((b == 1) & (a == 1)).sum() / d) if d else 0.0
    def rec(a, b):
        d = (a == 1).sum(); return float(((b == 1) & (a == 1)).sum() / d) if d else 0.0
    m, ci = boot_ci(y, p, f1)
    out = {"n": int(len(y)), "f1": m, "f1_ci95": ci}
    for name, f in [("accuracy", acc), ("precision", prec), ("recall", rec)]:
        mm, cc = boot_ci(y, p, f)
        out[name] = mm; out[f"{name}_ci95"] = cc
    return out


def fpr_ci(n_human, n_false_pos):
    """Bootstrap CI on a human false-positive rate from counts."""
    outcomes = np.array([1] * n_false_pos + [0] * (n_human - n_false_pos))
    vals = [RNG.choice(outcomes, size=n_human, replace=True).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"n_human": n_human, "false_positives": n_false_pos,
            "rate": round(n_false_pos / n_human, 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)]}


def load(name):
    p = OUTDIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def stylometric_seed_stability() -> dict:
    if not FEATS.exists():
        return {"note": "run shap_stylometric.py first to cache features"}
    df = pd.read_parquet(FEATS)
    feat_cols = [c for c in df.columns if c not in DROP]
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    f1s = []
    for s in SEEDS:
        clf = GradientBoostingClassifier(random_state=s).fit(tr[feat_cols], tr["label"])
        f1s.append(f1_score(te["label"], clf.predict(te[feat_cols]), zero_division=0))
    return {"seeds": SEEDS, "test_f1_per_seed": [round(float(x), 4) for x in f1s],
            "test_f1_mean": round(float(np.mean(f1s)), 4),
            "test_f1_std": round(float(np.std(f1s)), 4)}


def main() -> int:
    report = {"n_bootstrap": N_BOOT, "models": {}}
    sources = {
        "DeBERTa (clean)": ("detector_metrics_clean.json", "test"),
        "RoBERTa (clean)": ("detector_metrics_clean_roberta.json", "test"),
        "Stylometric": ("stylometric_shap.json", "test"),
    }
    for label, (fname, key) in sources.items():
        d = load(fname)
        if not d:
            continue
        cm = d[key]["confusion_matrix_[hu,ai]x[pred_hu,pred_ai]"]
        report["models"][label] = metrics_with_ci(cm)
        fair = d.get("fairness_human_FPR_by_L1", {})
        if fair:
            report["models"][label]["fairness"] = {
                g: fpr_ci(v["n_human"], round(v["false_positive_rate"] * v["n_human"]))
                for g, v in fair.items()}

    m4 = load("m4_transfer.json")
    if m4:
        report["models"]["M4 cross-generator"] = metrics_with_ci(
            m4["A_cross_generator_essays"]["overall"]["confusion_matrix_[hu,ai]x[pred_hu,pred_ai]"])
        report["models"]["M4 cross-domain"] = metrics_with_ci(
            m4["B_cross_domain"]["overall"]["confusion_matrix_[hu,ai]x[pred_hu,pred_ai]"])

    report["stylometric_seed_stability"] = stylometric_seed_stability()
    report["reading"] = ("Bootstrap 95% CIs (resampling the test outcomes) quantify sampling "
                         "uncertainty on n=200; seed stability quantifies training randomness for "
                         "the stylometric model. Transformer multi-seed retraining is the heavier "
                         "confirmation and is the documented next step.")
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_figure(report)

    print("=== CONFIDENCE INTERVALS (test F1, 95% bootstrap) ===")
    for k, v in report["models"].items():
        print(f"  {k:22s} F1 {v['f1']:.3f}  CI {v['f1_ci95']}")
    print("stylometric seed stability:", report["stylometric_seed_stability"])
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(report) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    items = [(k, v["f1"], v["f1_ci95"]) for k, v in report["models"].items()]
    labels = [k for k, _, _ in items]
    vals = [f for _, f, _ in items]
    lo = [f - c[0] for _, f, c in items]
    hi = [c[1] - f for _, f, c in items]
    colors = ["#2b6777", "#2b6777", "#3a7d44", "#d98e3b", "#b0483b"][:len(items)]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(labels, vals, color=colors, width=0.6,
           yerr=[lo, hi], capsize=5, error_kw={"ecolor": "#222831", "elinewidth": 1.3})
    for i, (_, f, c) in enumerate(items):
        ax.text(i, c[1] + 0.015, f"{f:.2f}\n[{c[0]:.2f}, {c[1]:.2f}]", ha="center",
                va="bottom", fontsize=8.5, color="#222831")
    ax.axhline(0.5, ls="--", lw=1, color="#888")
    ax.set_ylim(0, 1.18); ax.set_ylabel("test F1")
    ax.set_title("Detector F1 with 95% bootstrap confidence intervals\n"
                 "(in-domain tests n = 200; M4 transfer tests are larger, so their intervals are tighter)",
                 fontsize=11.5, fontweight="bold", color="#222831")
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_confidence_intervals.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_confidence_intervals.png")


if __name__ == "__main__":
    raise SystemExit(main())
