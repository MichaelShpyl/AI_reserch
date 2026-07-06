"""Stylometric detector and its SHAP feature-level explanation.

This builds the stylometric half of the hybrid detector (pipeline component 1) and delivers
the faithful, feature-level explanation that Chapter 5 pointed to (component 2). Chapter 5
showed that token-level highlights are weak for this detector because the signal is diffuse;
the honest explanation is at the level of writing-style features, and that is what this gives.

It computes hand-crafted style features (sentence-length variation, vocabulary richness, POS
mix, punctuation) on the cleaned corpus, trains a gradient-boosted classifier on the
student-level splits, reports how well style features alone separate the classes, and uses
SHAP to show which features push a decision toward AI or human.

    python src/explainability/shap_stylometric.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

REPO = Path(__file__).resolve().parents[2]
CLEAN = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
FEATS = REPO / "data" / "processed" / "stylometric_features.parquet"
FIGS = REPO / "dissertation" / "figures"
OUT = REPO / "outputs" / "stylometric_shap.json"

# Drop length-related columns so this is a pure writing-style model (lengths are matched anyway).
DROP = ["n_words", "n_sents", "id", "label", "split", "native"]

# Friendly names for the figure.
PRETTY = {
    "std_sent_len": "sentence-length variation", "sent_len_cv": "sentence-length spread",
    "burstiness": "burstiness", "mean_sent_len": "mean sentence length",
    "ttr": "vocabulary richness (TTR)", "root_ttr": "vocabulary richness (root TTR)",
    "hapax_ratio": "rare (one-off) words", "mean_word_len": "word length",
    "punct_ratio": "punctuation density", "pos_NOUN": "noun density", "pos_VERB": "verb density",
    "pos_ADJ": "adjective density", "pos_ADV": "adverb density", "pos_PRON": "pronoun density",
    "pos_PROPN": "proper-noun density", "pos_ADP": "preposition density", "pos_DET": "determiner density",
    "pos_AUX": "auxiliary-verb density", "pos_CCONJ": "coord. conjunction density",
    "pos_SCONJ": "subord. conjunction density", "pos_NUM": "number density",
    "pos_PART": "particle density", "pos_PUNCT": "punctuation-token density",
}


def compute_features() -> pd.DataFrame:
    if FEATS.exists():
        print(f"Loading cached features from {FEATS.relative_to(REPO)}", flush=True)
        return pd.read_parquet(FEATS)
    import sys
    sys.path.insert(0, str(REPO / "src" / "detection"))
    from stylometric import load_nlp, stylometric_features
    df = pd.read_parquet(CLEAN)
    nlp = load_nlp()
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        f = stylometric_features(r.text, nlp)   # text is already markup-cleaned
        f.update(id=r.id, label=int(r.label), split=r.split, native=bool(r.native))
        rows.append(f)
        if i % 200 == 0:
            print(f"  features {i}/{len(df)}", flush=True)
    out = pd.DataFrame(rows)
    FEATS.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATS, index=False)
    print(f"Saved {FEATS.relative_to(REPO)}", flush=True)
    return out


def main() -> int:
    df = compute_features()
    feat_cols = [c for c in df.columns if c not in DROP]
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    Xtr, ytr = tr[feat_cols], tr["label"]
    Xte, yte = te[feat_cols], te["label"]

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = {
        "n_features": len(feat_cols),
        "test": {"accuracy": round(accuracy_score(yte, pred), 4),
                 "precision": round(precision_score(yte, pred, zero_division=0), 4),
                 "recall": round(recall_score(yte, pred, zero_division=0), 4),
                 "f1": round(f1_score(yte, pred, zero_division=0), 4),
                 "confusion_matrix_[hu,ai]x[pred_hu,pred_ai]": confusion_matrix(yte, pred).tolist()},
    }
    native = te["native"].values
    fairness = {}
    for grp, mask in [("native", native), ("non_native", ~native)]:
        hu = mask & (yte.values == 0)
        if hu.sum():
            fpr = float(((pred == 1) & hu).sum() / hu.sum())
            fairness[grp] = {"n_human": int(hu.sum()), "false_positive_rate": round(fpr, 3)}
    metrics["fairness_human_FPR_by_L1"] = fairness

    # ---- SHAP ----
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xte)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = [{"feature": feat_cols[i], "pretty": PRETTY.get(feat_cols[i], feat_cols[i]),
            "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in order[:12]]
    metrics["top_features_by_shap"] = top

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_shap(sv, Xte, feat_cols)

    print("\n=== STYLOMETRIC DETECTOR (style features only) ===")
    print("test:", metrics["test"])
    print("fairness:", fairness)
    print("top SHAP features:", [t["pretty"] for t in top])
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def plot_shap(sv, Xte, feat_cols) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    pretty = [PRETTY.get(c, c) for c in feat_cols]
    Xp = Xte.copy(); Xp.columns = pretty
    plt.figure()
    shap.summary_plot(sv, Xp, max_display=14, show=False, color_bar_label="feature value")
    fig = plt.gcf()
    fig.set_size_inches(9, 6.2)
    fig.suptitle("SHAP: which style features push a decision toward AI (right) or human (left)",
                 fontsize=12, fontweight="bold", color="#222831", y=1.02)
    fig.savefig(FIGS / "fig_shap_stylometric.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig_shap_stylometric.png")


if __name__ == "__main__":
    raise SystemExit(main())
