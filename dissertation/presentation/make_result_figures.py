"""Result figures for the Meeting 3 visual deck.

  fig_stylometry_human_vs_ai.png  small-multiple bars, human vs AI on five style features
  fig_detector_confusion.png      detector test-set confusion matrix

    python dissertation/presentation/make_result_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "detection"))
from stylometric import load_nlp, stylometric_features  # noqa: E402
from text_normalize import normalize_text  # noqa: E402

AI = REPO / "data" / "processed" / "ai_essays"
CORPUS = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
FIGS = REPO / "dissertation" / "figures"
# Honest headline detector: DeBERTa on the cleaned (markup-stripped) corpus.
METRICS = REPO / "outputs" / "detector_metrics_clean.json"

HUMAN_C, AI_C, INK = "#2b6777", "#d98e3b", "#222831"
N_PAIRS = 200


def stylometry_figure():
    nlp = load_nlp()
    ids = sorted(p.stem for p in AI.glob("*.txt"))[:N_PAIRS]
    rows = []
    for rid in ids:
        hp = CORPUS / f"{rid}.txt"
        if not hp.exists():
            continue
        h = stylometric_features(normalize_text(hp.read_text(encoding="utf-8", errors="ignore")), nlp); h["who"] = "human"
        a = stylometric_features(normalize_text((AI / f"{rid}.txt").read_text(encoding="utf-8", errors="ignore")), nlp); a["who"] = "AI"
        rows += [h, a]
    df = pd.DataFrame(rows)
    feats = [("std_sent_len", "Sentence-length\nvariation"),
             ("root_ttr", "Vocabulary\nrichness"),
             ("hapax_ratio", "Rare (one-off)\nwords"),
             ("mean_word_len", "Word\nlength"),
             ("pos_NOUN", "Noun\ndensity")]
    means = df.groupby("who")[ [f for f, _ in feats] ].mean()
    fig, axes = plt.subplots(1, len(feats), figsize=(13, 4.2))
    for ax, (f, label) in zip(axes, feats):
        vals = [means.loc["human", f], means.loc["AI", f]]
        ax.bar(["human", "AI"], vals, color=[HUMAN_C, AI_C], width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if v < 5 else f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(label, fontsize=11, color=INK)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_yticks([])
    fig.suptitle(f"Writing style: human vs AI essays (lengths matched, {N_PAIRS} pairs)",
                 fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "fig_stylometry_human_vs_ai.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_stylometry_human_vs_ai.png")


def confusion_figure():
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    cm = np.array(m["test"]["confusion_matrix_[hu,ai]x[pred_hu,pred_ai]"])
    f1 = m["test"]["f1"]
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    labels = ["Human", "AI"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([f"predicted\n{l}" for l in labels], fontsize=11)
    ax.set_yticklabels([f"actually\n{l}" for l in labels], fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=22,
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontweight="bold")
    ax.set_title(f"Held-out test, markup removed\nDeBERTa-v3 cleaned, F1 = {f1:.2f}, n = 200",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_detector_confusion.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_detector_confusion.png")


if __name__ == "__main__":
    stylometry_figure()
    confusion_figure()
