"""Adversarial audit of the human-vs-AI detector: is the near-perfect score real?

Run after the detector reports F1 = 1.00 to find out WHY, with evidence. It does four
things and writes a JSON report plus figures:

  1. Split integrity. No student appears in more than one split (style leakage), each
     human/AI pair sits in the same split, and no exact-duplicate text crosses splits.
  2. Markup artifact. How often the raw human text carries BAWE export tags (<heading>,
     <fnote> ...) versus the AI text, and how accurate the dumb rule "has a tag -> human"
     is on its own. This is the leakage the cleaned corpus removes.
  3. Interpretable linear baselines. A TF-IDF + logistic-regression classifier on the RAW
     text and on the CLEANED text, trained on the train split and scored on the held-out
     test split, with the top weighted features for each class. If a simple linear model
     also separates them, the signal is broad and not a single bug; the features show what
     it keys on (markup and layout on raw, lexical style on clean).
  4. Style-not-topic control. A logistic regression restricted to function words only
     (no content words) on the cleaned text. If that still separates the classes, the
     remaining signal is writing style, not the essay topic.

    python src/detection/audit_detector.py
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from text_normalize import has_markup

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "processed" / "bawe_human_sample_manifest.csv"
RAW = REPO / "data" / "processed" / "detection_corpus.parquet"
CLEAN = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
FIGS = REPO / "dissertation" / "figures"
OUT = REPO / "outputs" / "audit_report.json"


def split_integrity(man: pd.DataFrame, raw: pd.DataFrame) -> dict:
    # A student in two splits would let the model memorise a person's style.
    by_student = man.groupby("student_id")["split"].nunique()
    leaked = by_student[by_student > 1]
    # Each id (human + AI) must share one split.
    pair_split = raw.groupby("id")["split"].nunique()
    pair_bad = pair_split[pair_split > 1]
    # Exact-duplicate text crossing splits would inflate the test score.
    raw = raw.assign(h=raw["text"].map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()))
    dup_cross = 0
    for _, g in raw.groupby("h"):
        if g["split"].nunique() > 1:
            dup_cross += 1
    return {
        "n_students": int(man["student_id"].nunique()),
        "students_in_multiple_splits": int(len(leaked)),
        "pairs_split_inconsistent": int(len(pair_bad)),
        "exact_duplicate_texts_across_splits": int(dup_cross),
        "n_exact_duplicate_text_groups": int((raw.groupby("h").size() > 1).sum()),
        "verdict": "clean" if len(leaked) == 0 and len(pair_bad) == 0 and dup_cross == 0
        else "LEAK",
    }


def markup_artifact(raw: pd.DataFrame) -> dict:
    raw = raw.assign(tag=raw["text"].map(has_markup))
    rate = raw.groupby("source")["tag"].mean()
    test = raw[raw["split"] == "test"]
    # Dumb rule: predict human (0) when a tag is present, AI (1) otherwise.
    pred = (~test["text"].map(has_markup)).astype(int)
    acc = accuracy_score(test["label"], pred)
    return {
        "human_raw_tag_rate": round(float(rate.get("human", 0.0)), 4),
        "ai_raw_tag_rate": round(float(rate.get("ai", 0.0)), 4),
        "dumb_has_tag_rule_test_accuracy": round(float(acc), 4),
        "reading": "A tag-presence rule alone reaches this accuracy on the test set, so "
                   "raw-text markup is a strong giveaway the cleaned corpus removes.",
    }


def length_recap(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    def by_label(df):
        w = df.assign(n=df["text"].str.split().str.len())
        return {k: round(float(v), 1) for k, v in w.groupby("label")["n"].mean().items()}
    return {"raw_mean_words_by_label": by_label(raw),
            "clean_mean_words_by_label": by_label(clean)}


def linear_baseline(df: pd.DataFrame, vec: TfidfVectorizer, name: str) -> dict:
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    clf = Pipeline([("vec", vec),
                    ("lr", LogisticRegression(max_iter=2000, C=4.0))])
    clf.fit(tr["text"], tr["label"])
    pred = clf.predict(te["text"])
    acc = accuracy_score(te["label"], pred)
    f1 = f1_score(te["label"], pred)
    feats = np.array(clf.named_steps["vec"].get_feature_names_out())
    coef = clf.named_steps["lr"].coef_[0]
    ai_idx = np.argsort(coef)[-20:][::-1]
    hu_idx = np.argsort(coef)[:20]
    top_ai = [feats[i] for i in ai_idx]
    top_hu = [feats[i] for i in hu_idx]
    return {"name": name, "test_accuracy": round(float(acc), 4),
            "test_f1": round(float(f1), 4),
            "top_features_AI": top_ai, "top_features_human": top_hu,
            "top_weights_AI": [round(float(coef[i]), 3) for i in ai_idx],
            "top_weights_human": [round(float(abs(coef[i])), 3) for i in hu_idx],
            "n_features": int(len(feats))}


def make_figures(report: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    # Figure: raw vs clean separability for the linear model and the dumb rule.
    labels = ["Dumb rule\n(markup only)", "Linear model\nRAW text", "Linear model\nCLEAN text",
              "Function words\nonly (CLEAN)"]
    vals = [report["markup_artifact"]["dumb_has_tag_rule_test_accuracy"],
            report["linear_raw"]["test_accuracy"],
            report["linear_clean"]["test_accuracy"],
            report["function_words_clean"]["test_accuracy"]]
    colors = ["#b0483b", "#d98e3b", "#2b6777", "#3a7d44"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222831")
    ax.axhline(0.5, ls="--", lw=1, color="#888")
    ax.text(3.45, 0.515, "chance", fontsize=9, color="#888", ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Why the detector separates human and AI: what each signal alone achieves",
                 fontsize=12.5, fontweight="bold", color="#222831")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_audit_separability.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_audit_separability.png")

    # Figure: top style features of the CLEAN linear model (human vs AI), bar = weight.
    hu = report["linear_clean"]["top_features_human"][:12][::-1]
    hu_w = report["linear_clean"]["top_weights_human"][:12][::-1]
    ai = report["linear_clean"]["top_features_AI"][:12][::-1]
    ai_w = report["linear_clean"]["top_weights_AI"][:12][::-1]
    wmax = max(max(hu_w), max(ai_w)) * 1.15
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].barh(range(len(hu)), hu_w, color="#2b6777")
    axes[0].set_yticks(range(len(hu))); axes[0].set_yticklabels(hu, fontsize=11)
    axes[0].set_title("Pushes toward HUMAN", fontsize=12, color="#2b6777", fontweight="bold")
    axes[1].barh(range(len(ai)), ai_w, color="#d98e3b")
    axes[1].set_yticks(range(len(ai))); axes[1].set_yticklabels(ai, fontsize=11)
    axes[1].set_title("Pushes toward AI", fontsize=12, color="#d98e3b", fontweight="bold")
    for ax in axes:
        ax.set_xticks([]); ax.set_xlim(0, wmax)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Top words the cleaned-text linear detector keys on (style, not topic)",
                 fontsize=13, fontweight="bold", color="#222831")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "fig_audit_top_features.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_audit_top_features.png")


def main() -> int:
    man = pd.read_csv(MANIFEST)
    raw = pd.read_parquet(RAW)
    clean = pd.read_parquet(CLEAN)

    report: dict = {}
    report["split_integrity"] = split_integrity(man, raw)
    report["markup_artifact"] = markup_artifact(raw)
    report["length_recap"] = length_recap(raw, clean)

    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                               sublinear_tf=True, lowercase=True)
    report["linear_raw"] = linear_baseline(raw, word_vec, "TF-IDF word 1-2gram, RAW")
    word_vec2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                                sublinear_tf=True, lowercase=True)
    report["linear_clean"] = linear_baseline(clean, word_vec2, "TF-IDF word 1-2gram, CLEAN")

    # Style-not-topic control: vocabulary limited to function words.
    fw = sorted(ENGLISH_STOP_WORDS)
    fw_vec = TfidfVectorizer(vocabulary=fw, sublinear_tf=True, lowercase=True)
    report["function_words_clean"] = linear_baseline(clean, fw_vec,
                                                     "Function words only, CLEAN")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_figures(report)

    print("\n=== AUDIT SUMMARY ===")
    print("Split integrity:", report["split_integrity"]["verdict"],
          report["split_integrity"])
    print("Markup artifact:", report["markup_artifact"])
    print("Linear RAW  : acc",
          report["linear_raw"]["test_accuracy"], "f1", report["linear_raw"]["test_f1"])
    print("Linear CLEAN: acc",
          report["linear_clean"]["test_accuracy"], "f1", report["linear_clean"]["test_f1"])
    print("Func-words CLEAN: acc", report["function_words_clean"]["test_accuracy"])
    print("\nTop AI features (CLEAN):", report["linear_clean"]["top_features_AI"][:12])
    print("Top human features (CLEAN):", report["linear_clean"]["top_features_human"][:12])
    print(f"\nSaved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
