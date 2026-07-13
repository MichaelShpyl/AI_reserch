"""A per-submission, plain-language explanation card (the lecturer-facing explainability upgrade).

The SHAP beeswarm in Chapter 5 convinces a researcher but not a busy lecturer. This module explains
ONE submission in words and one simple picture: which writing habits pushed this essay's score, each
stated in plain English and quantified against what is typical for real student essays in the corpus
(median and the middle 80 percent band). It uses the persisted hybrid's style model (TreeExplainer on
a single row is instant) so the explanation describes the very model that produced the score.

Outputs, for embedding in the Verification Interview Guide:
  - a horizontal bar card (PNG): top habits, arrows toward AI or human, plain names
  - matching sentences ("The words are longer than in typical student writing: average 5.3 letters
    against a typical 4.6.")

    from explain_submission import explain_submission
    card = explain_submission(text)   # {"png_path", "sentences", "features"}
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HYBRID = REPO / "models" / "hybrid"
FEATS = REPO / "data" / "processed" / "stylometric_features.parquet"
PPL_CACHE = REPO / "data" / "processed" / "gpt2_perplexity_home.parquet"

# Plain-English templates per feature: (short label, sentence when the VALUE is higher than the
# human median, sentence when it is lower). The observed direction always follows the numbers; the
# toward-AI / toward-human push is appended separately from the SHAP sign, so the words can never
# contradict the figures.
PLAIN = {
    "mean_word_len": ("word length",
                      "The words are longer than in typical student writing (average {v:.1f} letters against a typical {m:.1f})",
                      "The words are shorter than in typical student writing ({v:.1f} letters against a typical {m:.1f})"),
    "ttr": ("vocabulary variety",
            "The vocabulary varies more than typical (variety score {v:.2f} against {m:.2f})",
            "The vocabulary is narrower and more repetitive than typical (variety score {v:.2f} against {m:.2f})"),
    "root_ttr": ("vocabulary richness",
                 "The overall word variety is higher than typical ({v:.1f} against {m:.1f})",
                 "The overall word variety is lower than typical ({v:.1f} against {m:.1f})"),
    "hapax_ratio": ("rare words",
                    "More one-off, unusual words than typical ({v:.2f} against {m:.2f})",
                    "Fewer one-off, unusual words than typical ({v:.2f} against {m:.2f})"),
    "std_sent_len": ("sentence variation",
                     "The sentence lengths vary more than typical ({v:.1f} against {m:.1f})",
                     "The sentences are unusually uniform in length (variation {v:.1f} against a typical {m:.1f})"),
    "sent_len_cv": ("sentence rhythm",
                    "The sentence rhythm varies more than typical ({v:.2f} against {m:.2f})",
                    "The sentence rhythm is steadier than typical ({v:.2f} against {m:.2f})"),
    "burstiness": ("burstiness",
                   "The writing is burstier than typical ({v:.2f} against {m:.2f})",
                   "The writing flows more evenly than typical human writing ({v:.2f} against {m:.2f})"),
    "mean_sent_len": ("sentence length",
                      "Sentences run longer than typical ({v:.1f} words against {m:.1f})",
                      "Sentences are shorter than typical ({v:.1f} words against {m:.1f})"),
    "punct_ratio": ("punctuation",
                    "Punctuation is denser than typical ({v:.3f} against {m:.3f})",
                    "Punctuation is sparser than typical ({v:.3f} against {m:.3f})"),
    "gpt2_ppl": ("predictability",
                 "A language model finds this text harder to predict than typical (surprise score {v:.0f} against {m:.0f})",
                 "A language model finds this text easier to predict than typical (surprise score {v:.0f} against {m:.0f})"),
    "pos_AUX": ("auxiliary verbs",
                "More helper verbs (is, has, can) than typical ({v:.3f} against {m:.3f})",
                "Fewer helper verbs (is, has, can) than typical ({v:.3f} against {m:.3f})"),
    "pos_DET": ("determiners",
                "More small grammar words (the, this) than typical ({v:.3f} against {m:.3f})",
                "Fewer small grammar words (the, this) than typical ({v:.3f} against {m:.3f})"),
    "pos_NOUN": ("nouns",
                 "More nouns than typical ({v:.2f} against {m:.2f})",
                 "Fewer nouns than typical ({v:.2f} against {m:.2f})"),
    "pos_VERB": ("verbs",
                 "More verbs than typical ({v:.2f} against {m:.2f})",
                 "Fewer verbs than typical ({v:.2f} against {m:.2f})"),
    "pos_ADJ": ("adjectives",
                "More describing words than typical ({v:.2f} against {m:.2f})",
                "Fewer describing words than typical ({v:.2f} against {m:.2f})"),
    "pos_ADV": ("adverbs",
                "More adverbs than typical ({v:.2f} against {m:.2f})",
                "Fewer adverbs than typical ({v:.2f} against {m:.2f})"),
    "pos_PRON": ("pronouns",
                 "More pronouns (I, they, it) than typical ({v:.2f} against {m:.2f})",
                 "Fewer pronouns (I, they, it) than typical ({v:.2f} against {m:.2f})"),
}


def _fallback(feat: str) -> tuple[str, str, str]:
    name = feat.replace("pos_", "").replace("_", " ")
    return (name,
            "The measure '" + name + "' is higher than typical ({v:.2f} against {m:.2f})",
            "The measure '" + name + "' is lower than typical ({v:.2f} against {m:.2f})")


def explain_submission(text: str, out_png: Path, top_k: int = 5) -> dict:
    import sys
    import shap
    import torch
    sys.path.insert(0, str(REPO / "src" / "detection"))
    from text_normalize import normalize_text
    from stylometric import load_nlp, stylometric_features
    from hybrid_fusion import gpt2_perplexity

    feat_cols = json.loads((HYBRID / "feat_cols.json").read_text(encoding="utf-8"))
    gbm = pickle.load(open(HYBRID / "gbm_ppl.pkl", "rb"))
    cols = feat_cols + ["gpt2_ppl"]

    norm = normalize_text(text)
    feats = stylometric_features(norm, load_nlp())
    feats["gpt2_ppl"] = float(gpt2_perplexity([norm], torch.device("cpu"))[0])
    X = pd.DataFrame([feats])[cols]

    # Reference distribution: HUMAN essays in the corpus (what "typical student writing" means).
    ref = pd.read_parquet(FEATS)
    ppl = pd.read_parquet(PPL_CACHE)
    ref = ref.merge(ppl, on=["id", "label"], validate="one_to_one")
    human = ref[ref["label"] == 0]

    sv = shap.TreeExplainer(gbm).shap_values(X)[0]
    order = np.argsort(np.abs(sv))[::-1][:top_k]

    rows, sentences = [], []
    for i in order:
        feat = cols[i]
        v = float(X.iloc[0, i])
        med = float(human[feat].median())
        toward_ai = sv[i] > 0
        label, hi_t, lo_t = PLAIN.get(feat, _fallback(feat))
        body = (hi_t if v >= med else lo_t).format(v=v, m=med)
        push = "toward AI" if toward_ai else "toward human"
        sentences.append(f"{body}, which here points {push}.")
        rows.append({"feature": feat, "label": label, "shap": float(sv[i]),
                     "value": v, "human_median": med,
                     "human_p10": float(human[feat].quantile(0.10)),
                     "human_p90": float(human[feat].quantile(0.90))})

    _draw_card(rows, out_png)
    return {"png_path": str(out_png), "sentences": sentences, "features": rows}


def _fmt(x: float) -> str:
    ax = abs(x)
    return f"{x:.0f}" if ax >= 10 else f"{x:.1f}" if ax >= 1 else f"{x:.2f}" if ax >= 0.1 else f"{x:.3f}"


def _draw_card(rows: list[dict], out_png: Path) -> None:
    """One strip per habit: a grey band for the middle 80 percent of real student essays, a line
    at the typical value (median), and a dot where THIS essay sits, coloured by which way the
    habit pushed the score. No model units anywhere; a lecturer reads position, not magnitude."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    AI, HUMAN, BAND, INKC, GREYC = "#d98e3b", "#2b6777", "#e3e7ea", "#222831", "#52616b"
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.4, 0.78 * n + 2.0))

    for k, r in enumerate(rows):
        y = n - 1 - k
        lo, hi, med, v = r["human_p10"], r["human_p90"], r["human_median"], r["value"]
        span_lo = min(lo, v); span_hi = max(hi, v)
        pad = (span_hi - span_lo) * 0.12 or abs(span_hi) * 0.12 or 1.0
        a, b = span_lo - pad, span_hi + pad

        def nx(x):
            return (x - a) / (b - a)

        ax.barh(y, nx(hi) - nx(lo), left=nx(lo), height=0.34, color=BAND, zorder=1)
        ax.plot([nx(med), nx(med)], [y - 0.24, y + 0.24], color=GREYC, lw=1.6, zorder=2)
        dot = AI if r["shap"] > 0 else HUMAN
        ax.scatter([nx(v)], [y], s=150, color=dot, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(-0.02, y, r["label"], ha="right", va="center", fontsize=12, color=INKC)
        ax.text(1.02, y, f"this essay {_fmt(v)}  ·  typical {_fmt(med)}",
                ha="left", va="center", fontsize=9.5, color=GREYC)
        push = "pushes toward AI" if r["shap"] > 0 else "pushes toward human"
        ax.text(nx(v), y + 0.34, push, ha="center", va="bottom", fontsize=8.5,
                color=dot, fontweight="bold")

    ax.set_xlim(-0.32, 1.42)
    ax.set_ylim(-0.7, n - 0.1)
    ax.axis("off")
    fig.suptitle("This essay's writing habits, against typical student writing",
                 fontsize=13, fontweight="bold", color=INKC, x=0.03, y=0.97, ha="left")
    fig.text(0.03, 0.03,
             "grey band = middle 80% of real student essays   |   line = typical (median)   |   "
             "dot = this essay, coloured by push direction",
             fontsize=8.5, color=GREYC)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)
