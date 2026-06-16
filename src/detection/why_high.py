"""Explain WHY the cleaned-text detector still scores ~99%: is it deep style, or is a
shallow tell still doing the work? Run after audit_detector.py.

Five probes on the cleaned corpus (markup already removed):

  1. Locale spelling. UK students write British spelling; Llama defaults to American.
     If the human side is heavily British and the AI side heavily American, that is a
     real but shallow giveaway (a generator-locale tell, not deep "AI-ness").
  2. Contractions. A quick second shallow-style check (don't, it's, can't ...).
  3. Self-similarity. Mean pairwise cosine similarity of the TF-IDF vectors within each
     class. If the AI essays are far more similar to each other than the human essays are,
     the AI class is a tight, uniform cluster, which is exactly why one generator is easy
     to spot.
  4. Style-space picture. Project the function-word vectors (no topic information) to 2D
     and plot the two classes, so the separation is visible.
  5. Borderline cases. The human test essays the style model rates most AI-like, i.e. the
     closest calls, so the errors can be characterised honestly.

    python src/detection/why_high.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

REPO = Path(__file__).resolve().parents[2]
CLEAN = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
FIGS = REPO / "dissertation" / "figures"
OUT = REPO / "outputs" / "why_high.json"

# Common British/American spelling pairs that show up in academic prose.
BR_AM = [
    ("colour", "color"), ("behaviour", "behavior"), ("favour", "favor"),
    ("labour", "labor"), ("organise", "organize"), ("organisation", "organization"),
    ("analyse", "analyze"), ("recognise", "recognize"), ("realise", "realize"),
    ("emphasise", "emphasize"), ("summarise", "summarize"), ("characterise", "characterize"),
    ("utilise", "utilize"), ("minimise", "minimize"), ("maximise", "maximize"),
    ("criticise", "criticize"), ("centre", "center"), ("theatre", "theater"),
    ("defence", "defense"), ("licence", "license"), ("programme", "program"),
    ("modelling", "modeling"), ("labelled", "labeled"), ("travelled", "traveled"),
    ("fulfil", "fulfill"), ("catalogue", "catalog"), ("dialogue", "dialog"),
    ("judgement", "judgment"), ("ageing", "aging"), ("metre", "meter"),
]
BR_ONLY = ["whilst", "amongst", "towards", "learnt", "spelt", "burnt"]
CONTRACTIONS = ["n't", "'re", "'ve", "'ll", "'m", "it's", "that's", "there's"]


def count_patterns(df, terms, word_boundary=True):
    """Return {source: hits per 1000 words} for a list of substrings/words."""
    res = {}
    for src, g in df.groupby("source"):
        words = int(g["text"].str.split().str.len().sum())
        hits = 0
        for t in g["text"]:
            low = t.lower()
            for term in terms:
                if word_boundary and term.isalpha():
                    hits += len(re.findall(rf"\b{re.escape(term)}\b", low))
                else:
                    hits += low.count(term)
        res[src] = round(1000 * hits / max(words, 1), 3)
    return res


def locale_spelling(df) -> dict:
    british = [b for b, _ in BR_AM] + BR_ONLY
    american = [a for _, a in BR_AM]
    return {
        "british_spellings_per_1k": count_patterns(df, british),
        "american_spellings_per_1k": count_patterns(df, american),
        "contractions_per_1k": count_patterns(df, CONTRACTIONS, word_boundary=False),
        "reading": "If human >> AI on British and AI >> human on American, locale spelling "
                   "is a shallow tell. It is real (UK students vs a US-default model) but "
                   "would vanish if the generator were prompted in British English.",
    }


def self_similarity(df) -> dict:
    # Measure in function-word (style) space, not full text: otherwise topic variety
    # swamps the style signal and both classes look equally diverse.
    fw = sorted(ENGLISH_STOP_WORDS)
    vec = TfidfVectorizer(vocabulary=fw, sublinear_tf=True)
    X = vec.fit_transform(df["text"])
    out = {}
    for src in ("human", "ai"):
        idx = np.where(df["source"].values == src)[0]
        S = cosine_similarity(X[idx])
        iu = np.triu_indices_from(S, k=1)
        out[src] = round(float(S[iu].mean()), 4)
    out["reading"] = ("Mean within-class similarity in style space. Higher means a tighter, "
                      "more uniform cluster. AI essays from one model tend to be more alike "
                      "in style than human essays, which is why one generator is easy to spot.")
    return out


def style_scatter(df) -> None:
    fw = sorted(ENGLISH_STOP_WORDS)
    vec = TfidfVectorizer(vocabulary=fw, sublinear_tf=True)
    X = vec.fit_transform(df["text"])
    XY = TruncatedSVD(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.6, 6))
    for src, color, label in [("human", "#2b6777", "Human"), ("ai", "#d98e3b", "AI")]:
        m = df["source"].values == src
        ax.scatter(XY[m, 0], XY[m, 1], s=14, c=color, alpha=0.55, label=label, edgecolors="none")
    ax.set_title("Essays in function-word style space (no topic words)\n"
                 "Two clean clusters: this is why the task is easy",
                 fontsize=12.5, fontweight="bold", color="#222831")
    ax.set_xlabel("style dimension 1"); ax.set_ylabel("style dimension 2")
    ax.legend(fontsize=11, loc="best")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_why_style_clusters.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_why_style_clusters.png")


def borderline(df) -> dict:
    fw = sorted(ENGLISH_STOP_WORDS)
    vec = TfidfVectorizer(vocabulary=fw, sublinear_tf=True)
    tr = df[df["split"] == "train"]; te = df[df["split"] == "test"]
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(vec.fit_transform(tr["text"]), tr["label"])
    proba = clf.predict_proba(vec.transform(te["text"]))[:, 1]  # P(AI)
    te = te.assign(p_ai=proba)
    humans = te[te["label"] == 0].sort_values("p_ai", ascending=False)
    ai = te[te["label"] == 1].sort_values("p_ai", ascending=True)
    return {
        "most_AI_like_humans": humans.head(5)[["id", "native", "p_ai"]]
        .assign(p_ai=lambda d: d["p_ai"].round(3)).to_dict("records"),
        "most_human_like_AI": ai.head(5)[["id", "p_ai"]]
        .assign(p_ai=lambda d: d["p_ai"].round(3)).to_dict("records"),
        "reading": "The closest calls. The most AI-like human essays are the ones to read "
                   "when characterising errors; they are usually the most polished and formal.",
    }


def main() -> int:
    df = pd.read_parquet(CLEAN)
    report = {
        "locale_spelling": locale_spelling(df),
        "self_similarity": self_similarity(df),
        "borderline": borderline(df),
    }
    style_scatter(df)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== WHY THE SCORE IS HIGH ===")
    ls = report["locale_spelling"]
    print("British per 1k:", ls["british_spellings_per_1k"])
    print("American per 1k:", ls["american_spellings_per_1k"])
    print("Contractions per 1k:", ls["contractions_per_1k"])
    print("Within-class similarity:", report["self_similarity"])
    print("Most AI-like humans:", report["borderline"]["most_AI_like_humans"])
    print(f"\nSaved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
