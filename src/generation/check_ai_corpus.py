"""Validate the generated AI essays against their human sources.

Checks the two properties the whole detector design depends on:
  - Length: the AI essays match the human source lengths (so the detector cannot
    learn length instead of style).
  - Topic: each AI essay is about the same topic as its human source (keyword overlap).

Also dedupes the metadata and saves an AI-vs-human length figure.

    python src/generation/check_ai_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "generation"))
from generate_ai_essays import extract_keywords  # noqa: E402

AI_DIR = REPO / "data" / "processed" / "ai_essays"
META = REPO / "data" / "processed" / "ai_essays_meta.csv"
SAMPLE = REPO / "data" / "processed" / "bawe_human_sample.csv"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
FIG = REPO / "dissertation" / "figures" / "fig7_ai_vs_human_length.png"
SUMMARY = REPO / "outputs" / "ai_corpus_check.txt"

SEED = 42


def main() -> int:
    out = []
    def say(s=""):
        print(s); out.append(s)

    # Dedupe metadata.
    meta = pd.read_csv(META)
    n0 = len(meta)
    meta = meta.drop_duplicates(subset="id", keep="last")
    meta.to_csv(META, index=False)
    n_files = len(list(AI_DIR.glob("*.txt")))
    say("=" * 64)
    say("AI CORPUS CHECK")
    say("=" * 64)
    say(f"AI essay files: {n_files}   metadata rows: {len(meta)} (deduped from {n0})")

    # Join with the human sample to compare against authoritative human lengths.
    sample = pd.read_csv(SAMPLE)[["id", "words", "title"]]  # meta already has discipline
    sample = sample.rename(columns={"words": "human_words"})
    df = meta.merge(sample, on="id", how="inner")
    df["ai_words"] = df["actual_words"]
    df["ratio"] = df["ai_words"] / df["human_words"]
    say(f"Matched AI/human pairs: {len(df)}")
    say("")

    # ---- Length cross-check ----
    r = float(np.corrcoef(df["ai_words"], df["human_words"])[0, 1])
    within10 = float((df["ratio"].between(0.9, 1.1)).mean() * 100)
    within20 = float((df["ratio"].between(0.8, 1.2)).mean() * 100)
    say("LENGTH CROSS-CHECK (AI essay words vs human source words):")
    say(f"    Pearson r:            {r:.4f}")
    say(f"    mean ratio:           {df['ratio'].mean():.3f}")
    say(f"    median ratio:         {df['ratio'].median():.3f}")
    say(f"    within +/-10%:        {within10:.1f}%")
    say(f"    within +/-20%:        {within20:.1f}%")
    say(f"    human words: mean {df['human_words'].mean():.0f}, median {df['human_words'].median():.0f}")
    say(f"    AI words:    mean {df['ai_words'].mean():.0f}, median {df['ai_words'].median():.0f}")
    # Distribution overlap via histogram intersection.
    hi = int(max(df["human_words"].max(), df["ai_words"].max()))
    bins = np.linspace(0, hi, 40)
    h_h, _ = np.histogram(df["human_words"], bins=bins)
    h_a, _ = np.histogram(df["ai_words"], bins=bins)
    p_h, p_a = h_h / h_h.sum(), h_a / h_a.sum()
    overlap = float(np.minimum(p_h, p_a).sum() * 100)
    say(f"    length-distribution overlap: {overlap:.1f}%")
    say("")

    # ---- Topic spot-check (keyword overlap) ----
    rng = np.random.RandomState(SEED)
    ids = df["id"].astype(str).tolist()
    pick = [ids[i] for i in rng.choice(len(ids), size=min(10, len(ids)), replace=False)]
    say("TOPIC SPOT-CHECK (keyword overlap, human source vs AI essay, 10 random):")
    overlaps = []
    for rid in pick:
        hp = CORPUS_TXT / f"{rid}.txt"
        ap = AI_DIR / f"{rid}.txt"
        if not (hp.exists() and ap.exists()):
            continue
        hk = set(extract_keywords(hp.read_text(encoding="utf-8", errors="ignore"), 10))
        ak = set(extract_keywords(ap.read_text(encoding="utf-8", errors="ignore"), 10))
        jac = len(hk & ak) / len(hk | ak) if (hk | ak) else 0.0
        overlaps.append(jac)
        disc = df.loc[df["id"] == rid, "discipline"].iloc[0]
        say(f"    {rid} ({disc[:18]:<18}) overlap={jac:.2f}  shared={sorted(hk & ak)[:5]}")
    if overlaps:
        say(f"    mean keyword overlap (Jaccard, top-10): {np.mean(overlaps):.2f}")
    say("")

    # ---- Figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5))
        b = np.linspace(0, 6000, 31)
        ax.hist(df["human_words"], bins=b, alpha=0.55, color="#2b6777",
                label=f"human (mean {df['human_words'].mean():.0f})")
        ax.hist(df["ai_words"], bins=b, alpha=0.55, color="#d98e3b",
                label=f"AI (mean {df['ai_words'].mean():.0f})")
        ax.set_xlabel("Essay length (words)")
        ax.set_ylabel("Number of essays")
        ax.set_title("Human vs AI essay length (matched at generation)",
                     fontsize=12.5, fontweight="bold", color="#222831")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG, dpi=200, facecolor="white"); plt.close(fig)
        say(f"Saved figure {FIG.relative_to(REPO)}")
    except Exception as e:
        say(f"(figure skipped: {e})")

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Saved {SUMMARY.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
