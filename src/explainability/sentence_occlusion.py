"""A per-essay account of the transformer half, built from behaviour rather than attribution.

Chapter 5 leaves one declared gap: the hybrid's transformer contributes to every score without a
per-essay explanation of its own, because its token-level attributions failed the faithfulness
test. This module tries the level between token and essay: delete one sentence at a time and
measure how the AI-probability moves. Occlusion is faithful by construction (it reports what the
model actually did when the text changed), and sentences are the unit a lecturer can quote.

Two jobs:
  explain_sentences(text)  the top flag-carrying sentences for one submission, for the guide
  main()                   the faithfulness experiment: on test AI essays, does removing the
                           top-3 occlusion sentences drop the score more than removing 3 random
                           sentences? The same comprehensiveness logic that failed at token level
                           (Section 5.3), now at sentence level. Either outcome is a finding.

    python src/explainability/sentence_occlusion.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "detection"))
from text_normalize import normalize_text  # noqa: E402

OUT = REPO / "outputs" / "sentence_occlusion.json"
FIGS = REPO / "dissertation" / "figures"
CORPUS = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
MAXLEN = 512
BATCH = 16
SEED = 42
N_ESSAYS = 30
TOP_K = 3
N_RANDOM = 5

_model, _tok, _device = None, None, None


def _load():
    global _model, _tok, _device
    if _model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        ckpts = sorted((REPO / "models" / "detector").glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tok = AutoTokenizer.from_pretrained(str(ckpts[-1]))
        _model = (AutoModelForSequenceClassification.from_pretrained(str(ckpts[-1]))
                  .to(_device).eval())
    return _model, _tok, _device


def _probs(texts: list[str]) -> np.ndarray:
    model, tok, device = _load()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to(device)
            out.extend(torch.softmax(model(**enc).logits, -1)[:, 1].tolist())
    return np.array(out)


def _logodds(texts: list[str]) -> np.ndarray:
    """Log-odds of the AI class. The in-domain detector saturates near probability 1.0, where
    sentence removals vanish below rounding; log-odds keeps the movement measurable."""
    model, tok, device = _load()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to(device)
            lg = model(**enc).logits
            out.extend((lg[:, 1] - lg[:, 0]).tolist())
    return np.array(out)


def split_sentences(text: str) -> list[str]:
    try:
        from stylometric import load_nlp
        nlp = load_nlp()
        sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    except Exception:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sents


def explain_sentences(text: str, top_k: int = TOP_K) -> dict:
    """The sentences whose removal most lowers the AI-probability, with the measured drops."""
    norm = normalize_text(text)
    sents = split_sentences(norm)
    full = float(_logodds([norm])[0])
    variants = [" ".join(sents[:i] + sents[i + 1:]) for i in range(len(sents))]
    drops = full - _logodds(variants)
    order = np.argsort(drops)[::-1][:top_k]
    return {"logodds_full": round(full, 3), "n_sentences": len(sents),
            "top": [{"sentence": sents[i], "drop_logodds": round(float(drops[i]), 3)}
                    for i in order]}


def main() -> int:
    import pandas as pd
    from scipy.stats import wilcoxon

    df = pd.read_parquet(CORPUS)
    ai = df[(df["label"] == 1) & (df["split"] == "test")].sample(n=N_ESSAYS, random_state=SEED)
    rng = random.Random(SEED)

    rows = []
    for k, (_, r) in enumerate(ai.iterrows(), 1):
        norm = normalize_text(r["text"])
        sents = split_sentences(norm)
        if len(sents) < TOP_K + 2:
            continue
        full = float(_logodds([norm])[0])
        variants = [" ".join(sents[:i] + sents[i + 1:]) for i in range(len(sents))]
        drops = full - _logodds(variants)
        top = list(np.argsort(drops)[::-1][:TOP_K])
        wo_top = " ".join(s for i, s in enumerate(sents) if i not in top)
        drop_top = full - float(_logodds([wo_top])[0])
        rand_drops = []
        for _ in range(N_RANDOM):
            idx = set(rng.sample(range(len(sents)), TOP_K))
            wo_r = " ".join(s for i, s in enumerate(sents) if i not in idx)
            rand_drops.append(full - float(_logodds([wo_r])[0]))
        rows.append({"id": str(r["id"]), "logodds_full": round(full, 3),
                     "n_sentences": len(sents),
                     "drop_top3": round(drop_top, 4),
                     "drop_random3_mean": round(float(np.mean(rand_drops)), 4)})
        print(f"[{k}/{N_ESSAYS}] {r['id']}: full {full:.3f}  top3 {drop_top:+.4f}  "
              f"rand3 {np.mean(rand_drops):+.4f}", flush=True)

    t = np.array([x["drop_top3"] for x in rows])
    rd = np.array([x["drop_random3_mean"] for x in rows])
    stat = wilcoxon(t, rd)
    report = {
        "design": f"{len(rows)} test AI essays; remove the top-{TOP_K} occlusion sentences "
                  f"versus {TOP_K} random sentences ({N_RANDOM} draws averaged), paired",
        "mean_drop_top3": round(float(t.mean()), 4),
        "mean_drop_random3": round(float(rd.mean()), 4),
        "ratio": round(float(t.mean() / rd.mean()), 2) if rd.mean() else None,
        "wilcoxon_p": round(float(stat.pvalue), 6),
        "top3_higher_in": int((t > rd).sum()), "n": len(rows),
        "per_essay": rows,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("mean_drop_top3", "mean_drop_random3", "ratio", "wilcoxon_p",
                       "top3_higher_in", "n")}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, (vals, lab, c) in enumerate([(t, f"top-{TOP_K} occlusion sentences", "#a63d2e"),
                                        (rd, f"{TOP_K} random sentences", "#52616b")]):
        ax.bar(i, vals.mean(), 0.55, color=c, label=lab)
        ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
        ax.errorbar(i, vals.mean(), yerr=ci, color="#222831", capsize=5, lw=1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["remove top-3\n(by occlusion)", "remove 3 random"])
    ax.set_ylabel("drop in log-odds of the AI class")
    passed = stat.pvalue < 0.05 and t.mean() > rd.mean()
    title = ("Sentence-level occlusion: targeted removal beats random (paired test)"
             if passed else
             "Sentence-level occlusion: targeted removal does not beat random")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#222831")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_sentence_occlusion.png", dpi=200, facecolor="white")
    print("Saved fig_sentence_occlusion.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
