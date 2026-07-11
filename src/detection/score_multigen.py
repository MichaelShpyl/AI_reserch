"""Score the multi-generator test slice: unseen commercial generators, on home ground.

Companion to generation/multigen_test_slice.py. Takes the Gemini and GPT essays generated with the
original corpus recipe, plus the matched human test essays they were sourced from, and scores all of
them with the transformer of record and the saved hybrid. Reports, per generator: detection rate
(how many of the unseen generator's essays are caught), with the human essays' false-positive rate
alongside as the base rate, and the length-match ratios so a length tell can be ruled in or out.

    python src/detection/score_multigen.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
SLICE = REPO / "data" / "processed" / "multigen_test"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
HYBRID = REPO / "models" / "hybrid"
OUT = REPO / "outputs" / "multigen_detection.json"
FIGS = REPO / "dissertation" / "figures"

import sys
sys.path.insert(0, str(REPO / "src" / "detection"))
from text_normalize import normalize_text  # noqa: E402
from hybrid_fusion import gpt2_perplexity, deberta_probs  # noqa: E402
from stylometric import load_nlp, stylometric_features  # noqa: E402


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = json.loads((SLICE / "state.json").read_text(encoding="utf-8"))

    rows = []
    human_seen = set()
    for key, meta in state["done"].items():
        eid, gen = key.rsplit("_", 1)
        f = SLICE / f"{key}.txt"
        if not f.exists():
            continue
        rows.append({"id": key, "group": gen, "label": 1,
                     "text": f.read_text(encoding="utf-8", errors="ignore"),
                     "len_ratio": round(meta["got_words"] / max(meta["source_words"], 1), 3)})
        if eid not in human_seen:
            hf = CORPUS_TXT / f"{eid}.txt"
            if hf.exists():
                rows.append({"id": eid, "group": "human", "label": 0,
                             "text": hf.read_text(encoding="utf-8", errors="ignore"),
                             "len_ratio": 1.0})
                human_seen.add(eid)
    df = pd.DataFrame(rows)
    print(f"{len(df)} texts: " + ", ".join(f"{g}={n}" for g, n in df.group.value_counts().items()),
          flush=True)

    texts = [normalize_text(t) for t in df["text"]]
    print("transformer probabilities ...", flush=True)
    p_deb = deberta_probs(texts, device)
    print("style features + perplexity ...", flush=True)
    nlp = load_nlp()
    F = pd.DataFrame([stylometric_features(t, nlp) for t in texts])
    F["gpt2_ppl"] = gpt2_perplexity(texts, device)
    feat_cols = json.loads((HYBRID / "feat_cols.json").read_text(encoding="utf-8"))
    gbm = pickle.load(open(HYBRID / "gbm_ppl.pkl", "rb"))
    fuser = pickle.load(open(HYBRID / "fuser.pkl", "rb"))
    p_style = gbm.predict_proba(F[feat_cols + ["gpt2_ppl"]])[:, 1]
    p_hy = fuser.predict_proba(np.column_stack([p_deb, p_style]))[:, 1]

    result = {"n": int(len(df)), "groups": {}}
    for g in sorted(df.group.unique()):
        m = (df.group == g).values
        det_t = float((p_deb[m] >= 0.5).mean())
        det_h = float((p_hy[m] >= 0.5).mean())
        entry = {"n": int(m.sum()),
                 "flag_rate_transformer": round(det_t, 4),
                 "flag_rate_hybrid": round(det_h, 4)}
        if g != "human":
            entry["mean_length_ratio"] = round(float(df.loc[m, "len_ratio"].mean()), 3)
        result["groups"][g] = entry
        print(f"{g:8s} n={m.sum():3d} flagged: transformer {det_t:.2f} hybrid {det_h:.2f}", flush=True)

    result["reading"] = ("For the AI groups the flag rate is the detection rate on a generator the "
                         "detector never saw, produced with the original corpus recipe; for the "
                         "human group it is the false-positive rate on the same essays' true authors.")
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(result)
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = ["human", "gemini", "openai"]
    groups = [g for g in groups if g in result["groups"]]
    names = {"human": "real human\n(false positives)", "gemini": "Gemini\n(never seen)",
             "openai": "GPT-4o-mini\n(never seen)"}
    x = np.arange(len(groups))
    t_rates = [result["groups"][g]["flag_rate_transformer"] * 100 for g in groups]
    h_rates = [result["groups"][g]["flag_rate_hybrid"] * 100 for g in groups]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x - 0.2, t_rates, 0.4, label="transformer", color="#52616B")
    ax.bar(x + 0.2, h_rates, 0.4, label="hybrid", color="#2b6777")
    for xi, (tv, hv) in enumerate(zip(t_rates, h_rates)):
        ax.text(xi - 0.2, tv + 1.5, f"{tv:.0f}%", ha="center", fontsize=9)
        ax.text(xi + 0.2, hv + 1.5, f"{hv:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([names[g] for g in groups], fontsize=10)
    ax.set_ylabel("flagged as AI (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Unseen commercial generators on the home corpus recipe:\n"
                 "does the detector catch what it never trained on?",
                 fontsize=11.5, fontweight="bold", color="#222831")
    ax.legend(fontsize=9.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_multigen_detection.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_multigen_detection.png")


if __name__ == "__main__":
    raise SystemExit(main())
