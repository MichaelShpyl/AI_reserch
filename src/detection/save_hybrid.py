"""Persist the fitted hybrid detector so the pipeline can call it on a single new submission.

hybrid_fusion.py evaluates the hybrid but fits its pieces in memory and throws them away. The output
assembler needs the trained detector at guide time, and the dissertation says the hybrid, not the bare
transformer, is the detector the pipeline should call (Section 6.7). This script fits the same pieces
on the home corpus and saves them to models/hybrid/, and it caches GPT-2 perplexity for the home
essays so this is cheap to repeat. The companion loader is hybrid_detect.py.

    python src/detection/save_hybrid.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
CLEAN = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
FEATS = REPO / "data" / "processed" / "stylometric_features.parquet"
PPL_CACHE = REPO / "data" / "processed" / "gpt2_perplexity_home.parquet"
HYBRID = REPO / "models" / "hybrid"
SEED = 42
DROP = ["n_words", "n_sents", "id", "label", "split", "native"]

import sys
sys.path.insert(0, str(REPO / "src" / "detection"))
from hybrid_fusion import gpt2_perplexity, deberta_probs  # noqa: E402


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = pd.read_parquet(CLEAN)
    feats = pd.read_parquet(FEATS)
    df = feats.merge(corpus[["id", "label", "text"]], on=["id", "label"], validate="one_to_one")

    if PPL_CACHE.exists():
        print("Loading cached perplexity", flush=True)
        ppl = pd.read_parquet(PPL_CACHE)
        df = df.merge(ppl, on=["id", "label"], validate="one_to_one")
    else:
        print("Computing GPT-2 perplexity for the home corpus (cached after this run)", flush=True)
        df["gpt2_ppl"] = gpt2_perplexity(df["text"].tolist(), device)
        df[["id", "label", "gpt2_ppl"]].to_parquet(PPL_CACHE, index=False)

    feat_cols = [c for c in feats.columns if c not in DROP]
    tr = df[df["split"] == "train"]; va = df[df["split"] == "val"]

    gbm = GradientBoostingClassifier(random_state=SEED)
    gbm.fit(tr[feat_cols + ["gpt2_ppl"]], tr["label"])

    print("DeBERTa probabilities on the validation split (to fit the fuser)", flush=True)
    p_deb_val = deberta_probs(va["text"].tolist(), device)
    p_style_val = gbm.predict_proba(va[feat_cols + ["gpt2_ppl"]])[:, 1]
    fuser = LogisticRegression()
    fuser.fit(np.column_stack([p_deb_val, p_style_val]), va["label"])

    HYBRID.mkdir(parents=True, exist_ok=True)
    with open(HYBRID / "gbm_ppl.pkl", "wb") as f:
        pickle.dump(gbm, f)
    with open(HYBRID / "fuser.pkl", "wb") as f:
        pickle.dump(fuser, f)
    (HYBRID / "feat_cols.json").write_text(json.dumps(feat_cols), encoding="utf-8")
    print(f"Saved hybrid to {HYBRID.relative_to(REPO)} "
          f"(fuser coefficients {fuser.coef_.round(3).tolist()})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
