"""Score a single submission with the saved hybrid detector (Section 6.7).

Loads the fitted pieces from models/hybrid/ (the perplexity-augmented style model and the logistic
fuser) and scores one text: DeBERTa probability and the style-plus-perplexity probability are fused
into the hybrid probability the pipeline reports. Falls back to the transformer alone if the hybrid
has not been saved, so the assembler never breaks.

    from hybrid_detect import hybrid_detect
    hybrid_detect(text)   # {"prob_ai", "flagged", "detector", "components"}
"""

from __future__ import annotations

import glob
import json
import pickle
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HYBRID = REPO / "models" / "hybrid"
DETECTOR = REPO / "models" / "detector"
MAXLEN = 512


def _latest_detector() -> str:
    cps = glob.glob(str(DETECTOR / "checkpoint-*"))
    return max(cps, key=lambda p: int(p.split("-")[-1])) if cps else str(DETECTOR)


def _transformer_prob(text: str) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ckpt = _latest_detector()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()
    import sys
    sys.path.insert(0, str(REPO / "src" / "detection"))
    from text_normalize import normalize_text
    enc = tok(normalize_text(text), truncation=True, max_length=MAXLEN, return_tensors="pt")
    with torch.no_grad():
        return float(torch.softmax(model(**enc).logits, -1)[0, 1])


def hybrid_detect(text: str) -> dict:
    p_deb = _transformer_prob(text)
    if not (HYBRID / "fuser.pkl").exists():
        return {"prob_ai": round(p_deb, 4), "flagged": p_deb >= 0.5, "detector": "transformer",
                "components": {"transformer": round(p_deb, 4)}}

    import sys
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO / "src" / "detection"))
    from text_normalize import normalize_text
    from stylometric import load_nlp, stylometric_features
    from hybrid_fusion import gpt2_perplexity

    feat_cols = json.loads((HYBRID / "feat_cols.json").read_text(encoding="utf-8"))
    gbm = pickle.load(open(HYBRID / "gbm_ppl.pkl", "rb"))
    fuser = pickle.load(open(HYBRID / "fuser.pkl", "rb"))

    norm = normalize_text(text)
    feats = stylometric_features(norm, load_nlp())
    feats["gpt2_ppl"] = float(gpt2_perplexity([norm], torch.device("cpu"))[0])
    import pandas as pd
    X = pd.DataFrame([feats])[feat_cols + ["gpt2_ppl"]]
    p_style = float(gbm.predict_proba(X)[0, 1])
    p_hy = float(fuser.predict_proba(np.array([[p_deb, p_style]]))[0, 1])
    return {"prob_ai": round(p_hy, 4), "flagged": p_hy >= 0.5, "detector": "hybrid",
            "components": {"transformer": round(p_deb, 4), "style_plus_perplexity": round(p_style, 4)}}
