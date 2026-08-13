"""Why every flagged essay scores about 0.957 (Section 6.7).

The reported hybrid score clusters so tightly on the AI side that the number looks suspicious: all
100 held-out AI essays land between 0.9539 and 0.9574. This script establishes why, and separates
the two causes, because they have different consequences.

The first cause is arithmetic and is a property of the combiner, not of any essay. The fuser is a
logistic regression over two probabilities. Both inputs are confined to [0, 1], so the logit is
confined to [b, w1 + w2 + b], and the reported score can never leave the interval that implies. No
submission can ever score above the ceiling or below the floor, whatever it contains.

The second cause is that in-domain both halves saturate, so nearly every essay is pushed to one end
of that interval. That is a fact about this corpus paired with one generator, and Chapter 6 already
treats it as a limitation.

The distinction matters for how the number should be read. A score of 0.957 is not a claim that an
essay is 95.7 percent likely to be machine written. It is what the combiner emits when both readers
agree and neither has doubt left to express.

The script also finds the essays where the two halves disagree, since those are the only ones where
fusion changes an in-domain decision.

    python src/detection/fusion_range.py
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
CLEAN = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
HYBRID = REPO / "models" / "hybrid"
DETECTOR = REPO / "models" / "detector"
OUT = REPO / "outputs" / "fusion_range.json"
MAXLEN = 512
BATCH = 8
THRESHOLD = 0.5
WORKED_ID = "3108a"   # the submission Chapter 7 walks through

sys.path.insert(0, str(REPO / "src" / "detection"))
from text_normalize import normalize_text  # noqa: E402
from stylometric import load_nlp, stylometric_features  # noqa: E402
from hybrid_fusion import gpt2_perplexity  # noqa: E402


def latest_checkpoint() -> str:
    cps = glob.glob(str(DETECTOR / "checkpoint-*"))
    return max(cps, key=lambda p: int(p.split("-")[-1])) if cps else str(DETECTOR)


def transformer_probs(texts: list[str]) -> np.ndarray:
    """DeBERTa probabilities, batched, one model load."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = latest_checkpoint()
    tok = AutoTokenizer.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(torch.softmax(model(**enc).logits, -1)[:, 1].cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def style_probs(texts: list[str]) -> np.ndarray:
    """Style-plus-perplexity probabilities from the saved gradient-boosted model."""
    feat_cols = json.loads((HYBRID / "feat_cols.json").read_text(encoding="utf-8"))
    gbm = pickle.load(open(HYBRID / "gbm_ppl.pkl", "rb"))
    nlp = load_nlp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for i, t in enumerate(texts):
        f = stylometric_features(t, nlp)
        f["gpt2_ppl"] = float(gpt2_perplexity([t], device)[0])
        rows.append(f)
        if (i + 1) % 25 == 0:
            print(f"    style features {i + 1}/{len(texts)}", flush=True)
    X = pd.DataFrame(rows)[feat_cols + ["gpt2_ppl"]]
    return gbm.predict_proba(X)[:, 1]


def main() -> None:
    fuser = pickle.load(open(HYBRID / "fuser.pkl", "rb"))
    w1, w2 = (float(x) for x in fuser.coef_[0])
    b = float(fuser.intercept_[0])
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))          # noqa: E731
    ceiling, floor = float(sig(w1 + w2 + b)), float(sig(b))

    print("Fuser: sigmoid(w1 * p_transformer + w2 * p_style + b)")
    print(f"  w1 ={w1:.4f}  w2 ={w2:.4f}  b ={b:.4f}")
    print(f"  Both inputs lie in [0, 1], so the score lies in [{floor:.4f}, {ceiling:.4f}].")
    print("  Those two numbers are properties of the combiner. No text can escape them.\n")

    df = pd.read_parquet(CLEAN)
    te = df[df.split == "test"].reset_index(drop=True)
    texts = [normalize_text(t) for t in te.text]
    print(f"Scoring the {len(te)} held-out test essays.")
    p_t = transformer_probs(texts)
    p_s = style_probs(texts)
    p_f = fuser.predict_proba(np.column_stack([p_t, p_s]))[:, 1]

    s = pd.DataFrame({"id": te.id, "label": te.label, "group": te.disciplinary_group,
                      "transformer": p_t, "style": p_s, "fused": p_f})
    ai, hu = s[s.label == 1], s[s.label == 0]
    # Pull the style column out by name. Attribute access (ai.style) resolves to the pandas
    # Styler object rather than this column, and fails later with a confusing error.
    ai_style, hu_style = ai["style"], hu["style"]

    # How much of the [floor, ceiling] interval does each class actually use?
    span = ceiling - floor
    ai_span = float(ai.fused.max() - ai.fused.min())
    hu_span = float(hu.fused.max() - hu.fused.min())

    # Disagreement: the only essays where fusion changes an in-domain decision.
    rescued = hu[(hu["transformer"] >= THRESHOLD) & (hu["fused"] < THRESHOLD)]
    missed = ai[(ai["transformer"] >= THRESHOLD) & (ai["fused"] < THRESHOLD)]

    print(f"\nAI essays (n={len(ai)}):")
    print(f"  transformer {ai.transformer.min():.4f} to {ai.transformer.max():.4f}")
    print(f"  style       {ai_style.min():.4f} to {ai_style.max():.4f}")
    print(f"  fused       {ai.fused.min():.4f} to {ai.fused.max():.4f}")
    print(f"  every AI essay sits within {(ceiling - ai.fused.min()) * 100:.2f} points of the ceiling")
    print(f"\nHuman essays (n={len(hu)}):")
    print(f"  fused       {hu.fused.min():.4f} to {hu.fused.max():.4f}")
    print(f"\nThe two classes together use {(ai_span + hu_span) / span * 100:.1f} percent of the "
          f"combiner's available range.")
    print(f"\nHuman essays the transformer alone would flag, and fusion rescues: {len(rescued)}")
    for r in rescued.sort_values("transformer", ascending=False).itertuples():
        print(f"  {r.id}: transformer {r.transformer:.4f}  style {r.style:.4f}  -> {r.fused:.4f}")
    print(f"AI essays the transformer catches and fusion loses: {len(missed)}")
    for r in missed.itertuples():
        print(f"  {r.id}: transformer {r.transformer:.4f}  style {r.style:.4f}  -> {r.fused:.4f}")

    # Section 7.4 quotes this submission's component scores when it explains that the gap between
    # the transformer's 0.9996 and the fused 0.957 is the ceiling rather than the style half
    # disagreeing. Record them here so that claim traces to a results file like every other number.
    worked = s[(s.id == WORKED_ID) & (s.label == 1)]
    worked_block = None
    if len(worked):
        w = worked.iloc[0]
        worked_block = {"id": WORKED_ID, "transformer": float(w.transformer),
                        "style": float(w["style"]), "fused": float(w.fused)}
        print(f"\nWorked submission {WORKED_ID} (Section 7.4): transformer "
              f"{w.transformer:.4f}, style {w['style']:.4f}, fused {w.fused:.4f}")
    else:
        print(f"\nWorked submission {WORKED_ID} is not in the test split; no entry written.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "worked_submission": worked_block,
        "fuser": {"w_transformer": w1, "w_style": w2, "intercept": b,
                  "ceiling": ceiling, "floor": floor,
                  "note": "Both inputs are probabilities in [0,1], so the logit is bounded by "
                          "[b, w1+w2+b] and the reported score can never leave [floor, ceiling]."},
        "n_test": int(len(te)),
        "ai": {"n": int(len(ai)),
               "transformer_min": float(ai.transformer.min()),
               "transformer_max": float(ai.transformer.max()),
               "style_min": float(ai_style.min()), "style_max": float(ai_style.max()),
               "fused_min": float(ai.fused.min()), "fused_max": float(ai.fused.max())},
        "human": {"n": int(len(hu)),
                  "fused_min": float(hu.fused.min()), "fused_max": float(hu.fused.max())},
        "range_used_pct": float((ai_span + hu_span) / span * 100),
        "rescued_by_fusion": [{"id": r.id, "transformer": float(r.transformer),
                               "style": float(r.style), "fused": float(r.fused)}
                              for r in rescued.itertuples()],
        "lost_by_fusion": [{"id": r.id, "transformer": float(r.transformer),
                            "style": float(r.style), "fused": float(r.fused)}
                           for r in missed.itertuples()],
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
