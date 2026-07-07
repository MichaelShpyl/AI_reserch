"""The hybrid detector: transformer fused with stylometric features (pipeline component 1, complete).

The scope defines the detector as a fine-tuned transformer combined with stylometric features
including GPT-2 perplexity. Both halves exist and are near the ceiling in-domain (DeBERTa 0.990,
style features 0.985), so gluing them together for the home corpus is necessary but not very
informative. The informative question is out of domain: Chapter 6 showed the transformer falls to
F1 0.79 on unseen domains and falsely flags most human arXiv abstracts, so this script also asks
whether the style half or the fusion softens that failure.

What it does:
  1. Adds the deferred GPT-2 perplexity feature to the stylometric set (mean token NLL over the
     first 512 tokens, exponentiated).
  2. Reports four arms on the home test split: transformer alone, style features alone, style
     features plus perplexity, and the fused hybrid (logistic regression over the two model
     probabilities, fitted on the validation split so the test stays untouched).
  3. Applies all arms zero-shot to the same cross-domain M4 sample as eval_m4_transfer.py
     (identical sampling, seed and normalisation, so numbers are comparable with Chapter 6).

    python src/detection/hybrid_fusion.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[2]
CLEAN = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
FEATS = REPO / "data" / "processed" / "stylometric_features.parquet"
M4 = REPO / "data" / "raw" / "m4"
MODELS = REPO / "models" / "detector"
OUT = REPO / "outputs" / "hybrid_fusion.json"
FIGS = REPO / "dissertation" / "figures"
PER_DOMAIN = 300
MAXLEN = 512
BATCH = 16
SEED = 42

import sys
sys.path.insert(0, str(REPO / "src" / "detection"))
from text_normalize import normalize_text  # noqa: E402
from stylometric import load_nlp, stylometric_features  # noqa: E402

DROP = ["n_words", "n_sents", "id", "label", "split", "native"]


def latest_checkpoint() -> Path:
    cps = glob.glob(str(MODELS / "checkpoint-*"))
    if not cps:
        raise SystemExit(f"No checkpoint under {MODELS}.")
    return Path(max(cps, key=lambda p: int(p.split("-")[-1])))


def gpt2_perplexity(texts, device) -> np.ndarray:
    """Mean-NLL perplexity of the first MAXLEN GPT-2 tokens of each text."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    lm = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    out = []
    with torch.no_grad():
        for i, t in enumerate(texts):
            ids = tok(t, truncation=True, max_length=MAXLEN, return_tensors="pt")["input_ids"].to(device)
            if ids.shape[1] < 2:
                out.append(float("nan"))
                continue
            loss = lm(ids, labels=ids).loss
            out.append(float(torch.exp(loss).item()))
            if (i + 1) % 200 == 0:
                print(f"    perplexity {i + 1}/{len(texts)}", flush=True)
    del lm
    torch.cuda.empty_cache()
    return np.array(out)


def deberta_probs(texts, device) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ckpt = latest_checkpoint()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to(device)
            probs.extend(torch.softmax(model(**enc).logits, -1)[:, 1].tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def metrics(y, yp) -> dict:
    return {"accuracy": round(accuracy_score(y, yp), 4),
            "precision": round(precision_score(y, yp, zero_division=0), 4),
            "recall": round(recall_score(y, yp, zero_division=0), 4),
            "f1": round(f1_score(y, yp, zero_division=0), 4)}


def style_frame(texts, nlp) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(texts):
        rows.append(stylometric_features(t, nlp))
        if (i + 1) % 300 == 0:
            print(f"    style features {i + 1}/{len(texts)}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = pd.read_parquet(CLEAN)
    feats = pd.read_parquet(FEATS)
    # ids are shared between a human essay and its AI twin, so the text join must use (id, label).
    df = feats.merge(corpus[["id", "label", "text"]], on=["id", "label"], validate="one_to_one")
    assert len(df) == len(feats), "feature/text join lost or duplicated rows"

    # ---- 1. perplexity feature for the home corpus ----
    print("== GPT-2 perplexity, home corpus ==", flush=True)
    df["gpt2_ppl"] = gpt2_perplexity(df["text"].tolist(), device)

    feat_cols = [c for c in feats.columns if c not in DROP]
    tr = df[df["split"] == "train"]; va = df[df["split"] == "val"]; te = df[df["split"] == "test"]

    # ---- 2. the four arms on the home test split ----
    print("== DeBERTa probabilities, home corpus ==", flush=True)
    p_deb = {s: deberta_probs(d["text"].tolist(), device) for s, d in
             [("val", va), ("test", te)]}

    def fit_gbm(cols):
        m = GradientBoostingClassifier(random_state=SEED)
        m.fit(tr[cols], tr["label"])
        return m

    gbm_style = fit_gbm(feat_cols)
    gbm_ppl = fit_gbm(feat_cols + ["gpt2_ppl"])

    p_style_val = gbm_ppl.predict_proba(va[feat_cols + ["gpt2_ppl"]])[:, 1]
    p_style_te = gbm_ppl.predict_proba(te[feat_cols + ["gpt2_ppl"]])[:, 1]
    fuser = LogisticRegression()
    fuser.fit(np.column_stack([p_deb["val"], p_style_val]), va["label"])
    p_hy_te = fuser.predict_proba(np.column_stack([p_deb["test"], p_style_te]))[:, 1]

    yte = te["label"].values
    arms = {
        "transformer": metrics(yte, (p_deb["test"] >= 0.5).astype(int)),
        "style": metrics(yte, gbm_style.predict(te[feat_cols])),
        "style_plus_perplexity": metrics(yte, gbm_ppl.predict(te[feat_cols + ["gpt2_ppl"]])),
        "hybrid": metrics(yte, (p_hy_te >= 0.5).astype(int)),
    }
    # where does perplexity rank among the features?
    import shap
    sv = shap.TreeExplainer(gbm_ppl).shap_values(te[feat_cols + ["gpt2_ppl"]])
    mean_abs = np.abs(sv).mean(axis=0)
    rank = int(np.argsort(mean_abs)[::-1].tolist().index(len(feat_cols))) + 1
    arms["perplexity_shap_rank"] = rank

    # ---- 3. zero-shot cross-domain, identical sample to eval_m4_transfer ----
    print("== cross-domain arm (M4, same sample as Chapter 6) ==", flush=True)
    train_m4 = pd.read_parquet(M4 / "train.parquet")
    domains = sorted(train_m4["source"].dropna().unique())
    parts = []
    for d in domains:
        sd = train_m4[train_m4["source"] == d]
        for lab in (0, 1):
            s = sd[sd["label"] == lab]
            if len(s):
                parts.append(s.sample(n=min(PER_DOMAIN, len(s)), random_state=SEED))
    B = pd.concat(parts).reset_index(drop=True)
    B_norm = [normalize_text(t) for t in B["text"].tolist()]
    print(f"  {len(B)} texts across {len(domains)} domains", flush=True)

    p_deb_B = deberta_probs(B_norm, device)
    print("  style features ...", flush=True)
    nlp = load_nlp()
    fB = style_frame(B_norm, nlp)
    print("  perplexity ...", flush=True)
    fB["gpt2_ppl"] = gpt2_perplexity(B_norm, device)
    p_style_B = gbm_ppl.predict_proba(fB[feat_cols + ["gpt2_ppl"]])[:, 1]
    p_hy_B = fuser.predict_proba(np.column_stack([p_deb_B, p_style_B]))[:, 1]

    yB = B["label"].values
    cross = {"n": int(len(B)), "overall": {
        "transformer": metrics(yB, (p_deb_B >= 0.5).astype(int)),
        "style_plus_perplexity": metrics(yB, (p_style_B >= 0.5).astype(int)),
        "hybrid": metrics(yB, (p_hy_B >= 0.5).astype(int))}, "per_domain": {}}
    for d in domains:
        m = (B["source"] == d).values
        hu = m & (yB == 0)
        cross["per_domain"][d] = {
            "human_FPR_transformer": round(float((p_deb_B[hu] >= 0.5).mean()), 4) if hu.sum() else None,
            "human_FPR_style": round(float((p_style_B[hu] >= 0.5).mean()), 4) if hu.sum() else None,
            "human_FPR_hybrid": round(float((p_hy_B[hu] >= 0.5).mean()), 4) if hu.sum() else None,
            "acc_transformer": round(float(((p_deb_B[m] >= 0.5).astype(int) == yB[m]).mean()), 4),
            "acc_hybrid": round(float(((p_hy_B[m] >= 0.5).astype(int) == yB[m]).mean()), 4),
        }

    result = {"home_test": arms, "cross_domain": cross,
              "fusion": "logistic regression over [P_transformer, P_style+ppl], fitted on the "
                        "validation split; cross-domain applied zero-shot",
              "note": "cross-domain sample identical to eval_m4_transfer.py (source, per-domain "
                      "300 per class, seed 42, same normalisation)"}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(result)
    print("\n=== HYBRID FUSION ===")
    for k, v in arms.items():
        print(f"home {k}: {v}")
    print("cross-domain overall:", json.dumps(cross["overall"], indent=1))
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    # Left: home-test F1 for the four arms.
    arms = ["transformer", "style", "style_plus_perplexity", "hybrid"]
    names = ["transformer", "style", "style\n+ perplexity", "hybrid"]
    f1s = [result["home_test"][a]["f1"] for a in arms]
    ax1.bar(names, f1s, width=0.6, color=["#52616B", "#9bb7bd", "#7fa8b0", "#2b6777"])
    for i, v in enumerate(f1s):
        ax1.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9.5, color="#222831")
    ax1.set_ylim(0.9, 1.01)
    ax1.set_ylabel("test F1 (home corpus)")
    ax1.set_title("In-domain: all arms at the ceiling", fontsize=11, fontweight="bold", color="#222831")
    ax1.spines[["top", "right"]].set_visible(False)
    # Right: cross-domain human FPR per domain, transformer vs hybrid.
    pd_ = result["cross_domain"]["per_domain"]
    doms = sorted(pd_)
    x = np.arange(len(doms))
    ax2.bar(x - 0.22, [pd_[d]["human_FPR_transformer"] for d in doms], 0.44,
            label="transformer", color="#a63d2e")
    ax2.bar(x + 0.22, [pd_[d]["human_FPR_hybrid"] for d in doms], 0.44,
            label="hybrid", color="#2b6777")
    ax2.set_xticks(x); ax2.set_xticklabels(doms, fontsize=9)
    ax2.set_ylabel("human false-positive rate")
    ax2.set_title("Out of domain: does the fusion soften the failure?",
                  fontsize=11, fontweight="bold", color="#222831")
    ax2.legend(fontsize=9, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The hybrid detector (component 1 complete): transformer + style + GPT-2 perplexity",
                 fontsize=12.5, fontweight="bold", color="#222831")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_hybrid_fusion.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_hybrid_fusion.png")


if __name__ == "__main__":
    raise SystemExit(main())
