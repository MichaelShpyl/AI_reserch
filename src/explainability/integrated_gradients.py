"""Explainability for the AI-text detector: Integrated Gradients token attributions
plus a faithfulness-by-ablation check.

This is the first piece of the explainability layer (CLAUDE.md component 2). It answers
"which words drove this decision?" in a way that can be put in front of a lecturer, and
then it tests whether that explanation is honest. The faithfulness test removes the tokens
the method says mattered most and measures how far the detector's confidence falls; if the
explanation is faithful, removing its top tokens should hurt confidence far more than
removing the same number of random tokens.

Method: Captum LayerIntegratedGradients on DeBERTa's word-embedding layer, attributing the
predicted class against an all-padding baseline. Runs on the fine-tuned cleaned-corpus
detector saved under models/detector.

    python src/explainability/integrated_gradients.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients

from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
MODELS = REPO / "models" / "detector"
FIGS = REPO / "dissertation" / "figures"
OUT = REPO / "outputs" / "explainability.json"
MAXLEN = 256          # shorter window keeps IG within 8 GB and is enough for the opening
N_STEPS = 32          # IG integration steps
N_FAITH = 50          # essays for the faithfulness test
KS = [1, 2, 3, 5, 8, 13, 21, 34]   # how many tokens to remove / keep in the sweep


def latest_checkpoint() -> Path:
    cps = glob.glob(str(MODELS / "checkpoint-*"))
    if not cps:
        raise SystemExit(f"No checkpoint under {MODELS}. Train the detector first.")
    return Path(max(cps, key=lambda p: int(p.split("-")[-1])))


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = latest_checkpoint()
    print(f"Loading model from {ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()

    df = pd.read_parquet(CORPUS)
    test = df[df["split"] == "test"].reset_index(drop=True)

    pad_id = tok.pad_token_id
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id
    emb_layer = model.get_input_embeddings()
    lig = LayerIntegratedGradients(
        lambda ids, mask: model(input_ids=ids, attention_mask=mask).logits, emb_layer)

    def encode(text):
        enc = tok(text, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    @torch.no_grad()
    def prob_ai(ids, mask):
        return torch.softmax(model(input_ids=ids, attention_mask=mask).logits, -1)[0, 1].item()

    def attributions(ids, mask, target):
        # Baseline: keep CLS/SEP, pad the rest, so attribution is "content vs nothing".
        base = ids.clone()
        base[(ids != cls_id) & (ids != sep_id)] = pad_id
        att = lig.attribute(inputs=ids, baselines=base, target=int(target),
                            additional_forward_args=(mask,), n_steps=N_STEPS,
                            internal_batch_size=8)
        att = att.sum(dim=-1).squeeze(0)            # sum over embedding dim -> per token
        att = att / (att.norm() + 1e-12)
        return att.detach().cpu().numpy()

    # ---- token-attribution figure for one human and one AI example ----
    examples = {}
    for label, want in [("AI", 1), ("human", 0)]:
        sub = test[test["label"] == want]
        for _, row in sub.iterrows():
            ids, mask = encode(row["text"])
            pred = int(model(input_ids=ids, attention_mask=mask).logits.argmax(-1).item())
            if pred == want:                         # use a correctly classified example
                # Always attribute to the AI class so the sign is consistent across panels:
                # positive (orange) pushes toward AI, negative (teal) pushes toward human.
                att = attributions(ids, mask, 1)
                toks = tok.convert_ids_to_tokens(ids[0].tolist())
                examples[label] = {"toks": toks, "att": att, "id": row["id"],
                                   "p_ai": prob_ai(ids, mask)}
                break
    plot_examples(examples)

    # ---- faithfulness by ablation (k-sweep: comprehensiveness and sufficiency) ----
    faith = faithfulness(test, encode, attributions, tok, model)
    plot_faithfulness(faith)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(faith, indent=2), encoding="utf-8")
    print("\n=== FAITHFULNESS (comprehensiveness: confidence drop when tokens removed) ===")
    print("k:", faith["ks"])
    print("drop, top tokens   :", faith["comprehensiveness_top"])
    print("drop, random tokens:", faith["comprehensiveness_random"])
    print("sufficiency (keep only top-k), mean retained P:", faith["sufficiency_top"])
    print("summary:", faith["summary"])
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


_PUNCT = {".": "“.” period", ",": "“,” comma", ";": "“;” semicolon", ":": "“:” colon",
          "-": "“-” dash", "!": "“!”", "?": "“?”", "(": "“(”", ")": "“)”",
          '"': "“ ” quote", "'": "“’” apostrophe", "’": "“’” apostrophe"}


def clean_tok(t: str) -> str:
    c = t.replace("▁", " ").replace("##", "").strip()
    if c in _PUNCT:
        return _PUNCT[c]
    return c or t


def plot_examples(examples: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2))
    for ax, label in zip(axes, ["AI", "human"]):
        if label not in examples:
            ax.axis("off"); continue
        e = examples[label]
        # top tokens by absolute attribution (skip special tokens); collapse repeated
        # labels (e.g. several commas) to the strongest instance so the chart is legible.
        best = {}
        for t, a in zip(e["toks"], e["att"]):
            if t in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                continue
            name = clean_tok(t)
            if name not in best or abs(a) > abs(best[name]):
                best[name] = a
        pairs = sorted(best.items(), key=lambda x: abs(x[1]), reverse=True)[:14][::-1]
        names = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        colors = ["#d98e3b" if v > 0 else "#2b6777" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=10)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_title(f"{label} essay ({e['id']}), detector P(AI) = {e['p_ai']:.2f}: "
                     f"tokens pushing toward AI (orange) / human (teal)",
                     fontsize=11, color="#222831")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xticks([])
    fig.suptitle("Integrated Gradients: which tokens drove each detection decision",
                 fontsize=13, fontweight="bold", color="#222831")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGS / "fig_explain_ig_tokens.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_explain_ig_tokens.png")


def faithfulness(test, encode, attributions, tok, model) -> dict:
    """Comprehensiveness (remove top-k, confidence should fall) and sufficiency (keep only
    top-k, confidence should stay high), swept over k, against a random-token baseline.
    Ablation is done by zeroing the attention mask so removed tokens are truly ignored."""
    rng = np.random.default_rng(42)
    comp_top = {k: [] for k in KS}
    comp_rand = {k: [] for k in KS}
    suff_top = {k: [] for k in KS}
    specials = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}

    def pred_prob(ids, mask, cls):
        with torch.no_grad():
            return torch.softmax(model(input_ids=ids, attention_mask=mask).logits, -1)[0, cls].item()

    sub = test.sample(n=min(N_FAITH, len(test)), random_state=42)
    for _, row in sub.iterrows():
        ids, mask = encode(row["text"])
        pred = int(model(input_ids=ids, attention_mask=mask).logits.argmax(-1).item())
        p0 = pred_prob(ids, mask, pred)
        att = np.abs(attributions(ids, mask, pred))
        n = ids.shape[1]
        cand = [i for i in range(n) if int(ids[0, i]) not in specials]
        if len(cand) <= max(KS):
            continue
        order = sorted(cand, key=lambda i: att[i], reverse=True)
        for k in KS:
            top = order[:k]
            randk = list(rng.choice(cand, size=k, replace=False))
            # comprehensiveness: remove the k tokens (mask them out)
            for idxs, store in [(top, comp_top[k]), (randk, comp_rand[k])]:
                m = mask.clone()
                for i in idxs:
                    m[0, i] = 0
                store.append(p0 - pred_prob(ids, m, pred))
            # sufficiency: keep ONLY the top-k content tokens (plus specials)
            keep = set(top)
            m = mask.clone()
            for i in cand:
                if i not in keep:
                    m[0, i] = 0
            suff_top[k].append(pred_prob(ids, m, pred))

    n_ess = len(comp_top[KS[0]])
    ct = [round(float(np.mean(comp_top[k])), 4) for k in KS]
    cr = [round(float(np.mean(comp_rand[k])), 4) for k in KS]
    sf = [round(float(np.mean(suff_top[k])), 4) for k in KS]
    # Summary: gap between top and random comprehensiveness at the largest k.
    gap = round(ct[-1] - cr[-1], 4)
    ratio = round(ct[-1] / (cr[-1] + 1e-9), 2)
    return {
        "n_essays": int(n_ess),
        "ks": KS,
        "comprehensiveness_top": ct,
        "comprehensiveness_random": cr,
        "sufficiency_top": sf,
        "summary": {"largest_k": KS[-1], "drop_gap_top_minus_random": gap,
                    "comprehensiveness_ratio_top_over_random": ratio},
        "reading": "Comprehensiveness: removing the top attributed tokens should drop the "
                   "detector's confidence more than removing random tokens. Sufficiency: keeping "
                   "only the top tokens should retain confidence. A small gap means the signal is "
                   "diffuse (spread across many ordinary words), so no few tokens are decisive.",
    }


def plot_faithfulness(f: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(f["ks"], f["comprehensiveness_top"], "-o", color="#d98e3b",
            label="remove top tokens (by Integrated Gradients)")
    ax.plot(f["ks"], f["comprehensiveness_random"], "-o", color="#2b6777",
            label="remove random tokens (baseline)")
    ax.set_xlabel("number of tokens removed (k)")
    ax.set_ylabel("drop in detector confidence")
    ax.set_title("Faithfulness check: do the highlighted tokens actually drive the decision?",
                 fontsize=12.5, fontweight="bold", color="#222831")
    ax.legend(fontsize=10, loc="best")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_explain_faithfulness.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_explain_faithfulness.png")


if __name__ == "__main__":
    raise SystemExit(main())
