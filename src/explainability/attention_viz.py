"""Attention visualisation for the AI-text detector, with the same faithfulness test as IG.

Component 2 of the pipeline names three explanation methods: attention visualisation, Integrated
Gradients, and SHAP over the stylometric features. The other two are done; this closes the set.
The attention view is the classic one: how much the [CLS] position attends to each token in the
final layer, averaged over heads, which is often presented as "what the classifier looked at".

The point of this script is not to present that view uncritically. Attention weights are known to
be an unreliable guide to what drives a decision, so the same ERASER-style ablation used for
Integrated Gradients is applied here with identical settings (same test sample, same seed, same
k-sweep): remove the k most-attended tokens and measure the confidence drop against removing k
random tokens, and keep only the top-k and measure retained confidence. That makes the three
methods directly comparable on one yardstick, and the dissertation can rank them by measured
faithfulness rather than by how convincing their pictures look.

Runs on CPU deliberately (the GPU is often busy with training); DeBERTa-base forward passes are
cheap enough.

    python src/explainability/attention_viz.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
MODELS = REPO / "models" / "detector"
FIGS = REPO / "dissertation" / "figures"
OUT = REPO / "outputs" / "attention_viz.json"
MAXLEN = 256
N_FAITH = 50
KS = [1, 2, 3, 5, 8, 13, 21, 34]   # identical sweep to integrated_gradients.py


def latest_checkpoint() -> Path:
    cps = glob.glob(str(MODELS / "checkpoint-*"))
    if not cps:
        raise SystemExit(f"No checkpoint under {MODELS}. Train the detector first.")
    return Path(max(cps, key=lambda p: int(p.split("-")[-1])))


_PUNCT = {".": "“.” period", ",": "“,” comma", ";": "“;” semicolon", ":": "“:” colon",
          "-": "“-” dash", "!": "“!”", "?": "“?”", "(": "“(”", ")": "“)”",
          '"': "“ ” quote", "'": "“’” apostrophe", "’": "“’” apostrophe"}


def clean_tok(t: str) -> str:
    c = t.replace("▁", " ").replace("##", "").strip()
    if c in _PUNCT:
        return _PUNCT[c]
    return c or t


def main() -> int:
    device = torch.device("cpu")   # deliberate: leave the GPU to whatever is training
    ckpt = latest_checkpoint()
    print(f"Loading model from {ckpt} (CPU)", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()

    df = pd.read_parquet(CORPUS)
    test = df[df["split"] == "test"].reset_index(drop=True)

    def encode(text):
        enc = tok(text, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    @torch.no_grad()
    def cls_attention(ids, mask):
        """Final-layer attention from the [CLS] position to every token, averaged over heads."""
        out = model(input_ids=ids, attention_mask=mask, output_attentions=True)
        att = out.attentions[-1][0]          # heads x seq x seq
        return att.mean(dim=0)[0].cpu().numpy()   # CLS row -> per-token weight

    @torch.no_grad()
    def pred_and_prob(ids, mask):
        logits = model(input_ids=ids, attention_mask=mask).logits
        pred = int(logits.argmax(-1).item())
        return pred, torch.softmax(logits, -1)[0, pred].item()

    # ---- example figure: most-attended tokens for one AI and one human essay ----
    examples = {}
    for label, want in [("AI", 1), ("human", 0)]:
        sub = test[test["label"] == want]
        for _, row in sub.iterrows():
            ids, mask = encode(row["text"])
            pred, _ = pred_and_prob(ids, mask)
            if pred == want:
                w = cls_attention(ids, mask)
                toks = tok.convert_ids_to_tokens(ids[0].tolist())
                with torch.no_grad():
                    p_ai = torch.softmax(model(input_ids=ids, attention_mask=mask).logits, -1)[0, 1].item()
                examples[label] = {"toks": toks, "att": w, "id": row["id"], "p_ai": p_ai}
                break
    plot_examples(examples)

    # ---- faithfulness with the identical protocol to IG ----
    rng = np.random.default_rng(42)
    comp_top = {k: [] for k in KS}
    comp_rand = {k: [] for k in KS}
    suff_top = {k: [] for k in KS}
    specials = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}

    @torch.no_grad()
    def prob_of(ids, mask, cls):
        return torch.softmax(model(input_ids=ids, attention_mask=mask).logits, -1)[0, cls].item()

    sub = test.sample(n=min(N_FAITH, len(test)), random_state=42)
    for n_done, (_, row) in enumerate(sub.iterrows(), 1):
        ids, mask = encode(row["text"])
        pred, p0 = pred_and_prob(ids, mask)
        w = cls_attention(ids, mask)
        n = ids.shape[1]
        cand = [i for i in range(n) if int(ids[0, i]) not in specials]
        if len(cand) <= max(KS):
            continue
        order = sorted(cand, key=lambda i: w[i], reverse=True)
        for k in KS:
            top = order[:k]
            randk = list(rng.choice(cand, size=k, replace=False))
            for idxs, store in [(top, comp_top[k]), (randk, comp_rand[k])]:
                m = mask.clone()
                for i in idxs:
                    m[0, i] = 0
                store.append(p0 - prob_of(ids, m, pred))
            keep = set(top)
            m = mask.clone()
            for i in cand:
                if i not in keep:
                    m[0, i] = 0
            suff_top[k].append(prob_of(ids, m, pred))
        if n_done % 10 == 0:
            print(f"  faithfulness {n_done}/{len(sub)}", flush=True)

    ct = [round(float(np.mean(comp_top[k])), 4) for k in KS]
    cr = [round(float(np.mean(comp_rand[k])), 4) for k in KS]
    sf = [round(float(np.mean(suff_top[k])), 4) for k in KS]
    result = {
        "n_essays": len(comp_top[KS[0]]),
        "ks": KS,
        "comprehensiveness_top": ct,
        "comprehensiveness_random": cr,
        "sufficiency_top": sf,
        "summary": {"largest_k": KS[-1], "drop_gap_top_minus_random": round(ct[-1] - cr[-1], 4),
                    "comprehensiveness_ratio_top_over_random": round(ct[-1] / (cr[-1] + 1e-9), 2)},
        "protocol": "identical to integrated_gradients.py (same test sample, seed 42, same k-sweep), "
                    "so attention, IG and SHAP are comparable on one faithfulness yardstick",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_faithfulness(result)
    print("\n=== ATTENTION FAITHFULNESS (same protocol as IG) ===")
    print("k:", KS)
    print("drop, top-attended :", ct)
    print("drop, random       :", cr)
    print("sufficiency top-k  :", sf)
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def plot_examples(examples: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, label in zip(axes, ["AI", "human"]):
        if label not in examples:
            ax.axis("off")
            continue
        e = examples[label]
        best = {}
        for t, a in zip(e["toks"], e["att"]):
            if t in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                continue
            name = clean_tok(t)
            if name not in best or a > best[name]:
                best[name] = a
        pairs = sorted(best.items(), key=lambda x: x[1], reverse=True)[:12][::-1]
        ax.barh(range(len(pairs)), [p[1] for p in pairs], color="#52616B")
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([p[0] for p in pairs], fontsize=10)
        ax.set_title(f"{label} essay ({e['id']}), P(AI) = {e['p_ai']:.2f}",
                     fontsize=11, color="#222831")
        ax.set_xticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Final-layer [CLS] attention: the tokens the detector attends to most",
                 fontsize=12.5, fontweight="bold", color="#222831")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGS / "fig_attention_tokens.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_attention_tokens.png")


def plot_faithfulness(f: dict) -> None:
    # Overlay against the saved IG faithfulness so the comparison is one picture.
    ig = None
    ig_path = REPO / "outputs" / "explainability.json"
    if ig_path.exists():
        ig = json.loads(ig_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(f["ks"], f["comprehensiveness_top"], "-o", color="#52616B",
            label="remove most-attended tokens (attention)")
    if ig:
        ax.plot(ig["ks"], ig["comprehensiveness_top"], "-o", color="#d98e3b",
                label="remove top tokens (Integrated Gradients)")
    ax.plot(f["ks"], f["comprehensiveness_random"], "--o", color="#2b6777",
            label="remove random tokens (baseline)")
    ax.set_xlabel("number of tokens removed (k)")
    ax.set_ylabel("drop in detector confidence")
    ax.set_title("Attention vs Integrated Gradients on the same faithfulness yardstick",
                 fontsize=12.5, fontweight="bold", color="#222831")
    ax.legend(fontsize=10, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_attention_faithfulness.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_attention_faithfulness.png")


if __name__ == "__main__":
    raise SystemExit(main())
