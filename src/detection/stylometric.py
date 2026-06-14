"""Stylometric features for the hybrid detector.

These are the non-transformer signals the detector combines with DeBERTa. They are
cheap and run on CPU (spaCy), so they do not need the GPU. The intuition is that human
academic writing varies more (sentence length, vocabulary) than AI text, which tends to
be smoother and more uniform.

Features per text:
  - sentence-length statistics and burstiness (variation in sentence length)
  - type-token ratio and a length-robust variant (root TTR), hapax ratio
  - mean word length, punctuation ratio
  - part-of-speech distribution (proportions of the main POS tags)
  - perplexity via GPT-2 (implemented here, but run it when the GPU is free; it is the
    one feature that wants a model on the GPU)

Demo: compute the features for the AI essays generated so far and their matched human
essays, and print the human-vs-AI means as an early signal.

    python src/detection/stylometric.py            # demo on current AI/human pairs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"

POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "PROPN", "ADP", "DET",
            "AUX", "CCONJ", "SCONJ", "NUM", "PART", "PUNCT"]


def load_nlp():
    import spacy
    # Keep the parser for sentence boundaries; drop NER for speed.
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def stylometric_features(text: str, nlp) -> dict:
    doc = nlp(text)
    toks = [t for t in doc if not t.is_space]
    words = [t for t in toks if t.is_alpha]
    lower = [t.text.lower() for t in words]
    sents = [s for s in doc.sents if len(s.text.strip()) > 0]
    sent_lens = np.array([sum(1 for t in s if t.is_alpha) for s in sents], dtype=float)
    sent_lens = sent_lens[sent_lens > 0]

    n_words = len(words)
    feats: dict[str, float] = {}
    feats["n_words"] = float(n_words)
    feats["n_sents"] = float(len(sent_lens))

    if len(sent_lens) > 0:
        mu, sigma = sent_lens.mean(), sent_lens.std()
        feats["mean_sent_len"] = float(mu)
        feats["std_sent_len"] = float(sigma)
        feats["sent_len_cv"] = float(sigma / mu) if mu else 0.0
        feats["burstiness"] = float((sigma - mu) / (sigma + mu)) if (sigma + mu) else 0.0
    else:
        feats.update(mean_sent_len=0.0, std_sent_len=0.0, sent_len_cv=0.0, burstiness=0.0)

    if n_words > 0:
        n_types = len(set(lower))
        counts = pd.Series(lower).value_counts()
        feats["ttr"] = n_types / n_words
        feats["root_ttr"] = n_types / np.sqrt(n_words)
        feats["hapax_ratio"] = float((counts == 1).sum()) / n_words
        feats["mean_word_len"] = float(np.mean([len(w.text) for w in words]))
    else:
        feats.update(ttr=0.0, root_ttr=0.0, hapax_ratio=0.0, mean_word_len=0.0)

    n_toks = max(len(toks), 1)
    feats["punct_ratio"] = sum(1 for t in toks if t.is_punct) / n_toks
    pos_counts = pd.Series([t.pos_ for t in toks]).value_counts()
    for tag in POS_TAGS:
        feats[f"pos_{tag}"] = float(pos_counts.get(tag, 0)) / n_toks
    return feats


def features_for_texts(texts, nlp) -> pd.DataFrame:
    return pd.DataFrame([stylometric_features(t, nlp) for t in texts])


class GPT2Perplexity:
    """Per-text perplexity from GPT-2. Run on GPU when it is free.

    Not used in the demo to avoid competing with the local generation. On first use it
    downloads gpt2 (~500 MB). For long texts it scores the first `max_tokens` tokens.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu", max_tokens: int = 1024):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.torch = torch
        self.device = device
        self.max_tokens = max_tokens
        self.tok = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    def score(self, text: str) -> float:
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_tokens).input_ids.to(self.device)
        if ids.size(1) < 2:
            return float("nan")
        with self.torch.no_grad():
            loss = self.model(ids, labels=ids).loss
        return float(self.torch.exp(loss).item())


def demo(limit: int) -> int:
    if not AI_DIR.exists() or not any(AI_DIR.glob("*.txt")):
        print("No AI essays yet; run the generator first.")
        return 1
    nlp = load_nlp()
    ids = sorted(p.stem for p in AI_DIR.glob("*.txt"))[:limit]
    rows = []
    for rid in ids:
        human_path = CORPUS_TXT / f"{rid}.txt"
        if not human_path.exists():
            continue
        ai_text = (AI_DIR / f"{rid}.txt").read_text(encoding="utf-8", errors="ignore")
        hu_text = human_path.read_text(encoding="utf-8", errors="ignore")
        h = stylometric_features(hu_text, nlp); h["label"] = "human"; h["id"] = rid
        a = stylometric_features(ai_text, nlp); a["label"] = "ai"; a["id"] = rid
        rows.extend([h, a])
    df = pd.DataFrame(rows)
    print(f"Computed stylometric features for {len(ids)} human/AI pairs.\n")
    show = ["burstiness", "sent_len_cv", "std_sent_len", "ttr", "root_ttr",
            "hapax_ratio", "mean_word_len", "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_PUNCT"]
    means = df.groupby("label")[show].mean()
    out = means.T
    out["diff(human-ai)"] = out["human"] - out["ai"]
    with pd.option_context("display.float_format", lambda v: f"{v:7.3f}"):
        print(out.to_string())
    print("\n(Early signal only, on the essays generated so far. Larger burstiness and "
          "vocabulary variation in human text would be the expected direction.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Stylometric features for the detector.")
    ap.add_argument("--limit", type=int, default=40, help="Max human/AI pairs in the demo.")
    args = ap.parse_args()
    return demo(args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
