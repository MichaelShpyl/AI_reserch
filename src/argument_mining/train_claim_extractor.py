"""Claim extractor (pipeline component 3): BIO sequence labelling on Persuasive Essays 2.0.

Trains a DeBERTa-v3-base token classifier to mark argument components (MajorClaim, Claim, Premise)
in essay text, following Stab and Gurevych (2017): the corpus's official 322/80 train/test split,
paragraph-level sequences (annotations never cross paragraphs; the longest paragraph fits in 256
subwords, so nothing is truncated), and strict span-level evaluation with seqeval.

Outputs: models/claim_extractor/ (weights), outputs/claim_extractor.json (metrics),
dissertation/figures/fig_claim_extractor.png (per-class span F1).

    python src/argument_mining/train_claim_extractor.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BRAT = REPO / "data" / "raw" / "persuasive_essays" / "ArgumentAnnotatedEssays-2.0" / "brat-project-final"
SPLIT = REPO / "data" / "raw" / "persuasive_essays" / "ArgumentAnnotatedEssays-2.0" / "train-test-split.csv"
OUTJ = REPO / "outputs" / "claim_extractor.json"
FIGS = REPO / "dissertation" / "figures"
MODELDIR = REPO / "models" / "claim_extractor"
SEED = 42
MODEL_NAME = "microsoft/deberta-v3-base"

TYPES = ["MajorClaim", "Claim", "Premise"]
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
LAB2ID = {l: i for i, l in enumerate(LABELS)}


def read_split() -> dict[str, str]:
    raw = SPLIT.read_text(encoding="utf-8-sig")
    return {m.group(1): m.group(2) for m in re.finditer(r'"(essay\d+)";"(TRAIN|TEST)"', raw)}


def parse_essay(stem: str) -> tuple[str, list[tuple[int, int, str]]]:
    text = (BRAT / f"{stem}.txt").read_text(encoding="utf-8")
    spans = []
    for line in (BRAT / f"{stem}.ann").read_text(encoding="utf-8").splitlines():
        if line.startswith("T"):
            _, meta, _ = line.split("\t", 2)
            typ, start, end = meta.split()[:3]
            if typ in TYPES:
                spans.append((int(start), int(end), typ))
    return text, spans


def paragraphs_with_offsets(text: str):
    """Yield (paragraph_text, char_start) for each non-empty line."""
    pos = 0
    for line in text.split("\n"):
        if line.strip():
            yield line, pos
        pos += len(line) + 1


def build_examples(stems: list[str]) -> list[dict]:
    out = []
    for stem in stems:
        text, spans = parse_essay(stem)
        for para, p0 in paragraphs_with_offsets(text):
            p1 = p0 + len(para)
            local = [(max(s, p0) - p0, min(e, p1) - p0, t) for s, e, t in spans
                     if s < p1 and e > p0]
            out.append({"essay": stem, "text": para, "spans": local})
    return out


def encode(examples: list[dict], tok):
    encodings = []
    for ex in examples:
        enc = tok(ex["text"], truncation=True, max_length=256, return_offsets_mapping=True)
        labels = []
        for (a, b) in enc["offset_mapping"]:
            if a == b:                       # special tokens
                labels.append(-100)
                continue
            lab = "O"
            for s, e, t in ex["spans"]:
                if a >= s and b <= e:
                    lab = ("B-" if a == s or (a <= s < b) else "I-") + t
                    break
                if a < s < b:                # token straddles the span start
                    lab = "B-" + t
                    break
            labels.append(LAB2ID[lab])
        enc.pop("offset_mapping")
        enc["labels"] = labels
        encodings.append(enc)
    return encodings


def main() -> int:
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer, TrainingArguments,
                              set_seed)
    from seqeval.metrics import classification_report, f1_score

    set_seed(SEED)
    split = read_split()
    stems = sorted(split)
    train_stems = [s for s in stems if split[s] == "TRAIN"]
    test_stems = [s for s in stems if split[s] == "TEST"]
    rng = np.random.default_rng(SEED)
    rng.shuffle(train_stems)
    n_val = max(1, len(train_stems) // 10)
    val_stems, train_stems = train_stems[:n_val], train_stems[n_val:]
    print(f"essays: train {len(train_stems)}, val {len(val_stems)}, test {len(test_stems)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = {}
    for name, st in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        exs = build_examples(st)
        ds[name] = Dataset.from_list(encode(exs, tok))
        print(f"{name}: {len(exs)} paragraphs")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        id2label={i: l for l, i in LAB2ID.items()}, label2id=LAB2ID)

    def metrics(p):
        preds = p.predictions.argmax(-1)
        true, hyp = [], []
        for pr, lb in zip(preds, p.label_ids):
            t_row, h_row = [], []
            for pi, li in zip(pr, lb):
                if li == -100:
                    continue
                t_row.append(LABELS[li]); h_row.append(LABELS[pi])
            true.append(t_row); hyp.append(h_row)
        return {"span_f1": f1_score(true, hyp)}

    args = TrainingArguments(
        output_dir=str(MODELDIR / "_runs"), seed=SEED,
        num_train_epochs=5, per_device_train_batch_size=8, per_device_eval_batch_size=32,
        learning_rate=2e-5, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="span_f1",
        logging_steps=50, report_to=[], fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=DataCollatorForTokenClassification(tok),
                      compute_metrics=metrics)
    trainer.train()

    # Held-out test, strict span-level.
    pred = trainer.predict(ds["test"])
    preds = pred.predictions.argmax(-1)
    true, hyp = [], []
    for pr, lb in zip(preds, pred.label_ids):
        t_row, h_row = [], []
        for pi, li in zip(pr, lb):
            if li == -100:
                continue
            t_row.append(LABELS[li]); h_row.append(LABELS[pi])
        true.append(t_row); hyp.append(h_row)
    rep = classification_report(true, hyp, output_dict=True, zero_division=0)
    result = {
        "model": MODEL_NAME,
        "data": "Persuasive Essays 2.0 (Stab and Gurevych 2017), official 322/80 split, "
                "paragraph-level sequences",
        "n_test_essays": len(test_stems),
        "span_f1_micro": round(f1_score(true, hyp), 4),
        "per_class": {k: {"precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                          "f1": round(v["f1-score"], 3), "support": int(v["support"])}
                      for k, v in rep.items() if k in TYPES},
        "reading": "Strict span-level scores (exact boundary and type match, seqeval). "
                   "Reference points from the literature: Stab and Gurevych's ILP parser and "
                   "later BiLSTM/transformer taggers report component F1 in the 0.80s.",
    }
    OUTJ.write_text(json.dumps(result, indent=2), encoding="utf-8")
    MODELDIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODELDIR))
    tok.save_pretrained(str(MODELDIR))
    make_figure(result)
    print(json.dumps(result, indent=2))
    print(f"Saved {OUTJ.relative_to(REPO)} and model to {MODELDIR.relative_to(REPO)}")
    return 0


def make_figure(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = result["per_class"]
    names = list(per)
    f1s = [per[n]["f1"] for n in names]
    sup = [per[n]["support"] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(names, f1s, width=0.55, color=["#a63d2e", "#2b6777", "#d98e3b"])
    for b, f, s in zip(bars, f1s, sup):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.02, f"{f:.2f}\n(n={s})",
                ha="center", fontsize=10, color="#222831")
    ax.axhline(result["span_f1_micro"], color="#52616B", ls="--", lw=1.2,
               label=f"micro span-F1 {result['span_f1_micro']:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("strict span-level F1")
    ax.set_title("Argument-component extraction on the official Persuasive Essays test set",
                 fontsize=12, fontweight="bold", color="#222831")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_claim_extractor.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_claim_extractor.png")


if __name__ == "__main__":
    raise SystemExit(main())
