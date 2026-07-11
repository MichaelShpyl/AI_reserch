"""Pairwise relation classification on Persuasive Essays 2.0 (the second half of component 3).

The scope defines argument mining as span tagging plus pairwise relation classification. The span
tagger exists (train_claim_extractor.py); this trains the relation side: given an ordered pair of
argument components from the same paragraph, does the first support the second, attack it, or is
there no direct link? Following Stab and Gurevych (2017), candidate pairs are ordered pairs of gold
components within one paragraph (their argument trees never cross paragraphs), the official 322/80
essay split is used, and the honest baseline is the majority class, since most candidate pairs are
unlinked. DeBERTa-v3-base reads "source [SEP] target" and classifies supports / attacks / none.

    python src/argument_mining/train_relation_classifier.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BRAT = REPO / "data" / "raw" / "persuasive_essays" / "ArgumentAnnotatedEssays-2.0" / "brat-project-final"
SPLIT = REPO / "data" / "raw" / "persuasive_essays" / "ArgumentAnnotatedEssays-2.0" / "train-test-split.csv"
MODELDIR = REPO / "models" / "relation_classifier"
OUT = REPO / "outputs" / "relation_classifier.json"
SEED = 42
MODEL_NAME = "microsoft/deberta-v3-base"
LABELS = ["none", "supports", "attacks"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}


def read_split() -> dict[str, str]:
    raw = SPLIT.read_text(encoding="utf-8", errors="ignore")
    return {m.group(1): m.group(2) for m in re.finditer(r'"(essay\d+)";"(TRAIN|TEST)"', raw)}


def parse_essay(stem: str):
    text = (BRAT / f"{stem}.txt").read_text(encoding="utf-8")
    comps, rels = {}, []
    for line in (BRAT / f"{stem}.ann").read_text(encoding="utf-8").splitlines():
        if line.startswith("T"):
            tid, meta, ctext = line.split("\t", 2)
            typ, start, end = meta.split()[:3]
            if typ in ("MajorClaim", "Claim", "Premise"):
                comps[tid] = {"type": typ, "start": int(start), "end": int(end),
                              "text": ctext.strip()}
        elif line.startswith("R"):
            m = re.match(r"R\d+\t(supports|attacks) Arg1:(T\d+) Arg2:(T\d+)", line)
            if m:
                rels.append((m.group(2), m.group(3), m.group(1)))
    return text, comps, rels


def paragraph_index(text: str, pos: int) -> int:
    """Which paragraph (non-empty line) a character offset falls in."""
    idx, cursor = 0, 0
    for line in text.split("\n"):
        if cursor <= pos < cursor + len(line):
            return idx
        if line.strip():
            idx += 1
        cursor += len(line) + 1
    return idx


def build_pairs(stems: list[str]) -> list[dict]:
    pairs = []
    for stem in stems:
        text, comps, rels = parse_essay(stem)
        linked = {(a, b): t for a, b, t in rels}
        paras = {tid: paragraph_index(text, c["start"]) for tid, c in comps.items()}
        tids = sorted(comps)
        for a in tids:
            for b in tids:
                if a == b or paras[a] != paras[b]:
                    continue
                pairs.append({"essay": stem,
                              "src": f"{comps[a]['type']}: {comps[a]['text']}",
                              "tgt": f"{comps[b]['type']}: {comps[b]['text']}",
                              "label": LAB2ID[linked.get((a, b), "none")]})
    return pairs


def main() -> int:
    import torch
    from datasets import Dataset
    from sklearn.metrics import classification_report, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    set_seed(SEED)

    split = read_split()
    train_stems = sorted(s for s, v in split.items() if v == "TRAIN")
    test_stems = sorted(s for s, v in split.items() if v == "TEST")
    train_pairs = build_pairs(train_stems)
    test_pairs = build_pairs(test_stems)
    dist = Counter(p["label"] for p in train_pairs)
    print(f"pairs: train {len(train_pairs)}, test {len(test_pairs)}; "
          f"train dist {[f'{LABELS[k]}:{v}' for k, v in sorted(dist.items())]}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def make(pairs):
        ds = Dataset.from_list(pairs)
        return ds.map(lambda b: tok(b["src"], b["tgt"], truncation=True, max_length=256),
                      batched=True)

    train_ds, test_ds = make(train_pairs), make(test_pairs)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3,
        id2label={i: l for l, i in LAB2ID.items()}, label2id=LAB2ID)

    # class weights against the heavy none-majority
    counts = np.array([dist.get(i, 1) for i in range(3)], dtype=float)
    weights = torch.tensor((counts.sum() / (3 * counts)), dtype=torch.float)
    print(f"class weights: {weights.tolist()}", flush=True)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(MODELDIR / "_runs"), overwrite_output_dir=True,
        num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=8, per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        eval_strategy="no", save_strategy="no",
        fp16=torch.cuda.is_available(), report_to="none", logging_steps=50, seed=SEED,
    )
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=train_ds,
                              processing_class=tok, data_collator=DataCollatorWithPadding(tok))
    trainer.train()

    pred = trainer.predict(test_ds)
    preds = pred.predictions.argmax(-1)
    y = pred.label_ids
    rep = classification_report(y, preds, target_names=LABELS, output_dict=True, zero_division=0)
    majority = int(np.bincount(y).argmax())
    base = f1_score(y, np.full_like(y, majority), average="macro", zero_division=0)

    MODELDIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODELDIR))
    tok.save_pretrained(str(MODELDIR))
    result = {
        "task": "ordered within-paragraph component pairs -> supports / attacks / none "
                "(gold components, official 322/80 split, class-weighted loss)",
        "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
        "test_macro_f1": round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
        "majority_baseline_macro_f1": round(float(base), 4),
        "per_class": {k: {"precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                          "f1": round(v["f1-score"], 3), "n": int(v["support"])}
                      for k, v in rep.items() if k in LABELS},
        "note": "basic pairwise model completing scope component 3; gold component boundaries are "
                "given, so this measures relation classification alone, not the full pipeline",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== RELATION CLASSIFIER ===")
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    print(json.dumps(result["per_class"], indent=2))
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
