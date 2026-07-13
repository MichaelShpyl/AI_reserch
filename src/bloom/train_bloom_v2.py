"""Bloom classifier v2: gold EduQG training data plus validated silver labels.

The v1 classifier's higher-order classes are starved (19 "analyse" questions in the whole gold
set), so this variant adds silver labels produced by an LLM annotator that first had to prove
itself against the gold labels (src/bloom/llm_annotate.py; the few-shot variant is the one that
earns trust or not). Silver rows join the TRAIN split only. The validation and test splits stay
byte-identical to v1 (same seed, same stratified split), so the v1-versus-v2 comparison on
outputs/bloom_classifier.json versus outputs/bloom_classifier_v2.json is apples to apples.

    python src/bloom/train_bloom_v2.py --silver-key "v3pool|commercial:anthropic:claude-opus-4-8|fs"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "bloom"))
from train_bloom_classifier import LEVELS, SEED, load_rows  # noqa: E402

ANNOT = REPO / "outputs" / "bloom_llm_annotation.json"
OUTJ = REPO / "outputs" / "bloom_classifier_v2.json"
MODELDIR = REPO / "models" / "bloom_classifier_v2"
NAME_TO_INT = {v: k for k, v in LEVELS.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-key", required=True,
                    help="key in bloom_llm_annotation.json holding the silver labels")
    ap.add_argument("--cap", type=int, default=400,
                    help="max silver rows per class (seeded sample), so silver cannot swamp gold")
    args = ap.parse_args()

    from datasets import Dataset

    set_seed(SEED)
    rows = load_rows()
    labels = sorted({r["label"] for r in rows})
    lab2id = {l: i for i, l in enumerate(labels)}

    X = [r["text"] for r in rows]
    y = [lab2id[r["label"]] for r in rows]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                              random_state=SEED)

    silver_all = json.loads(ANNOT.read_text(encoding="utf-8"))[args.silver_key]["labels"]
    gold_qs = set(X)
    rng = random.Random(SEED)
    by_class: dict[str, list[str]] = {}
    for q, lvl in silver_all.items():
        if q not in gold_qs:
            by_class.setdefault(lvl, []).append(q)
    X_sil, y_sil = [], []
    for lvl, qs in sorted(by_class.items()):
        rng.shuffle(qs)
        for q in qs[:args.cap]:
            X_sil.append(q)
            y_sil.append(lab2id[NAME_TO_INT[lvl]])
    print(f"gold train {len(X_tr)}, silver added {len(X_sil)} "
          f"({dict(Counter(LEVELS[labels[i]] for i in y_sil))}), cap {args.cap}/class")

    X_tr2, y_tr2 = X_tr + X_sil, y_tr + y_sil

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    def mk(ds_x, ds_y):
        d = Dataset.from_dict({"text": ds_x, "labels": ds_y})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)

    dtr, dva, dte = mk(X_tr2, y_tr2), mk(X_va, y_va), mk(X_te, y_te)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(labels))
    counts = np.bincount(y_tr2, minlength=len(labels))
    weights = torch.tensor((counts.sum() / (len(labels) * counts)).astype(np.float32))
    print("class weights:", [round(float(w), 2) for w in weights])

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            lbl = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, lbl,
                                                     weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(MODELDIR / "_runs"), seed=SEED,
        num_train_epochs=6, per_device_train_batch_size=16, per_device_eval_batch_size=64,
        learning_rate=2e-5, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        logging_steps=20, report_to=[],
    )

    def metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "macro_f1": f1_score(p.label_ids, preds, average="macro")}

    trainer = WeightedTrainer(model=model, args=targs, train_dataset=dtr, eval_dataset=dva,
                              data_collator=DataCollatorWithPadding(tok),
                              compute_metrics=metrics)
    trainer.train()

    pred = trainer.predict(dte).predictions.argmax(-1)
    names = [LEVELS[l] for l in labels]
    rep = classification_report(y_te, pred, target_names=names, output_dict=True, zero_division=0)
    v1 = json.loads((REPO / "outputs" / "bloom_classifier.json").read_text(encoding="utf-8"))
    result = {
        "data": "EduQG gold train + validated LLM silver labels on the project's own "
                "verification questions (train split only; gold val/test identical to v1)",
        "silver_key": args.silver_key, "n_silver": len(X_sil), "cap_per_class": args.cap,
        "silver_class_counts": dict(Counter(LEVELS[labels[i]] for i in y_sil)),
        "n_test": len(X_te),
        "bert_v2": {"accuracy": round(accuracy_score(y_te, pred), 4),
                    "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
                    "per_class_f1": {n: round(rep[n]["f1-score"], 3) for n in names},
                    "confusion": confusion_matrix(y_te, pred).tolist()},
        "bert_v1_reference": v1["bert"],
        "reading": "Same gold test set as v1. The question is whether in-domain silver data "
                   "lifts the starved higher-order classes without hurting the others.",
    }
    OUTJ.write_text(json.dumps(result, indent=2), encoding="utf-8")
    MODELDIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODELDIR))
    tok.save_pretrained(str(MODELDIR))
    print(json.dumps({"v2": result["bert_v2"], "v1": v1["bert"]["per_class_f1"]}, indent=2))
    print(f"Saved {OUTJ.relative_to(REPO)} and model to {MODELDIR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
