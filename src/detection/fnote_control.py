"""The fnote control: retrain the detector with the residual token stripped (Chapter 3's promise).

The audit found one residual token, the bare word "fnote", surviving the markup cleaning: 115 of the
640 AI essays echo it from keyword prompts, so in the cleaned corpus it flips from a human marker to
a weak AI one. Chapter 3 records that it does not touch the function-word or stylometric results and
promises a retraining pass with the token stripped. This is that pass, run as a control rather than a
replacement: the token is stripped from BOTH classes, the same DeBERTa configuration is retrained on
the stripped corpus, and the test metrics are compared with the detector of record. If the numbers
hold, the promise is discharged and "fnote" is confirmed as non-load-bearing; if they move, the
chapter's numbers get updated. The detector of record in models/detector is left untouched.

    python src/detection/fnote_control.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
OUTDIR = REPO / "models" / "detector_fnote_control"
OUT = REPO / "outputs" / "fnote_control.json"
SEED = 42


def strip_fnote(text: str) -> str:
    return re.sub(r"\bfnote\b", " ", text, flags=re.IGNORECASE)


def main() -> int:
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    set_seed(SEED)

    df = pd.read_parquet(CORPUS)
    n_before = int(df["text"].str.contains(r"\bfnote\b", case=False).sum())
    df["text"] = df["text"].map(strip_fnote)
    n_after = int(df["text"].str.contains(r"\bfnote\b", case=False).sum())
    print(f"essays containing 'fnote': {n_before} -> {n_after}", flush=True)

    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def make(split):
        d = df[df["split"] == split]
        ds = Dataset.from_pandas(d[["text", "label", "native"]], preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=512), batched=True)

    train_ds, val_ds, test_ds = make("train"), make("val"), make("test")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)

    def cm(p):
        return {"f1": f1_score(p.label_ids, p.predictions.argmax(-1), zero_division=0)}

    targs = TrainingArguments(
        output_dir=str(OUTDIR), overwrite_output_dir=True,
        num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=4, per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), report_to="none", logging_steps=25, seed=SEED,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=cm)
    print(f"Training the fnote-stripped control on {len(train_ds)} essays", flush=True)
    trainer.train()

    pred = trainer.predict(test_ds)
    preds = pred.predictions.argmax(-1)
    y = pred.label_ids
    native = np.array(test_ds["native"])
    fair = {}
    for grp, mask in [("native", native), ("non_native", ~native)]:
        hu = mask & (y == 0)
        if hu.sum():
            fair[grp] = round(float(((preds == 1) & hu).sum() / hu.sum()), 4)

    result = {
        "control": "identical config and splits to the detector of record; only change is stripping "
                   "the bare token 'fnote' from both classes before training",
        "essays_containing_fnote_before": n_before,
        "test": {"accuracy": round(float(accuracy_score(y, preds)), 4),
                 "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
                 "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
                 "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
                 "confusion_matrix_[hu,ai]x[pred_hu,pred_ai]": confusion_matrix(y, preds).tolist()},
        "human_FPR_by_L1": fair,
        "detector_of_record_f1": 0.990,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== FNOTE CONTROL ===")
    print(json.dumps(result["test"], indent=2))
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
