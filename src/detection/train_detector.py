"""Fine-tune a transformer detector on the human-vs-AI corpus (first baseline).

This is the transformer half of the hybrid detector, on its own first. It fine-tunes
DeBERTa-v3-base (per CLAUDE.md) to classify Human (0) vs AI (1), using the student-level
splits. Reports accuracy, precision, recall, F1, the confusion matrix, and a native vs
non-native breakdown (the fairness analysis seed). Stylometric fusion and M4 pre-training
come next.

    python src/detection/train_detector.py
    python src/detection/train_detector.py --model roberta-base   # comparison run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "processed" / "detection_corpus.parquet"
OUTDIR = REPO / "models" / "detector"
METRICS = REPO / "outputs" / "detector_metrics.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--corpus", default=str(CORPUS),
                    help="parquet corpus to train on (raw or cleaned)")
    ap.add_argument("--out", default=str(METRICS),
                    help="where to write the metrics JSON")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    metrics_path = Path(args.out)

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              set_seed)
    set_seed(args.seed)

    df = pd.read_parquet(args.corpus)
    tok = AutoTokenizer.from_pretrained(args.model)

    def make(split):
        d = df[df["split"] == split]
        ds = Dataset.from_pandas(d[["text", "label", "native"]], preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True)

    train_ds, val_ds, test_ds = make("train"), make("val"), make("test")

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        y = p.label_ids
        return {"accuracy": accuracy_score(y, preds),
                "precision": precision_score(y, preds, zero_division=0),
                "recall": recall_score(y, preds, zero_division=0),
                "f1": f1_score(y, preds, zero_division=0)}

    targs = TrainingArguments(
        output_dir=str(OUTDIR), overwrite_output_dir=True,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.accum,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), report_to="none",
        logging_steps=25, seed=args.seed,
    )
    collator = DataCollatorWithPadding(tok)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, processing_class=tok,
                      data_collator=collator, compute_metrics=compute_metrics)

    print(f"Training {args.model} on {len(train_ds)} examples ...", flush=True)
    trainer.train()

    val_m = trainer.evaluate(val_ds)
    test_pred = trainer.predict(test_ds)
    preds = test_pred.predictions.argmax(-1)
    y = test_pred.label_ids
    cm = confusion_matrix(y, preds).tolist()
    test_m = {"accuracy": accuracy_score(y, preds),
              "precision": precision_score(y, preds, zero_division=0),
              "recall": recall_score(y, preds, zero_division=0),
              "f1": f1_score(y, preds, zero_division=0),
              "confusion_matrix_[hu,ai]x[pred_hu,pred_ai]": cm}

    # Fairness: false-positive rate (human flagged as AI) by first-language status.
    native = np.array(test_ds["native"])
    fairness = {}
    for grp, mask in [("native", native), ("non_native", ~native)]:
        hu = mask & (y == 0)
        if hu.sum():
            fpr = float(((preds == 1) & hu).sum() / hu.sum())
            fairness[grp] = {"n_human": int(hu.sum()), "false_positive_rate": round(fpr, 3)}

    result = {"model": args.model, "val": {k: round(float(v), 4) for k, v in val_m.items()
                                           if k.startswith("eval_")},
              "test": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in test_m.items()},
              "fairness_human_FPR_by_L1": fairness}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== TEST RESULTS ===", flush=True)
    print(json.dumps(result["test"], indent=2), flush=True)
    print("fairness (human false-positive rate by L1):", json.dumps(fairness), flush=True)
    print(f"\nSaved {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
