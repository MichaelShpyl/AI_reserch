"""Bloom's-taxonomy question classifier (pipeline component 5): fine-tuned BERT-base.

Trains on the 903 Bloom-labelled questions in EduQG (Hadifar et al. 2022). Labels are revised-Bloom
cognitive levels; only levels 1 to 4 occur in the data (remember, understand, apply, analyse) and the
distribution is heavily imbalanced (660/114/110/19), so training uses class weights and results are
reported as macro-F1 with per-class detail, not accuracy alone.

Two models are compared on the same held-out test split:
  1. the keyword heuristic currently used in the question generator (the baseline), and
  2. fine-tuned BERT-base (the component the locked scope specifies).

Outputs: models/bloom_classifier/ (weights), outputs/bloom_classifier.json (metrics),
dissertation/figures/fig_bloom_classifier.png (confusion matrices side by side).

    python src/bloom/train_bloom_classifier.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "raw" / "eduqg"
OUTJ = REPO / "outputs" / "bloom_classifier.json"
FIGS = REPO / "dissertation" / "figures"
MODELDIR = REPO / "models" / "bloom_classifier"
SEED = 42

LEVELS = {1: "remember", 2: "understand", 3: "apply", 4: "analyse"}

# The transparent keyword heuristic from src/question_gen/generate_questions.py, mapped to level ids.
HEURISTIC = [
    (6, r"\b(design|propose|create|devise|what if|how might|suggest a)\b"),
    (5, r"\b(evaluate|justify|critique|assess|defend|do you agree|to what extent|how convincing|weigh)\b"),
    (4, r"\b(compare|contrast|analyse|analyze|distinguish|why does|how does|what evidence|relationship between|implication)\b"),
    (3, r"\b(how would|apply|calculate|use the|give an example|in practice|what would happen)\b"),
    (2, r"\b(explain|describe|summaris|summariz|what is meant|why|interpret|in your own words)\b"),
    (1, r"\b(define|list|name|state|identify|what is|who|when|where)\b"),
]


def heuristic_level(q: str) -> int:
    ql = q.lower()
    for level, pat in HEURISTIC:
        if re.search(pat, ql):
            return min(level, 4)  # data has no level 5/6; cap so the comparison is fair
    return 2


def load_rows() -> list[dict]:
    rows = []
    for f in ["qg_train_v0.json", "qg_valid_v0.json"]:
        for ch in json.loads((DATA / f).read_text(encoding="utf-8")):
            for q in ch.get("questions", []):
                b = q.get("bloom")
                if b:
                    qt = q["question"].get("normal_format") or q["question"].get("question_text")
                    if qt and str(b).strip().isdigit():
                        rows.append({"text": qt.strip(), "label": int(str(b).strip())})
    return rows


def main() -> int:
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    from datasets import Dataset

    set_seed(SEED)
    rows = load_rows()
    labels = sorted({r["label"] for r in rows})
    lab2id = {l: i for i, l in enumerate(labels)}
    print(f"{len(rows)} labelled questions; levels {labels} "
          f"({[sum(r['label'] == l for r in rows) for l in labels]})")

    X = [r["text"] for r in rows]
    y = [lab2id[r["label"]] for r in rows]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                              random_state=SEED)
    print(f"splits: train {len(X_tr)}, val {len(X_va)}, test {len(X_te)}")

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    def mk(ds_x, ds_y):
        d = Dataset.from_dict({"text": ds_x, "labels": ds_y})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)

    dtr, dva, dte = mk(X_tr, y_tr), mk(X_va, y_va), mk(X_te, y_te)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(labels))
    # Class weights against the 660/114/110/19 imbalance.
    counts = np.bincount(y_tr, minlength=len(labels))
    weights = torch.tensor((counts.sum() / (len(labels) * counts)).astype(np.float32))
    print("class weights:", [round(float(w), 2) for w in weights])

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            lbl = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, lbl,
                                                     weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    args = TrainingArguments(
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

    trainer = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                              data_collator=DataCollatorWithPadding(tok),
                              compute_metrics=metrics)
    trainer.train()

    # Held-out test: BERT vs the heuristic baseline.
    pred = trainer.predict(dte).predictions.argmax(-1)
    heur = [lab2id[heuristic_level(x)] for x in X_te]
    names = [LEVELS[l] for l in labels]
    rep_bert = classification_report(y_te, pred, target_names=names, output_dict=True,
                                     zero_division=0)
    rep_heur = classification_report(y_te, heur, target_names=names, output_dict=True,
                                     zero_division=0)
    result = {
        "data": "EduQG Bloom-labelled subset (Hadifar et al. 2022)",
        "n_total": len(rows), "n_test": len(X_te),
        "levels": names,
        "class_counts": {LEVELS[l]: int(sum(r['label'] == l for r in rows)) for l in labels},
        "bert": {"accuracy": round(accuracy_score(y_te, pred), 4),
                 "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
                 "per_class_f1": {n: round(rep_bert[n]["f1-score"], 3) for n in names},
                 "confusion": confusion_matrix(y_te, pred).tolist()},
        "heuristic": {"accuracy": round(accuracy_score(y_te, heur), 4),
                      "macro_f1": round(f1_score(y_te, heur, average="macro"), 4),
                      "per_class_f1": {n: round(rep_heur[n]["f1-score"], 3) for n in names},
                      "confusion": confusion_matrix(y_te, heur).tolist()},
        "reading": "Macro-F1 is the fair headline: the data is 73% level-1, so accuracy alone "
                   "flatters majority-class behaviour. Class-weighted training; stratified splits.",
    }
    OUTJ.write_text(json.dumps(result, indent=2), encoding="utf-8")
    MODELDIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODELDIR))
    tok.save_pretrained(str(MODELDIR))
    make_figure(result, names)

    print(json.dumps({k: result[k] for k in ("bert", "heuristic")}, indent=2))
    print(f"Saved {OUTJ.relative_to(REPO)} and model to {MODELDIR.relative_to(REPO)}")
    return 0


def make_figure(result: dict, names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, key, title in [(axes[0], "heuristic", "Keyword heuristic (baseline)"),
                           (axes[1], "bert", "Fine-tuned BERT-base")]:
        cm = np.array(result[key]["confusion"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "#222831", fontsize=10)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        mf = result[key]["macro_f1"]
        ax.set_title(f"{title}\nmacro-F1 {mf:.2f}", fontsize=11, fontweight="bold", color="#222831")
    fig.suptitle("Bloom's-level classification on the EduQG test split", fontsize=12.5,
                 fontweight="bold", color="#222831")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_bloom_classifier.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_bloom_classifier.png")


if __name__ == "__main__":
    raise SystemExit(main())
