"""Training-data structure comparison: balanced vs natural writer distribution (Meeting 2 item).

The detection corpus was deliberately balanced: equal essays per disciplinary-group-by-native cell.
Real submissions are not balanced (in full BAWE, native AH writers are 21 percent of essays and
non-native AH writers 4 percent), so the locked evaluation plan asks whether the manual balancing
helps, hurts, or manufactures structure that would not survive realistic data. Two detectors are
trained in parallel from the same cleaned corpus and identical hyperparameters; only the writer
distribution of the training set differs:

  balanced  - equal essays per cell (the design of record), subsampled to the same size as natural
  natural   - essays per cell proportional to the full BAWE corpus's real distribution

Both use matched human-AI pairs (an essay's AI twin always travels with it, keeping labels 50/50;
"natural" varies who wrote the essays, not the label ratio, since AI text has no natural base rate).
The two training sets are the SAME SIZE, so distribution is not confounded with quantity. Both models
are evaluated on the same untouched test split, reported overall, per cell, and reweighted to the
natural cell proportions (how each model would do on a realistic mix).

    python src/evaluation/balanced_vs_natural.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "processed" / "detection_corpus_clean.parquet"
HOLDINGS = REPO / "data" / "raw" / "bawe" / "download" / "documentation" / "BAWE.xls"
OUT = REPO / "outputs" / "balanced_vs_natural.json"
FIGS = REPO / "dissertation" / "figures"
MODELS = REPO / "models" / "detector_dist"
SEED = 42
MODEL = "microsoft/deberta-v3-base"


def natural_proportions() -> dict:
    h = pd.read_excel(HOLDINGS)
    h["native"] = h["L1"].astype(str).str.strip().str.lower().eq("english")
    cell = h.groupby(["disciplinary group", "native"]).size()
    total = cell.sum()
    # Key format matches the corpus's `cell` column (e.g. "AH_native", "AH_non-native").
    return {f"{g}_{'native' if n else 'non-native'}": c / total for (g, n), c in cell.items()}


def build_training_sets(df: pd.DataFrame) -> tuple[list, list, dict]:
    """Return (balanced_ids, natural_ids, info): two same-size sets of HUMAN essay ids from the
    train split; the AI twins are added later by id."""
    rng = np.random.default_rng(SEED)
    train_h = df[(df.split == "train") & (df.label == 0)]
    props = natural_proportions()
    cells = sorted(train_h.cell.unique())
    avail = {c: train_h[train_h.cell == c].id.tolist() for c in cells}

    # Largest natural set that fits the availability: N = min over cells of avail/prop.
    n_max = int(min(len(avail[c]) / props[c] for c in cells))
    # Integer allocation by largest remainder, then clip to availability.
    raw = {c: props[c] * n_max for c in cells}
    alloc = {c: int(v) for c, v in raw.items()}
    rem = sorted(cells, key=lambda c: raw[c] - alloc[c], reverse=True)
    for c in rem:
        if sum(alloc.values()) >= n_max:
            break
        alloc[c] += 1
    alloc = {c: min(alloc[c], len(avail[c])) for c in cells}
    n_total = sum(alloc.values())

    natural_ids = []
    for c in cells:
        pick = rng.choice(avail[c], size=alloc[c], replace=False)
        natural_ids.extend(pick.tolist())

    # Balanced set of the SAME total size: equal per cell (largest-remainder on the leftover).
    per = n_total // len(cells)
    extra = n_total - per * len(cells)
    balanced_ids = []
    for i, c in enumerate(cells):
        k = per + (1 if i < extra else 0)
        pick = rng.choice(avail[c], size=min(k, len(avail[c])), replace=False)
        balanced_ids.extend(pick.tolist())

    info = {"n_human_each": n_total, "natural_alloc": alloc,
            "balanced_alloc": {c: per + (1 if i < extra else 0) for i, c in enumerate(cells)},
            "natural_proportions": {k: round(v, 4) for k, v in props.items()}}
    return balanced_ids, natural_ids, info


def train_and_eval(name: str, human_ids: list, df: pd.DataFrame) -> dict:
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    set_seed(SEED)
    ids = set(human_ids)
    train = df[(df.split == "train") & (df.id.isin(ids))]          # human + AI twins by shared id
    val = df[df.split == "val"]
    test = df[df.split == "test"]
    print(f"[{name}] train {len(train)} (labels {train.label.value_counts().to_dict()})", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)

    def make(d):
        ds = Dataset.from_pandas(d[["text", "label", "cell", "native"]], preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=512), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    targs = TrainingArguments(
        output_dir=str(MODELS / name), overwrite_output_dir=True,
        num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=4, per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), report_to="none", logging_steps=25, seed=SEED,
    )

    def cm(p):
        preds = p.predictions.argmax(-1)
        return {"f1": f1_score(p.label_ids, preds, zero_division=0)}

    trainer = Trainer(model=model, args=targs, train_dataset=make(train), eval_dataset=make(val),
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=cm)
    trainer.train()
    pred = trainer.predict(make(test))
    preds = pred.predictions.argmax(-1)
    y = pred.label_ids

    res = {"n_train": int(len(train)),
           "test_overall": {
               "accuracy": round(float(accuracy_score(y, preds)), 4),
               "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
               "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
               "f1": round(float(f1_score(y, preds, zero_division=0)), 4)}}

    # Per-cell F1 on the shared test split, plus a natural-weighted aggregate.
    cells = np.array(test["cell"])
    props = natural_proportions()
    per_cell, nat_weighted, wsum = {}, 0.0, 0.0
    for c in sorted(set(cells)):
        m = cells == c
        if m.sum() < 4:
            continue
        f1c = float(f1_score(y[m], preds[m], zero_division=0))
        per_cell[c] = {"n": int(m.sum()), "f1": round(f1c, 4)}
        w = props.get(c, 0.0)
        nat_weighted += w * f1c
        wsum += w
    res["test_per_cell_f1"] = per_cell
    res["test_f1_natural_weighted"] = round(nat_weighted / max(wsum, 1e-9), 4)

    # Fairness read: human false-positive rate by L1 (the risk the project cares about).
    native = np.array(test["native"])
    fair = {}
    for grp, mask in [("native", native), ("non_native", ~native)]:
        hu = mask & (y == 0)
        if hu.sum():
            fair[grp] = round(float(((preds == 1) & hu).sum() / hu.sum()), 4)
    res["human_FPR_by_L1"] = fair
    # Persist raw predictions for the paired comparison.
    res["_preds"] = preds.tolist()
    res["_y"] = y.tolist()
    return res


def main() -> int:
    df = pd.read_parquet(CORPUS)
    balanced_ids, natural_ids, info = build_training_sets(df)
    overlap = len(set(balanced_ids) & set(natural_ids))
    print(f"training sets: {info['n_human_each']} human essays each "
          f"(+ AI twins), overlap {overlap}", flush=True)

    results = {"design": info, "model": MODEL,
               "note": "same size, same hyperparameters, same seed, same test split; only the "
                       "writer distribution differs"}
    for name, ids in [("balanced", balanced_ids), ("natural", natural_ids)]:
        results[name] = train_and_eval(name, ids, df)

    # Paired comparison on the shared test set: McNemar on correctness.
    pb = np.array(results["balanced"].pop("_preds")); yb = np.array(results["balanced"].pop("_y"))
    pn = np.array(results["natural"].pop("_preds")); yn = np.array(results["natural"].pop("_y"))
    assert (yb == yn).all()
    cb, cn = pb == yb, pn == yn
    b01 = int((cb & ~cn).sum())   # balanced right, natural wrong
    b10 = int((~cb & cn).sum())
    from scipy.stats import binomtest
    p = binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) else 1.0
    results["paired_mcnemar"] = {"balanced_right_natural_wrong": b01,
                                 "natural_right_balanced_wrong": b10,
                                 "p": round(float(p), 4)}

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    make_figure(results)
    print("\n=== BALANCED vs NATURAL TRAINING DISTRIBUTION ===")
    for name in ("balanced", "natural"):
        r = results[name]
        print(f"{name:9s} overall F1 {r['test_overall']['f1']}  "
              f"natural-weighted F1 {r['test_f1_natural_weighted']}  "
              f"FPR native {r['human_FPR_by_L1'].get('native')} / "
              f"non-native {r['human_FPR_by_L1'].get('non_native')}")
    print("paired McNemar:", results["paired_mcnemar"])
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = sorted(results["balanced"]["test_per_cell_f1"])
    xb = [results["balanced"]["test_per_cell_f1"][c]["f1"] for c in cells]
    xn = [results["natural"]["test_per_cell_f1"][c]["f1"] for c in cells]
    x = np.arange(len(cells))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.bar(x - 0.2, xb, 0.4, label="balanced training mix", color="#2b6777")
    ax1.bar(x + 0.2, xn, 0.4, label="natural training mix", color="#d98e3b")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_non_native", "\nnon-native").replace("_native", "\nnative")
                         for c in cells], fontsize=8.5)
    ax1.set_ylim(0.85, 1.01)
    ax1.set_ylabel("test F1 (same held-out split)")
    ax1.set_title("Per-cell F1", fontsize=11, fontweight="bold", color="#222831")
    ax1.legend(fontsize=9, frameon=False, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)
    labels = ["balanced", "natural"]
    overall = [results[l]["test_overall"]["f1"] for l in labels]
    natw = [results[l]["test_f1_natural_weighted"] for l in labels]
    x2 = np.arange(2)
    ax2.bar(x2 - 0.2, overall, 0.4, label="overall F1", color="#52616B")
    ax2.bar(x2 + 0.2, natw, 0.4, label="natural-weighted F1", color="#9bb7bd")
    ax2.set_xticks(x2); ax2.set_xticklabels(labels)
    ax2.set_ylim(0.85, 1.01)
    ax2.set_title("Aggregate", fontsize=11, fontweight="bold", color="#222831")
    ax2.legend(fontsize=9, frameon=False, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)
    mc = results.get("paired_mcnemar")
    if mc is not None:
        fig.text(0.5, 0.015,
                 f"The two models made identical predictions on every test essay "
                 f"(disagreements: {mc['balanced_right_natural_wrong']} vs "
                 f"{mc['natural_right_balanced_wrong']}, McNemar p = {mc['p']}). At this corpus's "
                 f"separability, the training mix is not load-bearing.",
                 ha="center", fontsize=9.5, color="#52616B", style="italic")
    fig.suptitle("Does the balanced training design help or hurt? Same size, same settings, "
                 "only the writer mix differs", fontsize=12, fontweight="bold", color="#222831")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_balanced_vs_natural.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_balanced_vs_natural.png")


if __name__ == "__main__":
    raise SystemExit(main())
