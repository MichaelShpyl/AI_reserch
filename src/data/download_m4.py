"""Download the M4 / SemEval-2024 Task 8 English binary set and summarise it.

This is the detector's pre-training data: Subtask A, monolingual track, which is a
binary human (0) vs machine (1) corpus across several generators and domains. Source:
the d0rj/SemEval2024-task8 mirror on the Hugging Face Hub (the official data is on Google
Drive; this mirror avoids the flaky Drive download and matches the documented counts).

The official documented size is 119,757 train and 5,000 dev for Subtask A monolingual;
the script prints the actual counts so the provenance can be checked.

Saves each split to data/raw/m4/<split>.parquet (gitignored) and prints a summary.

    python src/data/download_m4.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
from datasets import load_dataset

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "raw" / "m4"
DS_ID = "d0rj/SemEval2024-task8"
CONFIG = "subtaskA_monolingual"
EXPECTED_TRAIN = 119757  # documented Subtask A monolingual train size


def pick(cols, *candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DS_ID} [{CONFIG}] ...", flush=True)
    ds = load_dataset(DS_ID, CONFIG)
    print("Splits:", {k: ds[k].num_rows for k in ds}, flush=True)

    for split in ds:
        df = ds[split].to_pandas()
        path = OUT / f"{split}.parquet"
        df.to_parquet(path, index=False)
        print(f"\n=== split: {split} ({len(df)} rows) -> {path.relative_to(REPO)} ===")
        print("columns:", list(df.columns))

        label_col = pick(df.columns, "label", "labels")
        text_col = pick(df.columns, "text")
        if label_col:
            vc = df[label_col].value_counts(dropna=False).sort_index()
            print("label balance (0=human, 1=machine):")
            for k, v in vc.items():
                print(f"    {k}: {v}  ({100*v/len(df):.1f}%)")
        # Low-cardinality breakdowns (source domain, generator model).
        for col in df.columns:
            if col in (label_col, text_col):
                continue
            if df[col].dtype == object and df[col].nunique() <= 30:
                print(f"by {col}:")
                for k, v in df[col].value_counts().head(12).items():
                    print(f"    {str(k):<24} {v}")
        if text_col:
            wlen = df[text_col].astype(str).str.split().str.len()
            print(f"text length (words): min {wlen.min()} median {int(wlen.median())} "
                  f"mean {wlen.mean():.0f} max {wlen.max()}")

        if split == "train":
            ok = "OK" if len(df) == EXPECTED_TRAIN else f"DIFFERS (expected {EXPECTED_TRAIN})"
            print(f"provenance check: train rows {len(df)} -> {ok}")

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
