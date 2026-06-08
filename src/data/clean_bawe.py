"""Clean the BAWE metadata into a sampling frame and report availability.

Input is the metadata produced by ``explore_bawe.py``
(``data/interim/bawe_metadata.csv``). This script fixes the known data quirks,
drops rows that cannot be used for stratified sampling, and prints the numbers
we need to design the sample: the disciplinary group x native/non-native
availability table, and how clustered the texts are by student.

Cleaning applied:
  - decode HTML entities in discipline labels (e.g. '&amp;' -> '&')
  - merge the case-split 'OTHER' into 'Other'
  - drop rows missing any stratification key (disciplinary group, L1) or the
    word count (needed for length matching at generation time)

Rows with a missing academic level are kept, because level is metadata here, not
a stratification variable. The drop reasons are all reported.

Output:
  - data/interim/bawe_clean.csv   the cleaned sampling frame

Run from the repo root (after explore_bawe.py):
    python src/data/clean_bawe.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "data" / "interim" / "bawe_metadata.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "interim" / "bawe_clean.csv"

GROUP_LABELS = {
    "AH": "Arts and Humanities",
    "LS": "Life Sciences",
    "PS": "Physical Sciences",
    "SS": "Social Sciences",
}

HTML_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'"}


def is_missing(series: pd.Series) -> pd.Series:
    """True where a value is NaN or an empty / 'nan' / 'none' string."""
    as_str = series.astype(str).str.strip().str.lower()
    return series.isna() | as_str.isin(["", "nan", "none"])


def clean_discipline(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    for entity, char in HTML_ENTITIES.items():
        out = out.str.replace(entity, char, regex=False)
    # Merge the case-split catch-all label.
    out = out.mask(out.str.lower() == "other", "Other")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean BAWE metadata into a sampling frame.")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.in_path.exists():
        print(f"ERROR: input not found: {args.in_path}")
        print("Run src/data/explore_bawe.py first to produce bawe_metadata.csv.")
        return 1

    df = pd.read_csv(args.in_path)
    n0 = len(df)
    print("=" * 70)
    print("BAWE CLEANING AND SAMPLING-FRAME AVAILABILITY")
    print("=" * 70)
    print(f"Input rows: {n0}")

    # ---- Fix discipline labels ----
    n_entity = int(df["discipline"].astype(str).str.contains("&amp;|&lt;|&gt;", regex=True).sum())
    n_other_upper = int((df["discipline"].astype(str).str.strip() == "OTHER").sum())
    df["discipline"] = clean_discipline(df["discipline"])
    print(f"Fixed HTML entities in discipline: {n_entity} row(s)")
    print(f"Merged 'OTHER' -> 'Other':         {n_other_upper} row(s)")

    # ---- Recompute native status from L1 (missing L1 -> NA, not False) ----
    native = df["L1"].astype(str).str.strip().str.lower().eq("english")
    df["native"] = native.mask(is_missing(df["L1"]))

    # ---- Identify rows to drop (missing a key field) ----
    miss_group = is_missing(df["disciplinary_group"])
    miss_l1 = is_missing(df["L1"])
    miss_words = df["words"].isna() if "words" in df else pd.Series(False, index=df.index)
    miss_level = is_missing(df["level"]) if "level" in df else pd.Series(False, index=df.index)

    drop_mask = miss_group | miss_l1 | miss_words
    print("\nDrop reasons (rows can fail more than one):")
    print(f"    missing disciplinary group: {int(miss_group.sum())}")
    print(f"    missing L1:                 {int(miss_l1.sum())}")
    print(f"    missing word count:         {int(miss_words.sum())}")
    print(f"    -> unique rows dropped:     {int(drop_mask.sum())}")
    print(f"    (kept, level missing but usable: {int((miss_level & ~drop_mask).sum())})")

    clean = df.loc[~drop_mask].copy()
    clean["native"] = clean["native"].astype(bool)
    clean["group_name"] = clean["disciplinary_group"].map(GROUP_LABELS).fillna(clean["disciplinary_group"])
    print(f"\nClean rows: {len(clean)}  (dropped {n0 - len(clean)})")
    print(f"Distinct disciplines after cleaning: {clean['discipline'].nunique()}")

    # ---- Availability table: group x native/non-native ----
    print("\nAvailability: disciplinary group x first-language status")
    status = clean["native"].map({True: "native", False: "non-native"})
    table = pd.crosstab(clean["disciplinary_group"], status, margins=True, margins_name="Total")
    # Order columns sensibly.
    cols = [c for c in ["native", "non-native", "Total"] if c in table.columns]
    table = table[cols]
    print(table.to_string())

    # Binding constraint for oversampling non-native.
    non_native_by_group = clean[~clean["native"]].groupby("disciplinary_group").size()
    native_by_group = clean[clean["native"]].groupby("disciplinary_group").size()
    print("\nPer-group availability (the constraint on cell sizes):")
    for g in sorted(clean["disciplinary_group"].unique()):
        nn = int(non_native_by_group.get(g, 0))
        na = int(native_by_group.get(g, 0))
        print(f"    {g} ({GROUP_LABELS.get(g, '?'):<20}) native {na:>4}   non-native {nn:>4}")
    print(f"\n    Smallest non-native group: {int(non_native_by_group.min())} "
          f"({non_native_by_group.idxmin()})")
    print(f"    Smallest native group:     {int(native_by_group.min())} "
          f"({native_by_group.idxmin()})")

    # ---- Student clustering ----
    if "student_id" in clean:
        sizes = clean.groupby("student_id").size()
        multi = int((sizes > 1).sum())
        print("\nStudent clustering (matters for train/test leakage later):")
        print(f"    unique students:            {sizes.shape[0]}")
        print(f"    texts per student: mean {sizes.mean():.2f}, median {sizes.median():.0f}, max {sizes.max()}")
        print(f"    students with >1 text:      {multi} ({100.0 * multi / sizes.shape[0]:.1f}%)")
        for k in (3, 5):
            share = 100.0 * int(sizes[sizes > k].sum()) / len(clean)
            print(f"    texts from students with >{k} texts: {int(sizes[sizes > k].sum())} ({share:.1f}%)")

    clean.to_csv(args.out_path, index=False, encoding="utf-8")
    print(f"\nSaved {args.out_path.relative_to(REPO_ROOT)}  ({len(clean)} rows, {clean.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
