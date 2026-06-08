"""Load the BAWE corpus, summarise it, and save reproducible artefacts.

This is the Phase 1 exploration step. It reads the BAWE holdings spreadsheet
(``BAWE.xls``), prints a summary of the corpus (counts by discipline, by
disciplinary group, by academic level, by native / non-native first language,
and the distribution of essay lengths in words), and validates the spreadsheet's
``words`` column against a whitespace word count of the plain-text essays.

Why validate length: the whole detection design depends on matching AI-written
essays to human ones on length. If we later trust a length number that is wrong,
the matched corpus is wrong, so we check the source of truth once, here.

Artefacts written:
  - data/interim/bawe_metadata.csv   cleaned metadata, one row per text
  - outputs/bawe_summary.txt         the full printed summary
  - outputs/bawe_length_hist.png     overall length histogram
  - outputs/bawe_length_by_group.png length by disciplinary group (boxplot)

Run from the repo root:
    python src/data/explore_bawe.py
Options:
    --bawe-root PATH      where the extracted corpus lives (default data/raw/bawe)
    --crosscheck N        how many text files to word-count for validation (default 200)
    --crosscheck-all      word-count every text file instead of a sample
    --no-figures          skip saving the PNG figures
    --seed N              random seed for the cross-check sample (default 42)
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is two levels up from this file: src/data/explore_bawe.py -> repo.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BAWE_ROOT = REPO_ROOT / "data" / "raw" / "bawe"
DEFAULT_INTERIM = REPO_ROOT / "data" / "interim"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"

# Standard BAWE level coding (see corpus manual).
LEVEL_LABELS = {
    1: "UG year 1",
    2: "UG year 2",
    3: "UG year 3",
    4: "Masters (taught)",
}

# Disciplinary group codes used in BAWE.
GROUP_LABELS = {
    "AH": "Arts and Humanities",
    "LS": "Life Sciences",
    "PS": "Physical Sciences",
    "SS": "Social Sciences",
}

# Candidate header names for each field we need. We verified the real headers,
# but match case-insensitively and by substring so a future re-download that
# renames a column slightly does not silently break the summary.
FIELD_CANDIDATES = {
    "id": ["id"],
    "student_id": ["student_id", "student id"],
    "title": ["title"],
    "module": ["module"],
    "course": ["course"],
    "discipline": ["discipline"],
    "group": ["disciplinary group", "disciplinary_group", "group"],
    "level": ["level"],
    "grade": ["grade"],
    "genre": ["genre family", "genre_family", "genre"],
    "words": ["words", "word count", "word_count"],
    "l1": ["l1", "first language", "first_language"],
    "gender": ["gender"],
    "yob": ["year of birth", "year_of_birth", "yob"],
    "education": ["education"],
}


class Tee:
    """Collects lines so we can print them and also write them to a file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def find_file(root: Path, name: str) -> Path | None:
    """Find the first file named ``name`` (case-insensitive) under ``root``."""
    name_low = name.lower()
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == name_low:
            return p
    return None


def find_dir(root: Path, name: str) -> Path | None:
    """Find the first directory named ``name`` (case-insensitive) under ``root``."""
    name_low = name.lower()
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == name_low:
            return p
    return None


def resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map each needed field to a real column name, or None if not found."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    resolved: dict[str, str | None] = {}
    for field, candidates in FIELD_CANDIDATES.items():
        chosen: str | None = None
        # Exact (normalised) match first.
        for cand in candidates:
            if cand in lookup:
                chosen = lookup[cand]
                break
        # Then substring match (handles the long 'education (...)' header).
        if chosen is None:
            for cand in candidates:
                for norm, original in lookup.items():
                    if cand in norm:
                        chosen = original
                        break
                if chosen is not None:
                    break
        resolved[field] = chosen
    return resolved


def fmt_counts(series: pd.Series, total: int) -> list[str]:
    """Format a value_counts series as 'value: count (pct%)' lines."""
    lines = []
    counts = series.value_counts(dropna=False)
    for value, count in counts.items():
        pct = 100.0 * count / total
        label = "(missing)" if pd.isna(value) else str(value)
        lines.append(f"    {label:<28} {count:>5}  ({pct:4.1f}%)")
    return lines


def summarise(df: pd.DataFrame, cols: dict[str, str | None], out: Tee) -> pd.DataFrame:
    """Print the summary and return a cleaned metadata frame."""
    total = len(df)

    out("=" * 70)
    out("BAWE CORPUS SUMMARY")
    out("=" * 70)
    out(f"Generated:        {datetime.now():%Y-%m-%d %H:%M}")
    out(f"pandas version:   {pd.__version__}")
    out(f"Total texts:      {total}")
    out("")

    out("Resolved columns (field -> spreadsheet column):")
    for field, col in cols.items():
        marker = "" if col is not None else "   <-- NOT FOUND"
        out(f"    {field:<12} -> {col}{marker}")
    out("")

    # ---- Build a tidy, cleaned frame keyed on the resolved columns ----
    clean = pd.DataFrame()
    rename_map = {
        "id": "id",
        "student_id": "student_id",
        "title": "title",
        "module": "module",
        "course": "course",
        "discipline": "discipline",
        "group": "disciplinary_group",
        "level": "level",
        "grade": "grade",
        "genre": "genre_family",
        "words": "words",
        "l1": "L1",
        "gender": "gender",
        "yob": "year_of_birth",
        "education": "education_code",
    }
    for field, new_name in rename_map.items():
        src = cols.get(field)
        if src is not None:
            clean[new_name] = df[src]

    # Numeric coercions.
    if "words" in clean:
        clean["words"] = pd.to_numeric(clean["words"], errors="coerce")
    if "level" in clean:
        clean["level"] = pd.to_numeric(clean["level"], errors="coerce")

    # Derived fields.
    if "level" in clean:
        clean["level_label"] = clean["level"].map(
            lambda x: LEVEL_LABELS.get(int(x), f"level {x}") if pd.notna(x) else "(missing)"
        )
    if "L1" in clean:
        clean["native"] = clean["L1"].astype(str).str.strip().str.lower().eq("english")

    # ---- Discipline ----
    if "discipline" in clean:
        n_disc = clean["discipline"].nunique(dropna=True)
        out(f"Disciplines: {n_disc} distinct")
        out("\n".join(fmt_counts(clean["discipline"], total)))
        out("")

    # ---- Disciplinary group ----
    if "disciplinary_group" in clean:
        out("Disciplinary group:")
        grp = clean["disciplinary_group"].astype(str).str.strip()
        grp_named = grp.map(lambda g: f"{g} ({GROUP_LABELS.get(g, '?')})")
        out("\n".join(fmt_counts(grp_named, total)))
        out("")

    # ---- Academic level ----
    if "level_label" in clean:
        out("Academic level:")
        out("\n".join(fmt_counts(clean["level_label"], total)))
        out("")

    # ---- Native / non-native ----
    if "native" in clean:
        out("First-language status (native = L1 is English):")
        n_native = int(clean["native"].sum())
        n_non = int((~clean["native"]).sum())
        out(f"    Native (L1 English)        {n_native:>5}  ({100.0 * n_native / total:4.1f}%)")
        out(f"    Non-native                 {n_non:>5}  ({100.0 * n_non / total:4.1f}%)")
        out("")
        out("    Top 12 first languages (L1):")
        top_l1 = clean["L1"].value_counts(dropna=False).head(12)
        for lang, count in top_l1.items():
            label = "(missing)" if pd.isna(lang) else str(lang)
            out(f"        {label:<22} {count:>5}")
        out("")

    # ---- Length distribution ----
    if "words" in clean:
        w = clean["words"].dropna()
        n_missing = int(clean["words"].isna().sum())
        out("Essay length in words (from spreadsheet 'words' column):")
        out(f"    count:   {len(w)}   (missing: {n_missing})")
        out(f"    min:     {w.min():.0f}")
        out(f"    25%:     {w.quantile(0.25):.0f}")
        out(f"    median:  {w.median():.0f}")
        out(f"    mean:    {w.mean():.0f}")
        out(f"    75%:     {w.quantile(0.75):.0f}")
        out(f"    max:     {w.max():.0f}")
        out(f"    std:     {w.std():.0f}")
        out("")
        bins = [0, 500, 1000, 2000, 3000, 4000, 5000, np.inf]
        labels = ["<500", "500-999", "1000-1999", "2000-2999",
                  "3000-3999", "4000-4999", "5000+"]
        banded = pd.cut(w, bins=bins, labels=labels, right=False)
        out("    Length bands:")
        band_counts = banded.value_counts().reindex(labels)
        for band, count in band_counts.items():
            pct = 100.0 * count / len(w)
            out(f"        {band:<12} {count:>5}  ({pct:4.1f}%)")
        out("")

    return clean


def crosscheck_lengths(
    clean: pd.DataFrame,
    txt_dir: Path | None,
    n_sample: int,
    do_all: bool,
    seed: int,
    out: Tee,
) -> None:
    """Compare the 'words' column to a whitespace count of the .txt files."""
    out("-" * 70)
    out("LENGTH CROSS-CHECK (spreadsheet 'words' vs whitespace count of .txt)")
    out("-" * 70)
    if txt_dir is None:
        out("    CORPUS_TXT directory not found, skipping cross-check.")
        out("")
        return
    if "id" not in clean or "words" not in clean:
        out("    Need both 'id' and 'words' columns, skipping cross-check.")
        out("")
        return

    pairs = clean[["id", "words"]].dropna()
    ids = pairs["id"].astype(str).tolist()
    if not do_all and n_sample < len(ids):
        rng = random.Random(seed)
        idx = rng.sample(range(len(ids)), n_sample)
        ids_to_check = [ids[i] for i in idx]
        out(f"    Sampled {n_sample} texts (seed {seed}).")
    else:
        ids_to_check = ids
        out(f"    Checking all {len(ids)} texts.")

    words_by_id = dict(zip(pairs["id"].astype(str), pairs["words"]))
    sheet_vals, file_vals, missing = [], [], 0
    for text_id in ids_to_check:
        f = txt_dir / f"{text_id}.txt"
        if not f.exists():
            missing += 1
            continue
        n_file = len(f.read_text(encoding="utf-8", errors="ignore").split())
        sheet_vals.append(float(words_by_id[text_id]))
        file_vals.append(float(n_file))

    if not sheet_vals:
        out(f"    No matching .txt files found (missing: {missing}).")
        out("")
        return

    s = np.array(sheet_vals)
    f_arr = np.array(file_vals)
    corr = float(np.corrcoef(s, f_arr)[0, 1])
    mad = float(np.mean(np.abs(s - f_arr)))
    median_ratio = float(np.median(f_arr / s))
    out(f"    Compared:        {len(s)}   (missing files: {missing})")
    out(f"    Pearson r:       {corr:.4f}")
    out(f"    Mean abs diff:   {mad:.1f} words")
    out(f"    Median ratio (file/sheet): {median_ratio:.3f}")
    out("    A high correlation means the 'words' column is reliable ground truth")
    out("    for the length-matching step later.")
    out("")


def save_figures(clean: pd.DataFrame, outdir: Path, out: Tee) -> None:
    """Save the length histogram and a length-by-group boxplot."""
    if "words" not in clean:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        out(f"    (figures skipped: {exc})")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    w = clean["words"].dropna()

    # Histogram.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(w, bins=40, color="#4878a8", edgecolor="white")
    ax.axvline(w.median(), color="#c44", linestyle="--", label=f"median {w.median():.0f}")
    ax.axvline(w.mean(), color="#2a2", linestyle=":", label=f"mean {w.mean():.0f}")
    ax.set_xlabel("Essay length (words)")
    ax.set_ylabel("Number of essays")
    ax.set_title("BAWE essay length distribution")
    ax.legend()
    fig.tight_layout()
    hist_path = outdir / "bawe_length_hist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    out(f"    Saved {hist_path.relative_to(REPO_ROOT)}")

    # Length by disciplinary group.
    if "disciplinary_group" in clean:
        groups = sorted(clean["disciplinary_group"].dropna().astype(str).str.strip().unique())
        data = [clean.loc[clean["disciplinary_group"].astype(str).str.strip() == g, "words"].dropna()
                for g in groups]
        if data and all(len(d) for d in data):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(data, tick_labels=[f"{g}\n{GROUP_LABELS.get(g, '')}" for g in groups], showfliers=False)
            ax.set_ylabel("Essay length (words)")
            ax.set_title("BAWE essay length by disciplinary group")
            fig.tight_layout()
            grp_path = outdir / "bawe_length_by_group.png"
            fig.savefig(grp_path, dpi=150)
            plt.close(fig)
            out(f"    Saved {grp_path.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore and summarise the BAWE corpus.")
    parser.add_argument("--bawe-root", type=Path, default=DEFAULT_BAWE_ROOT,
                        help="Folder containing the extracted BAWE corpus.")
    parser.add_argument("--crosscheck", type=int, default=200,
                        help="Number of text files to word-count for validation.")
    parser.add_argument("--crosscheck-all", action="store_true",
                        help="Word-count every text file instead of a sample.")
    parser.add_argument("--no-figures", action="store_true", help="Skip saving figures.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the cross-check sample.")
    parser.add_argument("--interim", type=Path, default=DEFAULT_INTERIM)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if not args.bawe_root.exists():
        print(f"ERROR: BAWE root not found: {args.bawe_root}", file=sys.stderr)
        print("Extract the corpus there, or pass --bawe-root.", file=sys.stderr)
        return 1

    xls_path = find_file(args.bawe_root, "BAWE.xls")
    if xls_path is None:
        # Fall back to any spreadsheet under the root.
        for ext in ("*.xls", "*.xlsx"):
            hits = list(args.bawe_root.rglob(ext))
            if hits:
                xls_path = hits[0]
                break
    if xls_path is None:
        print(f"ERROR: no BAWE.xls (or any .xls/.xlsx) found under {args.bawe_root}", file=sys.stderr)
        return 1

    out = Tee()
    out(f"Spreadsheet:      {xls_path.relative_to(REPO_ROOT)}")
    df = pd.read_excel(xls_path, sheet_name=0)
    cols = resolve_columns(df)

    clean = summarise(df, cols, out)

    txt_dir = find_dir(args.bawe_root, "CORPUS_TXT")
    crosscheck_lengths(clean, txt_dir, args.crosscheck, args.crosscheck_all, args.seed, out)

    # Save artefacts.
    out("-" * 70)
    out("ARTEFACTS")
    out("-" * 70)
    args.interim.mkdir(parents=True, exist_ok=True)
    interim_csv = args.interim / "bawe_metadata.csv"
    clean.to_csv(interim_csv, index=False, encoding="utf-8")
    out(f"    Saved {interim_csv.relative_to(REPO_ROOT)}  ({len(clean)} rows, {clean.shape[1]} cols)")

    if not args.no_figures:
        save_figures(clean, args.outputs, out)

    summary_txt = args.outputs / "bawe_summary.txt"
    out.save(summary_txt)
    print(f"    Saved {summary_txt.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
