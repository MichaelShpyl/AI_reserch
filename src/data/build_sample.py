"""Draw the stratified human sample from the cleaned BAWE frame.

Design (approved 2026-06-08):
  - Stratify on disciplinary group x first-language status: 8 cells.
  - 80 essays per cell, even across the four groups, 50/50 native / non-native.
    Total 640 human essays (320 native, 320 non-native).
  - Per-student cap of 4 within a cell, to stop one writer's style dominating.
  - Train / val / test split assigned at the student level (70 / 15 / 15) so no
    writer appears in two splits. Some students span groups, so assignment is
    global and cell-aware, not per cell.
  - Fixed seed (42), recorded, so the draw is reproducible.

Why this shape: groups are balanced in BAWE while disciplines are not, so we
stratify by group. Non-native writers are oversampled from their natural 29% up
to 50% so we can measure detector bias against them. Length differences between
groups are handled by group balance now and by per-essay length matching when we
generate the AI counterparts.

Outputs:
  - data/processed/bawe_human_sample.csv            full metadata + cell + split
  - data/processed/bawe_human_sample_manifest.csv   id + split only (versioned)
  - outputs/bawe_sample_summary.txt                 the printed report
  - outputs/bawe_sample_length_by_cell.png          length balance check

Run from the repo root (after clean_bawe.py):
    python src/data/build_sample.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "data" / "interim" / "bawe_clean.csv"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"

GROUPS = ["AH", "LS", "PS", "SS"]
GROUP_LABELS = {
    "AH": "Arts and Humanities", "LS": "Life Sciences",
    "PS": "Physical Sciences", "SS": "Social Sciences",
}
STATUSES = ["native", "non-native"]

SEED = 42
CELL_SIZE = 80
STUDENT_CAP = 4
TEST_FRAC = 0.15
VAL_FRAC = 0.15


class Tee:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def draw_sample(df: pd.DataFrame, rs: np.random.RandomState, out: Tee) -> pd.DataFrame:
    """Cap per student within a cell, then draw CELL_SIZE essays per cell."""
    parts = []
    out("Drawing cells (target {} each, per-student cap {}):".format(CELL_SIZE, STUDENT_CAP))
    for g in GROUPS:
        for s in STATUSES:
            cell = df[(df["disciplinary_group"] == g) & (df["status"] == s)]
            # Random shuffle, then keep at most CAP rows per student.
            shuffled = cell.sample(frac=1.0, random_state=rs)
            capped = shuffled.groupby("student_id", sort=False, group_keys=False).head(STUDENT_CAP)
            if len(capped) < CELL_SIZE:
                raise ValueError(
                    f"Cell {g}_{s} has only {len(capped)} essays after the cap, "
                    f"need {CELL_SIZE}. Lower CELL_SIZE or raise STUDENT_CAP."
                )
            drawn = capped.sample(n=CELL_SIZE, random_state=rs).copy()
            drawn["cell"] = f"{g}_{s}"
            parts.append(drawn)
            out(f"    {g}_{s:<11} pool(capped)={len(capped):>4}  drawn={len(drawn)}  "
                f"students={drawn['student_id'].nunique()}")
    return pd.concat(parts, ignore_index=True)


def assign_splits(sample: pd.DataFrame, rs: np.random.RandomState) -> dict:
    """Assign each student to one split (global, leakage-safe, cell-aware)."""
    target_test = round(CELL_SIZE * TEST_FRAC)
    target_val = round(CELL_SIZE * VAL_FRAC)
    student_split: dict = {}

    for g in GROUPS:
        for s in STATUSES:
            cell = sample[sample["cell"] == f"{g}_{s}"]
            counts = {"train": 0, "val": 0, "test": 0}
            per_student = cell.groupby("student_id").size()
            # Tally students already assigned in earlier cells.
            unassigned = []
            students = rs.permutation(per_student.index.to_numpy())
            for sid in students:
                n = int(per_student[sid])
                if sid in student_split:
                    counts[student_split[sid]] += n
                else:
                    unassigned.append((sid, n))
            # Fill test, then val, then train, to approximate the target shares.
            for sid, n in unassigned:
                if counts["test"] < target_test:
                    student_split[sid] = "test"
                elif counts["val"] < target_val:
                    student_split[sid] = "val"
                else:
                    student_split[sid] = "train"
                counts[student_split[sid]] += n
    return student_split


def report(sample: pd.DataFrame, out: Tee) -> None:
    total = len(sample)
    out("")
    out("=" * 70)
    out("SAMPLE REPORT")
    out("=" * 70)
    out(f"Total essays: {total}")
    out(f"Native: {int((sample['native']).sum())}   "
        f"Non-native: {int((~sample['native']).sum())}")
    out(f"Distinct students: {sample['student_id'].nunique()}")
    out("")

    # Per-cell counts and split breakdown.
    out("Per-cell counts and split breakdown:")
    ct = pd.crosstab(sample["cell"], sample["split"])
    ct = ct[[c for c in ["train", "val", "test"] if c in ct.columns]]
    ct["total"] = ct.sum(axis=1)
    out(ct.to_string())
    out("")

    # Overall split sizes.
    out("Overall split sizes:")
    for split in ["train", "val", "test"]:
        n = int((sample["split"] == split).sum())
        out(f"    {split:<6} {n:>4}  ({100.0 * n / total:4.1f}%)")
    out("")

    # Leakage check.
    spanning = sample.groupby("student_id")["split"].nunique()
    n_span = int((spanning > 1).sum())
    out(f"Students appearing in more than one split (must be 0): {n_span}")
    out("")

    # Per-student load.
    sizes = sample.groupby("student_id").size()
    out(f"Essays per student in sample: mean {sizes.mean():.2f}, max {int(sizes.max())}")
    out("")

    # Discipline spread within each group (confirm no degenerate concentration).
    out("Discipline spread within each group (top 5):")
    for g in GROUPS:
        sub = sample[sample["disciplinary_group"] == g]
        top = sub["discipline"].value_counts().head(5)
        out(f"    {g} ({GROUP_LABELS[g]}), {sub['discipline'].nunique()} disciplines:")
        for disc, n in top.items():
            out(f"        {disc:<28} {n}")
    out("")

    # Academic level spread.
    if "level_label" in sample:
        out("Academic level spread:")
        for lvl, n in sample["level_label"].value_counts().items():
            out(f"    {str(lvl):<20} {n:>4}  ({100.0 * n / total:4.1f}%)")
        out("")

    # Length confound check: native vs non-native, and by group.
    out("Length (words) by first-language status, the confound to watch:")
    for s in STATUSES:
        w = sample.loc[sample["status"] == s, "words"]
        out(f"    {s:<11} mean {w.mean():6.0f}   median {w.median():6.0f}")
    out("    By group (mean words, native / non-native):")
    for g in GROUPS:
        wn = sample.loc[(sample["disciplinary_group"] == g) & (sample["native"]), "words"].mean()
        wnn = sample.loc[(sample["disciplinary_group"] == g) & (~sample["native"]), "words"].mean()
        out(f"        {g}: {wn:6.0f} / {wnn:6.0f}")
    out("    Note: human vs AI is not confounded by length, because each AI essay")
    out("    is generated to match its human source length. Any native vs non-native")
    out("    length gap here is a property of the human data and is reported, not hidden.")
    out("")


def save_length_figure(sample: pd.DataFrame, outdir: Path, out: Tee) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        out(f"    (figure skipped: {exc})")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    cells = [f"{g}_{s}" for g in GROUPS for s in STATUSES]
    data = [sample.loc[sample["cell"] == c, "words"] for c in cells]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=cells, showfliers=False)
    ax.set_ylabel("Essay length (words)")
    ax.set_title("Sampled essay length by cell (group x first-language status)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = outdir / "bawe_sample_length_by_cell.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    out(f"    Saved {path.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the stratified BAWE human sample.")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    if not args.in_path.exists():
        print(f"ERROR: input not found: {args.in_path}. Run clean_bawe.py first.")
        return 1

    np.random.seed(SEED)
    rs = np.random.RandomState(SEED)

    df = pd.read_csv(args.in_path)
    df["native"] = df["native"].astype(bool)
    df["status"] = df["native"].map({True: "native", False: "non-native"})

    out = Tee()
    out("=" * 70)
    out("BAWE STRATIFIED SAMPLE")
    out("=" * 70)
    out(f"Seed {SEED}, cell size {CELL_SIZE}, student cap {STUDENT_CAP}, "
        f"split {int((1 - TEST_FRAC - VAL_FRAC) * 100)}/{int(VAL_FRAC * 100)}/{int(TEST_FRAC * 100)}")
    out("")

    sample = draw_sample(df, rs, out)
    student_split = assign_splits(sample, rs)
    sample["split"] = sample["student_id"].map(student_split)

    report(sample, out)

    # Save artefacts.
    args.processed.mkdir(parents=True, exist_ok=True)
    full_path = args.processed / "bawe_human_sample.csv"
    sample.to_csv(full_path, index=False, encoding="utf-8")
    out(f"Saved {full_path.relative_to(REPO_ROOT)}  ({len(sample)} rows)")

    manifest = sample[["id", "student_id", "disciplinary_group", "native", "cell", "split"]]
    manifest_path = args.processed / "bawe_human_sample_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    out(f"Saved {manifest_path.relative_to(REPO_ROOT)}  (versioned manifest)")

    if not args.no_figure:
        save_length_figure(sample, args.outputs, out)

    summary_path = args.outputs / "bawe_sample_summary.txt"
    out.save(summary_path)
    print(f"Saved {summary_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
