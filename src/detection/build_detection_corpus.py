"""Assemble the labelled detection corpus from the human and AI essays.

Pairs each sampled human essay (label 0) with its matched AI essay (label 1), carrying
the student-level train/val/test split and the group / native metadata from the manifest.
Result: data/processed/detection_corpus.parquet, 1,280 rows (640 + 640).

    python src/detection/build_detection_corpus.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "processed" / "bawe_human_sample_manifest.csv"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
AI_DIR = REPO / "data" / "processed" / "ai_essays"
OUT = REPO / "data" / "processed" / "detection_corpus.parquet"


def main() -> int:
    man = pd.read_csv(MANIFEST)
    rows = []
    missing = 0
    for r in man.itertuples(index=False):
        rid = str(r.id)
        hp = CORPUS_TXT / f"{rid}.txt"
        ap = AI_DIR / f"{rid}.txt"
        if not (hp.exists() and ap.exists()):
            missing += 1
            continue
        common = dict(id=rid, split=r.split, disciplinary_group=r.disciplinary_group,
                      native=bool(r.native), cell=r.cell)
        rows.append({**common, "source": "human", "label": 0,
                     "text": hp.read_text(encoding="utf-8", errors="ignore").strip()})
        rows.append({**common, "source": "ai", "label": 1,
                     "text": ap.read_text(encoding="utf-8", errors="ignore").strip()})

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"Detection corpus: {len(df)} rows  (missing pairs: {missing})  -> {OUT.relative_to(REPO)}")
    print("\nBy split and label (0=human, 1=AI):")
    print(pd.crosstab(df["split"], df["label"]).to_string())
    print("\nBy split and source group:")
    print(pd.crosstab(df["split"], df["disciplinary_group"]).to_string())
    wl = df["text"].str.split().str.len()
    print(f"\nText length (words): min {wl.min()} median {int(wl.median())} mean {wl.mean():.0f} max {wl.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
