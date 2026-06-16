"""Assemble the labelled detection corpus from the human and AI essays.

Pairs each sampled human essay (label 0) with its matched AI essay (label 1), carrying
the student-level train/val/test split and the group / native metadata from the manifest.
Result: data/processed/detection_corpus.parquet, 1,280 rows (640 + 640).

    python src/detection/build_detection_corpus.py            # raw text
    python src/detection/build_detection_corpus.py --clean    # markup-stripped text

The --clean flag applies text_normalize.normalize_text to every essay, removing the BAWE
export tags from the human side and Llama's markdown from the AI side (see the audit,
2026-06-16). It writes a separate detection_corpus_clean.parquet so the raw and cleaned
detectors can be compared head to head.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from text_normalize import normalize_text

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "processed" / "bawe_human_sample_manifest.csv"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
AI_DIR = REPO / "data" / "processed" / "ai_essays"
OUT = REPO / "data" / "processed" / "detection_corpus.parquet"
OUT_CLEAN = REPO / "data" / "processed" / "detection_corpus_clean.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="strip BAWE/markdown markup and write the cleaned corpus")
    args = ap.parse_args()
    prep = normalize_text if args.clean else (lambda s: s.strip())
    out_path = OUT_CLEAN if args.clean else OUT

    man = pd.read_csv(MANIFEST)
    rows = []
    missing = 0
    for r in man.itertuples(index=False):
        rid = str(r.id)
        hp = CORPUS_TXT / f"{rid}.txt"
        ap_ = AI_DIR / f"{rid}.txt"
        if not (hp.exists() and ap_.exists()):
            missing += 1
            continue
        common = dict(id=rid, split=r.split, disciplinary_group=r.disciplinary_group,
                      native=bool(r.native), cell=r.cell)
        rows.append({**common, "source": "human", "label": 0,
                     "text": prep(hp.read_text(encoding="utf-8", errors="ignore"))})
        rows.append({**common, "source": "ai", "label": 1,
                     "text": prep(ap_.read_text(encoding="utf-8", errors="ignore"))})

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Detection corpus ({'clean' if args.clean else 'raw'}): {len(df)} rows  "
          f"(missing pairs: {missing})  -> {out_path.relative_to(REPO)}")
    print("\nBy split and label (0=human, 1=AI):")
    print(pd.crosstab(df["split"], df["label"]).to_string())
    print("\nBy split and source group:")
    print(pd.crosstab(df["split"], df["disciplinary_group"]).to_string())
    wl = df["text"].str.split().str.len()
    print(f"\nText length (words): min {wl.min()} median {int(wl.median())} mean {wl.mean():.0f} max {wl.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
