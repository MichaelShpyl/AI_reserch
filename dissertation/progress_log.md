# Progress log

One short entry per working session. Feeds the methodology chapter and the weekly
supervisor notes. Newest entries at the bottom.

## 2026-06-07

- Set up the repository scaffolding from CLAUDE.md: folder tree, an `src` package
  with one subpackage per pipeline component, and `data/raw`, `data/interim`,
  `data/processed`, plus `config`, `notebooks`, `outputs`, `tests`, `dissertation`.
- Added `.gitignore` (Python ML), `requirements.txt` (pinned late-2024 baseline,
  reproducibility over latest), and `README.md` (project summary and local Windows
  GPU setup).
- Initialised a local-only git repository (no remote, kept private) and committed
  the scaffolding.
- Decisions recorded: keep stable version pins, do not chase latest; `bitsandbytes`
  is only needed for the Phase 4 Llama QLoRA fine-tuning, which will run on HPC, so
  any local install trouble is non-blocking for now.
- Next: local environment install and GPU verification (Step 2), then BAWE download
  and corpus summary (Step 3).

## 2026-06-08

- Downloaded BAWE from the Oxford Text Archive (resource 2539) and extracted it to
  `data/raw/bawe/`. The package holds plain text (`CORPUS_TXT`, 2,761 files), several
  XML and encoding variants, and `documentation/BAWE.xls` (the holdings metadata).
- Added `xlrd==2.0.1` to requirements: `BAWE.xls` is the legacy `.xls` format, which
  `openpyxl` cannot read. Installed `xlrd` and `openpyxl` into the working interpreter.
- Wrote `src/data/explore_bawe.py`: it loads `BAWE.xls`, resolves the columns we need,
  prints a full summary, validates the `words` column against a whitespace count of the
  text files, and saves a cleaned `data/interim/bawe_metadata.csv`, a text summary, and
  two length figures.
- Key numbers (2,761 texts): disciplinary groups are well balanced (SS 28.1%, AH 25.5%,
  LS 24.7%, PS 21.6%); levels are reasonably even across UG years 1 to 3 and taught
  masters; first language is 70.7% native English, 29.3% non-native (Chinese is the
  largest non-native group). Length is right-skewed: median 2,091 words, mean 2,357,
  range 302 to 13,160. Two thirds of essays sit between 1,000 and 3,000 words.
- Length cross-check: spreadsheet `words` vs text-file word count gave Pearson r = 0.986
  and a median file/sheet ratio of 0.995, so the `words` column is reliable ground truth
  for the later length-matching step.
- Data-quality notes for the sampling step: 32 distinct discipline labels (not 35);
  `Other`/`OTHER` are split by case; one label carries an HTML entity
  (`Cybernetics &amp; Electronic Engineering`); one missing discipline/group and nine
  missing levels. These need light cleaning before we sample.
- Cleaned the metadata into a sampling frame (`src/data/clean_bawe.py`): fixed HTML
  entities in discipline labels, merged the case-split `OTHER` into `Other`, and dropped
  one row missing group and word count, leaving 2,760 usable essays. The cleaning report
  prints the group x L1/L2 availability table and the per-student clustering.
- Key constraints found: non-native essays are unevenly spread (AH 114, LS 185, PS 169,
  SS 340), and the corpus is heavily clustered by author (627 students, up to 20 essays
  each, 86% with more than one). AH non-native is the scarce stratum (114 essays from
  just 29 students).
- Agreed the sampling design: stratify on disciplinary group x first-language status
  (8 cells), 80 per cell, 640 human essays, 50/50 native/non-native (oversampling
  non-native from the natural 29% for a detector-bias analysis), per-student cap of 4,
  and a 70/15/15 train/val/test split assigned at the student level so no writer spans
  splits. Built it in `src/data/build_sample.py` with a fixed seed (42).
- Sample checks passed: exactly 80 per cell, 320/320 native/non-native, zero students in
  more than one split, healthy discipline spread within each group, and native vs
  non-native length close (mean 2,368 vs 2,471 words). Saved a versioned manifest for
  reproducibility and wrote the design to `dissertation/sample_design.md`.
- Flagged (not fixed, by request): installed packages drift from the pins (numpy 2.3.4
  vs pinned 1.26.4, plus transformers/spaCy/matplotlib) and there is no `.venv`. No
  problem for data work now. Reconcile before the detection phase and before HPC by
  making a dedicated env and re-pinning to the actual working versions.
- Next: architecture diagram and slide deck for the supervisor meeting (Step 3), then
  start the Introduction chapter (Step 4). Generation pipeline is next week, HPC-gated.
