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
- Built the supervisor-meeting materials (Step 3). Architecture diagram drawn with
  matplotlib (`dissertation/presentation/make_architecture.py` -> `architecture.png`):
  a clean left-to-right flow of the six components, showing the hybrid detector
  (transformer plus stylometric) and the two question-generation backends, with a note
  that steps 2 to 6 run only on flagged (AI) work. Reflects the two-class scope.
- Built an 8-slide plain deck with `python-pptx` (`make_deck.py` -> the .pptx, gitignored
  as it will be hand-edited): title, problem and verification gap, architecture, a
  step-by-step walkthrough, dataset approach, the Verification Interview Guide output,
  evaluation (one slide, discrimination simulation primary and LLM-as-judge supplementary
  with cross-model agreement and objective-proxy validation, to answer the coordinator's
  flag), and status and next steps. Speaker notes written per slide and exported to
  `talk_track.md` to rehearse from.
- QA: verified slide text, order and notes; bullet glyph correct; zero em dashes; no
  placeholders; diagram fits the slide. Could not auto-render slide images (no LibreOffice
  installed), so layout was checked by geometry, not pixels.
- Read the Meeting 1 record. Confirmed next meeting Tuesday 9 June 12:00, and that the
  two-class change must show in the architecture and the Introduction (it does in the
  deck). Open items from Vini: HPC access and assessment format (viva vs recorded
  presentation); both are noted as asks on the status slide.
- Foundation phase is effectively complete (environment, data, sample, architecture),
  slightly ahead of the week 1 to 2 plan.
- Proposed the Introduction chapter section structure
  (`dissertation/introduction_outline.md`): background, problem statement, motivation,
  verification gap, research question and objectives, scope (including the two-class
  decision), contributions, and dissertation outline. Headings only, to show Vini before
  drafting. Full prose comes after Meeting 2, in my own voice.
- Started a LibreOffice install (winget) so slide decks can be auto-rendered for visual QA
  in future sessions.
- Set up authoring and dev tooling (requested). Installed Pandoc (markdown to Word and
  PDF, verified by exporting `sample_design.md` to .docx), Graphviz (`dot` for diagrams),
  `markitdown` and `python-pptx`, and a curated set of VS Code extensions (Jupyter, Ruff,
  Markdown All in One, markdownlint, Code Spell Checker, GitLens, Rainbow CSV, LaTeX
  Workshop). LibreOffice (installed earlier) is now used to render decks to images for
  visual QA, which caught and fixed a crowded box on the architecture slide. MiKTeX
  (LaTeX), ruff and jupyter were already present. Recorded the Python dev packages in
  `requirements-dev.txt` and the toolchain in the README. Pandoc and `dot` need a
  terminal restart to be on PATH.
- Produced six dataset rationale figures (`dissertation/figures/`) and an explainer
  (`dissertation/dataset_rationale.md`) tying each design choice to its evidence: group
  balance versus discipline imbalance (stratify by group), non-native availability and the
  AH binding constraint (oversampling to 80 per cell), essays-per-student clustering
  (student-level split and per-student cap), the balanced 8-cell sample by split, and
  native versus non-native length in the sample (length is not a proxy for first language).
  Generated by `make_dataset_figures.py`.
- Next: after Meeting 2, draft the Introduction section by section; then the generation
  pipeline and detector (HPC-gated).

## 2026-06-09

- Meeting 2 with Dr. Vijayan. Walked through the architecture, the detector (DeBERTa plus
  perplexity and burstiness), explainability (SHAP, Integrated Gradients, ablation for
  faithfulness), the two-backend question generation, Bloom's, the output guide, and the
  dataset work. Supervisor was positive and called the work strong.
- Supervisor feedback:
  (1) The dataset may be over-structured. Do not over-invest in manual balancing. The
      balanced 640 sample is fine for the initial supervised run. For a structured-versus
      -natural comparison, use two separately trained models, not one reused model.
  (2) Keep two classes (human vs AI) for now. A later phase could add a third "partial AI"
      class if two-class works, accepting the heavy overlap. Two classes stay the scope.
  (3) Any chart in the dissertation must have category percentages summing to 100 (the
      group chart rounded to 101).
  (4) Native vs non-native length difference is acceptable.
- Acted on immediately: corrected the group-balance figure to one decimal so the
  percentages sum to 100.0; drafted the Meeting 2 formal record in
  `dissertation/meetings/` and generated a Word copy with Pandoc for the OneDrive upload.
- New action items: upload materials and the record to OneDrive and email the supervisor;
  draft the Introduction at a high level only (no method detail yet, methods may change);
  search NLP/AI conferences (Ireland or Europe, hybrid) and prepare a 200 to 250 word
  abstract; prepare for both a presentation and a viva; clarify the "application form" the
  supervisor mentioned.
- Compute: HPC / test PC still pending; supervisor to check availability and add me as a
  user.
- To confirm with Mykhailo before changing locked scope or evaluation: whether to record
  the possible 3-class extension as future work, and whether to add the structured-versus
  -natural two-model comparison to the methodology. Not edited yet.
- Meeting 2 record finalised (Mykhailo confirmed the format) and uploaded. Applied the
  agreed changes to the source of truth: added the structured-versus-natural two-model
  comparison to the evaluation in `CLAUDE.md`, recorded the possible 3-class extension as
  future work (core scope stays two-class), and noted the over-structuring guidance in
  `sample_design.md`.
- Drafted a 230-word conference abstract (`dissertation/conference_abstract_draft.md`) and
  shortlisted venues: AICS (Irish Conference on AI and Cognitive Science, December, best
  fit) plus AIES and academic-integrity venues to verify. BEA 2026 already closed (23
  March). The abstract is a draft to rewrite in my own voice; all deadlines to be verified
  on official CFP pages.
- Next: draft the Introduction at a high level while awaiting compute; then begin the
  detection phase (download M4 / SemEval-2024 Task 8, set up the detector, get a first
  local result), and generate the AI essays once HPC is confirmed.

## 2026-06-14

- Prepared the repository for a clean handoff to a new session (switching to a fresh
  context window). Wrote `HANDOFF.md` as the entry point: current state, the active task
  (local AI-essay generation), the publication task, the working norms, a repository map
  labelling each path as supervisor-facing, private/local-only, or published, and a
  command cheat sheet. Pointed `CLAUDE.md` and `README.md` to it.
- Confirmed local generation is viable without HPC: Ollama 0.22 is installed with
  `llama3.1:8b` pulled, on the RTX 4060 (8 GB). Plan for the next session: build
  `src/generation/generate_ai_essays.py`, test on a few essays, then run the full 640 in
  the background (resumable, length-matched), checking in periodically to save tokens.
- Decision since HPC keeps slipping: generate the AI essays locally now rather than wait.
- Fable 5 was unavailable, so continued directly. Built the local AI-essay generator
  `src/generation/generate_ai_essays.py` (Ollama, llama3.1:8b). It matches each human
  essay on topic, keywords, and length: keywords are extracted from the human text to
  anchor the topic (this caught and fixed an off-topic case where a vague Dickens title
  produced an essay about compilers), and a continuation loop tops the essay up to the
  target length (ratio about 1.0). Resumable, with per-essay metadata.
- Validated on short and medium essays, then launched the full 640-essay run in the
  background. It is slow on the laptop GPU (tens of seconds to a few minutes per essay,
  so several hours, likely overnight), and resumable if interrupted. Outputs go to
  `data/processed/ai_essays/` (gitignored, BAWE-derived).
- Started the dissertation document. Drafted Chapter 1 Introduction at outline level
  (`dissertation/chapters/01_introduction.md`), as a rough first-person draft to rewrite
  in my own voice, per Vini's guidance to keep methods out until the model is built.
- Verified AICS as the lead conference venue (Irish AI conference, December, Springer
  proceedings; 2025 at DCU). The 2026 call is not yet announced, so recorded the pages to
  watch rather than inventing a deadline (`dissertation/conference_abstract_draft.md`).
- Drafted the literature review skeleton (`dissertation/chapters/02_literature_review.md`):
  section structure mapped to the pipeline, what each section must cover, and search
  targets, with no citations yet. Every reference will be found and verified (2021 to 2026)
  before it goes in.
- Started the detection-phase groundwork (non-GPU, so generation keeps running). Installed
  the missing `datasets` package and wrote `src/data/download_m4.py`, which pulls the M4 /
  SemEval-2024 Task 8 Subtask A monolingual set (English, binary human vs machine) from the
  d0rj Hugging Face mirror and saves train/dev/test to `data/raw/m4/` (gitignored).
  Provenance check passed: train is exactly 119,757 rows as documented. Dev is 5,000
  (50/50), test 34,272 across generators (GPT4, davinci, chatGPT, cohere, dolly, bloomz).
  Columns: text, label (0 human / 1 machine), model, source, domain. This is the detector
  pre-training corpus.
- Generation interrupted at 32/640 (both Ollama and the generator process had stopped,
  most likely the machine slept). Recovered with no loss: restarted `ollama serve`,
  disabled sleep on AC power (`powercfg /change standby-timeout-ac 0`), and resumed the
  run, which skipped the 32 finished essays and continued from 33. Confirmed it is
  generating again.
- Next: when generation finishes, run the human-vs-AI length cross-check on the AI essays;
  then build the detector (stylometric feature extractor, then fine-tune a transformer,
  pre-trained on M4). Finalise the abstract once the AICS 2026 call opens.
