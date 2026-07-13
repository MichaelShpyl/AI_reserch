# Progress log

One short entry per working session. Feeds the methodology chapter and the weekly
supervisor notes. Newest entries at the bottom.

## 2026-06-07

- Set up the repository scaffolding from my project plan: folder tree, an `src` package
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
- QA: verified slide text, order and notes; bullet glyph correct; no placeholders; diagram
  fits the slide. Could not auto-render slide images (no LibreOffice
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
  comparison to the evaluation plan in my scope notes, recorded the possible 3-class extension as
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

- Prepared the repository so I can pick up cleanly after a break. Wrote `HANDOFF.md` as the
  entry point: current state, the active task
  (local AI-essay generation), the publication task, the working norms, a repository map
  labelling each path as supervisor-facing, private/local-only, or published, and a
  command cheat sheet. Pointed `README.md` to it.
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
- Built the stylometric feature extractor `src/detection/stylometric.py` (CPU, spaCy):
  sentence-length burstiness and variance, type-token ratio and root TTR, hapax ratio,
  mean word length, punctuation and POS-tag proportions, plus a GPT-2 perplexity function
  to wire up when the GPU is free. Validated on 35 human/AI pairs with a clear early signal
  in the expected direction: human text varies sentence length more (std 12.2 vs 8.6) and
  has higher vocabulary diversity (root TTR 15.2 vs 9.9, hapax 0.21 vs 0.10), while AI uses
  longer words and more nouns. Lengths are matched (~1.0), so these are style differences,
  not length artifacts. Encouraging for the hybrid detector and a good check that the
  generated essays have a distinct AI profile.
- Generation stalled a second time at 98/640 (same mode: Ollama and the generator both
  gone), so the cause is likely the machine sleeping, and possibly harness background-task
  teardown. Hardened it: added a keep-awake call to the generator (Windows
  SetThreadExecutionState, so the system will not idle-sleep while it runs), disabled sleep
  and hibernate on both AC and DC, restarted Ollama, and resumed (skips the 98 done). A
  periodic monitor restarts it if it dies again. (A detached Start-Process launch was
  tried and exited instantly, so the proven Bash background launch is used.)
- Generation hardening confirmed: launching the generator as a detached process
  (Start-Process, not a Bash background run) survives the harness teardown that was killing
  it. It ran unattended from 142 to 227 and is logging to `outputs/gen_out.log`. ETA ~8.6h.
- Ran a literature-review research workflow (8 sections, gather then adversarial verify) to
  find 2021 to 2026 sources. The scripted pass did not finish, so verification is incomplete:
  15 candidate titles were gathered (detection, explainability, evaluation) and saved as
  UNVERIFIED leads in `dissertation/litreview_sources.md`; five sections returned nothing.
  I will finish the verification by hand. Lesson: keep the automated research passes small
  and always verify the sources myself.
- Researched publication and application options for the next meeting (verified via web),
  compiled in `dissertation/publication_options.md`. Key finding: several ideal venues have
  closed for 2026 (ENAI/ECEIA abstract 9 Feb, BEA 23 Mar, ICNLSP 30 May). Live targets:
  the GenAIDetect workshop (machine-generated-text-detection shared task, likely with
  EMNLP 2026 Budapest or AACL 2026 Zhuhai, both hybrid) and AICS 2026 (Ireland, December,
  Springer; call not yet out). Immediate step: an arXiv preprint. Journals (slower):
  International Journal for Educational Integrity and Computers and Education: AI.
- Generation healthy and unattended at 257/640 (detached), ratio ~1.05, ETA ~7.9h.
- Meeting-3 readiness pass. Built a 5-slide progress-update deck for tomorrow
  (`dissertation/presentation/Meeting3_progress_update.pptx`, source `make_meeting3_deck.py`,
  notes in `talk_track_meeting3.md`): what is done since Meeting 2, the dataset with the
  length-match point and the sample-composition figure, the early stylometric result
  (human vs AI separation with lengths matched), and next steps plus publication and asks.
  Rendered and visually QA'd. Gitignored the private recordings folder.
- Readiness summary: dataset generating (around 295/640, on track), M4 ready, stylometric
  extractor done, scope updates applied, dataset figures done and the chart fixed,
  publication options researched, Introduction drafted and literature review structured.
  Carryover for me: upload the Meeting 2 record to OneDrive and email Vini.
- DATASET COMPLETE. All 640 matched AI essays generated locally (0 failures), so the
  detection corpus is 1,280 essays (640 human + 640 AI, two classes). Validated with
  `src/generation/check_ai_corpus.py`:
  - Length matched: Pearson r = 0.978 between AI and human source lengths, mean ratio 1.047,
    99.2% of AI essays within +/-20% of their source, length-distribution overlap 91.4%
    (figure `dissertation/figures/fig7_ai_vs_human_length.png`).
  - Topic matched: keyword spot-check shows the shared terms are the topic-defining ones
    (for example Merkel/Schroeder, phonological-loop/memory, china/tribute), so each AI
    essay tackled the same question without copying.
  - Combined with the earlier stylometric check (human more varied with lengths matched),
    the corpus is valid: human and AI differ in style, not length or topic.
- Built the labelled detection corpus (`src/detection/build_detection_corpus.py`):
  `data/processed/detection_corpus.parquet`, 1,280 rows, balanced (train 438/438, val
  102/102, test 100/100) on the student-level splits, zero missing pairs.
- Launched the first detector training (`src/detection/train_detector.py`): DeBERTa-v3-base
  fine-tune as the transformer baseline, reporting accuracy/precision/recall/F1, the
  confusion matrix, and a native-vs-non-native human false-positive-rate breakdown (the
  fairness seed). Running detached on the now-free GPU (Ollama stopped to free VRAM);
  result pending in `outputs/detector_metrics.json`.
- First detector trained: DeBERTa-v3-base on the 1,280-essay corpus. Test F1 = 1.00
  (confusion matrix 100/0/0/100), val F1 0.995, and the human false-positive rate is 0.0
  for both native and non-native writers. Metrics in `outputs/detector_metrics.json`.
- Treated the perfect score critically, not as a win. Checked for a trivial artifact:
  lengths are matched (so not length), and a citation count showed the AI essays actually
  have MORE in-text citations than human (mean 21.6 vs 13.0), so citation density is not
  the giveaway either. Conclusion: near-perfect is the expected result for in-domain,
  single-generator detection (a known-easy setting in the literature). The real, defensible
  work is robustness (M4 cross-generator, paraphrase) and the explainability layer, not the
  raw score. The deck frames the 100% honestly.
- Built a visual-first Meeting 3 deck for a quick presentation
  (`dissertation/presentation/Meeting3_visual.pptx`, source `make_meeting3_visual_deck.py`,
  notes `talk_track_meeting3_visual.md`): one graph per slide (architecture, balanced
  dataset, AI-vs-human length match, human-vs-AI writing style, detector confusion matrix,
  summary). New result figures via `make_result_figures.py` (stylometry comparison,
  confusion matrix).
- Next: a RoBERTa comparison run, the stylometric fusion, the M4 cross-generator
  robustness test (the honest harder case), and the explainability layer. Ready the
  abstract for GenAIDetect/AICS.

## 2026-06-16

- Ran the RoBERTa-base comparison detector (same corpus, same student-level splits,
  same training settings) as a second transformer baseline alongside DeBERTa-v3-base.
  Both land at test F1 = 1.00 with a clean 100/0/0/100 confusion matrix and a 0.0
  human false-positive rate for native and non-native writers alike. On validation
  DeBERTa is marginally ahead (val F1 0.9951, eval loss 0.025) of RoBERTa (val F1
  0.9903, eval loss 0.083). Metrics saved separately: `outputs/detector_metrics_deberta.json`
  and `outputs/detector_metrics_roberta.json`.
- Reading: two independent architectures agreeing at ceiling is further evidence that
  in-domain single-generator detection is genuinely easy on this corpus, not a quirk of
  one model. It is not new headroom to chase. DeBERTa stays the primary detector on the
  slightly better validation result; RoBERTa is the documented comparison. The defensible
  contribution remains robustness (M4 cross-generator, paraphrase) and explainability,
  which is where effort goes next once the design choices are agreed.

### Detector audit: why the score is perfect, and the artifact we found and fixed

- Ran a full adversarial audit of the 100% result rather than trusting it
  (`src/detection/audit_detector.py`, report in `outputs/audit_report.json`). The goal
  was to rule out a bug or a trivial shortcut before showing the result to anyone. Four
  checks:
  1. **Split integrity (clean).** 394 students, none in more than one split, every
     human/AI pair in the same split, and zero exact-duplicate texts crossing train and
     test. So the perfect test score is not train/test contamination.
  2. **Markup artifact (found, and it was large).** The human BAWE plain-text export keeps
     structural tags (`<heading>`, `<fnote>`, `<list>`, `<figure>`, `<quote>`, `<table>`).
     88.3% of the sampled human essays carry at least one; only 3.0% of the AI essays do
     (a few stray angle brackets). A brain-dead rule, "has a tag therefore human", scores
     **92.5%** on the test set by itself. Mirror problem on the AI side: Llama sometimes
     emits markdown (`**bold**`, `## headings`, `* bullets`, numbered lists) that human
     plain text never has. Both are corpus/format artifacts, not writing style, and they
     sit at the start of the text inside the model's 512-token window.
  3. **The real signal survives cleaning, and it is style not topic.** Built a normaliser
     (`src/detection/text_normalize.py`) that strips the BAWE tags from the human side and
     the markdown from the AI side and flattens layout, then a cleaned corpus
     (`detection_corpus_clean.parquet`, `build_detection_corpus.py --clean`). On the
     cleaned text a simple TF-IDF + logistic-regression model still reaches **100%** on
     the test set, and a model restricted to **function words only** (the, and, therefore,
     because; no content words, so no topic information) still reaches **99.5%**. That can
     only be writing style.
  4. **Interpretable fingerprint.** The cleaned linear model's top AI-leaning terms are the
     familiar LLM register ("in conclusion", "nuanced", "essential", "highlights",
     "insights", "complex", "significant"); the top human-leaning terms are blunter
     connectives ("therefore", "because", "so", "thus", "very"). Figures
     `dissertation/figures/fig_audit_separability.png` (what each signal alone achieves)
     and `fig_audit_top_features.png` (the words the cleaned detector keys on).
- Conclusion for the write-up and the meeting: the raw 100% was inflated by a corpus-markup
  artifact, which we identified, measured (a tag-only rule gets 92.5%), and removed. After
  removing it the classes remain almost perfectly separable on style alone, which is the
  expected, literature-consistent result for one-generator, one-domain detection. The honest
  headline detector is therefore reported on the **cleaned** corpus, with the raw number kept
  as the "before" to show the artifact's size. This does not weaken the project: the research
  contribution was never the in-domain score, it is explainability and robustness to harder
  settings (other generators, paraphrase, mixed human/AI), where the score will and should
  drop. Retrained DeBERTa on the cleaned corpus for the honest headline
  (`outputs/detector_metrics_clean.json`).
- Cleaned-corpus DeBERTa result (the number to present): test accuracy 0.99, precision
  0.980, recall 1.00, **F1 0.990**, confusion matrix [[98, 2], [0, 100]] (2 human essays
  flagged as AI, no AI missed). Native-writer false-positive rate 0.04, non-native 0.00.
  So removing the markup artifact moved the detector off the ceiling (1.000 -> 0.990), which
  is the honest picture: a real, slightly imperfect classifier with a genuine confusion
  matrix and a first fairness signal to discuss. The full-document linear probe still scores
  1.00 on clean text, while DeBERTa (which only reads the first 512 tokens) makes 2 errors,
  so the transformer is working from the essay openings while the signal is spread through
  the whole document.
- Updated the figures and the visual deck to this audited story:
  `fig_audit_separability.png` and `fig_audit_top_features.png` are new; the confusion figure
  and the stylometry figure now use cleaned text and the cleaned metrics; the deck gained two
  audit slides ("I stress-tested the perfect score", "What it actually keys on"). The deck
  was open in PowerPoint, so the rebuilt version is saved as `Meeting3_visual_audited.pptx`
  (close the old one and present this).
- Re-ran the RoBERTa comparison on the cleaned corpus too, so both architectures are
  reported post-fix. RoBERTa clean: test accuracy 0.995, F1 0.995, confusion [[99, 1],
  [0, 100]], native false-positive rate 0.02, non-native 0.00 (`detector_metrics_clean_roberta.json`).
  So on cleaned text DeBERTa is F1 0.990 and RoBERTa F1 0.995; both fall off the 1.000 ceiling
  once the markup shortcut is gone, and both make their few errors the same way (a handful of
  human essays flagged as AI, no AI essay missed). The cross-model agreement is reassuring:
  the residual signal is a real, architecture-independent style difference, not a quirk of one
  model. DeBERTa stays the primary detector (marginally better validation F1, 0.9855 vs 0.9808).
- Process note: this is exactly the kind of result that must be stress-tested before a viva.
  Finding the markup leak now (and being able to show the audit that caught it) is stronger
  evidence of rigour than a clean-looking 100% would have been.

### Why the cleaned score is still ~99% (the explainable answer)

- Pushed past "it is style" to a concrete, defensible explanation
  (`src/detection/why_high.py`, report `outputs/why_high.json`). The cleaned score stays
  high because the task as built is the easy end of the problem and several independent
  tells stack up:
  - **Single generator, single domain.** The "AI" class is really "Llama 3.1 with my
    prompt", so it is stylistically uniform. In function-word style space the AI essays are
    more self-similar (mean pairwise cosine 0.596) than the human essays (0.558), i.e. a
    tighter cluster, which is easy to separate from a varied class of different students.
    Detecting one known model in one subject area is the easy case in the literature.
  - **Lexical register.** AI overuses "in conclusion", "nuanced", "essential", "highlights",
    "insights"; humans use blunter connectives ("therefore", "because", "so").
  - **Locale spelling (a real but shallow tell).** Humans lean British (2.25 per 1k words),
    Llama leans American (2.35 per 1k); humans 0.77 American, AI 1.38 British. So it is not
    absolute, and it would shrink if the generator were prompted to write British English.
    Worth noting as both a contributor and a limitation.
  - **Formality.** Humans use about three times more contractions (0.83 vs 0.28 per 1k).
  - When this many signals all point the same way, ~99% is the expected result, not a bug.
- Visual: `dissertation/figures/fig_why_style_clusters.png` projects every essay into 2D
  function-word style space (no topic words) and shows two cleanly separated clouds. This is
  the most intuitive "why it is easy" picture and went into the deck as a new slide.
- Error sense-check: the human test essay the style model rates most AI-like is 3012c
  (non-native) at P(AI) 0.43, still correctly classified human; the next few are a small mix
  of native and non-native. No systematic bias against non-native writers in the closest calls.
- Honest framing for the meeting and the write-up: the in-domain number is high because the
  problem is easy by construction. The contribution is the explanation behind each decision
  plus robustness to the harder settings where the score will drop: other generators (M4),
  paraphrased and "humanised" AI text, mixed human/AI documents, and British-English-prompted
  AI (which removes the locale tell).
- Deck finalised to a single file: regenerated the canonical
  `dissertation/presentation/Meeting3_visual.pptx` (10 visual slides) and deleted the
  temporary `Meeting3_visual_audited.pptx`.

### Meeting 3 held (16 June 2026): outcome and decisions

- Presented the visual deck and the detector audit. The supervisor was satisfied with the
  approach of stress-testing the perfect score and the honest framing. Formal record in
  `dissertation/meetings/Meeting_Record_3_Mykhailo_Shpyl.md` (and `.docx`).
- Confirmed points and decisions from the meeting:
  - **AICS is running this year and is in September** (supervisor received the email the day
    before), so the proposal window is near. Corrected `publication_options.md` from the
    earlier "December" guess. GenAIDetect stays a second target.
  - The American-vs-British English locale tell I found is recognised as something to fix:
    the generation should match the expected submission language (British English for ATU),
    so the detector does not learn locale instead of style.
  - Paraphrased / partially human-edited AI text (AI written, human tweaked) is the
    partial-AI category from before and stays later-phase work, but the dataset will need to
    account for it because it would otherwise distort predictions. Multi-generator data
    (GPT, Claude, DeepSeek) is the other dataset priority; with one generator the task is
    easy by construction.
  - Useful context raised: many universities and journals now accept AI-assisted
    modifications, which supports the project framing (verify understanding, not ban AI use).
  - Timeline reassurance: about two and a half months left, keep submitting work as drafted.
- Next-week plan agreed: submit the drafted chapters now plus the presentation as a separate
  file; write the implementation chapters (dataset build and generation, detector, audit);
  research the AICS application/proposal; improve multi-generator detection and dataset
  consistency; then start the explainability work as the next main component.
- Next meeting: Tuesday 23 June 2026 (online).

### Dissertation document assembled (progress draft to send the supervisor)

- Built a single ATU-style dissertation document so the chapters can be sent to the supervisor
  for review: `dissertation/Dissertation_Shpyl_progress_draft.docx`. After first drafting it from
  last year's final year project, the official `Dissertation Template 2026.docx` was provided, so
  the document was rebuilt to conform to it: the official two title pages, the official ATU Donegal
  Declaration text (award confirmed as "Master of Science in Artificial Intelligence and Big Data
  Analytics"), Acknowledgements, Abstract, an Acronyms table, the Table of Contents, a Table of
  Figures, then the chapters. Body is 12pt, 1.5 line spacing, justified, with page numbers at the
  bottom right, per the template's stated rules.
- Generator lives in `dissertation/docgen/` (`build_dissertation.js`, Node + the docx library),
  so it is reproducible: it parses the markdown chapters, joins the hard-wrapped lines into
  proper paragraphs, applies heading styles, and embeds the detection figures with captions.
  The output `.docx` and `node_modules` are gitignored; the markdown chapters and the script are
  the source of truth. To regenerate: `cd dissertation/docgen && npm install && node build_dissertation.js`.
- Content included: Chapter 1 (Introduction, full prose), Chapter 2 (Literature Review, included
  honestly as a planned structure and reading plan with a visible note, since it has no verified
  citations yet), Chapter 3 (AI Text Detection, full prose with Figures 3.1 to 3.4: the
  separability bars, the top-feature words, the cleaned confusion matrix, and the style clusters).
  Added the four figure callouts into `chapters/03_detection.md` itself.
- The document is marked as a working draft for supervisor review. The chapter prose is still a
  rough first draft and will get a full polish before the final assessed submission. The TOC is a
  Word field; open in Word and press F9 to populate page numbers.
- Matched the document to the official template's typography after confirming it: the template
  uses Calibri body and Office-blue headings at 12pt with 1.5 line spacing and justified text,
  not Arial/teal. Changed the generator's fonts and heading colour to match. Committed the
  official `Dissertation Template 2026.docx` to the repo as the format reference.

### Compute decision: laptop only, no HPC (Meeting 3 follow-up)

- The supervisor confirmed ATU HPC / cloud will NOT be available. The whole project now runs on
  the laptop only (RTX 4060, 8 GB VRAM). There is no "scale up on HPC later" step, so every GPU
  component must fit 8 GB. Updated my scope and handover notes to match.
- Feasibility on 8 GB: detector fine-tuning (DeBERTa/RoBERTa base) already runs there; AI essay
  generation is done locally via Ollama; multi-generator data (GPT, Claude, DeepSeek) is API-based
  so needs API access and budget, not local GPU; explainability, argument mining, the Bloom's
  BERT-base classifier, and evaluation all fit.
- Risk flagged: the locked-scope Backend B is "Llama 3 8B fine-tuned with QLoRA". QLoRA of a full
  8B model on 8 GB is possible but very tight and slow (4-bit, batch 1, short sequences, gradient
  checkpointing, paged optimiser). Likely practical fallback: fine-tune a smaller open model
  (Llama 3.2 3B, Qwen2.5 3B, or Phi-3-mini) so it fits comfortably, keeping the commercial-vs-local
  comparison. This changes part of the locked scope, so it needs supervisor sign-off before any
  switch. Also, `bitsandbytes` on Windows is now on the critical path (QLoRA runs locally), not
  deferred.

- To regenerate `figdims.json` if the figures change:
  `python -c "import json;from pathlib import Path;from PIL import Image;figs=['fig_audit_separability','fig_audit_top_features','fig_detector_confusion','fig_why_style_clusters'];Path('dissertation/docgen/figdims.json').write_text(json.dumps({f:{'w':Image.open(Path('dissertation/figures')/(f+'.png')).size[0],'h':Image.open(Path('dissertation/figures')/(f+'.png')).size[1]} for f in figs},indent=2))"`

### Implementation chapter and the explainability layer (16 June 2026)

A long solo working day, kept to safe in-scope tasks only (no scope changes, nothing that
needs supervisor sign-off, laptop only).

- **Chapter 4 (Implementation)** drafted (`chapters/04_implementation.md`): the engineering story
  of what is built, environment and reproducibility on the 8 GB laptop, the BAWE data pipeline,
  the matched AI generation, the labelled corpus, the detector, and the audit tooling.
- **Explainability layer (locked-scope component 2)** implemented as a first pass
  (`src/explainability/integrated_gradients.py`, Captum). Retrained and saved the cleaned-corpus
  DeBERTa detector (F1 0.990) so the model was available, then:
  - Integrated Gradients token attributions, attributed to the AI class for a consistent sign
    (`fig_explain_ig_tokens.png`). The strongest tokens are punctuation and common words and
    fragments, not topic words, which agrees with the audit: the model reads style, not content.
  - A faithfulness test by ablation, swept over k (`fig_explain_faithfulness.png`, report
    `outputs/explainability.json`). Honest finding: removing the top attributed tokens lowers
    confidence only slightly more than removing random tokens (comprehensiveness ratio about 1.44
    at k=34), and keeping only the top tokens collapses the prediction to chance (sufficiency at
    ~0.50). So the signal is **diffuse**, spread across the whole essay, and a per-word heatmap is a
    weak standalone explanation for this detector. The faithful, lecturer-facing explanation is the
    feature-level / lexical-fingerprint view from the audit (Figures 3.2 and 3.4), plus SHAP on the
    stylometric features once the hybrid model exists.
- **Chapter 5 (Explainability)** drafted (`chapters/05_explainability.md`) reporting the above,
  including the negative faithfulness result, which is a genuine contribution: it tells us which
  explanation to trust.
- Wired Chapters 4 and 5 (with figures) into the dissertation document; it is now 28 pages and
  validates. Regenerated `dissertation/Dissertation_Shpyl_progress_draft.docx`.
- These chapters are still rough first drafts and the prose will get a full polish before any
  assessed submission. Nothing here changes the locked scope.

### M4 robustness evaluation (16 June 2026): the honest harder case

Ran the zero-shot transfer test the project needed (`src/detection/eval_m4_transfer.py`): apply the
in-domain DeBERTa detector, with no adaptation, to the M4 / SemEval-2024 Task 8 benchmark, which has
many generators and domains the detector never saw. Split into two honest tests.

- **Cross-generator, essays (M4 OUTFOX split).** F1 0.97. The detector caught all six unseen
  generators (GPT-4, ChatGPT, Cohere, BLOOMz, Dolly, davinci) at 96 to 100 percent, with a 5 percent
  human false-positive rate. The AI-style fingerprint is largely **generator-agnostic** on essays:
  a Llama-only detector still recognises text from quite different models. This corrected my prior
  expectation that cross-generator transfer would be weak.
- **Cross-domain (M4 reddit/wikihow/arxiv/wikipedia/peerread).** F1 falls to 0.79, and the failure is
  one-sided and important. The detector still catches machine text (86 to 98 percent) but **wrongly
  flags genuine human text as AI**: 79 percent on arXiv abstracts, about 40 percent on Wikipedia and
  WikiHow, 30 percent on peer-review, 23 percent on Reddit. It learned "human = student essay" and so
  misjudges more formal human writing. This is a **false-accusation** failure, exactly the harm the
  project exists to prevent, and it ties straight to the fairness concern (writers whose style differs
  from the training norm, including non-native writers, are most at risk).
- Reading: the detector is **robust across AI models but fragile across domains**, and the fragility
  hurts humans, not machines. The honest headline is that it spots AI essays from many models well, and
  is not safe to point at human writing from a domain it was not trained on. This is a lower bound
  (zero-shot, no adaptation); training on diverse human text and several generators should help, and is
  the next experiment.
- Figures: `fig_m4_per_generator.png`, `fig_m4_transfer_gap.png`, `fig_m4_per_domain.png`. Report:
  `outputs/m4_transfer.json`. Written up as Chapter 6 (`chapters/06_robustness.md`) and folded into the
  dissertation (now 31 pages) and the AICS paper draft (M4 result added; the earlier "will fall on
  other generators" claim corrected to "robust across generators, fragile across domains").
- Build note: the dissertation `.docx` was open in Word during the run (locked), so the canonical file
  was not overwritten. The builder now accepts a `DISS_OUT` override; to refresh the canonical file,
  close Word and run `node dissertation/docgen/build_dissertation.js`.

### Stylometric detector and SHAP explanation (16 June 2026)

Completed the feature-level explanation Chapter 5 promised, and built the stylometric half of the
hybrid detector (`src/explainability/shap_stylometric.py`).

- Computed hand-crafted style features (sentence-length variation, burstiness, vocabulary richness,
  word length, punctuation, POS mix) for the cleaned 1,280-essay corpus (cached to
  `data/processed/stylometric_features.parquet`), dropped the length-related columns to keep it pure
  style, and trained a gradient-boosted classifier on the student-level splits.
- **Result: a fully interpretable, style-only detector reaches F1 0.985** (confusion [[98, 2], [1, 99]]),
  almost matching the transformer's 0.99, with a balanced false-positive rate of 0.02 for both native
  and non-native writers. So the signal is captured nearly as well by transparent features as by a
  large model. This is also the non-transformer half of the planned hybrid detector (component 1).
- **SHAP** (`fig_shap_stylometric.png`, report `outputs/stylometric_shap.json`) gives the faithful,
  feature-level explanation: longer words and denser auxiliary-verb use push toward AI; more
  sentence-length variation, richer vocabulary, and more rare one-off words push toward human. This is
  the defensible "this was flagged because the sentences are uniform and the words longer" account the
  project is built around, and it is faithful because the model literally uses those features (unlike
  the diffuse token-level view in 5.3).
- Written up as Chapter 5 section 5.5 (with Figure 5.3); folded into the dissertation (now 32 pages,
  validates) and added to the AICS paper. The token-level finding from earlier in the chapter is what
  motivates preferring this feature-level explanation.
- Env note: installing `shap` downgraded numpy 2.3.4 -> 2.0.2 (shap needs < 2.3). All tools still
  import and run (spaCy, sklearn, transformers, torch), and this actually satisfies the earlier
  scipy/sklearn "numpy < 2.3" warning. Keep this in mind for the eventual version re-pin.

### Critical review and rigour/honesty improvements (16 June 2026)

Did a deep critical review of the detection/explainability/robustness work and the dissertation,
attacking it from several angles (data leakage, statistics, overclaiming, reproducibility) and
verifying every finding against the code. It returned 29 findings, all well-evidenced. Acted on the
empirical-rigour and honesty findings now; the strategic/scope findings are summarised for the
supervisor meeting.

Implemented (new results):
- **Confidence intervals + seed stability** (`src/evaluation/confidence.py`, `fig_confidence_intervals.png`,
  `outputs/confidence_intervals.json`), the review's top finding. Bootstrap 95% CIs on the test
  metrics: DeBERTa F1 0.99 [0.97, 1.00], RoBERTa 0.995 [0.98, 1.00], stylometric 0.985 [0.97, 1.00],
  M4 cross-generator 0.97 [0.97, 0.98], M4 cross-domain 0.79 [0.77, 0.80]. The in-domain intervals
  overlap heavily (n=200), so the model differences are within sampling noise; the M4 intervals are
  tight (large n). The stylometric model is identical (0.985) across five seeds, so it is stable.
- **Style-vs-locale control** (`src/detection/style_locale_control.py`, `outputs/style_locale_control.json`):
  removing every British/American spelling variant and contraction (1,251/1,280 essays changed)
  leaves the function-words accuracy unchanged (0.995 -> 1.00) and the TF-IDF model at 1.00. So the
  locale/formality tell is real but non-load-bearing; the separability is genuine distributional
  style. This answers the reviewer's concern and strengthens the claim.

Honesty/wording fixes folded into the chapters and the AICS paper:
- Softened the DeBERTa-vs-RoBERTa "architecture agreement" claim: the 0.990/0.995 gap is one essay
  on n=200 and within the overlapping CIs; the meaningful agreement is the shared one-sided error
  (false positives on human text, no AI missed), not the near-identical scores.
- Reported the in-domain fairness FPR as counts with the caveat that n=50/group is too small to
  establish a fairness gap; pointed to the Chapter 6 cross-domain result as the real fairness evidence.
- Named the two robustness corpora separately (Test A = OUTFOX essays, Test B = M4/SemEval-2024 Task 8),
  noted the generator sets differ, and made the human-side false-positive rate the load-bearing
  evidence for the domain claim (it cannot be a generator artefact).
- Noted the TF-IDF 100% had one residual markup token ("fnote") in its vocabulary, so the
  function-words-only 99.5% is the cleaner evidence.
- Flagged the IG sufficiency curve as an empty-input artefact (rely on comprehensiveness), and added
  the like-for-like caveat (stylometric model sees the full essay, DeBERTa only the first 512 tokens),
  the length-sensitivity of TTR/hapax, and that perplexity is deferred to the hybrid.
- Dissertation rebuilt to 35 pages, validates. Added Figure 3.5 (CIs).

Strategic findings for the 23 June supervisor meeting (not acted on, they touch locked scope):
- Detection and explainability are genuinely complete and well-validated; resist further polishing.
- The core contribution, question generation (Component 4, commercial vs local), is unbuilt
  (`src/question_gen/` empty). On-plan (weeks 7-8) but it is the bottleneck for the primary research
  question, the evaluation, and the output assembler. Recommended next: stand up Backend A
  (commercial via API, structured prompting over flagged passages) as a first slice.
- Backend B QLoRA toolchain (bitsandbytes/peft/trl) is not installed and untested; the 8B-on-8GB fit
  is unresolved. Smoke-test on a 3B model and bring the 8B-vs-3B decision to the supervisor.
- Chapter 2 (literature review) is still a skeleton with zero verified citations: the biggest writing
  gap, needs several sessions; the non-native-bias citation is load-bearing for Ch1/Ch6.
- Timeline: argument mining, Bloom's, output assembler, and the full evaluation are unstarted with
  ~10 weeks left and writing weeks protected; bring a depth-vs-breadth re-scoping decision (e.g.
  simplify argument mining to claim extraction) to the supervisor.
- Good news the review confirmed: numbers are internally consistent across chapters, the split-leakage
  audit is sound, and a style pass over the prose found nothing that needs rewording.

### Question generation: first working slice of the core contribution (16 June 2026)

Built the first end-to-end slice of the argument-aware question generation (the project's core
contribution, components 3 to 6 of the locked scope), running on the local model today.

- `src/question_gen/generate_questions.py`: takes a flagged essay and produces a Verification
  Interview Guide. Steps: number the sentences; extract the student's main claims with each claim
  citing the sentence numbers it comes from (so provenance is guaranteed, the model cites indices and
  we look up the real sentence, it cannot hallucinate a quote); generate grounded verification
  questions per claim that cannot be answered from the claim alone; tag each with a Bloom's level
  (transparent heuristic, a placeholder for the Component 5 BERT classifier); assemble JSON + Markdown.
- Backends are pluggable behind one interface. This slice uses the local Llama 3.1 8B via Ollama (the
  basis for Backend B); the commercial Backend A is a stub that plugs into the same interface once an
  API key is available, which is what enables the core commercial-versus-local comparison. The backend
  is recorded in every guide so runs are comparable.
- Demo on flagged essay 3108a: 4 claims, each tied to exact source sentences, 3 grounded questions
  each (for example, for a claim about Shakespeare's "dramatic intensity" it asked what specific
  features of the dialogue justify that description). Output in
  `outputs/verification_guides/3108a_ai.md` and `.json`. The questions genuinely need understanding,
  not just the claim, which is the whole point.
- Written up as Chapter 7 (`chapters/07_question_generation.md`); dissertation now 37 pages, validates.
- Honest first-slice caveats recorded in the chapter: claim extraction is prompted (a stand-in for the
  trained Persuasive-Essays argument miner), Bloom's is a heuristic, and the questions are not yet
  evaluated (the discrimination simulation is the next step). Backend A (commercial) and the 8B-vs-3B
  Backend B decision are for the supervisor meeting.
- Ops: restarted Ollama (`ollama serve`) to run the local backend; left running idle.

### Discrimination simulation: the primary, judge-free question evaluation (16 June 2026)

Stood up the project's main evaluation method (`src/evaluation/discrimination_sim.py`) and got the
first measured question-generation result, with a finding I did not expect.

- Method: for each question, a context-aware model answers it WITH the source passage and a
  context-blind model answers with the question ONLY; discrimination = similarity(aware-answer,
  source) minus similarity(blind-answer, source), via local chat + local embeddings (nomic-embed-text),
  so no API key and no judge. Bootstrap 95% CIs included.
- Result (flagged essay 3108a): claim-grounded questions mean discrimination ~0.05 [0.01, 0.10];
  generic essay questions ("what is your main argument", "what evidence supports your conclusion")
  ~0.31 [0.27, 0.36]. Non-overlapping CIs. The generic questions discriminate MORE, the opposite of
  what I assumed (`fig_discrimination_sim.png`, `outputs/discrimination_sim.json`).
- Why, and the finding: the context-blind model is a full LLM with broad world knowledge, so a
  question that names specific content ("what makes Shakespeare's dialogue intense") is answerable
  without the essay. Generic "explain your own argument/evidence" questions force retrieval of the
  student's actual essay, so they discriminate. The simulation therefore measures essay-specificity,
  and the lesson is that good verification questions must make the student reproduce their own
  reasoning/evidence/choices, not recite subject facts.
- Did a finding -> fix -> re-measure loop: rewrote the question prompt to demand the student's own
  reasoning and evidence (kept in `generate_questions.py`); grounded discrimination rose from ~0.03
  (v1, preserved in `discrimination_sim_v1.json`) to ~0.05 (v2), still below generic, because a
  specific question still tends to name the content.
- Honest caveat recorded: the blind LLM is a strong, conservative stand-in (more knowledge than a
  non-understanding student), so a low score means "answerable by a knowledgeable non-reader", not
  "useless"; the real picture sits between the two lines.
- Written up as Chapter 8 (`chapters/08_evaluation_questions.md`); dissertation now 40 pages, validates.
- Next: run the simulation across many essays (not one), add the supplementary LLM-as-judge rubric
  with cross-model agreement, and bring in the commercial backend for the commercial-vs-local
  comparison on the same measure.

### Meeting 4 pack prepared (21 June 2026)

Built the full pack for the 23 June supervision meeting, matching the established per-meeting pattern
(visual deck + talk track + a written brief, plus sendable PDFs).

- New deck builder `dissertation/presentation/make_meeting4_visual_deck.py` -> `Meeting4_visual.pptx`
  and `talk_track_meeting4_visual.md` (12 slides: recap of the audited detector, then the week's new
  work, explainability/SHAP, faithfulness, M4 transfer, the fairness false-positive result, rigour,
  the question-gen slice as a worked example, the discrimination-sim finding, the five decisions, and
  the plan). Every figure caption and number is read from `outputs/*.json` so nothing on screen
  contradicts the files.
- Written brief `dissertation/meetings/Meeting_4_Agenda_Brief.md` (+ `.docx`, `.pdf`): agenda, the
  five-point progress summary, the five decisions, and the timeline.
- Rendered sendables via LibreOffice: `Meeting4_visual.pdf`, `Meeting_4_Agenda_Brief.pdf`, and the
  draft `Dissertation_Shpyl_progress_draft.pdf` (40 pages, confirmed).
- Went back through the pack critically from three angles (checking every number against the JSONs,
  my own style rules, and the questions Vini is likely to ask). The numbers all held up, and the pass
  flagged real improvements that I applied: dropped the repetitive "New:" slide
  titles; relabelled the cross-generator result as the OUTFOX split (not "M4" broadly); reworked the
  worked-example slide to show two positive-discrimination questions instead of one negative-scoring
  one; named the AI-on-AI smoke-test caveat (the demo and the sim ran on a generated essay);
  downgraded the discrimination "finding" to a single-essay signal and added the
  similarity-to-source metric caveat; added a fairness-mitigation direction for the 79% false-positive
  rate; made decisions 2 and 3 decidable (8B QLoRA smoke-test first, name Qwen2.5 3B; state what the
  argument-mining re-scope costs); and recorded the scope note that the explainable detector is now
  the style-only model with the transformer kept as the black-box comparison.
- No new experiments or scope changes; this was meeting preparation only. The decisions themselves
  wait on Vini (commercial API access, the 8B-vs-3B fallback, the argument-mining re-scope).

### Meeting 4 held, record written (23 June 2026)

Met Vini (online). Wrote up the formal record in the ATU template
(`dissertation/meetings/Meeting_Record_4_Mykhailo_Shpyl.docx` and `.pdf`, same format as records 1 to 3).
Key outcomes and new obligations:

- **New graded deadline: intermediate presentation on 14 July 2026 (Week 6, online), carries marks.**
  Must cover progress, writing, and implementation. The introduction and literature review chapters
  should be finished before it. This is now the top priority.
- **No meeting next week (Vini travelling). Next supervision meeting 7 July 2026, 12:00 online.**
- **Chapter 1 feedback.** Add two or three opening paragraphs that introduce the work before the
  background; elaborate the background; expand the problem statement (currently too concise).
  References are mandatory and must be added as I write (Zotero already in use). Per-chapter outline can
  stay brief for now. Fix the bullet-point spacing. Vini wants the Word document (not only the PDF) to
  comment on; final submission stays PDF.
- **API and compute.** Vini supportive. I am to contact the programme coordinator Aoife Hill about
  student API credits and compute (Paul Rini pointed me to her). 8 GB VRAM is tight for 8B QLoRA;
  Qwen2.5 3B is the fallback. (Draft email to Aoife already prepared.)
- **AICS.** Vini to forward the call/email; deadline window around October; Student Track.
- Vini reviewed the work and called the question-generation framework clear. She suggested looking more
  at paragraph openings and closings, where AI text tends to give itself away. She also noted, as a
  journal reviewer, that high AI usage in submissions is now common and increasingly tolerated when the
  research is strong, which supports this project's defensible-verification framing.
- Immediate priorities from here: Chapter 1 expansion + references, the literature review, and the
  14 July presentation; in parallel, contact Aoife and run the 8B-vs-3B smoke test.

### Backend A built: real commercial backends, free-tier first (25 June 2026)

Aoife Hill confirmed ATU cannot provide API access/credits or a GPU better than the 8 GB laptop, and
referred the decision to Vini. My decision, pending her sign-off: free-tier commercial API first, small
self-funded spend if needed. Implemented Backend A accordingly so the commercial-vs-local comparison
(a core contribution) can run as soon as a key is in place.

- New `src/question_gen/commercial_backend.py`: three providers behind the same
  `chat_json(system, user)` interface as the local OllamaBackend.
  - Gemini (free tier, default) via REST; Anthropic Claude via the official `anthropic` SDK
    (default model claude-opus-4-8, overridable to a cheaper tier for cost); OpenAI GPT via REST. Robust JSON extraction (tolerates ```json fences and prose),
    retry/backoff on 429/5xx, and clear errors when a key is missing.
  - Keys come from env or a local `.env` (already gitignored); added `.env.example`.
- Wired into `generate_questions.py`: `--backend commercial --provider {gemini,anthropic,openai}
  [--model ...]`. The old placeholder stub is gone.
- New `src/evaluation/compare_backends.py`: generates a guide with each backend and scores all of
  them on the SAME discrimination measure, so the gap reflects the generator. Reports mean
  discrimination + bootstrap CIs, Bloom mix, and provenance; writes `outputs/backend_comparison_<id>.json`
  and `fig_backend_comparison.png`. Backends with no key are skipped and reported, so it runs
  local-only today and gains the commercial arm once a key exists.
- requirements.txt: added `requests`, `python-dotenv`, and `anthropic` (the last only for the Claude
  path; Gemini/OpenAI use REST).
- Verified offline (no key/network): module imports, JSON extraction across fenced/prose/plain
  replies, clean missing-key errors for all three providers, both CLIs, and compare_backends import.
  Not yet run live: needs a free Gemini key in `.env` (https://aistudio.google.com/apikey); the local
  arm of the comparison also needs `ollama serve` running (it was idle at time of writing).
- Next: drop a free Gemini key in `.env`, then `python src/evaluation/compare_backends.py --id 3108a
  --source ai --commercial gemini` for the first commercial-vs-local numbers. Still pending Vini for
  the locked-scope sign-off (commercial backend funding model + 8B-to-3B fallback).

### First commercial-vs-local result (25 June 2026)

Ran the comparison live with a free-tier Gemini key. Note: the default `gemini-2.0-flash` was
quota-blocked on this key (429 on the first call), so the Gemini default is now `gemini-2.5-flash`
(works, free tier) in `commercial_backend.py`. Result on essay 3108a (12 grounded questions each,
scored on the same local discrimination model; `outputs/backend_comparison_3108a.json`,
`fig_backend_comparison.png`):

- Local Llama 3.1 8B: mean discrimination 0.032, 95% CI [-0.005, 0.083].
- Commercial Gemini 2.5 Flash: mean discrimination 0.022, 95% CI [-0.002, 0.048].
- The CIs overlap heavily, so on this one essay there is no detectable difference between the free
  local model and the commercial model. Both sit far below the generic-question baseline (0.31),
  consistent with the earlier finding that content-specific grounded questions discriminate weakly.
- First read on the research question ("can a local open model match a commercial LLM"): so far yes,
  it matches (even nominally edges it) on this measure. Heavy caveats: n=1 essay, the local model is
  base llama3.1:8b (not yet QLoRA-fine-tuned), both are zero/few-shot prompted, and both score near
  the floor, so this compares two backends on a measure where neither shines. Real test: scale across
  essays, blend the retrieval-forcing question style so discrimination is not floored, then re-run
  with the fine-tuned local model.

### Scaled the comparison + Chapter 1/2 for the 14 July presentation (1 July 2026)

Two parallel threads for the graded 14 July intermediate presentation.

- **Scaled commercial-vs-local comparison.** New `src/evaluation/compare_backends_batch.py`: samples N
  essays (seeded), builds a guide with each backend, pools per-question discrimination, saves after every
  essay (resumable) and records per-backend failures. The free-tier Gemini quota blocked
  `gemini-2.0-flash` and then `gemini-2.5-flash` mid-run; `gemini-flash-latest` still had quota, so the
  batch now runs on that, launched detached so it keeps running if the terminal closes (same as the
  Ollama server). At 9 of
  15 balanced essays: local pooled 0.048 [0.032, 0.063], commercial 0.057 [0.037, 0.078], CIs overlap,
  so still no detectable difference (local matches commercial). Figure `fig_backend_comparison_batch.png`.
- **Chapter 1 expanded** per Vini's feedback: a new 1.1 overview that introduces the work before the
  background, an elaborated background, and a fuller problem statement (opacity, fairness, and the
  flag-is-not-understanding gap). Verified citations added inline.
- **Chapter 2 literature review written** as real prose (was a skeleton), grounded only in sources I
  verified by direct search against arXiv / ACL Anthology / dblp / publisher pages: Mitchell et al. 2023
  (DetectGPT), Wang et al. 2023/2024 (M4, SemEval-2024 Task 8), Krishna et al. 2023 (paraphrase attack),
  Wu et al. 2025 (survey), He et al. 2021 (DeBERTa), Lundberg and Lee 2017 (SHAP), Sundararajan et al.
  2017 (Integrated Gradients), DeYoung et al. 2020 (ERASER faithfulness), Stab and Gurevych 2017
  (Persuasive Essays), Anderson and Krathwohl 2001 (Bloom's), Hadifar et al. 2022 (EduQG),
  Mohammadshahi et al. 2023 (RQUGE), Liang et al. 2023 (non-native bias). `litreview_sources.md`
  regenerated as a verified dossier. Remaining `[ref: ...]` markers are honestly flagged for the next pass.
- I tried to script the citation checks first but the automation did not finish, so I verified each
  source myself with direct web search. Rebuilt the dissertation docx and PDF (now 44
  pages). No em dashes in the new prose.
- Next: finish the batch to 15, fill the remaining citation markers, and add a reference list from Zotero.

### Citations filled + balanced comparison (2 July 2026)

- Verified seven more sources by direct search and filled their markers: Liu et al. 2019 (RoBERTa),
  Jain and Wallace 2019 (attention is not explanation), Zheng et al. 2023 (LLM-as-judge), Hayes and
  Krippendorff 2007 (alpha), Guo et al. 2024 (neural QG survey), Kumar et al. 2025 (automatic Bloom
  classification), Jin et al. 2024 (GenAI adoption in higher education). Also caught one Liang marker
  that `replace_all` missed because it wrapped across a line break. `litreview_sources.md` updated to 20
  verified sources. Five markers remain, all genuinely hard to cite cleanly (hybrid stylometry, recent
  transformer argument mining, and a general local-vs-commercial LLM evaluation), flagged in the file.
- Batch reached all 15 essays: local completed 14, commercial 9 (free-tier Gemini quota capped the
  commercial arm). Computed the fair head-to-head on the 9 essays where both ran: local pooled 0.0475
  [0.032, 0.063] vs commercial 0.057 [0.037, 0.078], CIs overlap, per-essay split 4-5. Regenerated
  `fig_backend_comparison_batch.png` on the balanced set (paired per-essay + pooled CIs). Saved the
  balanced numbers into `backend_comparison_batch.json`.
- Rebuilt the dissertation docx and PDF (44 pages), no em dashes. To reach commercial n=15 the Gemini
  daily quota needs to reset (or a small self-funded key), then re-run the batch (it is resumable).

### Gap-closing for the graded 14 July presentation (2 July 2026)

Systematically closed what was missing for the intermediate presentation.

- **All citation markers filled.** Verified three final sources: Mindner et al. 2023 (features for
  ChatGPT detection, the hybrid-stylometry anchor), Pietron et al. 2024 (compact-LM argument
  classification), and Oketch et al. 2025 (closed-vs-open LLMs for automated essay scoring: open models
  match GPT-4 at up to 37x lower cost, the perfect local-vs-commercial anchor). Dossier now 24 verified
  sources; zero [ref:] markers remain in any chapter.
- **References section added.** New `chapters/09_references.md`, author-date, generated from the
  verified dossier only. While writing it I re-verified author lists and page numbers against the ACL
  Anthology and caught two of my own errors (SemEval-2024 pages are 2057 to 2079, not what I had; two
  overlong author lists trimmed to the confirmed names plus et al.). Builder now includes the chapter.
- **Stale text fixed in the doc build.** Removed the Chapter 2 "planned structure, citations in
  progress" note (now false); rewrote the abstract to cover the full current state (audit, faithfulness
  finding, robustness/fairness numbers, question generation, discrimination finding, and the
  commercial-vs-local result).
- **Comparison written into the dissertation.** New section 8.6 with the balanced commercial-vs-local
  result and Figure 8.2 (`fig_backend_comparison_batch.png`), with the caveats stated. Document now
  48 pages with References.
- **Built the graded presentation deck.** `presentation/make_intermediate_presentation.py` ->
  `Intermediate_Presentation_Shpyl.pptx` + PDF + `talk_track_intermediate.md`: 12 slides covering
  problem, research question, dataset, audited detector, explainability + faithfulness, robustness +
  fairness, worked verification-guide example, discrimination finding, commercial-vs-local, writing
  progress, and plan to submission. Every number read from the result JSONs; slides visually verified.
- **QLoRA smoke test (promised to Vini) run.** torch 2.5.1+cu124 sees the 4060; bitsandbytes 0.45.0 and
  peft 0.14.0 were pinned but not installed, so installed them and ran a real test: a 4-bit NF4
  Linear4bit forward pass executes on CUDA and LoraConfig builds. The Windows toolchain blocker is
  RESOLVED; the remaining question is only whether a full 8B fits in 8 GB (needs a model download and
  a training-step probe) or the agreed Qwen2.5 3B fallback is used. Scope sign-off still with Vini.
- Gemini daily quota reset, so the batch resumed detached and is filling the commercial arm (12/15 at
  the time of writing, reusing all local scores). Final balanced numbers and figure regenerate when it
  finishes. Proofread all the new content before the rebuild.

### Final comparison numbers, and the honest story changed (2 July 2026, later)

The batch finished at 14 of 15 (one essay's commercial run stuck on a per-minute 429; retried, still
blocked). One further essay (0038a) was excluded because local claim extraction returned an empty
guide. Final balanced set: 13 essays, 117 questions per backend, scored identically.

- Pooled: local 0.046 [0.033, 0.059] vs commercial 0.070 [0.051, 0.090]. Because the essays are the
  same, the fair test is paired: commercial higher on 9 of 13, mean paired difference +0.024 with
  bootstrap 95% CI [0.003, 0.044] (excludes zero), paired t-test p=0.053, Wilcoxon p=0.057.
- So the story moved from "indistinguishable at n=9" to "weak evidence of a small commercial edge at
  n=13, on the boundary of significance, with the local model competitive before any fine-tuning"
  (gap ~0.02 on a scale whose generic-question ceiling is 0.31). Section 8.6, the deck slide, and the
  abstract all updated to say exactly this; the n=9 snapshot is kept in the chapter as a lesson about
  small-n instability. Paired stats saved into `backend_comparison_batch.json` under `balanced`.
- Regenerated `fig_backend_comparison_batch.png` on the 13-essay balanced set (the batch's own figure
  writer had overwritten the balanced version with unbalanced aggregates; worth fixing in the script
  later).
- Final deliverables rebuilt and verified: dissertation 48 pages (References included, paired stats
  present in the PDF text), intermediate deck 12 slides + talk track. All Presentation-ready.

### QLoRA fit probes answered, critical review applied, handover notes refreshed (2 July 2026, later)

- **The 8B-vs-3B question is now measured, not argued** (`src/question_gen/qlora_fit_probe.py`,
  `outputs/qlora_fit_probe.json`). Llama 3.1 8B (4-bit NF4, LoRA r=16, grad checkpointing, paged
  8-bit optimiser, batch 1): loads at 5.3 GB, completes steps at seq 256/512/1024 but only by
  spilling past the physical 8 GB into system RAM (peak alloc 8.0/8.6/9.8 GB) with step times
  5.3 s / 12.3 s / 214 s. Qwen2.5 3B: peak 3.2/3.9/5.2 GB, steps 1.5/1.5/2.8 s, about 3 GB headroom.
  Practical answer for Vini: the 3B fallback, with the 8B numbers as the documented reason. On the
  presentation plan slide.
- **Adversarial review of the graded deck applied** (three lenses; examiner, statistics referee,
  style editor). Highest-value catches, all fixed: the example slide quoted a question I had
  silently paraphrased (now verbatim from the logged guide, and the notes say so); "QLoRA can only
  narrow the gap" was an unsupported directional claim (now "what the fine-tuning experiment will
  test"); the balanced paired stats were not produced by any checked-in script (compute_balanced()
  added to compare_backends_batch.py, verified to reproduce the committed numbers); the Gemini
  endpoint switch mid-run (4 essays gemini-2.5-flash, 9 gemini-flash-latest) is now disclosed in
  8.6; Wilcoxon p=0.057 added alongside the t-test; the paired CI is rounded outward [0.002, 0.045];
  the 15-sampled/13-used exclusions are spoken on the slide; ethics lines added to the plan slide
  (no human participants, licensed data, conversation-not-accusation); week markers added to the
  plan; the "X rather than Y" cadence thinned from 7+ instances to one; "this project" drumbeat in
  Chapter 2 varied; number formats unified (0.995 not 99.5%); assorted wording tells fixed.
- **The 'fnote' puzzle from the audit is solved and written up**: the bare word "fnote" appears in
  115 of 640 AI essays, echoed by the generator from keyword prompts built on human source text, so
  after tag-stripping it flips to a weak AI-side marker. It is a generation artefact, not leftover
  markup; it does not affect the function-word or stylometric results; stripping the bare token is
  queued for the next retraining pass. One added paragraph in Chapter 3.
- **Handover notes rewritten** (three weeks stale, still describing essay generation as the active
  task). Now state the current position, deliverables, ops notes, and norms.
- All deliverables rebuilt and re-verified: 48-page dissertation, 12-slide deck, talk track;
  proofread, with every number script-reproducible.

### Bloom classifier trained, references fully audited, Meeting 4 feedback closed out (2 July 2026)

- **Bloom's classifier (component 5) trained.** BERT-base on EduQG's 903 Bloom-labelled questions
  (levels 1 to 4 only, skewed 660/114/110/19), class-weighted, stratified 632/135/136 splits. First
  run crashed on unpadded batching; fixed with DataCollatorWithPadding. Results vs the keyword
  heuristic on the same test split: macro-F1 0.31 vs 0.16, accuracy 0.57 vs 0.26; per-class F1
  remember 0.76, understand 0.38, apply 0.09, analyse 0.0. Lower-vs-higher-order coarsening:
  higher-order F1 0.23 (BERT) vs 0.17 (heuristic); the heuristic's binary accuracy 0.86 is
  majority-class bias (all-lower scores 0.85). Honest conclusion: the component doubles the baseline
  and replaces the heuristic, and the label supply (73% one class), not the model, is the ceiling.
  Written up as new section 7.5 with Figure 7.1 (`fig_bloom_classifier.png`); model in
  `models/bloom_classifier/`; metrics in `outputs/bloom_classifier.json`. Deck plan slide updated
  (classifier moved from planned to done).
- **Full reference legitimacy audit (all 24 entries).** All 15 arXiv-carrying entries verified
  programmatically against the arXiv API (title, authors, year); RQUGE, SemEval-2024, ERASER, and
  Jain and Wallace verified against their ACL Anthology pages; Sundararajan against PMLR (pages
  3319 to 3328 confirmed); DeBERTa against dblp (He, Liu, Gao, Chen, ICLR 2021); the book/journal
  entries confirmed via publisher/dblp records. No fabricated or wrong entries found in the final
  list (the two errors caught earlier, SemEval pages and overlong author lists, were already fixed).
  Audit method recorded at the top of `litreview_sources.md`.
- **Meeting 4 "too vague" feedback now fully closed.** Already done earlier: opening paragraphs
  before the background, elaborated background, expanded problem statement, mandatory references
  (now a verified References chapter). Newly done: the dissertation outline (1.9) restructured into
  a small paragraph per chapter as Vini asked, and the bullet-spacing complaint root-caused in the
  doc builder (list items lacked the 1.5 line spacing body paragraphs have; fixed in
  `build_dissertation.js`) and verified in the rendered PDF.
- Deliverables rebuilt: dissertation now 49 pages (Bloom section added), deck 12 slides, all style
  gates clean.

### Output assembler built: the pipeline now produces its end product (2 July 2026, evening)

- **Component 6 works end to end.** New `src/pipeline/assemble_guide.py` produces the lecturer's
  Verification Interview Guide for a submission: runs the TRAINED DeBERTa detector live (3108a
  scores 0.9996), states the score's limits in plain language (in-domain F1, out-of-domain
  false-positive risk, "a reason to talk, never proof", and "no score of 1.0 exists on this scale"),
  lists the SHAP-validated drivers in lecturer terms, quotes each claim to its exact source
  sentences, re-tags all questions with the TRAINED Bloom classifier (higher levels marked advisory
  per its measured limits), and closes with a three-level suggested rubric. Renders Markdown ->
  Word -> PDF; the worked example is `outputs/verification_guides/3108a_ai_guide.pdf` (4 pages),
  generated with live models, visually verified. First render showed "1.00 probability" (a 2dp
  rounding of 0.9996), which overclaims certainty; fixed to show the raw value plus an explicit
  never-certain line.
- Chapter 7 gains section 7.6 describing the assembler; the deck's plan slide moves the assembler
  from planned to done. Dissertation now 50 pages.
- Retried essay 0315a's commercial arm: still 429-blocked on the free tier. The retry did confirm
  full reproducibility though: the checked-in batch script now recomputes the balanced paired stats
  and figure in one command, matching the dissertation exactly (13 essays, 117 questions each,
  0.046 vs 0.070, diff 0.024, t p=0.0529, Wilcoxon p=0.0574).
- Remaining build items: the claim extractor (Persuasive Essays corpus not yet downloaded; the
  prompted stand-in works meanwhile), the LLM-as-judge evaluation, and the Qwen2.5 3B QLoRA
  fine-tune plus re-comparison. All post-14-July per the plan slide.

### Persuasive Essays downloaded, claim extractor training, LLM-as-judge stood up (2 July 2026, night)

- **Persuasive Essays 2.0 downloaded** from the official TUdatalib repository (open bitstream, no
  login; local research use only, data/raw is gitignored). 402 essays with BRAT annotations
  (MajorClaim/Claim/Premise spans plus relations), the official 322/80 train/test split, license and
  guidelines included. Parser verified: span offsets match exactly at essay and paragraph level.
- **Claim extractor (component 3) training**: DeBERTa-v3-base token classification, BIO over the
  three component types, paragraph-level sequences (max paragraph 173 words, so nothing truncates),
  official split, strict span-level seqeval scoring. First run was learning (val span-F1 0.29 after
  epoch 1, 0.45 after epoch 2) when the DISK FILLED (2 GB free of 929) and a checkpoint save
  crashed it. Freed ~12 GB (deleted the two probe models from the HF cache, results already saved;
  roberta-large, unused; stale _runs checkpoint dirs) and relaunched.
- **LLM-as-judge built and first-run** (`src/evaluation/llm_judge.py`): four-dimension rubric
  (relevance, specificity, discrimination potential, cognitive appropriateness), judge-agnostic
  behind the same backend interface, resumable per question, and anchored to the objective
  simulation by Spearman correlation as the scope requires. Gemini judged 7 of 12 pilot questions
  before the daily quota cut it off. Preliminary and worth attention: the judge scores everything
  4.5 to 5.0 (ceiling effect) and correlates NEGATIVELY with the simulation (judge-mean vs sim
  rho -0.70, p 0.08, n 7): the judge loves exactly the claim-grounded questions the simulation
  shows are answerable from world knowledge. Small n, resumes when quota resets, but it is a live
  demonstration of why the scope anchors the judge to the objective measure instead of trusting it.
- Ops note: the laptop's disk is essentially full at the system level (915 of 929 GB used even
  after cleanup); I need to keep clearing room before any further model downloads or checkpoints.
- **Claim extractor trained (component 3 complete).** After the disk cleanup the retrained run
  finished: DeBERTa-v3-base BIO tagger, strict span-level micro-F1 0.63 on the official 80-essay
  test set (premises 0.72, major claims 0.54, claims 0.44, the expected ordering; dedicated
  CRF/joint architectures report higher, said plainly in the write-up). New section 7.6 with
  Figure 7.2; the assembler section renumbered to 7.7; deck plan slide updated (claim extractor
  moved from planned to done, integration design flagged for Vini). Dissertation now 51 pages.
  Every locked-scope component now has a trained model on disk: detector, explainability layer,
  claim extractor, question generation (two backends), Bloom classifier, and the output assembler.

### Meeting 5 pack prepared (2 July 2026, late)

Built the 7 July supervision pack in the established per-meeting format.

- `make_meeting5_deck.py` -> `Meeting5_visual.pptx` + `.pdf` + `talk_track_meeting5.md`: 9 slides.
  Title; since-Meeting-4 map; the paired commercial-vs-local result; the Bloom classifier; the claim
  extractor; the assembled guide (page 1 of the real PDF rasterised as `fig_guide_page1.png`, the
  showcase slide); the two measured findings needing decisions (QLoRA fit probes, the preliminary
  negative judge-vs-simulation correlation); Meeting-4 feedback closed out; decisions + a request
  for a 10-minute run-through of the graded presentation.
- `Meeting_5_Agenda_Brief.md` (+ `.docx`, `.pdf`, 2 pages): agenda, six-point progress summary,
  four decisions (Qwen2.5 3B sign-off, claim-extraction design, capped spend for judges two and
  three, AICS call), and the 14 July note (Chapters 1 and 2 finished as she asked).
- Every number on the slides reads from the result JSONs; key slides visually verified.
- The meeting record (ATU template) gets filled after the meeting, as with Meetings 1 to 4.

### Judge complete, comparison complete at 14 essays, and the headline moved (3 July 2026)

The Gemini quota reset overnight; a quota-waiter finished both pending jobs.

- **LLM-as-judge complete for judge one (12 of 12 questions).** Final: every rating 4.5 to 5.0
  (ceiling effect); no correlation with the objective simulation (judge-mean Spearman rho -0.14,
  p 0.67; the judge's own discrimination dimension rho -0.22, p 0.49). The correlation's history is
  its own small-n lesson: -0.70 at n=7, -0.25 at n=10, -0.14 at n=12. Honest summary: near-ceiling
  judge ratings carry no signal about measured discrimination, which is the live argument for
  anchoring judges to the simulation. Written up as new section 8.7.
- **Essay 0315a completed on retry, so the comparison is now 14 balanced essays (126 questions per
  backend), and the result crossed into significance**: local 0.042 [0.030, 0.055] vs commercial
  0.078 [0.058, 0.098]; paired difference 0.036, CI [0.008, 0.067], t p=0.040, Wilcoxon p=0.030,
  commercial higher on 10 of 14. The claim moves from "boundary" to "a small but statistically
  significant commercial edge, from a local model not yet fine-tuned". The full trajectory (n=9 no
  difference, n=13 boundary, n=14 significant) is reported in 8.6 as a small-n lesson. Updated:
  chapter 8.6, the intermediate deck slide 10, the Meeting 5 deck and brief, and the abstract.
- **Generic baseline re-measured on the same 14 essays** (the review flagged that 0.31 came from one
  pilot essay): `src/evaluation/generic_baseline_batch.py`, 84 generic questions, pooled 0.30 with CI
  [0.28, 0.32], so the pilot's 0.31 held up almost exactly. Chapter 8.6 and the decks now cite the
  same-essays baseline; the comparison figure's dashed line is relabelled "generic baseline, same
  essays (0.30)".
- **Figure 8.2 made properly paired.** The batch figure writer was regenerating the left panel as
  two jittered scatters, but the caption and the analysis are paired; rewrote `make_figure` to draw
  one connecting line per essay between its two backends (exact pairing from the saved per-question
  scores). Regenerated from saved data, no model calls.
- Final rebuild of all four deliverables with everything consistent: dissertation 52 pages (new
  section 8.7 on the judge; updated 8.6), intermediate deck 12 slides, Meeting 5 deck 9 slides,
  Meeting 5 brief 2 pages. Everything proofread; every number on every slide traces to a saved
  result JSON. Every locked-scope component is trained, the core
  comparison is complete and significant, and both the graded-presentation pack and the 7 July
  meeting pack are ready.

### QLoRA fine-tune experiment (Backend B), preliminary and pending sign-off (3 July 2026)

I freed more disk space, so the paused fine-tune experiment is running again. Framed as PRELIMINARY
evidence for the 8B-to-3B decision (like the fit probes), not a change to the locked-scope document.

- `src/question_gen/finetune_qg.py`: manual QLoRA SFT (no trl) of Qwen2.5-3B-Instruct on 2,600 EduQG
  passage->question pairs, prompt tokens masked so it learns only the question; adapter saved to
  `models/qg_finetune_qwen3b/`. Chose the 3B because the fit probe proved the locked-scope 8B is
  impractical on the 8 GB card.
- `src/question_gen/hf_backend.py`: a transformers generation backend (base or +LoRA adapter) with the
  same `chat_json` interface, so the fine-tuned model plugs into the pipeline without Ollama/GGUF
  conversion. Robust JSON/plain-text parsing.
- `src/question_gen/eval_finetune.py`: isolates the fine-tuning effect. On a fixed claim set (extracted
  once with the local model), base Qwen 3B and fine-tuned Qwen 3B each write verification questions,
  both scored by the same discrimination simulation. Phases are sequenced for the 8 GB VRAM (Ollama for
  claims/scoring, HF for generation, freed between; never co-resident).
- Running detached via a waiter (finish fine-tune -> run eval -> figure `fig_finetune_eval.png`,
  `outputs/finetune_eval.json`). Results and the write-up (a Chapter 7 section, deck mention) follow
  when it lands; honest either way, since a narrow QG fine-tune can help or can hurt open verification
  questions.

### Supervisor sign-off received on all Meeting 5 asks (3 July 2026)

Vini approved everything proposed in the Meeting 5 pack:

- **Backend B: Qwen2.5 3B approved** (the 8B-to-3B scope change). I updated the scope notes to record
  the 3B as the scope of record (components list and compute section) and marked the bitsandbytes
  question as settled rather than pending. The fit probes were the evidence.
- **Claim-extraction design approved**: trained spans (Persuasive Essays extractor) for provenance,
  combined with the prompted extractor's readable phrasing.
- **Judges two and three approved**: a small capped self-funded spend on Claude and GPT for the
  cross-model agreement (Krippendorff's alpha) that the LLM-as-judge validation needs.
- **First download attempt of the Qwen 3B stalled** on a Windows symlink-privilege error plus a hung
  connection; completed cleanly on a retry via `snapshot_download`. The fine-tune then started properly
  (650 steps, ~4.7 s/step, GPU 97%). Deck/dissertation prose that still says "agreed as an option" or
  "settle with my supervisor" gets updated to "approved" in the same pass as the fine-tune results,
  once the evaluation lands.

### QLoRA fine-tune of Backend B: measured result (5 July 2026)

The Qwen2.5 3B QLoRA fine-tune finished and the isolate-the-effect evaluation ran to completion.

- **Training** (`src/question_gen/finetune_qg.py`): 2,600 EduQG passage-to-question pairs, 4-bit NF4
  base, LoRA r=16 on q/k/v/o, prompt tokens masked in the loss, 2 epochs, batch 1 x grad-accum 8,
  paged AdamW 8-bit, gradient checkpointing. Ran in under an hour on the 4060, final train loss 1.42.
  Adapter saved to `models/qg_finetune_qwen3b/`.
- **Result** (`src/question_gen/eval_finetune.py`, `outputs/finetune_eval.json`): on the same 18 claims
  from 6 essays and the same discrimination scorer, only the question-writing model changing. Base
  Qwen 3B mean discrimination 0.0325 (95% CI [0.015, 0.051], 54 questions); fine-tuned 0.154 (95% CI
  [0.085, 0.227], 26 questions). Mann-Whitney p = 0.007, intervals do not overlap. Roughly a 4.7x gain.
- **Two honest caveats, both written into the chapter.** The counts differ (54 vs 26) because the
  fine-tuned model learned EduQG's one-question-per-passage habit and emits a single, tighter question
  per claim. And the discrimination metric rewards passage-specificity, which is the direction
  fine-tuning pushed the model, so the effect is reported as "more discriminative", not "better" in a
  broader sense. Small test (6 essays), reported as a clear isolated signal, not a sweeping claim.
- **Not cross-compared** with the commercial-vs-local batch (Gemini 0.078 vs Llama 8B 0.042): that
  batch used the 8B as the local arm and ran on different essays. The clean like-for-like step still
  outstanding is to run the fine-tuned 3B through that same paired comparison.
- **Written up:** new Chapter 7 section 7.8 with Figure 7.3 (`fig_finetune_eval.png`), registered in
  the docx builder (figdims + Table of Figures) and rebuilt to PDF. The graded intermediate deck gains
  a dedicated fine-tune slide and the plan/prose move from "planned"/"agreed fallback" to
  "done"/"approved". The two result JSON note fields updated from "pending sign-off" to the 3 July
  approval (numbers untouched).

### Like-for-like 4-way comparison: the fine-tuned local model wins (5 July 2026)

Ran the definitive commercial-vs-local comparison on a fixed claim set
(`src/evaluation/likeforlike_4way.py`, `outputs/likeforlike_4way.json`). One claim set per essay is
extracted once with the neutral Llama-8B extractor, then all four question writers (Llama 8B, free-tier
Gemini, base Qwen 3B, fine-tuned Qwen 3B) answer the SAME claims, scored by the same discrimination
simulation. Only the question writer varies, so this is tighter than the Section 8.6 batch (where each
backend also chose its own claims). Same 14 balanced essays. VRAM-sequenced, resumable per essay.

- **Pooled means:** fine-tuned 3B **0.153** [0.106, 0.202] (n=62); base 3B 0.041 [0.028, 0.055] (n=126);
  Llama 8B 0.031 [0.016, 0.045] (n=126); commercial Gemini 0.017 [-0.001, 0.036] (n=66, 9/14 essays).
- **Paired (per-essay means):** fine-tuned vs commercial +0.166 [0.108, 0.229], t p=0.001, Wilcoxon
  p=0.004, higher on all 9 shared essays. Fine-tuned vs base 3B +0.123, p<0.001, higher on 13/14
  (replicates Section 7.8 at scale; 0.153 here vs 0.154 there). Commercial vs Llama 8B -0.016, p=0.20
  (not significant): the Section 8.6 commercial edge disappears once claims are fixed.
- **Headline:** the locally fine-tuned open 3B, which fits the 8 GB laptop, beats commercial Gemini on
  the discrimination measure when both write about the same claims. Direct evidence for the second half
  of the research question.
- **Caveats kept in the write-up.** Commercial is quota-crippled (free-tier 429s -> 9/14 essays, pooled
  CI touches zero), so the commercial pair is provisional pending a refill to n=14 when quota resets;
  the resume guards now retry empty entries so the refill is a one-command re-run. The fine-tuned model
  writes ~1 tight question per claim (62 vs 126), and the metric rewards specificity, so the effect is
  "more discriminative", not "better" in general. The commercial-vs-8B flip is offered as a hypothesis
  (part of Gemini's Section 8.6 edge may have come from choosing easier claims), not a settled result,
  given the paired n of 9.
- **Written up:** new Chapter 8 Section 8.8 with Figure 8.3 (`fig_likeforlike_4way.png`, bars labelled
  with question count and essay coverage); Section 7.8 closing updated to point to it; figure registered
  in the docx builder; dissertation rebuilt to PDF. The graded deck's fine-tune slide upgraded to the
  4-way result ("Fix the claims, and the fine-tuned local model wins") and the plan slide updated.

### CORRECTION: the fine-tune "win" is a metric-gaming artifact (5 July 2026)

This supersedes the headline of the previous two entries. Auditing the actual questions
(`src/evaluation/qg_quality_audit.py`, `outputs/qg_quality_audit.json`) shows the QLoRA fine-tune did
NOT produce better verification questions. It failed.

- **95% of the fine-tuned model's questions are degenerate** (59/62): multiple-choice stems like
  "Which of the following is correct?" with no options ever supplied, plus a few raw JSON fragments.
  The fine-tune overfit EduQG's mostly-multiple-choice format. The base 3B, Llama 8B and commercial
  arms produce 0% degenerate questions.
- **These stems game the discrimination metric.** The literal string "Which of the following is
  correct?" scores mean discrimination 0.44 (11 occurrences), higher than any real question from any
  model, because a contentless question makes the source-aware and source-blind answers diverge at
  random. So the fine-tuned arm's 0.153/0.154 measures emptiness, not quality.
- **What this means:** the earlier claims ("fine-tuned local beats commercial", "materially stronger
  question writer", "+0.166 p=0.001") are retracted as artifacts. The second half of the research
  question stays open. Two real findings replace the false one: (1) fine-tune data format dominates,
  so Backend B needs an open-ended QG set, not EduQG; (2) the judge-free discrimination sim is gameable
  by degenerate questions and needs a well-formedness gate. This vindicates the evaluation plan's rule
  of never trusting a single automatic score.
- **What still stands:** the commercial-vs-local comparison among well-formed writers (Section 8.6, and
  Section 8.8's other three arms) is unaffected; on fixed claims the small open models are competitive
  with free-tier commercial Gemini.
- **Corrected across deliverables:** Chapter 7 Section 7.8 rewritten ("a result that looked too good"),
  new Chapter 8 Section 8.9 with Figure 8.4 (`fig_qg_quality_audit.png`), abstract sentence added, the
  deck's "wins" slide replaced by the honest audit slide, plan slide corrected. Caught by inspecting
  the text behind the number, which is the whole point.

### v2 fine-tune on open-ended data: diagnosis confirmed, component half-fixed (6 July 2026)

Acted on the Section 8.9 diagnosis: re-fine-tuned Qwen 3B with everything identical except the data,
EduQG -> SQuAD (open-ended, passage-grounded), `src/question_gen/finetune_qg_v2.py`, adapter in
`models/qg_finetune_qwen3b_v2/`. Evaluated the honest way, degeneracy audited before any score
(`src/question_gen/eval_qg_v2.py`, `outputs/qg_v2_eval.json`), on the same 18 claims as v1.

- **Diagnosis confirmed:** v2 is 0% degenerate vs v1's 95%. The multiple-choice collapse was caused by
  EduQG's format, not by fine-tuning as such.
- **Real (not artifact) gain:** v2 discrimination 0.102 [0.060, 0.144] vs base 0.027 [0.015, 0.040],
  Mann-Whitney p = 0.0003, on well-formed questions. v1's higher 0.154 was 95% degenerate stems gaming
  the metric; v2's lower 0.102 is genuine.
- **Honest limits:** v2 is modest, still below the generic baseline (~0.30); SQuAD trains a factual
  style ("What is the purpose of judicial review?") that is partly answerable without the essay. So
  Backend B is diagnosed and half-fixed; the next step is training on reasoning-demanding verification
  questions distilled from the pipeline's own prompts.
- **Also fixed:** `hf_backend._extract_json` cut-at-first-'?' plus JSON-block stripping (v2 sometimes
  appended JSON after the question); unit-tested on the captured bad strings. The degeneracy audit now
  runs before every score, as the standing rule.
- **Written up:** Chapter 8 Section 8.10 with Figure 8.5 (`fig_qg_v2_eval.png`, base vs v1-artifact vs
  v2); Section 8.9 pointer and the abstract updated; deck plan slide + notes updated. Dissertation and
  deck rebuilt.

### Meeting 5 pack refreshed, chapters tidied, work committed (6 July 2026)

- Refreshed the 7 July pack with the week's real story: the Meeting 5 brief and deck now cover the
  approvals put to work, the fine-tune round trip (the artifact caught by the audit, then the v2 fix),
  the tighter fixed-claim comparison, and this week's plan (reasoning-style training data for Backend
  B, judges two and three on the approved spend, the commercial-arm refill when quota resets). Two new
  slides walk the fine-tune story from the figures.
- Tidied the draft chapters before committing: removed the leftover draft banners and notes-to-self
  from the chapter tops, fixed a wrong-voice sentence in Chapter 3 (the corpus writers are not "my
  students"), and reworded a few log entries that had drifted into the third person. The abstract's
  closing line now reads plainly as a working-draft notice.
- Rebuilt everything after the tidy-up: the 60-page dissertation (docx and PDF), the Meeting 5 deck
  and brief, and the intermediate presentation. Committed the full body of work since 23 June.

### Well-formedness gate implemented; v3 pipeline built; refill and judges (6 July 2026, later)

- **The gate from Section 8.9 is now code, not a promise.** One shared rule
  (`src/question_gen/wellformed.py`) is used by the quality audit, the guide builder, and the
  evaluations. `build_guide` now filters malformed questions before they can reach a lecturer and
  records `n_dropped_malformed` in every guide, so a misbehaving backend is visible rather than
  hidden. Evaluations keep scoring raw output but always report the degeneracy rate next to the
  score. Unit-checked against the captured degenerate stems.
- **v3 of the fine-tune is ready to run.** `build_v3_dataset.py` distills reasoning-style
  verification questions from the local Llama 3.1 8B running the production prompt over the training
  essays (the 15 evaluation essays are excluded to avoid contamination; only well-formed teacher
  questions become targets; Llama's licence permits training other models on its output).
  `finetune_qg_v3.py` keeps every setting identical to v1 and v2 so the three-way comparison isolates
  the data format: multiple-choice (EduQG) vs factual open-ended (SQuAD) vs verification-style
  (self-distilled). `eval_qg_v3.py` audits degeneracy before scoring, on the same 18 claims.
- **Commercial refill running.** Gemini quota reopened, so the fixed-claim comparison is filling its
  five missing commercial essays (the framework reuses everything else). One transient hang in an
  Ollama stop call was fixed with a timeout; the run is otherwise cache-driven.
- **Judges two and three are still blocked on keys.** The `.env` has placeholder entries for the
  Anthropic and OpenAI keys but no real values yet; the judge script is ready and resumable once the
  keys arrive.

### Commercial arm refilled: the no-edge result now stands on the full sample (6 July 2026, later)

Gemini quota reopened and the fixed-claim comparison refilled its five missing commercial essays
(105 commercial questions, 14 of 14 essays on every arm; a few claims inside covered essays were lost
to per-minute rate limits). The headline sharpened: on identical claims the commercial edge is gone
entirely. Gemini 0.024 [0.008, 0.041] vs Llama 8B 0.031 and base Qwen 3B 0.041; the paired difference
is -0.005 (p = 0.62), with Gemini higher on 7 of 14 essays, a coin flip. So the Section 8.6 edge
(+0.036, p = 0.040, own-claims design) does not survive fixing the claims, measured on the complete
sample rather than hypothesised from a partial one. The reading: it came substantially from claim
selection. Chapter 8.8, Figure 8.3 (now 14/14 on every bar), both decks, and the Meeting 5 brief all
updated and rebuilt. The v3 chain (distil reasoning-style pairs -> train -> audited eval) is running
in the background at about 1,200 pairs per hour.

### Attention visualisation done: component 2's three methods on one yardstick (6 July 2026, later)

- Built `src/explainability/attention_viz.py`: final-layer [CLS] attention averaged over heads, on
  the same matched essay pair as the IG figure, then the identical ERASER-style ablation protocol
  (same 50-essay test sample, same seed, same k-sweep), run on CPU so the GPU stayed free for the v3
  chain. This closes the third method the scope lists for the explainability layer.
- Result: attention and IG are nearly indistinguishable on faithfulness (comprehensiveness ratio
  1.43 vs 1.44 at k=34; both leave the detector at a coin flip when only their top tokens are kept).
  The diffuse-signal conclusion now stands on three methods rather than two, and the SHAP feature
  view stays the lecturer-facing explanation. A nice detail: the AI essay's single most-attended
  token is an em dash, echoing the stylometric register story.
- New Section 5.6 with Figures 5.4 and 5.5; dissertation rebuilt (now 62 pages).
- Also built and dry-ran `src/evaluation/balanced_vs_natural.py` for the Meeting 2 training-data
  structure comparison: 247 human essays per arm (plus AI twins), natural allocation matching full
  BAWE proportions (AH-native 53 vs AH-non-native 10) against a balanced allocation (31 per cell),
  same size, same hyperparameters, same test split; per-cell F1, natural-weighted aggregate,
  fairness FPR by L1, paired McNemar. Queued to train when the v3 chain frees the GPU.
- Judge agreement is ready to compute automatically (Krippendorff's alpha, interval metric,
  unit-tested) as soon as a second judge completes; both commercial judges await account credit.

### Judge machinery smoke-tested with a preliminary manual rating (6 July 2026, later)

The second and third commercial judges are still blocked on account credit, so the cross-judge
agreement machinery was smoke-tested with a clearly-labelled stand-in: I rated the 12 pilot questions
against the exact rubric myself as a manual stand-in for the intended Claude judge, stored in
`outputs/llm_judge_preliminary_claude.json` with the provenance spelled out (manual, not
reproducible by script, rater not context-free, NOT a judge of record; the canonical llm_judge.json
is untouched). Nothing from this goes in the dissertation; the funded API judges replace it.

What the preview shows is still useful:
- The machinery works end to end (Krippendorff's alpha, pairwise Spearman, anchoring).
- The preliminary ratings spread the scale (3.0 to 4.75, mean 3.9) where Gemini sits at the ceiling
  (4.5 to 5.0). Agreement between them is consequently poor (alpha -0.40, Spearman 0.26 ns), which
  previews the likely real finding: a ceiling-effect judge cannot agree with any judge that actually
  discriminates, which is more support for anchoring judges to the objective simulation.
- The preliminary ratings also show no correlation with the simulation (mean rho -0.14, and the
  discrimination dimension alone -0.02), matching Gemini's -0.14. Two judge-like raters, zero signal
  about measured discrimination between them.

### All three LLM judges complete: the judge validation lands a decisive negative (6 July 2026, later)

API credit arrived, so judges two and three ran on the approved capped spend (Claude Opus 4.8 via the
Anthropic API, GPT-4o-mini via the OpenAI API; the anthropic SDK installed on the way). All three
judges have now rated the 12 pilot questions, and the scope's validation criteria are all computed
(`outputs/llm_judge.json`):

- **Ceiling effects in two of three:** Gemini 4.5 to 5.0 (mean 4.81), GPT 4.75 to 5.0 (mean 4.94).
  Claude uses the scale, 2.5 to 4.5 (mean 3.67), marking down the content-naming questions.
- **Cross-model agreement: poor.** Krippendorff's alpha (interval) across the three judges is -0.25;
  no pairwise Spearman is significant (0.20 to 0.45, all p > 0.13).
- **Anchoring: no judge tracks the objective measure.** Gemini rho -0.14 (ns), Claude -0.30 (ns), and
  GPT significantly negative at -0.75 (p = 0.005): its few below-ceiling ratings fell on the questions
  that objectively discriminate best.
- **Reading:** rubric ratings measure how good a question looks, not whether it works. The judge-free
  simulation carries the empirical weight, and the anchored design is vindicated with evidence. This is
  the second independent instance of the project's central lesson (after the fine-tune artifact): no
  automatic score is trusted until checked against something it cannot game.
- The preliminary manual rating from earlier today is marked superseded in its own file; notably it
  predicted the pattern (spread scale, poor agreement with the ceiling judge) almost exactly.
- Chapter 8.7 rewritten as "three judges, anchored, and the anchoring was needed"; abstract, Meeting 5
  brief and deck, and the intermediate deck all updated; every deliverable rebuilt for tomorrow's
  meeting (dissertation now 63 pages).

### v3 complete: the data-format experiment has its answer (7 July 2026)

The overnight chain finished: 2,604 self-distilled verification questions (307 essays, evaluation
essays excluded, teacher Llama 8B via the production prompt, only gate-passing questions kept), the
identical QLoRA fine-tune (final loss 0.72 vs v1's 1.42 and v2's 1.27, on-format data is easier to
learn), and the audited evaluation on the same 18 claims.

- **v3 output is exactly what the product needs:** zero degenerate questions of 42, 2.3 per claim
  (v2 managed 1.3), and the style is the pipeline's own ("How did you decide to use the example of X
  as evidence", "how did you connect it to your broader argument").
- **Metric: 0.064 [0.033, 0.096]**, about 2.5x base (0.027) and double the 8B teacher's own score in
  the fixed-claim comparison (0.031), so the 3B student overtook its teacher.
- **The twist:** v2 still posts the higher raw number (0.102) with terse factual one-liners that are
  not verification questions at all. The metric alone would pick the unfit adapter (it rewards
  specificity and penalises the content-quoting that verification style requires, Section 8.3). Third
  instance of the central lesson after the v1 artifact and the judge panel; the metric does not get to
  decide alone. v3 is the working Backend B on style-fit plus real gain, v2 documented.
- Written up as Section 8.11 with Figure 8.6 (base / v1-artifact / v2 / v3); 8.10's ending now hands
  over to it. Meeting 5 brief restructured (v3 into results, a real plan section: balanced-vs-natural
  running, hybrid fusion, claim-extraction integration, the relation-classification decision, then the
  Discussion and Conclusions chapters). All deliverables rebuilt; dissertation 65 pages.
- Balanced-vs-natural launched on the freed GPU (both models ~20 minutes; near-ceiling expected
  overall, the per-cell, natural-weighted and fairness numbers are the informative part).

### Morning integration: v3 written up, natural-distribution control done, references completed (7 July 2026)

- **Section 8.11 and Figure 8.6**: the completed data-format experiment (base 0.027 / v1 artifact /
  v2 0.102 / v3 0.064 with 0% degeneracy), including the honest reading that the metric alone would
  rank the unfit v2 first, so it does not decide alone. Meeting 5 brief restructured around it.
- **Section 6.6 and Figure 6.4**: the balanced-vs-natural training control (Meeting 2 item). Both
  same-size detectors score identically everywhere (F1 1.000, zero prediction disagreements, McNemar
  p = 1.0), so the balanced design is not what makes the corpus separable; at this separability the
  comparison has no resolution, which is stated plainly.
- **Nine method references added, every one verified against its primary source** (NeurIPS
  proceedings, ACL Anthology, CrossRef, arXiv, the OpenAI report PDF): QLoRA, LoRA, SQuAD, Qwen2.5,
  Llama 3, GPT-2, BERT, the BAWE corpus article (Alsop and Nesi 2009), and Nomic Embed. In-text
  citations placed at first use in Chapters 3, 4, 7 and 8. Reference list now 33 entries; the
  verification evidence is recorded in litreview_sources.md.
- **Chapter 7 staleness pass**: title no longer says "first slice"; 7.2 and 7.4 now read as a
  point-in-time record whose stand-ins are resolved by 7.5, 7.6, 7.8 and Chapter 8, instead of
  promising work that is already done.
- **Hybrid fusion running** (component 1 completion): GPT-2 perplexity feature added, four arms on
  the home test split, and a zero-shot cross-domain arm on the identical M4 sample as Chapter 6 to
  test whether the fusion softens the false-positive failure. One real bug caught before the run:
  the feature-to-text join must use (id, label) because human essays and their AI twins share ids.

### The integrated guide: every trained component in one artifact (7 July 2026)

Built the final form of the pipeline's output, the design Vini approved at Meeting 5.
`src/argument_mining/extract_spans.py` is the inference side of the trained claim extractor (BIO
decoding over paragraph sequences, verbatim spans with char offsets and roles).
`src/question_gen/integrated_guide.py` orchestrates the approved "spans for provenance, prompt for
phrasing" design: prompted claims give readable phrasing and sentence citations, the trained miner
attaches the verbatim major-claim/claim/premise spans found in those sentences (matched by word
overlap), the v3 fine-tuned backend writes the questions, and the well-formedness gate filters them.
`src/detection/save_hybrid.py` persists the fitted hybrid (perplexity-augmented GBM plus the logistic
fuser, with home-corpus perplexity cached), and `src/detection/hybrid_detect.py` scores a single
submission with it, so the assembler now reports the hybrid detector rather than the bare transformer.

On the worked submission (3108a) the guide came out with four claims, twelve questions, zero dropped
by the gate, and the hybrid scored 0.957 against the transformer's own 0.9996: the style half pulls
the over-confident transformer back, which is the behaviour a lecturer facing a possible false
positive wants. Every element on the page is now a trained component's live output. Written up as
Section 7.9 with Figure 7.4 (the guide's first page, regenerated from the integrated build). One
honest touch kept in: short claim fragments from the miner (its known claim-boundary weakness) are
filtered from the display, so only substantive spans appear.

### Graded presentation rebuilt for a 20-minute general audience (7 July 2026)

Vini confirmed the 14 July intermediate presentation is 20 minutes with no questions, for a general
audience (basic computer literacy, no project knowledge). Rebuilt the deck accordingly:

- Three new plain-language concept figures (`make_concept_figures.py`): an end-to-end pipeline diagram
  ("a flag opens a conversation, not an accusation"), a picture explaining the two-AI question test,
  and a roadmap that marks "we are here" at the half-way point with Done on the left and Still-to-do
  on the right.
- 16 slides, plain language throughout, arc: the problem in everyday terms, the whole idea in one
  picture, the building blocks with the honesty stories (the 100% artefact, the fairer hybrid, the
  fine-tune that lied), how questions are graded without students, laptop-vs-cloud, then an explicit
  "where it is weak" slide, the half-way roadmap, the plan, and a one-sentence takeaway.
- Deliberately honest about being half done and about explainability being the weakest part (not yet
  clear enough for a non-technical lecturer), which is named as the biggest job for the second half.
- Ran a three-persona comprehension review (layperson, first-year CS student, skeptical examiner) via
  a workflow. Fixed everything it flagged: the one blocking issue (why empty questions fool the score,
  now spelled out), defined "word patterns vs writing style", replaced all "out-of-domain" jargon with
  "unfamiliar kinds of writing", relabelled the on-screen "Bloom" tag as "Thinking level" and named
  Bloom's taxonomy in the script, clarified the 61% base rate, broke the pipeline caption into steps,
  and added the stress-test bridge on the hybrid slide.

### Completion sprint, part 1: promises discharged and the weak points attacked (7 July 2026, evening)

The full completion plan is in motion; this entry records the first wave.

- **The fnote control (Chapter 3's promise) is discharged.** Stripped the bare token from both
  classes, retrained the identical DeBERTa configuration as a control (`fnote_control.py`): F1 0.995
  against the record's 0.990, one test essay of difference, confirming the token carried nothing.
  The detector of record stays untouched; Chapter 3's sentence now reports the control instead of
  promising it.
- **The relation classifier (the last unbuilt scope element) is training.** Ordered within-paragraph
  component pairs from Persuasive Essays 2.0, official split, supports / attacks / none with
  class-weighted loss (attacks are only 0.7 percent of pairs, which the write-up will state
  plainly). About 20,600 training pairs, 4,900 test pairs.
- **The explainability upgrade the presentation admits is needed has its first concrete piece:** a
  per-submission explanation card (`explain_submission.py`). For one essay it turns the hybrid's own
  style model into five plain sentences, each with its number set against the human-corpus median
  ("sentences unusually uniform: variation 7.5 against a typical 11.4") and its push direction, plus
  a small bar chart. One bug caught in testing: wording was driven by the SHAP sign and could
  contradict the numbers ("more helper verbs" over a below-median value); rewritten so the words
  always follow the values. The card is now embedded in every generated guide, and Chapter 5 gained
  Section 5.7 describing it and what it does not yet do (no lecturer testing yet).
- **The multi-generator test slice is generating**: matched essays from Gemini and GPT-4o-mini with
  the original corpus recipe (same system prompt, title plus keywords plus target length), for 40
  test-split sources; chat_text added to the commercial backends for raw prose. Length ratios are
  recorded per essay; GPT undershoots length notably on some essays, which the analysis will state.
- **Two measured-safeguard scripts are ready to run** once the GPU frees: the abstain-band sweep
  (turning the proposed uncertain band into abstain-rate versus false-accusation curves) and the
  multigen scorer (detection rates on generators the detector never saw, on home ground).
- **The scaled fixed-claim comparison is written and validated**: 30 essays, the working v3 backend
  replacing the artifact v1 arm, seeded from the 4-way state so the 14 original essays' claims and
  three arms are reused verbatim, and new essays drawn only from the pool that is neither an
  evaluation essay nor one of the 307 essays v3 trained on (guards checked: zero violations).
- **Appendices added as Chapter 12**: the pipeline's verbatim prompts, a model-and-training
  configuration table, the repository map, and the worked guide reference.

### The abstain band, measured, and it half-refutes its own proposal (7 July 2026, night)

The abstain-band sweep ran on the cross-domain sample with per-text hybrid probabilities kept.
Accuracy among judged texts climbs from 0.79 (no band) to 0.88 (band 0.2-0.8, declining 28.5 percent
of texts), so abstention genuinely concentrates verdicts on cases the detector gets right. But the
human false-positive rate among judged texts barely moves (about 0.19 across the sweep): the
surviving false accusations are confident errors, mostly arXiv abstracts, that no uncertainty band
can catch. I had drafted the figure title as "abstention cuts false accusations" before reading the
sweep; the data refused, the title was corrected, and the write-up (new Section 6.8, Figure 6.6)
records both the benefit and the limit. The mitigation plan updates: abstention is worth deploying
for accuracy, per-domain calibration is still needed for fairness, and the question stage remains the
backstop. Also tonight: the scaled 30-essay comparison relaunched with an Ollama self-healing guard
after the server died mid-run again (the guard detected and restarted it on first use), and the
multi-generator slice continues generating.

### Completion sprint, part 2: the scaled answer, and two more safeguards priced (8 July 2026, early)

The overnight experiments landed faster than planned thanks to state reuse, and the results close the
evaluation programme.

- **The scaled fixed-claim comparison (30 essays, 754 questions, all arms 0 percent degenerate).**
  Three findings in rising order: the commercial-vs-8B null replicates (-0.005, p = 0.62); the v3
  fine-tune's gain is solid at scale (0.081 vs base 0.038, +0.040 paired, p = 0.0001, higher on 22 of
  27 essays); and the claim retracted twice in weaker forms is now admissible, the fine-tuned local
  model beats free-tier Gemini on shared essays (+0.050, p = 0.0094, higher on 11 of 13), with the
  ranking and fitness-for-purpose finally agreeing because every question passed the gate. Honest
  frame kept: Gemini quota lapsed again at 14 of 30 essays, so the commercial pair rests on 13, and
  the claim is scoped to the free tier. Written as Section 8.12 with Figure 8.7; Chapter 9's answer
  and sample-size limitation updated.
- **The multi-generator slice, scored.** The transformer of record flags 100 percent of both unseen
  commercial generators' essays (Gemini n=19, GPT-4o-mini n=40, matched subset agrees) and none of
  the 40 human sources. The hybrid catches all of GPT but only 68 percent of Gemini: the same style
  half that cut false accusations of humans extends some protection to a generator whose style
  drifts human-ward. The trade the fusion was designed to make, now measured on its other side.
  Section 6.9 with Figure 6.7.
- **The consistency audit is now a script** (`dissertation/docgen/audit_consistency.py`): fourteen
  headline numbers checked against their result JSONs, citation cross-check (33 in-text citations,
  33 entries, all matched after adding three genuinely missing citations: Anderson and Krathwohl for
  Bloom's taxonomy, DeYoung for ERASER, Pietron for the argument-mining comparison), and figure
  numbering. ALL CLEAN, and rerunnable before every hand-in.
- Draft stands at 86 pages. The evaluation programme named in the completion plan is done; what
  remains is writing polish, the real-student study (future work by design), and submission process.

### Refill running; timing verified; README and AICS draft brought current (8 July 2026)

- Gemini quota reopened, so the scaled comparison's commercial arm is refilling (the rerun is also
  self-healing the two essays whose claims failed overnight, so all arms head toward 30 of 30).
- Timed the graded talk from the speaker notes: 2,499 words is about 19.2 minutes at a prepared
  130 words per minute, with no slide over 1.7 minutes, so the 20-minute slot fits with a small
  buffer and nothing needs cutting.
- The assembler now handles the not-flagged case: a human submission's guide states plainly that no
  interview is suggested and frames the generated sections as a contrast example.
- README rewritten to the current state: what the system does, how to run the pipeline end to end,
  results at a glance, the consistency audit, and an honest status. The stale "generating the AI
  half" status and the internal working-notes pointers are gone.
- The AICS Student Track draft is citation-complete: every placeholder resolved against the same
  verified reference list as the dissertation (Wu, Liang, Mitchell, Liu, He, Wang x2, Krishna,
  Mindner, Sundararajan, DeYoung, Alsop and Nesi), and the numbers already match the final results.
  What remains before submission is prose polish, CEURART formatting, and the call details from
  Vini.

### Scaled refill lands, judge study grows 5x, and a full style revision (12 July 2026)

- The commercial arm refill finished on gemini-flash-latest after gemini-2.5-flash's quota died:
  coverage rose from 14 to 22 of 30 essays (856 questions total, still zero degenerate anywhere).
  The headline hardened with it: v3 beats the free commercial tier +0.047 paired, p = 0.0002,
  higher on 20 of 22 shared essays, and beats its own base +0.046, p < 0.0001, on 25 of 30. The
  commercial-versus-8B null replicates (p = 0.60). Eight essays still lack commercial questions;
  the rerun after the next quota window tops them up.
- Split the commercial arm by model version for the disclosure: the refill questions are shorter
  (median 29 words vs 43) and score higher (0.035 vs 0.024), so the mixed arm does not flatter the
  local side. Disclosed in Section 8.12 next to the free-tier caveat.
- Expanded the judge study from 12 to 60 questions (20 per arm from the scaled run, one seeded
  sample, anchored to the same run's simulation scores). The anti-correlation is not a small-sample
  artifact: Claude rho -0.34 (p = 0.010), GPT -0.37 (p = 0.004). New nuance: the judges now agree
  with each other in rank (0.58) while both point away from the objective measure, and both rank
  v3, the best arm by simulation, at or near the bottom. Written into Section 8.7; the Gemini
  judge reruns when its quota allows.
- A full style revision of the dissertation and the deck: the prose leaned on the same few
  rhetorical habits (mirrored contrasts, punchline paragraph endings, trailing "which is"
  significance tags, recycled catchphrases, staged triads) and rewrote chapter by chapter, keeping
  every number, citation and figure reference fixed and re-running the consistency audit after.
- The graded deck now leads with the strong results (99% detection, 100% on two unseen commercial
  generators with zero false accusations, the laptop model beating the cloud tier 20 of 22) while
  keeping the honesty arc; talk re-timed at 18.5 minutes. Abstract rewritten to match.
