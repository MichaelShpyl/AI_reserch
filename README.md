# Explainable AI for Academic Integrity Verification

MSc dissertation project, Atlantic Technological University.
Author: Mykhailo Shpyl. Supervisor: Dr. Vini Vijayan.

## What this is

This project builds an explainable AI pipeline for academic integrity
verification. It does two connected things. First, it detects whether text in a
student submission is likely AI-generated. Second, for flagged work, it generates
verification questions drawn from the specific claims and arguments in that
submission, so a lecturer can check whether the student actually understands what
they handed in. The aim is not to accuse students. Current detectors are black
boxes that return a percentage a lecturer cannot defend if challenged. This
system makes every decision explainable, giving lecturers transparent, defensible
evidence. The target user is a lecturer running a large module (150 to 300
students) who has no time to prepare individual verification questions for every
flagged submission.

The full write-up, including scope, research question, datasets and results, is the dissertation
draft in `dissertation/` (built by `dissertation/docgen/build_dissertation.js`).

## Setup (local: Windows, NVIDIA GPU)

Target machine: Windows 11, RTX 4060 laptop GPU, Python 3.11.

1. Create and activate a virtual environment:

       python -m venv .venv
       .venv\Scripts\Activate.ps1

2. Upgrade pip, then install PyTorch with CUDA support first:

       python -m pip install --upgrade pip
       pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

3. Install the remaining requirements:

       pip install -r requirements.txt

4. Download the spaCy English model:

       python -m spacy download en_core_web_sm

5. Verify the GPU is visible to PyTorch:

       python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device')"

   Expect your torch version, `CUDA available: True`, and a line naming the RTX 4060.

## Repository layout

    config/          configuration (paths, model names, hyperparameters)
    data/raw/        downloaded datasets, never edited
    data/interim/    intermediate processing
    data/processed/  final corpora ready for training
    src/             pipeline source, one subpackage per component
    notebooks/       exploration only, not core logic
    outputs/         generated guides, figures, results
    tests/           24 tests, no GPU or corpus needed: `python -m pytest tests/ -q`
    dissertation/    the written document, drafted alongside the code

## Data

Datasets are not stored in this repository (licences are respected; BAWE is CC BY-NC-SA). Put raw
downloads in `data/raw/`: BAWE from the Oxford Text Archive, M4/SemEval-2024 Task 8, Persuasive
Essays 2.0, EduQG, and SQuAD via the Hugging Face hub. The loading and summary scripts live in
`src/data/`.

## Running the pipeline

The end product is the Verification Interview Guide, generated end to end with local models:

    python src/question_gen/integrated_guide.py --id 3108a --source ai --claims 4
    python src/pipeline/assemble_guide.py --id 3108a --source ai

The first command extracts claims (prompted phrasing plus the trained argument miner's verbatim
spans), writes questions with the fine-tuned local backend through the well-formedness gate; the
second scores the submission with the hybrid detector, builds the per-submission explanation card,
tags questions with the trained Bloom classifier, and renders Markdown, Word and PDF into
`outputs/verification_guides/`.

Every experiment writes its result to a JSON under `outputs/`, and the pre-submission consistency
audit checks the written chapters against those files:

    python dissertation/docgen/audit_consistency.py

## Results at a glance

- Detection: F1 0.99 in-domain after a corpus-artefact audit; transfers to six unseen generators at
  0.97; degrades cross-domain to 0.79 with false accusation as the failure mode, which motivates the
  design.
- The hybrid (transformer + stylometric features + GPT-2 perplexity) cuts cross-domain false
  accusations of humans by three to eight times at equal F1.
- Explanations are faithfulness-tested; the feature-level account passes where all three token-level
  methods fail, and each guide carries a plain-language per-submission explanation card.
- Question generation: on fixed claims across 30 essays (901 questions), the QLoRA fine-tuned 3B
  (trained on self-distilled verification questions) beats its base (p < 0.0001, 25 of 30 essays)
  and the free-tier commercial model on the 29 shared essays (p = 0.0003, 24 of 29), with every
  question passing the well-formedness gate.
- Three LLM judges neither agree with each other (Krippendorff's alpha -0.25) nor track the
  objective measure, and the anti-correlation replicates at a five-times-larger question set,
  which is why the judge-free simulation carries the evidence.
- Two controls with teeth: the training-distribution control, run where the task is hard, shows
  balancing the writer mix costs nothing in accuracy but slightly raises out-of-domain false
  accusations (p < 0.001); and sentence-level occlusion shows the style signal has no
  flag-carrying sentences to point at, which is why explanations stay at habit level.

## Tooling (writing, diagrams, slides)

Beyond the runtime in `requirements.txt`, these tools support writing the
dissertation and producing meeting materials. The Python dev packages are in
`requirements-dev.txt`.

- Writing and output: Pandoc (markdown to Word and PDF), MiKTeX (the LaTeX engine
  Pandoc uses for PDF).
- Diagrams: Graphviz (`dot`) and matplotlib.
- Document reading: `markitdown`.
- Editor: VS Code with Python, Pylance, Jupyter, Ruff, Markdown All in One,
  markdownlint, Code Spell Checker, GitLens, Rainbow CSV, and LaTeX Workshop.

Install the system tools (Windows) with winget, then restart the terminal and
VS Code so `pandoc` and `dot` are on PATH:

    winget install JohnMacFarlane.Pandoc
    winget install Graphviz.Graphviz
    winget install TheDocumentFoundation.LibreOffice

## Status

All six locked-scope components are built and trained, the pipeline runs end to end on an 8 GB
laptop, and the evaluation programme is complete. The dissertation draft (92 pages, twelve
chapters, 41 verified references) lives in `dissertation/`, with a per-session progress log in
`dissertation/progress_log.md` and the August validation-study protocol in
`dissertation/study_protocol.md`. Remaining work is the validation study, writing polish, and the
submission process.
