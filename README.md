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

See `CLAUDE.md` for the full scope, research questions, datasets, and phase plan.

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
    tests/           tests
    dissertation/    the written document, drafted alongside the code

## Data

Datasets are not stored in this repository. Download them per `CLAUDE.md` and put
raw downloads in `data/raw/`. The BAWE corpus is the starting point; the loading
and summary script lives in `src/data/`.

## Tooling (writing, diagrams, slides)

Beyond the runtime in `requirements.txt`, these tools support writing the
dissertation and producing meeting materials. The Python dev packages are in
`requirements-dev.txt`.

- Writing and output: Pandoc (markdown to Word and PDF), MiKTeX (the LaTeX engine
  Pandoc uses for PDF).
- Diagrams: Graphviz (`dot`) and matplotlib.
- Slides: `python-pptx` to build decks; LibreOffice to render a deck to images for a
  visual check.
- Document reading: `markitdown`.
- Editor: VS Code with Python, Pylance, Jupyter, Ruff, Markdown All in One,
  markdownlint, Code Spell Checker, GitLens, Rainbow CSV, and LaTeX Workshop.

Install the system tools (Windows) with winget, then restart the terminal and
VS Code so `pandoc` and `dot` are on PATH:

    winget install JohnMacFarlane.Pandoc
    winget install Graphviz.Graphviz
    winget install TheDocumentFoundation.LibreOffice

## Status

Foundation (data and environment) is complete. Current work: generating the AI half
of the detection corpus locally, then the detector. New contributors and new AI
sessions should start with `HANDOFF.md`.
