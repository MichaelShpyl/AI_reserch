# Copyright and third-party licences

## This work

Copyright (c) 2026 Mykhailo Shpyl.

The code, figures, written analysis and generated results in this repository were produced for an
MSc dissertation at Atlantic Technological University, supervised by Dr. Vini Vijayan.

No open-source licence is granted at present, so the default applies: all rights reserved. The
repository is public so that the dissertation's claims can be checked, every file path printed in
the document resolves to the file it names, and every number quoted can be traced to the results
file it was read from. If you want to reuse any of it, ask.

## Data used, and why none of it is here

No dataset is stored in this repository. Some of that is licensing and some is size, and the two
reasons are separated below because they have different consequences for reproducing the work.

| Source | Licence | Why it is absent |
|---|---|---|
| BAWE (British Academic Written English) | CC BY-NC-SA 3.0 | Licensed for research use, not for redistribution. A test in `tests/test_invariants.py` fails if any BAWE prose is committed |
| M4 / SemEval-2024 Task 8 | Per the shared task's terms | Available from the task organisers; not mirrored here |
| Persuasive Essays 2.0 (Stab and Gurevych, 2017) | TU Darmstadt terms | Obtained directly from the publisher |
| EduQG, SciQ, SQuAD, LearningQ | Open, per each dataset | Downloadable; not mirrored |

`README.md` gives the download location for each one, and the scripts under `src/data/` rebuild
every processed corpus from those downloads.

## Models

Model checkpoints are not committed because of their size, not their licence. Everything needed to
retrain them is here: the training scripts, the configuration, the pinned dependency versions and
the seeds.

Base models used and their terms: DeBERTa-v3-base and RoBERTa-base (MIT), BERT-base (Apache 2.0),
GPT-2 (modified MIT), Qwen2.5-3B (Qwen licence), and Llama 3.1 8B via Ollama (Llama 3.1 Community
Licence). The Llama licence permits use of its outputs, which matters here because the AI half of
the detection corpus was generated with it.

## If you are examining this work

Nothing in this repository needs a licence granted to read it, run the tests, or check a number
against the file it came from. `python -m pytest tests/ -q` runs without the corpus, the models, a
GPU or a network connection.
