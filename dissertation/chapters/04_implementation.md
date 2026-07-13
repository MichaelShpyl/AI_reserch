# Chapter 4: Implementation

## 4.1 Environment and reproducibility

The whole project runs on my own laptop. After Meeting 3 it became clear that no ATU HPC or
cloud would be available, so every part has to fit a single RTX 4060 with 8 GB of VRAM, or fall
back to the CPU. Most of the choices below follow from that limit, including the use of
base-sized models instead of large ones.

The stack is Python on Windows, with PyTorch and the Hugging Face libraries for the transformer
work, spaCy for the linguistic features, and scikit-learn for the simpler baselines and the
audit probes. To keep runs repeatable I set a single random seed (42) everywhere that randomness
enters: the sampling, the train and test split, and the model training. Dependency versions are
pinned. The work is committed to a local git repository in small steps with clear messages, so
any result can be traced back to the exact code that produced it. Long jobs (essay generation,
model training) are launched as detached background processes instead of being tied to an open
terminal session, so they keep running if the session that started them closes. This sounds like
a small detail, but it cost me two failed generation runs before I worked it out, and it is now
the standard way I start anything that takes more than a few minutes.

## 4.2 Data acquisition and cleaning

The human side of the corpus comes from the British Academic Written English (BAWE) collection
(Alsop and Nesi, 2009), which I downloaded from the Oxford Text Archive. A first script
(`src/data/explore_bawe.py`) reads the holdings spreadsheet, prints a summary, and cross-checks
the recorded word counts against a plain count of the text files so I know the metadata is
trustworthy. A second script (`src/data/clean_bawe.py`) produces a cleaned metadata table,
dropping a row that was labelled twice. Every later step samples from the cleaned table.

## 4.3 Sampling

The sample is drawn by `src/data/build_sample.py`. I stratify evenly across the four broad
disciplinary groups rather than by named discipline, and I balance native against non-native
English writers so that I can later test whether the detector is unfair to non-native writers.
A per-student cap stops any one person's style from dominating. The train, validation and test
split is made at the level of the student, not the essay, so the same writer never appears on
both sides of the split. The result is 640 human essays with a versioned manifest
(`bawe_human_sample_manifest.csv`) that records, for each essay, its group, first-language
status, and split. The reasoning behind the sizes is written up in
`dissertation/sample_design.md`.

## 4.4 Generating the matched AI essays

`src/generation/generate_ai_essays.py` builds the AI half of the corpus. For each human essay
it generates one AI essay on the same topic and at the same target length, using Llama 3.1 8B
running locally through Ollama. Matching on length and topic was the part I took most care over.
An AI half that ran systematically longer or shorter, or that drifted onto different subjects,
would let the detector separate the classes for the wrong reason, and the whole corpus would be
invalid.

The matching works through the prompt and a length loop. The prompt is anchored on keywords
pulled from the human source text, not just the essay title, after an early test with a title
alone produced an off-topic essay. The generator also runs a short continuation loop. If the
first draft falls short of the target length, it asks the model to continue, up to a few rounds,
so the AI essay lands close to its human counterpart. The script is resumable, so a run that
stops can be restarted and it skips what is already done, and it keeps the machine awake while
it runs. Each essay goes to disk along with a metadata row recording the target and actual
length, the number of rounds, and the time taken.

Generation of all 640 essays completed locally with no failures. A validation script
(`src/generation/check_ai_corpus.py`) confirmed the match: the correlation between each AI essay
and its human source length is about 0.98, the mean length ratio is about 1.05, and over 99
percent of AI essays are within 20 percent of their source length. A keyword spot-check
confirmed the AI essays stayed on the same topics. The detector needs the two halves to differ
only in the writing, and on these checks they do.

## 4.5 Building the labelled corpus

`src/detection/build_detection_corpus.py` pairs each human essay (label 0) with its matched AI
essay (label 1), carries the split and metadata across from the manifest, and writes a single
table of 1,280 rows. The same script has a cleaning mode (described in Section 4.7) that strips
markup before writing, so the raw and cleaned corpora can be compared directly.

## 4.6 The detector

Detector training lives in `src/detection/train_detector.py`, which uses the Hugging Face
Trainer to fine-tune a transformer to classify human against AI. DeBERTa-v3-base is the primary
model and RoBERTa-base the comparison. To fit the 8 GB card I keep the per-step batch small and
use gradient accumulation to reach a sensible effective batch, with mixed precision on the GPU.
The script reports accuracy, precision, recall, F1 and the confusion matrix. It also breaks out
the false-positive rate for native and non-native writers separately, and the fairness analysis
later builds on that breakdown. The stylometric feature extractor
(`src/detection/stylometric.py`) computes the linguistic features (sentence-length variation,
vocabulary richness, part-of-speech mix, and so on). Those features are fused with the
transformer into the hybrid detector in Section 6.7 (`src/detection/hybrid_fusion.py`).

## 4.7 The audit and the cleaning step

The first detector scored a perfect 100 percent, which I did not trust. Before accepting it I
ran an audit, implemented in three small scripts, and I count those scripts as part of the
contribution.

`src/detection/audit_detector.py` runs the diagnostic battery: it confirms there is no student
or duplicate-text leakage across the splits, measures how much of the score a markup-only rule
can reach on its own, and trains simple interpretable baselines on raw and cleaned text,
including a function-words-only model that removes all topic information.
`src/detection/text_normalize.py` is the cleaning function the audit relies on. It strips the
BAWE export tags from the human text and the markdown from the AI text, and flattens the
layout, so that whatever difference remains comes from the writing and not the formatting.
`src/detection/why_high.py` then explains why the cleaned score stays high. It measures locale
spelling, contraction use, and how tightly the AI essays cluster in style space, and it draws
the two classes in a two-dimensional style projection.

The finding, covered in Chapter 3, was that the original human text carried structural tags
that no AI essay had, and that this shortcut accounted for a large part of the perfect score.
The cleaning step removes the shortcut. The detector is then retrained on the cleaned corpus,
and the retrained score is the headline figure I report.

## 4.8 Engineering notes and lessons

A couple of practical points from this phase belong in the methodology. One is launching long
jobs as detached processes, which is what finally made the overnight generation reliable. The
other is the markup artefact itself. It shows why a strong result has to be stress-tested
before it is believed, and because the audit is scripted, anyone can rerun it and see the same
thing. Both points feed into how I will build and check the remaining components.
