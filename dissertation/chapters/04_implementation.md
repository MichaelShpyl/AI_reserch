# Chapter 4: Implementation

## 4.1 Environment and reproducibility

The whole project runs on my own laptop. After Meeting 3 it became clear that no ATU HPC or
cloud would be available, so every part has to fit a single RTX 4060 with 8 GB of VRAM, or fall
back to the CPU. That constraint shaped a lot of the choices below, and it is the reason the
models are base-sized rather than large.

The stack is Python on Windows, with PyTorch and the Hugging Face libraries for the transformer
work, spaCy for the linguistic features, and scikit-learn for the simpler baselines and the
audit probes. To keep runs repeatable I set a single random seed (42) everywhere that randomness
enters: the sampling, the train and test split, and the model training. Dependency versions are
pinned, and the work is committed to a local git repository in small steps with clear messages,
so any result can be traced back to the exact code that produced it. Long jobs (essay generation,
model training) are launched as detached background processes rather than tied to an open terminal
session, so they keep running if the session that started them closes. This sounds like a small
detail, but it cost me two failed generation runs before I worked it out, and it is now the
standard way I start anything that takes more than a few minutes.

## 4.2 Data acquisition and cleaning

The human side of the corpus is the British Academic Written English (BAWE) collection (Alsop
and Nesi, 2009), which I
downloaded from the Oxford Text Archive. A first script (`src/data/explore_bawe.py`) reads the
holdings spreadsheet, prints a summary, and cross-checks the recorded word counts against a plain
count of the text files so I know the metadata is trustworthy. A second script
(`src/data/clean_bawe.py`) produces a cleaned metadata table, dropping a row that was labelled
twice. The cleaned table is what every later step samples from.

## 4.3 Sampling

The sample is drawn by `src/data/build_sample.py`. Rather than sample by named discipline, I
stratify evenly across the four broad disciplinary groups and balance native against non-native
English writers, so that I can later test whether the detector is unfair to non-native writers.
A per-student cap stops any one person's style from dominating, and the train, validation and
test split is made at the level of the student, not the essay, so the same writer never appears
on both sides of the split. The result is 640 human essays with a versioned manifest
(`bawe_human_sample_manifest.csv`) that records, for each essay, its group, first-language status,
and split. The reasoning behind the sizes is written up in `dissertation/sample_design.md`.

## 4.4 Generating the matched AI essays

The AI half of the corpus is built by `src/generation/generate_ai_essays.py`. For each human
essay it generates one AI essay on the same topic and at the same target length, using Llama 3.1
8B running locally through Ollama. The length and topic matching is the most important part: if
the AI essays were systematically longer or shorter, or drifted onto different subjects, the
detector could separate the classes for the wrong reason and the whole corpus would be invalid.

Two things make the matching work. First, the prompt is anchored on keywords pulled from the
human source text, not just the essay title, after an early test produced an off-topic essay from
a title alone. Second, the generator runs a short continuation loop: if the first draft falls
short of the target length it asks the model to continue, up to a few rounds, so the AI essay
lands close to its human counterpart. The script is resumable, so a run that stops can be
restarted and it skips what is already done, and it keeps the machine awake while it runs. It
writes each essay to disk along with a metadata row recording the target and actual length, the
number of rounds, and the time taken.

Generation of all 640 essays completed locally with no failures. A validation script
(`src/generation/check_ai_corpus.py`) confirmed the match: the correlation between each AI essay
and its human source length is about 0.98, the mean length ratio is about 1.05, and over 99
percent of AI essays are within 20 percent of their source length. A keyword spot-check confirmed
the AI essays stayed on the same topics. So the two halves differ in writing, not in length or
subject, which is exactly the property the detector needs.

## 4.5 Building the labelled corpus

`src/detection/build_detection_corpus.py` pairs each human essay (label 0) with its matched AI
essay (label 1), carries the split and metadata across from the manifest, and writes a single
table of 1,280 rows. The same script has a cleaning mode (described in Section 4.7) that strips
markup before writing, so the raw and cleaned corpora can be compared directly.

## 4.6 The detector

The detector is fine-tuned by `src/detection/train_detector.py`. It uses the Hugging Face Trainer
to fine-tune a transformer to classify human against AI, with DeBERTa-v3-base as the primary model
and RoBERTa-base as a comparison. To fit the 8 GB card I keep the per-step batch small and use
gradient accumulation to reach a sensible effective batch, with mixed precision on the GPU. The
script reports accuracy, precision, recall, F1 and the confusion matrix, and on top of that it
breaks out the false-positive rate for native and non-native writers separately, which is the seed
of the fairness analysis. The stylometric feature extractor (`src/detection/stylometric.py`)
computes the linguistic features (sentence-length variation, vocabulary richness, part-of-speech
mix, and so on); those features are fused with the transformer into the hybrid detector in
Section 6.7 (`src/detection/hybrid_fusion.py`).

## 4.7 The audit and the cleaning step

The first detector scored a perfect 100 percent, which I did not trust. The audit that followed is
implemented in three small scripts, and they are part of the contribution, not throwaway checks.

`src/detection/audit_detector.py` runs the diagnostic battery: it confirms there is no student or
duplicate-text leakage across the splits, measures how much of the score a markup-only rule can
reach on its own, and trains simple interpretable baselines on raw and cleaned text, including a
function-words-only model that removes all topic information. `src/detection/text_normalize.py`
is the cleaning function that the audit relies on: it strips the BAWE export tags from the human
text and the markdown from the AI text, and flattens the layout, so that any remaining difference
is writing style rather than formatting. `src/detection/why_high.py` then explains why the cleaned
score stays high, by measuring locale spelling, contraction use, and how tightly the AI essays
cluster in style space, and by drawing the two classes in a two-dimensional style projection.

The finding, covered in Chapter 3, was that the original human text carried structural tags that
no AI essay had, and that this shortcut accounted for a large part of the perfect score. The
cleaning step removes it, and the detector is then retrained on the cleaned corpus to give the
honest headline figure.

## 4.8 Engineering notes and lessons

Two practical lessons are worth recording for the methodology. The first is the value of launching
long jobs as detached processes, which is what finally made the overnight generation reliable. The
second is the markup artefact itself: it is a clean example of why a strong result has to be
stress-tested before it is believed, and the fact that the audit is scripted means anyone can rerun
it and see the same thing. Both points feed directly into how I will build and check the remaining
components.
