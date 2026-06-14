# Chapter 2: Literature Review

> Draft note (delete before submission): this is a **skeleton and reading plan**, not
> prose. It sets out the sections, what each must cover, and search targets to find. It
> contains **no citations yet on purpose**. Every reference must be located, read, and
> verified by me before it goes in, using only 2021 to 2026 work (older items only where
> they are genuine foundations). Never paste a citation an AI hands over; find the real
> paper first. The prose is written after the model is built, so the methods are
> described as they actually turn out.

## How I will run the review

- Window: mainly 2021 to 2026, ideally 2023 to 2026, because the field moves fast.
  Foundational methods older than that are allowed where they are the original source
  (for example the attribution methods and the argument corpus).
- For each source: take the title or concept, find the real paper, read it, then cite it.
- Anchor the search on the methods and datasets already chosen for this project, then
  branch out to alternatives and criticisms.

## 2.1 Scope of the review

State what the review covers and why, and the search approach above. Map the sections to
the pipeline: detection, stylometry, explainability, argument mining, question
generation, question quality, and fairness.

## 2.2 Detecting AI-generated text

Cover the two broad families and where they fall short.
- Zero-shot and statistical detectors that use a language model's own probabilities
  (search: perplexity-based detection, log-probability curvature, "DetectGPT", watermarking).
- Supervised transformer detectors fine-tuned to classify human vs AI (search: RoBERTa
  AI text detection, DeBERTa text classification, GPT-generated text detectors).
- Benchmarks and shared tasks (anchor: M4 and SemEval-2024 Task 8, already in scope).
- Known weaknesses: brittleness to paraphrasing, domain shift, and the black-box score
  problem that motivates this project. Find recent survey papers on machine-generated
  text detection to frame the section.

## 2.3 Stylometric features

Explain the features the detector uses and the evidence behind them.
- Perplexity / predictability, burstiness (variance in sentence length and structure),
  type-token ratio, sentence-length variance, part-of-speech distributions.
- Work that combines stylometric features with transformer models (search: hybrid
  stylometric transformer AI detection). Note what each feature is supposed to capture.

## 2.4 Explainability and faithfulness

Cover attribution methods and how to test whether explanations are honest.
- Attention as explanation and its critics, Integrated Gradients, SHAP (search for the
  original method papers and recent NLP applications).
- Faithfulness testing by ablation or perturbation (search: faithfulness of explanations
  NLP, feature ablation, sufficiency and comprehensiveness). This justifies the ablation
  check in the pipeline.

## 2.5 Argument mining

Cover extracting claims, premises, and evidence, and linking them.
- Argument component identification as sequence labelling (BIO) and relation
  classification (anchor: the Persuasive Essays 2.0 corpus, Stab and Gurevych, already in
  scope; search recent transformer-based argument mining).
- Why provenance matters here: each extracted element must point back to its source
  passage so the questions are defensible.

## 2.6 Automatic question generation

Cover neural and LLM-based question generation, with an education focus.
- Neural QG from passages, controllable QG, and grounding or provenance in generated
  questions (search: educational question generation, controllable QG, LLM question
  generation grounding).
- The cost and control trade-off between commercial and local open models, which is the
  comparison at the centre of this project (search: open-source vs commercial LLM
  evaluation, local LLM deployment cost).

## 2.7 Bloom's taxonomy and cognitive level

Cover classifying questions by cognitive level for quality control.
- The revised Bloom's taxonomy as the labelling scheme (anchor: Anderson and Krathwohl,
  foundational, verify the reference).
- Automatic classification of question cognitive level (search: Bloom's taxonomy question
  classification, cognitive level prediction). Note the EduQG dataset already in scope.

## 2.8 Evaluating generated questions

Cover how question quality is judged, which is the project's main evaluation.
- Answerability and discrimination style evaluation: can a model answer the question
  without the source, and does access to the source change the answer (search: question
  answerability evaluation, reference-free QG evaluation).
- LLM-as-judge and its risks, plus inter-rater agreement with Krippendorff's alpha
  (search: LLM as judge reliability, Krippendorff alpha agreement).

## 2.9 Fairness and bias in detection

Cover bias against non-native English writers, which the project measures directly.
- Evidence that detectors flag non-native writing as AI more often (search: GPT detector
  bias non-native English writers). This grounds the fairness analysis.

## 2.10 Summary and the gap

Synthesise the sections and restate the gap: detection is improving but stays opaque, and
nothing turns a flag into source-grounded verification questions. That gap is what the
project addresses.
