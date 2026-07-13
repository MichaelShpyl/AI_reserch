# Appendices

## Appendix A: the prompts the pipeline uses

The pipeline's behaviour depends on its prompts, so this appendix reproduces them verbatim. Three
of them do the argument-aware work.

**Claim extraction** (system role): "You analyse a student essay for an academic-integrity
verification interview. A claim is a substantive position the student asserts and argues for, not
background, definition, or quotation. You cite the sentence numbers each claim is drawn from. Reply
with JSON only." The user turn supplies the essay as numbered sentences and asks for the n most
important claims, each with a one-line paraphrase and its sentence numbers. The model only returns
sentence numbers. The pipeline then looks up the source text from the real sentences, so an
invented quotation is impossible.

**Question writing** (system role): "You write verification questions for a lecturer to check
whether a student genuinely understands and wrote a claim in their own essay. The crucial property:
a knowledgeable person who has NOT read this essay should be unable to answer well from general
knowledge. So do not ask about general facts; ask the student to reconstruct their own reasoning,
name the specific evidence or examples they used, justify the choices they made, and explain how
this point connects to the rest of their essay. Avoid yes/no questions and avoid questions that
contain their own answer. Reply with JSON only." This wording came out of the Section 8.3 finding
that content-naming questions are answerable from general knowledge.

**Fine-tune training prompt** (identical across v1, v2 and v3, so only the data differs): "You are
a teacher. Read the passage and write one clear question that checks whether a student understood
it. Reply with the question only."

**Corpus generation** (system role, used for the original Llama corpus and reused verbatim for the
multi-generator test slice): "You are a university student writing a coursework essay. Output only
the essay itself as continuous academic prose. Do not write a title, headings, bullet points, a
reference list, or any framing such as 'Here is' or 'Sure'. Stay strictly on the given topic and
match the academic register of the stated discipline." The user turn anchors the topic with the
essay title, keywords extracted from the human source, and the target length.

## Appendix B: model and training configurations

| Component | Model | Key settings |
|---|---|---|
| Detector (transformer) | DeBERTa-v3-base | 3 epochs, lr 2e-5, batch 4 x accum 4, max 512 tokens, fp16, seed 42, student-level splits |
| Detector (comparison) | RoBERTa-base | identical settings |
| Stylometric model | Gradient boosting | 25 features incl. GPT-2 perplexity; seed 42 |
| Hybrid fuser | Logistic regression | over the two model probabilities, fitted on the validation split |
| Claim extractor | DeBERTa-v3-base, BIO | paragraph sequences, max 256 subwords, official 322/80 split, strict span scoring |
| Relation classifier | DeBERTa-v3-base, pairs | within-paragraph ordered pairs, class-weighted loss, 3 epochs |
| Bloom classifier | BERT-base | class weights, stratified 632/135/136, 6 epochs |
| QG fine-tunes v1/v2/v3 | Qwen2.5-3B-Instruct | QLoRA: 4-bit NF4, LoRA r=16 on q/k/v/o, 2 epochs, batch 1 x accum 8, paged AdamW 8-bit, prompt-masked loss, 2,600 pairs |
| Local generation | Llama 3.1 8B (Ollama) | temperature 0.2, seed 42 |
| Embeddings (simulation) | nomic-embed-text | local, via Ollama |

Each experiment writes its full result to a JSON file under `outputs/`. All the numbers quoted in
this document trace back to one of those files.

## Appendix C: repository map

```
src/
  data/            sampling and corpus construction (BAWE manifest, cleaning)
  generation/      matched AI-essay generation; the multi-generator test slice
  detection/       detector training, audit, normalisation, hybrid fusion, abstain band
  explainability/  Integrated Gradients, SHAP, attention, the per-submission card
  argument_mining/ claim extractor and relation classifier (training and inference)
  question_gen/    prompts, backends (local, commercial, fine-tuned), fine-tunes, gate
  bloom/           Bloom's-level classifier
  evaluation/      discrimination simulation, comparisons, judges, quality audit
  pipeline/        the output assembler (Verification Interview Guide)
outputs/           result JSONs, verification guides (generated)
dissertation/      chapters, figures, decks, meeting records, this document's builder
data/              raw and processed corpora (not committed; licences respected)
models/            trained checkpoints and adapters (not committed)
```

## Appendix D: the worked Verification Interview Guide

The complete guide for the worked submission (essay 3108a, AI-generated variant) is included with
the electronic submission as `outputs/verification_guides/3108a_ai_guide.pdf`. It is the file the
pipeline produced with live models, unedited. Its first page appears as Figure 7.4. The guide
contains the hybrid detection summary with component views, the per-submission explanation card,
four claims with two-way provenance (cited sentences plus the trained miner's verbatim argument
spans), twelve gated verification questions with thinking-level tags, and the three-level suggested
rubric for reading the student's answers.
