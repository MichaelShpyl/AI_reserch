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

Table: Model and training configuration for every trained component, with the settings needed to reproduce each run.

| Component | Model | Key settings |
|---|---|---|
| Detector (transformer) | DeBERTa-v3-base | 3 epochs, lr 2e-5, batch 4 x accum 4, max 512 tokens, fp16, seed 42, student-level splits |
| Detector (comparison) | RoBERTa-base | identical settings |
| Stylometric model | Gradient boosting | 23 stylometric features plus GPT-2 perplexity, 24 inputs; essay length excluded on purpose; seed 42 |
| Hybrid fuser | Logistic regression | over the two model probabilities, fitted on the validation split |
| Claim extractor | DeBERTa-v3-base, BIO | paragraph sequences, max 256 subwords, official 322/80 split, strict span scoring |
| Relation classifier | DeBERTa-v3-base, pairs | within-paragraph ordered pairs, class-weighted loss, 3 epochs |
| Bloom classifier | BERT-base | class weights, stratified 632/135/136, 6 epochs |
| QG fine-tunes v1/v2/v3 | Qwen2.5-3B-Instruct | QLoRA: 4-bit NF4, LoRA r=16 on q/k/v/o, 2 epochs, batch 1 x accum 8, paged AdamW 8-bit, prompt-masked loss, 2,600 pairs |
| Local generation | Llama 3.1 8B (Ollama) | temperature 0.2, seed 42 |
| Embeddings (simulation) | nomic-embed-text | local, via Ollama |

Each experiment writes its full result to a JSON file under `outputs/`. All the numbers quoted in
this document trace back to one of those files.

## Appendix C: the stylometric feature set

Every explanation a lecturer reads is built on these, so they are listed in full rather than
summarised. They are computed by `src/detection/stylometric.py` using spaCy, and all of them are
plain arithmetic over the text: nothing here needs a GPU or a trained model to reproduce.

Table: The 23 stylometric features, with what each one measures and why it is in the set.

| Feature | What it measures |
|---|---|
| `mean_sent_len` | Average sentence length in words |
| `std_sent_len` | Spread of sentence length. Human writing varies more |
| `sent_len_cv` | That spread relative to the mean, so it does not scale with length |
| `burstiness` | (sigma - mu) / (sigma + mu). Negative means unusually regular |
| `ttr` | Type-token ratio: distinct words over total words |
| `root_ttr` | The same, divided by the square root of length, which removes most of the length dependence |
| `hapax_ratio` | Share of words used exactly once. One-off vocabulary |
| `mean_word_len` | Average word length in characters |
| `punct_ratio` | Punctuation as a share of tokens |
| `pos_NOUN`, `pos_VERB`, `pos_ADJ`, `pos_ADV` | Content-word densities |
| `pos_PRON`, `pos_PROPN` | Pronoun and proper-noun density |
| `pos_ADP`, `pos_DET`, `pos_AUX` | Prepositions, determiners, auxiliaries: the function-word core |
| `pos_CCONJ`, `pos_SCONJ` | Coordinating and subordinating conjunctions, so sentence joining |
| `pos_NUM`, `pos_PART`, `pos_PUNCT` | Numerals, particles, punctuation tags |

Two things are deliberately absent. Essay length, as `n_words` and `n_sents`, is computed but
dropped before training, because a detector allowed to read length would learn the one shortcut
Section 4.6 exists to close. And there is no topic feature of any kind: the set contains no word
identities, which is what lets Section 3.6 argue that separation on these features is separation on
style.

GPT-2 perplexity is added as a twenty-fourth input to the fused style model in Section 6.7, where
it ranks first by mean absolute SHAP. It is kept separate here because it is the only feature that
needs a language model to compute, and therefore the only one a reader cannot reproduce with a
text file and a scripting language.

## Appendix D: repository map

Listing: Layout of the source tree, showing where each pipeline stage is implemented.

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
  webapp/          the lecturer-facing interface of Section 4.11 (FastAPI, one page)
outputs/           result JSONs, verification guides (generated)
dissertation/      chapters, figures, decks, meeting records, this document's builder
data/              raw and processed corpora (not committed; licences respected)
models/            trained checkpoints and adapters (not committed)
```

## Appendix E: the worked Verification Interview Guide

The complete guide for the worked submission (essay 3108a, AI-generated variant) is included with
the electronic submission as `outputs/verification_guides/3108a_ai_guide.pdf`. It is the file the
pipeline produced with live models, unedited. Its first page appears as Figure 7.4. The guide
contains the hybrid detection summary with component views, the per-submission explanation card,
four claims with two-way provenance (cited sentences plus the trained miner's verbatim argument
spans), twelve gated verification questions with thinking-level tags, and the three-level suggested
rubric for reading the student's answers.
