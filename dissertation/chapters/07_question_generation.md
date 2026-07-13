# Chapter 7: From a flag to verification questions

## 7.1 The step after detection

Detection on its own only produces a flag, and the earlier chapters showed how fragile a flag can
be out of domain. The real contribution of the project comes after the flag. A suspected submission
is turned into something a lecturer can act on fairly: a short set of questions, drawn from the
student's own claims, that check whether the student understands what they handed in. A student who
wrote and understood the work can answer them. A student who did not will struggle. This chapter
builds that step. It starts with a thin slice that runs end to end, then replaces the slice's
stand-ins one by one with trained components, and finishes with the fine-tuning of the local
backend.

## 7.2 How the slice works

The pipeline takes a flagged essay and produces a Verification Interview Guide
(`src/question_gen/generate_questions.py`). It runs in four steps.

First, it splits the essay into numbered sentences. Second, it extracts the student's main claims
and records the sentence numbers each claim comes from. The design choice that matters here is
provenance. The model is asked to cite sentence numbers instead of quoting text, and the source is
then looked up from the real numbered sentences, so the model cannot invent a quote. Every claim
points at exact lines in the submission. Third, for each claim it generates verification questions
that are grounded in the source sentences and written so that someone who only read the claim
cannot answer them well. Fourth, it tags each question with a Bloom's cognitive level (Anderson and
Krathwohl, 2001).

Backends sit behind one interface. This first slice runs on the local open-source model (Llama 3.1
8B through Ollama), which is the basis for Backend B. The commercial Backend A plugs into the same
interface, and the backend used is recorded with every guide, so the two can be compared directly.
The comparison between the two backends is the core commercial-versus-local research question, and
Chapter 8 carries it out.

## 7.3 An example

Run on a flagged essay that compares three literary texts, the system pulled out four claims and
wrote grounded questions for each. Two of them give the flavour.

One claim was that the three texts use language differently (metaphor and symbolism in the poetry,
dramatic intensity in the drama, subtle nuance in the prose), tied to two specific sentences in the
submission. For it the system asked: "What specific features of Shakespeare's dialogue lead you to
describe it as having dramatic intensity?" and "How does Austen's subtle and nuanced prose style
allow her to explore complex social issues in a way that might be less effective with more overt
language?" Answering either one takes actual reasoning about the texts. Repeating the claim back
would not be enough.

Another claim was that the texts share themes of love and personal growth, tied to a single
sentence. Here the system asked for an example from one of the texts that links personal growth to
relationships. Producing such an example is only possible for a student who genuinely engaged with
the texts. The full guide, with every claim, its exact source sentences, and the Bloom's levels, is
saved as `outputs/verification_guides/3108a_ai.md`.

## 7.4 Stand-ins in the first slice

The slice above was deliberately a thin path through the remaining pipeline, with three declared
stand-ins. They are worth recording here because the rest of this chapter replaces them one by one.
The claim extraction was prompted rather than the trained argument miner planned in the scope; the
trained miner arrives in Section 7.6. The Bloom's tag was a transparent keyword heuristic, replaced
by the trained classifier in Section 7.5. And the questions themselves were not yet evaluated; the
discrimination simulation that turns them into measured results is the subject of Chapter 8, where
the commercial backend also joins the comparison. One decision was still open at this point,
whether the local backend should be the full Llama 3 8B with QLoRA or a smaller model that fits the
laptop. My supervisor and I settled it on the evidence of the fit probes, and Section 7.8 picks it
up.

## 7.5 The Bloom's classifier

The keyword heuristic above was always a placeholder, and component 5 of the scope is a trained
classifier. I fine-tuned BERT-base (Devlin et al., 2019) on the Bloom-labelled subset of EduQG
(Hadifar et al., 2022). That subset holds 903 questions carrying a cognitive level, and only four
levels occur (remember, understand, apply, analyse), in a heavily skewed distribution (660, 114,
110 and 19 examples respectively). Training used class weights against the imbalance, stratified
splits (632 train, 135 validation, 136 test), and six epochs. The keyword heuristic was scored on
the same held-out test split as the baseline.

The trained model roughly doubles the heuristic: macro-F1 0.31 against 0.16, accuracy 0.57 against
0.26 (Figure 7.1). The per-class breakdown is less comfortable. BERT is reliable on the majority
class (remember, F1 0.76) and moderate on understand (0.38), but it cannot learn apply (0.09) or
analyse (0.0) from 110 and 19 examples. Coarsening the same predictions into the standard
lower-order versus higher-order split does not rescue the minority side either. Higher-order F1 is
0.23 for BERT and 0.17 for the heuristic, and the heuristic's apparently high binary accuracy
(0.86) comes from majority-class bias, since predicting lower-order for everything scores about
0.85 on this test set.

![Figure 7.1: Bloom's-level classification on the EduQG test split. The trained BERT-base doubles the keyword heuristic on macro-F1 (0.31 vs 0.16), but neither model can learn the two smallest classes from 110 and 19 examples.](../figures/fig_bloom_classifier.png)

The component works and clearly beats the transparent baseline, so it replaces the heuristic in the
pipeline. The bottleneck is the labelled data, not the model. In EduQG, 73 percent of the labels
sit at the lowest level, so a classifier that has to recognise higher-order questions needs either
more labelled examples of them or a coarser, better-populated label scheme. For the guide's
quality-control purpose this means Bloom's tags on remember and understand questions can be
trusted, while tags on the higher levels should be treated as suggestions until the label supply
improves. Both the model and the full metrics are saved
(`models/bloom_classifier/`, `outputs/bloom_classifier.json`).

## 7.6 The trained argument miner

The prompted claim extraction above was the declared stand-in for component 3, and the trained
version now exists. I downloaded the Persuasive Essays 2.0 corpus from its official repository (402
essays, BRAT annotations for major claims, claims, and premises, with the corpus's own 322/80
train/test split) and fine-tuned DeBERTa-v3-base as a BIO token classifier over the three component
types. Sequences are paragraphs. No annotated component crosses a paragraph boundary, and the
longest paragraph fits inside 256 subwords, so nothing is truncated. Evaluation is strict
span-level (exact boundary and exact type) with seqeval on the held-out test essays.

The result is a micro span-F1 of 0.63: premises 0.72, major claims 0.54, claims 0.44 (Figure 7.2).
Strict matching is a hard yardstick, and the per-class order is the expected one, since claim
boundaries are the classic ambiguity in this corpus. Dedicated argument-mining architectures with
CRF decoding or joint relation modelling report higher figures (Pietron et al., 2024), so this is a
working first version some way short of the state of the art, and I report it as such. For the
pipeline it opened a design choice, settled with my supervisor and implemented in Section 7.9. The
trained extractor finds spans in the student's own words. The prompted extractor produces readable
claim paraphrases with sentence citations. The guide combines them, using the spans for provenance
and the prompted paraphrases for phrasing. The model and metrics are saved
(`models/claim_extractor/`, `outputs/claim_extractor.json`).

![Figure 7.2: Argument-component extraction on the official Persuasive Essays test set, strict span-level F1 per class. Premises are learned well; claim boundaries are the hard case, matching the corpus literature.](../figures/fig_claim_extractor.png)

The component's second half, pairwise relation classification, is also built
(`src/argument_mining/train_relation_classifier.py`, `outputs/relation_classifier.json`). Following
the corpus's own structure, candidate pairs are ordered pairs of gold components within one
paragraph. DeBERTa reads the source and target components together and decides supports, attacks,
or no direct link, trained with class weights on the official split. The results divide along the
data's imbalance (Figure 7.3). Supports-links are learned well, F1 0.75 with recall 0.80, which
sits where the corpus literature puts link identification when component boundaries are given.
Unlinked pairs score 0.95. Attacks are not learned at all (F1 0.0), and the cause is the label
supply: attack relations are 0.7 percent of candidate pairs, about the same starvation that capped
the Bloom classifier's smallest classes. The macro-F1 of 0.57 against a 0.30 majority baseline is
therefore fairly computed but slightly misleading in both directions, pulled down by a class with
almost no training signal and pulled up by an easy one. The supports-F1 is the informative number. For the guide, the
practical use is ordering. Knowing which premises support which claim tells the lecturer which
evidence to probe first. The question generator itself only needs the claims, so this completes the
scope's argument-mining specification without changing the pipeline's behaviour.

![Figure 7.3: Relation classification on the official Persuasive Essays test split, per class. Supports-links reach F1 0.75 with gold components; attack relations, 0.7 percent of pairs, are unlearnable from this corpus, the same label-starvation pattern as the Bloom classifier's smallest classes.](../figures/fig_relation_classifier.png)

## 7.7 The assembled guide

With the classifier in place, the output assembler (`src/pipeline/assemble_guide.py`) produces the
Verification Interview Guide, the document a lecturer can take into a conversation and the reason
the pipeline exists. For a submission it runs the trained detector live and reports the probability
with its limits stated in plain language (the in-domain F1, the out-of-domain false-positive risk,
and the instruction to treat the score as a reason to talk, never as proof). It lists the
SHAP-validated drivers of the decision in lecturer-readable terms. It presents each extracted claim
quoted to its exact source sentences. It re-tags every question with the trained Bloom classifier
and marks the levels the classifier is weak on as advisory. It closes with a suggested three-level
rubric for reading the student's answers. The guide renders to Markdown, Word, and PDF
(`outputs/verification_guides/3108a_ai_guide.pdf` is the worked example, generated end to end with
live models).

One design note runs through the document. The first page frames the guide as evidence for a
conversation, not an accusation, and the fairness results of Chapter 6 made that framing necessary.
This was the assembler's first complete version, running the transformer detector and the prompted
claims. Section 7.9 returns to it once the trained argument miner and the fine-tuned backend are in
place and reports the fully integrated guide.

## 7.8 Fine-tuning the local backend

Section 7.4 left one question open for my supervisor: whether Backend B, the open-source side of
the core comparison, should be the full Llama 3 8B with QLoRA or a smaller model that fits the
laptop. The fit probes settled the practical half of it. In 4-bit with a LoRA adapter the 8B loads
at 5.3 GB but only trains by spilling past the physical 8 GB into system memory, at 214 seconds per
step for a 1024-token sequence. Qwen2.5 3B trains inside VRAM at 2.8 seconds per step with about
3 GB of headroom (`outputs/qlora_fit_probe.json`). On the strength of those probes my supervisor
signed off on Qwen2.5 3B as Backend B on 3 July 2026, so the fine-tuning experiment runs on the 3B.

I fine-tuned Qwen2.5-3B-Instruct (Yang et al., 2024) with QLoRA (Dettmers et al., 2023;
`src/question_gen/finetune_qg.py`): 4-bit NF4 base, a LoRA adapter (rank 16; Hu et al., 2022) on
the attention projections, gradient checkpointing and a paged 8-bit
optimiser, batch 1 with gradient accumulation of 8, two epochs over 2,600 passage-to-question pairs
from EduQG. The prompt tokens are masked in the loss, so the model learns only to produce the
question. Training finished in under an hour on the 4060 at a final loss of 1.42, and the adapter
is saved to `models/qg_finetune_qwen3b/`.

The evaluation is built to isolate the fine-tuning effect and nothing else
(`src/question_gen/eval_finetune.py`). One fixed set of 18 claims was extracted once from six
flagged essays. The base Qwen 3B and the fine-tuned Qwen 3B each wrote verification questions for
those same claims, and both sets were scored by the same discrimination simulation used in Chapter
8. The only thing that changes between the two conditions is the question-writing model, so any
difference comes from the adapter. The three phases never share the 8 GB. Claims are extracted with
the local model through Ollama, which is then unloaded. Each transformer model generates in turn,
and scoring loads the embedding model last.

On the raw metric the fine-tune looks like a clear success. The base model's questions score a mean
discrimination of 0.033 (95 percent CI 0.015 to 0.051); the fine-tuned model's score 0.154 (95
percent CI 0.085 to 0.227), with a Mann-Whitney p of 0.007 and confidence intervals that do not
overlap (Figure 7.3). Taken at face value that is roughly a five-fold gain, and the same 0.15 shows
up again at essay scale in Section 8.8.

![Figure 7.4: The QLoRA fine-tune of Qwen2.5 3B against its own base model on the same 18 claims, mean discrimination with 95 percent bootstrap intervals. The raw score rises from 0.033 to 0.154, but the quality audit in Section 8.9 shows this rise to be an artifact of degenerate output rather than better questions.](../figures/fig_finetune_eval.png)

It nearly went into the write-up as the headline result of the project. Then I read the questions
themselves. The fine-tuned model had overfit the format of its training
data. EduQG is largely a multiple-choice corpus, and the adapter learned to emit multiple-choice
stems: about 95 percent of its questions are strings like "Which of the following is correct?" or
"Which of the following is not a reason why Peugeot is successful?", with no options ever supplied,
and a handful are raw JSON fragments leaking from the prompt. These are not verification questions
at all. A student could not answer them, and a lecturer could not use them. The full audit across
all four models is in Section 8.9, where the base 3B, the Llama 8B and the commercial model each
produce zero degenerate questions and the fine-tuned model produces almost only degenerate ones.

The empty stems are also what drove the score up. The literal string "Which of the
following is correct?" scores a mean discrimination of 0.44, higher than any real question from any
model. A contentless question defeats the simulation. The source-aware and the source-blind
answerer both receive a stem with nothing to answer, their responses diverge more or less at
random, and the aware-minus-blind gap the metric reports reflects that randomness rather than
anything a real student would need the essay for. So the 0.154 measures how empty the fine-tuned
model's questions are, not how good they are. The conclusion is the opposite of what the raw number
suggests. The QLoRA fine-tune on EduQG failed as a question generator for this task.

I keep this in the dissertation instead of deleting it because the failure is informative. It shows
that the format of the training data matters more than its size. Two epochs on an overwhelmingly
multiple-choice corpus was enough to collapse a capable instruct model into producing
multiple-choice stems. More data of the same kind would not help; the fix is the right data, a
question-generation set of open-ended questions (or the verification-style prompts themselves), and
that is the clear next experiment for Backend B. The failure also exposes a real weakness in the
discrimination simulation. The measure assumes well-formed questions and can be gamed by degenerate
ones, so it needs a well-formedness guard (a simple filter, or the Bloom classifier, rejecting
non-questions) before its scores are trusted. This kind of failure is why the evaluation plan never
rested on a single automatic metric, and Section 8.9 turns the episode into that methodological
point.

## 7.9 The integrated guide

Sections 7.5 to 7.8 replaced the first slice's stand-ins one at a time. This section puts the
trained pieces together into the final guide, in the form my supervisor approved at our fifth
meeting (`src/question_gen/integrated_guide.py`, `src/pipeline/assemble_guide.py`). A submission
now flows through five trained models. The prompted extractor supplies each claim in readable
phrasing with sentence citations. The trained argument miner of Section 7.6 supplies the provenance
in the student's own words: each claim carries the verbatim major-claim, claim and premise spans
the miner found in its cited sentences, labelled by role. This is the "spans for provenance, prompt
for phrasing" design in one place, and it is stronger than either extractor alone, since the
lecturer sees both a sentence they can read and the exact words the model keyed on. The questions
are written by the v3 fine-tuned local backend, filtered through the well-formedness gate so
nothing degenerate reaches the page (the guide records how many, if any, were dropped), and tagged
with the trained Bloom classifier, with the levels the classifier is weak on marked advisory. The
detector at the top of the guide is the hybrid of Section 6.7, and it reports its component views.
On the worked submission it scores 0.957, noticeably calmer than the transformer's own 0.9996,
because the style half pulls the over-confident transformer back. A lecturer facing a possible
false positive would want that behaviour.

The document itself is organised for its reader, a lecturer with no time to prepare
(Figure 7.5). It opens with the framing the fairness results of Chapter 6 called for, that the
guide is there to support a conversation with the student, not to make an accusation. It then gives
the detection verdict with its limits in plain language, the SHAP-validated drivers of the decision
in lecturer terms, the claims with their two-way provenance and grounded questions, and a
three-level rubric for reading the answers. It renders to Markdown, Word and PDF, generated end to
end with live models on the 8 GB laptop (`outputs/verification_guides/3108a_ai_guide.pdf` is the
worked example). Nothing on the page is a mock-up; every element comes from a trained component of
this project. The pipeline as a whole now works, not just the individual parts.

![Figure 7.5: The first page of the assembled Verification Interview Guide, generated end to end with live models. The detection verdict is the hybrid detector with its component views; the claims that follow carry both readable phrasing and the trained argument miner's verbatim spans; the questions are written by the fine-tuned local backend and tagged by the trained Bloom classifier. The first page presents the guide as support for a conversation with the student rather than an accusation.](../figures/fig_guide_page1.png)
