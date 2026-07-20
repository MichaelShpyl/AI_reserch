# Chapter 8: Evaluating the questions (discrimination simulation)

## 8.1 The idea

A good verification question is one that a student who understands their own work can answer and a
student who does not cannot. I evaluate that without humans or an LLM judge, using a discrimination
simulation (`src/evaluation/discrimination_sim.py`). For each question, one model answers it WITH
the source passage from the essay (the "context-aware" answer) and another answers it with the
question ONLY (the "context-blind" answer). I then measure how close each answer is to the source,
using embedding similarity (nomic-embed-text; Nussbaum et al., 2024), and take the gap, aware
minus blind, as the discrimination score. A high gap means the question genuinely needs the source.
A near-zero gap means it can be answered without it. Everything runs on the local model and local
embeddings, so the evaluation needs no API key and no judge.

## 8.2 The first result

I expected my claim-grounded questions to discriminate well. They did not. On the flagged essay,
the claim-grounded questions scored a mean discrimination of about 0.05 (95% confidence interval
[0.01, 0.10]), while a set of generic essay questions ("what is your main argument", "what evidence
most supports your conclusion", "explain your reasoning in your own words") scored about 0.31
([0.27, 0.36]). The intervals do not overlap, so the difference is real, and it is the opposite of
what I assumed (Figure 8.1).

![Figure 8.1: Discrimination simulation. Claim-grounded questions (left) cluster low; generic essay questions (right) discriminate much more. The gap is the source-aware minus source-blind answer similarity, with 95% confidence intervals.](../figures/fig_discrimination_sim.png)

## 8.3 Why the generic questions score higher

The reason is the context-blind model. It is a large language model with broad world knowledge, so a
question that names specific content, "what features make Shakespeare's dialogue intense", is
answerable from general knowledge without ever reading this student's essay. The source barely
changes the answer, so the discrimination is low. A generic question like "what is the main argument
of your essay" is different. The blind model has no idea what this particular student argued, so its
answer is vague, while the source-aware model answers about the actual essay. The source matters a
lot, so the discrimination is high.

So the simulation really measures essay-specificity, whether answering the question requires this
student's own essay or only knowledge of the subject. That reframes what a good verification
question is. It should force the student to reproduce their own argument, evidence, and choices,
rather than recite facts about the topic that anyone knowledgeable could supply. When I rewrote the
question-generation prompt to ask for the student's own reasoning and chosen evidence rather than
general facts, the grounded score did improve, from about 0.03 to about 0.05, but it stayed well
below the generic questions, because a specific question still tends to name the content and hand a
knowledgeable answerer most of what it needs.

## 8.4 A caveat about the stand-in

The context-blind model is a strong stand-in, with far more world knowledge than a student who used
AI without understanding the work. A low discrimination score therefore does not mean a question is
useless in practice. It means a knowledgeable person who never read the essay can answer it. The
simulation is a conservative, hard test of essay-specificity. A real student who did not understand
their submission would do worse than the blind model on the grounded questions, so the true picture
sits somewhere between the two lines in Figure 8.1.

## 8.5 What this changes

This is the first measured result for the question generation, and it is more useful as a finding
than a score. The strongest verification questions make the student retrieve and justify their own
essay: the argument, the specific evidence they chose, and how their points connect. Questions that
restate subject facts do not do this. The next iteration of the generator will blend the claim
grounding (so each question is tied to a real passage) with this retrieval-forcing style (so the
answer cannot come from general knowledge). The remaining evaluation step is the supplementary
LLM-as-judge rubric with cross-model agreement as a second view. The two steps promised here
earlier, scaling the simulation across many essays and bringing in the commercial backend, are
reported next.

## 8.6 Commercial versus local: the first scaled comparison

With the simulation in place, I ran the comparison at the centre of the research question. ATU could
not provide API access or credits, so the commercial backend uses a free-tier Google Gemini API,
plugged in behind the same interface as the local model. The local backend is the same Llama 3.1 8B
used throughout. One disclosure on the commercial arm: free-tier quota limits forced a switch of
Gemini endpoint partway through (four essays ran on `gemini-2.5-flash`, ten on
`gemini-flash-latest`), so the commercial arm is one provider but not one fixed model snapshot. For
each essay in a seeded random sample, each backend generated a Verification Interview Guide (three
claims, grounded questions with sentence-level provenance), and every question from both backends was
scored by the same local discrimination simulation, so what varies is which model wrote the questions.

A small pilot on the single flagged essay from Section 8.2 came first. There, both backends' grounded
questions had discrimination intervals that included zero (local 0.032 with [-0.005, 0.083];
commercial 0.022 with [-0.002, 0.048]), too little signal to separate anything. The scaled run below
is the real comparison.

Fifteen essays were sampled; fourteen produced a scored guide from both backends (one essay was
excluded because claim extraction returned an empty guide; one commercial run was rate-limited for a
day and completed on retry), giving each backend 126 questions. The pooled means are local 0.042
with a 95% interval of [0.030, 0.055] against commercial 0.078 with [0.058, 0.098]. Because the same
essays sit under both backends, the fair test is paired. The commercial backend scores higher on 10
of the 14 essays. The mean paired difference is 0.036, with a bootstrap interval of [0.008, 0.067];
the paired t-test gives p = 0.040 and the Wilcoxon signed-rank p = 0.030. Unlike in the pilot, both
pooled means sit clearly above zero, so at this scale the grounded questions do measurably need the
source (Figure 8.2).

![Figure 8.2: Commercial versus local question generation, balanced across the same 14 essays. Left: per-essay mean discrimination, paired. Right: pooled means with bootstrap 95% confidence intervals; the dashed line is the generic-question baseline.](../figures/fig_backend_comparison_batch.png)

On this measure the commercial backend holds a small but statistically significant advantage, and
the local model remains competitive in absolute terms. The gap is about 0.04 against a
generic-question baseline of 0.30 with a 95% interval of [0.28, 0.32], now re-measured on the same
fourteen essays rather than taken from the pilot (84 generic questions; the pilot's single-essay
value of 0.31 barely moved). And the local number comes from a base model that has not yet been
fine-tuned. The fine-tuning experiment will test whether the planned QLoRA step closes the gap. The
sample-size trajectory is worth recording. Nine essays showed no detectable difference, thirteen sat
on the boundary (t p = 0.053), and the full fourteen crossed it. A small-n snapshot could have
supported whichever story one preferred, so the dissertation reports the complete set with paired
tests. The caveats from earlier still apply: the discrimination measure is conservative, both
backends sit well below the generic-question baseline, and the commercial arm is one provider's free
tier rather than a frontier model. Extending the comparison to more essays, more providers, and the
fine-tuned local backend is mechanical from here, since the framework saves per essay and resumes.

## 8.7 The LLM-as-judge study

The supplementary evaluation puts LLM judges over the questions with the four-dimension rubric
(relevance, specificity, discrimination potential, cognitive appropriateness, each 1 to 5), and the
plan has always been to validate judges rather than trust them. All three judges the scope calls for
have now rated the twelve grounded questions from the pilot guide: Gemini (2.5 Flash, free tier),
Claude (Opus 4.8) and GPT (4o-mini), the last two on a small capped spend approved by my supervisor
(`src/evaluation/llm_judge.py`, resumable across quota interruptions; results and the agreement
statistics in `outputs/llm_judge.json`).

The three judges do not tell one story. Gemini and GPT both sit at the ceiling: Gemini rates
everything 4.5 to 5.0 (mean 4.81) and GPT even tighter, 4.75 to 5.0 (mean 4.94), so their scores
mostly say the questions look well-formed. Claude behaves differently, using the scale from 2.5 to
4.5 (mean 3.67) and marking down the content-naming questions such as "how does the metaphor evoke
longing", which general knowledge can answer. Cross-model agreement, the scope's own validation
criterion, is poor as a result. Krippendorff's alpha across the three judges is -0.25, and no
pairwise rank correlation reaches significance (Spearman 0.20 to 0.45, all p > 0.13). With two
judges at a ceiling, there is no way for them to agree with a third that discriminates.

Anchoring to the objective measure settles what the disagreement leaves open. None of the three
judges correlates positively with the discrimination simulation on the same questions. Gemini's mean
rating sits at rho -0.14 (p = 0.67), Claude's at -0.30 (p = 0.34), and GPT's near-constant ratings
manage a significantly negative rho of -0.75 (p = 0.005). The few questions GPT rated below the
ceiling were among the ones that objectively discriminate best. With n = 12 the individual
correlations deserve caution (an earlier partial run of the first judge looked strongly negative at
seven questions and attenuated to nothing by twelve), but the pattern across three independent
providers is consistent. Rubric ratings track how good a question looks. They do not track how well
it separates someone who read the essay from someone who did not.

Twelve questions is a thin base for correlations, and Chapter 9 said so, so the study was rerun at
five times the size once the scaled comparison of Section 8.12 provided a large pool of scored,
well-formed questions to draw from. Sixty questions were sampled evenly from its three fully
well-formed arms (twenty each from the local 8B, the base 3B and the fine-tuned v3, one seeded
sample that every judge rates identically), and anchored to the same run's per-question simulation
scores. All three judges completed the set, the free-tier one across several quota
windows. The larger sample settles both open
points. First, the anti-correlation is not a small-sample fluke. At sixty questions Claude's mean
rating sits at rho -0.34 against the simulation (p = 0.010), GPT's at -0.37 (p = 0.004) with its
own "discrimination potential" dimension the worst of all at -0.51 (p = 0.0001), and Gemini's at
-0.08 (p = 0.56); no judge correlates positively. Second, the disagreement between judges turns
out to be calibration, not noise. Every pair correlates in rank (Spearman 0.42 to 0.58, all
p < 0.001), while interval alpha stays negative (-0.14) because two judges sit near the ceiling
(Gemini mean 4.63, GPT 4.87), a full point above Claude (3.56). The three judges share a stable
notion of what a good question looks like, and that notion points away from what discriminates.
At arm level the cost is concrete: the simulation ranks the sampled arms v3 first (0.071 against
0.039 and 0.017), yet Claude ranks v3 second, and GPT and Gemini both rank it below the plain 8B.
A backend selection delegated to this panel
would have rejected the backend the pipeline ships.

For the evaluation design this is the conclusion the plan anticipated, now with evidence behind it
instead of caution. Judge scores in this setup cannot certify question quality on their own. The
judges either sit at a ceiling or, when they do discriminate, rank against the objective measure.
The judge-free discrimination simulation carries the empirical weight, and the panel's value is
this negative result. It is also a second, independent instance of the lesson from Section 8.9: a
plausible-looking automatic score, whether a judge rating or a discrimination number, cannot be
trusted until it is checked against something that cannot be gamed.

## 8.8 Like-for-like: four question writers on one fixed claim set

Section 8.6 compared the backends but let each one run its whole pipeline, so the model that chose the
claims differed between arms as well as the model that wrote the questions. That leaves an ambiguity.
The commercial edge there could come from writing better questions, or simply from Gemini selecting
claims that happen to discriminate more. This section removes the ambiguity. One claim set is fixed
per essay, extracted once with the neutral local extractor (Llama 3.1 8B), and four question writers
answer the same claims: Llama 3.1 8B, free-tier Gemini, the base Qwen2.5 3B, and the QLoRA-fine-tuned
Qwen2.5 3B. Every question is scored by the same discrimination simulation, so the only variable
across the four arms is the model that writes the question (`src/evaluation/likeforlike_4way.py`,
`outputs/likeforlike_4way.json`). The 14 essays are the balanced set from Section 8.6.

On the discrimination score alone the ordering is dramatic (Figure 8.3). The fine-tuned 3B reaches a
pooled mean of 0.153 (95% CI [0.105, 0.202]), far above the base 3B at 0.041 [0.028, 0.055], Llama 8B
at 0.031 [0.016, 0.045], and commercial Gemini at 0.024 [0.008, 0.041]. Its paired lead is 0.138 over
commercial (p < 0.001) and 0.123 over its own base (p < 0.001), and it reproduces the 0.154 from
Section 7.8. Taken at face value, the locally fine-tuned open model beats the commercial one by a
wide margin.

The picture changes once the questions themselves are read, and Section 8.9 covers why. About 95
percent of the fine-tuned model's questions are degenerate multiple-choice stems that game the
simulation, so the 0.153 measures how empty they are rather than how good. I set the fine-tuned arm
aside for the interpretation here and compare the three models whose questions are well-formed.

![Figure 8.3: The four question writers on one fixed claim set per essay, same scorer, pooled mean discrimination with 95% bootstrap intervals, each bar labelled with its question count and essay coverage. The fine-tuned 3B posts the highest raw score, but Section 8.9 shows that score to be an artifact of degenerate questions; the meaningful comparison is among the other three arms.](../figures/fig_likeforlike_4way.png)

Among those three the picture is modest and close. On the fixed claims the three models are
statistically indistinguishable: base Qwen 3B at 0.041, Llama 8B at 0.031, and Gemini at 0.024, all low
and with overlapping intervals. The run initially covered only 9 of 14 essays on the commercial arm
because free-tier quota ran out mid-run; the framework resumed when quota returned, and the numbers
here are the complete set (105 commercial questions across all 14 essays, a few claims lost to
rate-limiting inside otherwise-covered essays). The complete set settles what the partial run could
only suggest. In Section 8.6, with each backend choosing its own claims, Gemini held a significant edge
over Llama 8B (0.078 to 0.042, paired p = 0.040). Here, on identical claims, that edge is gone
(paired difference -0.005, p = 0.62, Gemini higher on seven of fourteen essays, a coin flip). The
Section 8.6 advantage does not survive fixing the claims. I take this to mean the advantage came
substantially from claim selection, with Gemini choosing easier-to-discriminate claims when allowed
to, and not from writing stronger questions on the same ones. The design here cannot rule out every
alternative, but the disappearance is now measured on the full sample rather than hypothesised. The
firm conclusion of this section is narrower than the one I first drafted, and arguably more useful.
When the task is fixed, small open models running on a laptop write questions as discriminative as a
commercial model's. The second half of the research question, whether a fine-tuned local model can
beat a commercial one, stays open at this point, because the fine-tune that was supposed to answer
it produced unusable questions.

## 8.9 A quality audit of the generated questions

Section 8.8 leaned on a number without reading what it scored. This section reads the questions, and
the gap between the two is the biggest methodological lesson in the project
(`src/evaluation/qg_quality_audit.py`, `outputs/qg_quality_audit.json`). Every question from all four
arms was checked with a transparent rule for degeneracy: multiple-choice stems (which never carry
options here), raw JSON fragments leaking from the prompt, and contentless "which is correct" forms.

Figure 8.4 shows the result. The Llama 8B, base 3B and commercial arms produce no degenerate
questions at all. The fine-tuned 3B produces almost nothing else: 59 of its 62 questions, just over 95
percent, are stems like "Which of the following is correct?" or "Which of the following is not a reason
why Peugeot is successful?", with no options supplied. The fine-tune, two epochs on the largely
multiple-choice EduQG corpus, had overfit the format of the training data and forgotten how to write an
open question. None of these are usable in a verification interview.

The second panel matters for the evaluation method. The empty stems do not just pass the
discrimination simulation, they outscore the real questions. The single literal string "Which of the
following is correct?", scored eleven times across the run, averages a discrimination of 0.44, higher
than the mean of every model's real questions. The reason is structural. A contentless question gives
the source-aware answerer and the source-blind answerer nothing to latch onto, so their answers
wander apart for reasons unrelated to the source, and the aware-minus-blind gap the metric rewards is
large. The simulation assumes the questions it scores are well-formed. A degenerate generator can
exploit that assumption and post an arbitrarily high score.

![Figure 8.4: Auditing question quality behind the discrimination score. Left: the share of degenerate, unusable questions per model. Right: the mean discrimination of each model's questions against the score of the contentless stem "Which of the following is correct?" (dashed line), which out-scores every real question writer.](../figures/fig_qg_quality_audit.png)

Two conclusions follow. For Backend B, the format of the training data dominates, so fixing the
fine-tune means the right data rather than more data: an open-ended question-generation set
(SQuAD-style questions, or the pipeline's own verification prompts distilled into training pairs)
instead of a multiple-choice corpus. Section 8.10 runs that experiment. For the evaluation, the
judge-free discrimination simulation, valuable as it is for well-formed questions, needs a
well-formedness gate in front of it, and no automatic score should be read without inspecting the
text behind it. That gate is now part of the pipeline (`src/question_gen/wellformed.py`). The same
transparent rule the audit uses is applied wherever questions leave the system, the guide builder
drops malformed questions before a lecturer can see them and records how many it dropped, and every
fine-tune evaluation reports its degeneracy rate next to its score. The evaluation plan required
reading outputs before trusting scores, and that requirement is why this failure was caught here
rather than in a viva.

## 8.10 Fixing the fine-tune: v2 on open-ended data

Section 8.9 made a testable prediction: if the degeneracy was caused by EduQG's multiple-choice format,
then changing only the training data to open-ended questions should fix it. So I re-ran the fine-tune
with everything held identical except the data, swapping EduQG for SQuAD (Rajpurkar et al.,
2016), whose questions are
open-ended and passage-grounded (`src/question_gen/finetune_qg_v2.py`, adapter in
`models/qg_finetune_qwen3b_v2/`). The evaluation repeats the Section 7.8 design on the same 18 claims,
base against the new adapter, and it audits degeneracy before reading any score
(`src/question_gen/eval_qg_v2.py`, `outputs/qg_v2_eval.json`).

The prediction held. The v2 model's questions are 0 percent degenerate against v1's 95, so the cause
was the format of the training data and not fine-tuning as such (Figure 8.5). This time the
discrimination gain is real rather than an artifact, because the questions are well-formed. On
the same claims, v2 scores a mean discrimination of 0.102 (95% CI [0.060, 0.144]) against the base
model's 0.027 ([0.015, 0.040]), a Mann-Whitney p of 0.0003. The contrast with v1 matters here. v1's
higher 0.154 came from empty multiple-choice stems gaming the metric. v2's lower 0.102 comes from
genuine questions that need the source more than the base model's do.

![Figure 8.5: The fine-tune, corrected. Base 3B, the v1 EduQG adapter (hatched: its score is an artifact of 95 percent degenerate questions), and the v2 SQuAD adapter, mean discrimination on the same 18 claims with 95 percent intervals and each bar's degeneracy rate. v2 is a real gain over base (p = 0.0003) on well-formed questions, but modest and still below the generic-question baseline.](../figures/fig_qg_v2_eval.png)

v2 is a partial success, not a triumph, and I hold the interpretation to what was measured. Its
0.102 is well under the generic-question baseline of about 0.30, and reading its questions shows
why. SQuAD trains a factual style ("What is the purpose of judicial review?"), and factual questions
are partly answerable from general knowledge, the property Section 8.3 showed the simulation
penalises. Two smaller points. v2 needed a cleaned output parser, because the adapter sometimes
appended JSON fragments after the question that an earlier extractor would have kept; the parser now
takes the first well-formed question and drops the rest. And the degeneracy audit ran before the
score, as the standing rule now requires. Fine-tuning a small local model does help once the
training data has the right form. But SQuAD's factual questions are not the ideal signal for
verification, so the next step is a training set of reasoning-demanding questions, distilled from
the pipeline's own verification prompts. That is the v3 experiment of the next section. At this
stage Backend B was a diagnosed and partly fixed component, and it is reported that way rather than
as the false win of Section 7.8's first draft.

## 8.11 Completing the data-format experiment: v3 on the pipeline's own style

The third leg trains on data in the format the task itself uses. A training set of 2,604
verification questions was distilled from the pipeline itself (`src/question_gen/build_v3_dataset.py`):
the local Llama 3.1 8B ran the production claim-extraction and question-writing prompts over 307
training essays, and each well-formed question became one passage-to-question pair, with the passage
being the claim plus its source sentences, which is what the model sees at inference. The training
set needed some care. All fifteen evaluation essays were excluded from the training pool, so nothing
the fine-tunes are tested on was ever trained on. Only questions passing the well-formedness gate
became training targets, since a noisy teacher would poison the experiment. And the teacher is local
rather than a commercial API because Llama's licence permits training other models on its output.
Training kept every setting identical to v1 and v2 once again
(`src/question_gen/finetune_qg_v3.py`); the final loss of 0.72, against 1.42 for v1 and 1.27 for v2,
already suggests that on-format data is easier to learn.

The evaluation follows the standing protocol: degeneracy audited before any score, same 18 fixed
claims (`src/question_gen/eval_qg_v3.py`, `outputs/qg_v3_eval.json`). v3 is fully well-formed, zero
degenerate questions out of 42, and it answers the request for three questions per claim far better
than v2 did (2.3 per claim against 1.3). Its output reads like the pipeline's intended style:
"How did you decide to use the example of a decision made by a public authority as evidence", "what
role does the concept of authority play in your argument, and how did you connect it", "in what ways
does your discussion relate to the broader themes of your essay". On the metric it scores a mean
discrimination of 0.064 (95% CI [0.033, 0.096]), roughly two and a half times the base model's 0.027
([0.015, 0.040]), with the intervals barely brushing. That also doubles the 8B teacher's own score
in the fixed-claim comparison (0.031, Section 8.8), so the 3B student overtook the model that
taught it (Figure 8.6).

![Figure 8.6: The data-format experiment complete on the same 18 claims: base model, v1 (multiple-choice EduQG, hatched because its score is a degeneracy artifact), v2 (factual SQuAD) and v3 (verification-style self-distillation), with each bar's degeneracy rate. Both open-ended formats produce fully well-formed output and real gains; only the multiple-choice format collapses.](../figures/fig_qg_v3_eval.png)

One result does not fit a tidy ranking. v2 still posts the higher number (0.102 against 0.064), and
the reason matters more than the numbers themselves. The two adapters learned different trades.
v2 writes one terse factual question per claim; that style happens to suit the simulation, but "What
is the purpose of judicial review?" is not a verification question, since it never asks the student
about their own essay. v3 writes the questions the product actually needs, demanding the student's
reasoning, choices and connections, but in doing so it quotes the claim's content back, and Section
8.3 established that handing content to the source-blind answerer shrinks the measured gap. The
metric, taken alone, would therefore pick the adapter whose output is unfit for purpose. This is the
third time in this chapter that a single automatic number would have led to the wrong choice, after
the v1 artifact and the judge panel, and it is why the decision about Backend B is not delegated to
the metric. v3 becomes the working local backend because its output is well-formed, on-style and a
real improvement over base. v2's higher raw score is documented and explained rather than chased.

The completed experiment is the dissertation's answer on fine-tuning. The format of the training
data dominates: a multiple-choice corpus collapses the model, and either open-ended format fixes it.
Fine-tuning a 3B model on an 8 GB laptop reliably lifts it well clear of its base, whichever open
format is used, and self-distillation can lift a small student past its own larger teacher on the
target measure. No single automatic score, this one or a judge's, is allowed to settle a design
decision by itself in this project.

## 8.12 The comparison at scale: thirty essays

Chapter 9 names the comparison's sample size as a limitation, so the final experiment scales the
fixed-claim design from fourteen essays to thirty. The arms are v3 (the backend the pipeline
actually ships), the local 8B, the base 3B, and commercial Gemini
(`src/evaluation/likeforlike_scaled.py`, `outputs/likeforlike_scaled.json`). The protocol does not
change: one fixed claim set per essay from the neutral extractor, every writer answering the same
claims, one scorer, and the degeneracy rate reported next to every score. Contamination needed
guarding against at this size. The sixteen new essays come only from the pool that is neither an
evaluation essay nor one of the 307 essays that supplied v3's training questions. And the original
fourteen essays keep their claims and questions as first generated; nothing is regenerated. The run
scored 901 questions across the four arms. None is degenerate.

The headline result is the strongest in this dissertation (Figure 8.7). The fine-tuned v3 reaches a
mean discrimination of 0.085 (95% CI [0.071, 0.100]), more than double every other arm. Against its
own base model the paired difference is +0.046 (p < 0.0001, higher on 25 of 30 essays). Against the
commercial model, on the 29 essays both arms cover, it is +0.040 (p = 0.0003, higher on 24 of 29).
On this task, with the claims held fixed, a model fine-tuned on one consumer laptop writes
measurably better verification questions than the commercial free tier. The comparison also
replicates the earlier null. Commercial Gemini has no edge over the plain local 8B (paired
difference +0.003, p = 0.74, n = 29), which repeats the Section 8.8 finding that the commercial
advantage in Section 8.6 came from claim selection, not question writing.

I trust this result for a specific reason, having retracted a version of it once before. The v1
"win" in Section 8.9 came from degenerate output that gamed the metric. Here every one of v3's 184
questions passes the well-formedness gate, and reading them confirms they are ordinary verification
questions in the intended style. The score and the output quality agree this time. Section 8.9
established that agreement as the condition for believing any number this pipeline produces.

![Figure 8.7: The fixed-claim comparison at thirty essays with the working v3 backend. All 901 questions are well-formed. v3 clears the base 3B (p < 0.0001) and, on the 29 shared essays, the commercial model (p = 0.0003); the commercial-versus-8B null replicates. Coverage is printed on each bar: the free-tier commercial arm covers 29 of 30 essays.](../figures/fig_likeforlike_scaled.png)

Two disclosures apply. The commercial arm is a free-tier flash model, not a frontier one, so the
claim is "a QLoRA fine-tune on an 8 GB laptop beats the free commercial tier at writing verification
questions", not "local models beat commercial models". And quota interruptions shaped the commercial
arm's coverage: it spans 29 of the 30 essays and mixes two versions of the same model, fourteen
essays on gemini-2.5-flash (105 questions) and fifteen on gemini-flash-latest (99 questions) across
two refills. The refilled questions are shorter (median 30 words against 43) and score twice as high
(0.050 against 0.024), so the mixture strengthens the commercial arm rather than the local side, and
v3 clears it anyway. With both
stated, the second half of the research question has its answer. A locally fine-tuned open model can
match, and in this comparison beat, the commercial backend it was allowed to compare against.

## 8.13 Closing the baseline gap by construction: the v4 experiment

One number survived every comparison in this chapter untouched: the generic-question baseline of
0.30, which no grounded writer approached. Section 8.3 diagnosed why. The four generic champions
give a context-blind reader nothing to answer from, while grounded questions name their claim's
content, and the naming is the leak the blind model exploits. If that diagnosis is right, the gap
should close by construction: write questions that stay anchored to one claim but never name what
the claim says. The lecturer loses nothing, because the guide prints the claim beside its
questions. The blind model loses everything.

v4 tests exactly that, with everything except the prompt held fixed against v3: the same local
teacher, the same 2,600-pair target, the same QLoRA settings and the same 18-claim evaluation
(`src/question_gen/build_v4_dataset.py`, `outputs/qg_v4_eval.json`). One addition proved
necessary before the build could start. In the smoke test the 8B teacher ignored the no-naming
instruction whenever the context tempted it, so the rule is enforced by a mechanical gate in the
same family as the well-formedness gate: a candidate question that shares a content word with the
claim, or a capitalized entity from the claim or source, is rejected before it can become
training data.

The result is the largest single movement in the fine-tune series (Figure 8.8). v4 reaches a mean
discrimination of 0.266 (95% CI [0.236, 0.296]), four times v3 and within reach of the 0.30
baseline, with zero degenerate questions. And because this series has been burned before, the
number was not believed until the output was read and measured three more ways. The questions are
real, answerable interview questions, not stubs: 51 of 54 are unique as strings, and 89 percent
stay content-free at inference (the model saw the gate only through its training data, and drifts
back to naming content in 6 of 54 questions). The honest cost is visible in the phrasing: the
output converges toward a small family of angles, with nine questions opening on the evidence
choice and seven on the connection to the wider argument. v4 closes the gap largely by converging
toward the generic form, and what it adds over the four generic champions is the per-claim
anchoring the guide is built on, the Bloom spread, and mild adaptation to each claim.

Two conclusions follow, one scientific and one practical. Scientifically, the baseline gap is now
explained: most of it was the price of naming content, paid by every grounded writer in this
chapter, and a 3B model reproduces the generic effect once that price is removed. Practically,
the engineering step that a deployment would need has now been taken and measured
(`src/question_gen/eval_qg_v4_gated.py`, `outputs/qg_v4_gated_eval.json`). With the content gate
applied at inference, rejecting and regenerating any question that names claim content, the
output becomes fully uniform: 100 percent content-free across all 54 questions with no fallbacks,
at a mean discrimination of 0.256 (95% CI [0.230, 0.281]), statistically indistinguishable from
the ungated 0.266, for a retry cost of seven extra generation rounds across the eighteen claims.
Nothing technical now blocks v4; the pipeline still ships v3 because the remaining question is
not an engineering one. Choosing between v3's specific, quotable questions and v4's
harder-to-fake ones is a judgement about what works in a real conversation, and the validation
study protocol now includes a blind v3-versus-v4 arm designed to answer exactly that with human
evidence.

![Figure 8.8: The fine-tune series on the fixed 18 claims, with the v4 result. Trained to anchor without naming, v4 reaches 0.266 against the generic baseline's 0.30, with zero degenerate output, 51 of 54 unique questions and 89 percent content-free behaviour at inference. The gap to the baseline was mostly the price of naming content.](../figures/fig_qg_v4.png)
