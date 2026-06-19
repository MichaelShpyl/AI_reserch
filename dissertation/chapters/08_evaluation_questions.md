# Chapter 8: Evaluating the questions (discrimination simulation)

> Draft note (delete before submission): rough first-person draft for me to rewrite in my own
> words. This is the project's primary, judge-free evaluation of question quality, and it produced
> a result I did not expect that changed how I think about good verification questions. No em
> dashes; my own wording for the final version.

## 8.1 The idea

A good verification question is one that a student who understands their own work can answer and a
student who does not cannot. I evaluate that without humans or an LLM judge, using a discrimination
simulation (`src/evaluation/discrimination_sim.py`). For each question, one model answers it WITH
the source passage from the essay (the "context-aware" answer) and another answers it with the
question ONLY (the "context-blind" answer). I then measure how close each answer is to the source,
using embedding similarity, and take the gap, aware minus blind, as the discrimination score. A high
gap means the question genuinely needs the source; a near-zero gap means it can be answered without
it. Everything runs on the local model and local embeddings, so the evaluation needs no API key and
no judge.

## 8.2 The result, which surprised me

I expected my claim-grounded questions to discriminate well. They did not. On the flagged essay,
the claim-grounded questions scored a mean discrimination of about 0.05 (95% confidence interval
[0.01, 0.10]), while a set of generic essay questions ("what is your main argument", "what evidence
most supports your conclusion", "explain your reasoning in your own words") scored about 0.31
([0.27, 0.36]). The intervals do not overlap, so the difference is real, and it is the opposite of
what I assumed (Figure 8.1).

![Figure 8.1: Discrimination simulation. Claim-grounded questions (left) cluster low; generic essay questions (right) discriminate much more. The gap is the source-aware minus source-blind answer similarity, with 95% confidence intervals.](../figures/fig_discrimination_sim.png)

## 8.3 Why, and what it means

The reason is the context-blind model. It is a large language model with broad world knowledge, so a
question that names specific content, "what features make Shakespeare's dialogue intense", is
answerable from general knowledge without ever reading this student's essay. The source barely
changes the answer, so the discrimination is low. A generic question like "what is the main argument
of your essay" is different: the blind model has no idea what this particular student argued, so its
answer is vague, while the source-aware model answers about the actual essay. The source matters a
lot, so the discrimination is high.

So the simulation is really measuring essay-specificity: does answering the question require this
student's own essay, or only knowledge of the subject. That reframes what a good verification
question is. It should force the student to reproduce their own argument, evidence, and choices,
rather than recite facts about the topic that anyone knowledgeable could supply. When I rewrote the
question-generation prompt to ask for the student's own reasoning and chosen evidence rather than
general facts, the grounded score did improve, from about 0.03 to about 0.05, but it stayed well
below the generic questions, because a specific question still tends to name the content and hand a
knowledgeable answerer most of what it needs.

## 8.4 An honest caveat about the stand-in

The context-blind model is a strong stand-in, with far more world knowledge than a student who used
AI without understanding the work. So a low discrimination score does not mean a question is useless
in practice; it means the question is answerable by a knowledgeable person who never read this essay.
The simulation is therefore a conservative, hard test of essay-specificity. A real student who did
not understand their submission would do worse than the blind model on the grounded questions, so the
true picture sits somewhere between the two lines in Figure 8.1.

## 8.5 What this changes

This is the first measured result for the question generation, and it is more useful as a finding
than a score. It says the strongest verification questions are the ones that make the student
retrieve and justify their own essay, its argument, the specific evidence they chose, and how their
points connect, rather than questions that restate subject facts. The next iteration of the
generator will blend the claim grounding (so each question is tied to a real passage) with this
retrieval-forcing style (so the answer cannot come from general knowledge). The next evaluation steps
are to run this simulation across many essays rather than one, to add the supplementary LLM-as-judge
rubric with cross-model agreement as a second view, and to bring in the commercial backend so the
commercial-versus-local comparison can be scored on the same discrimination measure.
