# Chapter 1: Introduction

## 1.1 Overview

This dissertation builds an explainable pipeline that helps a lecturer check the integrity of a
student submission without having to treat a detector's score as the final word. It does two
connected things. First it decides whether a piece of writing is likely AI-generated, and rather than
returning a bare percentage it shows which features of the writing drove that decision. Second, when
a submission is flagged, it reads the student's own argument, pulls out the specific claims they made,
and turns those claims into verification questions a lecturer can ask in a short conversation. The aim
is not to catch students out. It is to give the lecturer transparent, defensible evidence, tied to the
student's own words, that a fair decision can rest on.

Two design choices run through the whole system. Every stage is meant to be explainable, so a lecturer
can see and justify why a submission was flagged and where each question came from, instead of pointing
at a number they cannot account for. And the question generation is built twice, once with a commercial
large language model and once with a smaller open model that runs on a single laptop, so the project can
ask a question that matters to any institution: how close can a locally run, low-cost model get to an
expensive commercial one for this task (Oketch et al., 2025). That comparison is a central
contribution.

The intended user is a lecturer on a large module, somewhere between 150 and 300 students, who has
neither the time to prepare a fair set of verification questions for every flagged essay nor a good way
to defend a black-box score if a student challenges it. The system is aimed squarely at that person. It
is designed to save time, to treat students fairly, including students who do not write in English as a
first language and whom existing detectors tend to penalise (Liang et al., 2023), and to produce something a lecturer can stand over.

## 1.2 Background and context

Over the last few years generative AI tools have moved from a curiosity to something most students have
used at least once (Jin et al., 2024). Drafting an opening paragraph,
rephrasing an awkward sentence, or getting a hard idea explained is now a few seconds of typing into a
chatbot. That is genuinely useful for learning, but it has also put pressure on a basic academic
question: is this piece of work the student's own, and does it reflect their own understanding?

Universities responded quickly, and a market of AI-text detectors appeared almost as fast as the
writing tools did (Wu et al., 2025). Broadly these detectors fall into two families. Some are zero-shot
or statistical: they use a language model's own probabilities to judge how predictable a passage is, on
the assumption that machine text sits in smoother, more probable regions than human text (Mitchell et
al., 2023). Others are supervised: a transformer is fine-tuned on labelled human and AI examples and
then classifies new text (He et al., 2021). Commercial services usually combine such methods behind an
interface and return a single figure, an AI-likelihood percentage.

The trouble is not that these tools never work. It is that they were built fast, they are opaque, and
they are fragile in the situations that matter most. Their accuracy drops sharply on text they were not
trained for, and they can be evaded by simple paraphrasing (Krishna et al., 2023). More seriously for a university, they hand back a number with nothing behind it, and there
is growing evidence that the errors are not spread evenly: writing by non-native English speakers is
flagged as AI far more often than it should be (Liang et al., 2023). A tool
that is confident, unexplained, and unfair is a poor basis for an academic-integrity decision.

## 1.3 Problem statement

The core problem I keep returning to is that current detectors give a lecturer a percentage and stop
there. Someone who sees "82 percent AI" has no way to explain that number if the student pushes back,
and no real way to know whether it is even right for this particular essay. That is a weak and risky
position to accuse a person from, and the cost of being wrong falls on the student, whose record and
standing are at stake. A score with no reasons behind it cannot carry that weight.

There is a fairness problem on top of the opacity. Because the errors are biased, the students most
likely to be wrongly flagged are often those who are already disadvantaged, including people writing in
a second language (Liang et al., 2023). A system that quietly repeats that
bias, and cannot even show its reasoning, is worse than no system at all.

The deeper problem is that even a correct flag does not answer the question a lecturer actually cares
about. Knowing that text is probably AI-generated does not tell you whether the student understands the
material they handed in. One student might have used a tool heavily and still grasp the argument;
another might have written every word themselves and understood little. A flag is a suspicion about how a
text was produced, not evidence about what the student knows. What a lecturer needs, and what no detector
provides, is a fair way to move from that suspicion to a check of understanding.

Closing that distance is the problem this project takes on: make the detection transparent and
defensible, and then turn a flag into specific, source-grounded questions that test whether the student
can account for their own work.

## 1.4 Motivation

I am aiming this at the lecturer who runs a large module and has no time to sit with each flagged essay
and work out a fair way to check it. That person needs something they can defend, that treats students
fairly, and that saves them time rather than adding to the pile. The fairness side matters to me in
particular, because the research already shows these detectors can be biased against people who do not
write in English as a first language (Liang et al., 2023), and I do not want
to build something that quietly repeats that. A system that is transparent about its reasoning is also a
system whose mistakes can be seen and argued with, which is exactly what a fair process needs.

## 1.5 The verification gap

So there is a gap. On one side is detection, which is improving but stays a black box. On the other side
is the question that actually matters: does this student understand their own submission? Nothing I
found connects the two. Nothing takes a flag and turns it into
specific questions, drawn from the student's own claims, that a lecturer can ask in a short conversation
to see whether the understanding is there. That gap is what this project tries to fill.

## 1.6 Research question and objectives

The main question I am asking is this: how can an explainable AI pipeline be designed to support
lecturers in academic integrity verification, by combining transparent AI text detection with
argument-aware question generation, and how far can locally fine-tuned open-source models match
commercial ones at producing interpretable, defensible outputs for this task.

To get there the objectives are roughly:

- detect AI-written text in a way that can be explained, not just scored;
- show which words and features drove each decision, and test that those explanations are faithful;
- pull out the claims and evidence in a flagged essay and tie each one to its source;
- generate verification questions from those claims, and compare a commercial model against a local
  open-source one;
- label each question by cognitive level as a quality check; and
- evaluate the whole thing with objective measures rather than opinion.

## 1.7 Scope and boundaries

To keep this doable in the time I have, the scope is deliberately tight. The detector works with two
classes only, human and AI. I dropped the earlier idea of partial-AI classes because they overlap too
much to separate reliably; that may come back later as future work if the two-class version works well.
The pipeline has six parts (detection, explainability, argument mining, question generation, a Bloom's
level check, and the output guide), and the datasets are fixed in advance. There are no human
participants in the study, which keeps it clear of needing ethics approval and keeps the focus on the
system itself.

## 1.8 Contributions

What I think this project adds:

- a pipeline that goes from a detection flag to transparent, defensible evidence, instead of stopping
  at a score;
- verification questions that are tied back to the student's own writing, so a lecturer can see where
  each one came from;
- a direct comparison of a locally run open model against a commercial one for this task, which matters
  for any institution thinking about cost and control;
- an objective way to measure whether a question really tests understanding; and
- an honest look at how the detector treats native versus non-native writers.

## 1.9 Dissertation outline

The rest of the document is laid out as follows.

Chapter 2 reviews the recent literature the project draws on: AI-text detection and its benchmarks,
stylometric features, explainability and faithfulness, argument mining, question generation and its
evaluation, and the fairness evidence that motivates the design.

Chapter 3 sets out the detection methodology and the first results: the dataset, how the matched AI
essays were built, and the audit that found and removed a corpus artefact behind an initially perfect
score.

Chapter 4 describes the implementation of the components built so far, with the engineering decisions
that the 8 GB laptop budget forced.

Chapter 5 explains the detector's decisions and tests whether those explanations are faithful,
comparing token-level attributions on the transformer against SHAP over the stylometric features.

Chapter 6 tests robustness: how the detector transfers to generators it never saw and to other kinds
of text, and what its failures mean for fairness.

Chapter 7 presents the first slice of the core contribution: turning a flag into verification
questions drawn from the student's own claims, with sentence-level provenance and a Bloom's level on
each question.

Chapter 8 evaluates those questions with a judge-free discrimination simulation and reports the first
scaled comparison between the commercial and local backends.

The chapters that follow in the final document cover the remaining components (the trained argument
miner and the Bloom's classifier), the fine-tuned local backend and the completed
commercial-versus-local comparison, the assembled Verification Interview Guide, a discussion of what
the results mean and where the limits are, and the conclusion with future work.
