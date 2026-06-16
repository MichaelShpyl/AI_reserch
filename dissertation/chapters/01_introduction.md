# Chapter 1: Introduction

> Draft note (delete before submission): this is a rough first-person draft for me to
> rewrite in my own words. It is written at outline level, as Vini asked, so detailed
> tools and methods are left for the Methodology chapter once the model is built. The
> project is about detecting AI text, so the final wording here has to be my own.

## 1.1 Background and context

In the last couple of years generative AI tools have gone from a novelty to something
most students have tried at least once. Writing a first draft, rephrasing a paragraph,
or explaining a hard concept is now a few seconds of typing into a chatbot. That is
useful, but it has also put real pressure on how universities judge whether a piece of
work is a student's own. Tools that claim to detect AI-written text appeared quickly in
response, and a lot of institutions started leaning on them. The trouble is that those
tools were built fast, and they tend to hand back a single number with nothing behind it.

## 1.2 Problem statement

The core problem I keep coming back to is that current detectors give you a percentage
and stop there. A lecturer who sees "82 percent AI" has no way to explain that number if
a student pushes back, and no way to know if it is even right. That is a weak position to
accuse someone from, and the cost of being wrong is high for the student. There is a
second, deeper issue. Even a correct flag does not tell you the thing that actually
matters, which is whether the student understands the work they handed in. A flag is a
suspicion, not evidence of understanding or the lack of it.

## 1.3 Motivation

I am aiming this at the lecturer who runs a large module, say 150 to 300 students, and
has no time to sit with each flagged essay and work out a fair way to check it. That
person needs something they can defend, that treats students fairly, and that saves them
time rather than adding to the pile. I also care about the fairness side because the
research already shows these detectors can be biased against people who do not write in
English as a first language, and I do not want to build something that quietly repeats
that.

## 1.4 The verification gap

So there is a gap. On one side we have detection, which is getting better but stays a
black box. On the other side we have the real question, does this student understand
their own submission. Nothing I found connects the two. Nothing takes a flag and turns it
into specific questions, drawn from the student's own claims, that a lecturer can ask in a
short conversation to see if the understanding is there. That gap is what this project
tries to fill.

## 1.5 Research question and objectives

The main question I am asking is this: how can an explainable AI pipeline be designed to
support lecturers in academic integrity verification, by combining transparent AI text
detection with argument-aware question generation, and how far can locally fine-tuned
open-source models match commercial ones at producing interpretable, defensible outputs
for this task.

To get there the objectives are roughly:

- detect AI-written text in a way that can be explained, not just scored;
- show which words and features drove each decision, and test that those explanations are
  faithful;
- pull out the claims and evidence in a flagged essay and tie each one to its source;
- generate verification questions from those claims, and compare a commercial model
  against a local open-source one;
- label each question by cognitive level as a quality check; and
- evaluate the whole thing with objective measures rather than opinion.

## 1.6 Scope and boundaries

To keep this doable in the time I have, the scope is deliberately tight. The detector
works with two classes only, human and AI. I dropped the earlier idea of partial-AI
classes because they overlap too much to separate reliably; that may come back later as
future work if the two-class version works well. The pipeline has six parts (detection,
explainability, argument mining, question generation, a Bloom's level check, and the
output guide), and the datasets are fixed in advance. There are no human participants in
the study, which keeps it clear of needing ethics approval and keeps the focus on the
system itself.

## 1.7 Contributions

What I think this project adds:

- a pipeline that goes from a detection flag to transparent, defensible evidence, instead
  of stopping at a score;
- verification questions that are tied back to the student's own writing, so a lecturer
  can see where each one came from;
- a direct comparison of a locally run open model against a commercial one for this task,
  which matters for any institution thinking about cost and control;
- an objective way to measure whether a question really tests understanding; and
- an honest look at how the detector treats native versus non-native writers.

## 1.8 Dissertation outline

The rest of the document is laid out as follows. Chapter 2 reviews the recent literature
on AI text detection, explainability, argument mining, and question generation. Chapter 3
sets out the detection methodology and the first results, including the dataset, how the AI
essays were built, and the audit that found and removed a corpus artefact. Chapter 4 covers
the implementation of the parts built so far. Chapter 5 explains the detector's decisions and
tests whether those explanations are faithful. The later chapters cover the remaining
components (argument mining, question generation, the Bloom's check, and the output guide),
the evaluation, a discussion of what the results mean and where the limits are, and the
conclusion with future work.
