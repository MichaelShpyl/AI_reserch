# Chapter 1: Introduction

## 1.1 Overview

This dissertation builds an explainable pipeline that helps a lecturer check the integrity of a
student submission without having to treat a detector's score as the final word. The pipeline does
two connected things. It decides whether a piece of writing is likely AI-generated, and rather than
returning a bare percentage it shows which features of the writing drove that decision. Then, for a
flagged submission, it reads the student's own argument, pulls out the specific claims they made,
and turns those claims into verification questions a lecturer can ask in a short conversation. The
aim is to give the lecturer evidence they can explain and defend, tied to the student's own words,
so a fair decision has something to rest on. Catching students out is not the goal.

Every stage is meant to be explainable, so a lecturer can see and justify why a submission was
flagged and where each question came from, instead of pointing at a number they cannot account for.
The question generation is also built twice, once with a commercial large language model and once
with a smaller open model that runs on a single laptop. Building it twice lets the project measure
how close a locally run, low-cost model can get to an expensive commercial one for this task
(Oketch et al., 2025). The question matters to any institution, and the comparison is a central
contribution of this dissertation.

I built the system for a lecturer on a large module, somewhere between 150 and 300 students, who
has neither the time to prepare a fair set of verification questions for every flagged essay nor a
good way to defend a black-box score if a student challenges it. The system has to save that
lecturer time. It also has to treat students fairly, including students who do not write in English
as a first language and whom existing detectors tend to penalise (Liang et al., 2023), and it has
to produce something the lecturer can stand behind.

## 1.2 Background and context

Over the last few years generative AI tools have moved from a curiosity to something most students
have used at least once (Jin et al., 2024). Drafting an opening paragraph, rephrasing an awkward
sentence, or getting a hard idea explained now takes a few seconds of typing into a chatbot. That
is genuinely useful for learning. It also puts pressure on a basic academic question, whether a
piece of work is the student's own and reflects their own understanding.

Universities responded quickly, and a market of AI-text detectors appeared almost as fast as the
writing tools did (Wu et al., 2025). Broadly the detectors fall into two families. Zero-shot or
statistical methods use a language model's own probabilities to judge how predictable a passage is,
on the assumption that machine text sits in smoother, more probable regions than human text
(Mitchell et al., 2023). Supervised methods fine-tune a transformer on labelled human and AI
examples and then classify new text (He et al., 2021). Commercial services usually combine such
methods behind an interface and return a single figure, an AI-likelihood percentage.

These tools do work some of the time. But they were built fast and they are opaque. Their accuracy
drops sharply on text they were not trained for, and simple paraphrasing can evade them (Krishna et
al., 2023). More seriously for a university, they hand back a number with nothing behind it, and
there is growing evidence that the errors are not spread evenly. Writing by non-native English
speakers gets flagged as AI far more often than it should be (Liang et al., 2023). A confident
score with no explanation behind it and a known bias in its errors is a poor basis for an
academic-integrity decision.

## 1.3 Problem statement

Current detectors give a lecturer a percentage and stop there. Someone who sees "82 percent AI" has
no way to explain that number if the student pushes back, and no real way to know whether it is
even right for this particular essay. Accusing a person from that position is weak and risky, and
the cost of being wrong falls on the student, whose record and standing are at stake. A bare
percentage is not enough to carry a decision with those stakes.

On top of the opacity there is a fairness problem. Because the errors are biased, the students most
likely to be wrongly flagged are often those who are already disadvantaged, including people
writing in a second language (Liang et al., 2023). A system that repeats that bias and cannot show
its reasoning is worse than no system at all.

Even a correct flag does not answer the question a lecturer actually cares about. Knowing that text
is probably AI-generated does not tell you whether the student understands the material they handed
in. One student might have used a tool heavily and still grasp the argument. Another might have
written every word themselves and understood little. A flag is a suspicion about how a text was
produced, and it carries no evidence about what the student knows. A lecturer needs a fair way to
move from that suspicion to a check of understanding, and no detector provides one.

This project takes on that distance. It makes the detection step transparent and defensible, then
turns a flag into specific, source-grounded questions that test whether the student can account for
their own work.

## 1.4 Motivation

I am aiming this at the lecturer who runs a large module and has no time to sit with each flagged
essay and work out a fair way to check it. The lecturer needs a process they can defend. It should
treat students fairly, and it has to save time rather than add to the pile. The fairness side
matters to me in particular. The research already shows these detectors can be biased against
people who do not write in English as a first language (Liang et al., 2023), and I do not want to
build something that quietly repeats that. If the system shows its reasoning, its mistakes can be
seen and argued with, and a fair process needs that to be possible.

## 1.5 The verification gap

There is a gap between detection and the question a lecturer actually needs answered. Detection is
improving but stays a black box, and whether the student understands their own submission is a
separate matter. Nothing I found connects the two, and nothing takes a flag and turns it into
specific questions, drawn from the student's own claims, that a lecturer can ask in a short
conversation to see whether the understanding is there. This project is built to fill that gap.

## 1.6 Research question and objectives

The main question is how an explainable AI pipeline can be designed to support lecturers in
academic integrity verification, by combining transparent AI text detection with argument-aware
question generation, and how far locally fine-tuned open-source models can match commercial ones at
producing interpretable, defensible outputs for this task.

To get there the objectives are roughly:

- detect AI-written text in a way that can be explained as well as scored;
- show which words and features drove each decision, and test that those explanations are faithful;
- pull out the claims and evidence in a flagged essay and tie each one to its source;
- generate verification questions from those claims, and compare a commercial model against a local
  open-source one;
- label each question by cognitive level as a quality check; and
- evaluate the whole thing with objective measures, not opinion.

## 1.7 Scope and boundaries

To keep this doable in the time I have, the scope is deliberately tight. The detector works with
two classes only, human and AI. I dropped the earlier idea of partial-AI classes because they
overlap too much to separate reliably; if the two-class version works well, they may come back as
future work. The pipeline has six parts (detection, explainability, argument mining, question
generation, a Bloom's level check, and the output guide), and the datasets are fixed in advance.
The study has no human participants, which keeps it clear of needing ethics approval and keeps the
focus on the system itself.

## 1.8 Contributions

What I think this project adds:

- a pipeline that carries a detection flag through to evidence a lecturer can inspect and defend,
  instead of stopping at a score, running end to end on one consumer laptop;
- verification questions tied back to the student's own writing, so a lecturer can see where each
  one came from;
- a controlled comparison of locally run open models against a commercial one for this task,
  relevant to any institution weighing cost and control, which ends with the fine-tuned local
  model ahead of the free commercial tier on the fixed task;
- an objective, judge-free way to measure whether a question really tests understanding, along
  with measured evidence for why the popular LLM-as-judge alternative cannot be trusted without
  an anchor;
- an examination of how the detector treats writers and writing it was not trained on, including
  what the make-up of the training data itself does to false accusations; and
- a demonstration, run on my own results, that an automatic score is only worth the checks behind
  it: three of this project's own headline numbers were caught and retracted on the way to the
  ones reported here.

## 1.9 Dissertation outline

The rest of the document is laid out as follows.

Chapter 2 reviews the recent literature the project draws on, covering AI-text detection and its
benchmarks, stylometric features, explainability and faithfulness, argument mining, question
generation and its evaluation, and the fairness evidence behind the design.

Chapter 3 sets out the detection methodology and the first results, including the dataset, the
construction of the matched AI essays, and the audit that found and removed a corpus artefact
behind an initially perfect score.

Chapter 4 describes the implementation of the components built so far, with the engineering
decisions that the 8 GB laptop budget forced.

Chapter 5 explains the detector's decisions and tests whether those explanations are faithful,
comparing token-level attributions on the transformer against SHAP over the stylometric features.

Chapter 6 tests robustness, looking at how the detector transfers to generators it never saw and to
other kinds of text, and what its errors mean for fairness.

Chapter 7 presents the core contribution, turning a flag into verification questions drawn from the
student's own claims, with sentence-level provenance and a Bloom's level on each question. It
builds the first thin slice, then replaces its stand-ins with the trained Bloom's classifier and
the trained claim extractor, assembles the lecturer's Verification Interview Guide, and fine-tunes
the local backend.

Chapter 8 evaluates the questions. It builds the judge-free discrimination simulation, runs the
commercial-versus-local comparison under two designs, validates a three-judge LLM panel against the
objective measure, and reports the fine-tuning data-format experiment together with the quality
audit that caught a metric-gaming artefact.

Chapter 9 discusses what the results mean, covering the answer to the research question, the
methodological lesson drawn from the negative results, the case for verification over accusation,
the limitations, and the implications for practice. Chapter 10 concludes and sets out future work,
and the references follow.
