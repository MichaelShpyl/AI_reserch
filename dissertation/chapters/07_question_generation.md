# Chapter 7: From a flag to verification questions (first slice)

> Draft note (delete before submission): rough first-person draft for me to rewrite in my own
> words. This is the first working slice of the core contribution, the argument-aware question
> generation, built end to end on the local model. The full commercial-versus-local comparison
> and the evaluation come next. No em dashes; my own wording for the final version.

## 7.1 Why this is the point of the project

Detection on its own only produces a flag, and the earlier chapters showed how fragile a flag can
be out of domain. The actual contribution of the project is what happens after the flag: turning a
suspected submission into something a lecturer can act on fairly, which is a short set of questions,
drawn from the student's own claims, that check whether the student understands what they handed in.
A student who wrote and understood the work can answer them; a student who did not will struggle.
This chapter is the first working version of that step.

## 7.2 How the slice works

The pipeline takes a flagged essay and produces a Verification Interview Guide
(`src/question_gen/generate_questions.py`). It runs in four steps.

First, it splits the essay into numbered sentences. Second, it extracts the student's main claims,
and for each claim it records the sentence numbers the claim comes from. This is the important
design choice for provenance: the model is asked to cite sentence numbers, not to quote the text, so
the source is looked up from the real numbered sentences and the model cannot invent a quote. Every
claim therefore points at exact lines in the submission. Third, for each claim it generates
verification questions that are grounded in the source sentences and written so that they cannot be
answered well by someone who only read the claim. Fourth, each question is tagged with a Bloom's
cognitive level.

Backends sit behind one interface. This slice runs on the local open-source model (Llama 3.1 8B
through Ollama), which is the basis for Backend B. The commercial Backend A plugs into the same
interface once an API key is available, and because the backend is recorded with every guide the
two can be compared directly, which is the core commercial-versus-local research question.

## 7.3 An example

Run on a flagged essay that compares three literary texts, the system pulled out four claims and
wrote grounded questions for each. Two of them give the flavour.

For the claim that the three texts use language differently (metaphor and symbolism in the poetry,
dramatic intensity in the drama, subtle nuance in the prose), tied to two specific sentences in the
submission, it asked: "What specific features of Shakespeare's dialogue lead you to describe it as
having dramatic intensity?" and "How does Austen's subtle and nuanced prose style allow her to
explore complex social issues in a way that might be less effective with more overt language?" Both
need the student to reason about the text, not repeat the claim.

For the claim that the texts share themes of love and personal growth, tied to a single sentence, it
asked for an example from one of the texts that links personal growth to relationships, which is a
question a student who genuinely engaged with the texts can answer and a student who did not cannot.
The full guide, with every claim, its exact source sentences, and the Bloom's levels, is saved as
`outputs/verification_guides/3108a_ai.md`.

## 7.4 What is first-slice and what comes next

This is deliberately a thin path through the remaining pipeline, and I am honest about the parts that
are stand-ins. The claim extraction is currently prompted rather than the trained argument miner
planned in the scope (the Persuasive Essays corpus, claim and premise labelling); a prompted
extractor can miss or merge claims, but it feeds question generation now and can be upgraded later.
The Bloom's tag is a transparent keyword heuristic, a placeholder for the BERT classifier that is its
own component. And the questions have not yet been evaluated: the planned discrimination simulation,
where a model with the source and a model without it both try to answer each question and the gap
shows how well the question targets understanding, is the next piece of work and is what will turn
these example questions into measured results.

The immediate next steps are therefore to add the commercial backend and run the commercial-versus-local
comparison, to stand up the discrimination-simulation evaluation, and to decide with my supervisor
whether the local backend is the full Llama 3 8B with QLoRA or a smaller model that fits the laptop,
since that choice affects the comparison.
