# Validation study protocol (draft for supervisor review)

Status: draft, 16 July 2026. This is the design for the August validation named in the
dissertation's future work. Nothing here has been submitted to ethics yet; the point of this
document is to agree the design early enough that an ethics application, if one is needed at
all for the chosen variant, can go in before the end of July.

## What needs validating

Two claims currently rest on proxies:

1. The discrimination simulation says the generated questions separate someone who knows the
   essay from someone who does not. The stand-in for "someone who does not" is a context-blind
   LLM. No human has yet answered these questions.
2. The lecturer-facing outputs (the explanation card and the interview guide) are designed for
   non-technical readers, but no non-technical reader has been observed using them.

Both fit inside one small study with two arms.

## Arm A: do the questions discriminate for humans?

**Design.** Within-subjects, counterbalanced. Each participant supplies a short piece of their
own writing (500 to 800 words, any coursework-like topic, written before the session so the
study adds no writing burden). The pipeline generates verification questions for every text.
In the session, each participant answers two question sets in randomised order: the questions
for their own text, and the questions for one other participant's text which they are given
five minutes to read (the "read-only" condition simulates a student defending work they did
not produce; reading it first makes the comparison conservative, since a contract-cheating
student has at least read what they submitted).

**Participants.** Eight to twelve postgraduate volunteers from the department. No students in
any live academic-integrity process; no real submissions; no deception.

**Measures.** Answers scored on the guide's own three-level rubric (strong, partial, weak) by
two blind raters working from anonymised transcripts; per-question scoring difference between
own-text and read-only conditions; correlation between the human own-versus-read gap and the
simulation's per-question discrimination score, which is the number that validates or corrects
the simulation.

**Analysis.** Paired per-participant comparison (Wilcoxon); Spearman correlation of human gaps
against simulated discrimination across questions; inter-rater agreement on the rubric.

## Arm B: can a lecturer actually use the outputs?

**Design.** Three to five lecturers, one thirty-minute session each. Each receives two real
guides from the pilot corpus (the flagged AI essay and its unflagged human twin), reads them
cold, and thinks aloud. Structured tasks: say what the detector concluded and why, in their own
words; identify which writing habits drove the flag from the explanation card; pick the three
questions they would actually ask. Ends with the System Usability Scale and open comments.

**Measures.** Task success, misreadings (especially of the card's typical-student bands),
usability score, and the comment list, which becomes the revision backlog for the card.

## Ethics and data

- Volunteers, written consent, right to withdraw, no compensation issues to manage.
- No real misconduct cases, no live submissions, no deception anywhere.
- Texts and answers anonymised at collection (participant codes); everything stays on the
  project laptop, consistent with the rest of the project; deleted after marking of the
  dissertation.
- Arm B may qualify as staff usability consultation rather than human-subjects research;
  whether Arm A's design needs full ethics review or a light-touch application is the first
  question for the supervisor and the department's process.

## Timeline

- Week of 21 July: agree design with supervisor, submit whatever approval the chosen variant
  needs, recruit by department mailing list.
- Mid August: run both arms (each participant session is under an hour; the whole study fits
  in one week).
- Late August: analysis is small and scripted in advance (the measures above are all simple
  paired statistics), so it cannot eat the protected writing time.

## Fallbacks

If Arm A cannot be approved or recruited in time, Arm B alone still discharges the most
criticised gap (outputs untested on their audience) with minimal ethics surface. If neither
runs, the study design itself, reviewed by the supervisor, goes into the dissertation as
specified future work with a concrete protocol, which is worth more than a vague intention.
