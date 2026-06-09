# Formal Record of Dissertation Meeting

**Student:** Mykhailo Shpyl  **Supervisor:** Dr. Vini Vijayan
**Programme:** MSc Artificial Intelligence and Big Data Analytics
**Date:** 9 June 2026 (Meeting 2)

## Key points discussed

**Architecture walkthrough.** Presented the full pipeline as a visual flow: student
submission, then the AI detector, explainability, argument mining, question
generation, Bloom's classifier, and the output guide. The supervisor had read the
draft material and was positive on it.

**Detector design.** The detector combines a fine-tuned DeBERTa transformer with two
stylometric features. The first is text predictability (perplexity from next-word
prediction): AI tends to choose high-probability continuations, so its text is more
predictable than typical student writing. The second is burstiness: humans vary
sentence length and structure more, while AI writes more uniformly. The supervisor
asked clarifying questions and was satisfied with the rationale.

**Explainability.** SHAP and Integrated Gradients show which passages drove a flag.
Faithfulness is tested by ablation: remove the highlighted evidence and check whether
the detector's confidence changes. If the score is unchanged after removing the stated
evidence, the explanation was not faithful. The supervisor noted, correctly, that this
is an assumption-based check.

**Question generation (a key research contribution).** Argument mining extracts the
student's claims and evidence and links them to questions. The project compares a
commercial LLM (more adaptive, higher cost) against a locally run open-source LLM (more
cost-effective and predictable at institutional scale, fewer external updates). Bloom's
taxonomy classifies each question by cognitive level.

**Output.** A guide for the lecturer with highlighted passages, the detection score,
recommended verification questions, and a suggested grading rubric.

**Dataset work.** BAWE was cleaned (one row was labelled twice and dropped). The sample
is stratified by the four disciplinary groups and balanced between native and
non-native English writers, because research shows non-native writing can read as more
AI-like and should not be under-represented. A per-student cap was applied (one student
contributed about twenty essays) so no single writing style is over-represented, and
train, validation and test were split by student so the same student does not appear in
more than one split. The current sample is 640 essays, balanced, for an initial local
run. Length is similar for native and non-native writers (non-native slightly longer).

**Supervisor feedback on the dataset.**
- The dataset may be over-structured. The model should be allowed to learn from less
  structured data, and the student should not spend too much time on manual balancing.
- The balanced set is fine for an initial supervised run.
- If the structured-versus-natural comparison is pursued, it must use two separately
  trained models run in parallel, not one model reused, so earlier training does not
  contaminate the comparison.
- Any chart used in the dissertation must have category percentages that sum to 100
  (the group chart had rounded values summing to 101). This has been corrected.

**Classification.** Start with two classes (human vs AI). If that works well, a later
phase could add a third "partial AI" class and test whether the three can be separated,
accepting that partial AI overlaps heavily with both other classes. Two classes remain
the scope for now.

**Compute.** ATU HPC / test PC access is still being arranged. The supervisor will check
availability and add the student as a user, then revert by email.

**Status.** The AI-written essay dataset has not been generated yet (waiting on compute),
and the detector has not been trained yet.

**Writing.** Draft the Introduction at a high level now, without fixing the method and
tool detail, since the methods may change as the model is built. Complete the
method-specific writing after the model exists.

**Assessment format.** There is currently no viva in the official position, but the
supervisor favours a viva and will discuss it with the coordinators. Prepare for both a
recorded presentation and a viva.

**Dissemination.** The supervisor encouraged submitting a short abstract (200 to 250
words) to an NLP or AI conference in Ireland or Europe (hybrid is fine) within the next
few weeks. A journal is unlikely to fit the 13-week timeline, but a conference abstract
can be accepted quickly and proceedings often include a journal.

## Action items for the student

- Upload the architecture and dataset materials, and this meeting record, to the
  OneDrive folder, and email the supervisor once uploaded.
- Draft the Introduction chapter at a high level, without detailed methods.
- Test the detection pipeline locally on the small balanced sample while awaiting compute.
- Generate the matched AI-written essays once compute is available.
- Search for suitable NLP / AI conferences (Ireland or Europe, hybrid) and prepare a 200
  to 250 word abstract.
- Prepare for both a presentation and a viva.
- Clarify the "application form" the supervisor mentioned at the end of the meeting.
- Correct the group-percentage chart so the values sum to 100 (done).

## Action items for the supervisor

- Check ATU HPC / test PC availability, add the student as a user, and revert by email.
- Discuss the assessment format (viva vs recorded presentation) with the coordinators.

## Date of next meeting

Tuesday 16 June 2026, 12:00 (online).

## Issues requiring immediate attention

- Compute access remains the main risk to the timeline and should be confirmed early.
- The conference abstract has a near-term window (next few weeks), so the conference
  search should start now.
