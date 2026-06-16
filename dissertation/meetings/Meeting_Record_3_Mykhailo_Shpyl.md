# Formal Record of Dissertation Meeting

**Student:** Mykhailo Shpyl  **Supervisor:** Dr. Vini Vijayan
**Programme:** MSc Artificial Intelligence and Big Data Analytics
**Date:** 16 June 2026 (Meeting 3)

> Note: the recording began about ten minutes into the meeting, so the earlier part of the
> progress walkthrough is reconstructed from the slides presented. The discussion from the
> detector audit onward is taken directly from the recording.

## Key points discussed

**Progress since Meeting 2.** Presented the work as a visual deck. The matched AI essay
dataset is complete (640 human and 640 AI essays, 1,280 total), generated locally with
Llama 3.1 8B and validated for length and topic matching. A first detector has been trained.

**Detector audit (main discussion).** The first detector scored 100 percent, which was
treated as a warning rather than a success. On inspection the human BAWE essays carried
structural export tags (for example `<heading>`) that the AI essays did not, so the model had
partly learned "has tags therefore human". The tags were stripped from both sides. The slide
shown breaks this down: a tag-only rule alone scores 92.5 percent (the size of the shortcut),
the raw-text model scores 100 percent (still able to see the tags), and after cleaning the
model still scores 100 percent, so a real difference remains. A second issue was identified:
Llama writes American English while the students write British English, so part of the signal
was language locale rather than AI authorship. To rule out both formatting and topic, a model
restricted to function words only still reached 99.5 percent, which shows the remaining signal
is genuine writing style.

**Explainability (now starting, a main research topic).** Showed the words that push a
prediction each way: humans use blunter connectives ("therefore", "because", "so"), while the
AI uses a more uniform, sophisticated register ("in conclusion", "understanding of", "complex",
"essential"). Also showed the essays plotted in style space, where human and AI form two
separate clusters.

**Why detection is easy at this stage.** With only human writing and one AI model, the two
classes separate cleanly. As more generators are added (GPT, Claude, DeepSeek), the clusters
will overlap and will need to be separated with more care. The dataset will also need to
include AI text that a human has rewritten or paraphrased, because partially human-edited AI
text would otherwise distort the predictions.

**Discussion on scope of "AI use".** The point was raised that many universities and journals
now accept work that has been modified with generative AI tools, so blanket detection is not
the goal. The case of interest here is AI-written text that a human has tweaked, which is the
partial-AI category discussed previously and remains later-phase work. The core scope stays
two classes for now.

**Language of submission.** Agreed that the detector should account for the expected language
variety of the submissions (for example British English for ATU students) so that language
locale is not mistaken for AI authorship. The generation step should match the expected
locale.

**Publication.** AICS (the Irish Conference on Artificial Intelligence and Cognitive Science)
is confirmed to be running, and the supervisor received an email about it the day before. The
event is in September, so the proposal window is near. GenAIDetect remains a second target.
The student is still locating the application and proposal forms.

**Submission of materials.** The student will submit the chapters drafted so far now, and the
presentation as a separate file, so the supervisor can see the progress. A small item that had
not yet been uploaded will be submitted straight away.

**Timeline.** About two and a half months remain to the end-of-August deadline. The supervisor
advised not to over-stress and to keep submitting work as it is drafted.

## Action items for the student

- Submit the drafted chapters now, and the presentation as a separate file, to the supervisor.
- Write the implementation chapters: how the dataset was built and generated, the detector,
  and the audit.
- Research the AICS application and proposal process (September event) and prepare a proposal;
  keep GenAIDetect as a second target.
- Improve multi-generator detection and dataset consistency, and plan for paraphrased or
  partially human-edited AI text.
- Match the AI generation to the expected submission language (British English) so locale is
  not learned instead of style.
- Begin the explainability work (the attribution methods), as the next main component.
- Carry over: clarify the "application form" mentioned previously, and confirm the HPC update.

## Action items for the supervisor

- Confirm the HPC / compute access position when available.
- Share any AICS proposal details received.

## Date of next meeting

Tuesday 23 June 2026 (online), to review the chapters and the explainability progress.

## Issues requiring immediate attention

- The AICS proposal window is near (September event), so the application details should be
  found and a proposal drafted soon.
- Multi-generator and paraphrased AI text are the next dataset priorities, since the current
  single-generator setup makes detection easier than the real task will be.
