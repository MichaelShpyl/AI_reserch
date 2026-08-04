# Chapter 10: Conclusions and future work

## 10.1 What this dissertation contributes

The project set out to give lecturers a process they can defend in place of a single unexplained
score, and to find out whether that process needs a commercial API behind it. The short answer to
both questions: the process exists and runs, and it does not need the API. Each contribution below
is backed by a result in the preceding chapters.

First, the working system. A Verification Interview Guide generates end to end from a submission,
with live models, on an 8 GB laptop. The guide presents its contents as starting material for a
conversation with the student, not as a verdict.

Second, a matched detection corpus and an audited detector. The corpus pairs 640 real student
essays with 640 AI essays matched on topic and length, so the two classes differ in writing and
not in anything a detector could use as a shortcut. When the first detector scored perfectly, an
audit traced the score to a corpus artefact, removed it from both classes, and re-established the
result on style alone: F1 0.99 in domain, and 100 percent detection of essays from two commercial
generators the model never saw in training, with no false accusations of the matching human
essays. The audit procedure is reusable beyond this project.

Third, explanations that have been tested rather than assumed. Three attribution methods were
measured on one ablation yardstick. The two token-level methods fail it, both with the same
diffuse signal, and the feature-level SHAP account passes, so that account is what a lecturer
sees. In this dissertation, "this explanation is faithful" is an empirical statement.

Fourth, question generation with provenance built in. Claims cite sentence numbers, the sentence
text is looked up from the submission itself, and a generated question can always be traced to
exact lines. A well-formedness gate keeps degenerate output away from the lecturer and away from
the metrics.

Fifth, the local-versus-commercial answer. The comparison ran under two designs, and the outcome
depended on the design, which is itself a finding. With self-chosen claims the commercial backend
held a significant edge. With the claims held fixed the edge disappeared, and after fine-tuning,
the local 3B wrote better questions than the free commercial tier on 24 of the 29 shared essays
(p = 0.0003) at thirty-essay scale. A university can run this kind of verification on ordinary
hardware, without sending student work to a third party.

Sixth, a controlled data-format experiment. The same model was fine-tuned three times with
identical settings, varying only the training corpus. A multiple-choice corpus collapsed the model
into unusable output, either open-ended corpus repaired it, and the verification-style data
produced the backend the pipeline ships. In the same runs, the 3B student overtook its 8B teacher
on the target measure.

Seventh, an evaluation method for verification questions that needs no judge, together with a
demonstration of how it can be gamed, and a resolution of its stubbornest number: the v4
experiment showed the generic-baseline gap to be mostly the price of naming the claim's content. The discrimination simulation measures whether a question
needs the source. Contentless questions defeat it, and the well-formedness gate closes that hole.
The discovery and the gate together are a methods contribution to an evaluation literature that
leans heavily on LLM judges (Zheng et al., 2023).

Eighth, a replicated negative result on those judges. At twelve questions, three commercial judges
neither agreed with each other nor tracked the objective measure. Rerun at sixty questions, all
three judges agree with each other in rank and still point away from the measure, and two of the
three rank the best backend below the plain 8B. That is direct evidence for anchoring judge studies to an
objective measure instead of trusting the panel.

## 10.2 Future work

The next step with the most to offer is a study with real students, run under ethics approval, to
test whether the generated questions separate authors from non-authors in practice as the
simulation predicts. This project excluded human participants by design, so that ground is
untested. The simulation's conservative bias gives the study a fair chance of confirming the
prediction, and the guide is the ready-made instrument for it.

Several extensions follow from measured limits. The commercial arm should widen to multiple
providers and a frontier model, now that the framework treats any API as a plug-in backend. The
detection corpus gained test essays from two more model families during the robustness work; the
training side should be widened the same way, so the transfer results can be shown on the corpus
itself without leaning on an external benchmark. Cross-domain deployment needs calibrated
per-domain thresholds before anyone should trust a flag outside student essays, and the measured
abstain band shows what calibration alone will not fix. The training-distribution control, re-run on the hard cross-domain task, now shows a real
trade-off (accuracy unchanged, more false flags from the balanced mix), and the next step is
testing whether that effect reaches writer-level fairness on larger corpora. The Bloom classifier is held back by label
supply, not architecture, and the cheap route to more labels is now closed: LLM annotators were
tried under a pre-registered validation against the gold labels and failed it on the higher-order
classes (Section 7.5), so the supply that would lift the guide's quality control has to be human. The relation classifier learns supports links well (F1 0.75) but attack
relations are too rare in the training corpus to learn at all; a corpus with denser attack
annotation would finish that component. And the agreed extension of the classification target, a
third partial-AI class, becomes worth piloting once mixed human-and-AI documents can be
constructed carefully, since partial assistance is the realistic case in coursework.

## 10.3 Closing

One rule ran through this work: do not present a number you cannot explain, and do not trust a
number you have not tried to break. Applying that rule to my own results removed three headline
claims, and each time the finding that replaced the headline was more useful and easier to defend.
The version of the local-beats-commercial claim that survives in Chapter 8 is the one that earned
its place. Where the pipeline can show its working it can be trusted, and where it cannot, the
document says so.
