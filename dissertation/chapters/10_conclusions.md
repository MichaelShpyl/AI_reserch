# Chapter 10: Conclusions and future work

## 10.1 The answer

This project asked whether an explainable pipeline can support a lecturer in academic integrity
verification by joining transparent detection to argument-aware question generation, and how far a
locally fine-tuned open model can go against a commercial one at that task.

The answer to the first half is yes, with a condition attached. The pipeline exists, runs end to
end on one 8 GB laptop, and produces a document a lecturer can take into a room. The condition is
that it works because it does not ask detection to carry the decision. Pointed at human arXiv
abstracts, the transformer falsely flags 79 percent of them and the shipped hybrid still flags 61.
Neither number can be the end of a disciplinary process, and the only reason the system is
defensible is that the flag opens a conversation instead of closing one.

The answer to the second half is yes, and rather more firmly than I expected. On thirty essays with
the claim set held fixed, a QLoRA fine-tune of Qwen2.5 3B beat the free commercial tier on 24 of the
29 essays both covered, at p = 0.0003, and beat the 8B model it was distilled from on 25 of 30. A
university can run this kind of verification on hardware it already owns, without sending student
work to anybody.

Both answers are narrower than they sound, and Chapter 9 sets out how narrow. What follows here is
what the work contributes, what should happen next, and what it does not settle.

## 10.2 The six objectives, and what happened to each

Section 1.6 set six objectives. Five were met and one was met with a qualification I would not want
buried, so they are worth closing off one at a time before the contributions are claimed.

Table: The six objectives from Section 1.6 against what the work actually delivered.

| Objective | Outcome |
|---|---|
| Detect AI-written text in a way that can be explained as well as scored | Met. F1 0.990 in domain on an audited corpus, every flag accompanied by a named-habit explanation (Chapters 3 and 5) |
| Show which words and features drove each decision, and test that those explanations are faithful | Met, and the test is the result. Both token-level methods failed the ablation; the feature-level account passed and is what ships (Chapter 5) |
| Pull out the claims and evidence in a flagged essay and tie each one to its source | Met. Strict span-F1 0.63, with sentence-level provenance that cannot be forged (Chapter 7) |
| Generate verification questions from those claims, and compare a commercial model against a local open-source one | Met. The local 3B beats the free commercial tier on 24 of 29 essays at p = 0.0003 (Chapter 8) |
| Label each question by cognitive level as a quality check | Met with a qualification. The classifier works and doubles a keyword baseline, but at macro-F1 0.31 it is the weakest model in the pipeline, capped by label supply rather than architecture (Section 7.5) |
| Evaluate the whole thing with objective measures, not opinion | Met, and it is the objective that shaped the rest. The judge-free simulation is the primary measure; the LLM panel brought in as a supplement failed its own validation and is reported as a negative result (Chapter 8) |

The one qualification is the Bloom classifier, and it is worth being plain about what it means. A
cognitive level printed beside a question is quality control on the generator rather than a claim
about the question, and at macro-F1 0.31 it should be read as a weak signal. Nothing else in the
pipeline depends on it.

One objective in the scope is not in that table, because it belongs to the design rather than the
components. Chapter 1 promised an examination of how the detector treats writers and writing it was
not trained on, including what the make-up of the training data does to false accusations. Chapter
6 delivers it: the 79 percent false-positive rate on unfamiliar academic prose, the fusion that
reduces it by a factor of three to five on four of five domains without curing it, the abstain band
that lifts accuracy and barely moves false accusation, and the supervisor-requested control showing
a balanced training mix to be harmless for accuracy and slightly costly for false flags. That work
produced no headline number, and it changed the shape of the system more than any result that did.

## 10.3 Contributions

Four contributions, each stated with the evidence behind it and the limit around it. The detail is
in the chapters; this is the summary a reader should be able to hold.

**A working pipeline, and the corpus that makes its numbers mean anything.** A Verification
Interview Guide generates end to end from a submission with live models. Behind it sits a corpus of
640 real student essays paired with 640 AI essays matched on topic and length, cleaned by identical
code, split at writer level. When the first detector scored perfectly, an audit traced the score to
a corpus artefact rather than to writing, removed it from both classes, and re-established the
result on style alone at F1 0.99. The audit procedure is reusable and is the part of this
contribution I would defend hardest, because the corpus is what every later number rests on. What
it does not do is show generator diversity: one human source paired with one generator demonstrates
the design and not the range.

**Explanations that were tested rather than asserted.** Attribution methods were measured on one
ablation yardstick. Both token-level methods failed it with the same diffuse signal, attention
included, and the feature-level SHAP account passed, so that account is what a lecturer sees. In
this dissertation "this explanation is faithful" is an empirical statement with a test behind it,
and the two methods that failed are reported alongside the one that passed. Faithfulness was
measured on a single yardstick, and a different ablation protocol might rank the three differently.
It would still have to account for the finding that removing the tokens an explanation names barely
moves the prediction.

**Question generation with provenance that cannot be forged.** Claims cite sentence numbers, the
sentence text is looked up from the submission, and the model is never asked for a quotation. An
invented quotation is therefore impossible by construction rather than by good behaviour, and
Section 4.12 describes the tests that hold that property in place. A well-formedness gate keeps
degenerate output away from both the lecturer and the metrics. Provenance is the strong part;
selection is not. The claims come from a miner at a strict span-F1 of 0.63, so which claims surface
is considerably less reliable than where the ones that do surface came from.

**Applying a judge-free measure to question selection, and a demonstration that it can be gamed.**
The discrimination simulation compares a source-aware against a source-blind model and needs no
human and no LLM judge. The underlying idea is not mine: Liusie et al. use the same comparison to
audit existing exam questions, and finding a published precedent strengthens the method rather than
weakening the claim, since it means the measure was not invented for convenience. What this project
adds is the shift from auditing a fixed test to selecting among generated questions, and the failure
that came with it. Contentless questions defeat the measure, which is how a fine-tune came to post a
score four times its base while producing nothing usable, and the gate that closes that hole is part
of the contribution rather than a patch on it. The limit is the one Chapter 9 states plainly: a
context-blind model is a stronger stand-in than a student who did not write the work, so the measure
is a conservative floor rather than a classroom prediction.

## 10.4 Future work

The order below is the order I would actually do these in, and the reasons for the order matter as
much as the list.

**First, a classroom study with real students under ethics approval.** Everything this dissertation
says about question quality rests on a simulation. The study would test whether the questions
separate authors from non-authors as the simulation predicts, and the simulation's conservative
bias gives it a fair chance of confirming rather than contradicting. Two arms are already designed:
a blind comparison of the shipped v3 questions against the v4 anchored ones, which settles a
judgement Chapter 8 had to make on reasoning rather than evidence, and a measurement of whether the
interview format disadvantages students who are less fluent in spoken English, which is the
fairness question this project raised and could not answer. The guide is the ready-made instrument
for all of it. Nothing else on this list changes what the work can claim as much as this does.

**Second, robustness against an adversary rather than against drift.** The detector has been
measured against domain shift and never against a paraphrase attack, which is the failure the
literature treats as decisive. Adding an adversarial paraphrase arm would either confirm the
detector is usable or show it is not, and either answer is worth more than another point of
in-domain F1. Alongside it, per-domain calibrated thresholds and the abstain band should be fitted
before a flag is trusted outside student essays, with the caveat Section 6.8 measured: calibration
lifts accuracy on what remains and barely moves the false accusation rate, because the surviving
errors are confident ones.

**Third, widening what the corpus and the comparison cover.** The test side already spans six
unseen generators from the external benchmark plus two commercial families; the training side
should be widened to match, so transfer can be demonstrated on the corpus itself rather than on
somebody else's. The commercial arm should widen to several
providers and at least one frontier model, which the framework already supports since any API is a
plug-in backend. Both are straightforward and neither changes a conclusion, which is why they sit
third.

**Fourth, the two components capped by label supply.** The Bloom classifier and the attack-relation
classifier are limited by data rather than architecture, and the cheap route to more data is now
closed: LLM annotators were tried under a validation against gold labels and failed it on the
higher-order classes. The supply has to be human, which makes this expensive rather than difficult.
A corpus with denser attack annotation would finish the relation component.

**Fifth, the scope extension agreed with my supervisor.** A third partial-AI class becomes worth
piloting once mixed human-and-AI documents can be constructed carefully, because partial assistance
is the realistic case in coursework and the two-class framing treats it as a boundary problem. It
sits last because it needs the corpus work above it to be done first.

## 10.5 What this work does not settle

Three questions stayed open, and they are open in a way that further tuning would not close.

Whether the questions work on students is the obvious one, and it is why the classroom study heads
the list above. No amount of simulation answers it.

Whether the pipeline helps a real lecturer is the second, and it is quieter. Every usability claim
here rests on design argument. Nobody other than me has ever run the interface or read a guide with
a real case in front of them, and the honest position is that the design is reasoned rather than
observed.

The third is whether moving evidential weight from a detector to a conversation is a net gain in
fairness. It removes a measured bias against second-language writers, which is the whole reason the
design points that way. It also introduces an interview, and an interview advantages students who
are fluent and confident in speech, which may be the same people. This dissertation argues the
first half and can only flag the second. A design that removes one bias and creates another has not
obviously improved anything, and finding out which way that balance falls needs the students this
project deliberately did without.

## 10.6 Closing

One rule ran through this work: do not present a number you cannot explain, and do not trust a
number you have not tried to break. Applying it to my own results removed three headline claims. A
detector that scored perfectly was reading corpus formatting. Ninety-five percent of the output
from the fine-tune that beat everything was multiple-choice stems with no content in them. And the
judge panel brought in to validate the question measure turned out to disagree with it. Each time,
the finding that replaced the headline was more useful and easier to defend than the headline had
been.

That is the argument the project makes twice over. It makes it about detectors, where a percentage
with nothing behind it is a poor basis for accusing a student. And it makes it about my own work,
where the checks are described in the chapters where they apply and the failures are reported next
to the successes.

If a reader takes one thing from this dissertation, it should be the shape of the answer rather
than any single number. A detection score is evidence of a weak and unevenly distributed kind. The
useful question is not how to make that score more confident, but what to put between the score and
a consequence. This project's answer is a conversation the student can take part in, grounded in
sentences they wrote themselves, with the reasoning visible to both sides.
