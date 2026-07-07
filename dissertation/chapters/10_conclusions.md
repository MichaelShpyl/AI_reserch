# Chapter 10: Conclusions and future work

## 10.1 What this dissertation contributes

The project set out to replace an indefensible number with a defensible process, and to find out
whether that process needs a commercial API behind it. Its contributions, each backed by a result in
the preceding chapters, are these.

First, a matched detection corpus and an audited detector. The corpus pairs 640 real student essays
with 640 AI essays matched on topic and length, so the classes differ in writing rather than in
anything a detector could cheat on; the audit methodology that caught a corpus artefact behind an
initially perfect score, removed it from both classes, and re-established the result on style alone
is reusable beyond this project.

Second, an explanation stack whose explanations are tested, not assumed. Three attribution methods
were measured on one ablation yardstick; the two token-level methods fail it for the same reason, a
diffuse signal, and the feature-level SHAP account passes it and becomes the lecturer-facing
explanation. The claim "this explanation is faithful" is, throughout, an empirical statement.

Third, an argument-aware question-generation pipeline with provenance by construction: claims cite
sentence numbers that are looked up rather than quoted, so a generated question can always be traced
to exact lines of the submission, and a well-formedness gate keeps degenerate output away from both
the lecturer and the metrics.

Fourth, a judge-free evaluation for verification questions, and the demonstration that it can be
gamed. The discrimination simulation measures whether a question needs the source; the discovery that
contentless questions defeat it, and the gate that closes that hole, are a methods contribution to an
evaluation literature that leans heavily on LLM judges.

Fifth, the three-judge validation with a decisive negative result: commercial judges neither agree
with each other nor track the objective measure on this task, which is direct evidence for anchoring
judge studies rather than trusting them.

Sixth, the commercial-versus-local comparison run under two designs, showing the comparison's outcome
is a property of the design: a significant commercial edge under self-chosen claims disappears
entirely under fixed claims. On the fixed task, models running on one consumer laptop matched the
commercial API.

Seventh, the data-format experiment across three fine-tunes of the same model with identical
settings: a multiple-choice corpus collapses the model into unusable output, either open-ended format
repairs it, the verification-style data produces the on-style backend, and the 3B student overtook
its 8B teacher on the target measure.

And eighth, the working artifact itself: a Verification Interview Guide that generates end to end
from a submission with live models on an 8 GB laptop, framing its contents as evidence for a
conversation.

## 10.2 Future work

The clearest next step is the one this project deliberately excluded: a study with real students,
under ethics approval, testing whether the generated questions separate authors from non-authors in
practice as the simulation predicts. The simulation's conservative bias makes this a hopeful
experiment rather than a hail-Mary, and the guide is the ready-made instrument for it.

Several extensions follow directly from measured limits. The commercial arm should widen to multiple
providers and a frontier model now that the framework treats any API as a plug-in backend. The
detection corpus should gain generations from more model families, letting the transfer results rest
on the corpus itself rather than an external benchmark. Cross-domain deployment needs calibrated
per-domain thresholds and an abstain band before anyone should trust a flag outside student essays,
and the hybrid detector's out-of-domain behaviour suggests where those calibrations should start. The
training-distribution control should be re-run on a task hard enough to have resolution. The Bloom
classifier is throttled by label supply, not architecture, so a better-populated labelling of
higher-order questions would lift the guide's quality control. Relation classification between claims
and premises would complete the argument-mining component as originally specified. And the agreed
future extension of the classification target, a third partial-AI class, becomes worth piloting once
mixed human-and-AI documents can be constructed carefully, since partial assistance is the realistic
case in coursework.

## 10.3 Closing

The dissertation's method can be stated in one sentence: never present a number you cannot explain,
and never trust one you have not tried to break. Practising that on my own results cost me three
headlines, an artifact fine-tune, an agreeing panel of judges, and a tidy ranking, and each time it
replaced a false claim with a finding that was more useful and easier to defend. That, more than any
single score, is what I would want a lecturer, or an examiner, to take from this work: the pipeline
is trustworthy in exactly the places where it can show its working, and it is honest about the places
where it cannot.
