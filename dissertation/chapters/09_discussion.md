# Chapter 9: Discussion

## 9.1 Answering the research question

The research question had two halves: whether an explainable pipeline can support lecturers in
academic integrity verification by combining transparent detection with argument-aware question
generation, and how far locally fine-tuned open-source models can match commercial LLMs at it. The
work gives each half a concrete answer, and neither answer is the simple one I expected at the start.

On the first half, the pipeline exists and works end to end. A submission goes in; a detector trained
on a matched, audited corpus scores it; the score arrives with a faithfulness-tested explanation in
lecturer language rather than a bare percentage; the submission's own claims are extracted with
sentence-level provenance; verification questions grounded in those claims are generated, gated for
well-formedness, tagged with a cognitive level, and assembled into an interview guide whose first page
says what the whole design says: this is evidence for a conversation, not an accusation. Every stage
was evaluated, and the evaluations are the reason the answer is credible. The detection is excellent
in domain (F1 0.990 with a bootstrap interval of roughly [0.97, 1.00]) and transfers across unseen
generators (0.97), but degrades across text domains (0.79) in the direction that matters most,
falsely flagging human academic writing. The explanation layer is honest about itself: all three
token-level methods, attention included, fail a faithfulness test that the feature-level SHAP account
passes, so the feature account is the one a lecturer sees.

On the second half, the answer turned out to depend on holding the task fixed, which is itself a
finding. When each backend chose its own claims, the commercial model held a small significant edge
(paired difference 0.036, p = 0.040). When every model wrote questions for identical claims, the edge
vanished (difference -0.005, p = 0.62): most of the apparent commercial advantage was claim selection,
not question writing. Fine-tuning then moved the local backend well clear of its own base once the
training data had the right format, and at thirty essays the fine-tuned 3B not only holds its gain
over its own base (+0.040, p = 0.0001) but beats the free-tier commercial model on the essays both
cover (+0.050, p = 0.0094, higher on eleven of thirteen), with every question well-formed, which is
why that claim is made here after being retracted twice in weaker forms. Put together, the honest
answer to "can locally fine-tuned open models match commercial LLMs at this task" is yes on this
evidence, and on the fixed task the fine-tuned local model does better than the free commercial tier;
the caveats are that the commercial arm is a flash model rather than a frontier one, that all
well-formed writers still sit below the generic-question baseline, and the deeper point that the
comparison's outcome moved with the experimental design, which is why the dissertation reports every
design it ran rather than the friendliest one.

## 9.2 What the failures taught: trust nothing you have not checked

The most valuable results in this dissertation are the ones that nearly deceived me, because the
project's thesis is that unexplained scores cannot be trusted, and three times my own scores proved
it.

The first was the fine-tune artifact. The v1 adapter posted a discrimination score four times its
base model, with non-overlapping confidence intervals, and for about an hour it was the headline of
this dissertation. Reading its actual output showed 95 percent degenerate multiple-choice stems, and
the audit showed those stems do not merely survive the metric but win it: a contentless question makes
the source-aware and source-blind answers diverge at random. The second was the judge panel. Three
commercial judges, the validation the scope called for, produced ratings that neither agree with each
other (Krippendorff's alpha of -0.25) nor track the objective measure, with one judge significantly
anti-correlated. Rubric ratings measure how good a question looks. The third was quieter but the same
shape: ranking the fine-tunes by raw discrimination would have picked v2, whose terse factual
one-liners are not verification questions at all, over v3, whose output is exactly what the product
needs.

One methodological posture answers all three, and it is now built into the pipeline rather than
stated as advice. Every automatic score is anchored to something it cannot game: the judges are
anchored to the simulation, the simulation is gated for well-formedness with the degeneracy rate
reported next to every score, and no design decision was delegated to a single number. I would argue
this posture, demonstrated three times on the project's own results, is as much a contribution as any
component, because it is precisely the discipline the field's detector vendors do not practise.

## 9.3 Verification over accusation

The fairness evidence shaped the design more than any other single input. The literature reports that
detectors flag non-native English writers at wildly disproportionate rates, and my own robustness
tests reproduced the pattern's mechanism at domain level: human arXiv abstracts were falsely flagged
79 percent of the time because dense academic prose looks, to these features, like machine text. A
detector with that failure mode cannot be the end of any fair process. The pipeline therefore treats
detection as the opening of a conversation and moves the evidential weight to verification: questions
a student who wrote and understood the work can answer and an impostor cannot, each question traceable
to exact sentences of the submission so nothing is invented.

The discrimination simulation sharpened what such questions must look like. Its central finding is
that content-naming questions are answerable from general knowledge, so good verification questions
force the student to reconstruct their own reasoning, name their own evidence, and connect their own
argument. That principle survived every subsequent experiment, reshaped the generation prompt, drove
the v3 training data, and explains the ranking twist between v2 and v3. It also bounds what the
simulation can see: the context-blind model is a much stronger stand-in than a student who submitted
work they do not understand, so measured discrimination is a conservative floor, not an estimate of
classroom performance.

## 9.4 Limitations

The limitations are stated throughout the chapters; gathering the important ones in one place keeps
the claims honest. The evaluation of question quality rests on a simulation, not on students; that
was a deliberate scope decision (it keeps the project clear of human-participants approval), but it
means the discrimination scores are proxies with a conservative bias, and classroom validation is the
single most important piece of future work. Sample sizes grew but stay modest: the final comparison
runs thirty essays with 754 questions and the fine-tune tests rest on 27-essay paired designs, while
the judge study is twelve questions; bootstrap intervals are used throughout, and the sample-size
trajectory of the first comparison is reported precisely because small snapshots would have supported
whichever story I preferred. The detection
corpus pairs one human corpus with one generator, and although the detector transfers to six unseen
generators, the corpus itself cannot show generator diversity. The commercial arm is one provider's
free tier rather than a frontier model, and quota limits caused disclosed irregularities. On the home
corpus the detection task saturates, which is why the training-distribution control could show the
balancing is harmless but not whether it helps on a harder task. And the argument-mining component
extracts claims at a working but unspectacular span-F1 of 0.63, and the relation classifier learns
supports-links well (F1 0.75 with gold components) but cannot learn attack relations from the 0.7
percent of pairs that carry them, a label-supply ceiling it shares with the Bloom classifier.

## 9.5 Implications for practice

For a lecturer, the practical output of this work is a defensible process rather than a score. A flag
arrives with named, tested reasons ("the sentences are unusually uniform, the vocabulary less varied
than a typical student's"), and with an interview guide rather than a verdict. For an institution,
two results matter. First, detectors, including good ones, must not be pointed at text from domains
they were not trained on, and any deployment needs domain checks, calibrated thresholds and an
abstain band, because the failure mode is false accusation and it lands unevenly. Second, the
local-versus-commercial result removes a dependency: models that run on a single consumer laptop
matched a commercial API on the fixed task, which means an institution can run verification without
sending student work to a third party, a real consideration given that submissions are personal data.
The pipeline was built entirely on that constraint, and the constraint turned out to be a feature.
