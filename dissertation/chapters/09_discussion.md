# Chapter 9: Discussion

## 9.1 Answering the research question

The research question had two halves. The first asked whether an explainable pipeline can support
lecturers in academic integrity verification by combining transparent detection with argument-aware
question generation. The second asked how far locally fine-tuned open-source models can match
commercial LLMs at the task. Both halves now have concrete answers, though neither is the simple
answer I expected at the start.

On the first half, the pipeline exists and works end to end. A submission goes in and a detector
trained on a matched, audited corpus scores it. The score arrives with a faithfulness-tested
explanation written in lecturer language instead of a bare percentage. The submission's own claims
are extracted with sentence-level provenance. Verification questions grounded in those claims are
generated, gated for well-formedness, tagged with a cognitive level, and assembled into an interview
guide. The guide's first page frames what follows as evidence for a conversation with the student,
not an accusation, and the rest of the design takes the same position.

Every stage was evaluated, and the evaluations are what make this answer credible. In domain the
detection is excellent, an F1 of 0.990 with a bootstrap interval of roughly [0.97, 1.00]. The
detector also transfers across unseen generators at 0.97. Across text domains it degrades to 0.79,
and it degrades in the direction that matters most, by falsely flagging human academic writing. A
detector that holds up in the setting it was trained on and falls away outside it is the pattern
the field already reports, whether the shift comes from paraphrasing (Krishna et al., 2023) or from
the tools themselves being tested in the wild (Weber-Wulff et al., 2023). The contribution here is
not that the drop happens but that it is measured per domain and priced, so a deployment decision
can be made with the number in front of it. The
explanation layer went through the same scrutiny. All three token-level methods, attention included,
fail a faithfulness test that the feature-level SHAP account passes, so the lecturer sees the
feature account.

On the second half, the strongest result in the dissertation is where the comparison ended up. At
thirty essays the fine-tuned 3B holds its gain over its own base (+0.046, p < 0.0001, higher on 25
of 30 essays) and beats the free-tier commercial model on the 29 essays both cover (+0.040,
p = 0.0003, higher on 24 of 29), with every question well-formed. I make that claim after retracting
two weaker forms of it. On this evidence the answer to "can locally fine-tuned open models match
commercial LLMs at this task" is yes, and on the fixed task the fine-tuned local model does better
than the free commercial tier. That direction agrees with Oketch et al. (2025), who report open
models reaching comparable performance to closed ones at lower cost on annotation-style tasks. The
result here is narrower, one task and one free-tier commercial arm, but it points the same way and
carries the same practical consequence for an institution weighing an API subscription against a
machine it already owns.

Getting there depended on holding the task fixed, and that dependence is a finding in itself. When
each backend chose its own claims, the commercial model held a small significant edge (paired
difference 0.036, p = 0.040). When every model wrote questions for identical claims, the edge
vanished (difference +0.003, p = 0.74). Most of the apparent commercial advantage was claim
selection, not question writing. Fine-tuning then moved the local backend well clear of its own base
once the training data had the right format.

There are caveats. The commercial arm is a flash model, not a frontier one. Every writer in the
main comparison sits below the generic-question baseline, though the v4 experiment (Section 8.13)
later showed most of that gap to be the price of naming the claim's content, closable by
construction at some cost in phrasing variety. And because the comparison's outcome moved with the
experimental design, the dissertation reports every design it ran, not only the friendliest one.

## 9.2 What the internal checks caught

The project's thesis is that unexplained scores cannot be trusted. Three of my own scores
demonstrated the point, and the checks that caught them are among the most useful results in the
dissertation.

The first case was the fine-tune artifact. The v1 adapter posted a discrimination score four times
its base model, with non-overlapping confidence intervals, and for about an hour it looked like the headline
result of the project. Reading the actual output showed 95 percent degenerate multiple-choice stems.
The audit then showed that those stems win the metric, because a contentless question makes the
source-aware and source-blind answers diverge at random.

The second case was the judge panel. Three commercial judges, the validation the scope called for,
produced ratings that neither agree with each other (Krippendorff's alpha of -0.25) nor track the
objective measure. That LLM judges carry position, verbosity and self-enhancement biases is
documented (Zheng et al., 2023); what this study adds is a case where the panel agrees with itself
and still points away from the measured truth, which is the harder failure to notice. Rerunning the panel at sixty questions made the verdict firmer. At sixty all
three judges agree with each other in rank, none correlates positively with the simulation, and
the two funded ones anti-correlate significantly. Rubric ratings measure how good a question looks. The panel study stands as a
replicated negative result.

The third case was quieter but had the same shape. Ranking the fine-tunes by raw discrimination
would have picked v2, whose terse factual one-liners are not verification questions at all, over v3,
whose output is what the product needs.

One methodological posture covers all three cases, and it is built into the pipeline now rather
than left as advice. Every automatic score is anchored to something it cannot game. The judges are
anchored to the simulation. The simulation is gated for well-formedness, and the degeneracy rate is
reported next to every score. No design decision was delegated to a single number. That posture was
demonstrated three times on the project's own results, and I would argue it is as much a
contribution as any component. Weber-Wulff et al. (2023) tested fourteen detection tools that
universities license and found them neither accurate nor reliable, and none of them exposes the
kind of internal check described here.

## 9.3 Fairness and verification

The fairness evidence influenced the design more than any other single result. Liang et al. (2023)
found that detectors flag non-native English writers at wildly disproportionate rates, an average
false-positive rate of 61.22 percent on TOEFL essays while the same detectors handled US
eighth-grade essays almost perfectly, and my own robustness tests reproduced the mechanism behind
that pattern at domain level. Human arXiv abstracts were falsely flagged 79 percent of the time,
because dense academic prose looks, to these features, like machine text. The trigger in both cases
is the same. Text that is fluent but low in variation reads as machine-written, whether it comes
from a second-language writer or from a discipline whose register is dense by convention. A detector with that failure mode cannot be the end of any fair process. The
pipeline therefore treats detection as the opening of a conversation and moves the evidential weight
to verification. The questions are ones a student who wrote and understood the work can answer and
an impostor cannot. Each question is traceable to exact sentences of the submission, so nothing is
invented.

The discrimination simulation made clear what such questions have to look like. Its central finding
is that content-naming questions are answerable from general knowledge. Good verification questions
instead force the student to reconstruct their own reasoning, name their own evidence, and show how
their own argument fits together. That finding held in every later experiment. I rewrote the
generation prompt around it and built the v3 training data on it, and it explains why v2 and v3 rank
the way they do on the raw metric. It also sets a limit on what the simulation can measure. The
context-blind model is a much stronger stand-in than a student who submitted work they do not
understand, so measured discrimination is a conservative floor. It is not an estimate of classroom
performance.

## 9.4 Limitations

The limitations are stated where they arise in the chapters. This section gathers the important
ones in one place.

The evaluation of question quality rests on a simulation, not on students. That was a deliberate
scope decision (it keeps the project clear of human-participants approval), but it means the
discrimination scores are proxies with a conservative bias. Classroom validation is the single most
important piece of future work.

Sample sizes grew but stay modest. The final comparison runs thirty essays with 901 questions, and
the fine-tune tests rest on 27-essay paired designs. The judge study grew from twelve questions to
sixty. At sixty, its anti-correlation with the simulation could no longer be attributed to sample
size. Bootstrap intervals are used throughout, and the sample-size trajectory of the first
comparison is reported in full because small snapshots would have supported whichever story I
preferred.

The detection corpus pairs one human corpus with one generator. The detector transfers to six unseen
generators, but the corpus itself cannot show generator diversity. The commercial arm is one
provider's free tier rather than a frontier model, and quota limits caused disclosed irregularities.
On the home corpus the detection task saturates, so the training-distribution control needed the
harder cross-domain setting before it had resolution. There it found the balancing harmless for
accuracy but slightly costly for false accusations on unfamiliar human text (Section 6.6), a
trade-off measured on one corpus pair and worth re-testing at larger scale.

The argument-mining component extracts claims at a working but unspectacular span-F1 of 0.63. The
relation classifier learns supports-links well (F1 0.75 with gold components) but cannot learn
attack relations from the 0.7 percent of pairs that carry them. That is a label-supply ceiling, and
the Bloom classifier hits the same one.

## 9.5 Implications for practice

For a lecturer, the practical output of this work is a defensible process, not a score. A flag
arrives with named, tested reasons ("the sentences are unusually uniform, the vocabulary less varied
than a typical student's") and with an interview guide instead of a verdict.

For an institution, two of the results carry direct consequences. Detectors, including good ones,
must not be pointed at text from domains they were not trained on, and any deployment needs domain
checks, calibrated thresholds and an abstain band, because the failure mode is false accusation and
it lands unevenly. The local-versus-commercial result removes a dependency. Models that run on a
single consumer laptop matched a commercial API on the fixed task. The pipeline was built entirely
under that constraint, and the constraint turned out to be an advantage, because it means an
institution can run verification without sending student work to a third party. Given that
submissions are personal data, that is a real consideration.
