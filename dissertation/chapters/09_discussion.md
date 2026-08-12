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
detector reaches F1 0.990, with a bootstrap interval of roughly [0.97, 1.00]. Section 9.2 explains
why that figure is the least interesting one here. The
detector also transfers across unseen generators at 0.97. Across text domains it degrades to 0.79,
and it degrades in the direction that matters most, by falsely flagging human academic writing. A
detector that holds up in the setting it was trained on and falls away outside it is the pattern the
field already reports, whether the shift comes from paraphrasing (Krishna et al., 2023) or from the
tools themselves being tested in the wild (Weber-Wulff et al., 2023). The contribution here is not
that the drop happens but that it is measured per domain and priced, so a deployment decision can be
made with the number in front of it. The explanation layer went through the same scrutiny. Both
token-level methods, attention included, fail a faithfulness test that the feature-level SHAP
account passes, so the lecturer sees the feature account.

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

There are caveats. The commercial arm is a flash model, not a frontier one. Every writer in the main
comparison sits below the generic-question baseline, though the v4 experiment (Section 8.13) later
showed most of that gap to be the price of naming the claim's content, closable by construction at
some cost in phrasing variety. And because the comparison's outcome moved with the experimental
design, the dissertation reports every design it ran, not only the friendliest one.

## 9.2 How these results sit against the published work

Four of this project's numbers can be placed directly against figures in the literature, and the
comparison is more useful where it is unflattering than where it is not.

**Detection in domain is not where the contribution is.** An F1 of 0.990 sounds strong until you
notice that near-perfect in-domain scores are the norm in this field rather than the exception, and
that Chapter 3 spent most of its length explaining why the first version of that number was worth
nothing. The interesting comparison is the false-positive rate. Binoculars (Hans et al., 2024)
detects over 90 percent of ChatGPT output at a false-positive rate of 0.01 percent with no training
data at all. My hybrid, on unfamiliar human prose, sits at 61 percent on arXiv abstracts. Those two
numbers are measured on different tasks and are not a like-for-like contest, but the gap between a
benchmark false-positive rate of 0.01 percent and a real one measured in tens of percent is the
whole reason this project treats a flag as an opening rather than a finding. A tool used on a module
of three hundred students cannot tolerate percentage-level false accusation, and none of the systems
in this comparison can currently promise otherwise outside their training domain.

**Cross-domain degradation is smaller here than the paraphrase literature reports, and that is not a
victory.** Krishna et al. (2023) show a strong paraphraser dropping DetectGPT from 70.3 percent
accuracy to 4.6 percent, a collapse. My detector falls from F1 0.990 to 0.790 across domains, which
looks mild beside that. The comparison flatters me for the wrong reason: I never tested against an
adversarial paraphraser. Domain shift is a passive change in the input distribution and paraphrase
attack is an active one, and the second is strictly harder. The honest statement is that this
detector has been measured against distribution shift and has not been measured against an
adversary, and Section 9.7 lists that as the first thing I would add.

**Argument mining sits below the published ceiling, as expected for the strict setting.** The claim
extractor reaches a strict span-level F1 of 0.63 with exact boundary and type match. Stab and
Gurevych report macro F1 0.867 for component identification against a human upper bound of 0.886 on
the same corpus. Those are different measurements, theirs a token-level identification task and mine
a strict span task, so the gap is not a fair deficit of 0.24. What the comparison does establish is
the direction of the ceiling: even the reference system reaches only 97.9 percent of human
performance on the easier framing, and claim boundaries are the hard part in both. For this pipeline
the boundary does not have to be exact, because the quotation shown to a lecturer is looked up from
the submission by sentence number rather than reproduced from the model's span.

**The judge result is the one that departs from the literature's usual finding, and it departs in a
documented direction.** Zheng et al. (2023) report strong judges matching human preference around 80
percent of the time, which is why LLM-as-judge became a default. My three-judge panel reaches
Krippendorff's alpha of -0.14 across sixty questions and correlates negatively with the objective
measure. Read alone that looks like an outlier. Read against Feuer et al. (2024), who find judge
preferences failing to track measured safety, world knowledge or instruction following, and against
Norman et al. (2026), who separate reliability from validity across 21 judges and roughly 541,000
judgments, it is the same phenomenon at small scale. Their phrase, reliability without validity, is
exactly what Chapter 8 measures: three judges that agree about the ranking and none that agrees with
the measurement. The contribution is not the discovery but the demonstration on a task where an
independent objective measure existed to check them against, which is the condition under which the
failure is visible at all.

Taken together, the positioning is modest and I would rather state it that way. This project does
not beat the field on detection and does not claim to. What it does is join components the field has
built separately, measure each one against something it cannot influence, and report the three
occasions on which that measurement contradicted a result I had already written down.

## 9.3 What the internal checks caught

The project's thesis is that unexplained scores cannot be trusted. Three of my own headline
numbers demonstrated the point, and the checks that caught them are among the most useful results
in the dissertation.

The first was the detector that scored a perfect 100 percent, covered in Section 3.4. A rule using
nothing but leftover corpus markup reached 92.5 percent on the same text, which located the score
in how the corpus had been exported rather than in how the essays were written. Cleaning both
classes and retraining put the honest figure at 0.990.

The second was the fine-tune artifact. The v1 adapter posted a discrimination score four times
its base model, with non-overlapping confidence intervals, and for about an hour it looked like the
headline result of the project. Reading the actual output showed 95 percent degenerate
multiple-choice stems. The audit then showed that those stems win the metric, because a contentless
question makes the source-aware and source-blind answers diverge at random.

The third was the judge panel. Three commercial judges, the validation the scope called for,
produced ratings that neither agree with each other (Krippendorff's alpha of -0.25) nor track the
objective measure. That LLM judges carry position, verbosity and self-enhancement biases is
documented (Zheng et al., 2023); what this study adds is a case where the panel agrees with itself
and still points away from the measured truth, which is the harder failure to notice. Rerunning the
panel at sixty questions made the verdict firmer. At sixty all three judges agree with each other in
rank, none correlates positively with the simulation, and the two funded ones anti-correlate
significantly. What a rubric measures is how a question reads, and this project needed to know
whether it works, which turns out to be a different property. The panel study stands as a
replicated negative result.

A fourth case was quieter and had the same shape. Ranking the fine-tunes by raw discrimination
alone would have picked v2, whose terse factual one-liners are not verification questions at all,
over v3, whose output is what the pipeline needs. I count three rather than four because that one
was caught before anything had been written down, which is the difference between a near miss and a
retraction.

One methodological posture covers all three cases, and it is built into the pipeline now rather than
left as advice. Every automatic score is anchored to something it cannot game. The judges are
anchored to the simulation. The simulation is gated for well-formedness, and the degeneracy rate is
reported next to every score. No design decision was delegated to a single number. That posture was
demonstrated three times on the project's own results, and I would argue it is as much a
contribution as any component. Weber-Wulff et al. (2023) tested fourteen detection tools that
universities license and found them neither accurate nor reliable, and none of them exposes the kind
of internal check described here.

The same posture is applied to the write-up itself. Section 4.12 describes a test suite in which
each test encodes a claim this document makes, and Section 4.10 an audit that reads every number in
the text back against the results files. Both were written after the fact and both found errors:
a summary claim about the hybrid detector that had been wrong in three places, and a feature count
that disagreed with the code. A document arguing that scores need checking should be able to say
what checks its own sentences.

## 9.4 Fairness and verification

The fairness evidence influenced the design more than any other single result. Liang et al. (2023)
found that detectors flag non-native English writers at wildly disproportionate rates, an average
false-positive rate of 61.22 percent on TOEFL essays against about 5 percent on essays by native
writers, and my own robustness tests reproduced the mechanism behind that pattern at domain level.
Human arXiv abstracts were falsely flagged 79 percent of the time, because dense academic prose
looks, to these features, like machine text. The trigger in both cases is the same. Text that is
fluent but low in variation reads as machine-written, whether it comes from a second-language writer
or from a discipline whose register is dense by convention.

Two things follow, and only one of them is technical. The technical one is that fusion helps and
does not cure: the hybrid cuts the human false-positive rate by a factor of between three and five
on four of the five out-of-domain test sets, and on arXiv abstracts it moves 79 percent to 61
percent, which is better and still unusable alone. The other is a question of where evidential
weight is allowed to sit. A detector with that failure mode cannot be the end of any fair process,
so the pipeline treats detection as the opening of a conversation and moves the weight to
verification. The questions are ones a student who wrote and understood the work can answer and an
impostor cannot. Each question is traceable to exact sentences of the submission, so nothing is
invented.

It is worth being precise about what the in-domain fairness result does and does not show. On the
held-out test set the detector falsely flagged 0 of 50 essays by non-native writers and 2 of 50 by
native writers, with overlapping confidence intervals. The honest reading is no detectable
difference on a deliberately balanced sample of student essays, not that the detector is fair in
general. The population that gets hurt by these tools is characterised by how they write rather than
by a label in a corpus, and the arXiv result is the same mechanism showing up where the corpus has
no fairness label at all.

The discrimination simulation made clear what such questions have to look like. Its central finding
is that content-naming questions are answerable from general knowledge. A good verification question does the
opposite. It asks why one example was chosen over another, what a phrase in the third paragraph was
doing, or how an objection raised in the middle connects to the conclusion. None of that is
answerable from general knowledge, because none of it is general. That finding held in every later experiment. I rewrote the
generation prompt around it and built the v3 training data on it, and it explains why v2 and v3 rank
the way they do on the raw metric. It also sets a limit on what the simulation can measure. The
context-blind model is a much stronger stand-in than a student who submitted work they do not
understand, so measured discrimination is a conservative floor. It is not an estimate of classroom
performance.

There is a fairness question about the verification stage too, and this project cannot answer it. An
interview format advantages students who are fluent and confident in spoken English, which is the
same group the detector already treats unfairly. Written answers, extra time, or a first-language
option are obvious mitigations and none of them has been tested here. Moving the evidential weight
from a detector to a conversation removes one bias and introduces the possibility of another, and a
classroom study would have to measure that rather than assume it away.

## 9.5 Limitations and threats to validity

The limitations are stated where they arise in the chapters. This section gathers the important ones
and sorts them by what they threaten, because a sample-size worry and a construct worry are not the
same kind of problem and do not have the same remedy.

**Construct.** These are the serious ones. Discrimination is a proxy for understanding and not a
measure of it: a context-blind language model is not a student who did not write the essay, although
it is a considerably harder opponent, which is why the proxy is conservative rather than merely
approximate. Bloom's level is a proxy for cognitive demand, assigned by the weakest model in the
pipeline at a macro-F1 of 0.31. And a detection flag is a proxy for misconduct, which is precisely
the equivalence the whole design refuses to make. Each of those three gaps is between what was
measured and what a lecturer actually cares about, and no amount of extra data closes them.

**Internal.** The evaluation of question quality rests on a simulation rather than on students. That
was a deliberate scope decision, taken with my supervisor, and it is what keeps the project clear of
human-participants approval. It also means classroom validation is the single most important piece
of future work. Sample sizes grew but stay modest: the final comparison runs thirty essays with 901
questions, the fine-tune tests rest on 27-essay paired designs, and the judge study grew from twelve
questions to sixty. At sixty, its anti-correlation with the simulation could no longer be attributed
to sample size. Bootstrap intervals are used throughout, and the sample-size trajectory of the first
comparison is reported in full because small snapshots would have supported whichever story I
preferred. Transformer runs use a single seed, where the stylometric model was checked across five
and returned an identical F1 each time.

**External.** The detection corpus pairs one human corpus with one generator. The detector transfers
to six unseen generators, but the corpus itself cannot show generator diversity on the training
side. One country, one level of study, one discipline mix. The commercial arm is one provider's free
tier rather than a frontier model, and its version can change without notice; quota limits caused
disclosed irregularities. On the home corpus the detection task saturates, so the
training-distribution control needed the harder cross-domain setting before it had resolution. There
it found the balancing harmless for accuracy but slightly costly for false accusations on unfamiliar
human text (Section 6.6), a trade-off measured on one corpus pair and worth re-testing at larger
scale.

**Component ceilings.** The argument miner extracts claims at a working but unspectacular span-F1 of
0.63. The relation classifier learns supports-links well (F1 0.75 with gold components) but cannot
learn attack relations from the 42 of 4,922 test pairs that carry them. That is a label-supply
ceiling, and the Bloom classifier hits the same one from the same cause. Neither is an architecture
problem and neither is fixed by a larger model.

**Not tested at all.** Two absences are worth naming rather than leaving implied. The detector has
never been shown an adversarial paraphrase, which is the attack the literature treats as decisive.
And no part of the pipeline has been run by anyone other than me, so every usability claim about the
lecturer's guide and the interface rests on design argument rather than observation.

## 9.6 Implications for practice

**For a lecturer**, the practical output of this work is a defensible process, not a score. A flag
arrives with named, tested reasons ("the sentences are unusually uniform, the vocabulary less varied
than a typical student's") and with an interview guide instead of a verdict. The change that matters
is in what the lecturer is asked to defend. Defending a percentage means defending a model;
defending a question drawn from the student's own third paragraph means asking the student about
their own third paragraph, which is something a lecturer is already qualified to do.

**For an institution**, three of the results carry direct consequences. First, detectors,
including good ones, must not be pointed at text from domains they were not trained on, and any deployment needs
domain checks, calibrated thresholds and an abstain band, because the failure mode is false
accusation and it lands unevenly. Second, the local-versus-commercial result removes a
dependency: models
that run on a single consumer laptop matched and then beat a commercial tier on the fixed task, and
the pipeline was built entirely under that constraint. The constraint turned out to be an advantage,
because it means verification can run without sending student work to a third party, and submissions
are personal data. Third, an abstain band is cheaper than it looks and less effective than it
sounds. Section 6.8 measures it: abstaining on the middle band lifts accuracy on what remains from
0.79 to 0.88 but barely moves the false accusation rate, because the surviving errors are confident
ones. It is worth deploying and it is not a fix.

**For a student**, the output is something to answer rather than something to rebut. Gorichanaz
(2023) describes students accused of AI use adopting a legalistic posture and gathering version
histories to prove authorship, with the burden effectively shifted onto them. A guide that quotes
their own sentences and asks them to explain their own reasoning does not remove that burden, but it
changes it from proving a negative to demonstrating something they should be able to demonstrate. If
the pipeline has a claim on being fairer, that is where it sits.

**For policy**, the finding with the widest reach is the least technical, and Section 9.2 has
already stated it: the checks built into this project overturned three of its own results.
Institutional policy that treats a
detector's percentage as a threshold for action has no equivalent check anywhere in it. The
practical recommendation is not a better detector. It is that no automated score should trigger a
consequence without a human step in which the student can account for their own work.

## 9.7 What I would do differently

Four things, in the order I would change them.

The audit should have come first. The markup artefact cost a week of work built on a detector that was
reading formatting, and the audit that found it took an afternoon to write. Every subsequent
component was checked before it was believed, and the only reason the first one was not is that I
had not yet learned to.

Testing against an adversarial paraphraser belonged in the plan from the start. Domain shift is the failure I
measured because it is the failure I thought of; paraphrase attack is the one the literature treats
as decisive, and by the time that was clear the evaluation design was fixed. It is the first thing I
would add with more time.

Counting the available Bloom's labels should have preceded choosing a classifier. Two components in this pipeline are
capped by label supply rather than by architecture, and both caps were discovered after the models
were trained. An hour spent counting available labels for the smallest classes would have redirected
the effort.

Last, the question-generation comparison should have used fixed claims from the first run rather
than the third. The finding that most of the commercial advantage was claim selection is a good result,
but it arrived by accident after two designs that measured something other than what I intended.
Deciding what has to be held constant, before running anything, is the cheapest methodological
improvement available and it is the one I most consistently skipped.
