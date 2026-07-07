# Chapter 6: Robustness, transfer to unseen generators and domains

## 6.1 The question

The in-domain detector scores about 0.99, but that is on the easy setting: one generator
(Llama) and one kind of text (student essays). The honest question is what happens when neither
holds. To test it I took the trained detector, with no retraining or adaptation, and ran it on
the M4 benchmark (SemEval-2024 Task 8), which has many generators and many domains that the
detector never saw. I split this into two tests so I do not accidentally call the easy half
"robust".

A note on sources, so the two tests are not confused. Test A uses the OUTFOX essay set that ships
inside the M4 release: human essays against six generators. Test B uses the M4 / SemEval-2024 Task 8
monolingual data across five web and academic domains. They are different corpora with different
generator sets, which I chose deliberately: A isolates a change of generator while holding the text
type fixed (essays), and B changes the domain. I name them separately rather than calling both
simply "M4".

## 6.2 Test A: transfer to unseen generators (still essays)

The first test uses the OUTFOX essay set, where humans and six different models (GPT-4, ChatGPT,
Cohere, BLOOMz, Dolly, and davinci) all wrote essays. This keeps the kind of text the same as
training and only changes the generator. The detector held up well: an F1 of 0.97 (95% confidence
interval [0.97, 0.98], on about 4,800 essays so the interval is tight), and it caught every
generator at between 96 and 100 percent (Figure 6.1), with a human false-positive rate of about
5 percent. This is a useful and slightly surprising finding. A detector trained only on Llama still
recognises text from quite different models. The reason is that current large models share a lot of
the same smooth, uniform style, so the AI fingerprint is largely generator-agnostic, at least on
essays.

![Figure 6.1: Transfer to unseen generators on essays. A detector trained only on Llama still flags GPT-4, ChatGPT, Cohere, BLOOMz, Dolly and davinci at 96 to 100 percent.](../figures/fig_m4_per_generator.png)

## 6.3 Test B: transfer to unseen domains

The second test is the hard one, on the M4 monolingual data. Here the human text is not essays at
all: it is Reddit posts, WikiHow articles, arXiv abstracts, Wikipedia, and peer-review text, against
four generators. The overall F1 falls to 0.79 (95% confidence interval [0.77, 0.80], tight because
the sample is large), and the way it fails is the important part. The generator set here differs
from Test A, so the headline drop mixes a change of domain with a change of generators. The
load-bearing evidence is therefore not the combined F1 but the human side of the result, because a
false-positive rate measured on human text cannot be a generator artefact. The detector still
catches the machine text well in every domain (86 to 98 percent). What breaks is the human side: it
wrongly flags genuine human writing as AI at high rates on the more formal domains, 79 percent on
arXiv abstracts, about 40 percent on Wikipedia and WikiHow, 30 percent on peer-review, and 23
percent on Reddit (Figure 6.3). That human-side failure is what carries the argument.

![Figure 6.2: The in-domain score does not transfer everywhere. F1 stays high across generators on essays but falls on out-of-domain human and AI text.](../figures/fig_m4_transfer_gap.png)

![Figure 6.3: Cross-domain failure modes. The detector still catches AI text (orange) but wrongly flags formal human text as AI (teal), worst on arXiv abstracts.](../figures/fig_m4_per_domain.png)

## 6.4 What this means

The picture is clear and it matters for the project. The detector is robust across AI models but
fragile across domains, and the fragility is not random. It learned that "human" looks like a
student essay, so when it meets human writing that is more formal or more technical, an arXiv
abstract for example, it calls it AI. In other words, the out-of-domain failure is a
false-accusation failure, exactly the harm the project is trying to avoid.

This connects directly to the fairness concern from the start of the project. A detector trained
on one narrow idea of human writing will misjudge writers whose style sits outside that idea. The
arXiv result is the clearest warning: formal, dense, careful human prose reads as machine to this
model. The same mechanism is what makes these tools risky for non-native writers, whose style also
differs from the training norm. It is also the strongest argument for why the pipeline does not
stop at a score. A number that is confidently wrong on a whole domain is dangerous on its own; the
verification questions are what turn a flag into something a lecturer can check rather than act on
blindly.

## 6.5 Caveats and next steps

This is a zero-shot test with no adaptation, so it is a lower bound on what is achievable: a
detector trained on diverse human text and several generators would almost certainly do better
across domains, and that is a clear next experiment (train on a slice of M4 and re-test). The
in-domain "human" being essays is the root of the false positives, so the fix is more varied
human data and per-domain calibration. The remaining robustness gaps to test are paraphrased and
"humanised" AI text, and documents that mix human and AI writing. For the project as it stands,
the honest headline is that the detector is good at spotting AI essays from many models, and not
safe to point at human writing from a domain it was not trained on.

## 6.6 Does the balanced corpus design distort? The natural-distribution control

One design worry was agreed with my supervisor early (Meeting 2) and stayed open until now. The
detection corpus is deliberately balanced, equal essays per disciplinary-group-by-native cell, but
real submissions are nothing like that: in the full BAWE corpus, native Arts and Humanities writers
supply about 21 percent of essays and non-native AH writers about 4 percent. If the detector's strong
results depended on that artificial structure, the whole evaluation would be over-fitted to a corpus
shape that never occurs in practice. The control is two detectors trained in parallel from the same
cleaned corpus with identical hyperparameters, seeds and size (247 human essays each, plus their
matched AI twins), differing only in writer mix: one balanced (31 per cell), one matching full BAWE's
natural skew (53 native AH essays down to 10 non-native AH). Both are evaluated on the same untouched
test split, per cell, reweighted to the natural proportions, and with the fairness read
(`src/evaluation/balanced_vs_natural.py`, `outputs/balanced_vs_natural.json`).

The answer is that the balancing is not load-bearing (Figure 6.4). The two models score identically,
F1 1.000 overall, in every cell, and under natural reweighting, and they make the same prediction on
every one of the 200 test essays (zero disagreements, McNemar p = 1.0), with a zero false-positive
rate on human essays in both native-language groups. Read against Chapter 3 this is unsurprising:
the cleaned corpus is separable on function words alone, so 494 training essays saturate the test set
regardless of how their writers are distributed. Two conclusions follow, one reassuring and one
honest. The reassurance is that the balanced design did not manufacture the result; a realistically
skewed training mix reproduces it exactly, which is what the Meeting 2 guard was for. The honesty is
that at this level of separability the comparison has no resolution: whether balancing helps or hurts
can only be measured where the task is hard, such as the cross-domain settings of this chapter, and
that is where a future version of this control belongs.

![Figure 6.4: The training-distribution control. Two same-size detectors, one trained on the balanced writer mix and one on BAWE's natural skew, evaluated per cell and in aggregate on the same held-out split. They are indistinguishable everywhere, so the balanced design is not what makes the corpus separable.](../figures/fig_balanced_vs_natural.png)

## 6.7 The hybrid detector, and what fusion actually buys

The scope's first component is a hybrid: the transformer combined with the stylometric features,
including the GPT-2 perplexity signal (Radford et al., 2019) that Chapter 5 deferred because it needs
a GPU pass. Both halves have existed for some time at 0.990 and 0.985 in-domain, so assembling the
hybrid could not be about the headline number; the interesting question, given everything this
chapter found, is whether the fusion changes the detector's behaviour where it fails. The assembly is
simple and disclosed: perplexity joins the feature set, the feature model is retrained, and a
logistic regression fuses the two models' probabilities, fitted on the validation split so the test
set stays untouched, then applied zero-shot out of domain (`src/detection/hybrid_fusion.py`,
`outputs/hybrid_fusion.json`).

In-domain the answer is quick: everything sits at the ceiling. Adding perplexity lifts the feature
model from 0.985 to 1.000 on the 200-essay test split, the hybrid scores the same, and with two
essays' worth of movement separating all the arms these differences should not be over-read. The
detail worth keeping is that perplexity enters the feature model as its strongest single feature,
first of twenty-five by mean absolute SHAP, which confirms the literature's regard for it and
completes the feature set the scope specified.

Out of domain is where the fusion earns its place (Figure 6.5). On the same cross-domain sample as
Section 6.3, the three arms have almost identical F1 (transformer 0.790, hybrid 0.791), but they make
completely different errors. The transformer over-flags humans: recall 0.93 at precision 0.69, which
is the false-accusation profile. The style-plus-perplexity model is the mirror image, precision 0.82
at recall 0.72. The fusion balances the two (precision 0.80, recall 0.78) and lifts accuracy from
0.753 to 0.794, and the effect on the failure that matters is large: the human false-positive rate
falls in every domain, from 0.79 to 0.61 on arXiv abstracts, 0.30 to 0.09 on peer reviews, 0.23 to
0.05 on reddit, 0.41 to 0.10 on wikihow and 0.40 to 0.12 on wikipedia. In four of the five domains
false accusations drop by a factor of three to eight; arXiv remains the hard case, better but not
fixed. The style half is what does it: hand-crafted features generalise across registers where the
transformer's learned representation of "human" stays anchored to student essays.

![Figure 6.5: The hybrid detector. In-domain (left) every arm is at the ceiling and the differences are within noise. Out of domain (right) the fusion barely moves F1 but changes the error mix: the human false-positive rate, the false-accusation failure of Section 6.3, falls in every domain, by three to eight times in four of the five.](../figures/fig_hybrid_fusion.png)

The reading connects back to Section 6.4's mitigation plan. Fusion does not solve domain shift, and
the headline F1 barely moves, so this is not a robustness cure. What it does is redistribute the
remaining errors away from falsely accusing human writers, which is exactly the direction a deployed
system must err in, and it does so for the price of a feature extractor that runs anywhere. Component
1 of the scope is now complete as specified, and the hybrid, not the bare transformer, is the
detector the rest of the pipeline should call.
