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
