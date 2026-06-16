# Chapter 3: AI text detection (methodology and first results)

> Draft note (delete before submission): rough first-person draft for me to rewrite in my
> own words. It records what I actually did and what I found, including the mistake I caught,
> so the methodology and results chapters can be built from it. No em dashes, my own wording
> for the final version.

## 3.1 What this component has to do

The first stage of the pipeline decides whether a piece of submitted writing is the
student's own or was produced by a generative model. Everything downstream depends on it,
but the score on its own is not the point of the project. The point is a decision a lecturer
can defend. So I am as interested in why the model decides what it decides as in how often it
is right.

## 3.2 The detection corpus

I needed text that is labelled human or AI and that does not give the answer away for the
wrong reason. The human side is a stratified sample of 640 real student essays from BAWE
(the sampling is described in the sample design note). For the AI side I generated one essay
per human essay with Llama 3.1 8B running locally, matched on topic, keywords and target
length. Length matching matters: if the AI essays were systematically longer or shorter, a
detector could score well just by counting words and would learn nothing about style. After
generation the lengths line up closely (Pearson r about 0.98 between each AI essay and its
human source, mean length ratio about 1.05), so length is not the signal.

That gives 1,280 essays, 640 human and 640 AI. I split them at the level of the student, not
the essay, so no student appears in both training and test. If the same person's writing sat
on both sides of the split, the model could memorise that person rather than learn the
general difference. The split is 438 / 102 / 100 pairs for train / validation / test.

## 3.3 The detector

The detector is a fine-tuned transformer. DeBERTa-v3-base is the primary model and
RoBERTa-base is a comparison, both fine-tuned to classify human (0) against AI (1) with the
same settings and the same student-level splits. I report accuracy, precision, recall, F1
and the confusion matrix, plus a separate false-positive rate for native and non-native
English writers, because wrongly flagging a non-native student as AI is the failure mode I
most want to avoid. A stylometric feature model (perplexity, burstiness, vocabulary richness,
part-of-speech mix) is being added as the second half of the hybrid; this chapter is the
transformer half on its own.

## 3.4 The result, and why I did not trust it at first

The first trained detector scored a perfect F1 of 1.00 on the held-out test set, for both
DeBERTa and RoBERTa. A perfect score is not a result to celebrate, it is a result to
investigate. In a project that is about making detection defensible, presenting a number I
could not explain would be the opposite of the point. So before using it I ran an audit
(`src/detection/audit_detector.py`) to find out whether the score was real or a shortcut.

## 3.5 The audit

The audit asked four questions.

**Is the test set contaminated?** No. I checked that no student sits in more than one split,
that each human and AI pair stays in the same split, and that no exact-duplicate text crosses
from training into test. All three came back clean across 394 students. So the perfect score
is not the model having seen the answers.

**Is there a trivial giveaway in the text?** Yes, and this is the important finding. The
human essays come from the BAWE plain-text export, which keeps structural tags such as
`<heading>`, `<fnote>`, `<list>`, `<figure>` and `<quote>`. About 88 percent of the human
essays in my sample contain at least one of these tags. None of the AI essays do, because
Llama writes prose, not corpus markup. A rule as simple as "if the text contains a tag, call
it human" reaches 92.5 percent accuracy on the test set on its own, with no understanding of
language at all. The AI side has the mirror version of the same problem: Llama sometimes
writes markdown (bold, headings, bulleted and numbered lists) that human plain text never
has. Both of these are artefacts of how the two halves of the corpus were produced, not
differences in writing, and they sit right at the start of the text where the model's 512-token
window sees them. This is the shortcut behind a large part of the perfect score.

**Does anything real remain once the artefact is removed?** Yes. I wrote a normaliser
(`src/detection/text_normalize.py`) that strips the BAWE tags from the human side and the
markdown from the AI side and flattens the layout, then rebuilt the corpus from the cleaned
text. On the cleaned text a simple TF-IDF and logistic-regression model still separates the
two classes at 100 percent on the test set. More tellingly, a model that is allowed to use
only function words (the, and, because, therefore, and the like, which carry no topic
information at all) still reaches 99.5 percent. Topic cannot explain that. The thing that
remains after the markup is gone is writing style.

**What is the model keying on?** The cleaned linear model is interpretable, so I can read off
the words that push each way. The AI-leaning terms are the familiar large-model register: "in
conclusion", "nuanced", "essential", "highlights", "insights", "complex", "significant". The
human-leaning terms are blunter connectives: "therefore", "because", "so", "thus", "very".
That matches what the stylometric features already showed, that human writing here varies more
and the AI writing is smoother and more uniform.

## 3.6 The honest headline

After removing the artefact I retrained DeBERTa on the cleaned corpus. The score moved off
the ceiling, which is what I wanted: test accuracy 0.99, precision 0.98, recall 1.00, F1
0.990, with a confusion matrix of [[98, 2], [0, 100]]. Two human essays are flagged as AI and
no AI essay is missed. The native-writer false-positive rate is 0.04 and the non-native rate
is 0.00 on this small test set. The full-document linear probe still scores 1.00 on the same
cleaned text, while DeBERTa makes two mistakes, which fits the fact that the transformer only
reads the opening 512 tokens while the linear model sees the whole essay.

RoBERTa, trained the same way on the cleaned corpus, lands in the same place: F1 0.995, a
confusion matrix of [[99, 1], [0, 100]], native false-positive rate 0.02. Both models fall off
the perfect ceiling once the markup is gone, and both make their few mistakes the same way, by
flagging a small number of human essays as AI rather than missing AI. Two different
architectures agreeing is a sign that the residual signal is a real style difference and not a
quirk of one model.

So the cleaned F1 of 0.990 is the number I report, and the raw 1.000 is kept only to show how
large the artefact was. Either way the in-domain task is easy, which is consistent with the
literature: separating one known generator from human writing in one domain is close to a
solved problem. That is not the contribution of this project. The contribution is the
explanation behind each decision and, after this stage, robustness to the harder settings that
matter in practice, which are other generators, paraphrased AI text, and submissions that mix
human and AI writing. I expect the score to drop there, and that drop is where the real
research is.

## 3.7 Why the score is still high after cleaning

A fair question is why a cleaned score of 0.99 is not itself suspicious. I dug into this
(`src/detection/why_high.py`) so I can answer it rather than wave it away. The score is high
because the task as I have built it is the easy version of the problem, and several separate
signals reinforce each other.

The biggest reason is that all of my AI text comes from one model with one prompt. The AI
class is really "Llama 3.1 writing the way I asked it to", so it has a consistent style. When
I measure how similar essays are to each other in style space, the AI essays are more alike
(average similarity 0.60) than the human essays (0.56), because the humans are a crowd of
different people and the AI is one voice repeated 640 times. Separating one consistent voice
from a varied crowd is not hard, and the detection literature treats single-generator,
in-domain detection as close to solved.

On top of that sit several smaller tells that all point the same way. The vocabulary differs
(the AI register I described above). The spelling differs: my students write British English
(about 2.3 British spellings per thousand words) and Llama defaults to American (about 2.4
American spellings per thousand words). This one is real but shallow, and I am honest that it
is a generator-locale artefact rather than deep evidence of AI authorship; it would shrink if
I prompted the model to write British English, which is a fix I will try. Formality differs
too: the human essays use about three times as many contractions. None of these on its own is
decisive, but they stack, and a 2D picture of the essays in pure style space (Figure, function
words only) shows two clouds that barely touch. So 0.99 is the expected result for an easy
setting, not a hidden error.

The closest call is reassuring rather than worrying. The human essay the model rates most
AI-like scores 0.43 on the AI scale, so it is still correctly called human, and the borderline
cases are a small mix of native and non-native writers rather than a pile of non-native essays,
which is the bias I was watching for.

## 3.8 What this means for the next steps

The audit also seeds the explainability layer. The system can already point at the words that
drove a decision, which is the first concrete version of the defensible evidence the whole
project is built around. The immediate next steps are to fuse the stylometric features with
the transformer, to test the detector against the M4 multi-generator benchmark so I have an
honest cross-generator number, and to build the attribution methods (Integrated Gradients and
SHAP) on top of the model so the word-level explanation is faithful and not just a linear
proxy.
