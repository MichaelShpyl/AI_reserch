# Chapter 5: Explaining the detector's decisions

> Draft note (delete before submission): rough first-person draft for me to rewrite in my own
> words. It reports the first explainability results and one finding I did not expect, that the
> token-level explanation is only weakly faithful because the signal is diffuse. No em dashes;
> my own wording for the final version.

## 5.1 Why this matters for the project

The whole point of the project is that a detector should hand a lecturer defensible evidence,
not just a percentage. So the detector is only half the story. The other half is being able to
say why a piece of text was flagged, and being honest about whether that explanation is real.
This chapter is the first version of the explainability layer. It does two things: it shows which
parts of an essay drove the decision, and then it tests whether that explanation is faithful,
which means whether the words it points to are actually the words the model used.

## 5.2 Token attributions with Integrated Gradients

For the word-level view I used Integrated Gradients, a standard attribution method, through the
Captum library (`src/explainability/integrated_gradients.py`). It works on the detector's word
embeddings and measures how much each token pushes the decision toward the AI class, compared to
a neutral baseline where the content is removed. I attribute everything to the AI class so the
sign is consistent: a positive score pushes toward AI, a negative score pushes toward human.

Figure 5.1 shows this for a matched pair, the AI and human versions of the same essay. Two things
stand out. The detector is confident and in the right direction for both (the AI essay scores 1.00
on the AI scale, the human essay 0.01). And the tokens that carry the most weight are not topic
words. They are punctuation and very common words and word-fragments: full stops, commas, a dash,
words like "have", "be", "can", "similarly". That fits everything from the audit. The model is
reading style and rhythm, the small machinery of how a sentence is put together, rather than what
the essay is about.

![Figure 5.1: Integrated Gradients token attributions for a matched AI and human essay. Orange pushes toward AI, teal toward human. The strongest tokens are punctuation and common words, not topic words, so even at the word level the detector is reading style.](../figures/fig_explain_ig_tokens.png)

## 5.3 Are the highlights honest? A faithfulness test

A highlight is only useful if it is faithful. I tested this by ablation (the same script). For a
sample of test essays I ranked the tokens by their attribution and then asked two questions.
First, comprehensiveness: if I remove the top tokens, does the detector's confidence fall, and
does it fall more than if I remove the same number of random tokens? Second, sufficiency: if I
keep only the top tokens and hide the rest, does the confidence hold up?

Figure 5.2 shows the comprehensiveness result as I remove more and more tokens. Removing the
top-ranked tokens does hurt the detector more than removing random ones, but only by a little, and
only once a fair number are removed. This comprehensiveness sweep is the load-bearing, honest part
of the test. I also ran a sufficiency check (keep only the top tokens, hide the rest), but with at
most a few dozen of 256 tokens kept the model sees almost nothing and sits at a coin-flip for every
k, so that flat result is an empty-input artefact of the design rather than a finding, and I do not
rely on it. The comprehensiveness result on its own already says what matters: no small set of words
is enough to carry the decision.

![Figure 5.2: Faithfulness by ablation. Removing the top attributed tokens (orange) lowers detector confidence only slightly more than removing random tokens (teal). Keeping only the top tokens collapses the prediction to chance. The signal is spread across the whole essay.](../figures/fig_explain_faithfulness.png)

## 5.4 What this tells me: the signal is diffuse

This is an honest and useful negative result. The reason a few words cannot explain the decision
is that the difference between human and AI writing here is spread across the whole essay, in the
ordinary words and the punctuation, not concentrated in a handful of give-away terms. It is the same
thing the audit found from the other direction: a model using only function words still separated
the classes at 99.5 percent, and the essays form two clean clouds in pure style space. A diffuse
signal is hard to summarise with a word-level highlight, so for this detector a per-word heatmap is
a weak explanation on its own.

The explanation that is faithful for this detector is the feature-level one. The interpretable
linear view from the audit (Chapter 3) names the style markers that separate the classes, the
large-model register on one side and the blunter connectives on the other, and that view is backed
by the function-words-only result rather than contradicted by it. So the defensible, lecturer-facing
explanation is the style and feature picture, supported by the source-grounded verification
questions that come later in the pipeline, not a single highlighted sentence.

## 5.5 The faithful explanation: stylometric features and SHAP

To make the feature-level explanation concrete I built a detector from hand-crafted style features
alone, with no transformer: sentence-length variation and burstiness, vocabulary richness, word
length, punctuation, and the part-of-speech mix (`src/explainability/shap_stylometric.py`). Trained
on the same student-level splits, this transparent model reaches an F1 of 0.985 on the test set
(95% bootstrap confidence interval about [0.97, 1.00], and identical, 0.985, across five training
seeds, so it is stable). Two honest caveats. The comparison with the transformer's 0.99 is not
quite like for like: the feature model reads the whole essay while DeBERTa reads only the first 512
tokens, which helps the feature model, so I read this as "competitive" rather than "equal". And the
false-positive rate of 0.02 for native and non-native writers is one essay in fifty each, too small
to claim fairness; the cross-domain result in Chapter 6 is the real fairness evidence. Two of the
features (type-token ratio and the rare-word ratio) are length-sensitive, but since the essay lengths
are matched across the classes this carries little class signal; the length-robust vocabulary measure
behaves the same way. Perplexity, one of the strongest stylometric signals in the literature, is
implemented but not yet in this model (it needs a GPU pass) and is deferred to the hybrid. Even so,
the headline holds: the signal is captured almost as well by a handful of interpretable features as
by a large model.

Because the model is built from named features, I can explain it faithfully with SHAP, which
attributes each decision to the features that drove it. Figure 5.3 shows the picture across the test
set. Longer words and a denser use of auxiliary verbs push a text toward AI, while more
sentence-length variation, richer vocabulary, and more rare one-off words push it toward human. This
is the explanation the project actually needs: a lecturer can be told that a piece was flagged
because its sentences are unusually uniform, its words longer, and its vocabulary less varied than a
typical student's, and that account is faithful because the model literally uses those features. It
is the opposite of the black-box percentage the project set out to replace.

![Figure 5.3: SHAP on the stylometric detector. Each dot is an essay; position shows how much a feature pushed the decision toward AI (right) or human (left), and colour shows the feature value. Longer words push toward AI; richer vocabulary and more varied sentence length push toward human.](../figures/fig_shap_stylometric.png)

This stylometric model is also the non-transformer half of the planned hybrid detector, so this step
serves both the explainability layer and the detector itself.

## 5.6 Next steps for explainability

Three things follow. First, fuse the stylometric model with the transformer into the hybrid detector,
and add perplexity (a strong feature that needs the GPU) to the feature set. Second, keep the token
attributions as a supporting visual, useful for showing a lecturer roughly where the model looked,
but always paired with the honest caveat from the faithfulness test, with the SHAP feature view as
the primary explanation. Third, carry the faithfulness check forward as a standard step: any
explanation the system shows should be one I have tested, not one I have assumed.
