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
only once a fair number are removed. The sufficiency result is starker: keeping only the top tokens
drops the detector to a coin-flip, whatever number I keep. In other words, no small set of words is
enough to carry the decision.

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

## 5.5 Next steps for explainability

Three things follow. First, add SHAP on the stylometric features once the hybrid detector is built,
so the feature-level explanation has per-feature attributions to sit alongside the lexical view.
Second, keep the token attributions as a supporting visual, useful for showing a lecturer roughly
where the model looked, but always paired with the honest caveat from the faithfulness test. Third,
carry this faithfulness check forward as a standard step: any explanation the system shows should be
one I have tested, not one I have assumed.
