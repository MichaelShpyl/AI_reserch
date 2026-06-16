# Auditing a Near-Perfect AI-Text Detector: A Corpus-Markup Artifact and What Survives

**Mykhailo Shpyl, supervised by Dr. Vini Vijayan**
Department of Computing, Atlantic Technological University, Donegal, Ireland

> DRAFT NOTE (delete before submission). This is a draft for the AICS Student Track, built from
> verified results in the project. TWO things must be done before submission. (1) Rewrite the
> prose in my own words; this is scaffolding, not final text. (2) Every citation below is a
> PLACEHOLDER marked "[verify: ...]". I must find the real paper, read it, and confirm it before
> it goes in. No citation here has been checked yet. Target format is CEURART single column,
> 6 to 12 pages including references. Submit via EasyChair.

## Abstract

Automatic detectors of AI-generated text are increasingly used in education, but they typically
return a single score with no defensible explanation. We build a transparent two-class
(human vs AI) detector and, more importantly, we audit it. On a balanced corpus of 640 human
essays from the British Academic Written English (BAWE) corpus and 640 length-matched and
topic-matched essays generated locally with Llama 3.1, a fine-tuned DeBERTa-v3 detector reached a
perfect test score. Rather than report it, we treated it as a warning. A reproducible audit found
that the human texts retained structural export markup that the AI texts lacked, so that a rule as
trivial as "contains a tag, therefore human" already reached 92.5 percent. After removing the
markup from both classes, the detector still scores an F1 of 0.99, and a classifier restricted to
function words alone reaches 99.5 percent, which shows the residual signal is writing style rather
than topic or formatting. An Integrated Gradients analysis with a faithfulness-by-ablation test
shows the style signal is diffuse, spread across ordinary words and punctuation, so per-word
highlights are a weak standalone explanation and a feature-level explanation is preferable. Tested
zero-shot on the M4 benchmark, the detector transfers across unseen generators (F1 0.97 on essays)
but is fragile across domains (F1 0.79), failing by flagging formal human text such as arXiv
abstracts as AI, which underlines the fairness risk of single-domain detectors. We argue that for
single-generator, in-domain detection a near-perfect score is expected, and that the useful
contributions are the audit method, the honest robustness picture, and an interpretable account of
what the detector uses.

## 1. Introduction

Generative writing tools are now part of how many students work, which has put pressure on how
institutions judge whether a submission is a student's own. Detectors of AI-generated text appeared
quickly, but most return a percentage with nothing behind it. A lecturer who is shown "82 percent
AI" cannot defend that number if a student challenges it, and the number says nothing about whether
the student understands what they submitted [verify: survey of machine-generated text detection,
2023 to 2025]. There is also evidence that such detectors are biased against non-native English
writers [verify: Liang et al., GPT detectors biased against non-native writers, 2023].

This paper focuses on the detection component of a larger explainable pipeline for academic
integrity verification. Our contribution here is not a new detector. It is a cautionary, reproducible
audit of a detector that scored perfectly, and an honest account of why it scored that way and what
survives once the easy shortcuts are removed.

## 2. Related work

Detection of machine-generated text falls into two broad families: zero-shot statistical methods that
use a language model's own probabilities, such as perplexity and curvature based detectors
[verify: DetectGPT, Mitchell et al., 2023; log-probability methods], and supervised transformer
classifiers fine-tuned to separate human from machine text [verify: RoBERTa/DeBERTa detector papers].
Shared tasks such as SemEval-2024 Task 8 / M4 provide multi-generator benchmarks [verify: Wang et al.,
M4, 2024]. A recurring weakness is brittleness to domain shift and paraphrasing [verify]. Stylometric
features (perplexity, burstiness, type-token ratio, part-of-speech distributions) have long been used
to characterise authorship and are increasingly combined with transformers [verify]. For
explainability we use Integrated Gradients [verify: Sundararajan et al., 2017] and evaluate it with
faithfulness criteria of comprehensiveness and sufficiency [verify: DeYoung et al., ERASER, 2020].

## 3. Data and method

**Human corpus.** We sampled 640 essays from BAWE [verify: BAWE corpus reference], stratified evenly
across four disciplinary groups and balanced between native and non-native English writers, with a
per-student cap and a student-level train, validation and test split so that no writer appears in
more than one split.

**AI corpus.** For each human essay we generated one AI essay on the same topic and at the same target
length, locally, using Llama 3.1 8B through Ollama. Topic anchoring used keywords extracted from the
human source, and a short continuation loop matched the target length. Length matching is essential:
without it a detector could separate the classes on length alone. The generated essays correlate with
their human sources at about 0.98 on length, with a mean ratio of about 1.05.

**Detector.** We fine-tuned DeBERTa-v3-base (with RoBERTa-base as a comparison) to classify human (0)
against AI (1), on a single 8 GB GPU using a small batch with gradient accumulation. We report
accuracy, precision, recall, F1, the confusion matrix, and a false-positive rate broken down by
first-language status.

## 4. The audit

The first detector reached F1 = 1.00 on the held-out test set. We audited this before reporting it
(`audit_detector.py`). Three findings follow.

**No leakage.** Across 394 students, none appeared in more than one split, every human and AI pair
shared a split, and no exact-duplicate text crossed train and test. The perfect score is not
contamination.

**A markup artifact.** The BAWE plain-text export retains structural tags such as heading, footnote
and list markers. About 88 percent of the human essays carried at least one; almost none of the AI
essays did. A rule that predicts human whenever a tag is present reaches 92.5 percent on the test set
on its own, with no language understanding. The AI side has the mirror problem, occasional markdown
that human plain text never contains. Both are artifacts of how the corpus halves were produced.

**What survives cleaning.** We normalised both classes (`text_normalize.py`), removing the export
tags and the markdown and flattening layout, and rebuilt the corpus. On cleaned text a TF-IDF and
logistic-regression model still separates the classes at 100 percent, and a model restricted to
function words alone reaches 99.5 percent. Function words carry no topic, so the residual signal is
writing style. The cleaned-text DeBERTa detector scores F1 = 0.99 (confusion matrix [[98, 2], [0,
100]]), with a native-writer false-positive rate of 0.04. RoBERTa agrees at F1 = 0.995.

## 5. Explainability and a faithfulness check

We applied Integrated Gradients to the cleaned-text detector and tested the result by ablation
(`integrated_gradients.py`). The most influential tokens are punctuation and common words and
fragments rather than topic words, consistent with the audit. The faithfulness test is the
interesting part. Removing the top attributed tokens lowers the detector's confidence only slightly
more than removing the same number of random tokens, and keeping only the top tokens collapses the
prediction to chance. The style signal is diffuse, spread across the whole essay, so a per-word
highlight is a weak explanation on its own. The faithful, interpretable explanation is the
feature-level one: the function-word fingerprint and the two-cluster style space from the audit.

## 6. Robustness: zero-shot transfer to M4

To test whether the in-domain score means anything beyond the training setting, we applied the
detector with no adaptation to the M4 benchmark, which spans many generators and domains the detector
never saw. On M4's essay split (human essays versus six unseen generators: GPT-4, ChatGPT, Cohere,
BLOOMz, Dolly, davinci) the detector reached F1 0.97 and flagged every generator at 96 to 100 percent,
so the AI-style fingerprint is largely generator-agnostic. On out-of-domain text (reddit, wikihow,
arxiv, wikipedia, peer-review) the F1 fell to 0.79, and the failure was one-sided: the detector still
caught machine text (86 to 98 percent) but wrongly flagged genuine human text as AI, at 79 percent on
arXiv abstracts and about 40 percent on Wikipedia and WikiHow. The detector had learned that "human"
looks like a student essay, so it misjudges more formal human writing. This is a false-accusation
failure, the exact harm the wider project aims to prevent, and it mirrors the known bias of these
detectors against writing that differs from the training norm.

## 7. Discussion

A near-perfect in-domain score is the expected outcome when separating one known generator from human
writing in one domain, which the literature treats as close to solved [verify]. The value here is
methodological. First, a strong result must be stress-tested before it is believed, and the markup
artifact is a clean example of why. Second, faithfulness testing tells us which explanation to trust,
and for this detector it rules out naive token highlighting in favour of feature-level explanation.
Third, the robustness picture is honest rather than flattering: strong across models, fragile across
domains, and fragile in the direction that hurts students. All three points generalise to anyone
building an interpretable detector on a constructed corpus.

## 8. Limitations and future work

The transfer test above is zero-shot, with no adaptation, so it is a lower bound: a detector trained
on diverse human text and several generators would likely do better across domains, and that is the
clear next experiment. The cross-domain false positives come from the in-domain "human" being student
essays, so the fix is more varied human data and per-domain calibration. We also found a locale tell:
the human students write British English while the model defaults to American, which is shallow and
would shrink if the generator were prompted for British English. The remaining robustness gaps to test
are paraphrased and "humanised" AI text and documents that mix human and AI writing. Beyond detection,
the wider pipeline (argument mining, source-grounded verification questions, a Bloom's-level check)
is future work. The project runs entirely on a laptop GPU, which constrains model sizes.

## 9. Conclusion

We presented an audit of a near-perfect AI-text detector. The perfect score was partly a corpus-markup
artifact, which we identified, measured and removed. After cleaning, the classes remain almost
perfectly separable on writing style alone, and a faithfulness test shows that style signal is diffuse
and best explained at the feature level. The honest framing, that in-domain detection is easy and the
real work is robustness and explanation, is the contribution we hope is useful to others.

## References

> Placeholders only. Find, read and verify each before submission; use 2021 to 2026 sources where
> possible, with foundational methods cited from their original papers.

- [verify] Survey of machine-generated text detection (2023 to 2025).
- [verify] Liang et al. (2023), GPT detectors are biased against non-native English writers.
- [verify] Mitchell et al. (2023), DetectGPT.
- [verify] Sundararajan, Taly and Yan (2017), Axiomatic Attribution for Deep Networks (Integrated
  Gradients).
- [verify] DeYoung et al. (2020), ERASER: faithfulness (comprehensiveness and sufficiency).
- [verify] Wang et al. (2024), M4 / SemEval-2024 Task 8.
- [verify] He et al., DeBERTa / DeBERTa-v3.
- [verify] BAWE (British Academic Written English) corpus reference.
