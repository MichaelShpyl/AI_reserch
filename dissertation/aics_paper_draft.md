# Auditing a Near-Perfect AI-Text Detector: A Corpus-Markup Artifact and What Survives

**Mykhailo Shpyl, supervised by Dr. Vini Vijayan**
Department of Computing, Atlantic Technological University, Donegal, Ireland

> Working draft for the AICS Student Track. Every citation has been verified against its primary
> source (the same verified list as the dissertation's reference chapter). Before submission: a
> final prose polish, format to CEURART single column (6 to 12 pages including references), and
> submit via EasyChair once the call details are confirmed with my supervisor.

## Abstract

Automatic detectors of AI-generated text are increasingly used in education, but they typically
return a single score with no defensible explanation. We build a transparent two-class
(human vs AI) detector and, more importantly, we audit it. On a balanced corpus of 640 human
essays from the British Academic Written English (BAWE) corpus and 640 length-matched and
topic-matched essays generated locally with Llama 3.1, a fine-tuned DeBERTa-v3 detector reached a
perfect test score. We treated the score as a warning and audited it before reporting anything.
The audit found
that the human texts retained structural export markup that the AI texts lacked, so that a rule as
trivial as "contains a tag, therefore human" already reached 92.5 percent. After removing the
markup from both classes, the detector still scores an F1 of 0.99, and a classifier restricted to
function words alone reaches 99.5 percent, which shows the residual signal is writing style rather
than topic or formatting. An Integrated Gradients analysis with a faithfulness-by-ablation test
shows the style signal is diffuse, spread across ordinary words and punctuation, so per-word
highlights are a weak standalone explanation and a feature-level explanation is preferable. Tested
zero-shot on the M4 benchmark, the detector transfers across unseen generators (F1 0.97 on essays)
but is fragile across domains (F1 0.79), failing by flagging formal human text such as arXiv
abstracts as AI, which underlines the fairness risk of single-domain detectors. We argue that a
near-perfect score is the expected outcome for single-generator, in-domain detection. The useful
contributions are the audit method, the robustness picture, and an account of what the detector
actually uses.

## 1. Introduction

Generative writing tools are now part of how many students work, which has put pressure on how
institutions judge whether a submission is a student's own. Detectors of AI-generated text appeared
quickly, but most return a percentage with nothing behind it. A lecturer who is shown "82 percent
AI" cannot defend that number if a student challenges it, and the number says nothing about whether
the student understands what they submitted (Wu et al., 2025). There is also evidence that such
detectors are biased against non-native English writers (Liang et al., 2023).

This paper focuses on the detection component of a larger explainable pipeline for academic
integrity verification. Our contribution is a cautionary, reproducible audit of a detector that
scored perfectly: why it scored that way, and what survives once the easy shortcuts are removed.

## 2. Related work

Detection of machine-generated text falls into two broad families: zero-shot statistical methods that
use a language model's own probabilities, such as perplexity and curvature based detectors
(Mitchell et al., 2023), and supervised transformer classifiers fine-tuned to separate human from
machine text (Liu et al., 2019; He et al., 2021). Shared tasks such as SemEval-2024 Task 8 and its
M4 corpus provide multi-generator benchmarks (Wang et al., 2023; Wang et al., 2024). A recurring
weakness is brittleness to domain shift and paraphrasing (Krishna et al., 2023). Stylometric
features (perplexity, burstiness, type-token ratio, part-of-speech distributions) have long been
used to characterise authorship and are increasingly combined with transformers (Mindner et al.,
2023). For explainability we use Integrated Gradients (Sundararajan et al., 2017) and evaluate it
with the faithfulness criteria of comprehensiveness and sufficiency (DeYoung et al., 2020).

## 3. Data and method

**Human corpus.** We sampled 640 essays from BAWE (Alsop and Nesi, 2009), stratified evenly
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
logistic-regression model still separates the classes at 100 percent (with one residual tag token
surviving into its vocabulary, so we rely on the next result), and a model restricted to function
words alone reaches 99.5 percent. Function words carry no topic and no markup, so the residual signal
is writing style. We also controlled for the British-versus-American spelling and contraction tells:
removing every spelling variant and contraction leaves the function-word accuracy unchanged
(0.995 to 1.00), so the locale tell is a real but non-load-bearing add-on. The cleaned-text DeBERTa
detector scores F1 = 0.99 (confusion matrix [[98, 2], [0, 100]]); on a 200-essay test set the 95%
bootstrap interval is about [0.97, 1.00], so we read it as "about 0.99". RoBERTa scores 0.995, but
its interval overlaps DeBERTa's almost completely, so the two are statistically indistinguishable.
The native-writer false-positive rate is 2 of 50 essays, too few to claim fairness; the cross-domain
result in Section 6 is the stronger fairness evidence.

## 5. Explainability and a faithfulness check

We applied Integrated Gradients to the cleaned-text detector and tested the result by ablation
(`integrated_gradients.py`). The most influential tokens are punctuation and common words and
fragments rather than topic words, consistent with the audit. The faithfulness test changed how we
read those attributions. Removing the top attributed tokens lowers the detector's confidence only slightly
more than removing the same number of random tokens, and keeping only the top tokens collapses the
prediction to chance. The style signal is diffuse, spread across the whole essay, so a per-word
highlight is a weak explanation on its own. The faithful, interpretable explanation is the
feature-level one: the function-word fingerprint and the two-cluster style space from the audit.
A detector built from hand-crafted style features alone (sentence-length variation, vocabulary
richness, word length, punctuation, part-of-speech mix) reaches F1 0.985, almost matching the
transformer, and SHAP attributes each decision to named features (longer words push toward AI;
richer vocabulary and more varied sentence length push toward human), giving the defensible,
lecturer-facing explanation the project is built around.

## 6. Robustness: zero-shot transfer

To test whether the in-domain score means anything beyond the training setting, we applied the
detector with no adaptation to two held-out corpora that span generators and domains it never saw.
On the OUTFOX essay set (human essays versus six unseen generators: GPT-4, ChatGPT, Cohere, BLOOMz,
Dolly, davinci) the detector reached F1 0.97 and flagged every generator at 96 to 100 percent, so the
AI-style fingerprint is largely generator-agnostic. On out-of-domain M4 / SemEval-2024 Task 8 text
(reddit, wikihow,
arxiv, wikipedia, peer-review) the F1 fell to 0.79. Because the generator set here also differs from
the OUTFOX test, the combined F1 mixes domain shift with generator shift, so the load-bearing evidence
is the one-sided failure on the human side, which cannot be a generator artefact: the detector still
caught machine text (86 to 98 percent) but wrongly flagged genuine human text as AI, at 79 percent on
arXiv abstracts and about 40 percent on Wikipedia and WikiHow. The detector had learned that "human"
looks like a student essay, so it misjudges more formal human writing. These failures are false
accusations, the harm the wider project is designed to prevent, and they mirror the known bias of
these detectors against writing that differs from the training norm.

## 7. Discussion

A near-perfect in-domain score is the expected outcome when separating one known generator from human
writing in one domain, which the recent survey literature treats as close to solved (Wu et al.,
2025). The value here is
methodological. First, a strong result must be stress-tested before it is believed, and the markup
artifact is a clean example of why. Second, faithfulness testing tells us which explanation to trust,
and for this detector it rules out naive token highlighting in favour of feature-level explanation.
Third, the robustness picture: strong across models, fragile across domains, with the failures
landing on human writers. All three points generalise to anyone building an interpretable detector
on a constructed corpus.

## 8. Limitations and future work

The transfer test above is zero-shot, with no adaptation, so it is a lower bound: a detector trained
on diverse human text and several generators would likely do better across domains, and that is the
clear next experiment. The cross-domain false positives come from the in-domain "human" being student
essays, so the fix is more varied human data and per-domain calibration. Two follow-up measurements
in the wider project support this reading: fusing the transformer with the stylometric features cuts
the cross-domain human false-positive rate by three to eight times (arXiv abstracts from 79 to about
61 percent), and on new test essays generated by two further commercial models (Gemini and GPT) the
detector caught every AI essay with no human false positives, consistent with the generator-agnostic
result above. We also found a locale tell:
the human students write British English while the model defaults to American, which is shallow and
would shrink if the generator were prompted for British English. The remaining robustness gaps to test
are paraphrased and "humanised" AI text and documents that mix human and AI writing. Beyond detection,
the wider pipeline (argument mining, source-grounded verification questions, a Bloom's-level check)
is future work. The project runs entirely on a laptop GPU, which constrains model sizes.

## 9. Conclusion

We presented an audit of a near-perfect AI-text detector. The perfect score was partly a corpus-markup
artifact, which we identified, measured and removed. After cleaning, the classes remain almost
perfectly separable on writing style alone, and a faithfulness test shows that style signal is diffuse
and best explained at the feature level. Our conclusion is that in-domain detection of a single known
generator is easy, and the work that matters is robustness and explanation. We hope the audit method
travels to other constructed corpora.

## References

Alsop, S. and Nesi, H. (2009). Issues in the Development of the British Academic Written English (BAWE) Corpus. Corpora, 4(1), pp. 71 to 83.

DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R. and Wallace, B. C. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. In Proceedings of ACL 2020, pp. 4443 to 4458.

He, P., Liu, X., Gao, J. and Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. In Proceedings of ICLR 2021.

Krishna, K., Song, Y., Karpinska, M., Wieting, J. and Iyyer, M. (2023). Paraphrasing Evades Detectors of AI-Generated Text, but Retrieval is an Effective Defense. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023).

Liang, W., Yuksekgonul, M., Mao, Y., Wu, E. and Zou, J. (2023). GPT Detectors Are Biased Against Non-Native English Writers. Patterns, 4(7), 100779.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L. and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.

Mindner, L., Schlippe, T. and Schaaff, K. (2023). Classification of Human- and AI-Generated Texts: Investigating Features for ChatGPT. arXiv:2308.05341.

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D. and Finn, C. (2023). DetectGPT: Zero-Shot Machine-Generated Text Detection Using Probability Curvature. In Proceedings of ICML 2023, PMLR 202, pp. 24950 to 24962.

Sundararajan, M., Taly, A. and Yan, Q. (2017). Axiomatic Attribution for Deep Networks. In Proceedings of ICML 2017, PMLR 70, pp. 3319 to 3328.

Wang, Y., Mansurov, J., Ivanov, P., Su, J., Shelmanov, A., Tsvigun, A., et al. (2023). M4: Multi-Generator, Multi-Domain, and Multi-Lingual Black-Box Machine-Generated Text Detection. arXiv:2305.14902.

Wang, Y., Mansurov, J., Ivanov, P., Su, J., Shelmanov, A., Tsvigun, A., Mohammed Afzal, O., Mahmoud, T., Puccetti, G., Arnold, T., et al. (2024). SemEval-2024 Task 8: Multidomain, Multimodel and Multilingual Machine-Generated Text Detection. In Proceedings of SemEval-2024, pp. 2057 to 2079.

Wu, J., Yang, S., Zhan, R., Yuan, Y., Chao, L. S. and Wong, D. F. (2025). A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions. Computational Linguistics, 51(1), pp. 275 to 338.
