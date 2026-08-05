# Literature review: verified sources

> Status: ALL 24 references fully verified (final audit 2 July 2026). Verification method, per entry:
> every arXiv-carrying reference (15 of 24) was checked programmatically against the arXiv API itself
> (title, author list, and year returned by export.arxiv.org compared to the entry); the ACL Anthology
> pages were fetched directly for RQUGE, SemEval-2024 Task 8, ERASER, and Jain and Wallace (author
> lists and page numbers taken from the Anthology, which corrected two earlier errors); PMLR was
> fetched for Sundararajan (pages 3319 to 3328 confirmed); dblp for DeBERTa (He, Liu, Gao, Chen, ICLR
> 2021 confirmed); and the remaining book/journal entries (Anderson and Krathwohl 2001; Hayes and
> Krippendorff 2007; Stab and Gurevych 2017; Lundberg and Lee 2017; Mitchell et al. 2023) were
> confirmed via publisher pages and dblp records found by direct search. Where a full author list
> could not be confirmed from a primary source, the entry lists the confirmed authors plus "et al."
> rather than guessing. The formal list lives in `chapters/09_references.md`; keep Zotero as the
> master copy going forward and regenerate the chapter from it for final submission.

## Detection of AI-generated text
- **Mitchell, Lee, Khazatsky, Manning, Finn (2023).** DetectGPT: Zero-Shot Machine-Generated Text
  Detection using Probability Curvature. ICML 2023. arXiv:2301.11305.
  Machine text sits in negative-curvature regions of a model's log-probability; a zero-shot curvature
  test detects it without a trained classifier.
- **Wang et al. (2024).** SemEval-2024 Task 8: Multidomain, Multimodel and Multilingual
  Machine-Generated Text Detection. Proc. SemEval-2024 (ACL). arXiv:2404.14183;
  https://aclanthology.org/2024.semeval-1.279/. The shared task (binary human-vs-machine, source
  attribution, and change-point) built on the M4 corpus. Used in this project for cross-generator and
  cross-domain robustness testing.
- **Wang et al. (2023).** M4: Multi-Generator, Multi-Domain, and Multi-Lingual Black-Box
  Machine-Generated Text Detection. arXiv:2305.14902. The corpus behind SemEval-2024 Task 8.
- **Krishna, Song, Karpinska, Wieting, Iyyer (2023).** Paraphrasing evades detectors of AI-generated
  text, but retrieval is an effective defense. NeurIPS 2023. arXiv:2303.13408. Their DIPPER paraphraser
  drops DetectGPT accuracy from 70.3% to 4.6%; motivates the robustness concern.
- **Wu, Yang, et al. (2025).** A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future
  Directions. Computational Linguistics 51(1):275-338. arXiv:2310.14724. Frames the detection families
  (watermarking, statistical, neural) and open problems (out-of-distribution, attacks, evaluation).

## Stylometric features
- (Liang et al. 2023, below, gives the perplexity/linguistic-variability evidence used here.)
- DONE (4 Aug 2026): Kumarage et al. 2023 supplies the hybrid stylometric-plus-transformer precedent; Opara 2024 the dedicated stylometry paper; Hans et al. 2024 the perplexity bar.

## Explainability and faithfulness
- **Lundberg, Lee (2017).** A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS 2017,
  pp. 4766-4777. Shapley-value feature attributions; the basis for the stylometric-feature explanation.
- **Sundararajan, Taly, Yan (2017).** Axiomatic Attribution for Deep Networks (Integrated Gradients).
  ICML 2017 (PMLR v70). Attribution satisfying Sensitivity and Implementation Invariance; used for the
  token-level attributions on the transformer detector.
- **DeYoung, Jain, Rajani, Lehman, Xiong, Socher, Wallace (2020).** ERASER: A Benchmark to Evaluate
  Rationalized NLP Models. ACL 2020. https://aclanthology.org/2020.acl-main.408/. Source of the
  comprehensiveness and sufficiency faithfulness metrics used in the ablation test.

## Argument mining
- **Stab, Gurevych (2017).** Parsing Argumentation Structures in Persuasive Essays. Computational
  Linguistics 43(3):619-659. https://aclanthology.org/J17-3005/. The Persuasive Essays corpus and the
  token-level sequence-labelling plus relation-classification approach this project builds on.

## Automatic question generation
- (RQUGE, below, is the closest verified anchor; add a neural/controllable QG and a local-vs-commercial
  LLM paper on the next pass.)

## Bloom's taxonomy and cognitive level
- **Anderson, Krathwohl (2001).** A Taxonomy for Learning, Teaching, and Assessing: A Revision of
  Bloom's Taxonomy of Educational Objectives. Longman, New York. The Remember-Understand-Apply-Analyse-
  Evaluate-Create scheme used to label questions.
- **Hadifar, Bitew, Deleu, Develder, Demeester (2022).** EduQG: A Multi-format Multiple Choice Dataset
  for the Educational Domain. arXiv:2210.06104 (also IEEE Access 2023). 3,397 questions, 903 tagged with
  a Bloom's cognitive level; in scope for the Bloom's classifier.

## Evaluating generated questions
- **Mohammadshahi et al. (2023).** RQUGE: Reference-Free Metric for Evaluating Question Generation by
  Answering the Question. Findings of ACL 2023. arXiv:2211.01482;
  https://aclanthology.org/2023.findings-acl.428/. Scores a question by whether a QA module can answer
  it from the context; the reference-free, answerability-based idea behind the discrimination
  simulation. Add LLM-as-judge and Krippendorff-alpha references next.

## Fairness and bias in detection
- **Liang, Yuksekgonul, Mao, Wu, Zou (2023).** GPT detectors are biased against non-native English
  writers. Patterns 4(7):100779. arXiv:2304.02819. Seven detectors flagged 61.3% of TOEFL essays by
  non-native writers as AI, versus about 5.1% for native writers; the direct evidence for the fairness
  analysis and the false-accusation finding.

## Detector model
- **He, Liu, Gao, Chen (2021).** DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR 2021.
  https://openreview.net/forum?id=XPZIaotutsD. The transformer architecture used as the project's
  primary detector.
- **Liu, Ott, Goyal, et al. (2019).** RoBERTa: A Robustly Optimized BERT Pretraining Approach.
  arXiv:1907.11692. The comparison detector.

## Second pass (verified 1 July 2026)
- **Jain, Wallace (2019).** Attention is not Explanation. NAACL 2019. https://aclanthology.org/N19-1357/;
  arXiv:1902.10186. Attention weights do not reliably explain predictions; a reason this project does not
  rely on attention. [Explainability]
- **Zheng, Chiang, Sheng, et al. (2023).** Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
  NeurIPS 2023. arXiv:2306.05685. Strong LLM judges match human preference about 80% of the time but show
  position, verbosity, and self-enhancement biases; frames the supplementary LLM-as-judge evaluation.
  [Question evaluation]
- **Hayes, Krippendorff (2007).** Answering the Call for a Standard Reliability Measure for Coding Data.
  Communication Methods and Measures 1:77-89. Krippendorff's alpha as a general inter-rater reliability
  measure; used for cross-model agreement. [Question evaluation]
- **Guo, Liao, Li, Chua (2024).** A Survey on Neural Question Generation: Methods, Applications, and
  Prospects. IJCAI 2024. arXiv:2402.18267. Organises neural QG (structured, unstructured, hybrid); the
  survey anchor for the question-generation section. [Question generation]
- **Kumar, Gulwani, Singh (2025).** Automated Analysis of Learning Outcomes and Exam Questions Based on
  Bloom's Taxonomy. arXiv:2511.10903. Compares classical, transformer, and LLM approaches to automatic
  Bloom-level classification. [Bloom's classification]
- **Jin, Yan, Echeverria, Gašević, Martinez-Maldonado (2024).** Generative AI in Higher Education: A
  Global Perspective of Institutional Adoption Policies and Guidelines. arXiv:2405.11800. Adoption of
  generative AI across 40 universities; the background citation for uptake in higher education.
  [Background]

## Still to add (markers left in the chapters)
- Hybrid stylometric-plus-transformer detector (Chapter 2.3).
- A recent transformer-based argument-mining paper beyond Stab and Gurevych (Chapter 2.5).
- A general local-versus-commercial / open-versus-proprietary LLM evaluation (Chapters 1.1 and 2.6); the
  comparisons found so far are domain-specific (for example medical), so a cleaner general reference is
  still wanted.

## Method references added 7 July 2026 (all verified against primary sources)

Each entry below was verified by locating the actual publication before it entered Chapter 9.

- Dettmers et al. (2023), QLoRA. Verified: NeurIPS 2023 proceedings page (oral) + arXiv:2305.14314.
- Hu et al. (2022), LoRA. Verified: arXiv:2106.09685 + dblp record for ICLR 2022 (OpenReview nZeVKeeFYf9).
- Rajpurkar et al. (2016), SQuAD. Verified: ACL Anthology D16-1264 (DOI 10.18653/v1/D16-1264), pp. 2383 to 2392.
- Yang et al. (2024), Qwen2.5 Technical Report. Verified: arXiv:2412.15115 (collective author "Qwen"; v2 Jan 2025).
- Grattafiori et al. (2024), The Llama 3 Herd of Models. Verified: arXiv:2407.21783 (v1 listed Dubey first; current v3 lists Grattafiori).
- Radford et al. (2019), GPT-2 report. Verified: the OpenAI-hosted PDF (no arXiv version exists; technical report).
- Devlin et al. (2019), BERT. Verified: ACL Anthology N19-1423, NAACL-HLT 2019, pp. 4171 to 4186.
- Alsop and Nesi (2009), the BAWE corpus article. Verified: CrossRef metadata for DOI 10.3366/E1749503209000227, Corpora 4(1), pp. 71 to 83. The dataset itself is OTA record 2539 (Nesi, Gardner, Thompson, Wickens; CC BY-NC-SA 3.0).
- Nussbaum et al. (2024), Nomic Embed. Verified: arXiv:2402.01613 (later published in TMLR 02/2025; the arXiv preprint is cited).

## Expansion pass, 4 August 2026 (all verified against the arXiv API before use)

Method: each candidate's arXiv id was queried against export.arxiv.org and the returned title and
author list compared with the claimed entry. 39 of 39 arXiv candidates passed; 8 were already in
the reference list. Four author attributions in draft prose were corrected as a result.

- **Abkenar et al. (2024).** Assessing Open-Source Large Language Models on Argumentation Mining Subtasks. arXiv:2411.05639.
- **Beale et al. (2025).** Adapting University Policies for Generative AI: Opportunities, Challenges, and Policy Solutions in Higher Education. arXiv:2506.22231.
- **Church et al. (2025).** Using LLMs to support assessment of student work in higher education: a viva voce simulator. arXiv:2511.05530.
- **Delphino et al. (2025).** Assessing the Prevalence of AI-assisted Cheating in Programming Courses: A Pilot Study. arXiv:2507.06438.
- **Dik et al. (2025).** Assessing GPTZero's Accuracy in Identifying AI vs. Human-Written Essays. arXiv:2506.23517.
- **Favero et al. (2025).** Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessment. arXiv:2502.14389.
- **Feuer et al. (2024).** Style Outweighs Substance: Failure Modes of LLM Judges in Alignment Benchmarking. arXiv:2409.15268.
- **Fu et al. (2024).** QGEval: Benchmarking Multi-dimensional Evaluation for Question Generation. arXiv:2406.05707.
- **Gorichanaz et al. (2023).** Accused: How students respond to allegations of using ChatGPT on assessments. arXiv:2308.16374.
- **Hans et al. (2024).** Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text. arXiv:2401.12070.
- **Ipeirotis et al. (2026).** Scalable and Personalized Oral Assessments Using Voice AI. arXiv:2603.18221.
- **Kumarage et al. (2023).** Stylometric Detection of AI-Generated Text in Twitter Timelines. arXiv:2303.03697.
- **Lee et al. (2025).** Beyond Static Scoring: Enhancing Assessment Validity via AI-Generated Interactive Verification. arXiv:2512.12592.
- **Li et al. (2025).** Large Language Models in Argument Mining: A Survey. arXiv:2506.16383.
- **Liusie et al. (2022).** World Knowledge in Multiple Choice Reading Comprehension. arXiv:2211.07040.
- **Nguyen et al. (2024).** Reference-based Metrics Disprove Themselves in Question Generation. arXiv:2403.12242.
- **Norman et al. (2026).** Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias. arXiv:2606.19544.
- **Opara et al. (2024).** StyloAI: Distinguishing AI-Generated Content with Stylometric Analysis. arXiv:2405.10129.
- **Perkins et al. (2024).** GenAI Detection Tools, Adversarial Techniques and Implications for Inclusivity in Higher Education. arXiv:2403.19148.
