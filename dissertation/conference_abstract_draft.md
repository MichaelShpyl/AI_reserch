# Conference abstract (draft)

Drafted after Meeting 2, where Dr. Vijayan suggested submitting a short abstract
(200 to 250 words) to an NLP or AI conference. This is a rough draft to get the
content down. **Rewrite it in my own voice before submitting.** This project is about
detecting AI-generated text, so my own abstract must not read as AI-generated.

## Working title

Explainable AI for Academic Integrity Verification: from detection to argument-aware
verification questions.

## Draft abstract (about 230 words)

Current detectors of AI-generated text return a single probability and little else. A
lecturer who flags a student essay cannot easily defend that score, and a probability
says nothing about whether the student understands the work they submitted. We present
an explainable pipeline that turns a detection flag into transparent, defensible
evidence for academic integrity verification. Detection combines a fine-tuned DeBERTa
transformer with stylometric features, including next-word-prediction perplexity and
burstiness, and every decision is explained with Integrated Gradients, SHAP and
attention, with faithfulness checked by feature ablation. For flagged work, an argument
mining stage extracts the student's claims and evidence and links each to its source
passage. These feed an argument-aware question generator that produces verification
questions a lecturer can ask in a short interview. A core research question compares a
commercial large language model against a locally fine-tuned open model (Llama 3 8B
with QLoRA), since cost and control matter for institution-scale deployment. A Bloom's
taxonomy classifier labels each question's cognitive level for quality control. We
evaluate question quality with a discrimination simulation that compares a context-aware
against a context-blind model answering each question, supported by multi-model
agreement, and report detection performance with a fairness analysis across native and
non-native writers. The data is public academic writing (BAWE) with length-matched and
topic-matched AI counterparts, under a two-class human versus AI design.

## Candidate venues (verify deadlines on the official call for papers)

Do not trust these dates. Check each official page before relying on it, the same rule
we use for paper citations.

- **AICS, the Irish Conference on Artificial Intelligence and Cognitive Science.** Best
  fit and confirmed as the right kind of venue: Ireland's main AI and cognitive science
  conference since 1988, held in December, with Springer proceedings. The 33rd edition
  (AICS 2025) was hosted by DCU at the Hyatt Centric Liberties, Dublin, on 1 to 2 December
  2025. AICS 2026 (the 34th) should fall in December 2026, but as of mid-June 2026 the
  2026 call for papers and its deadline are not yet announced (and the official site
  blocks automated checks). Action: watch https://aicsconf.org/, the AI Association of
  Ireland at https://aiai.ucd.ie/conferences.html, and WikiCFP
  (http://www.wikicfp.com/cfp/program?id=100) for the 2026 call. By the usual pattern a
  paper deadline around September or October 2026 is likely, but do not rely on that until
  the official call confirms it.
- **AIES, the AAAI/ACM Conference on AI, Ethics and Society.** Relevant to the fairness
  and academic-integrity angle. Check the 2026 dates at
  https://www.aies-conference.com/2026/call-for-papers/ (these often close in spring, so
  it may have passed).
- **BEA, the Workshop on Innovative Use of NLP for Building Educational Applications.**
  The ideal topical venue, but BEA 2026 already closed (submissions were due 23 March
  2026, co-located with ACL 2026 in July). Note it for BEA 2027.
- Also worth a look: European or Irish academic-integrity venues (for example the ENAI
  European Conference on Academic Integrity and Plagiarism) and SemEval-style shared
  tasks on machine-generated text detection, which the detector already builds on.

## Next steps

1. Rewrite the abstract in my own words and trim to the venue's word limit.
2. Confirm a venue and its real deadline from the official page.
3. Check whether the abstract needs co-author and affiliation details (supervisor as
   co-author is normal; confirm with Dr. Vijayan).

Sources: [AICS / aicsconf.org](https://aicsconf.org/),
[WikiCFP AICS](http://www.wikicfp.com/cfp/program?id=100),
[BEA 2026 call](https://sig-edu.org/news/bea21-call-for-papers/),
[AIES 2026 call](https://www.aies-conference.com/2026/call-for-papers/).
