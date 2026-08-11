# Conference abstract

Drafted after Meeting 2, where Dr. Vijayan suggested submitting a short abstract (200 to 250 words)
to an NLP or AI conference. **Rewritten 10 August 2026**, once the project was finished, because
the first version described what I intended to build rather than what I found. This project is
about detecting AI-generated text, so my own abstract must not read as AI-generated.

Every number below is read from a results file in `outputs/`. The repository is public at
https://github.com/MichaelShpyl/AI_reserch.

## Working title

From detection to conversation: explainable AI-text detection with argument-aware verification
questions.

## Abstract (244 words)

Detectors of AI-generated text return a probability and little else. A lecturer cannot defend that
number if a student challenges it, and it says nothing about whether the student understands the
work they submitted. This paper reports a pipeline that turns a flag into evidence for a
conversation, running end to end on one 8 GB laptop.

Detection fuses a fine-tuned DeBERTa with a gradient-boosting model over 23 stylometric features
and GPT-2 perplexity, trained on 640 academic essays paired with AI essays matched on topic and
length. It reaches F1 0.99 in domain and 0.97 across six unseen generators. Out of domain it is unsafe alone: 79 percent of human arXiv abstracts are falsely flagged, which
fusion reduces to 61 percent rather than fixing. That failure is why the pipeline continues past
the flag.

Explanations were tested rather than asserted. Under ablation, Integrated Gradients and attention
both failed; a feature-level SHAP account passed, and only that is shown to a lecturer. An argument
miner then extracts the student's claims, keeping sentence numbers so quoted text is looked up from
the submission and cannot be invented, and a generator writes verification questions from them.

Question quality is measured by a judge-free discrimination simulation. On 30 essays with claim
sets held fixed, a QLoRA fine-tune of Qwen2.5 3B beats a commercial tier on 24 of 29 essays
(p = 0.0003). A three-judge LLM panel, validated against that measure, failed: Krippendorff's
alpha -0.14, with negative correlations throughout.

## What changed from the first draft, and why

The June draft was a proposal written before the results existed. Four things in it were wrong by
the end, and three of them stated a finding backwards:

- **Backend.** It said Llama 3 8B with QLoRA. Fit probes measured 9.8 GB peak and 214 seconds a
  step on an 8 GB card, against 5.2 GB and 2.8 seconds for Qwen2.5 3B. My supervisor approved the
  change on that evidence on 3 July, and the 3B is what the result rests on.
- **Explainability.** It listed Integrated Gradients, SHAP and attention as the explanation methods,
  "with faithfulness checked by feature ablation". The ablation is the point: two of the three
  failed it. Presenting all three as the method inverts the finding.
- **Judges.** It offered "multi-model agreement" as supporting evidence. The panel failed its own
  validation three ways. It is a negative result, not support.
- **No results.** The strongest thing in the paper is that three of my own headline numbers were
  retracted after checking. An abstract with no numbers cannot say that.

## Candidate venues (verify deadlines on the official call for papers)

Do not trust these dates. Check each official page before relying on it, the same rule I use for
paper citations.

- **AICS, the Irish Conference on Artificial Intelligence and Cognitive Science.** Best fit and
  confirmed as the right kind of venue: Ireland's main AI and cognitive science conference since
  1988, held in December, with Springer proceedings. The 33rd edition (AICS 2025) was hosted by DCU
  at the Hyatt Centric Liberties, Dublin, on 1 to 2 December 2025. AICS 2026 (the 34th) should fall
  in December 2026, but as of mid-June 2026 the 2026 call for papers and its deadline were not yet
  announced (and the official site blocks automated checks). Action: watch https://aicsconf.org/,
  the AI Association of Ireland at https://aiai.ucd.ie/conferences.html, and WikiCFP
  (http://www.wikicfp.com/cfp/program?id=100) for the 2026 call. By the usual pattern a paper
  deadline around September or October 2026 is likely, but do not rely on that until the official
  call confirms it.
- **AIES, the AAAI/ACM Conference on AI, Ethics and Society.** Relevant to the fairness and
  academic-integrity angle. Check the 2026 dates at
  https://www.aies-conference.com/2026/call-for-papers/ (these often close in spring, so it may
  have passed).
- **BEA, the Workshop on Innovative Use of NLP for Building Educational Applications.** The ideal
  topical venue, but BEA 2026 already closed (submissions were due 23 March 2026, co-located with
  ACL 2026 in July). Note it for BEA 2027.
- Also worth a look: European or Irish academic-integrity venues (for example the ENAI European
  Conference on Academic Integrity and Plagiarism) and SemEval-style shared tasks on
  machine-generated text detection, which the detector already builds on.

## Next steps

1. Confirm a venue and its real deadline from the official page.
2. Trim to the venue's word limit. At 244 words the abstract fits a 250 limit; a 200-word venue
   means cutting the out-of-domain sentence, which is the one I would least like to lose.
3. Check whether the abstract needs co-author and affiliation details. A supervisor as co-author is
   normal; confirm with Dr. Vijayan.
4. Decide whether to submit before or after the 28 August dissertation deadline. Nothing in the
   abstract depends on work that is still outstanding.

Sources: [AICS / aicsconf.org](https://aicsconf.org/),
[WikiCFP AICS](http://www.wikicfp.com/cfp/program?id=100),
[BEA 2026 call](https://sig-edu.org/news/bea21-call-for-papers/),
[AIES 2026 call](https://www.aies-conference.com/2026/call-for-papers/).
