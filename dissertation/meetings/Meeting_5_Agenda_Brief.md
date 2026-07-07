% Supervision Meeting 5: Agenda and Progress Brief
% Mykhailo Shpyl (MSc AI and Big Data Analytics, ATU). Supervisor: Dr. Vini Vijayan
% Monday 7 July 2026

## Purpose of this meeting

Progress since Meeting 4 (23 June), what your approvals enabled this week, and, most usefully, a
run-through of the graded intermediate presentation (14 July). The detail is in the deck
(`Meeting5_visual.pptx`); this page is the short version.

## Agenda

1. Progress since Meeting 4 (8 to 10 minutes, from the deck).
2. This week's plan (below).
3. Run-through of the 14 July presentation, if time allows (about 10 minutes plus your feedback).

## What happened since Meeting 4

1. **The commercial backend is live and the core comparison ran at scale.** ATU could not provide
   API access (Aoife Hill, 25 June), so Backend A runs on a free-tier Gemini API behind the same
   interface as the local model. Over fourteen essays with 126 questions per backend, scored
   identically: local 0.042 [0.030, 0.055] vs commercial 0.078 [0.058, 0.098]; the paired difference
   is 0.036 (t p = 0.040, Wilcoxon p = 0.030). Reading: a small but statistically significant
   commercial edge, and a local model that is competitive before any fine-tuning.

2. **Three more components trained.** The Bloom's classifier (BERT-base on EduQG; macro-F1 0.31
   against the keyword baseline's 0.16; the skewed label supply is the ceiling). The claim extractor
   (DeBERTa on Persuasive Essays 2.0, official split; strict span-F1 0.63, premises 0.72). And the
   output assembler, which now produces the end product: a lecturer's Verification Interview Guide
   PDF, generated with live models, opening with the position that it is evidence for a conversation,
   not an accusation.

3. **Your sign-off on Qwen2.5 3B was put to work the same week, and the result is the most
   instructive of the project so far.** The QLoRA fine-tune ran and its discrimination score more
   than quadrupled, which for about an hour looked like the headline of the dissertation. Before
   writing that up I read the actual questions: 95 percent were degenerate multiple-choice stems
   ("Which of the following is correct?", no options), because EduQG is mostly a multiple-choice
   corpus and the model overfit its format. Worse, those empty stems game the discrimination metric;
   the bare stem alone scores higher than any real question. I retracted the result, wrote the audit
   up as its own section, and made a testable prediction: change only the data and the failure should
   vanish. Re-fine-tuned on SQuAD (open-ended questions, everything else identical): degeneracy went
   from 95 percent to zero, and the gain over the base model is now real (0.102 vs 0.027, p = 0.0003)
   though still modest. Conclusion: the training data's format is the lever, and the metric now has a
   well-formedness gate in front of it.

4. **A tighter comparison design.** In the scaled run each backend chose its own claims, so a backend
   could look better by picking easier claims. I re-ran the comparison with one fixed claim set per
   essay and every question writer answering the same claims, now complete at 14 of 14 essays per
   arm. On that design the commercial edge disappears entirely (Gemini 0.024, Llama 8B 0.031, base
   Qwen 3B 0.041; paired difference -0.005, p = 0.62, Gemini higher on 7 of 14, a coin flip). The
   earlier edge came substantially from claim selection: fix the task and the laptop models write
   questions as discriminative as the commercial one.

5. **LLM-as-judge, now complete for all three judges** (Gemini free tier; Claude and GPT on the
   capped spend you approved). The result is decisive: Gemini and GPT sit at the rating ceiling
   (means 4.81 and 4.94) while Claude uses the scale (mean 3.67), cross-model agreement is poor
   (Krippendorff's alpha -0.25, no pairwise correlation significant), and no judge correlates
   positively with the objective simulation (GPT's near-ceiling ratings even reach significantly
   negative, rho -0.75). Judge ratings measure how good a question looks, not whether it works;
   the anchoring the evaluation plan insisted on was needed.

6. **Backend B, v3 done overnight, completing the data-format experiment.** 2,604 verification
   questions distilled from the pipeline's own prompt (teacher: local Llama 8B; evaluation essays
   excluded; only gate-passing questions kept), then the identical fine-tune. Result: zero degenerate
   output, 2.3 questions per claim, on-style verification questions, discrimination 0.064 (about 2.5
   times base, and double the 8B teacher itself). One twist worth two minutes today: v2 still posts a
   higher raw score (0.102) with terse factual questions that are not verification questions at all,
   so the metric alone would pick the wrong adapter. Third instance of the project's central lesson;
   v3 is the working Backend B on style-fit plus real gain, with v2 documented.

7. **Writing kept pace.** Your Meeting 4 feedback stayed closed out (opening paragraphs, elaborated
   background, per-chapter outline, references verified against the real papers). This week added
   the fine-tune story (Sections 8.8 to 8.10), the completed three-judge validation (8.7), and the
   third explanation method, attention, measured on the same faithfulness yardstick as the others
   (5.6). The draft is 62 pages and you have the Word version.

## This week's plan

1. **The balanced-vs-natural training comparison** (your Meeting 2 evaluation item) is running
   today: two same-size detectors, identical settings, one trained on the balanced writer mix and one
   on BAWE's natural skewed mix, both evaluated on the same test split with per-cell and
   natural-weighted metrics plus the fairness read.

2. **Hybrid detector fusion.** The scope's component 1 is a transformer combined with stylometric
   features; both halves exist and score 0.99 and 0.985, so this week they get fused (with the
   deferred GPT-2 perplexity feature added) and evaluated against each half.

3. **Claim-extraction integration** per the design you approved: trained spans for provenance plus
   prompted phrasing, then a regenerated showcase guide with every trained component and the new
   well-formedness gate.

4. **One decision when convenient: relation classification.** The scope's argument-mining component
   mentions pairwise relation classification alongside the span tagger. The question generator only
   consumes claims, so I see two honest options: a basic relation classifier on Persuasive Essays
   (about two days), or descoping it with your sign-off and a sentence in the write-up. Your call.

5. **Then the writing:** the Discussion and Conclusions chapters, a consistency pass over the earlier
   chapters (several passages still promise work that is now done), and completing the reference list
   with the method papers (QLoRA, LoRA, SQuAD, SHAP, Integrated Gradients, Krippendorff), each
   verified against the real publication as always.

6. **AICS.** The call or email when convenient; the Student-Track draft is ready to align to it.

## For 14 July

The intermediate presentation is Monday 14 July, online, graded. The deck is updated with everything
above, including the fine-tune story told honestly, which I think is its strongest slide: it shows
the evaluation discipline working exactly as designed. A ten-minute run-through with your feedback
today would be worth more than anything else I could prepare this week.
