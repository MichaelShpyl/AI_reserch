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
   essay and every question writer answering the same claims. On that design the commercial edge
   disappears (Gemini 0.017, Llama 8B 0.031, base Qwen 3B 0.041; paired p = 0.20), which suggests
   part of the earlier edge was claim selection. Held as a hypothesis for now: the commercial arm
   covered 9 of 14 essays before free-tier quota ran out, and the run completes when quota resets.

5. **LLM-as-judge, complete for judge one.** The free-tier judge rated all 12 pilot questions 4.5 to
   5.0 (a ceiling effect) with no correlation to the objective simulation (Spearman -0.14, p = 0.67).
   Judge scores alone would certify questions the objective measure contradicts, which reinforces the
   same lesson as the fine-tune episode: no single automatic number gets trusted in this project.

6. **Writing kept pace.** Your Meeting 4 feedback stayed closed out (opening paragraphs, elaborated
   background, per-chapter outline, references verified against the real papers). The fine-tune
   episode added three new results sections (8.8 to 8.10) and two figures. The draft is 60 pages and
   you have the Word version.

## This week's plan

1. **Backend B, next step.** SQuAD proved the data format is the lever, but it trains a factual
   style. Next: a training set of reasoning-demanding verification questions distilled from the
   pipeline's own prompts, then re-fine-tune and re-measure on the same fixed claims.

2. **Judges two and three.** Executing the capped spend you approved (Claude and GPT) for
   cross-model agreement (Krippendorff's alpha), anchored to the simulation.

3. **Commercial arm refill.** The fixed-claim comparison completes its remaining five essays when
   the free-tier quota window reopens; the framework resumes automatically.

4. **AICS.** The call or email when convenient; the Student-Track draft is ready to align to it.

## For 14 July

The intermediate presentation is Monday 14 July, online, graded. The deck is updated with everything
above, including the fine-tune story told honestly, which I think is its strongest slide: it shows
the evaluation discipline working exactly as designed. A ten-minute run-through with your feedback
today would be worth more than anything else I could prepare this week.
