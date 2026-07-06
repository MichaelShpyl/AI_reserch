% Supervision Meeting 4: Agenda and Progress Brief
% Mykhailo Shpyl (MSc AI and Big Data Analytics, ATU). Supervisor: Dr. Vini Vijayan
% Tuesday 23 June 2026

## Purpose of this meeting

Update on one week of work since Meeting 3 (16 June), and five decisions I need from you. The detail and the figures are in the deck (`Meeting4_visual.pptx`); this is the one-page version to read first.

## Agenda

1. Progress since Meeting 3 (5 minutes, talk from the deck).
2. Decisions I need from you (10 minutes, the list below).
3. Plan to submission and the writing weeks (5 minutes).

## What I did since Meeting 3

Last meeting you saw the dataset and a first detector that I had audited. Since then I made the detector explainable, found where it breaks, built the first slice of the question generation, and evaluated it.

1. **Explained the detector, which is the whole point of the project.** A style-only detector (F1 0.985) gives a lecturer an explanation they can defend: longer words push toward AI, while richer vocabulary and more varied sentence length push toward human. I also tested the explanations rather than just producing them. The transformer's token-level highlights are diffuse and only weakly faithful, so I would not put those in front of a lecturer; the style-based explanation is the faithful one. One scope note: because the transformer's own explanations failed that faithfulness test, this style-only model is now the explainable detector I would deploy, with the transformer kept as the higher-ceiling black-box comparison.

2. **Tested how well it transfers (the external M4 benchmark).** Good news first: it catches AI models it never trained on. On the OUTFOX essay split (shipped inside M4), against six generators it never saw, cross-generator F1 is 0.97, with GPT-4, ChatGPT, Cohere and davinci caught about 100% of the time (dolly and bloomz a little lower). The caveat matters more: on the M4 web and academic domains it gets fragile (cross-domain F1 0.79), and the failure mode is the dangerous one, false-accusing humans. Real human arXiv abstracts are flagged as AI 79% of the time. That is the fairness risk the whole project is about, now measured. My intended answer is domain-aware thresholds, a low-confidence band where the system abstains rather than accuses, and the question-generation stage as the backstop (a flag opens a conversation, it does not convict).

3. **Added statistical rigour after a self-review.** Bootstrap 95% confidence intervals (on 200 essays DeBERTa at 0.99 and RoBERTa at 0.995 overlap, so there is no detectable difference between them), seed stability, and a locale control showing British versus American spelling is not what the detector relies on.

4. **Built the first slice of the core contribution.** A Verification Interview Guide: it pulls the main claims, cites each to the exact sentences it came from (so provenance is guaranteed and the model cannot invent a quote), generates questions aimed at the author's own reasoning and evidence, and tags each by Bloom's level. It runs end to end on the local model. Honest caveat: I demonstrated it on one of my own AI-generated essays as a pipeline smoke-test, so the demo is currently AI answering about an AI essay; the real target is a flagged human submission, which I will re-run it on.

5. **Stood up the primary, judge-free evaluation, and got a surprise.** A discrimination simulation that needs no human and no LLM judge. On one flagged essay (18 questions, local model), generic "explain your own argument and evidence" questions discriminate more (0.31) than content-specific ones (0.05), because a knowledgeable model can answer specific factual questions from world knowledge without ever reading the essay. Two caveats I will be upfront about: this is a single-essay early signal that I will confirm by scaling the simulation, and the metric scores similarity to the source, which may flatter broad answers, so I want to rule out a measurement artefact first. Taken carefully, the working hypothesis is that the strongest verification questions make the student reproduce their own reasoning, not recite subject facts.

## Decisions I need from you

1. **Commercial API access (Backend A).** The core commercial-versus-local comparison needs a Claude or GPT API key and a small budget. Can ATU provide one, or should I fund a capped amount myself?

2. **Backend B model size (this changes locked scope).** Full Llama 3 8B fine-tuned with QLoRA may not fit the 8 GB laptop now that there is no HPC. I will not ask you to decide blind: I will run an 8B QLoRA smoke-test on the 4060 this week. If it does not fit, my fallback is Qwen2.5 3B, which fits comfortably and keeps the commercial-versus-local comparison intact. What I want today is your pre-approval of that fallback so the smoke-test result does not leave me blocked.

3. **Re-scope to protect the timeline.** About ten weeks left, with the writing weeks protected. Argument mining, the Bloom's classifier, the output assembler and the full evaluation are still unbuilt. I propose simplifying argument mining to prompted claim extraction rather than training the full BIO tagging plus relation classification. The trade is explicit: it protects a thin, working end-to-end system, but it drops a trained-model contribution, leaves the Persuasive Essays datasets unused, and gives me no benchmarked argument-mining number. With the time left I think it is worth it, but I want your call.

4. **AICS submission.** I have confirmed it is the Irish AICS (AIAI / Springer-CEUR), Student Track, submitted via EasyChair, roughly an October deadline, 6 to 12 pages in CEURART format. Could you forward the call or email, and confirm you are happy with the angle (the detector audit and the fairness finding as the paper's core)?

5. **Literature review.** Chapter 2 is still a skeleton without citations and it is on the critical path. Which sources do you consider essential, so I am not reading blind?

## Plan to submission

- Now to mid-July: question generation (both backends, pending the decisions above) and the Bloom's classifier.
- Mid-July to early August: integration into the Verification Interview Guide and the full evaluation (scale the simulation across many essays, add the supplementary LLM-judge view).
- August: writing, revision, submission. These weeks are protected; if anything slips I cut analysis depth, not write-up time.

## Already drafted, ready to send

- A 40-page working draft in the official 2026 ATU template, with Chapters 1 and 3 to 8 substantive and Chapter 2 still a skeleton (`Dissertation_Shpyl_progress_draft.docx`).
- An AICS Student-Track paper draft (`aics_paper_draft.docx`).
- This brief and the Meeting 4 deck.
