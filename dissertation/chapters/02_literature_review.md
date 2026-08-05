# Chapter 2: Literature Review

## 2.1 Scope of the review

This review covers the parts of the literature the project draws on and, in places, argues against:
detection of AI-generated text, the stylometric features behind the detector, explainability and
faithfulness testing, argument mining, question generation, evaluation of question quality,
fairness, and the assessment literature on what universities can do beyond detection. The sections
follow the order of the pipeline components and end with the gap in the
literature. I anchored the search on the methods and datasets already chosen (the DeBERTa detector,
the M4 benchmark, the Persuasive Essays corpus, Bloom's taxonomy, the answerability idea behind the
evaluation) and then branched out to alternatives and criticisms.

## 2.2 Detecting AI-generated text

Work on detecting machine-generated text falls into two broad families. A recent survey by Wu et
al. (2025) organises the field along these lines and adds watermarking and human-assisted methods.
Watermarking is worth a note because it attacks the problem at the source: Kirchenbauer et al.
(2023) embed a statistical signal in the model's own sampling that a verifier can later test for.
It only works if the generator cooperates, which no institution can assume of the tools students
actually use, so post-hoc detection remains the practical setting for this project.

The first family is zero-shot and statistical. These methods score text using a language model's own
probabilities and need no trained classifier. The clearest example is DetectGPT (Mitchell et al.,
2023), which observes that machine-generated passages tend to occupy negative-curvature regions of a
model's log-probability surface, and turns that observation into a curvature test requiring no
labelled data. Nothing has to be trained. But the signal is subtle and can be disturbed.

The second family is supervised. A pretrained transformer is fine-tuned on labelled human and AI
examples and then classifies new text. I chose DeBERTa (He et al., 2021) as the primary detector
because its disentangled attention mechanism improves on earlier encoders, with RoBERTa (Liu et al.,
2019) as a comparison. Supervised detectors are accurate in-domain but depend heavily on what they
were trained on.

Progress is measured on shared benchmarks. The M4 corpus (Wang et al., 2023) collects human and
machine text across several generators, domains, and languages, and the SemEval-2024 Task 8 shared
task built on it (Wang et al., 2024). I use that benchmark to test whether a detector trained on one
generator and one kind of text transfers to unseen generators and unseen domains.

Both approaches have weaknesses that matter for this project. Detectors are brittle: Krishna et al.
(2023) show that a strong paraphraser drops DetectGPT's accuracy from 70.3% to 4.6% without changing
the meaning, and accuracy also falls under domain shift. Koike et al. (2024) push the same point
further with OUTFOX, where an attacking model learns from a detector's own outputs to write harder
essays; their essay corpus, which pairs human student writing with multiple generators, is the one this
project uses for its cross-generator transfer test in Chapter 6. In the education setting the
weakness is not hypothetical. Weber-Wulff et al. (2023) tested fourteen detection tools, including
the systems universities license, and concluded they are neither accurate nor reliable on academic
text. The problem that matters more here is
opacity. Commercial detectors return a single score with nothing behind it, and a number without
reasons is a weak basis for an academic-integrity decision. This project starts from that problem.

## 2.3 Stylometric features

Alongside the transformer, the detector uses stylometric features. These describe how a text is
written, independent of its subject: perplexity or predictability under a language model, burstiness
(variation in sentence length and structure), type-token ratio, sentence-length variance, and
part-of-speech distributions. Fluent machine text tends to be smoother and less varied than human
writing, and each feature is meant to capture one form of that difference.

There is direct evidence that these surface properties carry a real signal, and sometimes a harmful
one. Liang et al. (2023) attribute the misclassification of non-native English writing to its lower
perplexity, that is, its more limited linguistic variability, which makes it look more machine-like
to a perplexity-based detector. So the same features that separate the classes can also encode bias.
The pipeline uses them as interpretable evidence and checks their behaviour separately. Hand-crafted
features (perplexity, readability, error-based and other lexical features) have also been shown to
classify AI-generated and even AI-rephrased text, alone and alongside neural models (Mindner et al.,
2023).

Two further results justify the specific choices made here. Opara (2024) reports that a purely
stylometric classifier over 31 features is competitive on its own, which matters because it means
the interpretable half of the hybrid is not decoration added for the sake of an explanation. More
importantly for the architecture, Kumarage et al. (2023) fuse a normalised stylometric feature
vector with the embedding of a fine-tuned language model and show the combination beating the
transformer alone, using feature families close to the ones here: phraseology, punctuation, and
linguistic diversity measured through a moving-average type-token ratio. That paper works on tweets
rather than essays, so the setting is different, but it is the closest published precedent for the
design in Chapter 6 and it supports the same reasoning: the two signals are partly independent, so
fusing them buys something.

Perplexity deserves its own note, since it is the feature the fairness problem attaches to. The
strongest current use of it is Binoculars (Hans et al., 2024), which contrasts the perplexity of one
model against the cross-perplexity of a closely related second model and detects over 90 percent of
ChatGPT output at a false-positive rate of 0.01 percent, with no training data at all. That figure
sets a useful bar. An academic-integrity tool operating in a module of three hundred students cannot
tolerate a percentage-level false-positive rate, and the gap between 0.01 percent in a benchmark and
the rates measured on unfamiliar human prose in Chapter 6 is a large part of why this project treats
a flag as the opening of a conversation rather than a finding.

## 2.4 Explainability and faithfulness

The explainability layer is central to the project, so this section covers the attribution methods
and the work on testing whether an explanation reflects what the model actually did.

The pipeline applies Integrated Gradients (Sundararajan et al., 2017) to the transformer's tokens.
The method attributes a prediction to its input features and satisfies two axioms, Sensitivity and
Implementation Invariance. SHAP (Lundberg and Lee, 2017) assigns each feature a Shapley-value
importance; here it runs on the stylometric features, where it gives a lecturer-facing,
feature-level explanation. Attention weights are sometimes offered as explanations too, but that use
has been criticised (Jain and Wallace, 2019). Partly for that reason, this project does not rely on
attention.

An explanation is only useful if it is faithful, meaning the things it highlights actually drove the
decision. DeYoung et al. (2020) introduced the ERASER benchmark and the comprehensiveness and
sufficiency metrics, which measure faithfulness by removing or keeping the features an explanation
calls important and watching whether the prediction moves. Faithfulness has since become a field of
its own: Lyu et al. (2024) survey over a hundred explanation methods through that single lens,
working from the definition that an explanation must reflect the model's actual reasoning.
That requirement is load-bearing in this dissertation, where a convincing token
highlight failed the ablation test and a plainer feature account passed it. This project uses the
same kind of
ablation test. The test showed the transformer's token-level attributions were diffuse and only
weakly faithful, while the stylometric SHAP explanation held up.

## 2.5 Argument mining

The pipeline needs the claims and evidence in a student's essay before it can generate questions
grounded in them. The foundational resource is the Persuasive Essays corpus of Stab and Gurevych
(2017), which annotates argument components (claims, premises) and the relations between them, and
frames the task as token-level sequence labelling plus relation classification. The argument-mining
component here follows that setup. Their own system pairs a CRF over a large hand-engineered
feature set with Integer Linear Programming that optimises component types and relations jointly,
and reaches a macro F1 of 0.867 for component identification against a human upper bound of 0.886,
which is the benchmark Chapter 7 measures itself against. More recent work applies transformer
ensembles, and LLM refinement, to argument component classification, though on debate corpora such
as Args.me rather than on persuasive essays (Pietron et al., 2024).

The field has moved quickly since, and Li et al. (2025) survey where large language models now sit
within it. Two results are directly relevant to the constraint this project works under. Abkenar et
al. (2024) assess open-source models on argument-mining tasks, and Favero et al. (2025) test
specifically whether small models can do argument component classification in an educational
setting. Both matter because the component here has to run on the same 8 GB machine as everything
else, so the question is not what the largest available model can do but what a small local one can
be trusted with.

For this project the extraction also has to carry provenance. Each extracted claim points back to
the exact source passage it came from, so a generated question can be tied to where it originated
and a lecturer can defend it. Extraction without that link would bring the black-box problem back in
at a different stage. This is the requirement that rules out simply asking a model to summarise the
argument, however fluent the summary, because a paraphrase with no anchor cannot be checked.

## 2.6 Automatic question generation

Automatic question generation for education has a long history before LLMs; the systematic review
of Kurdi et al. (2020) covers 93 pre-LLM studies and already identifies the field's persistent
gap, that little of the work controls the difficulty or form of what gets generated. The field has since
moved from neural models that generate a question from a passage to large
language models prompted to do the same, with growing interest in controllable and grounded
generation (Guo et al., 2024). Grounding matters most for verification. A question is only
defensible if it can be traced to a specific claim in the student's own text, so the generator works
per extracted claim and records the source sentences.

The comparison at the centre of this project, a commercial model against a locally run open model,
sits inside a wider question about cost and control in LLM deployment. In the closest study to this
setting, Oketch et al. (2025) compared closed and open LLMs for automated essay scoring and found
open models such as Llama 3 and Qwen2.5 comparable to GPT-4 in performance at up to 37
times lower cost. An institution that could run verification on a laptop instead of a paid API would
have a cheaper and more controllable option, if the quality holds. The project tests whether it
does.

## 2.7 Bloom's taxonomy and cognitive level

Each generated question is labelled by cognitive level as a quality control. The labels come from
the revised Bloom's taxonomy of Anderson and Krathwohl (2001), which orders cognitive processes from
Remember and Understand through Apply and Analyse to Evaluate and Create. The taxonomy gives a
principled way to describe what a question demands of a student.

There is a directly relevant dataset for the educational side. EduQG (Hadifar et al., 2022) provides
3,397 questions with their source documents, and 903 of them are annotated with a Bloom's cognitive
level. I use that labelled subset to train and check the Bloom's classifier. Automatic
classification of question cognitive level is an active research task; approaches run from classical
classifiers through transformers to large language models (Kumar et al., 2025).

## 2.8 Evaluating generated questions

This section carries more weight than its length suggests, because the evaluation in Chapter 8 rests
on it. The question is how to tell a good verification question from a bad one without asking a
human, and the literature offers three answers: compare against a reference, ask a model to judge,
or test whether the question actually behaves the way a question should.

The reference-based answer is the oldest and the weakest. Metrics such as BLEU and BERTScore score a
generated question against a human-written one. Nguyen et al. (2024) show the approach undermines
itself: most question-generation benchmarks carry a single reference, so they replicated the
annotation process, collected a second valid human reference, and found that reference-based metrics
graded that second human question no better than machine output. A metric that cannot recognise a
human-written question as good is not measuring question quality. They propose a reference-free
alternative built on naturalness, answerability and complexity instead.

The second answer is answerability, and it is the family this project belongs to. RQUGE
(Mohammadshahi et al., 2023) drops the reference entirely and scores a question by whether a
question-answering module can answer it from the given context. QGEval (Fu et al., 2024) generalises
the idea into a benchmark across seven dimensions, fluency, clarity, conciseness, relevance,
consistency, answerability and answer consistency, and uses it to evaluate the automatic metrics
themselves rather than only the generators.

The most directly relevant work predates the current interest in LLM question generation. Liusie et
al. (2022) observed that multiple-choice reading comprehension systems answer questions
significantly better than chance with no access to the passage at all, by falling back on world
knowledge, and they turned that observation into a tool for test designers: information-theoretic
measures of how much world knowledge a question set leaks, including a contextual mutual information
term for how much the passage actually matters to a given question. That is the same instinct behind
the discrimination simulation in Chapter 8, which compares a model answering with the source against
the same model answering without it and treats the gap as the question's value. The framing differs,
since Liusie et al. are auditing an existing test while this project is selecting among generated
questions, but the underlying claim is shared. A question that can be answered without the source is
not testing what it appears to test. Finding that the idea has a published precedent strengthens the
method rather than weakening the contribution, because it means the measure is not an invention of
convenience.

The third answer is to ask a large model to judge, which is now the default in the field and is the
supplementary evaluation here. Its problems are well documented. Zheng et al. (2023) report that
strong judges match human preference around 80 percent of the time but carry position, verbosity and
self-enhancement biases. Feuer et al. (2024) go further and test whether judge preferences track
anything concrete, finding they do not correlate with measured safety, world knowledge or
instruction following, and that judges systematically prioritise style over substance. Norman et al.
(2026) audit 21 judges across three benchmarks and roughly 541,000 judgments, and separate two
things that are easy to confuse: a judge can be highly reliable, reproducing its own verdicts above
0.95, while still being invalid, showing severe position bias and shifting rankings by up to
fourteen places between benchmarks. Their phrase for it, reliability without validity, describes
what Chapter 8 finds at smaller scale, where three judges agree with each other about the ranking
and none of them agrees with the objective measure.

Cross-model agreement here is quantified with Krippendorff's alpha (Hayes and Krippendorff, 2007)
and, following the warning in all three of those papers, anchored to the discrimination results
rather than trusted on its own.

## 2.9 Fairness and bias in detection

Fairness was one of the motivations for this project. Liang et al. (2023) tested seven widely used
GPT detectors on 91 human-written TOEFL essays and 88 US eighth-grade essays. The detectors handled
the US essays almost perfectly but misclassified more than half the TOEFL essays as AI-generated, an
average false-positive rate of 61.22 percent, with 18 of the 91 flagged unanimously by all seven and
97.8 percent flagged by at least one. Two details of that study matter for the design here. The
unanimously flagged essays had significantly lower perplexity, and prompting a model to enrich the
word choice of those same essays cut the average false-positive rate from 61.22 percent to 11.77
percent. So the detectors were not recognising machine authorship at all. They were responding to
constrained linguistic expression, which is a property of the writer rather than of the writing's
origin. Chapter 6 measures the same mechanism on my own detector at domain level and reports it as a
main result rather than a footnote.

The problem is not confined to one study. Weber-Wulff et al. (2023) tested fourteen detection tools,
including several universities license, and judged them neither accurate nor reliable. Perkins et
al. (2024) show the tools degrade further under simple adversarial edits, which is the same
brittleness Krishna et al. (2023) demonstrate with paraphrasing. Dik et al. (2025) evaluate a
single widely used commercial detector on human and AI essays and report the accuracy achieved in
practice rather than in marketing. None of this says detection is worthless. It says a detector
output is evidence of a weak and unevenly distributed kind, which is precisely the argument for
putting a human conversation between the flag and any consequence.

What that means for students is documented too, and it is the part of the literature most often
skipped. Gorichanaz (2023) analysed forty-nine Reddit threads in which students described being
accused of using ChatGPT, most of them, on the author's reading, falsely. The thematic analysis
found students adopting a legalistic posture, gathering evidence such as version histories to prove
authorship, and struggling with the fact that the burden had effectively shifted onto them. The
paper's own conclusion is that assessment needs rethinking rather than better policing. That finding
shaped this project's output more than any technical result: the guide is written so that a student
can answer, and so that the lecturer has something concrete to ask about rather than a number to
defend.

## 2.10 What universities can do beyond detection

The education literature has been arguing since ChatGPT's release that detection alone cannot carry
an integrity policy. Perkins (2023) reviews the academic-integrity implications of LLMs and
concludes that what defines misconduct is undisclosed use, and that updated integrity policies,
not tool bans, are where institutions have real leverage.
Cotton et al. (2024) reach a similar position, recommending policies and procedures, training and
support, and detection used as one method among several rather than as the verdict.
The strand of that literature closest to this project is authentic and oral assessment. Sotiriadou
et al. (2020) show that scaffolded assessment ending in an interactive oral, a short structured
conversation about the student's own work, both deters misconduct and gives genuine evidence of
understanding. That is precisely the conversation this project's Verification Interview Guide is
built to equip, with the difference that here the questions are generated automatically from the
submission itself, at a scale where a lecturer with three hundred scripts could not prepare them
by hand.

Others have reached the same conclusion and gone at the scale problem from the other side.
Ipeirotis and Rizakos (2026) build oral assessment delivered by voice AI, automating the conversation
itself in order to make it affordable at class scale. That is the natural extension of the same
premise and it marks the boundary of this project deliberately: the pipeline here prepares a human
conversation and stops, because the moment the interview is automated the fairness argument that
motivated the whole design starts to erode. Church et al. (2025) look at LLM support for assessing
student work more generally, and Lee et al. (2025) argue for AI-generated variation as a way to
protect assessment validity rather than to police it. On the institutional side, Beale (2025)
survey how universities are adapting policy, updating the picture Jin et al. (2024) give of adoption
across forty institutions. Delphino (2025) supply a number the policy discussion usually
lacks by estimating the actual prevalence of AI-assisted cheating in programming courses, which is a
reminder that the size of the problem is itself contested.

## 2.11 Summary and the gap

Detection is improving but stays opaque, and on the evidence above it is sometimes unfair.
Explainability methods exist, and there are established ways to test whether they are faithful.
Argument mining can recover a student's claims with provenance. Question generation and its
evaluation are maturing, and Bloom's taxonomy gives a way to grade cognitive level. Nothing in this
literature joins the pieces together. No existing work takes a detection flag and produces
transparent, source-grounded verification questions that let a lecturer fairly check whether a
student understands their own work. This project addresses that gap. It also takes on the practical
question of whether a local model can do the job as well as a commercial one.
