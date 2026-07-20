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
2023). I took that result as support for the hybrid design.

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
component here follows that setup. More recent work applies transformer ensembles, and even LLM
refinement, to argument component classification across several corpora (Pietron et al., 2024).

For this project the extraction also has to carry provenance. Each extracted claim points back to
the exact source passage it came from, so a generated question can be tied to where it originated
and a lecturer can defend it. Extraction without that link would bring the black-box problem back in
at a different stage.

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

The project's main evaluation asks whether a question genuinely tests understanding. The closest
prior idea is answerability-based, reference-free evaluation. RQUGE (Mohammadshahi et al., 2023)
drops the reference question and instead scores a generated question by whether a
question-answering module can answer it from the given context. The discrimination simulation used
in this project works in the same spirit. It asks whether a model can answer a question with the
source versus without it, and treats the gap as the question's discriminative value.

The supplementary evaluation uses an LLM-as-judge. The approach is increasingly common but has known
reliability and bias problems, including position, verbosity, and self-enhancement biases (Zheng et
al., 2023), so judge output has to be checked against something independent before it counts.
Cross-model agreement is quantified with Krippendorff's alpha (Hayes and Krippendorff, 2007) and
anchored to the objective discrimination results.

## 2.9 Fairness and bias in detection

Fairness was one of the motivations for this project. Liang et al. (2023) tested seven widely used
GPT detectors on TOEFL essays by non-native English writers and found an average false-positive rate
of 61.3%, while the same detectors classified essays by native US student writers accurately, at
around 5 percent false positives read from the paper's first figure. The detectors systematically
misclassified competent non-native writing as AI-generated. Chapter 6 measures the same failure mode
directly on my own detector and reports it as a main result rather than a footnote. It is also why
transparency and defensibility run through the design.

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

## 2.11 Summary and the gap

Detection is improving but stays opaque, and on the evidence above it is sometimes unfair.
Explainability methods exist, and there are established ways to test whether they are faithful.
Argument mining can recover a student's claims with provenance. Question generation and its
evaluation are maturing, and Bloom's taxonomy gives a way to grade cognitive level. Nothing in this
literature joins the pieces together. No existing work takes a detection flag and produces
transparent, source-grounded verification questions that let a lecturer fairly check whether a
student understands their own work. This project addresses that gap. It also takes on the practical
question of whether a local model can do the job as well as a commercial one.
