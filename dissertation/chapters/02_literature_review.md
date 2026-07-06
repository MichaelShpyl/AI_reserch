# Chapter 2: Literature Review

## 2.1 Scope of the review

This review covers the parts of the literature the project draws on and argues against: detecting
AI-generated text, the stylometric features behind the detector, explainability and how to test whether
an explanation is honest, argument mining, question generation, how question quality is judged, and
fairness. The sections follow the pipeline so the review builds toward the gap the project fills. I
anchored the search on the specific methods and datasets already chosen (the DeBERTa detector, the M4
benchmark, the Persuasive Essays corpus, Bloom's taxonomy, the answerability idea behind the evaluation)
and then branched out to alternatives and criticisms.

## 2.2 Detecting AI-generated text

Work on detecting machine-generated text splits into two broad families, and a recent survey by Wu et
al. (2025) organises the field along these lines, adding watermarking and human-assisted methods.

The first family is zero-shot and statistical. These methods use a language model's own probabilities
rather than a trained classifier. The clearest example is DetectGPT (Mitchell et al., 2023), which
observes that machine-generated passages tend to occupy negative-curvature regions of a model's
log-probability surface, and turns that observation into a curvature test that needs no labelled data.
The appeal is that nothing has to be trained; the weakness is that the signal is subtle and can be
disturbed.

The second family is supervised. A pretrained transformer is fine-tuned on labelled human and AI
examples and then classifies new text. The transformer this project uses, DeBERTa (He et al., 2021),
improves on earlier encoders with a disentangled attention mechanism and was chosen as the primary
detector for that reason, with RoBERTa (Liu et al., 2019) as a comparison. Supervised detectors are
accurate in-domain but depend heavily on what they were trained on.

Progress is measured on shared benchmarks. The M4 corpus (Wang et al., 2023) collects human and
machine text across several generators, domains, and languages, and the SemEval-2024 Task 8 shared task
built on it (Wang et al., 2024). I use that benchmark to test whether a detector trained on one
generator and one kind of text transfers to unseen generators and unseen domains.

Two weaknesses matter for this project. First, detectors are brittle: Krishna et al. (2023) show that a
strong paraphraser drops DetectGPT's accuracy from 70.3% to 4.6% without changing the meaning, and
accuracy also falls under domain shift. Second, and more important here, commercial detectors return a
single opaque score. A number with no reasons behind it is a weak basis for an academic-integrity
decision, and that black-box problem is the starting point for this project.

## 2.3 Stylometric features

Alongside the transformer, the detector uses stylometric features, which describe how a text is written
rather than what it is about: perplexity or predictability under a language model, burstiness (variation
in sentence length and structure), type-token ratio, sentence-length variance, and part-of-speech
distributions. Each is meant to capture a way in which fluent machine text tends to differ from human
writing, being smoother and less varied.

There is direct evidence that these surface properties carry a real, and sometimes harmful, signal.
Liang et al. (2023) attribute the misclassification of non-native English writing to its lower
perplexity, that is, its more limited linguistic variability, which makes it look more machine-like to a
perplexity-based detector. That finding is a double lesson: stylometric features do separate the
classes, but they can also encode bias, so the pipeline treats them as interpretable evidence whose
behaviour must itself be checked. Recent work confirms that hand-crafted features (perplexity,
readability, error-based and lexical features) can classify AI-generated and even AI-rephrased text,
alone and alongside neural models (Mindner et al., 2023), which supports the hybrid design used here.

## 2.4 Explainability and faithfulness

The explainability layer is the point of the project, so this section covers both the attribution
methods and, just as important, how to check whether an explanation is honest.

Two attribution methods are used. Integrated Gradients (Sundararajan et al., 2017) attributes a
prediction to its input features while satisfying two axioms, Sensitivity and Implementation Invariance,
and is applied here to the transformer's tokens. SHAP (Lundberg and Lee, 2017) assigns each feature a
Shapley-value importance and is applied to the stylometric features, where it gives a lecturer-facing,
feature-level explanation. Attention weights are sometimes offered as explanations too, but that use has
been criticised (Jain and Wallace, 2019), which is one reason this project does not rely on attention.

An explanation is only useful if it is faithful, meaning the things it highlights are actually what drove
the decision. DeYoung et al. (2020) introduced the ERASER benchmark and the comprehensiveness and
sufficiency metrics, which measure faithfulness by removing or keeping the features an explanation calls
important and seeing whether the prediction moves. This project uses exactly that kind of ablation test,
and it is what showed the transformer's token-level attributions to be diffuse and only weakly faithful,
while the stylometric SHAP explanation held up.

## 2.5 Argument mining

To turn a flag into questions grounded in the student's own argument, the pipeline has to extract the
claims and evidence in an essay. The foundational resource is the Persuasive Essays corpus of Stab and
Gurevych (2017), which annotates argument components (claims, premises) and the relations between them,
and frames the task as token-level sequence labelling plus relation classification. The
argument-mining component here follows that setup; more recent work applies transformer ensembles,
and even LLM refinement, to argument component classification across several corpora (Pietron et al.,
2024).

Provenance is the reason argument mining matters here rather than mere extraction. Each extracted claim
must point back to the exact source passage it came from, so that a generated question can be tied to
where it originated and a lecturer can defend it. Extraction without provenance would reintroduce the
black-box problem the project is trying to remove.

## 2.6 Automatic question generation

Question generation has moved from neural models that generate a question from a passage to large
language models prompted to do the same, with growing interest in controllable and grounded generation
(Guo et al., 2024). For verification, grounding is the crucial property: a question is only defensible if
it can be traced to a specific claim in the student's own text, so the generator works per extracted
claim and records the source sentences.

The comparison at the centre of this project, a commercial model against a locally run open model, sits
in the wider question of cost and control in LLM deployment. In the closest study to this setting,
Oketch et al. (2025) compared closed and open LLMs for automated essay scoring and found open models
such as Llama 3 and Qwen2.5 comparable to GPT-4 in performance and fairness at up to 37 times lower
cost. An institution that could run verification on a laptop rather than a paid API would have a cheaper
and more controllable option, if the quality holds, which is the empirical question the project tests.

## 2.7 Bloom's taxonomy and cognitive level

Questions are labelled by cognitive level as a quality control, using the revised Bloom's taxonomy of
Anderson and Krathwohl (2001), which orders cognitive processes from Remember and Understand through
Apply and Analyse to Evaluate and Create. The taxonomy gives a principled way to describe what a
question demands of a student.

For the educational side there is a directly relevant dataset. EduQG (Hadifar et al., 2022) provides
3,397 questions with their source documents, of which 903 are annotated with a Bloom's cognitive level,
which makes it suitable for training and checking the Bloom's classifier used here. Automatic
classification of question cognitive level is an active research task, where approaches from classical
classifiers through transformers to large language models have been applied (Kumar et al., 2025).

## 2.8 Evaluating generated questions

The project's main evaluation asks whether a question genuinely tests understanding, and the closest
prior idea is answerability-based, reference-free evaluation. RQUGE (Mohammadshahi et al., 2023) scores a
generated question not by comparing it to a reference question but by whether a question-answering module
can answer it from the given context. The discrimination simulation used in this project is in the same
spirit: it asks whether a model can answer a question with the source versus without it, and treats the
gap as the question's discriminative value.

The supplementary evaluation uses an LLM-as-judge, which is increasingly common but has known
reliability and bias problems, including position, verbosity, and self-enhancement biases, so its
output has to be checked against something independent before it counts (Zheng et al., 2023).
Cross-model agreement is quantified with
Krippendorff's alpha (Hayes and Krippendorff, 2007), and anchored to the objective discrimination
results.

## 2.9 Fairness and bias in detection

Fairness is not a side issue for this project; it is one of its motivations. Liang et al. (2023) tested
seven widely used GPT detectors on TOEFL essays by non-native English writers and found an average
false-positive rate of 61.3%, against about 5.1% for essays by native writers. In other words, the
detectors systematically misclassified competent non-native writing as AI-generated. Chapter 6
measures the same failure mode directly on my own detector and treats it as evidence, not a footnote,
which is why transparency and defensibility run through the design.

## 2.10 Summary and the gap

Detection is improving but stays opaque and, on the evidence above, sometimes unfair. Explainability
methods exist and can be tested for faithfulness. Argument mining can recover a student's claims with
provenance. Question generation and its evaluation are maturing, and Bloom's taxonomy gives a way to
grade cognitive level. What is missing is anything that joins these up: nothing takes a detection flag
and turns it into transparent, source-grounded verification questions that let a lecturer fairly check
whether a student understands their own work. That gap, and the practical question of whether a local
model can do the job as well as a commercial one, is what this project addresses.
