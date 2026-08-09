# Interactive app

A local web app that runs the whole pipeline on any text you paste: detection, the plain-language
explanation, the sentences the detector reacted to, and verification questions drawn from the
submission's own claims.

Nothing is uploaded. Every model runs in this process, which is the same argument the dissertation
makes for the pipeline as a whole: a student submission is personal data and should not be sent to
a third party to be judged.

## Running it

```
python src/webapp/server.py
```

Then open <http://127.0.0.1:8000>.

The models load once at startup, which takes roughly 40 seconds. After that a detection takes about
1.5 seconds. The batch scripts reload every model on each call, so the app does not use them
directly: `pipeline_service.py` holds one copy of each model in process and reuses it.

Step 4, the questions, needs a local language model. Start Ollama first:

```
ollama serve
```

`llama3.1:8b` should be pulled. If Ollama is not running the first three steps still work and the
page says plainly what is missing rather than failing silently.

## What it does, stage by stage

1. **Detection.** The hybrid of Section 6.7: a fine-tuned DeBERTa fused with a gradient-boosting
   model over stylometric features and GPT-2 perplexity. Both component scores are shown, because
   the fusion only reads high when the two readers agree.
2. **Explanation.** The habit card of Section 5.7, in plain sentences and as positions against the
   middle 80 percent of real student essays. This is the account that passed the faithfulness test;
   the word-level highlighting people expect did not, which is why it is not offered.
3. **Corpus position.** The same measurements as percentiles against the 640 human essays. The habit
   card says whether a measurement is outside the normal range; this says by how far.
4. **Sentence marks.** Sentence-level occlusion (Section 5.8): each sentence is deleted in turn and
   the change in the score is measured, in log-odds because the detector saturates in probability.
   Expect the marks to be spread out. There is no single guilty sentence, and the app does not
   pretend otherwise. Click a marked sentence for what is known about it, what that supports, and
   what to do next. Underneath, the strongest sentences are removed and the rest re-scored, so you
   can see whether the verdict survives losing its best evidence.
5. **Claims and questions.** The claim extractor keeps the sentence numbers each claim came from,
   and the quoted text is looked up from the submission rather than echoed by the model, so an
   invented quotation is impossible. Questions are written per claim and tagged with a Bloom level.
   How many claims, how many questions each, and which backend writes them are all controls on the
   page.

**Compare two submissions** runs two texts through the same models side by side, with their habits
and percentiles set against each other. It is the check that makes a flag defensible: if a
submission you know to be genuine sits at the same extremes as the flagged one, the flag is
describing the genre rather than the author.

## Correctness

The app is wired to the same trained artefacts the dissertation reports on, so the two cannot
disagree. On the worked example it reproduces the reported figures exactly: the AI-written version
scores 0.9572 and is flagged, the real student's essay scores 0.0234 and is not.

A submission under 120 words is refused. Below that the writing-habit measurements are unstable,
and reporting them anyway would be the kind of unearned confidence this project argues against.

## Endpoints

`GET /api/status`, `POST /api/warmup`, `POST /api/detect`, `POST /api/explain`,
`POST /api/percentiles`, `POST /api/marks`, `POST /api/counterfactual`, `POST /api/questions`.
Interactive docs at `/api/docs`.

## Figures

`dissertation/presentation/make_webapp_figures.py` drives a headless Chrome through the whole page
and writes the screenshots used in Section 4.11. Start this app first, then run it. Scripting the
capture means the figures cannot quietly go stale when the layout changes.

## The bundled examples

`static/example_ai.txt` is the machine-written essay from the worked pair and ships with the repo.

The matching human essay is **not** included: it comes from BAWE, which is licensed for research
and must not be redistributed. The "Load a real student example" button therefore only works if you
create `static/example_human.txt` yourself from the corpus. Without it the button explains that the
file is missing, and you can paste any essay instead.
