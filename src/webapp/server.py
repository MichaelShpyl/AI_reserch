"""Local web app for the verification pipeline: paste a submission, get the whole analysis.

Everything runs on this machine against the trained models. No text leaves the process, which is
the same argument the dissertation makes for the pipeline as a whole: a student submission is
personal data and should not be sent to a third party to be judged.

    python src/webapp/server.py
    then open http://127.0.0.1:8000

The stages are separate endpoints on purpose. Detection and the explanation are fast local models,
so the page shows them within a couple of seconds, while claim extraction and question writing need
a language model and take longer. The UI requests them in that order rather than making the user
wait for everything at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fastapi import FastAPI, HTTPException                      # noqa: E402
from fastapi.responses import FileResponse, JSONResponse        # noqa: E402
from fastapi.staticfiles import StaticFiles                     # noqa: E402
from pydantic import BaseModel, Field                           # noqa: E402

import pipeline_service as ps                                   # noqa: E402

app = FastAPI(title="Verification Interview Guide", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

MIN_WORDS = 120          # below this the stylometric features are too noisy to mean anything
MAX_CHARS = 40_000


class Submission(BaseModel):
    text: str = Field(min_length=1)


class QuestionRequest(Submission):
    n_claims: int = Field(default=4, ge=1, le=8)
    k_questions: int = Field(default=3, ge=1, le=5)
    backend: str = Field(default="local")


def _check(text: str) -> str:
    t = text.strip()
    if len(t) > MAX_CHARS:
        raise HTTPException(413, f"Submission is longer than {MAX_CHARS} characters.")
    if len(t.split()) < MIN_WORDS:
        raise HTTPException(
            422,
            f"Needs at least {MIN_WORDS} words. The writing-habit measurements are unstable on "
            f"short text, and reporting them anyway would be the kind of unearned confidence this "
            f"project argues against.")
    return t


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    return ps.status()


@app.post("/api/warmup")
def warmup():
    return ps.warm_up()


@app.post("/api/detect")
def api_detect(sub: Submission):
    return ps.detect(_check(sub.text))


@app.post("/api/explain")
def api_explain(sub: Submission):
    return ps.explain(_check(sub.text))


@app.post("/api/marks")
def api_marks(sub: Submission):
    return ps.sentence_marks(_check(sub.text))


@app.post("/api/counterfactual")
def api_counterfactual(sub: Submission):
    """Delete the most-reacted-to sentences and re-score, against a random-removal control."""
    return ps.counterfactual(_check(sub.text))


@app.post("/api/percentiles")
def api_percentiles(sub: Submission):
    """Where this text sits in the distribution of the 640 real student essays, feature by feature."""
    return ps.percentiles(_check(sub.text))


@app.post("/api/questions")
def api_questions(req: QuestionRequest):
    try:
        return ps.claims_and_questions(_check(req.text), req.n_claims,
                                       req.k_questions, req.backend)
    except HTTPException:
        raise
    except Exception as e:
        # The question stage needs a language model running locally. Say so plainly rather than
        # returning a 500 the user cannot act on.
        return JSONResponse(
            status_code=503,
            content={"detail": "The question stage needs a local language model. Start Ollama "
                               "(`ollama serve`) and make sure llama3.1:8b is pulled, then try "
                               f"again. Underlying error: {type(e).__name__}: {e}"})


if __name__ == "__main__":
    import uvicorn
    print("Loading models once, this takes about half a minute...")
    print(ps.warm_up())
    print("Ready on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
