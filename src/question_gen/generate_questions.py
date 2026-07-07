"""First slice of the argument-aware question-generation pipeline (pipeline components 3 to 6).

Given a flagged essay this builds a lecturer's Verification Interview Guide:
  1. Split the essay into numbered sentences.
  2. Extract the student's main claims, each cited to the sentence number(s) it comes from, so
     provenance is guaranteed: we look up the real sentence by index, the model never supplies the
     quote, so it cannot hallucinate a source.
  3. Generate verification questions per claim, grounded in the source, that test whether the
     student understands what they wrote (not answerable from the claim sentence alone).
  4. Tag each question with a Bloom's cognitive level (a transparent heuristic for now; the
     Component 5 BERT classifier replaces it later).
  5. Assemble the guide as JSON and Markdown.

Backends are pluggable behind one interface. This slice runs on the LOCAL model (Ollama
llama3.1:8b), which is the basis for Backend B; the commercial Backend A drops into the same
interface once an API key is available, which is what enables the core commercial-vs-local
comparison. The backend is recorded in the output so runs are comparable.

    python src/question_gen/generate_questions.py --id 3108a --source ai --claims 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
OUTDIR = REPO / "outputs" / "verification_guides"
sys.path.insert(0, str(REPO / "src" / "detection"))
from text_normalize import normalize_text  # noqa: E402

OLLAMA = "http://localhost:11434"


# ---------------- backends ----------------
class Backend:
    name = "base"

    def chat_json(self, system: str, user: str) -> dict:
        raise NotImplementedError


class OllamaBackend(Backend):
    """Local open-source model (the basis for Backend B)."""

    def __init__(self, model: str = "llama3.1:8b", temperature: float = 0.2):
        self.model = model
        self.name = f"local:{model}"
        self.temperature = temperature

    def chat_json(self, system: str, user: str) -> dict:
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": self.model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": self.temperature, "num_ctx": 8192},
        }, timeout=180)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return json.loads(content)


# Backend A (commercial) lives in commercial_backend.py: Gemini (free tier), Anthropic, OpenAI.
# It is built via make_commercial_backend() in main(), behind the same chat_json interface.


# ---------------- pipeline ----------------
def sentences(text: str, cap: int = 100) -> list[str]:
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        sents = [s.text.strip() for s in nlp(text).sents if len(s.text.strip()) > 0]
    except Exception:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sents[:cap]


def extract_claims(sents: list[str], backend: Backend, n: int) -> list[dict]:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
    system = ("You analyse a student essay for an academic-integrity verification interview. "
              "A claim is a substantive position the student asserts and argues for, not background, "
              "definition, or quotation. You cite the sentence numbers each claim is drawn from. "
              "Reply with JSON only.")
    user = (f"Numbered sentences of the essay:\n{numbered}\n\n"
            f"Identify the {n} most important claims the student makes. For each, give a one-line "
            f"paraphrase of the claim and the sentence numbers it is based on. "
            f'Return JSON: {{"claims":[{{"claim":"...","sentences":[0,1]}}]}}')
    out = backend.chat_json(system, user)
    claims = []
    for c in out.get("claims", [])[:n]:
        idxs = [i for i in c.get("sentences", []) if isinstance(i, int) and 0 <= i < len(sents)]
        if not idxs or not c.get("claim"):
            continue
        claims.append({"claim": c["claim"].strip(),
                       "source_sentences": [{"n": i, "text": sents[i]} for i in idxs[:3]]})
    return claims


def questions_for_claim(claim: str, source: str, backend: Backend, k: int = 3) -> list[str]:
    # Discrimination-simulation finding (see src/evaluation/discrimination_sim.py): questions that
    # state specific domain facts can be answered from world knowledge without the essay and so do
    # not discriminate. Good verification questions must require the student's OWN essay: their
    # reasoning, the specific evidence they used, the choices they made, and how their points connect.
    system = ("You write verification questions for a lecturer to check whether a student genuinely "
              "understands and wrote a claim in their own essay. The crucial property: a knowledgeable "
              "person who has NOT read this essay should be unable to answer well from general "
              "knowledge. So do not ask about general facts; ask the student to reconstruct their own "
              "reasoning, name the specific evidence or examples they used, justify the choices they "
              "made, and explain how this point connects to the rest of their essay. Avoid yes/no "
              "questions and avoid questions that contain their own answer. Reply with JSON only.")
    user = (f"The student's claim: {claim}\nThe sentences in their essay it comes from: {source}\n\n"
            f"Write {k} verification questions that require the student's own essay to answer "
            f"(their reasoning, their chosen evidence, their connections), not general subject "
            f'knowledge. Return JSON: {{"questions":["...","..."]}}')
    out = backend.chat_json(system, user)
    return [q.strip() for q in out.get("questions", []) if isinstance(q, str) and q.strip()][:k]


BLOOM = [
    ("create", r"\b(design|propose|create|devise|what if|how might|suggest a)\b"),
    ("evaluate", r"\b(evaluate|justify|critique|assess|defend|do you agree|to what extent|how convincing|weigh)\b"),
    ("analyse", r"\b(compare|contrast|analyse|analyze|distinguish|why does|how does|what evidence|relationship between|implication)\b"),
    ("apply", r"\b(how would|apply|calculate|use the|give an example|in practice|what would happen)\b"),
    ("understand", r"\b(explain|describe|summaris|summariz|what is meant|why|interpret|in your own words)\b"),
    ("remember", r"\b(define|list|name|state|identify|what is|who|when|where)\b"),
]


def bloom_level(q: str) -> str:
    ql = q.lower()
    for level, pat in BLOOM:
        if re.search(pat, ql):
            return level
    return "understand"


def build_guide(essay_id: str, source_label: str, text: str, backend: Backend,
                n_claims: int) -> dict:
    # Production gate (see wellformed.py): a degenerate question must never reach a lecturer, and
    # the number dropped is recorded so a misbehaving backend is visible rather than hidden.
    from wellformed import well_formed
    sents = sentences(normalize_text(text))
    claims = extract_claims(sents, backend, n_claims)
    n_dropped = 0
    for c in claims:
        src = " ".join(s["text"] for s in c["source_sentences"])
        qs = questions_for_claim(c["claim"], src, backend)
        kept = [q for q in qs if well_formed(q)]
        n_dropped += len(qs) - len(kept)
        c["questions"] = [{"question": q, "bloom_level": bloom_level(q)} for q in kept]
    return {"essay_id": essay_id, "source": source_label, "backend": backend.name,
            "n_sentences": len(sents), "n_dropped_malformed": n_dropped, "claims": claims}


def to_markdown(guide: dict) -> str:
    lines = [f"# Verification Interview Guide: {guide['essay_id']}", "",
             f"Generated by backend `{guide['backend']}` over {guide['n_sentences']} sentences. "
             "Each question is tied to the student's own claim and its source sentences, so the "
             "lecturer can point to where it came from.", ""]
    for i, c in enumerate(guide["claims"], 1):
        lines += [f"## Claim {i}: {c['claim']}", "",
                  "**Source in the submission:**"]
        for s in c["source_sentences"]:
            lines.append(f"> [{s['n']}] {s['text']}")
        lines += ["", "**Verification questions:**"]
        for q in c["questions"]:
            lines.append(f"- {q['question']}  _(Bloom: {q['bloom_level']})_")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="3108a", help="essay id from the corpus")
    ap.add_argument("--source", choices=["ai", "human"], default="ai")
    ap.add_argument("--claims", type=int, default=4)
    ap.add_argument("--backend", choices=["local", "commercial"], default="local")
    ap.add_argument("--provider", choices=["gemini", "anthropic", "openai"], default="gemini",
                    help="commercial provider (Backend A); ignored for --backend local")
    ap.add_argument("--model", default=None,
                    help="model id; defaults to llama3.1:8b (local) or the provider default")
    args = ap.parse_args()

    path = (AI_DIR / f"{args.id}.txt") if args.source == "ai" else (CORPUS_TXT / f"{args.id}.txt")
    if not path.exists():
        raise SystemExit(f"Essay not found: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    if args.backend == "local":
        backend = OllamaBackend(args.model or "llama3.1:8b")
    else:
        from commercial_backend import make_commercial_backend
        try:
            backend = make_commercial_backend(args.provider, args.model)
        except RuntimeError as e:
            raise SystemExit(str(e))

    print(f"Building guide for {args.id} ({args.source}) via {backend.name} ...", flush=True)
    guide = build_guide(args.id, args.source, text, backend, args.claims)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{args.id}_{args.source}.json").write_text(json.dumps(guide, indent=2), encoding="utf-8")
    md = to_markdown(guide)
    (OUTDIR / f"{args.id}_{args.source}.md").write_text(md, encoding="utf-8")
    print(f"\n{md}\n")
    print(f"Saved outputs/verification_guides/{args.id}_{args.source}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
