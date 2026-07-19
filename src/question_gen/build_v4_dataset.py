"""Build the v4 fine-tuning dataset: claim-anchored questions that never name the claim.

The scaled comparison left one gap standing: every well-formed question writer, v3 included,
sits below the generic-question baseline (about 0.30). Section 8.3 explains why. The four
generic champions ("what is your main argument?") give a context-blind reader nothing to work
with, while grounded questions leak content by naming what the claim says, and the leak is what
the blind model answers from.

v4 tests whether that gap is closable by construction. The teacher writes questions that stay
anchored to one claim (the guide prints the claim right next to the question, so the lecturer
loses nothing) but refer to it only as "this claim" or "the point you make here", never quoting
or paraphrasing its content. If it works, discrimination should rise toward the generic baseline
while keeping per-claim provenance and Bloom spread. The known risk is mode collapse into a few
stock phrasings, so the evaluation must report question uniqueness next to the score, the same
way degeneracy is reported everywhere else.

Same teacher (local Llama 3.1 8B), same contamination guard, same target size and gate as v3, so
the v3-versus-v4 comparison isolates the prompting change.

    python src/question_gen/build_v4_dataset.py --target 2600
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
OUT = REPO / "data" / "interim" / "qg_v4_pairs.json"
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "detection"))

SEED = 42


STOP = set("""a an the this that these those of in on at to for with by from about as is are was
were be been being have has had do does did will would could should may might your you their
they it its and or but not no than then so such what which who whom whose when where why how""".split())
ANCHORS = {"claim", "point", "position", "argument", "essay", "evidence", "reasoning",
           "paragraph", "section", "conclusion", "sentence", "sentences", "example", "examples"}


def content_free(question: str, claim: str, source: str) -> bool:
    """The mechanical gate the instruction alone cannot enforce: a v4 question must share no
    content word with the claim, and no capitalized entity with the claim or source. Anchor
    words like 'claim' and 'essay' are allowed; everything else substantive is a leak."""
    import re as _re
    q_words = {w.lower() for w in _re.findall(r"[A-Za-z']+", question)}
    claim_content = {w.lower() for w in _re.findall(r"[A-Za-z']+", claim)
                     if len(w) > 3 and w.lower() not in STOP}
    if (q_words & claim_content) - ANCHORS:
        return False
    entities = {w.lower() for w in _re.findall(r"(?<=[a-z,;] )[A-Z][a-z]{2,}", claim + " " + source)}
    entities |= {w.lower() for w in _re.findall(r"[A-Z][a-z]{2,}", claim + " " + source)}
    if (q_words & entities) - ANCHORS - STOP:
        return False
    return True


def questions_v4(claim: str, source: str, backend, k: int = 3) -> list[str]:
    system = ("You write verification questions for a lecturer to check whether a student "
              "genuinely understands and wrote a claim in their own essay. Two hard rules. "
              "First, the question must NOT quote, paraphrase or name the specific content of "
              "the claim or its topic; refer to it only as 'this claim', 'this point' or 'the "
              "position you take here'. The lecturer shows the claim beside the question, so "
              "nothing is lost, and a question that names the content lets an outsider answer "
              "from general knowledge. Second, the question must demand the student's own essay "
              "to answer: the reasoning that led to the point, the evidence they chose and why "
              "that evidence over alternatives, what would weaken the point, how it connects to "
              "the essay's overall argument. Good examples of the FORM (vary yours, do not copy): "
              "'Walk me through the reasoning that led you to this claim.' / 'Which piece of "
              "evidence behind this point would you defend hardest if challenged, and why that "
              "one?' / 'If this claim turned out to be wrong, which other part of your essay "
              "would suffer most?' Vary the phrasing and angle across the questions. Avoid "
              "yes/no questions. Reply with JSON only.")
    user = (f"The student's claim (context for you, never to be named in the question): {claim}\n"
            f"Its source sentences (context only): {source}\n\n"
            f"Write {k + 5} distinct verification questions following both rules. "
            f'Return JSON: {{"questions":["...","..."]}}')
    out = backend.chat_json(system, user)
    cands = [q.strip() for q in out.get("questions", []) if isinstance(q, str) and q.strip()]
    passing = [q for q in cands if content_free(q, claim, source)]
    return passing[:k]


def eval_essay_ids() -> set[str]:
    ids = {"3108a"}
    batch = REPO / "outputs" / "backend_comparison_batch.json"
    if batch.exists():
        ids |= set(json.loads(batch.read_text(encoding="utf-8"))
                   .get("balanced", {}).get("essays", []))
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2600)
    ap.add_argument("--claims", type=int, default=3)
    args = ap.parse_args()

    from generate_questions import OllamaBackend, sentences, extract_claims
    from text_normalize import normalize_text
    from wellformed import well_formed

    excluded = eval_essay_ids()
    all_ids = sorted(p.stem for p in AI_DIR.glob("*.txt") if p.stem not in excluded)
    rng = random.Random(SEED)
    rng.shuffle(all_ids)
    print(f"{len(all_ids)} candidate essays ({len(excluded)} evaluation essays excluded)",
          flush=True)

    state = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"pairs": [],
                                                                              "done": []}
    done = set(state["done"])
    be = OllamaBackend("llama3.1:8b")
    t0 = time.time()

    for eid in all_ids:
        if len(state["pairs"]) >= args.target:
            break
        if eid in done:
            continue
        try:
            text = (AI_DIR / f"{eid}.txt").read_text(encoding="utf-8", errors="ignore")
            sents = sentences(normalize_text(text))
            claims = extract_claims(sents, be, args.claims)
            n_new = 0
            for c in claims:
                src = " ".join(s["text"] for s in c["source_sentences"])
                passage = f"Claim: {c['claim']}\nSource sentences from the essay: {src}"
                try:
                    qs = questions_v4(c["claim"], src, be, k=3)
                except Exception as ex:
                    print(f"    {eid} question gen failed: {type(ex).__name__}", flush=True)
                    continue
                for q in qs:
                    if well_formed(q):
                        state["pairs"].append({"passage": passage[:1500], "question": q})
                        n_new += 1
        except Exception as ex:
            print(f"  {eid} FAILED: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            n_new = 0
        state["done"].append(eid)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(state), encoding="utf-8")
        rate = len(state["pairs"]) / max((time.time() - t0) / 3600, 1e-9)
        print(f"  {eid}: +{n_new} pairs, total {len(state['pairs'])}/{args.target} "
              f"({rate:.0f}/h)", flush=True)

    print(f"\nDone: {len(state['pairs'])} pairs from {len(state['done'])} essays -> {OUT}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
