"""LLM-as-judge: the supplementary question-quality evaluation (supplementary arm of the evaluation plan).

A judge model rates each verification question on the four-dimension rubric (relevance,
specificity, discrimination potential, cognitive appropriateness), 1 to 5 each. The scope calls for
three commercial judges with cross-model agreement; this implementation is judge-agnostic behind
the same backend interface as question generation, so judges are added as keys/budget allow. The
validation the scope demands is built in: judge scores are correlated against the objective
discrimination-simulation scores for the same questions (Spearman), so the judge is anchored to the
primary measure rather than trusted on its own.

Resumable: scores save after every question, and a 429-blocked run just stops early and resumes
later. Results: outputs/llm_judge.json.

    python src/evaluation/llm_judge.py --provider gemini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "llm_judge.json"
SIM = REPO / "outputs" / "discrimination_sim.json"
sys.path.insert(0, str(REPO / "src" / "question_gen"))

RUBRIC = """You are an experienced university lecturer reviewing verification questions for an
academic-integrity interview. The question was generated from a specific claim in a student's essay.
Rate the question on four dimensions, each an integer 1 (poor) to 5 (excellent):
- relevance: does it target the claim and the essay rather than generic subject matter?
- specificity: does it demand specifics (the student's evidence, reasoning, choices) over vagueness?
- discrimination: would it separate a student who wrote and understood the essay from one who did not?
- cognitive: is the thinking it demands appropriate for checking understanding (not mere recall)?
Reply with JSON only: {"relevance": n, "specificity": n, "discrimination": n, "cognitive": n}"""


def judge_question(backend, claim: str, question: str) -> dict:
    user = (f"The claim from the student's essay: {claim}\n"
            f"The verification question to rate: {question}\n\n"
            f"Rate it now. JSON only.")
    out = backend.chat_json(RUBRIC, user)
    scores = {}
    for k in ("relevance", "specificity", "discrimination", "cognitive"):
        v = out.get(k)
        if not isinstance(v, (int, float)) or not 1 <= v <= 5:
            raise ValueError(f"bad {k} in judge reply: {out}")
        scores[k] = int(v)
    scores["mean"] = round(sum(scores.values()) / 4, 3)
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini", help="gemini|anthropic|openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--guide", default="3108a_ai",
                    help="guide stem whose questions to judge (must exist in verification_guides)")
    args = ap.parse_args()

    from commercial_backend import make_commercial_backend
    backend = make_commercial_backend(args.provider, args.model)

    guide = json.loads((REPO / "outputs" / "verification_guides" / f"{args.guide}.json")
                       .read_text(encoding="utf-8"))
    items = [(c["claim"], q["question"]) for c in guide["claims"] for q in c["questions"]]

    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"judges": {}}
    jkey = backend.name
    data["judges"].setdefault(jkey, {"guide": args.guide, "scores": {}})
    scores = data["judges"][jkey]["scores"]

    done = 0
    for claim, q in items:
        if q in scores:
            done += 1
            continue
        try:
            s = judge_question(backend, claim, q)
        except Exception as e:
            print(f"stopped at question {done + 1}/{len(items)}: {type(e).__name__}: {str(e)[:140]}")
            break
        scores[q] = s
        done += 1
        print(f"[{done}/{len(items)}] mean {s['mean']}  {q[:60]}", flush=True)
        OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Anchor to the objective measure: Spearman correlation of judge mean (and the discrimination
    # dimension alone) with the simulation's per-question discrimination scores.
    if SIM.exists() and scores:
        sim = json.loads(SIM.read_text(encoding="utf-8"))
        sim_scores = {d["question"]: d["discrimination"] for d in sim["questions"]["grounded"]}
        paired = [(scores[q]["mean"], scores[q]["discrimination"], sim_scores[q])
                  for q in scores if q in sim_scores]
        if len(paired) >= 5:
            from scipy import stats
            jm, jd, sd = zip(*paired)
            rho_m = stats.spearmanr(jm, sd)
            rho_d = stats.spearmanr(jd, sd)
            data["judges"][jkey]["anchoring"] = {
                "n_paired": len(paired),
                "spearman_judgeMean_vs_sim": {"rho": round(float(rho_m.statistic), 3),
                                              "p": round(float(rho_m.pvalue), 4)},
                "spearman_judgeDisc_vs_sim": {"rho": round(float(rho_d.statistic), 3),
                                              "p": round(float(rho_d.pvalue), 4)},
                "reading": "Spearman rank correlation between judge ratings and the judge-free "
                           "discrimination simulation for the same questions. Low agreement means "
                           "the judge measures something else; report it either way.",
            }
            print(f"anchoring: judge-mean vs sim rho={rho_m.statistic:.3f} (p={rho_m.pvalue:.3f}); "
                  f"judge-discrimination vs sim rho={rho_d.statistic:.3f} (p={rho_d.pvalue:.3f})")
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {OUT.relative_to(REPO)} ({done}/{len(items)} questions judged by {jkey})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
