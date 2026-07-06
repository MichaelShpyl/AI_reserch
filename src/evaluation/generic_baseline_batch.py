"""Measure the generic-question discrimination baseline on the SAME essays as the backend
comparison, instead of the single pilot essay.

The review of Chapter 8 flagged that the 0.31 generic baseline came from six questions on one essay
(3108a). This script runs the same six generic questions on every essay in the balanced comparison
set, with the essay's own normalised text as the source (capped at the same 100 sentences the guides
use), scored by the same discrimination function. Output feeds the dashed baseline in Figure 8.2 and
the chapter text. Resumable per essay.

    python src/evaluation/generic_baseline_batch.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
BATCH = REPO / "outputs" / "backend_comparison_batch.json"
OUT = REPO / "outputs" / "generic_baseline_batch.json"
sys.path.insert(0, str(REPO / "src" / "evaluation"))
sys.path.insert(0, str(REPO / "src" / "question_gen"))
from discrimination_sim import discrimination, boot_ci, GENERIC  # noqa: E402
from generate_questions import sentences  # noqa: E402
from text_normalize import normalize_text  # noqa: E402


def main() -> int:
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    essays = batch.get("balanced", {}).get("essays") or []
    if not essays:
        raise SystemExit("No balanced essay list in backend_comparison_batch.json")

    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"essays": {}}
    for i, eid in enumerate(essays, 1):
        if eid in data["essays"] and len(data["essays"][eid].get("scores", [])) == len(GENERIC):
            print(f"[{i}/{len(essays)}] {eid}: done, skip", flush=True)
            continue
        text = (AI_DIR / f"{eid}.txt").read_text(encoding="utf-8", errors="ignore")
        # Cap the source so it fits the embedding model's context (the full essay 500s Ollama's
        # embeddings endpoint): first 50 sentences, hard char guard, same text for answer and embed.
        source = " ".join(sentences(normalize_text(text), cap=50))[:6000]
        scores = []
        print(f"[{i}/{len(essays)}] {eid}: scoring {len(GENERIC)} generic questions", flush=True)
        for q in GENERIC:
            d = discrimination(q, source)
            scores.append(d["discrimination"])
        data["essays"][eid] = {"scores": scores, "mean": round(float(np.mean(scores)), 4)}
        OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")

    pooled = [s for e in data["essays"].values() for s in e["scores"]]
    mean, ci = boot_ci(pooled)
    per_essay = [e["mean"] for e in data["essays"].values()]
    data["pooled"] = {"n_essays": len(data["essays"]), "n_questions": len(pooled),
                      "mean": mean, "ci95": ci,
                      "per_essay_means": per_essay,
                      "reading": "The six generic questions from the pilot, scored on each balanced "
                                 "comparison essay with the essay's own text as source. This replaces "
                                 "the single-essay 0.31 baseline with a same-essays measurement."}
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nGENERIC BASELINE (batch): essays={len(data['essays'])} q={len(pooled)} "
          f"mean={mean} CI={ci}")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
