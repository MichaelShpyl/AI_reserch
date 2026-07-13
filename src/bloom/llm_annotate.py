"""LLM annotation for Bloom levels, validated against gold before it is trusted.

The Bloom classifier's higher-order classes are starved (19 "analyse" questions in all of
EduQG's 903), and no human labelling is in scope. The candidate fix is silver labels from a
strong LLM, but this project does not trust an automatic annotator it has not measured. So the
script has two modes, and the second is only worth running if the first goes well:

  --source gold    label all 903 gold EduQG questions and report agreement with the gold
                   labels, per class, which is the annotator's job interview
  --source v3pool  label the project's own generated verification questions
                   (data/interim/qg_v3_pairs.json), the classifier's actual deployment domain,
                   to populate the higher-order classes with in-domain silver data

Resumable; state in outputs/bloom_llm_annotation.json keyed by source and provider.

    python src/bloom/llm_annotate.py --source gold
    python src/bloom/llm_annotate.py --source v3pool --limit 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "bloom"))
OUT = REPO / "outputs" / "bloom_llm_annotation.json"

LEVELS = ["remember", "understand", "apply", "analyse"]

SYSTEM = """You label exam and verification questions with one revised-Bloom cognitive level.
Answer with JSON only: {"level": "<one of: remember, understand, apply, analyse>"}.

Definitions and decision rules:
- remember: the answer is a fact, name, date, definition or list that can be recalled verbatim.
  ("What is the powerhouse of the cell?")
- understand: the answer requires explaining, summarising, classifying or restating an idea in
  new words, but not using it on a new case. ("Explain why the moon has phases.")
- apply: the answer requires using a method, rule or concept on a concrete case or numbers.
  ("Use the formula to find the current at 12 volts.")
- analyse: the answer requires taking a position or artefact apart: comparing, contrasting,
  finding assumptions, relating evidence to a claim, or explaining how parts of an argument fit
  together. ("How does the author's second example undermine her first claim?")

Pick the HIGHEST level the question genuinely demands, not the topic's difficulty. A question
that merely names a hard concept is still remember. A question asking the writer to justify
their own choice of evidence is analyse."""


LEVEL_OF_INT = {1: "remember", 2: "understand", 3: "apply", 4: "analyse"}


def load_gold() -> list[dict]:
    from train_bloom_classifier import load_rows
    return [{"question": r["text"], "level": LEVEL_OF_INT[r["label"]]} for r in load_rows()]


def gold_splits() -> tuple[set[str], set[str]]:
    """Replicate the trainer's exact split (seed 42, stratified 70/15/15) so few-shot exemplars
    come only from the train split and agreement can be judged on the untouched test split."""
    from sklearn.model_selection import train_test_split
    rows = load_gold()
    X = [r["question"] for r in rows]
    y = [r["level"] for r in rows]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_va, X_te, _, _ = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                        random_state=42)
    return set(X_tr), set(X_te)


def few_shot_block() -> str:
    """Four seeded exemplars per class from the TRAIN split, teaching the gold convention."""
    import random
    train_qs, _ = gold_splits()
    rows = [r for r in load_gold() if r["question"] in train_qs]
    rng = random.Random(42)
    lines = ["", "Worked examples of this dataset's labelling convention (follow it even where",
             "it differs from your own reading of the definitions):"]
    for lvl in LEVELS:
        pool = [r for r in rows if r["level"] == lvl]
        rng.shuffle(pool)
        for r in pool[:4]:
            q = r["question"][:220]
            lines.append(f'- {lvl}: "{q}"')
    return "\n".join(lines)


def load_v3pool(limit: int) -> list[dict]:
    pairs = json.loads((REPO / "data" / "interim" / "qg_v3_pairs.json")
                       .read_text(encoding="utf-8"))["pairs"]
    seen, rows = set(), []
    for p in pairs:
        q = p["question"]
        if q not in seen:
            seen.add(q)
            rows.append({"question": q, "level": None})
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["gold", "v3pool"], default="gold")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap for v3pool")
    ap.add_argument("--few-shot", action="store_true",
                    help="add train-split gold exemplars to the prompt (convention teaching)")
    args = ap.parse_args()

    from commercial_backend import make_commercial_backend
    backend = make_commercial_backend(args.provider, args.model)

    system = SYSTEM + (few_shot_block() if args.few_shot else "")

    rows = load_gold() if args.source == "gold" else load_v3pool(args.limit)
    print(f"{len(rows)} questions to annotate ({args.source}) with {backend.name}"
          f"{' [few-shot]' if args.few_shot else ''}")

    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    key = f"{args.source}|{backend.name}" + ("|fs" if args.few_shot else "")
    data.setdefault(key, {"labels": {}})
    labels = data[key]["labels"]

    done = 0
    for r in rows:
        q = r["question"]
        if q in labels:
            done += 1
            continue
        try:
            reply = backend.chat_json(system, f"Question: {q}")
            lvl = str(reply.get("level", "")).strip().lower()
            if lvl not in LEVELS:
                raise ValueError(f"bad level {lvl!r}")
        except Exception as e:
            print(f"stopped at {done + 1}/{len(rows)}: {type(e).__name__}: {str(e)[:120]}")
            break
        labels[q] = lvl
        done += 1
        if done % 25 == 0:
            print(f"[{done}/{len(rows)}]", flush=True)
            OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")

    if args.source == "gold":
        def agreement(pairs):
            if not pairs:
                return None
            acc = sum(a == b for a, b in pairs) / len(pairs)
            per = {}
            for lvl in LEVELS:
                tp = sum(1 for a, b in pairs if a == lvl and b == lvl)
                fp = sum(1 for a, b in pairs if a == lvl and b != lvl)
                fn = sum(1 for a, b in pairs if a != lvl and b == lvl)
                p = tp / (tp + fp) if tp + fp else 0.0
                rc = tp / (tp + fn) if tp + fn else 0.0
                per[lvl] = {"precision": round(p, 3), "recall": round(rc, 3),
                            "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0,
                            "n_gold": sum(1 for _, b in pairs if b == lvl)}
            return {"n": len(pairs), "accuracy": round(acc, 4), "per_class": per,
                    "annotator_distribution": dict(Counter(a for a, _ in pairs))}

        paired = [(labels[r["question"]], r["level"], r["question"])
                  for r in rows if r["question"] in labels]
        _, test_qs = gold_splits()
        report = agreement([(a, b) for a, b, _ in paired])
        report_test = agreement([(a, b) for a, b, q in paired if q in test_qs])
        if report:
            data[key]["agreement_vs_gold"] = report
            data[key]["agreement_vs_gold_testsplit"] = report_test
            OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
            print("ALL:", json.dumps(report, indent=1))
            print("TEST SPLIT ONLY (fair for few-shot):", json.dumps(report_test, indent=1))
    else:
        dist = Counter(labels[r["question"]] for r in rows if r["question"] in labels)
        print("silver label distribution:", dict(dist))
    print(f"Saved {OUT.relative_to(REPO)} ({done}/{len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
