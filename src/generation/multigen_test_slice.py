"""A multi-generator test slice: does the detector catch generators it never trained on, at home?

Chapter 9 names the detection corpus's single generator (Llama 3.1) as a limitation: the OUTFOX
transfer test covers unseen generators, but on someone else's corpus design. This closes the gap on
home ground. For a sample of test-split human essays it generates matched AI counterparts with two
commercial generators the detector never saw (Gemini and GPT-4o-mini), using the SAME prompt recipe
as the original corpus (title plus extracted keywords plus target length, the identical system
prompt), then scores everything with the detector of record and the hybrid. Reported per generator:
detection rate, plus the length-match check so a trivial length tell cannot explain the result.

Generation is API-only and resumable; scoring runs afterwards (score_multigen.py).

    python src/generation/multigen_test_slice.py --n 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "processed" / "bawe_human_sample_manifest.csv"
HOLDINGS = REPO / "data" / "raw" / "bawe" / "download" / "documentation" / "BAWE.xls"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
OUTDIR = REPO / "data" / "processed" / "multigen_test"
STATE = OUTDIR / "state.json"
SEED = 42
GENERATORS = ["gemini", "openai"]

sys.path.insert(0, str(REPO / "src" / "generation"))
sys.path.insert(0, str(REPO / "src" / "question_gen"))
from generate_ai_essays import SYSTEM, build_user, extract_keywords  # noqa: E402


def sample_sources(n: int) -> pd.DataFrame:
    man = pd.read_csv(MANIFEST)
    test_ids = man[man["split"] == "test"]["id"].tolist()
    hold = pd.read_excel(HOLDINGS)
    hold = hold[hold["id"].isin(test_ids)][["id", "title", "module", "discipline", "words"]]
    hold = hold.dropna(subset=["title", "words"])
    return hold.sample(n=min(n, len(hold)), random_state=SEED).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    from commercial_backend import make_commercial_backend
    backends = {}
    for g in GENERATORS:
        try:
            backends[g] = make_commercial_backend(g, None)
        except RuntimeError as e:
            print(f"[skip] {g}: {e}", flush=True)

    src = sample_sources(args.n)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"done": {}}

    for _, row in src.iterrows():
        eid = row["id"]
        kw = []
        f = CORPUS_TXT / f"{eid}.txt"
        if f.exists():
            kw = extract_keywords(f.read_text(encoding="utf-8", errors="ignore"))
        target = int(row["words"])
        user = build_user(str(row["title"]), str(row["discipline"]), str(row["module"]), kw, target)
        for g, be in backends.items():
            key = f"{eid}_{g}"
            out_f = OUTDIR / f"{key}.txt"
            if state["done"].get(key) and out_f.exists():
                continue
            try:
                # chat_json parses JSON; for prose we need the raw completion, so use the
                # backend's raw chat if present, else ask for JSON-wrapped essay text.
                if hasattr(be, "chat_text"):
                    essay = be.chat_text(SYSTEM, user)
                else:
                    out = be.chat_json(
                        SYSTEM + " Reply with JSON only.",
                        user + '\nReturn JSON: {"essay": "<the full essay text>"}')
                    essay = out.get("essay", "")
            except Exception as e:
                print(f"  {key} FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
                continue
            if not essay or len(essay.split()) < 100:
                print(f"  {key}: too short ({len(essay.split())} words), skipped", flush=True)
                continue
            out_f.write_text(essay, encoding="utf-8")
            state["done"][key] = {"source_words": target, "got_words": len(essay.split())}
            STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
            ratio = len(essay.split()) / max(target, 1)
            print(f"  {key}: {len(essay.split())} words (target {target}, ratio {ratio:.2f})",
                  flush=True)
            time.sleep(1)

    n_ok = len(state["done"])
    print(f"\n{n_ok} generations saved in {OUTDIR.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
