"""Generate length-matched AI essays for the BAWE human sample, using local Ollama.

For each human essay in the sample, write one AI essay that answers the same essay
question, in the same discipline register, at about the same length. Topic and length
matching is the critical validity rule: if AI essays are systematically shorter or
longer than human ones, the detector learns length instead of style.

Local model: Ollama with llama3.1:8b (no HPC needed). The model tends to undershoot long
targets, so a continuation loop tops the essay up until it is close to the target length.

Outputs:
  data/processed/ai_essays/<id>.txt        one AI essay per human essay
  data/processed/ai_essays_meta.csv        id, target/actual words, rounds, timing, model

The script is resumable: ids whose .txt already exists are skipped, so a long run can be
stopped and restarted (re-run the same command).

Test small first, then run the full set in the background:
    python src/generation/generate_ai_essays.py --limit 3
    python src/generation/generate_ai_essays.py            # full 640
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "data" / "processed" / "bawe_human_sample.csv"
OUT_DIR = REPO / "data" / "processed" / "ai_essays"
META = REPO / "data" / "processed" / "ai_essays_meta.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
SEED = 42
TEMPERATURE = 0.8
NUM_CTX = 8192
MIN_RATIO = 0.95     # keep topping up until at least this fraction of target words
MAX_ROUNDS = 5       # cap continuation calls per essay
OVERSHOOT = 1.15     # trim back to target if longer than this fraction

SYSTEM = (
    "You are a university student writing a coursework essay. Output only the essay "
    "itself as continuous academic prose. Do not write a title, headings, bullet "
    "points, a reference list, or any framing such as 'Here is' or 'Sure'. Match the "
    "academic register of the stated discipline."
)

PREAMBLE_RE = re.compile(
    r"^(here is|here's|sure|certainly|below is|the following)\b.*?:\s*", re.IGNORECASE)


def wc(text: str) -> int:
    return len(text.split())


def clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    # Drop a leading meta line or markdown title if present.
    lines = text.split("\n")
    if lines and (PREAMBLE_RE.match(lines[0]) or lines[0].lstrip().startswith("#")
                  or (lines[0].startswith("**") and lines[0].endswith("**"))):
        lines = lines[1:]
        text = "\n".join(lines).strip()
    text = PREAMBLE_RE.sub("", text).strip()
    return text


def trim_to(text: str, target: int) -> str:
    """Trim to about target words, keeping whole sentences."""
    if wc(text) <= int(target * OVERSHOOT):
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, count = [], 0
    for s in sentences:
        out.append(s)
        count += wc(s)
        if count >= target:
            break
    return " ".join(out).strip()


def ollama_chat(messages: list[dict], num_predict: int, timeout: int = 600) -> str:
    payload = {
        "model": MODEL, "messages": messages, "stream": False,
        "options": {"temperature": TEMPERATURE, "seed": SEED,
                    "num_ctx": NUM_CTX, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def generate_one(title: str, discipline: str, module: str, target: int) -> tuple[str, int]:
    """Generate one essay near the target length. Returns (text, rounds)."""
    topic = title if isinstance(title, str) and title.strip() else discipline
    user = (f"Discipline: {discipline}. Module: {module}. Essay question: {topic}. "
            f"Write an essay of about {target} words. Aim close to {target} words.")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    first = clean(ollama_chat(messages, num_predict=min(int(target * 1.7) + 200, 6000)))
    essay = first
    rounds = 1
    while wc(essay) < int(target * MIN_RATIO) and rounds < MAX_ROUNDS:
        remaining = target - wc(essay)
        cont = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": essay},
            {"role": "user", "content": (
                f"Continue the same essay from where it stops. Do not repeat earlier "
                f"text and do not restart. Add about {remaining} more words, then a brief "
                f"conclusion if you are near the end.")},
        ]
        more = clean(ollama_chat(cont, num_predict=min(int(remaining * 1.7) + 200, 6000)))
        if not more:
            break
        essay = (essay + " " + more).strip()
        rounds += 1
    return trim_to(essay, target), rounds


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate length-matched AI essays via Ollama.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N essays (testing).")
    ap.add_argument("--sample", type=Path, default=SAMPLE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--meta", type=Path, default=META)
    args = ap.parse_args()

    df = pd.read_csv(args.sample)
    if args.limit:
        df = df.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta_exists = args.meta.exists()
    meta_f = args.meta.open("a", newline="", encoding="utf-8")
    writer = csv.writer(meta_f)
    if not meta_exists:
        writer.writerow(["id", "discipline", "target_words", "actual_words",
                         "ratio", "rounds", "seconds", "model"])

    total = len(df)
    done = skipped = failed = 0
    for i, row in enumerate(df.itertuples(index=False), start=1):
        rid = str(row.id)
        out_path = args.out_dir / f"{rid}.txt"
        if out_path.exists():
            skipped += 1
            continue
        target = int(row.words)
        t0 = time.time()
        try:
            essay, rounds = generate_one(row.title, row.discipline, row.module, target)
        except Exception as e:  # keep the batch alive on a single failure
            failed += 1
            print(f"[{i}/{total}] {rid} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        secs = time.time() - t0
        actual = wc(essay)
        out_path.write_text(essay, encoding="utf-8")
        writer.writerow([rid, row.discipline, target, actual,
                         round(actual / target, 3), rounds, round(secs, 1), MODEL])
        meta_f.flush()
        done += 1
        print(f"[{i}/{total}] {rid} target={target} actual={actual} "
              f"ratio={actual/target:.2f} rounds={rounds} {secs:.0f}s", flush=True)

    meta_f.close()
    print(f"\nDone. generated={done} skipped={skipped} failed={failed} of {total}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
