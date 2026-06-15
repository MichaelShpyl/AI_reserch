"""Generate length-matched AI essays for the BAWE human sample, using local Ollama.

For each human essay in the sample, write one AI essay that answers the same essay
question, in the same discipline register, at about the same length, and on the same
topic. Topic and length matching is the critical validity rule: if AI essays differ
systematically in length or topic, the detector learns those instead of writing style.

Topic is anchored two ways: the essay title (cleaned), and keywords extracted from the
human essay text. The keywords stop the model drifting off topic when a title is vague.

Local model: Ollama with llama3.1:8b (no HPC needed). The model undershoots long targets,
so a continuation loop tops the essay up until it is close to the target length.

Outputs:
  data/processed/ai_essays/<id>.txt        one AI essay per human essay
  data/processed/ai_essays_meta.csv        id, target/actual words, rounds, timing, model

Resumable: ids whose .txt already exists are skipped.

    python src/generation/generate_ai_essays.py --limit 3     # test
    python src/generation/generate_ai_essays.py               # full 640
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "data" / "processed" / "bawe_human_sample.csv"
OUT_DIR = REPO / "data" / "processed" / "ai_essays"
META = REPO / "data" / "processed" / "ai_essays_meta.csv"
BAWE_ROOT = REPO / "data" / "raw" / "bawe"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
SEED = 42
TEMPERATURE = 0.8
NUM_CTX = 8192
MIN_RATIO = 0.95
MAX_ROUNDS = 5
OVERSHOOT = 1.15

SYSTEM = (
    "You are a university student writing a coursework essay. Output only the essay "
    "itself as continuous academic prose. Do not write a title, headings, bullet "
    "points, a reference list, or any framing such as 'Here is' or 'Sure'. Stay strictly "
    "on the given topic and match the academic register of the stated discipline."
)

PREAMBLE_RE = re.compile(
    r"^(here is|here's|sure|certainly|below is|the following)\b.*?:\s*", re.IGNORECASE)

TOPIC_PREFIX_RE = re.compile(
    r"^\s*(dissertation field|field|essay|assignment|title|question)\s*:\s*", re.IGNORECASE)

# Small stopword list plus common academic filler, so keywords are content-bearing.
STOP = set("""
the a an and or but if then else of to in on at by for with from as is are was were be been
being this that these those it its their his her our your they them we you i he she who whom
which what when where why how not no nor so than too very can could should would may might must
will shall do does did done has have had having about into over under again further once here
there all any both each few more most other some such only own same out up down off above below
between through during before after while because thus therefore however moreover furthermore
also within without upon among against towards toward across behind beyond essay study studies
research paper text author students student university chapter section example examples used use
using one two three first second third many much several often general particular important
""".split())


def wc(text: str) -> int:
    return len(text.split())


def clean_text(text: str) -> str:
    text = text.strip().strip('"').strip()
    lines = text.split("\n")
    if lines and (PREAMBLE_RE.match(lines[0]) or lines[0].lstrip().startswith("#")
                  or (lines[0].startswith("**") and lines[0].endswith("**"))):
        lines = lines[1:]
        text = "\n".join(lines).strip()
    return PREAMBLE_RE.sub("", text).strip()


def clean_topic(title) -> str:
    t = "" if not isinstance(title, str) else title.strip()
    t = TOPIC_PREFIX_RE.sub("", t).strip().strip('"').strip()
    return t


def extract_keywords(text: str, n: int = 10) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text.lower())
    counts = Counter(t for t in tokens if t not in STOP and len(t) <= 20)
    return [w for w, _ in counts.most_common(n)]


def trim_to(text: str, target: int) -> str:
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


def ollama_chat(messages: list[dict], num_predict: int, timeout: int = 900) -> str:
    payload = {
        "model": MODEL, "messages": messages, "stream": False,
        "options": {"temperature": TEMPERATURE, "seed": SEED,
                    "num_ctx": NUM_CTX, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


def build_user(topic, discipline, module, keywords, target) -> str:
    kw = ", ".join(keywords) if keywords else topic
    return (f"Write a university coursework essay for the module '{module}' in the "
            f"discipline of {discipline}. The essay must be specifically about this "
            f"topic: {topic}. Address the topic directly and use these key terms "
            f"naturally where relevant: {kw}. Write about {target} words.")


def generate_one(topic, discipline, module, keywords, target) -> tuple[str, int]:
    user = build_user(topic, discipline, module, keywords, target)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    essay = clean_text(ollama_chat(messages, num_predict=min(int(target * 1.7) + 200, 6000)))
    rounds = 1
    while wc(essay) < int(target * MIN_RATIO) and rounds < MAX_ROUNDS:
        remaining = target - wc(essay)
        cont = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": essay},
            {"role": "user", "content": (
                f"Continue the same essay on the same topic from where it stops. Do not "
                f"repeat earlier text and do not restart. Add about {remaining} more "
                f"words, then a brief conclusion if near the end.")},
        ]
        more = clean_text(ollama_chat(cont, num_predict=min(int(remaining * 1.7) + 200, 6000)))
        if not more:
            break
        essay = (essay + " " + more).strip()
        rounds += 1
    return trim_to(essay, target), rounds


def find_corpus_txt(root: Path) -> Path | None:
    for p in root.rglob("CORPUS_TXT"):
        if p.is_dir():
            return p
    return None


def keep_awake() -> None:
    """On Windows, stop the system sleeping while this long run is active."""
    try:
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def main() -> int:
    keep_awake()
    ap = argparse.ArgumentParser(description="Generate length-matched AI essays via Ollama.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=Path, default=SAMPLE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--meta", type=Path, default=META)
    ap.add_argument("--ids", type=str, default=None, help="Comma-separated ids (testing).")
    args = ap.parse_args()

    df = pd.read_csv(args.sample)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        df = df[df["id"].astype(str).isin(wanted)]
    if args.limit:
        df = df.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = find_corpus_txt(BAWE_ROOT)
    if corpus is None:
        print("WARNING: CORPUS_TXT not found; keywords will come from titles only.", flush=True)

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
        topic = clean_topic(row.title) or str(row.discipline)
        keywords = []
        if corpus is not None:
            f = corpus / f"{rid}.txt"
            if f.exists():
                keywords = extract_keywords(f.read_text(encoding="utf-8", errors="ignore"))
        t0 = time.time()
        try:
            essay, rounds = generate_one(topic, row.discipline, row.module, keywords, target)
        except Exception as e:
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
    print(f"\nDone. generated={done} skipped={skipped} failed={failed} of {total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
