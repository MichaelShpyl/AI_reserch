"""The integrated Verification Interview Guide: every trained component, the approved design.

This is the pipeline in its final form, the design my supervisor approved (Meeting 5): the prompted
extractor supplies readable claim phrasing with sentence citations, and the trained argument miner
supplies the provenance in the student's own words, so each claim carries both a paraphrase and the
verbatim argument spans (major claim / claim / premise) the trained model found in the cited
sentences. Questions are written by the QLoRA-fine-tuned local backend (v3, the on-style adapter),
filtered through the well-formedness gate, and the assembler scores the submission with the hybrid
detector and tags each question with the trained Bloom classifier.

Every model here is a trained component of this project. VRAM is 8 GB, so the phases are sequenced:
prompted claims with Ollama (then freed), trained span extraction on CPU, question writing with the
v3 adapter on the GPU (then freed). The output JSON feeds assemble_guide.py.

    python src/question_gen/integrated_guide.py --id 3108a --source ai --claims 4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
GUIDES = REPO / "outputs" / "verification_guides"
V3_ADAPTER = REPO / "models" / "qg_finetune_qwen3b_v3"
OLLAMA_EXE = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
for p in ("question_gen", "detection", "argument_mining"):
    sys.path.insert(0, str(REPO / "src" / p))


def free_ollama():
    for m in ("llama3.1:8b", "nomic-embed-text"):
        try:
            subprocess.run([str(OLLAMA_EXE), "stop", m], capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            pass
    time.sleep(3)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def overlap(a: str, b: str) -> float:
    """Word-overlap of the shorter normalised string against the longer (0..1)."""
    wa, wb = set(norm(a).split()), set(norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="3108a")
    ap.add_argument("--source", choices=["ai", "human"], default="ai")
    ap.add_argument("--claims", type=int, default=4)
    args = ap.parse_args()

    path = (AI_DIR / f"{args.id}.txt") if args.source == "ai" else (CORPUS_TXT / f"{args.id}.txt")
    text = path.read_text(encoding="utf-8", errors="ignore")

    from generate_questions import OllamaBackend, sentences, extract_claims
    from text_normalize import normalize_text
    sents = sentences(normalize_text(text))

    # Phase 1: prompted claims (readable phrasing + sentence citations), Ollama.
    print("== Phase 1: prompted claims (Ollama) ==", flush=True)
    be = OllamaBackend("llama3.1:8b")
    claims = extract_claims(sents, be, args.claims)
    free_ollama()

    # Phase 2: trained argument spans (CPU), mapped to the cited sentences.
    print("== Phase 2: trained argument spans (CPU) ==", flush=True)
    from extract_spans import ClaimExtractor
    spans = ClaimExtractor().extract(text)
    for c in claims:
        cited = c["source_sentences"]
        c["argument_spans"] = []
        seen = set()
        for sp in spans:
            best = max((overlap(sp["text"], s["text"]) for s in cited), default=0.0)
            if best >= 0.5 and sp["text"] not in seen:
                c["argument_spans"].append({"type": sp["type"], "text": sp["text"]})
                seen.add(sp["text"])
        print(f"  claim '{c['claim'][:50]}...': {len(c['argument_spans'])} trained spans", flush=True)

    # Phase 3: questions from the v3 fine-tuned local backend, gated.
    print("== Phase 3: questions (v3 fine-tuned Qwen 3B) ==", flush=True)
    from hf_backend import HFBackend
    from generate_questions import questions_for_claim, bloom_level
    from wellformed import well_formed
    backend = HFBackend(adapter=str(V3_ADAPTER))
    n_dropped = 0
    for c in claims:
        src = " ".join(s["text"] for s in c["source_sentences"])
        qs = questions_for_claim(c["claim"], src, backend, k=3)
        kept = [q for q in qs if well_formed(q)]
        n_dropped += len(qs) - len(kept)
        c["questions"] = [{"question": q, "bloom_level": bloom_level(q)} for q in kept]
    backend.close()

    guide = {"essay_id": args.id, "source": args.source,
             "backend": f"local-hf:v3:{V3_ADAPTER.name}",
             "claim_provenance": "prompted phrasing + trained argument spans (approved design)",
             "n_sentences": len(sents), "n_dropped_malformed": n_dropped, "claims": claims}
    GUIDES.mkdir(parents=True, exist_ok=True)
    out = GUIDES / f"{args.id}_{args.source}.json"
    out.write_text(json.dumps(guide, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(REPO)} "
          f"({len(claims)} claims, {sum(len(c['questions']) for c in claims)} questions, "
          f"{n_dropped} dropped by the gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
