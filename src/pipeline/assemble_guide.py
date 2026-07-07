"""Output assembler (pipeline component 6): the lecturer's Verification Interview Guide.

This is the artifact the whole pipeline exists to produce. For one submission it assembles:
  1. the detection verdict from the trained DeBERTa detector (real inference, not a cached number),
  2. a plain-language explanation of what drives the detector (the SHAP-validated stylometric
     drivers), with the honesty caveats the dissertation establishes (domain fragility, the
     conversation-not-accusation position),
  3. the claims extracted from the submission, each quoted to its exact source sentences,
  4. the verification questions per claim, re-tagged by the TRAINED Bloom classifier (component 5),
     with higher-level tags marked advisory per the classifier's measured limits,
  5. a suggested marking rubric for the conversation.

Renders Markdown, then a .docx (pandoc) and a .pdf (LibreOffice headless) when available.

    python src/pipeline/assemble_guide.py --id 3108a --source ai
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
GUIDES = REPO / "outputs" / "verification_guides"
DETECTOR = REPO / "models" / "detector" / "checkpoint-110"
BLOOM_MODEL = REPO / "models" / "bloom_classifier"
PANDOC = Path.home() / "AppData" / "Local" / "Pandoc" / "pandoc.exe"
SOFFICE = Path("C:/Program Files/LibreOffice/program/soffice.exe")
sys.path.insert(0, str(REPO / "src" / "detection"))
from text_normalize import normalize_text  # noqa: E402

BLOOM_LEVELS = ["remember", "understand", "apply", "analyse"]
# The classifier's measured per-class reliability (outputs/bloom_classifier.json): tags above
# `understand` are advisory until the label supply improves.
ADVISORY = {"apply", "analyse"}


def detect(text: str) -> dict:
    """Score the submission with the hybrid detector (Section 6.7), falling back to the transformer
    alone if the hybrid has not been saved. Returns prob(AI), the verdict, and the components."""
    import sys as _sys
    _sys.path.insert(0, str(REPO / "src" / "detection"))
    from hybrid_detect import hybrid_detect
    return hybrid_detect(text)


def bloom_tag(questions: list[str]) -> list[str]:
    """Tag questions with the trained Bloom classifier (component 5)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BLOOM_MODEL))
    model = AutoModelForSequenceClassification.from_pretrained(str(BLOOM_MODEL)).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    out = []
    with torch.no_grad():
        for q in questions:
            enc = tok(q, truncation=True, max_length=64, return_tensors="pt")
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            pred = int(model(**enc).logits.argmax(-1))
            out.append(BLOOM_LEVELS[pred])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def drivers() -> list[str]:
    """The SHAP-validated stylometric drivers, phrased for a lecturer (Chapter 5)."""
    return [
        "Average word length: consistently longer words push a text toward the AI class.",
        "Vocabulary richness: a wider, less repetitive vocabulary pushes toward the human class.",
        "Sentence-length variation: humans vary sentence length more; uniform sentences look AI-written.",
        "Auxiliary-verb and determiner densities: small grammatical habits that differ between the classes.",
    ]


def build_markdown(essay_id: str, source: str, det: dict, guide: dict,
                   tags: dict[str, str]) -> str:
    L = []
    L.append(f"# Verification Interview Guide")
    L.append("")
    L.append(f"**Submission:** {essay_id}  |  **Generated:** {date.today().isoformat()}  |  "
             f"**Question backend:** {guide.get('backend', 'local')}")
    L.append("")
    L.append("> This guide is evidence for a conversation with the student, not an accusation. "
             "The detector's judgement is fallible, and the fair use of this document is to ask, "
             "listen, and weigh the answers.")
    L.append("")
    L.append("## 1. Detection summary")
    L.append("")
    verdict = "flagged as likely AI-generated" if det["flagged"] else "not flagged"
    which = ("hybrid detector (transformer fused with stylometric features and GPT-2 perplexity)"
             if det.get("detector") == "hybrid" else "trained transformer detector")
    L.append(f"The {which} scores this submission at **{det['prob_ai']}** probability "
             f"of being AI-generated, so it is **{verdict}**. No score of 1.0 exists on this scale: "
             f"the model is never certain, only confident. On matched in-domain test data the "
             f"detector's F1 is 0.99; on out-of-domain academic text its false-positive rate rises "
             f"sharply, so treat the score as a reason to talk, never as proof.")
    comp = det.get("components", {})
    if det.get("detector") == "hybrid" and comp:
        L.append("")
        L.append(f"*Component views: transformer {comp.get('transformer')}, "
                 f"style-plus-perplexity {comp.get('style_plus_perplexity')}. The hybrid fuses both "
                 f"because the style half keeps the detector calmer on unusual but human writing.*")
    L.append("")
    L.append("**What drives decisions like this one** (validated by faithfulness testing):")
    L.append("")
    for d in drivers():
        L.append(f"- {d}")
    L.append("")
    L.append("## 2. The student's claims and where they come from")
    L.append("")
    L.append("Each claim below is phrased for readability, then anchored two ways so nothing here is "
             "invented: to the exact sentence numbers it was drawn from, and to the verbatim "
             "argument spans the trained argument miner found in those sentences, each labelled by "
             "its role (major claim, claim, or premise). Every question can be traced to the "
             "student's own words.")
    L.append("")
    for i, c in enumerate(guide["claims"], 1):
        L.append(f"### Claim {i}: {c['claim']}")
        L.append("")
        L.append("**Source in the submission (cited sentences):**")
        L.append("")
        for s in c["source_sentences"]:
            L.append(f"> [{s['n']}] {s['text']}")
        L.append("")
        # Show only substantive spans: the trained miner is weakest on short claim fragments
        # (span-F1 0.44 on claims), so a two-word fragment is not useful provenance.
        spans = [sp for sp in c.get("argument_spans", []) if len(sp["text"].split()) >= 5]
        if spans:
            L.append("**Argument spans found by the trained miner (verbatim, in the student's words):**")
            L.append("")
            for sp in spans:
                L.append(f"- *{sp['type']}:* “{sp['text']}”")
            L.append("")
        L.append("**Verification questions:**")
        L.append("")
        for q in c["questions"]:
            tag = tags.get(q["question"], q.get("bloom_level", "understand"))
            adv = " (advisory)" if tag in ADVISORY else ""
            L.append(f"- {q['question']}  *(Bloom: {tag}{adv})*")
        L.append("")
    L.append("## 3. Suggested marking rubric for the conversation")
    L.append("")
    L.append("| Level | What you hear | Suggested reading |")
    L.append("|---|---|---|")
    L.append("| Strong | Reconstructs the claim, names their evidence, extends it unprompted | "
             "Understanding demonstrated; the flag is likely a false positive or the tool use was superficial |")
    L.append("| Partial | Recalls the claim but cannot say why the evidence supports it | "
             "Mixed picture; consider a follow-up task on the weak areas |")
    L.append("| Weak | Cannot restate their own claim or where it came from | "
             "Understanding not demonstrated; proceed per your institution's academic-integrity process |")
    L.append("")
    L.append("*Bloom tags above `understand` are marked advisory: the trained classifier is reliable "
             "on lower levels and under-trained on higher ones (see the technical report). The tags "
             "order the questions from recall to reasoning; start low, move up.*")
    L.append("")
    return "\n".join(L)


def render(md_path: Path) -> None:
    docx = md_path.with_suffix(".docx")
    pdf = md_path.with_suffix(".pdf")
    if PANDOC.exists():
        subprocess.run([str(PANDOC), str(md_path), "-o", str(docx)], check=True, timeout=120)
        print(f"Saved {docx.name}")
        if SOFFICE.exists():
            subprocess.run([str(SOFFICE), "--headless",
                            f"-env:UserInstallation=file:///{(REPO / 'outputs' / '.lo_guide').as_posix()}",
                            "--convert-to", "pdf", "--outdir", str(md_path.parent), str(docx)],
                           check=True, timeout=180, capture_output=True)
            print(f"Saved {pdf.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="3108a")
    ap.add_argument("--source", choices=["ai", "human"], default="ai")
    args = ap.parse_args()

    path = (AI_DIR / f"{args.id}.txt") if args.source == "ai" else (CORPUS_TXT / f"{args.id}.txt")
    if not path.exists():
        raise SystemExit(f"Essay not found: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")

    guide_path = GUIDES / f"{args.id}_{args.source}.json"
    if not guide_path.exists():
        raise SystemExit(f"No guide JSON at {guide_path}; run generate_questions.py first.")
    guide = json.loads(guide_path.read_text(encoding="utf-8"))

    print("Scoring submission with the trained detector ...", flush=True)
    det = detect(text)
    print(f"  prob(AI) = {det['prob_ai']}", flush=True)

    questions = [q["question"] for c in guide["claims"] for q in c["questions"]]
    print(f"Tagging {len(questions)} questions with the trained Bloom classifier ...", flush=True)
    tags = dict(zip(questions, bloom_tag(questions)))

    md = build_markdown(args.id, args.source, det, guide, tags)
    out_md = GUIDES / f"{args.id}_{args.source}_guide.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Saved {out_md.relative_to(REPO)}")
    render(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
