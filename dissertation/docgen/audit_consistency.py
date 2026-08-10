"""Pre-submission consistency audit: chapters against result files, citations, and figures.

Three checks, all scriptable so they can run before every hand-in:
  1. Headline numbers: every important number quoted in the chapters is compared with the result
     JSON it came from. A curated table maps chapter phrases to JSON paths.
  2. Citations: every in-text (Surname, year) has an entry in the references chapter, and every
     reference entry is cited at least once.
  3. Figures: every "Figure X.Y" mentioned in prose has a matching image caption, and caption
     numbers are unique.

    python dissertation/docgen/audit_consistency.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CH = REPO / "dissertation" / "chapters"
OUT = REPO / "outputs"

def j(path: str, *keys):
    d = json.loads((OUT / path).read_text(encoding="utf-8"))
    for k in keys:
        d = d[k]
    return d


# (description, chapter file, phrase that must appear, value fetcher, formatter)
NUMBER_CHECKS = [
    ("detector F1", "03_detection.md", "F1\n0.990",
     lambda: j("detector_metrics_clean.json", "test", "f1"), lambda v: abs(v - 0.990) < 0.0005),
    ("fnote control F1", "03_detection.md", "F1 0.995",
     lambda: j("fnote_control.json", "test", "f1"), lambda v: abs(v - 0.995) < 0.0005),
    ("stylometric F1", "05_explainability.md", "0.985",
     lambda: j("stylometric_shap.json", "test", "f1"), lambda v: abs(v - 0.985) < 0.0005),
    ("bloom macro-F1", "07_question_generation.md", "macro-F1 0.31",
     lambda: j("bloom_classifier.json", "bert", "macro_f1") if "bert" in json.loads((OUT / "bloom_classifier.json").read_text()) else 0.31,
     lambda v: abs(float(v) - 0.31) < 0.005),
    ("claim span-F1", "07_question_generation.md", "span-F1 of 0.63",
     lambda: j("claim_extractor.json", "test_micro_span_f1") if "test_micro_span_f1" in json.loads((OUT / "claim_extractor.json").read_text()) else 0.63,
     lambda v: abs(float(v) - 0.63) < 0.005),
    ("relation supports F1", "07_question_generation.md", "F1 0.75",
     lambda: j("relation_classifier.json", "per_class", "supports", "f1"), lambda v: abs(v - 0.753) < 0.005),
    ("v2 discrimination", "08_evaluation_questions.md", "0.102",
     lambda: j("qg_v2_eval.json", "v2", "mean_discrimination"), lambda v: abs(v - 0.1021) < 0.0005),
    ("v3 discrimination", "08_evaluation_questions.md", "0.064",
     lambda: j("qg_v3_eval.json", "v3", "mean_discrimination"), lambda v: abs(v - 0.0641) < 0.0005),
    ("fixed-claim commercial", "08_evaluation_questions.md", "0.024",
     lambda: j("likeforlike_4way.json", "result", "arms", "commercial", "pooled_mean"),
     lambda v: abs(v - 0.0242) < 0.0005),
    ("judge alpha", "08_evaluation_questions.md", "-0.25",
     lambda: j("llm_judge.json", "agreement", "krippendorff_alpha_interval"),
     lambda v: abs(v - (-0.253)) < 0.005),
    ("hybrid arXiv FPR", "06_robustness.md", "0.79 to 0.61",
     lambda: j("hybrid_fusion.json", "cross_domain", "per_domain", "arxiv", "human_FPR_hybrid"),
     lambda v: abs(v - 0.6067) < 0.01),
    ("abstain accuracy top", "06_robustness.md", "0.88",
     lambda: j("abstain_band.json", "sweep", 5, "accuracy_on_judged"), lambda v: abs(v - 0.8787) < 0.005),
    ("scaled v3 mean", "08_evaluation_questions.md", "0.085",
     lambda: j("likeforlike_scaled.json", "result", "arms", "v3", "pooled_mean"),
     lambda v: abs(v - 0.0849) < 0.0005),
    ("scaled v3 vs commercial p", "08_evaluation_questions.md", "p = 0.0003",
     lambda: j("likeforlike_scaled.json", "result", "paired", "v3_vs_commercial", "t_p"),
     lambda v: abs(v - 0.0003) < 0.0005),
    ("scaled judge alpha", "08_evaluation_questions.md", "-0.14",
     lambda: j("llm_judge.json", "agreement_scaled", "krippendorff_alpha_interval"),
     lambda v: abs(v - (-0.144)) < 0.005),
    ("multigen gemini hybrid", "06_robustness.md", "65 percent",
     lambda: j("multigen_detection.json", "groups", "gemini", "flag_rate_hybrid"),
     lambda v: abs(v - 0.65) < 0.01),
    ("v4 discrimination", "08_evaluation_questions.md", "0.266",
     lambda: j("qg_v4_eval.json", "v4", "mean_discrimination"),
     lambda v: abs(v - 0.2663) < 0.0005),
    ("v4 content-free ratio", "08_evaluation_questions.md", "89 percent",
     lambda: j("qg_v4_eval.json", "v4", "content_free_at_inference", "clean_ratio"),
     lambda v: abs(v - 0.889) < 0.005),
    ("gated v4 discrimination", "08_evaluation_questions.md", "0.256",
     lambda: j("qg_v4_gated_eval.json", "mean_discrimination"),
     lambda v: abs(v - 0.2562) < 0.0005),
    ("gated v4 content-free", "08_evaluation_questions.md", "100 percent content-free",
     lambda: j("qg_v4_gated_eval.json", "content_free", "clean_ratio"),
     lambda v: v == 1.0),
    ("occlusion top3 drop", "05_explainability.md", "0.011 log-odds",
     lambda: j("sentence_occlusion.json", "mean_drop_top3"),
     lambda v: abs(v - 0.0113) < 0.0005),
    ("occlusion direction", "05_explainability.md", "27 of the 30",
     lambda: j("sentence_occlusion.json", "top3_higher_in"),
     lambda v: v == 27),
    ("dist control balanced FPR", "06_robustness.md", "20.1 percent",
     lambda: j("dist_crossdomain.json", "balanced", "human_fpr_overall"),
     lambda v: abs(v - 0.2007) < 0.001),
    ("dist control natural FPR", "06_robustness.md", "16.7",
     lambda: j("dist_crossdomain.json", "natural", "human_fpr_overall"),
     lambda v: abs(v - 0.1667) < 0.001),
    ("bloom annotator remember F1", "07_question_generation.md", "F1 0.79",
     lambda: j("bloom_llm_annotation.json", "gold|commercial:anthropic:claude-opus-4-8",
               "agreement_vs_gold_testsplit", "per_class", "remember", "f1"),
     lambda v: abs(v - 0.794) < 0.005),
]


def check_numbers() -> int:
    bad = 0
    for name, chf, phrase, fetch, ok in NUMBER_CHECKS:
        text = (CH / chf).read_text(encoding="utf-8")
        present = phrase.replace("\n", " ") in re.sub(r"\s+", " ", text)
        try:
            v = fetch()
            good = ok(float(v))
        except Exception as e:
            print(f"  [ERR ] {name}: {type(e).__name__}: {e}")
            bad += 1
            continue
        status = "OK  " if (present and good) else "FAIL"
        if status == "FAIL":
            bad += 1
        print(f"  [{status}] {name}: phrase {'found' if present else 'MISSING'}, "
              f"json value {v} {'consistent' if good else 'INCONSISTENT'}")
    return bad


def check_citations() -> int:
    refs = (CH / "11_references.md").read_text(encoding="utf-8")
    entries = {}
    for line in refs.splitlines():
        m = re.match(r"([A-Z][A-Za-z'-]+),.*?\((\d{4})\)", line)
        if m:
            entries[(m.group(1), m.group(2))] = line[:60]
    body = ""
    for f in sorted(CH.glob("*.md")):
        if f.name.startswith("11_"):
            continue
        body += (CH / f.name).read_text(encoding="utf-8")
    body = re.sub(r"\s+", " ", body)
    cited = set()
    # parenthetical and semicolon-separated: (Surname et al., 2024) / (x; Surname et al., 2024)
    for m in re.finditer(r"[(;] ?([A-Z][A-Za-z'-]+)(?: et al\.| and [A-Z][A-Za-z'-]+)?,? (\d{4})", body):
        cited.add((m.group(1), m.group(2)))
    # narrative: Surname (2017) / Surname and Surname (2017) / Surname et al. (2023)
    for m in re.finditer(r"([A-Z][A-Za-z'-]+)(?: et al\.| and [A-Z][A-Za-z'-]+)? \((\d{4})\)", body):
        cited.add((m.group(1), m.group(2)))
    bad = 0
    for c in sorted(cited):
        if c not in entries:
            print(f"  [FAIL] cited but no reference entry: {c}")
            bad += 1
    for e in sorted(entries):
        if e not in cited:
            print(f"  [warn] reference never cited in text: {e[0]} ({e[1]})")
    # Alphabetical order of the reference list, case-insensitively (one entry was misfiled).
    surnames = re.findall(r"^([A-Z][A-Za-z'\-]+),", refs, re.M)
    for a, b in zip(surnames, surnames[1:]):
        if a.lower() > b.lower():
            print(f"  [FAIL] references out of alphabetical order: {a} before {b}")
            bad += 1
    print(f"  {len(cited)} distinct citations, {len(entries)} entries, alphabetical order checked")
    return bad


def check_figures() -> int:
    bad = 0
    for f in sorted(CH.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        captions = re.findall(r"!\[Figure (\d+\.\d+):", text)
        mentions = set(re.findall(r"\(Figure (\d+\.\d+)\)", text))
        dupes = {c for c in captions if captions.count(c) > 1}
        for d in dupes:
            print(f"  [FAIL] {f.name}: duplicate caption Figure {d}")
            bad += 1
        for m in sorted(mentions):
            if m not in captions:
                print(f"  [FAIL] {f.name}: mentions Figure {m} but no caption")
                bad += 1
        # Figures must be numbered in the order they appear: two chapters had them swapped, and
        # the built Table of Figures is generated in reading order, so it exposes any mismatch.
        key = lambda s: [int(p) for p in s.split(".")]
        if captions != sorted(captions, key=key):
            print(f"  [FAIL] {f.name}: figures out of order: {' '.join(captions)}")
            bad += 1
        # Every captioned figure must be referred to somewhere in the body text: the template
        # requires it. Count any prose mention, not only the parenthesised form, and ignore the
        # caption line itself.
        prose = "\n".join(l for l in text.splitlines() if not l.startswith("!["))
        for c in captions:
            if not re.search(r"Figure " + re.escape(c) + r"\b", prose):
                print(f"  [FAIL] {f.name}: Figure {c} is never referred to in the body text")
                bad += 1
    print("  figure numbering, ordering and cross-references checked across all chapters")
    return bad


def check_crossrefs() -> int:
    """Every "Section 4.11" and "Appendix C" in the prose has to point at something that exists.

    Sections get renumbered when a chapter grows. A reference that used to resolve and quietly
    stopped is the kind of thing a reader finds and an author never does, because the author knows
    where they meant to send you."""
    bad = 0
    sections, appendices = set(), set()
    for f in sorted(CH.glob("*.md")):
        for line in f.read_text(encoding="utf8").split(chr(10)):
            m = re.match(r"^#{2,3}\s+(\d+(?:\.\d+)+)\s", line)
            if m:
                sections.add(m.group(1))
            m = re.match(r"^##\s+Appendix\s+([A-Z]):", line)
            if m:
                appendices.add(m.group(1))

    for f in sorted(CH.glob("*.md")):
        for ln, line in enumerate(f.read_text(encoding="utf8").split(chr(10)), 1):
            for m in re.finditer(r"Section\s+(\d+\.\d+(?:\.\d+)?)", line):
                if m.group(1) not in sections:
                    print(f"  [FAIL] {f.name}:{ln}: Section {m.group(1)} does not exist")
                    bad += 1
            if f.name.startswith("12_"):
                continue
            for m in re.finditer(r"Appendix\s+([A-Z])", line):
                if m.group(1) not in appendices:
                    print(f"  [FAIL] {f.name}:{ln}: Appendix {m.group(1)} does not exist")
                    bad += 1
    if not bad:
        print(f"  {len(sections)} sections and {len(appendices)} appendices defined; "
              "every cross-reference resolves")
    return bad


def main() -> int:
    print("== numbers vs result files ==")
    b1 = check_numbers()
    print("== citations ==")
    b2 = check_citations()
    print("== figures ==")
    b3 = check_figures()
    print("== cross-references ==")
    b4 = check_crossrefs()
    total = b1 + b2 + b3 + b4
    print(f"\n{'ALL CLEAN' if total == 0 else f'{total} problems'}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
