"""Commercial-vs-local question-generation comparison (pipeline component 4, a core contribution).

For one flagged essay, generate a Verification Interview Guide with each backend, then score every
backend's questions on the SAME judge-free discrimination measure (discrimination_sim.py). The only
thing that varies is which model wrote the questions, so the gap is attributable to the generator.

Reports, per backend: mean discrimination with bootstrap 95% CIs, the Bloom-level mix, and a
provenance check (every claim cites real sentence numbers, guaranteed by construction). Backends
that cannot start (for example a commercial provider with no API key) are skipped and reported, so
this runs locally today and gains the commercial arm the moment a key is in .env.

  python src/evaluation/compare_backends.py --id 3108a --source ai --commercial gemini
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
FIGS = REPO / "dissertation" / "figures"
OUTDIR = REPO / "outputs"
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "evaluation"))
from generate_questions import OllamaBackend, build_guide  # noqa: E402
from discrimination_sim import discrimination, boot_ci  # noqa: E402


def score_guide(guide: dict) -> dict:
    """Score a guide's claim-grounded questions on the discrimination measure."""
    discs, blooms = [], []
    for c in guide["claims"]:
        src = " ".join(s["text"] for s in c["source_sentences"])
        for q in c["questions"]:
            d = discrimination(q["question"], src)
            discs.append(d["discrimination"])
            blooms.append(q["bloom_level"])
            print(f"    disc {d['discrimination']:+.3f}  {q['question'][:64]}", flush=True)
    mean, ci = boot_ci(discs) if discs else (0.0, [0.0, 0.0])
    return {"n_claims": len(guide["claims"]), "n_questions": len(discs),
            "mean_discrimination": mean, "ci95": ci,
            "bloom_mix": dict(Counter(blooms)),
            "provenance_ok": all(c["source_sentences"] for c in guide["claims"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="3108a")
    ap.add_argument("--source", choices=["ai", "human"], default="ai")
    ap.add_argument("--claims", type=int, default=4)
    ap.add_argument("--local-model", default="llama3.1:8b")
    ap.add_argument("--commercial", default="gemini",
                    help="commercial provider to compare (gemini|anthropic|openai), or 'none'")
    ap.add_argument("--commercial-model", default=None)
    args = ap.parse_args()

    path = (AI_DIR / f"{args.id}.txt") if args.source == "ai" else (CORPUS_TXT / f"{args.id}.txt")
    if not path.exists():
        raise SystemExit(f"Essay not found: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Assemble the backends to compare. Local always; commercial only if it can start.
    backends, skipped = [("local", OllamaBackend(args.local_model))], {}
    if args.commercial.lower() != "none":
        from commercial_backend import make_commercial_backend
        try:
            backends.append(("commercial", make_commercial_backend(args.commercial, args.commercial_model)))
        except RuntimeError as e:
            skipped[args.commercial] = str(e)
            print(f"[skip] commercial backend '{args.commercial}' unavailable: {e}\n", flush=True)

    results = {}
    for label, backend in backends:
        print(f"== {label} ({backend.name}): generating guide ==", flush=True)
        guide = build_guide(args.id, args.source, text, backend, args.claims)
        (OUTDIR / "verification_guides").mkdir(parents=True, exist_ok=True)
        (OUTDIR / "verification_guides" / f"{args.id}_{args.source}_{label}.json").write_text(
            json.dumps(guide, indent=2), encoding="utf-8")
        print(f"== {label}: scoring discrimination ==", flush=True)
        results[label] = {"backend": backend.name, **score_guide(guide)}

    report = {"essay_id": args.id, "source": args.source, "results": results,
              "skipped": skipped,
              "reading": "Same essay, same discrimination scorer; the gap reflects which model "
                         "generated the questions. Higher mean discrimination is better."}
    out = OUTDIR / f"backend_comparison_{args.id}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_figure(results, args.id)

    print("\n=== COMMERCIAL vs LOCAL ===")
    for label, r in results.items():
        print(f"{label:11s} {r['backend']:32s} mean disc {r['mean_discrimination']:.3f} "
              f"CI {r['ci95']}  (n={r['n_questions']})")
    if skipped:
        print(f"skipped: {', '.join(skipped)} (no key/SDK)")
    print(f"Saved {out.relative_to(REPO)}")
    return 0


def make_figure(results: dict, essay_id: str) -> None:
    if not results:
        return
    FIGS.mkdir(parents=True, exist_ok=True)
    labels = list(results)
    means = [results[k]["mean_discrimination"] for k in labels]
    cis = [results[k]["ci95"] for k in labels]
    err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
    colors = {"local": "#2b6777", "commercial": "#d98e3b"}
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(labels, means, yerr=err, capsize=6, width=0.55,
           color=[colors.get(k, "#888") for k in labels], error_kw={"ecolor": "#222831"})
    # Context: the generic-baseline mean from the earlier single-backend run, if present.
    sim = OUTDIR / "discrimination_sim.json"
    if sim.exists():
        g = json.loads(sim.read_text(encoding="utf-8")).get("generic", {}).get("mean_discrimination")
        if g is not None:
            ax.axhline(g, color="#a63d2e", ls="--", lw=1.2, label=f"generic-question baseline ({g:.2f})")
            ax.legend(fontsize=9, frameon=False)
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylabel("mean discrimination (aware - blind)")
    ax.set_title(f"Question generation: commercial vs local ({essay_id})",
                 fontsize=12, fontweight="bold", color="#222831")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_backend_comparison.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_backend_comparison.png")


if __name__ == "__main__":
    raise SystemExit(main())
