"""Scale the commercial-vs-local comparison across many essays (pipeline component 4).

The single-essay run (compare_backends.py) gives an n=1 signal. This runs the same procedure over a
sample of essays and pools the per-question discrimination scores, so the commercial-vs-local
comparison rests on real statistics rather than one essay.

For each sampled essay it builds a Verification Interview Guide with each backend and scores every
question on the SAME local discrimination model, so the only thing that varies is which model wrote
the questions. It saves after every essay and skips essays already done, so a long run is resumable
and never loses progress. Backend failures (for example a commercial rate-limit) are recorded per
essay and the run continues.

  python src/evaluation/compare_backends_batch.py --n 15 --claims 3 --commercial gemini
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
CORPUS_TXT = REPO / "data" / "raw" / "bawe" / "download" / "CORPUS_TXT"
FIGS = REPO / "dissertation" / "figures"
OUTDIR = REPO / "outputs"
OUT = OUTDIR / "backend_comparison_batch.json"
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "evaluation"))
from generate_questions import OllamaBackend, build_guide  # noqa: E402
from discrimination_sim import discrimination, boot_ci  # noqa: E402


def score_backend(guide: dict) -> tuple[list[float], list[str]]:
    """Raw per-question discrimination scores + Bloom levels for one guide (so essays can be pooled)."""
    discs, blooms = [], []
    for c in guide["claims"]:
        src = " ".join(s["text"] for s in c["source_sentences"])
        for q in c["questions"]:
            d = discrimination(q["question"], src)
            discs.append(d["discrimination"])
            blooms.append(q["bloom_level"])
    return discs, blooms


def essay_path(essay_id: str, source: str) -> Path:
    return (AI_DIR / f"{essay_id}.txt") if source == "ai" else (CORPUS_TXT / f"{essay_id}.txt")


def load_prior() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"essays": {}}


def run_essay(essay_id: str, source: str, backends: list, n_claims: int,
              prior: dict | None = None) -> dict:
    prior = prior or {}
    text = essay_path(essay_id, source).read_text(encoding="utf-8", errors="ignore")
    row = {}
    for label, backend in backends:
        if label in prior and "discs" in prior[label]:  # keep a backend that already succeeded
            row[label] = prior[label]
            print(f"    {label:11s} reuse (n={prior[label]['n_questions']})", flush=True)
            continue
        try:
            guide = build_guide(essay_id, source, text, backend, n_claims)
            discs, blooms = score_backend(guide)
            row[label] = {"backend": backend.name, "discs": discs, "blooms": blooms,
                          "n_questions": len(discs),
                          "mean": round(float(np.mean(discs)), 4) if discs else None,
                          "provenance_ok": all(c["source_sentences"] for c in guide["claims"])}
            m = row[label]["mean"]
            print(f"    {label:11s} n={len(discs):2d} mean={m}", flush=True)
        except Exception as e:  # rate-limit, model hiccup: record and keep going
            row[label] = {"backend": backend.name, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            print(f"    {label:11s} FAILED: {row[label]['error']}", flush=True)
    return row


def compute_balanced(essays: dict) -> dict | None:
    """The headline statistics: restrict to essays where every backend produced a non-empty scored
    guide, pool per-question scores per backend, and run PAIRED tests on the per-essay means (the
    essays are the same under both backends, so pairing is the fair test). This writes the
    `balanced` block the dissertation reports, so the numbers are reproducible from this script."""
    from scipy import stats as sstats
    present = [l for l in ("local", "commercial")
               if any(l in r and r[l].get("discs") for r in essays.values())]
    if len(present) < 2:
        return None
    both = [e for e, r in essays.items() if all(l in r and r[l].get("discs") for l in present)]
    excluded = [e for e, r in essays.items() if e not in both]
    if len(both) < 3:
        return None
    rng = np.random.default_rng(42)

    def boot(vals, n=5000):
        vals = np.array(vals)
        ms = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
        return round(float(vals.mean()), 4), [round(float(np.percentile(ms, 2.5)), 4),
                                              round(float(np.percentile(ms, 97.5)), 4)]

    out = {"essays": both, "excluded": excluded}
    for lbl in present:
        pooled = [x for e in both for x in essays[e][lbl]["discs"]]
        m, ci = boot(pooled)
        out[lbl] = {"n_essays": len(both), "n_questions": len(pooled),
                    "pooled_mean": m, "pooled_ci95": ci}
    lm = np.array([np.mean(essays[e]["local"]["discs"]) for e in both])
    cm = np.array([np.mean(essays[e]["commercial"]["discs"]) for e in both])
    diff = cm - lm
    dboots = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)]
    t, p = sstats.ttest_rel(cm, lm)
    w = sstats.wilcoxon(cm, lm)
    out["paired"] = {
        "mean_diff_commercial_minus_local": round(float(diff.mean()), 4),
        "diff_ci95": [round(float(np.percentile(dboots, 2.5)), 4),
                      round(float(np.percentile(dboots, 97.5)), 4)],
        "t_p": round(float(p), 4), "wilcoxon_p": round(float(w.pvalue), 4),
        "essays_commercial_higher": int((diff > 0).sum()),
    }
    return out


def aggregate(essays: dict) -> dict:
    agg = {}
    for label in ("local", "commercial"):
        pooled, blooms, per_essay = [], [], []
        for eid, row in essays.items():
            r = row.get(label)
            if not r or "discs" not in r or not r["discs"]:
                continue
            pooled.extend(r["discs"])
            blooms.extend(r["blooms"])
            per_essay.append(r["mean"])
        if pooled:
            mean, ci = boot_ci(pooled)
            agg[label] = {"n_essays": len(per_essay), "n_questions": len(pooled),
                          "pooled_mean": mean, "pooled_ci95": ci,
                          "per_essay_means": per_essay,
                          "bloom_mix": dict(Counter(blooms))}
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="number of essays to sample")
    ap.add_argument("--claims", type=int, default=3)
    ap.add_argument("--source", choices=["ai", "human"], default="ai")
    ap.add_argument("--commercial", default="gemini", help="gemini|anthropic|openai, or none")
    ap.add_argument("--commercial-model", default=None)
    ap.add_argument("--local-model", default="llama3.1:8b")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Deterministic sample of essay ids.
    all_ids = sorted(p.stem for p in AI_DIR.glob("*.txt"))
    rng = random.Random(args.seed)
    sample = rng.sample(all_ids, min(args.n, len(all_ids)))

    backends = [("local", OllamaBackend(args.local_model))]
    if args.commercial.lower() != "none":
        from commercial_backend import make_commercial_backend
        try:
            backends.append(("commercial", make_commercial_backend(args.commercial, args.commercial_model)))
        except RuntimeError as e:
            print(f"[skip] commercial backend unavailable: {e}", flush=True)

    data = load_prior()
    data.setdefault("essays", {})
    data["config"] = {"n": args.n, "claims": args.claims, "source": args.source,
                      "commercial": args.commercial, "seed": args.seed,
                      "backends": [b.name for _, b in backends]}

    for i, eid in enumerate(sample, 1):
        if eid in data["essays"] and all(lbl in data["essays"][eid] and "discs" in data["essays"][eid][lbl]
                                         for lbl, _ in backends):
            print(f"[{i}/{len(sample)}] {eid}: already done, skip", flush=True)
            continue
        print(f"[{i}/{len(sample)}] {eid}: running", flush=True)
        data["essays"][eid] = run_essay(eid, args.source, backends, args.claims,
                                        data["essays"].get(eid))
        data["aggregate"] = aggregate(data["essays"])
        OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")  # save after every essay

    balanced = compute_balanced(data["essays"])
    if balanced:
        data["balanced"] = balanced
        OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    make_figure(data)
    print("\n=== COMMERCIAL vs LOCAL ===")
    if balanced:
        for label in ("local", "commercial"):
            b = balanced[label]
            print(f"balanced {label:11s} essays={b['n_essays']:2d} q={b['n_questions']:3d} "
                  f"pooled mean {b['pooled_mean']:.3f} CI {b['pooled_ci95']}")
        print(f"paired: diff {balanced['paired']['mean_diff_commercial_minus_local']} "
              f"CI {balanced['paired']['diff_ci95']} t_p={balanced['paired']['t_p']} "
              f"wilcoxon_p={balanced['paired']['wilcoxon_p']}")
    else:
        for label, a in data.get("aggregate", {}).items():
            print(f"{label:11s} essays={a['n_essays']:2d} q={a['n_questions']:3d} "
                  f"pooled mean {a['pooled_mean']:.3f} CI {a['pooled_ci95']}")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(data: dict) -> None:
    # Balance first: the honest figure compares backends on the SAME essays, so restrict to essays
    # where every present backend produced a non-empty question list. (An earlier version plotted
    # unbalanced aggregates, which silently compared different essay sets.)
    present = [l for l in ("local", "commercial")
               if any(l in r and r[l].get("discs") for r in data.get("essays", {}).values())]
    balanced_ids = [e for e, r in data.get("essays", {}).items()
                    if all(l in r and r[l].get("discs") for l in present)]
    if present and balanced_ids:
        data = {"essays": {e: data["essays"][e] for e in balanced_ids}}
        data["aggregate"] = aggregate(data["essays"])
    agg = data.get("aggregate", {})
    if not agg:
        return
    FIGS.mkdir(parents=True, exist_ok=True)
    labels = [l for l in ("local", "commercial") if l in agg]
    colors = {"local": "#2b6777", "commercial": "#d98e3b"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: per-essay mean discrimination, PAIRED (a line per essay joins its two backends), since
    # the essays are the same under both and the headline test is paired.
    if len(labels) == 2:
        pairs = [(float(np.mean(r[labels[0]]["discs"])), float(np.mean(r[labels[1]]["discs"])))
                 for r in data["essays"].values()
                 if all(l in r and r[l].get("discs") for l in labels)]
        for a, b in pairs:
            ax1.plot([0, 1], [a, b], color="#bbb", lw=0.8, zorder=1)
        ax1.scatter([0] * len(pairs), [a for a, _ in pairs], color=colors[labels[0]], zorder=3)
        ax1.scatter([1] * len(pairs), [b for _, b in pairs], color=colors[labels[1]], zorder=3)
        ax1.set_xlim(-0.4, 1.4)
    else:
        for x, label in enumerate(labels):
            ys = agg[label]["per_essay_means"]
            ax1.scatter(np.random.default_rng(x + 1).normal(x, 0.05, len(ys)), ys,
                        color=colors[label], alpha=0.75)
    ax1.axhline(0, color="#888", lw=1, ls="--")
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels)
    ax1.set_ylabel("per-essay mean discrimination")
    ax1.set_title("Same essays, paired", fontsize=11, fontweight="bold", color="#222831")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Right: pooled mean with CI + generic baseline for context.
    means = [agg[l]["pooled_mean"] for l in labels]
    cis = [agg[l]["pooled_ci95"] for l in labels]
    err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
    ax2.bar(labels, means, yerr=err, capsize=6, width=0.55,
            color=[colors[l] for l in labels], error_kw={"ecolor": "#222831"})
    # Baseline for context: prefer the batch-level generic measurement (same essays as the bars);
    # fall back to the single-essay pilot value.
    g, src = None, ""
    gb = OUTDIR / "generic_baseline_batch.json"
    if gb.exists():
        pooled = json.loads(gb.read_text(encoding="utf-8")).get("pooled", {})
        if pooled.get("mean") is not None:
            g, src = pooled["mean"], f"generic baseline, same essays ({pooled['mean']:.2f})"
    if g is None:
        sim = OUTDIR / "discrimination_sim.json"
        if sim.exists():
            m = json.loads(sim.read_text(encoding="utf-8")).get("generic", {}).get("mean_discrimination")
            if m is not None:
                g, src = m, f"generic baseline, pilot essay ({m:.2f})"
    if g is not None:
        ax2.axhline(g, color="#a63d2e", ls="--", lw=1.2, label=src)
        ax2.legend(fontsize=9, frameon=False)
    ax2.axhline(0, color="#888", lw=1)
    ax2.set_ylabel("pooled mean discrimination")
    ax2.set_title("Pooled (95% CI)", fontsize=11, fontweight="bold", color="#222831")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    n_e = agg[labels[0]]["n_essays"]
    fig.suptitle(f"Question generation: commercial vs local across {n_e} essays",
                 fontsize=12.5, fontweight="bold", color="#222831")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "fig_backend_comparison_batch.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_backend_comparison_batch.png")


if __name__ == "__main__":
    raise SystemExit(main())
