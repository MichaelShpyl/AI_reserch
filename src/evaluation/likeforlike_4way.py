"""Like-for-like 4-way question-generation comparison on a fixed claim set.

The scaled batch (compare_backends_batch.py) let each backend run its OWN full pipeline, so the
claim-extraction model differed between arms as well as the question writer. This experiment tightens
that: it fixes one claim set per essay (extracted once with the neutral local extractor, Llama 3.1
8B), then has every question-writing model answer the SAME claims. The only variable across the four
arms is the model that writes the questions, so the comparison is genuinely like-for-like.

Four arms, all scored by the same discrimination simulation:
  local8b     - Llama 3.1 8B via Ollama (the open baseline, no fine-tuning)
  commercial  - free-tier Gemini via API (the commercial reference)
  base3b      - base Qwen2.5 3B (the approved Backend B, before fine-tuning)
  ft3b        - QLoRA-fine-tuned Qwen2.5 3B (the approved Backend B, after fine-tuning)

8 GB VRAM, so models are never co-resident: claims and the Ollama arm run with Ollama up, then it is
freed; each HF 3B model loads in turn (freed between); Gemini is API-only; scoring reloads Ollama
last. The run saves after every essay in every phase and skips finished work, so it is resumable
across a rate-limit or a crash.

    python src/evaluation/likeforlike_4way.py --essays 14 --claims 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
BATCH = REPO / "outputs" / "backend_comparison_batch.json"
ADAPTER = REPO / "models" / "qg_finetune_qwen3b"
OUT = REPO / "outputs" / "likeforlike_4way.json"
FIGS = REPO / "dissertation" / "figures"
OLLAMA_EXE = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
sys.path.insert(0, str(REPO / "src" / "question_gen"))
sys.path.insert(0, str(REPO / "src" / "evaluation"))
sys.path.insert(0, str(REPO / "src" / "detection"))

ARMS = ["local8b", "commercial", "base3b", "ft3b"]
ARM_LABEL = {"local8b": "local 8B", "commercial": "commercial\nGemini",
             "base3b": "base 3B", "ft3b": "fine-tuned 3B"}


def free_ollama():
    for m in ("llama3.1:8b", "nomic-embed-text"):
        try:
            subprocess.run([str(OLLAMA_EXE), "stop", m], capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            print(f"  (ollama stop {m} timed out; continuing)", flush=True)
    time.sleep(3)


def load_state() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"essays": {}}


def save(state: dict):
    OUT.write_text(json.dumps(state, indent=2), encoding="utf-8")


def ensure_claims(state, ids, n_claims):
    """Phase A part 1: fixed claim set per essay, extracted once with Llama 8B."""
    from generate_questions import OllamaBackend, sentences, extract_claims
    from text_normalize import normalize_text
    be = None
    for eid in ids:
        e = state["essays"].setdefault(eid, {})
        if e.get("claims"):
            print(f"  claims {eid}: reuse ({len(e['claims'])})", flush=True)
            continue
        if be is None:
            be = OllamaBackend("llama3.1:8b")
        text = (AI_DIR / f"{eid}.txt").read_text(encoding="utf-8", errors="ignore")
        sents = sentences(normalize_text(text))
        claims = []
        for c in extract_claims(sents, be, n_claims):
            src = " ".join(s["text"] for s in c["source_sentences"])
            claims.append({"claim": c["claim"], "source": src})
        e["claims"] = claims
        e.setdefault("gen", {})
        e.setdefault("score", {})
        print(f"  claims {eid}: {len(claims)}", flush=True)
        save(state)


def generate_arm(state, ids, arm, make_backend):
    """Generate k questions per fixed claim for one arm. make_backend() is called lazily and only
    if some essay still needs this arm, so we do not load a model we do not need."""
    from generate_questions import questions_for_claim
    backend = None
    for eid in ids:
        e = state["essays"][eid]
        # Reuse only a NON-empty prior result. An all-empty entry means the arm failed on this essay
        # (e.g. a commercial rate-limit), so a later run retries it once the limit clears.
        if arm in e["gen"] and any(e["gen"][arm]):
            print(f"  gen[{arm}] {eid}: reuse", flush=True)
            continue
        if backend is None:
            backend = make_backend()
        per_claim = []
        for c in e["claims"]:
            try:
                qs = questions_for_claim(c["claim"], c["source"], backend, k=3)
            except Exception as ex:
                print(f"    {eid} claim gen failed: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
                qs = []
            per_claim.append(qs)
        e["gen"][arm] = per_claim
        n = sum(len(q) for q in per_claim)
        print(f"  gen[{arm}] {eid}: {n} questions", flush=True)
        save(state)
    return backend


def score_arm(state, ids, arm):
    """Phase E: score one arm's questions with the discrimination simulation (Ollama)."""
    from discrimination_sim import discrimination
    for eid in ids:
        e = state["essays"][eid]
        if arm in e["score"] and any(e["score"][arm]):
            print(f"  score[{arm}] {eid}: reuse", flush=True)
            continue
        if arm not in e["gen"]:
            continue
        per_claim = []
        for c, qs in zip(e["claims"], e["gen"][arm]):
            ds = []
            for q in qs:
                try:
                    ds.append(discrimination(q, c["source"])["discrimination"])
                except Exception as ex:
                    print(f"    {eid} score failed: {type(ex).__name__}", flush=True)
            per_claim.append(ds)
        e["score"][arm] = per_claim
        flat = [d for ds in per_claim for d in ds]
        m = round(float(np.mean(flat)), 4) if flat else None
        print(f"  score[{arm}] {eid}: n={len(flat)} mean={m}", flush=True)
        save(state)


def pooled(state, ids, arm):
    return [d for eid in ids for ds in state["essays"][eid]["score"].get(arm, []) for d in ds]


def per_essay_means(state, ids, arm):
    out = []
    for eid in ids:
        flat = [d for ds in state["essays"][eid]["score"].get(arm, []) for d in ds]
        out.append(float(np.mean(flat)) if flat else np.nan)
    return np.array(out)


def boot_ci(vals, rng, n=10000):
    vals = np.array(vals)
    if len(vals) == 0:
        return None, [None, None]
    ms = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return round(float(vals.mean()), 4), [round(float(np.percentile(ms, 2.5)), 4),
                                          round(float(np.percentile(ms, 97.5)), 4)]


def analyse(state, ids):
    from scipy import stats
    rng = np.random.default_rng(42)
    result = {"essays": ids, "n_essays": len(ids), "arms": {}}
    for arm in ARMS:
        p = pooled(state, ids, arm)
        m, ci = boot_ci(p, rng)
        covered = sum(1 for eid in ids
                      if any(ds for ds in state["essays"][eid]["score"].get(arm, [])))
        result["arms"][arm] = {"n_questions": len(p), "n_essays_covered": covered,
                               "pooled_mean": m, "pooled_ci95": ci}

    # Paired tests on per-essay means (same essays under both arms; drop essays missing either).
    def paired(a, b):
        ma, mb = per_essay_means(state, ids, a), per_essay_means(state, ids, b)
        ok = ~(np.isnan(ma) | np.isnan(mb))
        ma, mb = ma[ok], mb[ok]
        if len(ma) < 3:
            return None
        diff = mb - ma
        t = stats.ttest_rel(mb, ma)
        w = stats.wilcoxon(mb, ma) if np.any(diff != 0) else None
        dboot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)]
        return {"n_essays": int(len(ma)),
                "mean_diff_b_minus_a": round(float(diff.mean()), 4),
                "diff_ci95": [round(float(np.percentile(dboot, 2.5)), 4),
                              round(float(np.percentile(dboot, 97.5)), 4)],
                "t_p": round(float(t.pvalue), 4),
                "wilcoxon_p": round(float(w.pvalue), 4) if w is not None else None,
                "b_higher_in": int((diff > 0).sum())}

    result["paired"] = {
        "ft3b_vs_commercial": paired("commercial", "ft3b"),   # the research question
        "ft3b_vs_base3b": paired("base3b", "ft3b"),            # the fine-tune effect, at 14 essays
        "commercial_vs_local8b": paired("local8b", "commercial"),
        "ft3b_vs_local8b": paired("local8b", "ft3b"),
    }
    return result


def make_figure(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [a for a in ARMS if result["arms"][a]["pooled_mean"] is not None]
    means = [result["arms"][a]["pooled_mean"] for a in arms]
    cis = [result["arms"][a]["pooled_ci95"] for a in arms]
    err = [[m - c[0] for m, c in zip(means, cis)], [c[1] - m for m, c in zip(means, cis)]]
    colors = {"local8b": "#8a8a8a", "commercial": "#d98e3b",
              "base3b": "#9bb7bd", "ft3b": "#2b6777"}
    total = result["n_essays"]
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    bars = ax.bar([ARM_LABEL[a] for a in arms], means, yerr=err, capsize=6, width=0.6,
                  color=[colors[a] for a in arms], error_kw={"ecolor": "#222831"})
    # Honest per-bar coverage: question count and how many of the 14 essays the arm actually covers
    # (the commercial arm is short because free-tier quota ran out on some essays).
    for a, b, m, ci in zip(arms, bars, means, [result["arms"][x]["pooled_ci95"] for x in arms]):
        cov = result["arms"][a]["n_essays_covered"]
        nq = result["arms"][a]["n_questions"]
        tag = f"n={nq}\n{cov}/{total} essays" + ("*" if cov < total else "")
        ax.text(b.get_x() + b.get_width() / 2, ci[1] + 0.012, tag,
                ha="center", va="bottom", fontsize=8.5, color="#52616B")
    # Generic-question baseline for context (same essays if available).
    gb = REPO / "outputs" / "generic_baseline_batch.json"
    if gb.exists():
        g = json.loads(gb.read_text(encoding="utf-8")).get("pooled", {}).get("mean")
        if g is not None:
            ax.axhline(g, color="#a63d2e", ls="--", lw=1.2, label=f"generic baseline ({g:.2f})")
            ax.legend(fontsize=9, frameon=False, loc="center left")
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylim(top=max(0.32, max(ci[1] for ci in [result["arms"][x]["pooled_ci95"] for x in arms]) + 0.06))
    ax.set_ylabel("pooled mean discrimination (aware - blind)")
    short = [a for a in arms if result["arms"][a]["n_essays_covered"] < total]
    subtitle = ("\n(*short of 14 essays: free-tier quota)" if short else "")
    ax.set_title("Same fixed claims, same scorer: only the question writer changes" + subtitle,
                 fontsize=11, fontweight="bold", color="#222831")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_likeforlike_4way.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_likeforlike_4way.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--essays", type=int, default=14)
    ap.add_argument("--claims", type=int, default=3)
    ap.add_argument("--commercial", default="gemini")
    args = ap.parse_args()
    if not ADAPTER.exists():
        raise SystemExit(f"No fine-tuned adapter at {ADAPTER}; run finetune_qg.py first.")

    ids = (json.loads(BATCH.read_text(encoding="utf-8")).get("balanced", {}).get("essays")
           or ["3108a"])[:args.essays]
    state = load_state()
    print(f"== {len(ids)} essays, {args.claims} claims each ==", flush=True)

    # Phase A: fixed claims + the Ollama arm, both while Ollama is up.
    print("== Phase A: fixed claims + local 8B questions (Ollama) ==", flush=True)
    ensure_claims(state, ids, args.claims)
    from generate_questions import OllamaBackend
    generate_arm(state, ids, "local8b", lambda: OllamaBackend("llama3.1:8b"))
    free_ollama()

    # Phase B: base Qwen 3B.
    print("== Phase B: base Qwen 3B questions (HF) ==", flush=True)
    from hf_backend import HFBackend
    if any("base3b" not in state["essays"][e]["gen"] for e in ids):
        b = generate_arm(state, ids, "base3b", lambda: HFBackend(adapter=None))
        if b:
            b.close()

    # Phase C: fine-tuned Qwen 3B.
    print("== Phase C: fine-tuned Qwen 3B questions (HF) ==", flush=True)
    if any("ft3b" not in state["essays"][e]["gen"] for e in ids):
        f = generate_arm(state, ids, "ft3b", lambda: HFBackend(adapter=str(ADAPTER)))
        if f:
            f.close()

    # Phase D: commercial (Gemini API, no GPU).
    print("== Phase D: commercial Gemini questions (API) ==", flush=True)
    from commercial_backend import make_commercial_backend
    try:
        generate_arm(state, ids, "commercial", lambda: make_commercial_backend(args.commercial, None))
    except RuntimeError as e:
        print(f"[skip] commercial backend unavailable: {e}", flush=True)

    # Phase E: score every arm with the discrimination sim (Ollama).
    print("== Phase E: score all arms (Ollama discrimination sim) ==", flush=True)
    for arm in ARMS:
        score_arm(state, ids, arm)
    free_ollama()

    result = analyse(state, ids)
    state["result"] = result
    save(state)
    make_figure(result)

    print("\n=== LIKE-FOR-LIKE 4-WAY ===")
    for arm in ARMS:
        a = result["arms"][arm]
        print(f"{arm:11s} n={a['n_questions']:3d} pooled mean {a['pooled_mean']} CI {a['pooled_ci95']}")
    for name, pr in result["paired"].items():
        if pr:
            print(f"paired {name:24s} diff {pr['mean_diff_b_minus_a']:+.4f} "
                  f"CI {pr['diff_ci95']} t_p={pr['t_p']} wilcoxon_p={pr['wilcoxon_p']} "
                  f"(n={pr['n_essays']}, b_higher_in={pr['b_higher_in']})")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
