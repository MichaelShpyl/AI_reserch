"""The scaled fixed-claim comparison: 30 essays, with the working v3 backend as an arm.

The 14-essay like-for-like run answered the design question (the commercial edge vanishes on fixed
claims) but Chapter 9 names its size as a limitation. This scales it to 30 essays and swaps the
artifact v1 arm for the backend the pipeline actually uses, v3. Same protocol throughout: one fixed
claim set per essay from the neutral extractor, every writer answers the same claims, one scorer,
resumable per essay per phase, VRAM-sequenced.

Contamination guards: new essays are drawn only from the pool that is neither an existing evaluation
essay nor one of the 307 essays v3's training data was distilled from. The 14 original essays' claims
and their local8b / base3b / commercial arms are reused verbatim from the 4-way state, so only the v3
arm and the 16 new essays cost fresh compute.

    python src/evaluation/likeforlike_scaled.py --essays 30
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AI_DIR = REPO / "data" / "processed" / "ai_essays"
OLD = REPO / "outputs" / "likeforlike_4way.json"
V3_TRAIN = REPO / "data" / "interim" / "qg_v3_pairs.json"
V3_ADAPTER = REPO / "models" / "qg_finetune_qwen3b_v3"
OUT = REPO / "outputs" / "likeforlike_scaled.json"
FIGS = REPO / "dissertation" / "figures"
OLLAMA_EXE = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
for p in ("question_gen", "evaluation", "detection"):
    sys.path.insert(0, str(REPO / "src" / p))

ARMS = ["local8b", "commercial", "base3b", "v3"]
ARM_LABEL = {"local8b": "local 8B", "commercial": "commercial\nGemini",
             "base3b": "base 3B", "v3": "fine-tuned 3B (v3)"}
SEED = 42


def free_ollama():
    for m in ("llama3.1:8b", "nomic-embed-text"):
        try:
            subprocess.run([str(OLLAMA_EXE), "stop", m], capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            pass
    time.sleep(3)


def load_state() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    # Seed the state from the 4-way run: claims plus the three reusable arms for the 14 essays.
    old = json.loads(OLD.read_text(encoding="utf-8"))
    state = {"essays": {}}
    for eid, e in old["essays"].items():
        state["essays"][eid] = {
            "claims": e["claims"],
            "gen": {a: e["gen"][a] for a in ("local8b", "commercial", "base3b") if a in e.get("gen", {})},
            "score": {a: e["score"][a] for a in ("local8b", "commercial", "base3b") if a in e.get("score", {})},
        }
    print(f"seeded from the 4-way state: {len(state['essays'])} essays reused", flush=True)
    return state


def save(state: dict):
    OUT.write_text(json.dumps(state), encoding="utf-8")


def pick_new_essays(existing: set[str], n_more: int) -> list[str]:
    v3_train = set(json.loads(V3_TRAIN.read_text(encoding="utf-8"))["done"])
    banned = existing | v3_train | {"3108a"}
    pool = sorted(p.stem for p in AI_DIR.glob("*.txt") if p.stem not in banned)
    rng = random.Random(SEED)
    picked = rng.sample(pool, n_more)
    print(f"pool after guards: {len(pool)}; picked {n_more} new essays", flush=True)
    return picked


def ensure_claims(state, ids, n_claims=3):
    from generate_questions import OllamaBackend, sentences, extract_claims
    from text_normalize import normalize_text
    be = None
    for eid in ids:
        e = state["essays"].setdefault(eid, {"claims": None, "gen": {}, "score": {}})
        if e.get("claims"):
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
        print(f"  claims {eid}: {len(claims)}", flush=True)
        save(state)


def generate_arm(state, ids, arm, make_backend):
    from generate_questions import questions_for_claim
    backend = None
    for eid in ids:
        e = state["essays"][eid]
        if arm in e["gen"] and any(e["gen"][arm]):
            continue
        if backend is None:
            backend = make_backend()
        per_claim = []
        for c in e["claims"]:
            try:
                qs = questions_for_claim(c["claim"], c["source"], backend, k=3)
            except Exception as ex:
                print(f"    {eid} gen failed: {type(ex).__name__}: {str(ex)[:100]}", flush=True)
                qs = []
            per_claim.append(qs)
        e["gen"][arm] = per_claim
        print(f"  gen[{arm}] {eid}: {sum(len(q) for q in per_claim)} questions", flush=True)
        save(state)
    return backend


def score_arm(state, ids, arm):
    from discrimination_sim import discrimination
    for eid in ids:
        e = state["essays"][eid]
        if (arm in e["score"] and any(e["score"][arm])) or arm not in e["gen"]:
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
        print(f"  score[{arm}] {eid}: n={len(flat)} "
              f"mean={round(float(np.mean(flat)), 4) if flat else None}", flush=True)
        save(state)


def analyse(state, ids):
    from scipy import stats
    from wellformed import is_degenerate
    rng = np.random.default_rng(SEED)

    def pooled(arm):
        return [d for eid in ids for ds in state["essays"][eid]["score"].get(arm, []) for d in ds]

    def per_essay(arm):
        out = []
        for eid in ids:
            flat = [d for ds in state["essays"][eid]["score"].get(arm, []) for d in ds]
            out.append(float(np.mean(flat)) if flat else np.nan)
        return np.array(out)

    def boot(vals, n=10000):
        vals = np.array(vals)
        if len(vals) == 0:
            return None, [None, None]
        ms = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
        return round(float(vals.mean()), 4), [round(float(np.percentile(ms, 2.5)), 4),
                                              round(float(np.percentile(ms, 97.5)), 4)]

    result = {"essays": ids, "n_essays": len(ids), "arms": {}}
    for arm in ARMS:
        p = pooled(arm)
        qs = [q for eid in ids for qs_ in state["essays"][eid]["gen"].get(arm, []) for q in qs_]
        deg = sum(1 for q in qs if is_degenerate(q))
        m, ci = boot(p)
        covered = sum(1 for eid in ids
                      if any(ds for ds in state["essays"][eid]["score"].get(arm, [])))
        result["arms"][arm] = {"n_questions": len(p), "n_essays_covered": covered,
                               "pct_degenerate": round(100 * deg / max(len(qs), 1), 1),
                               "pooled_mean": m, "pooled_ci95": ci}

    def paired(a, b):
        ma, mb = per_essay(a), per_essay(b)
        ok = ~(np.isnan(ma) | np.isnan(mb))
        ma, mb = ma[ok], mb[ok]
        if len(ma) < 3:
            return None
        diff = mb - ma
        t = stats.ttest_rel(mb, ma)
        w = stats.wilcoxon(mb, ma) if np.any(diff != 0) else None
        dboot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)]
        return {"n_essays": int(len(ma)), "mean_diff_b_minus_a": round(float(diff.mean()), 4),
                "diff_ci95": [round(float(np.percentile(dboot, 2.5)), 4),
                              round(float(np.percentile(dboot, 97.5)), 4)],
                "t_p": round(float(t.pvalue), 4),
                "wilcoxon_p": round(float(w.pvalue), 4) if w is not None else None,
                "b_higher_in": int((diff > 0).sum())}

    result["paired"] = {
        "commercial_vs_local8b": paired("local8b", "commercial"),
        "v3_vs_base3b": paired("base3b", "v3"),
        "v3_vs_commercial": paired("commercial", "v3"),
        "v3_vs_local8b": paired("local8b", "v3"),
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
    colors = {"local8b": "#8a8a8a", "commercial": "#d98e3b", "base3b": "#9bb7bd", "v3": "#2b6777"}
    total = result["n_essays"]
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    bars = ax.bar([ARM_LABEL[a] for a in arms], means, yerr=err, capsize=6, width=0.6,
                  color=[colors[a] for a in arms], error_kw={"ecolor": "#222831"})
    for a, b, ci in zip(arms, bars, cis):
        r = result["arms"][a]
        tag = f"n={r['n_questions']}\n{r['n_essays_covered']}/{total} essays"
        if r["pct_degenerate"]:
            tag += f"\n{r['pct_degenerate']:.0f}% degen"
        ax.text(b.get_x() + b.get_width() / 2, ci[1] + 0.008, tag, ha="center", va="bottom",
                fontsize=8.5, color="#52616B")
    gb = REPO / "outputs" / "generic_baseline_batch.json"
    if gb.exists():
        g = json.loads(gb.read_text(encoding="utf-8")).get("pooled", {}).get("mean")
        if g is not None:
            ax.axhline(g, color="#a63d2e", ls="--", lw=1.1, label=f"generic baseline ({g:.2f})")
            ax.legend(fontsize=9, frameon=False, loc="center left")
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylim(top=max(0.32, max(c[1] for c in cis) + 0.06))
    ax.set_ylabel("pooled mean discrimination (aware - blind)")
    ax.set_title(f"The fixed-claim comparison at scale: {total} essays,\n"
                 f"the working v3 backend against local, base and commercial",
                 fontsize=11.5, fontweight="bold", color="#222831")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_likeforlike_scaled.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_likeforlike_scaled.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--essays", type=int, default=30)
    ap.add_argument("--commercial", default="gemini")
    args = ap.parse_args()
    if not V3_ADAPTER.exists():
        raise SystemExit(f"No v3 adapter at {V3_ADAPTER}")

    state = load_state()
    existing = list(state["essays"].keys())
    ids = existing + pick_new_essays(set(existing), max(0, args.essays - len(existing)))
    ids = ids[:args.essays]
    print(f"== {len(ids)} essays ==", flush=True)

    print("== Phase A: claims + local 8B (Ollama) ==", flush=True)
    ensure_claims(state, ids)
    from generate_questions import OllamaBackend
    generate_arm(state, ids, "local8b", lambda: OllamaBackend("llama3.1:8b"))
    free_ollama()

    from hf_backend import HFBackend
    print("== Phase B: base Qwen 3B (HF) ==", flush=True)
    if any(not (("base3b" in state["essays"][e]["gen"]) and any(state["essays"][e]["gen"]["base3b"]))
           for e in ids):
        b = generate_arm(state, ids, "base3b", lambda: HFBackend(adapter=None))
        if b:
            b.close()

    print("== Phase C: v3 fine-tuned Qwen 3B (HF) ==", flush=True)
    if any(not (("v3" in state["essays"][e]["gen"]) and any(state["essays"][e]["gen"]["v3"]))
           for e in ids):
        f = generate_arm(state, ids, "v3", lambda: HFBackend(adapter=str(V3_ADAPTER)))
        if f:
            f.close()

    print("== Phase D: commercial Gemini (API) ==", flush=True)
    from commercial_backend import make_commercial_backend
    try:
        generate_arm(state, ids, "commercial", lambda: make_commercial_backend(args.commercial, None))
    except RuntimeError as e:
        print(f"[skip] commercial unavailable: {e}", flush=True)

    print("== Phase E: scoring (Ollama) ==", flush=True)
    for arm in ARMS:
        score_arm(state, ids, arm)
    free_ollama()

    result = analyse(state, ids)
    state["result"] = result
    save(state)
    make_figure(result)
    print("\n=== SCALED FIXED-CLAIM COMPARISON ===")
    for arm in ARMS:
        a = result["arms"][arm]
        print(f"{arm:11s} n={a['n_questions']:3d} cover {a['n_essays_covered']}/{result['n_essays']} "
              f"degen {a['pct_degenerate']:4.1f}% mean {a['pooled_mean']} CI {a['pooled_ci95']}")
    for name, pr in result["paired"].items():
        if pr:
            print(f"paired {name:22s} diff {pr['mean_diff_b_minus_a']:+.4f} CI {pr['diff_ci95']} "
                  f"t_p={pr['t_p']} (n={pr['n_essays']}, b_higher={pr['b_higher_in']})")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
