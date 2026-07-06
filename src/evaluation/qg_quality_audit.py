"""Audit the QUALITY of generated questions, not just their discrimination score.

The like-for-like study (likeforlike_4way.py) found the QLoRA-fine-tuned Qwen 3B scoring far higher
discrimination than every other model. Inspecting the actual questions shows why, and it is not good
news: the fine-tuned model overfit EduQG's multiple-choice format and collapsed into emitting
degenerate stems such as "Which of the following is correct?" with no options and no content. Those
contentless stems GAME the discrimination simulation: a source-aware and a source-blind answerer both
receive an empty question, their answers diverge at random, and the aware-minus-blind gap is large for
reasons that have nothing to do with question quality.

This script quantifies that. For each model arm it counts degenerate questions with a transparent
rule, and it compares the discrimination of degenerate stems against real questions, so the artifact
is measured rather than asserted. It is the evidence behind the corrected write-up in Chapters 7 and 8.

    python src/evaluation/qg_quality_audit.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs" / "likeforlike_4way.json"
OUT = REPO / "outputs" / "qg_quality_audit.json"
FIGS = REPO / "dissertation" / "figures"
ARMS = ["local8b", "commercial", "base3b", "ft3b"]
ARM_LABEL = {"local8b": "local 8B", "commercial": "commercial\nGemini",
             "base3b": "base 3B", "ft3b": "fine-tuned 3B"}

# Transparent degeneracy rule: multiple-choice stems (no options are ever supplied), raw JSON
# leakage, and contentless "which/what is correct" fragments. These are not usable verification
# questions: a verification question must ask the student to reconstruct their own reasoning.
DEGEN_PATTERNS = [
    r"which of the following",
    r"^\s*\{?\"?questions\"?\s*:",              # raw JSON leaked into the text
    r"^which (statement|option|answer|case)\b",
    r"^what is the (correct|right) answer",
    r"^(choose|select) the\b",
    r"^true or false",
]


def is_degenerate(q: str) -> bool:
    ql = q.lower().strip()
    if any(re.search(p, ql) for p in DEGEN_PATTERNS):
        return True
    words = re.findall(r"[a-z]+", ql)
    if len(words) <= 6 and any(w in ql for w in ("following", "correct", "statement")):
        return True
    return False


def collect(es, arm):
    out = []  # (question, discrimination)
    for e in es.values():
        gen = e.get("gen", {}).get(arm)
        sc = e.get("score", {}).get(arm)
        if not gen:
            continue
        for qs, ds in zip(gen, sc or []):
            for q, dd in zip(qs, ds):
                out.append((q, dd))
    return out


def main() -> int:
    es = json.loads(SRC.read_text(encoding="utf-8"))["essays"]
    report = {"source": SRC.name, "arms": {}}
    for arm in ARMS:
        qs = collect(es, arm)
        deg = [x for x in qs if is_degenerate(x[0])]
        clean = [x for x in qs if not is_degenerate(x[0])]
        report["arms"][arm] = {
            "n_questions": len(qs),
            "n_degenerate": len(deg),
            "pct_degenerate": round(100 * len(deg) / max(len(qs), 1), 1),
            "mean_disc_all": round(float(np.mean([x[1] for x in qs])), 4) if qs else None,
            "mean_disc_degenerate": round(float(np.mean([x[1] for x in deg])), 4) if deg else None,
            "mean_disc_clean": round(float(np.mean([x[1] for x in clean])), 4) if clean else None,
        }
    # The smoking gun: the literal contentless stem out-scores every real-question arm.
    empty = [d for arm in ARMS for (q, d) in collect(es, arm)
             if q.lower().strip().rstrip("?") in ("which of the following is correct",
                                                  "which of the following statements is correct")]
    report["contentless_stem"] = {
        "text": "Which of the following is correct?",
        "n_occurrences": len(empty),
        "mean_discrimination": round(float(np.mean(empty)), 4) if empty else None,
        "note": "A contentless string scoring above every model's real questions is direct evidence "
                "that the discrimination simulation can be gamed by degenerate questions.",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_figure(report)
    print("=== QG QUALITY AUDIT ===")
    for arm in ARMS:
        a = report["arms"][arm]
        print(f"{arm:11s} n={a['n_questions']:3d}  degenerate {a['pct_degenerate']:5.1f}%  "
              f"mean disc all={a['mean_disc_all']}")
    c = report["contentless_stem"]
    print(f"contentless stem '{c['text']}' x{c['n_occurrences']} -> mean disc {c['mean_discrimination']}")
    print(f"Saved {OUT.relative_to(REPO)}")
    return 0


def make_figure(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1, 1]})
    colors = {"local8b": "#8a8a8a", "commercial": "#d98e3b", "base3b": "#9bb7bd", "ft3b": "#2b6777"}

    # Left: share of degenerate questions per arm.
    pct = [report["arms"][a]["pct_degenerate"] for a in ARMS]
    ax1.bar([ARM_LABEL[a] for a in ARMS], pct, width=0.6, color=[colors[a] for a in ARMS])
    for i, p in enumerate(pct):
        ax1.text(i, p + 1.5, f"{p:.0f}%", ha="center", fontsize=10, color="#222831")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("degenerate / unusable questions (%)")
    ax1.set_title("The fine-tune collapsed into\nmultiple-choice stems", fontsize=11,
                  fontweight="bold", color="#222831")
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: mean discrimination of real questions per arm vs the contentless stem.
    means = [report["arms"][a]["mean_disc_all"] for a in ARMS]
    ax2.bar([ARM_LABEL[a] for a in ARMS], means, width=0.6, color=[colors[a] for a in ARMS])
    stem = report["contentless_stem"]["mean_discrimination"]
    ax2.axhline(stem, color="#a63d2e", ls="--", lw=1.4,
                label=f'"Which of the following\nis correct?" ({stem:.2f})')
    ax2.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax2.axhline(0, color="#888", lw=1)
    ax2.set_ylabel("mean discrimination")
    ax2.set_title("A contentless stem out-scores\nevery real question writer", fontsize=11,
                  fontweight="bold", color="#222831")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Why the fine-tuned model's discrimination score is an artifact",
                 fontsize=12.5, fontweight="bold", color="#222831")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_qg_quality_audit.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved fig_qg_quality_audit.png")


if __name__ == "__main__":
    raise SystemExit(main())
