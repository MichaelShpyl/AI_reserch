"""Redraw the question-generation version comparison.

The previous version of this figure put its annotation at y = 0.30, which is exactly where the
generic-baseline line runs, so the dashed line struck the text out. This one puts the annotation
under the title, where nothing crosses it, and reads every number from outputs/qg_v4_eval.json
rather than carrying them in the code.

    python dissertation/presentation/make_qg_v4_figure.py

Writes dissertation/figures/fig_qg_v4.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "dissertation" / "figures"
DATA = ROOT / "outputs" / "qg_v4_eval.json"

INK, TEAL, RUST, GREY, PALE, AMBER = "#222831", "#2B6777", "#A63D2E", "#5B6770", "#AEC8CE", "#D98E3B"


def main():
    d = json.loads(DATA.read_text(encoding="utf8"))
    # The JSON labels are historical: the key "v4" holds the anchored, content-free run, and the
    # shipped v3 sits under reference.v3_mean. Read them by meaning, not by key name.
    bars = [
        ("base 3B", d["base"]["mean_discrimination"], GREY),
        ("v2: SQuAD", d["v2"]["mean_discrimination"], PALE),
        ("v3: verification\n(ships in the pipeline)", d["reference"]["v3_mean"], TEAL),
        ("v4: anchored,\ncontent-free", d["v4"]["mean_discrimination"], AMBER),
    ]
    baseline = d["reference"]["generic_baseline"]
    err = d["v4"].get("ci95_halfwidth", 0.0305)
    clean = d["v4"]["content_free_at_inference"]
    uniq = d["uniqueness"]

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=180)
    xs = range(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62, zorder=3)
    ax.errorbar([len(bars) - 1], [bars[-1][1]], yerr=[err], fmt="none",
                ecolor=INK, elinewidth=1.8, capsize=6, zorder=4)

    for i, (_, v, _) in enumerate(bars):
        # A label above the tallest bar would collide with the baseline line, so that one goes
        # inside the bar instead.
        inside = v + 0.03 > baseline
        ax.text(i, v - 0.014 if inside else v + 0.012, f"{v:.3f}",
                ha="center", va="top" if inside else "bottom",
                fontsize=13, fontweight="bold", color="white" if inside else INK, zorder=5)

    ax.axhline(baseline, color=RUST, linestyle="--", linewidth=1.8, zorder=2)
    ax.text(len(bars) - 0.5, baseline + 0.006, f"generic baseline {baseline:.2f}",
            ha="right", va="bottom", fontsize=12, color=RUST)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=12, color=INK)
    ax.set_ylabel(f"mean discrimination (fixed {d['n_claims']} claims)", fontsize=12, color=INK)
    ax.set_ylim(0, baseline * 1.20)
    ax.set_title("v4 closes most of the baseline gap by never naming the claim",
                 fontsize=15, fontweight="bold", color=INK, pad=30)
    # The annotation sits between the title and the top of the plot, clear of the baseline line.
    ax.text(0.5, 1.015,
            f"v4: {d['v4']['pct_degenerate']:.0f}% degenerate,  "
            f"{uniq['n_unique']}/{uniq['n_questions']} unique,  "
            f"{clean['clean_ratio'] * 100:.0f}% content-free at inference",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5, color=GREY)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=11)
    fig.tight_layout()

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig_qg_v4.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
