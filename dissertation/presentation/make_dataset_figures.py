"""Generate dataset rationale figures into dissertation/figures/.

Each figure is tied to one design decision, so the "why" is visible:
  1 group_balance          groups are balanced
  2 discipline_imbalance   disciplines are not (long tail)        -> stratify by group
  3 l1_availability        non-native scarce and uneven           -> oversample, AH binds
  4 student_clustering     authorship is clustered                -> split by student, cap 4
  5 sample_composition     the final 8 balanced cells, by split   -> the design
  6 sample_length_by_l1    length similar across L1 in the sample -> length not a proxy

Reads data/interim/bawe_clean.csv (full cleaned corpus) and
data/processed/bawe_human_sample.csv (the drawn sample).

    python dissertation/presentation/make_dataset_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CLEAN = REPO / "data" / "interim" / "bawe_clean.csv"
SAMPLE = REPO / "data" / "processed" / "bawe_human_sample.csv"
FIGS = REPO / "dissertation" / "figures"

GROUPS = ["AH", "LS", "PS", "SS"]
GLABEL = {"AH": "Arts and\nHumanities", "LS": "Life\nSciences",
          "PS": "Physical\nSciences", "SS": "Social\nSciences"}
GCOLOR = {"AH": "#2b6777", "LS": "#4e9d6c", "PS": "#d98e3b", "SS": "#7d6b9e"}

NATIVE_C = "#2b6777"
NONNAT_C = "#d98e3b"
SPLIT_C = {"train": "#2b6777", "val": "#7fb0bd", "test": "#d98e3b"}
INK = "#222831"
GREY = "#52616b"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, color="#e6ebed", zorder=0)
    ax.set_axisbelow(True)


def labelbars(ax, bars, fmt="{:.0f}", dy=2, fs=9):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs, color=INK)


def fig_group_balance(clean):
    counts = clean["disciplinary_group"].value_counts().reindex(GROUPS)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar([GLABEL[g] for g in GROUPS], counts.values,
                  color=[GCOLOR[g] for g in GROUPS], zorder=3, width=0.6)
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v}\n({100*v/total:.0f}%)",
                ha="center", va="bottom", fontsize=9.5, color=INK)
    style(ax)
    ax.set_ylabel("Essays in corpus")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.set_title("Disciplinary groups are balanced", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_group_balance.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_discipline_imbalance(clean):
    vc = clean["discipline"].value_counts().sort_values()
    disc_group = clean.groupby("discipline")["disciplinary_group"].agg(
        lambda s: s.mode().iat[0])
    colors = [GCOLOR.get(disc_group.get(d, ""), "#999999") for d in vc.index]
    fig, ax = plt.subplots(figsize=(8.5, 9))
    ax.barh(range(len(vc)), vc.values, color=colors, zorder=3)
    ax.set_yticks(range(len(vc)))
    ax.set_yticklabels(vc.index, fontsize=8)
    for i, v in enumerate(vc.values):
        ax.text(v + 2, i, str(v), va="center", fontsize=7.5, color=GREY)
    style(ax)
    ax.xaxis.grid(True, color="#e6ebed")
    ax.yaxis.grid(False)
    ax.set_xlabel("Essays in corpus")
    ax.set_xlim(0, max(vc.values) * 1.08)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GCOLOR[g]) for g in GROUPS]
    ax.legend(handles, [g for g in GROUPS], title="group", loc="lower right", frameon=False)
    ax.set_title("Individual disciplines are uneven (long tail)",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_discipline_imbalance.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_l1_availability(clean):
    nat = clean[clean["native"]].groupby("disciplinary_group").size().reindex(GROUPS)
    non = clean[~clean["native"]].groupby("disciplinary_group").size().reindex(GROUPS)
    x = np.arange(len(GROUPS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w / 2, nat.values, w, label="native", color=NATIVE_C, zorder=3)
    b2 = ax.bar(x + w / 2, non.values, w, label="non-native", color=NONNAT_C, zorder=3)
    labelbars(ax, b1, dy=6)
    labelbars(ax, b2, dy=6)
    ax.axhline(80, color="#c0392b", linestyle="--", linewidth=1.5, zorder=4)
    ax.text(len(GROUPS) - 0.5, 92, "sampled: 80 per cell", color="#c0392b",
            ha="right", fontsize=9, fontweight="bold")
    ax.annotate("binding constraint\n(AH non-native = 114)", xy=(0 + w / 2, 114),
                xytext=(0.7, 300), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_xticks(x)
    ax.set_xticklabels([g for g in GROUPS])
    style(ax)
    ax.set_ylabel("Available essays in corpus")
    ax.set_ylim(0, max(nat.values) * 1.15)
    ax.legend(frameon=False)
    ax.set_title("Non-native writers are scarce and unevenly spread",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_l1_availability.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_student_clustering(clean):
    sizes = clean.groupby("student_id").size()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bins = np.arange(1, sizes.max() + 2) - 0.5
    ax.hist(sizes.values, bins=bins, color=NATIVE_C, zorder=3, edgecolor="white")
    ax.axvline(sizes.mean(), color="#c0392b", linestyle="--", linewidth=1.5)
    ax.text(sizes.mean() + 0.3, ax.get_ylim()[1] * 0.9, f"mean {sizes.mean():.1f}",
            color="#c0392b", fontsize=9)
    ax.axvline(4, color="#2a7", linestyle=":", linewidth=1.8)
    ax.text(4 + 0.3, ax.get_ylim()[1] * 0.78, "per-student cap = 4", color="#1c7a4a", fontsize=9)
    txt = (f"{sizes.shape[0]} students\nmean {sizes.mean():.1f}, max {int(sizes.max())}\n"
           f"{100*(sizes>1).mean():.0f}% write more than one")
    ax.text(0.62, 0.62, txt, transform=ax.transAxes, fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round", fc="#f3f5f6", ec="#cdd6da"))
    style(ax)
    ax.set_xlabel("Essays per student (full corpus)")
    ax.set_ylabel("Number of students")
    ax.set_title("Authorship is clustered, so we split by student and cap per cell",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_student_clustering.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_sample_composition(sample):
    cells = [f"{g}_{s}" for g in GROUPS for s in ["native", "non-native"]]
    ct = pd.crosstab(sample["cell"], sample["split"]).reindex(cells)
    order = [s for s in ["train", "val", "test"] if s in ct.columns]
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(cells))
    for s in order:
        vals = ct[s].values
        ax.bar(range(len(cells)), vals, bottom=bottom, label=s,
               color=SPLIT_C[s], zorder=3, width=0.7)
        bottom += vals
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cells], fontsize=9)
    for i, tot in enumerate(bottom):
        ax.text(i, tot + 1, f"{int(tot)}", ha="center", fontsize=9, color=INK)
    style(ax)
    ax.set_ylabel("Essays")
    ax.set_ylim(0, 92)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.set_title("Final sample: 80 per cell, balanced across groups and first language",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_sample_composition.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_sample_length_by_l1(sample):
    nat = sample.loc[sample["native"], "words"]
    non = sample.loc[~sample["native"], "words"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bins = np.linspace(0, 6000, 31)
    ax.hist(nat, bins=bins, color=NATIVE_C, alpha=0.55, label=f"native (mean {nat.mean():.0f})", zorder=3)
    ax.hist(non, bins=bins, color=NONNAT_C, alpha=0.55, label=f"non-native (mean {non.mean():.0f})", zorder=3)
    ax.axvline(nat.mean(), color=NATIVE_C, linestyle="--", linewidth=1.4)
    ax.axvline(non.mean(), color=NONNAT_C, linestyle="--", linewidth=1.4)
    style(ax)
    ax.set_xlabel("Essay length (words)")
    ax.set_ylabel("Number of essays")
    ax.legend(frameon=False)
    ax.set_title("In the sample, length is similar for native and non-native writers",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_sample_length_by_l1.png", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    if not CLEAN.exists():
        raise SystemExit(f"Missing {CLEAN}. Run clean_bawe.py first.")
    if not SAMPLE.exists():
        raise SystemExit(f"Missing {SAMPLE}. Run build_sample.py first.")
    FIGS.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(CLEAN)
    clean["native"] = clean["native"].astype(bool)
    sample = pd.read_csv(SAMPLE)
    sample["native"] = sample["native"].astype(bool)

    fig_group_balance(clean)
    fig_discipline_imbalance(clean)
    fig_l1_availability(clean)
    fig_student_clustering(clean)
    fig_sample_composition(sample)
    fig_sample_length_by_l1(sample)
    print(f"Saved 6 figures to {FIGS}")


if __name__ == "__main__":
    main()
