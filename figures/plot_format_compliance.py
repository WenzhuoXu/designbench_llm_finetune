"""
Format Compliance Results Visualization.

Reads per-sample JSON from check_format_compliance.py --save-results
and produces a multi-panel breakdown figure.

Usage:
    python figures/plot_format_compliance.py \
        --results logs/compliance/warmstart_qwen3_14b_39083858_<JOBID>.json \
        --out figures/warmstart_format_compliance.png

Panels:
  A) Summary scorecard (compliance / think / answer / output-length)
  B) Action type distribution (bar chart)
  C) Output token length distribution (histogram)
  D) Per-sample status grid (50 tiles: green=TAG, amber=BARE, red=FAIL)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────────
BG    = "#FAFAF8"
DARK  = "#1C1C2E"
GREY  = "#6B7280"
LIGHT = "#E8E8E4"
WHITE = "#FFFFFF"

C_TAG   = "#10B981"   # green  — ✓ TAG
C_BARE  = "#F59E0B"   # amber  — ~ BARE
C_FAIL  = "#EF4444"   # red    — ✗ FAIL
C_THINK = "#3B82F6"   # blue   — think tag
C_ANS   = "#8B5CF6"   # purple — answer tag
C_LEN   = "#06B6D4"   # cyan   — output length

ACTION_COLORS = {
    "SCALE_PARAM":       "#3B82F6",
    "SCALE_MULTI_PARAM": "#6366F1",
    "ADD_MEMBER":        "#10B981",
    "MODIFY_PARAM":      "#F59E0B",
    "REMOVE_MEMBER":     "#EF4444",
    "MOVE_JOINT":        "#8B5CF6",
    "OPTIMAL_STATE":     "#06B6D4",
    None:                "#9CA3AF",
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def make_figure(data: dict, out_path: str):
    samples = data["samples"]
    n = data["n_total"]
    ckpt = Path(data["checkpoint"]).parent.name  # e.g. warmstart_qwen3_14b_39083858

    # ── Summary stats ────────────────────────────────────────────────────────
    status_counts = Counter(s["status"] for s in samples)
    action_counts = Counter(s["action_type"] for s in samples)
    output_tokens = [s["output_tokens"] for s in samples]
    think_rate  = data["think_tag_rate"]
    answer_rate = data["answer_rate"]
    compliance  = data["format_compliance"]
    tag_rate    = data["action_tag_rate"]

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(BG)

    ax_score  = fig.add_axes([0.04, 0.70, 0.44, 0.24])   # A — scorecard
    ax_action = fig.add_axes([0.56, 0.70, 0.40, 0.24])   # B — action dist
    ax_hist   = fig.add_axes([0.04, 0.38, 0.44, 0.24])   # C — token length histogram
    ax_grid   = fig.add_axes([0.56, 0.38, 0.40, 0.24])   # D — status grid (unused area below)
    ax_think  = fig.add_axes([0.04, 0.06, 0.92, 0.24])   # E — per-sample think/answer/status detail

    for ax in [ax_score, ax_action, ax_hist, ax_grid, ax_think]:
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(LIGHT)
        ax.spines["bottom"].set_color(LIGHT)
        ax.tick_params(colors=GREY, labelsize=8.5)

    fig.text(0.5, 0.975,
             f"Warmstart Format Compliance — {ckpt}",
             ha="center", va="top", fontsize=14, fontweight="bold", color=DARK)
    fig.text(0.5, 0.955,
             f"{n} dev prompts  ·  greedy decode  ·  max_new_tokens=256  ·  "
             f"threshold={data['threshold']:.0%}  ·  "
             f"GRPO-ready: {'✓ YES' if data['ready_for_grpo'] else '✗ NO'}",
             ha="center", va="top", fontsize=9.5, color=GREY)

    # ── A: Scorecard ─────────────────────────────────────────────────────────
    ax_score.axis("off")
    ax_score.text(0, 1.0, "A  Summary scorecard", fontsize=10,
                  fontweight="bold", color=DARK, transform=ax_score.transAxes, va="top")

    metrics = [
        ("format_compliance",  compliance,  C_TAG,   "fraction with parseable action\n(TAG or BARE)"),
        ("action_tag_rate",    tag_rate,    C_THINK, "fraction with <action>…</action>"),
        ("think_tag_rate",     think_rate,  C_THINK, "fraction with <think>…</think>"),
        ("answer_rate",        answer_rate, C_ANS,   "fraction with <answer>…</answer>"),
    ]

    bar_y_positions = [0.75, 0.55, 0.35, 0.15]
    for (label, value, color, desc), ypos in zip(metrics, bar_y_positions):
        # Background track
        ax_score.add_patch(FancyBboxPatch(
            (0.30, ypos - 0.06), 0.60, 0.12,
            boxstyle="round,pad=0.01", fc=LIGHT, ec=LIGHT, lw=0,
            transform=ax_score.transAxes, zorder=2))
        # Filled bar
        ax_score.add_patch(FancyBboxPatch(
            (0.30, ypos - 0.06), 0.60 * value, 0.12,
            boxstyle="round,pad=0.01", fc=color, ec=color, lw=0, alpha=0.85,
            transform=ax_score.transAxes, zorder=3))
        # Label
        ax_score.text(0.28, ypos, label, ha="right", va="center", fontsize=8.5,
                      color=DARK, transform=ax_score.transAxes)
        # Value
        ax_score.text(0.92, ypos, f"{value:.1%}", ha="left", va="center",
                      fontsize=9, color=color, fontweight="bold",
                      transform=ax_score.transAxes)
        # Description
        ax_score.text(0.30 + 0.60 * value + 0.01, ypos, desc,
                      ha="left", va="center", fontsize=6.5, color=GREY,
                      transform=ax_score.transAxes)

    # Threshold line drawn in axes fraction coordinates
    thresh_x = 0.30 + 0.60 * data["threshold"]
    ax_score.plot([thresh_x, thresh_x], [0.06, 0.90],
                  color=C_FAIL, lw=1.5, ls="--",
                  transform=ax_score.transAxes, zorder=5)
    ax_score.text(thresh_x + 0.01, 0.92,
                  f"threshold\n{data['threshold']:.0%}",
                  ha="left", va="top", fontsize=7, color=C_FAIL,
                  transform=ax_score.transAxes)

    # ── B: Action type distribution ──────────────────────────────────────────
    ax_action.set_title("B  Action type distribution (turn 1)", fontweight="bold",
                        color=DARK, fontsize=10, pad=6)
    ordered_actions = sorted(action_counts.items(), key=lambda x: -x[1])
    labels = [act if act else "(none)" for act, _ in ordered_actions]
    counts = [c for _, c in ordered_actions]
    colors = [ACTION_COLORS.get(a, GREY) for a, _ in ordered_actions]

    bars = ax_action.barh(range(len(labels)), counts, color=colors,
                          edgecolor=BG, height=0.65)
    ax_action.set_yticks(range(len(labels)))
    ax_action.set_yticklabels(labels, fontsize=8.5)
    ax_action.set_xlabel("Count", fontsize=8.5, color=GREY)
    ax_action.invert_yaxis()
    # Value labels
    for bar, count in zip(bars, counts):
        ax_action.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                       str(count), va="center", fontsize=8.5, color=DARK)
    ax_action.set_xlim(0, max(counts) * 1.25)
    ax_action.axvline(0, color=LIGHT, lw=0.5)

    # ── C: Output token length histogram ─────────────────────────────────────
    ax_hist.set_title("C  Output token length distribution", fontweight="bold",
                      color=DARK, fontsize=10, pad=6)
    bins = np.linspace(min(output_tokens), max(output_tokens) + 1, 20)
    ax_hist.hist(output_tokens, bins=bins, color=C_LEN, edgecolor=BG, alpha=0.85)
    ax_hist.axvline(np.mean(output_tokens), color=DARK, lw=1.5, ls="--")
    ax_hist.text(np.mean(output_tokens) + 1, ax_hist.get_ylim()[1] * 0.9,
                 f"mean={np.mean(output_tokens):.0f}",
                 fontsize=8, color=DARK)
    ax_hist.set_xlabel("Output tokens (generated)", fontsize=8.5, color=GREY)
    ax_hist.set_ylabel("Count", fontsize=8.5, color=GREY)

    # ── D: Status grid — 10×5 tiles ─────────────────────────────────────────
    ax_grid.axis("off")
    ax_grid.set_xlim(0, 10)
    ax_grid.set_ylim(0, 5)
    ax_grid.set_title("D  Per-sample status  (green=TAG · amber=BARE · red=FAIL)",
                       fontweight="bold", color=DARK, fontsize=10, pad=6)

    status_color = {"TAG": C_TAG, "BARE": C_BARE, "FAIL": C_FAIL}
    for idx, s in enumerate(samples[:50]):
        col = idx % 10
        row = idx // 10
        y = 4 - row
        color = status_color.get(s["status"], GREY)
        ax_grid.add_patch(FancyBboxPatch(
            (col + 0.05, y + 0.10), 0.90, 0.80,
            boxstyle="round,pad=0.04", fc=color, ec=WHITE, lw=1.0,
            alpha=0.85, zorder=3))
        ax_grid.text(col + 0.50, y + 0.50, str(idx + 1),
                     ha="center", va="center", fontsize=6.5, color=WHITE,
                     fontweight="bold", zorder=4)

    # Legend
    ax_grid.legend(loc="lower right", fontsize=8, framealpha=0.9,
                   facecolor=BG, edgecolor=LIGHT,
                   handles=[mpatches.FancyArrow(0, 0, 0, 0, width=0, fc=c, ec=c, label=l)
                             for l, c in [("TAG", C_TAG), ("BARE", C_BARE), ("FAIL", C_FAIL)]])

    # ── E: Per-sample detail strip (think / answer booleans) ─────────────────
    ax_think.axis("off")
    ax_think.set_xlim(0, len(samples))
    ax_think.set_ylim(0, 3.5)
    ax_think.set_title("E  Per-sample tag presence  (top row = <think>  ·  bottom row = <answer>)",
                        fontweight="bold", color=DARK, fontsize=10, pad=6)

    tile_w = 0.80
    for idx, s in enumerate(samples):
        x = idx + 0.10
        # <think> row
        think_color = C_THINK if s["has_think"] else LIGHT
        ax_think.add_patch(FancyBboxPatch(
            (x, 2.0), tile_w, 0.9,
            boxstyle="round,pad=0.03", fc=think_color, ec=WHITE, lw=0.5,
            alpha=0.85, zorder=3))
        # <answer> row
        ans_color = C_ANS if s["has_answer"] else LIGHT
        ax_think.add_patch(FancyBboxPatch(
            (x, 0.9), tile_w, 0.9,
            boxstyle="round,pad=0.03", fc=ans_color, ec=WHITE, lw=0.5,
            alpha=0.85, zorder=3))

    ax_think.text(-0.5, 2.45, "<think>",  ha="right", va="center",
                  fontsize=8, color=C_THINK, fontweight="bold")
    ax_think.text(-0.5, 1.35, "<answer>", ha="right", va="center",
                  fontsize=8, color=C_ANS, fontweight="bold")

    # x-axis tick labels (every 5)
    tick_positions = list(range(0, len(samples), 5))
    ax_think.set_xticks([p + 0.5 for p in tick_positions])
    ax_think.set_xticklabels([str(p + 1) for p in tick_positions], fontsize=7.5)
    ax_think.tick_params(bottom=True, labelbottom=True, left=False)
    ax_think.spines["bottom"].set_visible(True)
    ax_think.spines["bottom"].set_color(LIGHT)
    ax_think.set_xlabel("Sample index", fontsize=8.5, color=GREY)

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to per-sample JSON from check_format_compliance.py")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: figures/warmstart_format_compliance.png)")
    args = parser.parse_args()

    out = args.out or str(
        Path(__file__).parent / "warmstart_format_compliance.png"
    )
    data = load_results(args.results)
    make_figure(data, out)


if __name__ == "__main__":
    main()
