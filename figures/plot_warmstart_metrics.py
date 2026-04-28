"""
Warmstart Training Metrics Dashboard.

4-panel figure from actual run: warmstart_qwen3_14b_39083858
  (a) Training loss + eval checkpoints
  (b) Token-level accuracy
  (c) Cosine LR schedule
  (d) Gradient norms
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────────
BG    = "#FAFAF8"
DARK  = "#1C1C2E"
GREY  = "#6B7280"
LIGHT = "#E8E8E4"

C_TRAIN = "#3B82F6"   # blue  — training curve
C_EVAL  = "#EF4444"   # red   — eval checkpoints
C_LR    = "#8B5CF6"   # purple — LR
C_GRAD  = "#F59E0B"   # amber  — grad norm
C_STOP  = "#10B981"   # green  — early stop line

# ── Hard-coded data from metrics.jsonl ───────────────────────────────────────
train_steps    = [10,   20,     30,     40,     50,     60,     70,     80   ]
train_loss     = [1.2430, 0.3285, 0.3028, 0.2970, 0.2908, 0.2865, 0.2770, 0.2790]
train_acc      = [0.7478, 0.8659, 0.8720, 0.8750, 0.8786, 0.8806, 0.8864, 0.8858]
train_lr       = [4.845e-6, 4.336e-6, 3.547e-6, 2.598e-6, 1.635e-6, 8.030e-7, 2.296e-7, 1.927e-9]
train_grad     = [1.9815, 0.5745, 0.3897, 1.1984, 0.4144, 0.4962, 0.5796, 1.9278]

eval_steps     = [50,    80   ]
eval_loss      = [0.2997, 0.2979]
eval_acc       = [0.8745, 0.8770]

# ── Figure setup ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Warmstart SFT — Training Metrics  ·  Qwen3-14B  ·  8×H100-80GB  ·  2 epochs  ·  ~67 min",
    fontsize=13, fontweight="bold", color=DARK, y=0.98
)

for ax in axes.flat:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)

EARLY_STOP_STEP = 80
EPOCH1_STEP = 40   # epoch=1.0 at step 40
EPOCH2_STEP = 80   # epoch=2.0 at step 80


def add_epoch_lines(ax, ymin, ymax):
    for step, label in [(EPOCH1_STEP, "epoch 1"), (EPOCH2_STEP, "epoch 2")]:
        ax.axvline(step, color=LIGHT, lw=1.2, ls="--", zorder=1)
        ax.text(step + 0.5, ymax * 0.97, label, fontsize=7.5, color=GREY,
                va="top", rotation=0)


# ── (a) Training loss ─────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(train_steps, train_loss, color=C_TRAIN, lw=2.0, marker="o",
        markersize=5, zorder=3, label="train loss")
ax.plot(eval_steps, eval_loss, color=C_EVAL, lw=0, marker="D",
        markersize=8, zorder=4, label="eval loss")
# Connect eval points with thin line
ax.plot(eval_steps, eval_loss, color=C_EVAL, lw=1.2, ls="--", zorder=3)
# Early stop annotation
ax.axvline(EARLY_STOP_STEP, color=C_STOP, lw=1.5, ls="-.", zorder=2)
ax.text(EARLY_STOP_STEP - 1, 0.80,
        "early stop\n(Δloss=0.006\n< 0.05 thresh.)",
        fontsize=7.5, color=C_STOP, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=C_STOP, lw=0.8))

add_epoch_lines(ax, 0.25, 1.30)
ax.set_xlim(5, 85)
ax.set_ylim(0.24, 1.35)
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (cross-entropy)")
ax.set_title("(a) Loss", fontweight="bold", color=DARK, fontsize=11)
ax.legend(fontsize=8.5, framealpha=0.9, facecolor=BG, edgecolor=LIGHT)

# Annotate rapid drop
ax.annotate("Rapid convergence\nstep 10→20 (1.24→0.33)",
            xy=(20, 0.3285), xytext=(28, 0.62),
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0),
            fontsize=7.5, color=GREY, ha="center")


# ── (b) Token accuracy ────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(train_steps, [v * 100 for v in train_acc], color=C_TRAIN, lw=2.0,
        marker="o", markersize=5, zorder=3, label="train acc.")
ax.plot(eval_steps, [v * 100 for v in eval_acc], color=C_EVAL, lw=0,
        marker="D", markersize=8, zorder=4, label="eval acc.")
ax.plot(eval_steps, [v * 100 for v in eval_acc], color=C_EVAL, lw=1.2, ls="--", zorder=3)
ax.axvline(EARLY_STOP_STEP, color=C_STOP, lw=1.5, ls="-.", zorder=2)
ax.axhline(87.0, color=GREY, lw=0.8, ls=":", zorder=1)
ax.text(6, 87.3, "87% reference", fontsize=7.5, color=GREY)

add_epoch_lines(ax, 74, 91)
ax.set_xlim(5, 85)
ax.set_ylim(73, 91)
ax.set_xlabel("Training step")
ax.set_ylabel("Mean token accuracy (%)")
ax.set_title("(b) Token accuracy", fontweight="bold", color=DARK, fontsize=11)
ax.legend(fontsize=8.5, framealpha=0.9, facecolor=BG, edgecolor=LIGHT)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))

# Annotate final value
ax.annotate(f"Final: {train_acc[-1]*100:.1f}%",
            xy=(80, train_acc[-1]*100), xytext=(70, 89.0),
            arrowprops=dict(arrowstyle="->", color=C_TRAIN, lw=1.0),
            fontsize=8, color=C_TRAIN)


# ── (c) Cosine LR schedule ────────────────────────────────────────────────────
ax = axes[1, 0]
ax.semilogy(train_steps, train_lr, color=C_LR, lw=2.0, marker="o",
            markersize=5, zorder=3)
ax.axvline(EARLY_STOP_STEP, color=C_STOP, lw=1.5, ls="-.", zorder=2)

# Shade warmup region (0→8 steps = 10% of 80)
ax.axvspan(0, 8, alpha=0.08, color=C_LR, zorder=1)
ax.text(4, 6e-6, "warmup\n(10%)", ha="center", fontsize=7.5, color=C_LR)

add_epoch_lines(ax, 1e-9, 6e-6)
ax.set_xlim(5, 85)
ax.set_ylim(1e-9, 6e-6)
ax.set_xlabel("Training step")
ax.set_ylabel("Learning rate")
ax.set_title("(c) Cosine LR schedule  (base: 5×10⁻⁶)", fontweight="bold", color=DARK, fontsize=11)

# Annotate start/end values
ax.text(10, train_lr[0] * 1.5, f"{train_lr[0]:.2e}", fontsize=7.5, color=C_LR)
ax.text(72, train_lr[-1] * 3, f"~0\n(cosine end)", fontsize=7.5, color=C_LR, ha="center")


# ── (d) Gradient norms ───────────────────────────────────────────────────────
ax = axes[1, 1]
bars = ax.bar(train_steps, train_grad, width=6, color=C_GRAD, alpha=0.8,
              edgecolor=BG, zorder=3)
ax.axhline(0.5, color=GREY, lw=0.8, ls=":", zorder=2)
ax.text(6, 0.52, "0.5 reference", fontsize=7.5, color=GREY)
ax.axvline(EARLY_STOP_STEP, color=C_STOP, lw=1.5, ls="-.", zorder=2)
# Annotate initial and final spikes
ax.annotate("Initial\nspike\n1.98",
            xy=(10, 1.9815), xytext=(18, 1.75),
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0),
            fontsize=7.5, color=GREY, ha="center")
ax.annotate("Final step\nspike 1.93",
            xy=(80, 1.9278), xytext=(72, 1.75),
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0),
            fontsize=7.5, color=GREY, ha="center")

add_epoch_lines(ax, 0, 2.1)
ax.set_xlim(5, 85)
ax.set_ylim(0, 2.1)
ax.set_xlabel("Training step")
ax.set_ylabel("Gradient norm  (clip: 0.5)")
ax.set_title("(d) Gradient norms", fontweight="bold", color=DARK, fontsize=11)

# ── Footer ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "Run: warmstart_qwen3_14b_39083858  ·  SLURM job 39083858  ·  "
         "DeepSpeed ZeRO-2  ·  BF16  ·  batch 1×16 grad-accum  ·  "
         "Max seq len 8192  ·  5,000 training traces",
         ha="center", fontsize=8, color=GREY)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

out = "/ocean/projects/mch250030p/wxu7/llm_finetune/figures/warmstart_metrics.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
