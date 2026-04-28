"""
Single-panel overview diagram for llm_finetune.

Core story:
  SFT  — imitation learning from expert traces; supervision target is the hook
  RL   — reinforcement from FEA environment; reward/cost functions are the hooks
  Both fine-tune the same LLM backbone toward DesignBench truss design.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Palette ────────────────────────────────────────────────────────────────
BG      = "#FAFAF8"
DARK    = "#1C1C2E"
GREY    = "#6B7280"
LIGHT   = "#E8E8E4"

C_SFT   = "#3B82F6"   # blue   — SFT
C_RL    = "#EF4444"   # red    — RL / GRPO
C_HOOK  = "#F59E0B"   # amber  — research hooks
C_DATA  = "#10B981"   # green  — data / environment
C_MODEL = "#8B5CF6"   # purple — LLM model
C_ENV   = "#06B6D4"   # cyan   — FEA environment

WHITE = "#FFFFFF"

fig, ax = plt.subplots(figsize=(20, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 9)
ax.axis("off")

# ── helpers ────────────────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, text, fc, ec=None, fontsize=9, tc=WHITE,
         bold=True, alpha=1.0, radius=0.35, lw=1.5):
    ec = ec or fc
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={radius}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=tc, fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center", linespacing=1.4)

def outline_box(ax, cx, cy, w, h, text, fc, fontsize=8.5, tc=None, lw=2.0,
                dash=False, radius=0.35):
    tc = tc or fc
    ls = (0, (5, 4)) if dash else "solid"
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={radius}",
                       fc=BG, ec=fc, lw=lw, linestyle=ls, alpha=1.0, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=tc, fontweight="bold", zorder=4, multialignment="center",
            linespacing=1.4)

def arr(ax, x0, y0, x1, y1, color=DARK, lw=2.0, hw=0.25, hl=0.4,
        conn=None, style="->"):
    kw = dict(arrowstyle=f"{style},head_width={hw},head_length={hl}",
              color=color, lw=lw)
    if conn:
        kw["connectionstyle"] = conn
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=kw, zorder=5)

def label(ax, x, y, text, color=DARK, size=8, ha="center", va="center",
          bold=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
            fontweight="bold" if bold else "normal", zorder=6)

def section_bg(ax, x, y, w, h, color, alpha=0.06, label_text="", lsize=8.5):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.1,rounding_size=0.4",
                       fc=color, ec=color, lw=1.5, alpha=alpha, zorder=1)
    ax.add_patch(p)
    if label_text:
        ax.text(x + w/2, y + h + 0.18, label_text, ha="center",
                fontsize=lsize, color=color, fontweight="bold", zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
ax.text(10, 8.65, "llm_finetune  —  Fine-tuning LLMs for Truss Design",
        ha="center", fontsize=15, fontweight="bold", color=DARK)
ax.text(10, 8.25,
        "SFT learns the design grammar from expert traces  ·  "
        "RL optimizes for real FEA quality  ·  "
        "Research hooks control the learning objective",
        ha="center", fontsize=9, color=GREY)

# ══════════════════════════════════════════════════════════════════════════════
# Three vertical zones: SFT (left)  |  Model (centre)  |  RL (right)
# ══════════════════════════════════════════════════════════════════════════════

# background zones
section_bg(ax,  0.3, 0.3, 5.8, 7.5, C_SFT,  alpha=0.06)
section_bg(ax,  7.0, 0.3, 6.0, 7.5, C_MODEL,alpha=0.06)
section_bg(ax, 13.9, 0.3, 5.8, 7.5, C_RL,   alpha=0.06)

ax.text(3.2,  8.0, "① SFT — Imitation Learning",    ha="center", fontsize=11, fontweight="bold", color=C_SFT)
ax.text(10.0, 8.0, "② LLM Backbone",                ha="center", fontsize=11, fontweight="bold", color=C_MODEL)
ax.text(16.8, 8.0, "③ RL — Reward-Driven Learning", ha="center", fontsize=11, fontweight="bold", color=C_RL)

# ── ① SFT zone ─────────────────────────────────────────────────────────────

# Data source
rbox(ax, 3.2, 7.1, 4.8, 0.75,
     "benchmark.json  (5 000 expert traces)\n"
     "problem · gold action sequence · FEA state history",
     C_DATA, fontsize=8.5)

arr(ax, 3.2, 6.72, 3.2, 6.25, color=C_DATA)

# Chat formatter
rbox(ax, 3.2, 5.9, 4.8, 0.60,
     "ChatFormatter  →  tokenize\n"
     "(Qwen3 · DeepSeek-R1 · Phi-4 · Llama4 · Gemma3)",
     C_SFT, fontsize=8)

arr(ax, 3.2, 5.60, 3.2, 5.15, color=C_SFT)

# SFT Dataset
rbox(ax, 3.2, 4.80, 4.8, 0.60,
     "SFTDataset  /  PackedSFTDataset\n"
     "input_ids · labels · attention_mask",
     C_SFT, fontsize=8)

arr(ax, 3.2, 4.50, 3.2, 4.00, color=C_SFT)

# ── HOOK: SFT Target ────────────────────────────────
hook_y = 3.65
# hook box
outline_box(ax, 3.2, hook_y, 4.8, 0.80,
            "RESEARCH HOOK — SFTTarget\n"
            "Which tokens carry the loss?",
            C_HOOK, fontsize=9, lw=2.5)
# hook variants below
for i, (name, desc) in enumerate([
    ("action_only",         "only grammar commands"),
    ("thinking_and_action", "<think> blocks + commands"),
    ("full_sequence",       "all assistant tokens"),
    ("final_answer",        "last action only"),
]):
    xi = 0.6 + i * 1.2
    rbox(ax, xi, 2.90, 1.08, 0.52, f"{name}\n({desc})",
         C_HOOK, fontsize=6.5, radius=0.2)
    arr(ax, 3.2, 3.25, xi, 3.17, color=C_HOOK, lw=1.2, hw=0.15, hl=0.25,
        conn="arc3,rad=0.0")

arr(ax, 3.2, 4.00, 3.2, 3.65+0.40, color=C_SFT)

# SFT loss arrow → model
arr(ax, 5.60, 3.65, 7.1, 4.90, color=C_SFT, lw=2.2)
label(ax, 6.4, 4.45, "cross-entropy loss\n(masked by SFTTarget)",
      color=C_SFT, size=7.5, ha="center")

# TRL SFTTrainer
rbox(ax, 3.2, 2.05, 4.8, 0.72,
     "TRL SFTTrainer  (DeepSpeed ZeRO-2/3)\n"
     "cosine LR · gradient checkpointing · W&B logging",
     C_SFT, fontsize=8)
arr(ax, 3.2, 2.90+0.0-0.5, 3.2, 2.05+0.36, color=C_SFT, lw=1.5)

rbox(ax, 3.2, 1.10, 4.8, 0.60,
     "checkpoints/sft/{run}/final/\n"
     "(can seed GRPO warm-start)",
     C_SFT, fontsize=8)
arr(ax, 3.2, 1.69, 3.2, 1.40, color=C_SFT)

# ── ② LLM Backbone (centre) ────────────────────────────────────────────────

rbox(ax, 10.0, 5.25, 5.5, 1.80,
     "LLM  (14B–17B parameters)\n\n"
     "Flash Attention 2  ·  BF16  ·  torch.compile\n"
     "Gradient checkpointing  ·  optional LoRA\n"
     "Qwen3-14B  /  DeepSeek-R1  /  Phi-4-reasoning …",
     C_MODEL, fontsize=8.5, radius=0.45)

# weight update arrows (both sides)
arr(ax, 8.75, 5.05, 8.75, 4.95, color=C_SFT,   lw=0, hw=0)  # invisible spacer
# SFT update label
label(ax, 8.6, 6.1, "weight update\n(SFT)", color=C_SFT, size=7.5)
arr(ax, 7.25, 5.8, 7.25, 5.8, color=C_SFT, lw=0)

# RL update → model
arr(ax, 12.75, 4.90, 13.9, 3.65+0.4, color=C_RL, lw=2.2)
label(ax, 13.55, 4.45, "GRPO gradient\n+ KL penalty",
      color=C_RL, size=7.5, ha="center")

# ── ③ RL zone ──────────────────────────────────────────────────────────────

# Problem prompts
rbox(ax, 16.8, 7.1, 4.8, 0.75,
     "RLPromptDataset  (20 truss problems)\n"
     "prompt only — no gold answer",
     C_DATA, fontsize=8.5)

arr(ax, 16.8, 6.72, 16.8, 6.25, color=C_DATA)

# vLLM generation
rbox(ax, 16.8, 5.9, 4.8, 0.60,
     "vLLM Server  (tensor-parallel, 8×H100)\n"
     "generate  K = 8  completions per prompt  (T = 0.8)",
     C_RL, fontsize=8)

# model → vLLM arrow
arr(ax, 12.75, 5.60, 14.3, 5.85, color=C_MODEL, lw=1.8,
    conn="arc3,rad=-0.15")
label(ax, 13.55, 6.05, "rollout\ngenerate", color=C_MODEL, size=7.5, ha="center")

arr(ax, 16.8, 5.60, 16.8, 5.15, color=C_RL)

# FEA environment
rbox(ax, 16.8, 4.80, 4.8, 0.60,
     "TrussRolloutEnv  (32 parallel FEA workers)\n"
     "parse grammar actions  →  FEA execute  →  state history",
     C_ENV, fontsize=8)

arr(ax, 16.8, 4.50, 16.8, 4.00, color=C_ENV)

# ── HOOK: Reward / Cost ─────────────────────────────
outline_box(ax, 16.8, 3.65, 4.8, 0.80,
            "RESEARCH HOOK — RewardFunction / CostFunction\n"
            "What counts as a good design?",
            C_HOOK, fontsize=9, lw=2.5)

for i, (name, desc) in enumerate([
    ("feasibility",    "+1 if FOS ≥ 1.5"),
    ("fos_improvement","ΔFOS continuous"),
    ("mass_reduction", "Δmass normalized"),
    ("grammar",        "valid action parse"),
]):
    xi = 14.4 + i * 1.2
    rbox(ax, xi, 2.90, 1.08, 0.52, f"{name}\n({desc})",
         C_HOOK, fontsize=6.5, radius=0.2)
    arr(ax, 16.8, 3.25, xi, 3.17, color=C_HOOK, lw=1.2, hw=0.15, hl=0.25)

arr(ax, 16.8, 4.00, 16.8, 3.65+0.40, color=C_ENV)

# GRPO normalization
rbox(ax, 16.8, 2.05, 4.8, 0.72,
     "TRL GRPOTrainer  —  Group Relative Policy Optimization\n"
     "advantage = (r − μ) / σ  within group  ·  KL(π ‖ π_ref)  penalty",
     C_RL, fontsize=8)
arr(ax, 16.8, 2.90+0.0-0.5, 16.8, 2.05+0.36, color=C_RL, lw=1.5)

rbox(ax, 16.8, 1.10, 4.8, 0.60,
     "checkpoints/grpo/{run}/final/\n"
     "optimized policy — feasible designs, low mass",
     C_RL, fontsize=8)
arr(ax, 16.8, 1.69, 16.8, 1.40, color=C_RL)

# sync updated weights back to vLLM
arr(ax, 16.8, 2.41, 16.8, 5.60, color=C_RL, lw=1.2,
    conn="arc3,rad=0.5", style="->")
label(ax, 19.5, 4.0, "sync\nweights", color=C_RL, size=7.5, ha="center")

# ══════════════════════════════════════════════════════════════════════════════
# Hook callout legend (bottom centre)
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_DATA,  "Data / Environment"),
    (C_SFT,   "SFT pipeline"),
    (C_MODEL, "LLM backbone"),
    (C_RL,    "RL / GRPO pipeline"),
    (C_ENV,   "FEA environment"),
    (C_HOOK,  "Research hook  (pluggable)"),
]
lx = 5.5
for col, txt in legend_items:
    ax.add_patch(mpatches.FancyBboxPatch((lx - 0.22, 0.38), 0.44, 0.35,
                 boxstyle="round,pad=0.05", fc=col, ec=col, lw=0, zorder=5))
    ax.text(lx + 0.38, 0.56, txt, fontsize=8, color=DARK, va="center", zorder=6)
    lx += 2.45

# ══════════════════════════════════════════════════════════════════════════════
# Divider lines
# ══════════════════════════════════════════════════════════════════════════════
for xd in [6.95, 13.05]:
    ax.plot([xd, xd], [0.3, 8.05], color=LIGHT, lw=1.5, zorder=1)

# ── Save ──────────────────────────────────────────────────────────────────
out = "/ocean/projects/mch250030p/wxu7/llm_finetune/figures/llm_finetune_overview.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor=BG)
print(f"Saved: {out.replace('.png', '.pdf')}")
