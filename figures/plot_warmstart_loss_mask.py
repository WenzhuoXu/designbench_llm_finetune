"""
Warmstart Loss Mask Visualization.

Shows which tokens in a training sequence are supervised (loss computed)
vs masked (labels=-100) under the WarmstartReasoningTarget.

Three panels:
  A) Raw DesignBench JSONL assistant message (before transform)
  B) Transformed sequence with color-coded loss regions
  C) Comparison bar: warmstart_reasoning vs full_sequence supervision ratio
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Palette (matches plot_overview.py) ─────────────────────────────────────
BG     = "#FAFAF8"
DARK   = "#1C1C2E"
GREY   = "#6B7280"
LIGHT  = "#E8E8E4"
WHITE  = "#FFFFFF"

C_SYSTEM   = "#9CA3AF"   # gray   — system / prompt tokens (masked)
C_USER_FBK = "#D1D5DB"   # light gray — user feedback tokens (masked)
C_THINK    = "#10B981"   # green  — <think> span (supervised)
C_ACTION   = "#3B82F6"   # blue   — <action> span (supervised)
C_ANSWER   = "#F59E0B"   # amber  — <answer> span (supervised)
C_MASK     = "#E5E7EB"   # very light gray — other tokens (masked)
C_FULL     = "#8B5CF6"   # purple — full_sequence supervised

# ── Token sequence definitions ──────────────────────────────────────────────
# We use symbolic "token blocks" (not real tokenizer output) to illustrate
# the masking logic. Each block has (label, color, width, is_supervised).

# Panel A: Raw DesignBench JSONL — single assistant message as-is
RAW_TOKENS = [
    # <think> tag + content
    ("<think>",       C_MASK,   0.7, False),
    ("Modification:", C_MASK,   1.2, False),
    ("Add",           C_MASK,   0.5, False),
    ("member",        C_MASK,   0.7, False),
    ("connecting",    C_MASK,   1.0, False),
    ("joints",        C_MASK,   0.7, False),
    ("1",             C_MASK,   0.3, False),
    ("and",           C_MASK,   0.4, False),
    ("6",             C_MASK,   0.3, False),
    ("\\n",           C_MASK,   0.4, False),
    ("Action:",       C_MASK,   0.7, False),
    ("ADD_MEMBER",    C_MASK,   1.1, False),
    ("(1,",           C_MASK,   0.5, False),
    ("6,",            C_MASK,   0.4, False),
    ("A36_Steel,",    C_MASK,   1.0, False),
    ("Pipe,",         C_MASK,   0.6, False),
    ("0.023,",        C_MASK,   0.7, False),
    ("0.0039)",       C_MASK,   0.8, False),
    ("</think>",      C_MASK,   0.8, False),
]

# Panel B: After WarmstartReasoningTarget transform — full training sequence
TRANSFORMED_TOKENS = [
    # System prompt (masked)
    ("[SYS]",         C_SYSTEM, 0.5, False),
    ("PROBLEM:",      C_SYSTEM, 0.8, False),
    ("Truss",         C_SYSTEM, 0.6, False),
    ("opt.",          C_SYSTEM, 0.5, False),
    ("...",           C_SYSTEM, 0.4, False),
    ("INITIAL",       C_SYSTEM, 0.7, False),
    ("STATE:",        C_SYSTEM, 0.6, False),
    ("Mass=213",      C_SYSTEM, 0.8, False),
    ("FOS=0.73",      C_SYSTEM, 0.8, False),
    # <think> supervised
    ("<think>",       C_THINK,  0.7, True),
    ("Add",           C_THINK,  0.5, True),
    ("member",        C_THINK,  0.7, True),
    ("1→6",           C_THINK,  0.6, True),
    ("</think>",      C_THINK,  0.8, True),
    # <action> supervised
    ("<action>",      C_ACTION, 0.7, True),
    ("ADD_MEMBER",    C_ACTION, 1.1, True),
    ("(1,6,",         C_ACTION, 0.6, True),
    ("A36,",          C_ACTION, 0.5, True),
    ("Pipe,",         C_ACTION, 0.5, True),
    ("0.023,",        C_ACTION, 0.6, True),
    ("0.004)",        C_ACTION, 0.6, True),
    ("</action>",     C_ACTION, 0.8, True),
    # User feedback (masked)
    ("[USR]",         C_USER_FBK, 0.5, False),
    ("[Sim",          C_USER_FBK, 0.5, False),
    ("Result]",       C_USER_FBK, 0.7, False),
    ("Mass=230",      C_USER_FBK, 0.8, False),
    ("FOS=0.73",      C_USER_FBK, 0.8, False),
    ("INFEAS.",       C_USER_FBK, 0.7, False),
    # Second <think>/<action> (supervised)
    ("<think>",       C_THINK,  0.7, True),
    ("Scale",         C_THINK,  0.5, True),
    ("member",        C_THINK,  0.7, True),
    ("6",             C_THINK,  0.3, True),
    ("thick.",        C_THINK,  0.6, True),
    ("</think>",      C_THINK,  0.8, True),
    ("<action>",      C_ACTION, 0.7, True),
    ("SCALE_PARAM",   C_ACTION, 1.1, True),
    ("(6,thick,",     C_ACTION, 0.9, True),
    ("1.631)",        C_ACTION, 0.7, True),
    ("</action>",     C_ACTION, 0.8, True),
    # More feedback (masked)
    ("[USR]",         C_USER_FBK, 0.5, False),
    ("...",           C_USER_FBK, 0.4, False),
    ("FEASIBLE✓",     C_USER_FBK, 0.9, False),
    # Final <answer> (supervised)
    ("<answer>",      C_ANSWER, 0.8, True),
    ("Optimal",       C_ANSWER, 0.8, True),
    ("solution",      C_ANSWER, 0.8, True),
    ("found",         C_ANSWER, 0.6, True),
    ("</answer>",     C_ANSWER, 0.9, True),
]

# ── Layout ───────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(BG)

# Three panels
ax_a  = fig.add_axes([0.04, 0.72, 0.92, 0.20])  # Panel A — raw message
ax_b  = fig.add_axes([0.04, 0.38, 0.92, 0.28])  # Panel B — transformed sequence
ax_c  = fig.add_axes([0.10, 0.06, 0.80, 0.22])  # Panel C — supervision comparison

for ax in [ax_a, ax_b, ax_c]:
    ax.set_facecolor(BG)
    ax.set_xlim(0, 20)
    ax.axis("off")

# ── Title ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, "Warmstart SFT — Loss Mask Visualization",
         ha="center", va="top", fontsize=16, fontweight="bold", color=DARK)
fig.text(0.5, 0.945, "Cross-entropy loss is computed only on supervised (colored) tokens  ·  "
         "Masked tokens get labels = −100 in DataCollatorForSFT",
         ha="center", va="top", fontsize=10, color=GREY)


def draw_token_row(ax, tokens, y_center, height=0.55, gap=0.08, start_x=0.15, label=None):
    """Draw a horizontal row of token tiles."""
    x = start_x
    for (text, color, width, supervised) in tokens:
        rect = FancyBboxPatch(
            (x, y_center - height / 2), width - gap, height,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            fc=color, ec=WHITE if supervised else LIGHT,
            lw=1.5 if supervised else 0.8, zorder=3
        )
        ax.add_patch(rect)
        fontsize = 6.5 if len(text) > 9 else 7.5
        ax.text(x + (width - gap) / 2, y_center, text,
                ha="center", va="center", fontsize=fontsize,
                color=DARK if not supervised else WHITE,
                fontweight="bold" if supervised else "normal", zorder=4)
        x += width
    if label:
        ax.text(0.05, y_center, label, ha="right", va="center",
                fontsize=8.5, color=GREY, fontstyle="italic")
    return x  # return end x for annotation


# ── Panel A: Raw JSONL message ────────────────────────────────────────────────
ax_a.set_ylim(0, 2)
ax_a.set_xlim(0, 20)
ax_a.text(0.0, 1.85, "A  Raw DesignBench JSONL — assistant message (before transform)",
          fontsize=10, fontweight="bold", color=DARK)
ax_a.text(0.0, 1.60,
          "role: \"assistant\"  ·  content: \"<think>Modification: Add member connecting joints 1 and 6\\nAction: ADD_MEMBER(1, 6, ...)</think>\"",
          fontsize=8, color=GREY, fontstyle="italic")

draw_token_row(ax_a, RAW_TOKENS, y_center=0.75, start_x=0.15)
# Label all tokens as "no supervision yet"
ax_a.annotate("", xy=(19.8, 0.75), xytext=(19.0, 0.75),
              arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
ax_a.text(19.85, 0.75, "All tokens:\nno explicit\nmask yet",
          va="center", fontsize=7, color=GREY)

# Bottom brace annotation
ax_a.annotate("", xy=(0.15, 0.2), xytext=(13.2, 0.2),
              arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0))
ax_a.text(6.7, 0.08, "Raw <think> content — action buried inside (no <action> tag)",
          ha="center", fontsize=8, color=GREY, fontstyle="italic")


# ── Panel B: Transformed sequence ────────────────────────────────────────────
ax_b.set_ylim(0, 3)
ax_b.set_xlim(0, 20)
ax_b.text(0.0, 2.85, "B  After WarmstartReasoningTarget — full tokenized training sequence",
          fontsize=10, fontweight="bold", color=DARK)
ax_b.text(0.0, 2.62,
          "Colored tiles = supervised (loss computed)  ·  Gray tiles = masked (labels = −100)",
          fontsize=8.5, color=GREY)

draw_token_row(ax_b, TRANSFORMED_TOKENS, y_center=1.55, height=0.60, start_x=0.15)

# Bracket annotations below the token row
brackets = [
    (0.15, 4.55,  "System prompt\n(masked)", C_SYSTEM),
    (4.55, 9.85,  "<think>+<action>\nturn 1 (supervised)", C_THINK),
    (9.85, 13.45, "Simulation\nfeedback (masked)", C_USER_FBK),
    (13.45, 18.65,"<think>+<action>\nturn 2 (supervised)", C_ACTION),
    (18.65, 20.45,"<answer>\n(supervised)", C_ANSWER),
]
for (x0, x1, label, color) in brackets:
    mid = (x0 + x1) / 2
    ax_b.annotate("", xy=(x0 + 0.05, 1.10), xytext=(x1 - 0.05, 1.10),
                  arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
    ax_b.text(mid, 0.85, label, ha="center", va="top", fontsize=7,
              color=color, fontweight="bold", multialignment="center")

# Legend
legend_items = [
    mpatches.Patch(fc=C_THINK,    ec=WHITE, label="<think> span — supervised"),
    mpatches.Patch(fc=C_ACTION,   ec=WHITE, label="<action> span — supervised"),
    mpatches.Patch(fc=C_ANSWER,   ec=WHITE, label="<answer> span — supervised"),
    mpatches.Patch(fc=C_SYSTEM,   ec=LIGHT, label="System prompt — masked (labels=−100)"),
    mpatches.Patch(fc=C_USER_FBK, ec=LIGHT, label="[Simulation Result] user feedback — masked"),
]
ax_b.legend(handles=legend_items, loc="upper right", fontsize=7.5,
            framealpha=0.9, facecolor=BG, edgecolor=LIGHT, ncol=2,
            bbox_to_anchor=(1.0, 2.82))


# ── Panel C: Supervision ratio comparison ────────────────────────────────────
ax_c.set_ylim(0, 3)
ax_c.set_xlim(0, 20)
ax_c.text(0.0, 2.85, "C  Supervision ratio: WarmstartReasoningTarget  vs  FullSequenceTarget",
          fontsize=10, fontweight="bold", color=DARK)

# Approximate token counts from the symbolic sequence above
total_tokens = sum(w for (_, _, w, _) in TRANSFORMED_TOKENS)
supervised_tokens = sum(w for (_, _, w, sup) in TRANSFORMED_TOKENS if sup)
masked_tokens = total_tokens - supervised_tokens

# Full sequence: supervises all non-system tokens
# system tokens approx (first 9 entries) — everything else is supervised
system_tokens = sum(w for (_, _, w, _) in TRANSFORMED_TOKENS[:9])
full_supervised = total_tokens - system_tokens
full_masked = system_tokens

bar_y = [1.6, 0.7]
bar_labels = ["warmstart_reasoning\n(selective masking)", "full_sequence\n(full response)"]
supervised_vals = [supervised_tokens / total_tokens, full_supervised / total_tokens]
masked_vals     = [masked_tokens / total_tokens,     full_masked / total_tokens]

bar_width = 16
bar_height = 0.55

for i, (y, lbl, sup_frac, msk_frac) in enumerate(
        zip(bar_y, bar_labels, supervised_vals, masked_vals)):

    sup_w = sup_frac * bar_width
    msk_w = msk_frac * bar_width

    # Supervised (colored gradient for warmstart)
    color = C_THINK if i == 0 else C_FULL
    ax_c.add_patch(FancyBboxPatch((2.0, y - bar_height/2), sup_w, bar_height,
                                  boxstyle="round,pad=0.03", fc=color, ec=WHITE,
                                  lw=1.0, zorder=3))
    # Masked
    ax_c.add_patch(FancyBboxPatch((2.0 + sup_w, y - bar_height/2), msk_w, bar_height,
                                  boxstyle="round,pad=0.03", fc=LIGHT, ec=WHITE,
                                  lw=0.8, zorder=3))

    ax_c.text(1.85, y, lbl, ha="right", va="center", fontsize=8, color=DARK,
              multialignment="right")
    ax_c.text(2.0 + sup_w / 2, y, f"{sup_frac*100:.0f}% supervised",
              ha="center", va="center", fontsize=8, color=WHITE, fontweight="bold", zorder=4)
    ax_c.text(2.0 + sup_w + msk_w / 2, y, f"{msk_frac*100:.0f}% masked",
              ha="center", va="center", fontsize=8, color=GREY, zorder=4)

ax_c.text(10, 0.12,
          "WarmstartReasoningTarget supervises only <think>, <action>, <answer> spans  —  "
          "reduces overfitting to prompt/feedback text, preserving RL plasticity",
          ha="center", fontsize=8.5, color=GREY, fontstyle="italic")

# ── Save ─────────────────────────────────────────────────────────────────────
out = "/ocean/projects/mch250030p/wxu7/llm_finetune/figures/warmstart_loss_mask.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
