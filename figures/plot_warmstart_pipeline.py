"""
Warmstart Message Transformation Pipeline.

Shows WarmstartTransform.transform_messages():
  Left:   Raw DesignBench SFT JSONL message sequence
  Center: Transform annotations (what changes and why)
  Right:  Output message sequence (warmstart training format)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Palette ──────────────────────────────────────────────────────────────────
BG     = "#FAFAF8"
DARK   = "#1C1C2E"
GREY   = "#6B7280"
LIGHT  = "#E8E8E4"
WHITE  = "#FFFFFF"

C_SYS  = "#9CA3AF"   # gray   — system messages
C_AST  = "#3B82F6"   # blue   — assistant messages
C_USR  = "#10B981"   # green  — user messages (remapped from system)
C_ANS  = "#F59E0B"   # amber  — answer messages
C_HOOK = "#8B5CF6"   # purple — transform function highlight

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 13))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")

fig.text(0.5, 0.975,
         "WarmstartTransform.transform_messages()  —  Message Reformatting Pipeline",
         ha="center", va="top", fontsize=14, fontweight="bold", color=DARK)
fig.text(0.5, 0.955,
         "Converts raw DesignBench SFT traces into the <think>/<action> format required for warmstart SFT",
         ha="center", va="top", fontsize=9.5, color=GREY)


# ── Helper functions ──────────────────────────────────────────────────────────

def msg_box(ax, x, y, w, h, role_label, role_color, content_lines,
            fontsize_content=7.5, annotation=None):
    """Draw a message box with role badge + content."""
    # Outer box
    outer = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.08,rounding_size=0.15",
                           fc=role_color + "22",  # very light tint
                           ec=role_color, lw=1.6, zorder=3)
    ax.add_patch(outer)
    # Role badge
    badge_w = 0.85
    badge = FancyBboxPatch((x, y + h - 0.32), badge_w, 0.32,
                           boxstyle="round,pad=0.04,rounding_size=0.08",
                           fc=role_color, ec=role_color, lw=1.0, zorder=4)
    ax.add_patch(badge)
    ax.text(x + badge_w / 2, y + h - 0.16, role_label,
            ha="center", va="center", fontsize=7, color=WHITE,
            fontweight="bold", zorder=5)
    # Content
    content_y = y + h - 0.50
    for line in content_lines:
        ax.text(x + 0.12, content_y, line, ha="left", va="top",
                fontsize=fontsize_content, color=DARK, zorder=5,
                wrap=True)
        content_y -= 0.22
    # Optional annotation tag (e.g., "① merge")
    if annotation:
        ax.text(x + w - 0.08, y + h - 0.16, annotation,
                ha="right", va="center", fontsize=7.5,
                color=C_HOOK, fontweight="bold", zorder=5)
    return y + h / 2  # return center y


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.2, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=6)


def transform_label(ax, x, y, text, color=C_HOOK):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=color, lw=1.0),
            zorder=7)


# ── Column positions ──────────────────────────────────────────────────────────
X_LEFT   = 0.3    # left column start x
X_RIGHT  = 10.1   # right column start x
BOX_W    = 7.5    # box width for both columns
X_MID    = X_LEFT + BOX_W + 0.5   # center of transform zone

# ── LEFT column: Raw JSONL messages ──────────────────────────────────────────
ax.text(X_LEFT + BOX_W / 2, 12.55,
        "Raw DesignBench SFT JSONL", ha="center", fontsize=11,
        fontweight="bold", color=DARK)
ax.text(X_LEFT + BOX_W / 2, 12.25,
        "DesignBench/data/sft/train.jsonl", ha="center", fontsize=8.5,
        color=GREY, fontstyle="italic")

# Message 0: system — problem
msg_box(ax, X_LEFT, 10.60, BOX_W, 1.50, "system", C_SYS,
        ["PROBLEM: Truss optimization (variant 79)",
         "INITIAL STRUCTURE: 8 joints, 13 members",
         "Joints: J0=(0,0)[pinned], J1=(3.48,0)[free], ...",
         "Members: M0: J0-J1 [A36_Steel, Pipe(r=0.028, t=0.004)]",
         "Goals: FOS_buck≥1.5, FOS_yield≥1.5, min mass"])

# Message 1: system — initial state
msg_box(ax, X_LEFT, 9.00, BOX_W, 1.45, "system", C_SYS,
        ["INITIAL STATE ANALYSIS:",
         "Mass: 213.21 kg",
         "FOS (buckling): 0.73  ← INFEASIBLE",
         "FOS (yielding): 2.68",
         "Status: INFEASIBLE ✗"])

# Message 2: assistant — first action
msg_box(ax, X_LEFT, 7.45, BOX_W, 1.40, "assistant", C_AST,
        ['<think>Modification: Add member connecting',
         '        joints 1 and 6',
         '  Action: ADD_MEMBER(1, 6, A36_Steel,',
         '          Pipe, 0.023, 0.004)',
         '</think>'])

# Message 3: system — FEA result
msg_box(ax, X_LEFT, 5.95, BOX_W, 1.35, "system", C_SYS,
        ["STRUCTURAL ANALYSIS RESULT:",
         "Mass: 229.89 kg",
         "FOS (buckling): 0.73  ← still infeasible",
         "Status: INFEASIBLE ✗"])

# Message 4: assistant — second action
msg_box(ax, X_LEFT, 4.45, BOX_W, 1.35, "assistant", C_AST,
        ['<think>Modification: Increase member 6',
         '        thickness by 63%',
         '  Action: SCALE_PARAM(6, thickness, 1.631)',
         '</think>'])

ax.text(X_LEFT + BOX_W / 2, 4.15, "⋯  (more turns)  ⋯",
        ha="center", fontsize=9, color=GREY)

# Final assistant — answer
msg_box(ax, X_LEFT, 2.55, BOX_W, 1.45, "assistant", C_ANS,
        ['<answer>',
         '  Optimal feasible solution found via LP',
         '</answer>',
         'Final: Mass=247.9 kg, FOS_buck=1.52 ✓'])


# ── RIGHT column: Transformed messages ───────────────────────────────────────
ax.text(X_RIGHT + BOX_W / 2, 12.55,
        "Transformed (warmstart training format)", ha="center", fontsize=11,
        fontweight="bold", color=DARK)
ax.text(X_RIGHT + BOX_W / 2, 12.25,
        "WarmstartReasoningTarget.transform_example()", ha="center", fontsize=8.5,
        color=C_HOOK, fontstyle="italic")

# Message 0 (out): system — merged
msg_box(ax, X_RIGHT, 10.10, BOX_W, 1.95, "system", C_SYS,
        ["PROBLEM: Truss optimization (variant 79)",
         "INITIAL STRUCTURE: 8 joints, 13 members...",
         "",
         "INITIAL STATE ANALYSIS:  ← merged here",
         "Mass: 213.21 kg · FOS_buck: 0.73 · INFEASIBLE"],
        annotation="① merge")

# Message 1 (out): assistant — restructured think+action
msg_box(ax, X_RIGHT, 8.45, BOX_W, 1.50, "assistant", C_AST,
        ["<think>",
         "  Add member connecting joints 1 and 6",
         "</think>",
         "<action>ADD_MEMBER(1, 6, A36_Steel, Pipe, 0.023, 0.004)</action>"],
        annotation="② extract")

# Message 2 (out): user — remapped from system
msg_box(ax, X_RIGHT, 7.00, BOX_W, 1.30, "user", C_USR,
        ["[Simulation Result]",
         "Mass: 229.89 kg · FOS_buck: 0.73",
         "Status: INFEASIBLE ✗"],
        annotation="③ remap")

# Message 3 (out): assistant — second turn
msg_box(ax, X_RIGHT, 5.50, BOX_W, 1.35, "assistant", C_AST,
        ["<think>",
         "  Increase member 6 thickness by 63%",
         "</think>",
         "<action>SCALE_PARAM(6, thickness, 1.631)</action>"],
        annotation="② extract")

ax.text(X_RIGHT + BOX_W / 2, 5.20, "⋯  (more turns)  ⋯",
        ha="center", fontsize=9, color=GREY)

# Final (out): assistant — answer (unchanged)
msg_box(ax, X_RIGHT, 2.55, BOX_W, 1.45, "assistant", C_ANS,
        ['<answer>',
         '  Optimal feasible solution found via LP',
         '</answer>',
         '← unchanged (pass-through)'])


# ── CENTER: Transform annotations ────────────────────────────────────────────
cx = (X_LEFT + BOX_W + X_RIGHT) / 2  # 9.05

# Main transform box
transform_box = FancyBboxPatch((cx - 0.75, 5.5), 1.5, 4.5,
                                boxstyle="round,pad=0.1,rounding_size=0.2",
                                fc=C_HOOK + "15", ec=C_HOOK, lw=1.8, zorder=3)
ax.add_patch(transform_box)
ax.text(cx, 7.75, "Warmstart\nTransform",
        ha="center", va="center", fontsize=8.5, color=C_HOOK,
        fontweight="bold", multialignment="center", zorder=4)

# Step labels in center column
steps = [
    (9.65, "① Merge\nsys[0]+sys[1]"),
    (8.60, "② Extract\naction→<action>"),
    (7.35, "③ sys→user\n+ [Sim Result]"),
    (6.20, "② Extract\naction→<action>"),
]
for (y, label) in steps:
    ax.text(cx, y, label, ha="center", va="center", fontsize=7,
            color=C_HOOK, fontweight="bold", multialignment="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=C_HOOK + "88", lw=0.8),
            zorder=5)

# Arrows left→right
arrow_pairs = [
    # (left_y, right_y)
    (11.35, 11.08),  # merge: two system → one
    (11.35, 11.08),
    (8.15,  9.20),   # ast[0] → think+action
    (6.625, 7.65),   # sys[1] → user
    (4.825, 6.175),  # ast[1] → think+action
    (3.275, 3.275),  # answer passthrough
]

# Draw the main horizontal arrow
ax.annotate("", xy=(X_RIGHT - 0.05, 8.0), xytext=(X_LEFT + BOX_W + 0.05, 8.0),
            arrowprops=dict(arrowstyle="->, head_width=0.25, head_length=0.15",
                            color=C_HOOK, lw=2.5), zorder=6)

# ── Role legend ───────────────────────────────────────────────────────────────
legend_y = 1.8
ax.text(0.3, legend_y + 0.35, "Message roles:", fontsize=9, color=DARK, fontweight="bold")
for i, (color, label) in enumerate([
    (C_SYS, "system  (problem spec / FEA feedback)"),
    (C_AST, "assistant  (model's reasoning + action)"),
    (C_USR, "user  (remapped from system in output)"),
    (C_ANS, "assistant + <answer>  (final solution)"),
]):
    patch = FancyBboxPatch((0.3 + i * 4.3, legend_y - 0.05), 0.45, 0.30,
                           boxstyle="round,pad=0.04", fc=color, ec=WHITE, lw=1.0, zorder=3)
    ax.add_patch(patch)
    ax.text(0.85 + i * 4.3, legend_y + 0.10, label,
            fontsize=8, color=DARK, va="center")

# Key insight callout
ax.text(9.0, 1.20,
        "Key insight: supervised loss applies to <think>, <action>, and <answer> tokens only — "
        "not to system/user prompt tokens",
        ha="center", fontsize=8.5, color=C_HOOK, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc=C_HOOK + "10", ec=C_HOOK, lw=1.0))

ax.text(9.0, 0.55,
        "llm_finetune/data/processors/warmstart_transform.py  ·  "
        "llm_finetune/training/sft/targets.py  ·  "
        "llm_finetune/data/collators.py",
        ha="center", fontsize=8, color=GREY, fontstyle="italic")

# ── Save ─────────────────────────────────────────────────────────────────────
out = "/ocean/projects/mch250030p/wxu7/llm_finetune/figures/warmstart_pipeline.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
