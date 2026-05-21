"""
Generate ablation-study PPT: §6.3 reward decomposition training results.
Each slide: title + result figure (plot or rendered equation PNG).
White background. Equations rendered via matplotlib mathtext → embedded PNG.
"""

import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
LOGS = PROJECT / "logs"
OUT = PROJECT / "docs" / "ablation_training_results.pptx"

# ── Ablation run registry ──────────────────────────────────────────────────────
RUNS = [
    ("abl01a_base_a2",        "r_env α=2",        "grpo_abl_01a_baseline_alpha2",    "#1f77b4"),
    ("abl01b_base_a5",        "r_env α=5",        "grpo_abl_01b_baseline_alpha5",    "#ff7f0e"),
    ("abl01c_base_a10",       "r_env α=10",       "grpo_abl_01c_baseline_alpha10",   "#2ca02c"),
    ("abl02_tree",            "+tree expansion",  "grpo_abl_02_tree_expansion",      "#d62728"),
    ("abl04_patterns",        "+pattern signals", "grpo_abl_04_pattern_signals",     "#9467bd"),
    ("abl05_llm_cap",         "+LLM capability",  "grpo_abl_05_llm_capability",      "#8c564b"),
    ("abl06_reasoning",       "+reasoning grnd",  "grpo_abl_06_reasoning_grounding", "#e377c2"),
    ("abl07_full_posterior",  "full posterior",   "grpo_posterior",                  "#17becf"),
]
SUFFIX = "_20260519_1537"

# ── Style ──────────────────────────────────────────────────────────────────────
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
BLACK  = RGBColor(0x00, 0x00, 0x00)
GRAY   = RGBColor(0x55, 0x55, 0x55)
ACCENT = RGBColor(0x1F, 0x77, 0xB4)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ── Data loading ───────────────────────────────────────────────────────────────
def load_metrics(slug):
    slug_name = f"qwen3_14b_{slug}{SUFFIX}"
    path = LOGS / slug_name / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "train" and "reward" in r and "train_loss" not in r:
            rows.append(r)
    return rows

ALL_DATA = {slug: load_metrics(slug) for slug, *_ in RUNS}

# ── Helpers ────────────────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


def eq_to_bytes(tex, fontsize=22, pad=0.3):
    """Render a LaTeX-style math string via matplotlib mathtext → PNG bytes."""
    fig, ax = plt.subplots(figsize=(10, 1.2))
    ax.axis("off")
    ax.text(0.5, 0.5, f"${tex}$", transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize,
            fontfamily="DejaVu Sans")
    fig.patch.set_facecolor("white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=pad)
    buf.seek(0)
    plt.close(fig)
    return buf


def multi_eq_to_bytes(lines, fontsize=18, line_spacing=1.6):
    """Render multiple equation/text lines stacked vertically.
    Lines starting with '#' are rendered as plain text (no $ wrapping).
    """
    n = len(lines)
    fig = plt.figure(figsize=(12, n * line_spacing))
    for i, tex in enumerate(lines):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.axis("off")
        if tex.startswith("#"):
            ax.text(0.5, 0.5, tex[1:], transform=ax.transAxes,
                    ha="center", va="center", fontsize=fontsize - 2,
                    color="#444444")
        else:
            ax.text(0.5, 0.5, f"${tex}$", transform=ax.transAxes,
                    ha="center", va="center", fontsize=fontsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf


def add_slide(prs, title_text):
    layout = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(layout)
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE

    # title bar
    txb = slide.shapes.add_textbox(Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.7))
    tf = txb.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = title_text
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = BLACK

    # thin rule
    from pptx.util import Pt as PtU
    from pptx.oxml.ns import qn
    from lxml import etree
    ln = slide.shapes.add_connector(1,
        Inches(0.4), Inches(0.85), Inches(12.93), Inches(0.85))
    ln.line.color.rgb = ACCENT
    ln.line.width = Pt(1.5)

    return slide


def place_image(slide, buf, left, top, width=None, height=None):
    buf.seek(0)
    pic = slide.shapes.add_picture(buf, left, top, width=width, height=height)
    return pic


def add_caption(slide, text, left, top, width, fontsize=10):
    txb = slide.shapes.add_textbox(left, top, width, Inches(0.35))
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(fontsize)
    run.font.color.rgb = GRAY


def steps(data):
    return [r["step"] for r in data]

def vals(data, key):
    return [r[key] for r in data]

# ══════════════════════════════════════════════════════════════════════════════
# Build presentation
# ══════════════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H


# ── Slide 1: Title ─────────────────────────────────────────────────────────────
slide = add_slide(prs, "")
bg = slide.background.fill
bg.solid()
bg.fore_color.rgb = WHITE

txb = slide.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(11), Inches(1.2))
tf = txb.text_frame
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = "§6.3 Ablation Study — Training Diagnostics"
r.font.size = Pt(36)
r.font.bold = True
r.font.color.rgb = BLACK

txb2 = slide.shapes.add_textbox(Inches(1.2), Inches(3.4), Inches(11), Inches(0.6))
tf2 = txb2.text_frame
p2 = tf2.paragraphs[0]
p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run()
r2.text = "qwen3-14B + LoRA r=32 · 2×H100-80GB · 50 steps · 2026-05-19"
r2.font.size = Pt(18)
r2.font.color.rgb = GRAY

txb3 = slide.shapes.add_textbox(Inches(1.2), Inches(4.1), Inches(11), Inches(0.5))
tf3 = txb3.text_frame
p3 = tf3.paragraphs[0]
p3.alignment = PP_ALIGN.CENTER
r3 = p3.add_run()
r3.text = "Theory: docs/plan_posterior_reward_walkthrough.md"
r3.font.size = Pt(14)
r3.font.color.rgb = GRAY


# ── Slide 2: MDP Setup ─────────────────────────────────────────────────────────
slide = add_slide(prs, "§0 — Truss Design MDP")

eqs = [
    r"\mathcal{M} = (\mathcal{S},\, \mathcal{A},\, T,\, r,\, \gamma,\, H)"
    r"\qquad T:\mathcal{S}\times\mathcal{A}\to\mathcal{S},\quad H\leq 9,\quad \gamma=0.99",
    r"\min_{\pi}\; m(s_H)\quad\mathrm{s.t.}\quad"
    r"\mathrm{FOS}_b(s_H)\geq 1.5,\quad"
    r"\mathrm{FOS}_y(s_H)\geq 1.5,\quad"
    r"\delta(s_H)\leq 0.01\,\mathrm{m}",
    r"f(s)=\mathbb{1}\left["
    r"\mathrm{FOS}_b(s)\geq 1.5\;\wedge\;"
    r"\mathrm{FOS}_y(s)\geq 1.5\;\wedge\;"
    r"\delta(s)\leq 0.01\right]",
]
buf = multi_eq_to_bytes(eqs, fontsize=17, line_spacing=1.5)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(12.5))

# state feature mini-table as figure
fig, ax = plt.subplots(figsize=(10, 2.4))
ax.axis("off")
rows_t = [
    ["fos_b_ratio", r"$\mathrm{FOS}_b / 1.5$"],
    ["fos_y_ratio", r"$\mathrm{FOS}_y / 1.5$"],
    ["defl_ratio",  r"$\delta / 0.01\,\mathrm{m}$"],
    ["mass_ratio",  r"$m / m_0$"],
    ["is_feasible", r"$f(s)\in\{0,1\}$"],
    ["buckling_slack", r"$\max(0,1.5-\mathrm{FOS}_b)$"],
    ["dv_*",        r"one-hot dominant violation"],
    ["n_members",   "member count"],
    ["depth_norm",  r"$t/\bar H$"],
]
tbl = ax.table(
    cellText=rows_t,
    colLabels=["Feature", "Formula"],
    loc="center", cellLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.3)
for (r_i, c_i), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r_i == 0:
        cell.set_facecolor("#e8f0fb")
    else:
        cell.set_facecolor("white")
fig.patch.set_facecolor("white")
buf2 = fig_to_bytes(fig)
place_image(slide, buf2, Inches(0.4), Inches(3.8), width=Inches(7.5))

add_caption(slide,
    "State vector φ(s) ∈ ℝ¹⁴ (features.py:STATE_FEATURE_NAMES). "
    "Used in dead-end avoidance (§2.4) and forward-prediction grounding (§4.1).",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 3: r_env — Lagrangian Potential ──────────────────────────────────────
slide = add_slide(prs, "§1 — r_env: Lagrangian Potential Reward")

eqs = [
    r"V(s) := \mathrm{sp}(1.5-\mathrm{FOS}_b) + \mathrm{sp}(1.5-\mathrm{FOS}_y)"
    r" + \mathrm{sp}\!\left(\frac{\delta}{0.01}-1\right),\qquad \mathrm{sp}(x):=\log(1+e^x)",
    r"\Phi(s) := \log\!\left(\frac{m_0}{m(s)}\right) - \alpha \cdot V(s)"
    r"\qquad \alpha=5 \;(\mathrm{mass\,utility} - \alpha \times \mathrm{constraint\,violation})",
    r"r_{\mathrm{env}}(s,a,s') = \gamma\,\Phi(s') - \Phi(s)"
    r"\qquad [\mathrm{NHR\;potential\;shaping}]",
    r"\sum_{t=0}^{H-1}\gamma^t\,r_{\mathrm{env},t}"
    r" = \gamma^H\Phi(s_H)-\Phi(s_0)"
    r"\quad\Rightarrow\quad \mathrm{optimal\;policy\;maximises\;}\Phi(s_H)",
]
buf = multi_eq_to_bytes(eqs, fontsize=16, line_spacing=1.6)
place_image(slide, buf, Inches(0.4), Inches(1.0), width=Inches(12.5))

add_caption(slide,
    "NHR potential-based shaping (Ng, Harada & Russell, ICML 1999). "
    "Single hyperparameter α (Lagrange multiplier on feasibility). "
    "Telescoping property: no terminal bonus needed. "
    "Dense, deterministic, policy-agnostic (~10 ms/step via trussme FEA).",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 4: §2 Pattern Signals ────────────────────────────────────────────────
slide = add_slide(prs, "§2 — Modification Tree Signals: Macros & Dead-End Avoidance")

eqs = [
    r"r_{\mathrm{macro}}(s,a,t) = \mathbb{1}[(a_{t-k+1},\ldots,a_t)"
    r"\;\mathrm{matches\;macro}\;M] \cdot w_M,\qquad \beta_2=0.05",
    r"r_{\mathrm{dead}}(s,a) = \mathbb{1}\left["
    r"\exists\,(s',a')\in\mathcal{D}_{\mathrm{dead}}:\;"
    r"\mathrm{class}(a)=\mathrm{class}(a'),\;"
    r"\|\varphi(s)-\varphi(s')\| < \rho_{nn}\right],\qquad \xi=0.20",
]
buf = multi_eq_to_bytes(eqs, fontsize=17, line_spacing=1.8)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(12.5))

# channel table
fig, ax = plt.subplots(figsize=(11, 1.8))
ax.axis("off")
rows_t = [
    ["SFT pretraining", "Action-class KL (§2.1), sibling-contrast DPO (§2.2)",
     "Gold distribution — no covariate shift"],
    ["GRPO-time pattern", "Macros r_macro (§2.3), dead-end r_dead (§2.4)",
     "Self-gating: silent on OOD states, never misleads"],
]
tbl = ax.table(cellText=rows_t,
               colLabels=["Channel", "Mechanism", "Why robust"],
               loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.5)
for (r_i, c_i), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    cell.set_facecolor("#e8f0fb" if r_i == 0 else "white")
fig.patch.set_facecolor("white")
buf2 = fig_to_bytes(fig)
place_image(slide, buf2, Inches(0.4), Inches(3.8), width=Inches(12.4))

add_caption(slide,
    "ρ_nn = 5th-percentile of within-gold pairwise φ-distances (data constant, not a YAML param). "
    "State-conditioned regressors on gold data are NOT used at GRPO time (§2 rationale).",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 5: §3 Tree Expansion Estimator ───────────────────────────────────────
slide = add_slide(prs, "§3 — Online Tree-Expanded Value Estimation")

eqs = [
    r"\hat{V}^{(D,b)}(s,a) := r(s,a,T(s,a)) + \gamma\cdot\tilde{V}^{(D-1,b)}(T(s,a))",
    r"\tilde{V}^{(D,b)}(s) := \max_{a'\in\mathcal{A}_b(s)}\hat{V}^{(D,b)}(s,a')"
    r"\quad(D\geq 1);\qquad \tilde{V}^{(0,b)}(s) := V^G(s)\;[\mathrm{greedy\;continuation}]",
    r"A^{(D,b)}(s_t,a_t) = \hat{V}^{(D,b)}(s_t,a_t)"
    r" - \frac{1}{K}\sum_{j=1}^{K}\hat{V}^{(D,b)}(s_t,a_j^{(t)})"
    r"\quad[\mathrm{replaces\;group\;baseline}]",
]
buf = multi_eq_to_bytes(eqs, fontsize=16, line_spacing=1.7)
place_image(slide, buf, Inches(0.4), Inches(1.0), width=Inches(7.8))

# cost table
fig, ax = plt.subplots(figsize=(4.5, 2.2))
ax.axis("off")
rows_t = [
    ["No tree (D=0)", "4", "36", "0.36 s"],
    ["D=1, b=5", "24", "216", "2.16 s"],
    ["D=2, b=5,3", "64", "576", "5.76 s"],
    ["D=3, b=5,3,3", "184", "1656", "16.6 s"],
]
tbl = ax.table(cellText=rows_t,
               colLabels=["Config", "FEA/step", "FEA/rollout", "Time (10ms)"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.4)
for (r_i, c_i), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r_i == 0:
        cell.set_facecolor("#e8f0fb")
    elif r_i == 2:
        cell.set_facecolor("#fff3e0")  # highlight D=1 default
    else:
        cell.set_facecolor("white")
fig.patch.set_facecolor("white")
buf2 = fig_to_bytes(fig)
place_image(slide, buf2, Inches(8.5), Inches(1.0), width=Inches(4.4))

add_caption(slide,
    "Default: D=1, b=5 (stratified across 5 action classes). "
    "Per-step advantage replaces GRPO's trajectory-level group-relative baseline. "
    "V^G = greedy-on-Φ continuation to horizon H.",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 6: §3 Bias & Ranking Bounds ─────────────────────────────────────────
slide = add_slide(prs, "§3 — Theorem 1 (Bias) & Theorem 2 (Ranking)")

eqs = [
    r"\mathbf{Theorem\;1\;(Bias):}\quad"
    r"\left|\hat{V}^{(D,b)}(s,a)-V^*(s,a)\right| \leq \gamma^{D+1}\cdot\bar{\Delta}_G"
    r"\qquad[\gamma=0.99,\;\bar{\Delta}_G\approx 0.2\;\Rightarrow\;\mathrm{bound}\approx 0.196]",
    r"\mathbf{Theorem\;2\;(Ranking):}\quad"
    r"\Delta_{\mathrm{rank}} > 2\,\gamma^{D+1}\,\sigma_G(s)"
    r"\;\Rightarrow\;\mathrm{ranking\;preserved}"
    r"\qquad[\sigma_G\approx 0.05\;\Rightarrow\;\mathrm{threshold}\approx 0.098]",
    r"\mathbf{Theorem\;3\;(Coverage):}\quad"
    r"\Pr[\mathrm{miss\;optimal\;class}] = 1 - \frac{b}{|\mathcal{A}_{\mathrm{class}}|}"
    r"\;\rightarrow\; 0 \;\mathrm{with\;stratified}\;b=5",
]
buf = multi_eq_to_bytes(eqs, fontsize=15, line_spacing=1.6)
place_image(slide, buf, Inches(0.4), Inches(1.0), width=Inches(12.5))

# numerical tables side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.0))
for ax in (ax1, ax2):
    ax.axis("off")

rows1 = [["0","0.198"],["1","0.196"],["2","0.194"],["3","0.192"]]
t1 = ax1.table(cellText=rows1, colLabels=["D", "Bias ≤ γ^(D+1)·ΔG"],
               loc="center", cellLoc="center")
t1.auto_set_font_size(False); t1.set_fontsize(11); t1.scale(1,1.4)
ax1.set_title("Bias bound (γ=0.99, ΔG≈0.2)", fontsize=10, pad=4)

rows2 = [["0","0.099"],["1","0.098"],["2","0.097"],["3","0.096"]]
t2 = ax2.table(cellText=rows2, colLabels=["D", "Δrank needed"],
               loc="center", cellLoc="center")
t2.auto_set_font_size(False); t2.set_fontsize(11); t2.scale(1,1.4)
ax2.set_title("Ranking threshold (σG≈0.05)", fontsize=10, pad=4)

for t in (t1, t2):
    for (r_i, c_i), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_facecolor("#e8f0fb" if r_i == 0 else "white")

fig.patch.set_facecolor("white")
buf2 = fig_to_bytes(fig)
place_image(slide, buf2, Inches(1.5), Inches(4.5), width=Inches(9.5))

add_caption(slide,
    "Key insight: benefit of depth is ranking accuracy, not absolute bias reduction. "
    "Bias shrinks slowly with depth (γ≈1); variance σG across continuations is what limits ranking. "
    "D=0 sufficient for action-class ranking (Δrank~0.2); D=1 helps within-class (Δrank~0.05–0.15).",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 7: §4 LLM-Capability Terms ──────────────────────────────────────────
slide = add_slide(prs, "§4 — LLM-Capability Shaping Terms")

eqs = [
    r"r_{\mathrm{pred}}(s,a,s')=\exp\!\left(-\frac{\|\hat{\varphi}(s')-\varphi(s')\|_2^2}"
    r"{2\sigma_\varphi^2}\right),\quad\mu=0.10\quad(\S4.1\;\mathrm{forward\;prediction})",
    r"\tilde{R}(\tau)=R(\tau)\cdot(1+\lambda\,d(s_0)),\quad"
    r"d(s_0)=\max(1-\mathrm{FOS}_{\min}/1.5,\,0)+\max(\delta/0.01-1,\,0)"
    r"\quad\lambda=0.50\quad(\S4.2\;\mathrm{difficulty\;weight})",
    r"r_{\mathrm{escape}}=\mathbb{1}[\,\mathrm{stag}(t)\wedge\mathrm{improved}(t)"
    r"\wedge\mathrm{class}(a_t)\neq\mathrm{class}_{\mathrm{stag}}]"
    r"\quad\kappa=0.30\quad(\S4.3\;\mathrm{stagnation\;escape})",
    r"\rho_{\mathrm{adapt}}=\frac{1}{B}\sum_{b=1}^{B}"
    r"D_{\mathrm{KL}}\left(p(\mathrm{a\text{-}class}\mid b)\,\|\,p(\mathrm{a\text{-}class})\right)"
    r"\quad\nu=0.05\quad(\S4.4,\;\mathrm{not\;yet\;wired})",
]
buf = multi_eq_to_bytes(eqs, fontsize=14, line_spacing=1.5)
place_image(slide, buf, Inches(0.4), Inches(1.0), width=Inches(12.5))

add_caption(slide,
    "§4.1 forces causal CoT–action coupling (vague post-hoc reasoning cannot earn r_pred). "
    "§4.3 stagnation escape requires action-class switch, not just any improvement. "
    "§4.4 ρ_adapt not yet wired (requires batch-level view across B problems).",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 8: §6.1 Full Reward Function ────────────────────────────────────────
slide = add_slide(prs, "§6.1 — Full Reward Function & Policy Gradient")

eqs = [
    r"r_{\mathrm{step}} = [\gamma\Phi(s')-\Phi(s)]_{\,r_{\mathrm{env}}\;\S1}"
    r"+\beta_2\,r_{\mathrm{macro}}-\xi\,r_{\mathrm{dead}}"
    r"+\mu\,r_{\mathrm{pred}}+\kappa\,r_{\mathrm{escape}}",
    r"R(\tau)=(1+\lambda\,d(s_0))\cdot(1+\nu\,\rho_{\mathrm{adapt}})"
    r"\cdot\sum_{t=0}^{H-1}\gamma^t\,r_{\mathrm{step},t}",
    r"\nabla J(\theta)=\mathbb{E}_{\tau}\!\left["
    r"(1+\lambda\,d(s_0))(1+\nu\,\rho_{\mathrm{adapt}})"
    r"\sum_{t=0}^{H-1}A^{(D,b)}(s_t,a_t)\,\nabla\log\pi_\theta(a_t\mid s_t)\right]",
]
buf = multi_eq_to_bytes(eqs, fontsize=15, line_spacing=1.8)
place_image(slide, buf, Inches(0.4), Inches(1.0), width=Inches(12.5))

# coefficient table
fig, ax = plt.subplots(figsize=(10, 2.0))
ax.axis("off")
rows_t = [
    ["α=5","γ=0.99","D=1","b=5","β₂=0.05","ξ=0.20","μ=0.10","κ=0.30","λ=0.50","ν=0.05"],
    ["§1.1","§1.2","§3.5","§3.4","§2.3","§2.4","§4.1","§4.3","§4.2","§4.4"],
]
tbl = ax.table(cellText=rows_t,
               colLabels=["α","γ","D","b","β₂","ξ","μ","κ","λ","ν"],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1,1.4)
for (r_i, c_i), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    cell.set_facecolor("#e8f0fb" if r_i == 0 else "white")
fig.patch.set_facecolor("white")
buf2 = fig_to_bytes(fig)
place_image(slide, buf2, Inches(1.5), Inches(5.1), width=Inches(10))

add_caption(slide,
    "Trajectory multipliers (1+λd)(1+νρ_adapt) act as global gradient scaling; "
    "per-step tree advantage A^(D,b) handles within-trajectory credit assignment.",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 9: §6.3 Ablation Map ─────────────────────────────────────────────────
slide = add_slide(prs, "§6.3 — Ablation Sequence → This Experiment")

fig, ax = plt.subplots(figsize=(12, 4.0))
ax.axis("off")
rows_t = [
    ["1a/b/c", "r_env α∈{2,5,10}", "§1", "abl01a/b/c", "COMPLETED ✓"],
    ["2", "+tree expansion D=1,b=5", "§3", "abl02", "COMPLETED ✓"],
    ["3", "+SFT priors (KL + DPO)", "§2.1/2.2", "—", "PENDING (needs SFT ckpt)"],
    ["4", "+pattern signals r_macro, r_dead", "§2.3/2.4", "abl04", "COMPLETED ✓"],
    ["5", "+LLM-capability r_escape, diff. weight", "§4.2/4.3", "abl05", "COMPLETED ✓"],
    ["6", "+reasoning grounding r_pred", "§4.1", "abl06", "COMPLETED ✓"],
    ["7", "full posterior reward", "§6.1", "abl07", "COMPLETED ✓"],
]
tbl = ax.table(cellText=rows_t,
               colLabels=["Step","Component added","Theory §","Run slug","Status"],
               loc="center", cellLoc="left")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1,1.5)
for (r_i, c_i), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r_i == 0:
        cell.set_facecolor("#e8f0fb")
    elif r_i == 4:  # step 3 not run
        cell.set_facecolor("#fff9e6")
    else:
        cell.set_facecolor("white")
fig.patch.set_facecolor("white")
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.4), Inches(1.0), width=Inches(12.4))

add_caption(slide,
    "All runs: qwen3_14b + LoRA r=32, ALLOW_BASE_GRPO=1 (no SFT warmstart), 50 steps, cosine LR from 5e-7. "
    "Step 3 requires warmstart_sft checkpoint (job 40929020 completed separately).",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 10: Step 1 — α Sweep Reward ─────────────────────────────────────────
slide = add_slide(prs, "Step 1 — Baseline: r_env Only, α Sweep")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for slug, label, _, color in RUNS[:3]:
    d = ALL_DATA[slug]
    if not d: continue
    s = steps(d); r = vals(d, "reward"); rstd = vals(d, "reward_std")
    ax1.plot(s, r, "o-", color=color, label=label, linewidth=2, markersize=5)
    ax1.fill_between(s,
        [rv - sv*0.3 for rv, sv in zip(r, rstd)],
        [rv + sv*0.3 for rv, sv in zip(r, rstd)],
        alpha=0.12, color=color)

ax1.set_xlabel("Training step"); ax1.set_ylabel("Mean reward (r_env scale)")
ax1.set_title("Reward vs Step"); ax1.legend(fontsize=9)
ax1.set_xlim(0, 55)

for slug, label, _, color in RUNS[:3]:
    d = ALL_DATA[slug]
    if not d: continue
    s = steps(d)
    snr = [r_/rs_ if rs_ > 0 else 0 for r_, rs_ in zip(vals(d,"reward"), vals(d,"reward_std"))]
    ax2.plot(s, snr, "s--", color=color, label=label, linewidth=1.5, markersize=5)

ax2.axhline(1.0, color="gray", linewidth=1, linestyle=":", alpha=0.7, label="SNR=1")
ax2.set_xlabel("Training step"); ax2.set_ylabel("SNR = reward / reward_std")
ax2.set_title("Signal-to-Noise Ratio"); ax2.legend(fontsize=9)
ax2.set_xlim(0, 55)

plt.tight_layout()
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(12.5))

add_caption(slide,
    "Shaded band = ±0.3 reward_std. All three runs converge to similar final reward (~20–22). "
    "Consistent dip at step 20 across all runs — driven by LR schedule and data sampling, not α. "
    "SNR < 1 at step 20 (reward variance exceeds mean); recovers by step 40.",
    Inches(0.4), Inches(6.5), Inches(12))


# ── Slide 11: Step 2 — +Tree Expansion ────────────────────────────────────────
slide = add_slide(prs, "Step 2 — +Tree Expansion D=1, b=5")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

compare = [("abl01b_base_a5", "r_env α=5 (baseline)", "#ff7f0e"),
           ("abl02_tree",    "+tree expansion",       "#d62728")]
for slug, label, color in compare:
    d = ALL_DATA[slug]
    if not d: continue
    s = steps(d); r = vals(d, "reward")
    ax1.plot(s, r, "o-", color=color, label=label, linewidth=2, markersize=5)

ax1.set_xlabel("Training step"); ax1.set_ylabel("Reward")
ax1.set_title("r_env α=5 vs +Tree Expansion"); ax1.legend(fontsize=10)
ax1.set_xlim(0, 55)

# KL comparison
for slug, label, color in compare:
    d = ALL_DATA[slug]
    if not d: continue
    s = steps(d); kl = vals(d, "kl")
    ax2.plot(s, [k * 1000 for k in kl], "s-", color=color, label=label,
             linewidth=2, markersize=5)

ax2.set_xlabel("Training step"); ax2.set_ylabel("KL from base (×10⁻³)")
ax2.set_title("KL Divergence from Base Model"); ax2.legend(fontsize=10)
ax2.set_xlim(0, 55)

plt.tight_layout()
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(12.5))

eq_buf = eq_to_bytes(
    r"A^{(1,5)}(s_t,a_t)=\hat{V}^{(1,5)}(s_t,a_t)-\frac{1}{K}\sum_j\hat{V}^{(1,5)}(s_t,a_j)"
    r"\quad[\mathrm{replaces\;group\text{-}normalised\;trajectory\;baseline}]",
    fontsize=13)
place_image(slide, eq_buf, Inches(0.4), Inches(6.0), width=Inches(12.5))

add_caption(slide,
    "Tree expansion (abl02) matches α=2 baseline in final reward — tree advantage signal is entering "
    "but not yet separating from the r_env-only baseline at 50 steps. KL nearly identical.",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 12: Step 4 — +Pattern Signals ───────────────────────────────────────
slide = add_slide(prs, "Step 4 — +Pattern Signals (r_macro, r_dead)")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

compare4 = [
    ("abl01b_base_a5",  "r_env α=5",  "#ff7f0e"),
    ("abl02_tree",      "+tree",       "#d62728"),
    ("abl04_patterns",  "+patterns",   "#9467bd"),
]
for ax_i, (metric, ylabel, title) in enumerate([
    ("reward",      "Reward",        "Mean Reward"),
    ("reward_std",  "Reward Std",    "Reward Std (diversity)"),
    ("frac_reward_zero_std", "Frac zero-std", "Zero-Std Groups"),
]):
    ax = axes[ax_i]
    for slug, label, color in compare4:
        d = ALL_DATA[slug]
        if not d: continue
        ax.plot(steps(d), vals(d, metric), "o-", color=color, label=label,
                linewidth=2, markersize=4)
    ax.set_xlabel("Step"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.set_xlim(0, 55)
    if ax_i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(12.5))

add_caption(slide,
    "Pattern signals (abl04) produce no visible separation from tree baseline at 50 steps. "
    "frac_zero_std = 0.15 at steps 30–40 across all runs (same batches — driven by data sampling). "
    "Reward_std grows with training, indicating increasing rollout diversity.",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 13: Steps 5–7 — LLM Capability & Full Posterior ─────────────────────
slide = add_slide(prs, "Steps 5–7 — LLM-Capability Terms & Full Posterior")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# left: steps 5-7 reward (different scale from 1-4)
cap_runs = [("abl05_llm_cap","#8c564b"), ("abl06_reasoning","#e377c2"), ("abl07_full_posterior","#17becf")]
labels5 = {"abl05_llm_cap": "+LLM cap (§4.2/4.3)", "abl06_reasoning": "+reasoning grnd (§4.1)", "abl07_full_posterior": "full posterior"}
for slug, color in cap_runs:
    d = ALL_DATA[slug]
    if not d: continue
    s = steps(d); r = vals(d, "reward"); rs = vals(d, "reward_std")
    ax1.plot(s, r, "o-", color=color, label=labels5[slug], linewidth=2, markersize=5)
    ax1.fill_between(s,
        [rv - sv*0.3 for rv, sv in zip(r, rs)],
        [rv + sv*0.3 for rv, sv in zip(r, rs)],
        alpha=0.1, color=color)

ax1.set_xlabel("Training step"); ax1.set_ylabel("Reward (composite scale)")
ax1.set_title("Steps 5–7: Reward vs Step\n(reward scale ≠ steps 1–4)")
ax1.legend(fontsize=9); ax1.set_xlim(0, 55)

# right: SNR comparison across all 8 runs at final step
final_snr = []
final_labels = []
final_colors = []
for slug, label, _, color in RUNS:
    d = ALL_DATA[slug]
    if not d: continue
    last = d[-1]
    snr = last["reward"] / last["reward_std"] if last["reward_std"] > 0 else 0
    final_snr.append(snr)
    final_labels.append(label)
    final_colors.append(color)

bars = ax2.barh(range(len(final_snr)), final_snr, color=final_colors, edgecolor="white")
ax2.set_yticks(range(len(final_labels)))
ax2.set_yticklabels(final_labels, fontsize=9)
ax2.axvline(1.0, color="gray", linewidth=1, linestyle=":", alpha=0.7)
ax2.set_xlabel("SNR at step 50"); ax2.set_title("Final SNR — All Runs")

plt.tight_layout()
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(12.5))

add_caption(slide,
    "Steps 5–7 reward scale is ~8–9× larger than steps 1–4 (additional reward terms accumulate). "
    "NOT directly comparable across step groups. Final SNR ~1.0–1.7 across all runs.",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 14: Diagnostic — Clip Ratio (all runs) ──────────────────────────────
slide = add_slide(prs, "Diagnostic — PPO Clip Ratio (All Runs, All Steps)")

fig, ax = plt.subplots(figsize=(10, 4.0))
for slug, label, _, color in RUNS:
    d = ALL_DATA[slug]
    if not d: continue
    clip = vals(d, "clip_ratio/high_mean")
    ax.plot(steps(d), clip, "o-", color=color, label=label, linewidth=1.5, markersize=4)

ax.set_xlabel("Training step")
ax.set_ylabel("clip_ratio/high_mean")
ax.set_title("PPO High-Clip Ratio vs Step — All 8 Runs")
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.set_xlim(0, 55)
ax.set_ylim(-0.001, 0.005)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
plt.tight_layout()
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(9.5))

eq_buf = eq_to_bytes(
    r"\mathrm{clip\_ratio/high} = \frac{1}{N}\sum_i"
    r"\mathbb{1}\!\left[\frac{\pi_\theta(a_i|s_i)}{\pi_{\theta_{\mathrm{old}}}(a_i|s_i)}"
    r"> 1+\varepsilon\right]",
    fontsize=14)
place_image(slide, eq_buf, Inches(9.8), Inches(1.8), width=Inches(3.3))

add_caption(slide,
    "clip_ratio/high_mean = 0.0000 at every logged step across all 8 runs (400 data points). "
    "The PPO clipping mechanism is never triggered. Policy ratio π_θ/π_θ_old stays within [1-ε, 1+ε] "
    "for all tokens. Consistent with KL~0.0013 — policy barely moves from base model. "
    "Indicates learning rate may be too small or ε too large for current setup.",
    Inches(0.4), Inches(6.5), Inches(12))


# ── Slide 15: Diagnostic — KL Divergence ──────────────────────────────────────
slide = add_slide(prs, "Diagnostic — KL Divergence from Base Model (All Runs)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for slug, label, _, color in RUNS:
    d = ALL_DATA[slug]
    if not d: continue
    kl = [k * 1000 for k in vals(d, "kl")]
    ax1.plot(steps(d), kl, "o-", color=color, label=label, linewidth=1.5, markersize=4)

ax1.set_xlabel("Training step"); ax1.set_ylabel("KL (×10⁻³)")
ax1.set_title("KL from Base — All 8 Runs")
ax1.legend(fontsize=8, ncol=1); ax1.set_xlim(0, 55)

# LR schedule
d_ref = ALL_DATA["abl01a_base_a2"]
if d_ref:
    lr = vals(d_ref, "learning_rate")
    s  = steps(d_ref)
    ax2.semilogy(s, lr, "k-o", linewidth=2, markersize=5)
    ax2.set_xlabel("Training step"); ax2.set_ylabel("Learning rate (log scale)")
    ax2.set_title("LR Schedule (cosine decay)\nsame across all runs")
    ax2.set_xlim(0, 55)
    for xi, yi in zip(s, lr):
        ax2.annotate(f"{yi:.1e}", (xi, yi), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=7)

plt.tight_layout()
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(12.5))

add_caption(slide,
    "KL plateaus at ~0.0013–0.0015 after step 10 and does not increase further. "
    "LR decays from 4.8×10⁻⁷ to 5.6×10⁻¹⁰ (860× reduction) over 50 steps — "
    "effective training ceases around step 45. Policy at step 50 is not converged; it is stopped.",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 16: Diagnostic — Loss & Training Speed ──────────────────────────────
slide = add_slide(prs, "Diagnostic — Policy Loss & Step Time (All Runs)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for slug, label, _, color in RUNS:
    d = ALL_DATA[slug]
    if not d: continue
    ax1.plot(steps(d), vals(d, "loss"), "o-", color=color, label=label,
             linewidth=1.5, markersize=4)

ax1.set_xlabel("Training step"); ax1.set_ylabel("Policy loss")
ax1.set_title("Policy Loss vs Step")
ax1.legend(fontsize=7, ncol=2); ax1.set_xlim(0, 55)
ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

for slug, label, _, color in RUNS:
    d = ALL_DATA[slug]
    if not d: continue
    step_times = vals(d, "step_time")
    ax2.scatter(steps(d), step_times, color=color, label=label, s=30, alpha=0.7)

ax2.set_xlabel("Training step"); ax2.set_ylabel("Step time (s)")
ax2.set_title("Wall-Clock per Step")
ax2.legend(fontsize=7, ncol=2); ax2.set_xlim(0, 55)

plt.tight_layout()
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(12.5))

add_caption(slide,
    "Negative loss is expected in GRPO (policy gradient minimises negative expected reward). "
    "Loss magnitude increases as policy learns to differentiate. "
    "Step time ~80–94 s/step consistent across all runs (~75 min for 50 steps).",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 17: Diagnostic — frac_zero_std ──────────────────────────────────────
slide = add_slide(prs, "Diagnostic — Fraction of Zero-Std Rollout Groups")

fig, ax = plt.subplots(figsize=(10, 4.2))
for slug, label, _, color in RUNS:
    d = ALL_DATA[slug]
    if not d: continue
    frac = vals(d, "frac_reward_zero_std")
    ax.plot(steps(d), frac, "o-", color=color, label=label, linewidth=1.5, markersize=5)

ax.set_xlabel("Training step"); ax.set_ylabel("Fraction zero-std groups")
ax.set_title("Groups with Zero Reward Variance (no learning signal)")
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.set_xlim(0, 55); ax.set_ylim(-0.01, 0.22)
ax.axhline(0.15, color="red", linewidth=0.8, linestyle=":", alpha=0.6, label="0.15 observed peak")
plt.tight_layout()
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.4), Inches(1.1), width=Inches(9.5))

add_caption(slide,
    "frac_zero_std = 0.0 / 0.05 / 0.15 / 0.15 / 0.05 pattern is IDENTICAL across all 8 runs — "
    "driven by data sampling order, not reward function. 15% of groups at steps 30–40 yield zero "
    "advantage (all K rollouts received equal reward). These samples contribute nothing to the gradient.",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Slide 18: Summary Table ────────────────────────────────────────────────────
slide = add_slide(prs, "Summary — All Runs at Step 50")

fig, ax = plt.subplots(figsize=(12, 3.8))
ax.axis("off")
rows_t = []
for slug, label, _, color in RUNS:
    d = ALL_DATA[slug]
    if not d:
        rows_t.append([label, "—","—","—","—","—","—"])
        continue
    last = d[-1]
    snr = last["reward"] / last["reward_std"] if last["reward_std"] > 0 else 0
    # final training record
    all_rows = [json.loads(l) for l in (LOGS / f"qwen3_14b_{slug}{SUFFIX}" / "metrics.jsonl").read_text().splitlines()]
    final = [r for r in all_rows if r.get("type")=="train" and "train_loss" in r]
    tl = f"{final[0]['train_loss']:.4f}" if final else "—"
    rt = f"{int(final[0]['train_runtime']//60)} min" if final else "—"
    rows_t.append([
        label,
        f"{last['reward']:.1f}",
        f"{last['reward_std']:.1f}",
        f"{snr:.2f}",
        f"{last['kl']*1000:.3f}",
        tl,
        rt,
    ])

tbl = ax.table(
    cellText=rows_t,
    colLabels=["Run","Reward↑","Reward Std","SNR","KL×10⁻³","Train Loss","Runtime"],
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.55)

for (r_i, c_i), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r_i == 0:
        cell.set_facecolor("#e8f0fb")
    else:
        run_slug = RUNS[r_i-1][0] if r_i <= len(RUNS) else None
        color = RUNS[r_i-1][3] if r_i <= len(RUNS) else "#ffffff"
        # light tint of run colour
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        cell.set_facecolor(tuple(0.9 + 0.1*c for c in rgb))

fig.patch.set_facecolor("white")
buf = fig_to_bytes(fig)
place_image(slide, buf, Inches(0.15), Inches(1.1), width=Inches(13.0))

add_caption(slide,
    "Reward scale differs between steps 1–4 (r_env only) and steps 5–7 (composite). "
    "clip_ratio/high = 0.0 and KL < 0.0015 across all runs indicate minimal policy movement. "
    "50 steps = early training; LR reached ~0 before convergence.",
    Inches(0.4), Inches(6.8), Inches(12))


# ── Save ────────────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"Saved: {OUT}")
print(f"Slides: {len(prs.slides)}")
