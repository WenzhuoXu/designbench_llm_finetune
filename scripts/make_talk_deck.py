"""Build the 8-minute DesignBench talk deck (method and results only).

Nine slides, no title/agenda/background. Every slide is rendered to a PNG at
240 dpi and placed full-bleed into the PPTX, so the rendered image is exactly
what the deck displays.

Sources for every number:
  docs/status_report.md, docs/ablation_ledger.md, docs/reward_rho_connection.md,
  docs/plan_posterior_reward_walkthrough.md,
  results/eval/*/eval_results.json, results/distill/traj_v{2,3}*.jsonl,
  logs/*/metrics.jsonl

Run:  python scripts/make_talk_deck.py
Out:  docs/DesignBench_8min_Talk.pptx  +  results/report/talk/slide_XX.png
"""
import json, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Circle, Polygon
from pptx import Presentation
from pptx.util import Inches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "results/report/talk")
PPTX = os.path.join(ROOT, "docs/DesignBench_8min_Talk.pptx")
os.makedirs(OUTDIR, exist_ok=True)

W, H = 960.0, 540.0          # points; 13.333 x 7.5 in at 72 pt/in
M = 34.0                     # page margin
DPI = 240

INK    = "#161616"
MUTED  = "#6E6A62"
RULE   = "#CFCABF"
PANEL  = "#F4F2ED"
PANEL2 = "#EAE7DF"
WHITE  = "#FFFFFF"
BLUE   = "#1F4E8C"
BLUEL  = "#DCE4F0"
GREEN  = "#1B6B3F"
GREENL = "#DCEBE1"
RUST   = "#A6401E"
RUSTL  = "#F2E1D9"
GREY   = "#9A958B"

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

FS_TITLE = 26
FS_H     = 16
FS_BODY  = 15
FS_SMALL = 14        # hard floor, nothing smaller anywhere


# --------------------------------------------------------------- primitives --
from fontTools.ttLib import TTFont
from matplotlib.font_manager import findfont, FontProperties

_MET = {}


def _met(fam, weight):
    """Per glyph advance widths in em units, so text width can be measured exactly."""
    key = (fam, weight)
    if key not in _MET:
        path = findfont(FontProperties(family=fam, weight=weight))
        f = TTFont(path, fontNumber=0)
        upm = f["head"].unitsPerEm
        cmap = {}
        for t in f["cmap"].tables:
            cmap.update(t.cmap)
        hm = f["hmtx"]
        adv = {}
        for cp, gn in cmap.items():
            try:
                adv[cp] = hm[gn][0] / upm
            except KeyError:
                pass
        _MET[key] = adv
    return _MET[key]


def twidth(s, size, fam=SANS, weight="normal"):
    adv = _met(fam, weight)
    return size * sum(adv.get(ord(c), 0.60) for c in s)


def wrap_w(s, size, maxw, fam=SANS, weight="normal"):
    """Greedy word wrap by measured width. Honours explicit newlines."""
    out = []
    for para in s.split("\n"):
        words, cur = para.split(), ""
        if not words:
            out.append("")
            continue
        for w_ in words:
            trial = (cur + " " + w_).strip()
            if not cur or twidth(trial, size, fam, weight) <= maxw:
                cur = trial
            else:
                out.append(cur)
                cur = w_
        out.append(cur)
    return out


def fit_size(s, maxw, size, fam=MONO, weight="bold", floor=FS_SMALL):
    """Largest size <= size at which s fits on one line, never below the floor."""
    while size > floor and twidth(s, size, fam, weight) > maxw:
        size -= 0.5
    return size


def new_slide():
    fig = plt.figure(figsize=(W / 72.0, H / 72.0), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, facecolor=WHITE, edgecolor="none", zorder=0))
    return fig, ax


def tx(ax, x, y, s, size=FS_BODY, color=INK, weight="normal", font=SANS,
       ha="left", va="top", lh=1.32, maxw=None, zorder=5, style="normal"):
    """Draw text, wrapping to maxw points if given. Returns the y below the block."""
    lines = wrap_w(s, size, maxw, font, weight) if maxw else s.split("\n")
    for i, ln in enumerate(lines):
        ax.text(x, y + i * size * lh, ln, fontsize=size, color=color,
                fontweight=weight, fontfamily=font, ha=ha, va=va,
                zorder=zorder, fontstyle=style)
    return y + len(lines) * size * lh


def txh(s, size, maxw, font=SANS, weight="normal", lh=1.32):
    """Height a wrapped block will occupy."""
    return len(wrap_w(s, size, maxw, font, weight)) * size * lh


def tx_bottom(ax, x, ybot, s, size=FS_SMALL, maxw=None, font=SANS,
              weight="normal", lh=1.32, **kw):
    """Draw a wrapped block so that it ENDS at ybot. Prevents panel overrun."""
    h = txh(s, size, maxw, font, weight, lh) if maxw else len(s.split("\n")) * size * lh
    return tx(ax, x, ybot - h, s, size=size, maxw=maxw, font=font,
              weight=weight, lh=lh, **kw)


def panel(ax, x, y, w, h, fill=PANEL, edge=None, lw=1.0, zorder=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                           edgecolor=edge if edge else "none",
                           linewidth=lw, zorder=zorder))


def hline(ax, x1, x2, y, color=RULE, lw=1.2, zorder=2, ls="-"):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, zorder=zorder, ls=ls,
            solid_capstyle="butt")


def vline(ax, x, y1, y2, color=RULE, lw=1.2, zorder=2, ls="-"):
    ax.plot([x, x], [y1, y2], color=color, lw=lw, zorder=zorder, ls=ls)


def title(ax, s, sub=None):
    """Assertion title, measured so it can never run off the page."""
    tw = W - 2 * M
    size = FS_TITLE
    while size > 19 and len(wrap_w(s, size, tw, SANS, "bold")) > 2:
        size -= 0.5
    lines = wrap_w(s, size, tw, SANS, "bold")
    y = M + 2
    for i, ln in enumerate(lines):
        ax.text(M, y + i * size * 1.20, ln, fontsize=size, color=INK,
                fontweight="bold", fontfamily=SANS, ha="left", va="top", zorder=5)
    y += len(lines) * size * 1.20 + 3
    if sub:
        y = tx(ax, M, y, sub, size=FS_SMALL, color=MUTED, maxw=tw, lh=1.28) + 3
    hline(ax, M, W - M, y + 4, color=INK, lw=1.6)
    return y + 18


def label(ax, x, y, s, color=MUTED, size=FS_SMALL, upper=True):
    ax.text(x, y, s.upper() if upper else s, fontsize=size, color=color,
            fontweight="bold", fontfamily=SANS, ha="left", va="top", zorder=5)
    return y + size * 1.5


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.6, head=7, zorder=4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=zorder,
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.42",
                                color=color, lw=lw, shrinkA=0, shrinkB=0))


def formula(ax, x, y, w, s, color=INK, size=19, weight="bold"):
    """Monospaced formula, shrunk until it fits the given width."""
    sz = fit_size(s, w, size, MONO, weight)
    ax.text(x, y, s, fontsize=sz, color=color, fontweight=weight,
            fontfamily=MONO, ha="left", va="top", zorder=5)
    return y + sz * 1.30


def truss(ax, ox, oy, sc=1.0, crit=(6, 7), critcolor=RUST):
    """Warren truss, joints J0..J8. crit highlights the top chord member J6-J7."""
    J = {0: (0, 78), 1: (70, 78), 2: (140, 78), 3: (210, 78), 4: (280, 78),
         5: (35, 5), 6: (105, 5), 7: (175, 5), 8: (245, 5)}
    P = lambda i: (ox + J[i][0] * sc, oy + J[i][1] * sc)
    mem = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
           (0, 5), (5, 1), (1, 6), (6, 2), (2, 7), (7, 3), (3, 8), (8, 4)]
    for a, b in mem:
        xa, ya = P(a); xb, yb = P(b)
        hot = (a, b) == crit or (b, a) == crit
        ax.plot([xa, xb], [ya, yb], color=critcolor if hot else "#8A857B",
                lw=5.0 if hot else 2.0, zorder=3, solid_capstyle="round")
    for i in J:
        x, y = P(i)
        ax.add_patch(Circle((x, y), 2.6 * sc, facecolor=INK, edgecolor="none", zorder=4))
    x0, y0 = P(0); x4, y4 = P(4)
    ax.add_patch(Polygon([(x0, y0 + 2), (x0 - 8, y0 + 16), (x0 + 8, y0 + 16)],
                         facecolor="#8A857B", edgecolor="none", zorder=3))
    ax.add_patch(Polygon([(x4, y4 + 2), (x4 - 8, y4 + 14), (x4 + 8, y4 + 14)],
                         facecolor="#8A857B", edgecolor="none", zorder=3))
    for dx in (-5, 5):
        ax.add_patch(Circle((x4 + dx, y4 + 17), 2.4, facecolor="#8A857B",
                            edgecolor="none", zorder=3))
    # load applied downward at the mid span joint, drawn clear of the web
    x2, y2 = P(2)
    arrow(ax, x2, y2 + 8, x2, y2 + 34, color=BLUE, lw=2.0)
    ax.text(x2 + 8, y2 + 24, "P", fontsize=FS_SMALL, color=BLUE,
            fontweight="bold", fontfamily=SANS, va="center", zorder=5)
    xa, ya = P(crit[0]); xb, yb = P(crit[1])
    ax.text((xa + xb) / 2, ya - 9, "M4", fontsize=FS_SMALL, color=critcolor,
            fontweight="bold", fontfamily=SANS, ha="center", va="bottom", zorder=5)


def barchart(ax, x, y, w, h, vals, labels, colors, vmax, valfmt="{:.0f}%",
             valsize=FS_H, labsize=FS_SMALL, gap=0.30, baseline=True):
    n = len(vals)
    pitch = w / n
    bw = pitch * (1 - gap)
    for i, v in enumerate(vals):
        bh = (v / vmax) * h
        cx = x + pitch * (i + 0.5)
        ax.add_patch(Rectangle((cx - bw / 2, y + h - bh), bw, max(bh, 1.2),
                               facecolor=colors[i], edgecolor="none", zorder=3))
        ax.text(cx, y + h - bh - 5, valfmt.format(v), fontsize=valsize,
                color=colors[i] if colors[i] != GREY else MUTED, fontweight="bold",
                fontfamily=MONO, ha="center", va="bottom", zorder=5)
        for j, ln in enumerate(labels[i].split("\n")):
            ax.text(cx, y + h + 8 + j * labsize * 1.28, ln, fontsize=labsize,
                    color=INK, fontfamily=SANS, ha="center", va="top", zorder=5)
    if baseline:
        hline(ax, x, x + w, y + h, color=RULE, lw=1.2)


def quote(ax, x, y, w, s, color=INK, accent=RUST, size=FS_SMALL):
    """Verbatim model text in a ruled block, wrapped to the block width."""
    lines = wrap_w(s, size, w - 24, SANS, "normal")
    h = len(lines) * size * 1.34 + 14
    panel(ax, x, y, w, h, fill=WHITE)
    ax.add_patch(Rectangle((x, y), 3.0, h, facecolor=accent, edgecolor="none", zorder=2))
    yy = y + 7
    for ln in lines:
        ax.text(x + 12, yy, ln, fontsize=size, color=color, fontfamily=SANS,
                ha="left", va="top", zorder=5, fontstyle="italic")
        yy += size * 1.34
    return y + h




# ------------------------------------------------------------------ slide 1 --
def slide1():
    fig, ax = new_slide()
    y0 = title(ax, "One grammar edit per turn; the solver returns a full re-analysis")
    cw = (W - 2 * M - 28) / 3.0
    iw = cw - 28
    rh = 232
    xs = [M, M + cw + 14, M + 2 * (cw + 14)]

    panel(ax, xs[0], y0, cw, rh, fill=PANEL)
    label(ax, xs[0] + 14, y0 + 12, "design state")
    truss(ax, xs[0] + 26, y0 + 56, sc=0.80, crit=(6, 7))
    tx_bottom(ax, xs[0] + 14, y0 + rh - 14,
              "auto_problem_000\n12 members, pipe sections\nFOS buckling 0.44",
              font=MONO, lh=1.42)

    panel(ax, xs[1], y0, cw, rh, fill=PANEL)
    label(ax, xs[1] + 14, y0 + 12, "policy output")
    q = ("The main issue is the buckling failure, especially on member M4. "
         "M4 is J6-J7 with r=0.0380.")
    ql = wrap_w(q, FS_SMALL, iw - 20, SANS)
    bh = 24 + len(ql) * FS_SMALL * 1.30 + 22
    panel(ax, xs[1] + 14, y0 + 40, iw, bh, fill=WHITE)
    tx(ax, xs[1] + 24, y0 + 46, "<think>", color=BLUE, font=MONO, weight="bold")
    yy = y0 + 46 + FS_SMALL * 1.55
    for ln in ql:
        tx(ax, xs[1] + 24, yy, ln, style="italic")
        yy += FS_SMALL * 1.30
    tx(ax, xs[1] + 24, yy + 2, "</think>", color=BLUE, font=MONO, weight="bold")
    ay = y0 + 40 + bh + 14
    panel(ax, xs[1] + 14, ay, iw, 46, fill=INK)
    tx(ax, xs[1] + 24, ay + 6, "<action>", color="#9C978B", font=MONO)
    formula(ax, xs[1] + 24, ay + 23, iw - 20, "SCALE_PARAM(4, r, 1.36)", color=WHITE, size=FS_H)

    panel(ax, xs[2], y0, cw, rh, fill=PANEL)
    label(ax, xs[2] + 14, y0 + 12, "solver response")
    rows = [("[Simulation Result]", RUST, "bold"), ("Mass: 116.73 kg", INK, "normal"),
            ("FOS buckling: 0.48", INK, "normal"), ("FOS yielding: 2.49", INK, "normal"),
            ("Status: INFEASIBLE", INK, "normal"), ("worst member(s): M4", RUST, "bold")]
    bh = len(rows) * FS_SMALL * 1.5 + 16
    panel(ax, xs[2] + 14, y0 + 40, iw, bh, fill=WHITE)
    ax.add_patch(Rectangle((xs[2] + 14, y0 + 40), 3.0, bh, facecolor=RUST, edgecolor="none", zorder=2))
    yy = y0 + 48
    for t_, c, wt in rows:
        ax.text(xs[2] + 26, yy, t_, fontsize=FS_SMALL, color=c, fontweight=wt,
                fontfamily=MONO, ha="left", va="top", zorder=5)
        yy += FS_SMALL * 1.5
    tx_bottom(ax, xs[2] + 14, y0 + rh - 14,
              "critical member, reported each turn",
              color=RUST, weight="bold", maxw=iw)

    yb = y0 + rh + 16
    hb = H - 28 - yb
    hw = (W - 2 * M - 14) / 2.0
    panel(ax, M, yb, hw, hb, fill=PANEL2)
    label(ax, M + 14, yb + 12, "action grammar, six operators")
    for i, o in enumerate(["SCALE_PARAM", "SCALE_MULTI_PARAM", "ADD_MEMBER",
                           "MODIFY_PARAM", "REMOVE_MEMBER", "MOVE_JOINT"]):
        tx(ax, M + 14 + (i % 2) * (hw / 2 - 4), yb + 42 + (i // 2) * 21, o, font=MONO)

    panel(ax, M + hw + 14, yb, hw, hb, fill=PANEL2)
    label(ax, M + hw + 28, yb + 12, "setup")
    tx(ax, M + hw + 28, yb + 42,
       "Feasible:  FOS_b ≥ 1.5,  FOS_y ≥ 1.5,  δ ≤ 0.01 m,  mass ≤ limit\n"
       "Policy:    Qwen3-14B, LoRA r = 32\n"
       "GRPO:      group 8, KL coef 0.04, 100 steps\n"
       "Horizon:   5 turns training, 20 turns evaluation",
       font=MONO, maxw=hw - 42, lh=1.55)
    return fig


# ------------------------------------------------------------------ slide 2 --
def slide2():
    fig, ax = new_slide()
    y0 = title(ax, "Potential-based reward: dense every turn, optimal policy preserved")
    lw, gap = 524.0, 16.0
    rx, rw = M + lw + gap, W - 2 * M - lw - gap
    liw, riw = lw - 28, rw - 28

    blocks = [("1.  potential of a design", ["Φ(s) = log( m_0 / m(s) )  −  α · V(s)"],
               "mass utility − α · violation"),
              ("2.  violation, smooth everywhere",
               ["V(s) = sp(1.5 − FOS_b) + sp(1.5 − FOS_y)", "       + sp(δ/0.01 − 1)"],
               "sp = softplus"),
              ("3.  reward for one turn", ["r(s, a, s′) = γ · Φ(s′)  −  Φ(s)"],
               "dense")]
    yy = y0
    for hd, fl, note in blocks:
        bh = 26 + len(fl) * 25 + txh(note, FS_SMALL, liw, SANS) + 12
        panel(ax, M, yy, lw, bh, fill=PANEL)
        label(ax, M + 14, yy + 10, hd)
        fy = yy + 32
        for f_ in fl:
            fy = formula(ax, M + 14, fy, liw, f_, size=19)
        tx(ax, M + 14, fy + 2, note, color=MUTED, maxw=liw)
        yy += bh + 9

    n4 = "Ng, Harada and Russell 1999: optimal policy unchanged."
    bh = 26 + 25 + txh(n4, FS_SMALL, liw, SANS) + 12
    panel(ax, M, yy, lw, bh, fill=INK)
    label(ax, M + 14, yy + 10, "4.  the return telescopes", color="#A7A196")
    fy = formula(ax, M + 14, yy + 32, liw, "Σ γᵗ r_t  =  γᴴ Φ(s_H)  −  Φ(s_0)", color=WHITE, size=19)
    tx(ax, M + 14, fy + 2, n4, color="#D6D1C7", maxw=liw)

    import math
    ch = 206
    panel(ax, rx, y0, rw, ch, fill=PANEL)
    label(ax, rx + 14, y0 + 10, "violation term V")
    px, py, pw, ph = rx + 40, y0 + 40, rw - 58, ch - 100
    xs_ = [i * 0.05 for i in range(61)]
    vm = math.log(1 + math.exp(1.5))
    X = [px + (v / 3.0) * pw for v in xs_]
    Y = [py + ph - (math.log(1 + math.exp(1.5 - v)) / (vm * 1.05)) * ph for v in xs_]
    ax.fill_between(X, Y, [py + ph] * len(X), color=BLUEL, zorder=2)
    ax.plot(X, Y, color=BLUE, lw=2.6, zorder=3)
    hline(ax, px, px + pw, py + ph, color=RULE)
    xt = px + 0.5 * pw
    ax.plot([xt, xt], [py, py + ph], color=RUST, lw=1.6, ls=(0, (4, 3)), zorder=3)
    ax.add_patch(Circle((xt, py + ph - (math.log(2) / (vm * 1.05)) * ph), 3.6,
                        facecolor=RUST, edgecolor="none", zorder=4))
    tx(ax, xt + 8, py + 2, "FOS = 1.5", color=RUST, font=MONO, weight="bold")
    tx(ax, px + pw / 2, py + ph + 6, "FOS buckling", color=MUTED, ha="center")
    tx_bottom(ax, rx + 14, y0 + ch - 12, "one free parameter:  α = 5\n(γ = 0.99 fixed)",
              font=MONO, maxw=riw, lh=1.45)

    yg = y0 + ch + 14
    hg = H - 28 - yg
    panel(ax, rx, yg, rw, hg, fill=RUSTL)
    label(ax, rx + 14, yg + 10, "what search adds", color=RUST)
    yy = yg + 36
    for it in ["lookahead past a dead end",
               "action class fitted to the state",
               "a reasoning-to-action link"]:
        tx(ax, rx + 14, yy, "–", color=RUST, weight="bold")
        yy = tx(ax, rx + 28, yy, it, maxw=riw - 14, lh=1.30) + 6
    tx_bottom(ax, rx + 14, yg + hg - 12, "→ next slide",
              color=RUST, weight="bold", maxw=riw, lh=1.30)
    return fig


# ------------------------------------------------------------------ slide 3 --
def slide3():
    fig, ax = new_slide()
    y0 = title(ax, "Depth-one lookahead supplies the GRPO baseline")
    lw, gap = 516.0, 16.0
    rx, rw = M + lw + gap, W - 2 * M - lw - gap
    riw = rw - 24
    th = H - 28 - y0

    panel(ax, M, y0, lw, th, fill=PANEL)
    label(ax, M + 14, y0 + 10, "one GRPO step at one state")
    sx, sy = M + 24, y0 + 46
    panel(ax, sx, sy + 84, 70, 42, fill=INK, zorder=3)
    tx(ax, sx + 35, sy + 97, "s(t)", size=FS_H, color=WHITE, font=MONO, weight="bold", ha="center")
    cx = sx + 104
    for i, cy in enumerate([sy + 18, sy + 88, sy + 158]):
        sel = (i == 1)
        panel(ax, cx, cy, 58, 32, fill=BLUE if sel else WHITE, edge=None if sel else RULE, zorder=3)
        tx(ax, cx + 29, cy + 8, f"a{i+1}", color=WHITE if sel else MUTED, font=MONO,
           weight="bold" if sel else "normal", ha="center")
        ax.plot([sx + 70, cx], [sy + 105, cy + 16], color=BLUE if sel else RULE,
                lw=2.2 if sel else 1.4, zorder=2)
    tx(ax, cx + 29, sy + 196, "a4 ... a8", color=MUTED, font=MONO, ha="center")
    bx = cx + 92
    bys = [sy + 2, sy + 44, sy + 86, sy + 128, sy + 170]
    for by, cl in zip(bys, ["SCALE", "ADD", "MODIFY", "REMOVE", "MOVE"]):
        panel(ax, bx, by, 116, 32, fill=BLUEL, edge=BLUE, lw=1.0, zorder=3)
        tx(ax, bx + 58, by + 8, cl, color=INK, font=MONO, ha="center")
        ax.plot([cx + 58, bx], [sy + 104, by + 16], color=BLUE, lw=2.0, zorder=2)
        ax.plot([bx + 116, bx + 152], [by + 16, by + 16], color=RUST, lw=1.4,
                ls=(0, (3, 3)), zorder=2)
    ax.plot([bx + 160, bx + 168, bx + 168, bx + 160],
            [bys[0] + 1, bys[0] + 1, bys[-1] + 31, bys[-1] + 31], color=INK, lw=1.4, zorder=3)
    tx(ax, bx + 176, sy + 88, "lookahead\nvalue", color=INK, lh=1.34)
    tx_bottom(ax, M + 14, y0 + th - 14,
              "one action per class · b = 5 = full grammar\n"
              "dashed: greedy continuation on Φ to the horizon\n"
              "6 solver calls per step · ≈ 2 s per rollout",
              color=MUTED, font=MONO, maxw=lw - 28, lh=1.50)

    yy = y0
    n1 = "D = 1, b = 5; max over five classes"
    bh = 26 + 24 + txh(n1, FS_SMALL, riw, SANS) + 12
    panel(ax, rx, yy, rw, bh, fill=PANEL)
    label(ax, rx + 12, yy + 10, "value estimator")
    fy = formula(ax, rx + 12, yy + 32, riw, "V(s,a) = r + γ · max V(s′,a′)", size=FS_H)
    tx(ax, rx + 12, fy + 2, n1, color=MUTED, maxw=riw)
    yy += bh + 12

    n2 = "a per-step baseline built from the lookahead values"
    bh = 26 + 24 + txh(n2, FS_SMALL, riw, SANS) + 12
    panel(ax, rx, yy, rw, bh, fill=INK)
    label(ax, rx + 12, yy + 10, "advantage GRPO trains on", color="#A7A196")
    fy = formula(ax, rx + 12, yy + 32, riw, "A(s,a_k) = V(s,a_k) − mean V", color=WHITE, size=FS_H)
    tx(ax, rx + 12, fy + 2, n2, color="#D6D1C7", maxw=riw)
    yy += bh + 12

    panel(ax, rx, yy, rw, H - 28 - yy, fill=PANEL)
    label(ax, rx + 12, yy + 10, "diagnostic")
    fy = formula(ax, rx + 12, yy + 32, riw, "ρ = Pr[ argmax π = argmax V ]", size=FS_H)
    tx(ax, rx + 12, fy + 2,
       "ρ = 1.0 throughout the D = 1 run.\n"
       "The trained policy reproduces one-step\n"
       "lookahead on its own, so search confirms\n"
       "the policy it is training.",
       color=INK, maxw=riw, lh=1.45)
    return fig


# ------------------------------------------------------------------ slide 4 --
def slide4():
    fig, ax = new_slide()
    y0 = title(ax, "Ranking accuracy sets the operating point: D = 1, b = 5")
    cw = (W - 2 * M - 28) / 3.0
    iw = cw - 28
    ch = H - 28 - y0 - 44
    xs = [M, M + cw + 14, M + 2 * (cw + 14)]

    panel(ax, xs[0], y0, cw, ch, fill=PANEL)
    label(ax, xs[0] + 14, y0 + 10, "theorem 1   bias")
    formula(ax, xs[0] + 14, y0 + 32, iw, "| V − V* |  ≤  γ^(D+1) · Δ_G", size=FS_H)
    barchart(ax, xs[0] + 22, y0 + 72, cw - 44, 96, [0.198, 0.196, 0.194, 0.192],
             ["D=0", "D=1", "D=2", "D=3"], [BLUE] * 4, 0.25,
             valfmt="{:.3f}", valsize=FS_SMALL)
    tx_bottom(ax, xs[0] + 14, y0 + ch - 14,
              "γ sets the bias bound.\nAt γ = 0.99 it is flat in depth.", font=MONO, maxw=iw, lh=1.50)

    panel(ax, xs[1], y0, cw, ch, fill=PANEL)
    label(ax, xs[1] + 14, y0 + 10, "theorem 2   ranking")
    formula(ax, xs[1] + 14, y0 + 32, iw, "Δ_rank > 2 γ^(D+1) σ_G ≈ 0.1", size=FS_H)
    px, py, pw = xs[1] + 86, y0 + 76, cw - 110
    x_of = lambda v: px + (v / 0.45) * pw
    ax.add_patch(Rectangle((x_of(0.20), py + 6), x_of(0.40) - x_of(0.20), 24,
                           facecolor=BLUE, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x_of(0.05), py + 46), x_of(0.15) - x_of(0.05), 24,
                           facecolor=RUST, edgecolor="none", zorder=3))
    ax.plot([x_of(0.098)] * 2, [py - 4, py + 80], color=INK, lw=1.6, ls=(0, (4, 3)), zorder=4)
    tx(ax, xs[1] + 78, py + 8, "between\nclasses", color=INK, ha="right", lh=1.24)
    tx(ax, xs[1] + 78, py + 48, "within a\nclass", color=INK, ha="right", lh=1.24)
    hline(ax, px, px + pw, py + 80, color=RULE)
    tx(ax, x_of(0.098), py + 84, "0.1", color=INK, font=MONO, ha="center")
    tx_bottom(ax, xs[1] + 14, y0 + ch - 14,
              "between classes: clears it\nwithin a class: sits on it\n→ the gap D = 1 closes",
              font=MONO, maxw=iw, lh=1.50)

    panel(ax, xs[2], y0, cw, ch, fill=PANEL)
    label(ax, xs[2] + 14, y0 + 10, "theorem 3   coverage")
    formula(ax, xs[2] + 14, y0 + 32, iw, "Pr[ miss best class ] = 1 − b/5", size=FS_H)
    px, py, ph = xs[2] + 30, y0 + 76, 84
    pw = cw - 50
    pitch = pw / 5.0
    for i in range(5):
        cxb = px + pitch * i + pitch * 0.14
        bw = pitch * 0.30
        for j, (series, col) in enumerate([([.8, .6, .4, .2, 0], RUST), ([.8, .6, 0, 0, 0], BLUE)]):
            bh = series[i] / 0.85 * ph
            ax.add_patch(Rectangle((cxb + j * (bw + 3), py + ph - max(bh, 2.0)), bw,
                                   max(bh, 2.0), facecolor=col, edgecolor="none", zorder=3))
        tx(ax, cxb + bw + 1, py + ph + 5, str(i + 1), color=INK, font=MONO, ha="center")
    hline(ax, px, px + pw, py + ph, color=RULE)
    ax.add_patch(Rectangle((px, py + ph + 26), 12, 10, facecolor=RUST, edgecolor="none", zorder=3))
    tx(ax, px + 18, py + ph + 24, "uniform", color=INK)
    ax.add_patch(Rectangle((px + 96, py + ph + 26), 12, 10, facecolor=BLUE, edgecolor="none", zorder=3))
    tx(ax, px + 114, py + ph + 24, "stratified", color=INK)
    tx_bottom(ax, xs[2] + 14, y0 + ch - 14,
              "stratified: zero miss rate at b ≥ 3\nb = 5 covers all five classes", font=MONO, maxw=iw, lh=1.50)

    tx_bottom(ax, M, H - 12,
              "D = 1 buys +15% ranking accuracy for 6× solver cost   ·   "
              "D = 2 buys +5% more for 16×",
              color=MUTED, font=MONO, maxw=W - 2 * M, lh=1.30)
    return fig


KL = {
 "single": [(10,.00111),(30,.00156),(50,.00174),(70,.00197),(90,.00254),(110,.00302),
            (130,.00303),(150,.00326),(170,.00334),(190,.00357),(200,.00351)],
 "mtnofb": [(5,.00117),(15,.00199),(25,.00215),(35,.00232),(45,.00264),(55,.00315),
            (65,.00386),(75,.00457),(85,.00481),(95,.00484),(100,.00493)],
 "tree":   [(5,.00118),(15,.002),(25,.00227),(35,.00283),(45,.00335),(55,.00421),
            (65,.00519),(75,.00602),(85,.00653),(95,.00675),(100,.00683)],
 "champ":  [(5,.00135),(15,.00237),(25,.00495),(30,.01084),(40,.01197),(50,.01652),
            (60,.0218),(65,.03563),(75,.03523),(85,.03434),(95,.03399),(100,.03463)],
}


# ------------------------------------------------------------------ slide 5 --
def slide5():
    fig, ax = new_slide()
    y0 = title(ax, "The policy moves once the environment names the failure")
    lw, gap = 604.0, 16.0
    rx, rw = M + lw + gap, W - 2 * M - lw - gap
    riw = rw - 24

    band = ("457 s / step, 2 × H100        ≈ 5,000 tokens / rollout\n"
            "turn-major batching; finished rollouts exit\n"
            "turn-synchronous generation: each turn's prompt depends on the solver reply")
    bh = 30 + 3 * FS_SMALL * 1.50 + 14
    yb = H - 28 - bh
    chh = yb - 16 - y0

    panel(ax, M, y0, lw, chh, fill=PANEL)
    label(ax, M + 14, y0 + 10, "KL divergence from the SFT reference policy")
    px, py = M + 58, y0 + 38
    pw, ph = lw - 82, chh - 84
    xf = lambda s_: px + (s_ / 200.0) * pw
    yf = lambda k: py + ph - (k / 0.04) * ph
    for g in [0.01, 0.02, 0.03, 0.04]:
        hline(ax, px, px + pw, yf(g), color="#E6E2D9", lw=1.0)
        tx(ax, px - 8, yf(g) - 6, f"{g:.2f}", color=MUTED, font=MONO, ha="right")
    tx(ax, px - 8, yf(0) - 6, "0", color=MUTED, font=MONO, ha="right")
    hline(ax, px, px + pw, yf(0), color=RULE)
    ax.plot([px, px + pw], [yf(0.01)] * 2, color=INK, lw=1.3, ls=(0, (5, 4)), zorder=3)
    for k, col, lwid in [("single", GREY, 2.0), ("mtnofb", "#7B766B", 2.0),
                         ("tree", RUST, 2.0), ("champ", BLUE, 3.4)]:
        ax.plot([xf(a) for a, _ in KL[k]], [yf(b) for _, b in KL[k]], color=col,
                lw=lwid, zorder=4, solid_joinstyle="round")
    ax.add_patch(Circle((xf(100), yf(0.03463)), 4.4, facecolor=BLUE, edgecolor="none", zorder=5))
    tx(ax, xf(100) + 10, yf(0.03463) - 8, "0.0346", size=FS_H, color=BLUE, font=MONO, weight="bold")
    ax.add_patch(Circle((xf(200), yf(0.00351)), 4.0, facecolor=GREY, edgecolor="none", zorder=5))
    tx(ax, xf(200), yf(0.00351) - 22, "0.0035 after 200 steps", color=MUTED, font=MONO, ha="right")
    for s_ in [0, 50, 100, 150, 200]:
        tx(ax, xf(s_), py + ph + 6, str(s_), color=MUTED, font=MONO, ha="center")
    tx(ax, px + pw / 2, py + ph + 26, "GRPO step", color=MUTED, ha="center")

    panel(ax, rx, y0, rw, chh, fill=PANEL2)
    label(ax, rx + 12, y0 + 10, "final KL / feasibility")
    yy = y0 + 40
    for name, kl, feas, col in [("single turn", "0.0035", "4%", GREY),
                                ("multi turn", "0.0049", "12%", "#7B766B"),
                                ("+ guided search", "0.0068", "12%", RUST),
                                ("+ critical member", "0.0346", "68%", BLUE)]:
        ax.add_patch(Rectangle((rx + 12, yy + 3), 10, 10, facecolor=col, edgecolor="none", zorder=3))
        tx(ax, rx + 28, yy, name, maxw=riw - 62)
        tx(ax, rx + 28, yy + 19, kl, color=MUTED, font=MONO)
        tx(ax, rx + rw - 12, yy + 12, feas, size=FS_H, color=col, font=MONO, weight="bold", ha="right")
        yy += 50
    tx_bottom(ax, rx + 12, y0 + chh - 12,
              "single turn: 200 steps\nat KL 0.0035", color=INK, font=MONO, maxw=riw, lh=1.42)

    panel(ax, M, yb, W - 2 * M, bh, fill=INK)
    label(ax, M + 14, yb + 10, "rollout cost", color="#A7A196")
    tx(ax, M + 14, yb + 32, band, color="#D6D1C7", font=MONO, lh=1.50)
    return fig


# ------------------------------------------------------------------ slide 6 --
def slide6():
    fig, ax = new_slide()
    y0 = title(ax, "Model, optimizer and reward fixed: environment drives 0 to 68 percent")
    lw, gap = 546.0, 16.0
    rx, rw = M + lw + gap, W - 2 * M - lw - gap

    tbl = 128.0
    yb = H - 28 - tbl
    chh = yb - 16 - y0

    panel(ax, M, y0, lw, chh, fill=PANEL)
    label(ax, M + 14, y0 + 10, "held out feasibility, one change at a time")
    barchart(ax, M + 26, y0 + 44, lw - 52, chh - 118,
             [0, 4, 12, 68, 52],
             ["SFT only", "single turn", "multi turn", "critical member", "mass rule"],
             [GREY, GREY, "#7B766B", BLUE, RUST], 80)
    tx_bottom(ax, M + 14, y0 + chh - 12,
              "identical model / optimizer / reward · n = 25 · ± 8 points",
              color=MUTED, font=MONO, maxw=lw - 28, lh=1.28)

    panel(ax, rx, y0, rw, chh, fill=PANEL)
    label(ax, rx + 14, y0 + 10, "CONSTRAINT PRICE α", upper=False)
    barchart(ax, rx + 24, y0 + 50, rw - 48, chh - 124, [8, 12, 4],
             ["α = 2", "α = 5", "α = 10"], [GREY, BLUE, GREY], 15)
    tx_bottom(ax, rx + 14, y0 + chh - 12,
              "peak at α = 5", color=MUTED, font=MONO, maxw=rw - 28, lh=1.28)

    panel(ax, M, yb, W - 2 * M, tbl, fill=PANEL2)
    label(ax, M + 14, yb + 10, "one change per run")
    cols = [M + 14, M + 320, M + 458, M + 600]
    for cxx, hd in zip(cols, ["change", "feasibility", "mean FOS", "what it shows"]):
        tx(ax, cxx, yb + 32, hd.upper(), color=MUTED, weight="bold")
    hline(ax, M + 14, W - M - 14, yb + 50, color=RULE)
    yy = yb + 58
    for name, feas, fos, note, col in [
            ("grammar-compliance reward", "12% → 4%", "1.42 → 1.53",
             "format loss is a fragility effect", RUST),
            ("hard mass rule in prompt", "68% → 52%", "1.38 → 1.50",
             "a hard rule shifts the policy onto FOS", RUST),
            ("guided search, D = 1, b = 5", "12% → 12%", "1.42 → 2.12",
             "best FOS in the study, ρ = 1.0", BLUE)]:
        tx(ax, cols[0], yy, name, maxw=300)
        tx(ax, cols[1], yy, feas, color=col, font=MONO, weight="bold")
        tx(ax, cols[2], yy, fos, color=INK, font=MONO)
        tx(ax, cols[3], yy, note, color=MUTED, maxw=W - M - 14 - cols[3])
        yy += 22
    return fig


FAIL_ROWS = [("0","SCALE_PARAM(6, r, 2.0)","0.442","−0.000"),("1","SCALE_PARAM(6, t, 2.0)","0.442","−0.000"),
             ("2","SCALE_PARAM(7, r, 2.0)","0.442","+0.000"),("3","SCALE_PARAM(11, r, 2.0)","0.441","−0.001"),
             ("4","SCALE_PARAM(6, r, 2.0)","0.441","−0.000"),("5","SCALE_PARAM(6, r, 2.0)","0.440","−0.001"),
             ("6","SCALE_PARAM(5, r, 2.0)","0.440","+0.000"),("7","SCALE_PARAM(6, r, 2.0)","0.439","−0.001"),
             ("8","SCALE_PARAM(10, r, 2.0)","0.438","−0.001"),("9","SCALE_PARAM(4, r, 2.0)","0.520","+0.082"),
             ("10","SCALE_PARAM(6, r, 2.0)","0.516","−0.003"),("11","SCALE_PARAM(6, r, 3.0)","0.503","−0.013")]
FAIL_FOS = [.4423,.4423,.4421,.4421,.4412,.4408,.4401,.4401,.4386,.4378,.5197,.5163,.5032]
OK_ROWS = [("0","M4","SCALE_PARAM(4, r, 1.36)","0.483"),("1","M7","SCALE_PARAM(7, r, 1.5)","0.508"),
           ("2","M11","SCALE_PARAM(11, r, 1.5)","0.524"),("3","M8","SCALE_PARAM(8, r, 1.3)","0.959"),
           ("4","M3","SCALE_PARAM(3, r, 1.5)","1.147"),("5","M4","SCALE_PARAM(4, r, 1.3)","1.184"),
           ("6","M8","SCALE_PARAM(8, r, 1.2)","1.694")]
OK_FOS = [.4423,.4831,.5083,.5239,.9588,1.1472,1.1837,1.6936]


def fos_chart(ax, x, y, w, h, series, color, ymax=1.85):
    n = len(series) - 1
    xf = lambda i: x + (i / n) * w
    yf = lambda v: y + h - (v / ymax) * h
    hline(ax, x, x + w, yf(0), color=RULE)
    ax.plot([x, x + w], [yf(1.5)] * 2, color=GREEN, lw=1.4, ls=(0, (5, 4)), zorder=3)
    tx(ax, x, yf(1.5) - 17, "feasible at 1.5", color=GREEN)
    ax.plot([xf(i) for i in range(len(series))], [yf(v) for v in series],
            color=color, lw=3.0, zorder=4, solid_joinstyle="round")
    for i, v in enumerate(series):
        ax.add_patch(Circle((xf(i), yf(v)), 3.0, facecolor=color, edgecolor="none", zorder=5))
    tx(ax, x + w, yf(series[-1]) - 19, f"{series[-1]:.2f}", size=FS_H, color=color,
       font=MONO, weight="bold", ha="right")


def _trace_slide(fig, ax, y0, rows, headers, colx, fosser, color, hot, quotes,
                 verdict, verdict_fill, verdict_col, note):
    lw, gap = 386.0, 16.0
    liw = lw - 28
    th = H - 28 - y0
    panel(ax, M, y0, lw, th, fill=PANEL)
    for cxx, hd in zip(colx, headers):
        tx(ax, cxx, y0 + 14, hd.upper(), color=MUTED, weight="bold")
    hline(ax, M + 14, M + lw - 14, y0 + 34, color=RULE)
    yy = y0 + 42
    for r in rows:
        is_hot = r[0] == hot
        if is_hot:
            panel(ax, M + 14, yy - 4, lw - 28, 22, fill="#E2EEE6", zorder=2)
        for j, cell in enumerate(r):
            if j == 0:
                c = MUTED
            elif len(r) == 4 and j == 1 and headers[1] == "named":
                c = RUST
            else:
                c = GREEN if is_hot else INK
            tx(ax, colx[j], yy, cell, font=MONO, color=c,
               weight="bold" if is_hot else "normal")
        yy += 22
    tx_bottom(ax, M + 14, y0 + th - 14, note, color=GREEN, maxw=liw, lh=1.30)

    rx = M + lw + gap
    rw = W - 2 * M - lw - gap
    vh = 24 + txh(verdict, FS_H, rw - 28, SANS, "bold", 1.30)
    panel(ax, rx, y0, rw, 96, fill=PANEL)
    fos_chart(ax, rx + 26, y0 + 24, rw - 46, 54, fosser, color)
    yy = y0 + 110
    for hd, q in quotes:
        tx(ax, rx, yy, hd, color=MUTED, weight="bold")
        yy = quote(ax, rx, yy + 18, rw, q, accent=color) + 12
    panel(ax, rx, H - 28 - vh, rw, vh, fill=verdict_fill)
    tx(ax, rx + 14, H - 28 - vh + 12, verdict, size=FS_H, color=verdict_col,
       maxw=rw - 28, lh=1.30, weight="bold")


# ------------------------------------------------------------------ slide 7 --
def slide7():
    fig, ax = new_slide()
    y0 = title(ax, "Without the critical-member field: 12 turns, FOS unchanged",
               sub="auto_problem_000 · Qwen3-30B reasoning + warmstarted 14B action head")
    _trace_slide(fig, ax, y0, FAIL_ROWS, ["#", "action", "FOS", "Δ"],
                 [M + 16, M + 44, M + 258, M + 322], FAIL_FOS, RUST, "9",
                 [("turn 0",
                   "Buckling usually happens in compression members, so I should "
                   "check which members are in compression."),
                  ("turn 4",
                   "The mass increased, but the buckling FOS stayed the same at 0.44. "
                   "That's strange. Maybe it is determined by a specific member that "
                   "hasn't been scaled yet.")],
                 "11 / 12 edits move FOS by less than 0.002\n0 / 6 problems feasible",
                 RUSTL, RUST,
                 "step 9 edits member 4, the critical member")
    return fig


# ------------------------------------------------------------------ slide 8 --
def slide8():
    fig, ax = new_slide()
    y0 = title(ax, "With the critical-member field: 7 turns to feasibility",
               sub="Same problem, same models, same reward; one field added to the solver reply")
    _trace_slide(fig, ax, y0, OK_ROWS, ["#", "named", "action", "FOS"],
                 [M + 16, M + 42, M + 104, M + 320], OK_FOS, GREEN, "6",
                 [("turn 0",
                   "The main issue is the buckling failure, especially on member M4. "
                   "First, I need to figure out why M4 is failing."),
                  ("turn 3",
                   "The worst member for buckling is M8 now. So I need to address "
                   "M8's buckling.")],
                 "7 / 7 actions target the named member\n4 / 6 problems reach feasibility",
                 GREENL, GREEN,
                 "")
    return fig


# ------------------------------------------------------------------ slide 9 --
def slide9():
    fig, ax = new_slide()
    y0 = title(ax, "Held out: 27 of 50 feasible, and environment feedback beat every algorithmic change")
    lw, gap = 396.0, 16.0
    rx, rw = M + lw + gap, W - 2 * M - lw - gap
    riw = rw - 28
    bot = 132.0
    yb = H - 28 - bot
    top = yb - 16 - y0

    panel(ax, M, y0, lw, top, fill=PANEL)
    tx(ax, M + 18, y0 + 12, "54%", size=42, color=GREEN, font=MONO, weight="bold")
    tx(ax, M + 124, y0 + 18, "27 of 50 held out problems\nreach a feasible design",
       size=FS_H, color=INK, lh=1.34)
    cell, cg = 22.0, 5.0
    gx, gy = M + 18, y0 + 78
    for i in range(50):
        col = GREEN if i < 27 else (RUST if i < 40 else "#D69A63")
        r, c = divmod(i, 10)
        ax.add_patch(Rectangle((gx + c * (cell + cg), gy + r * (cell + cg)), cell, cell,
                               facecolor=col, edgecolor="none", zorder=3))
    tx_bottom(ax, M + 18, y0 + top - 12,
              "one square = one problem\nfirst 25 problems: 68%   full 50: 54%",
              color=MUTED, font=MONO, maxw=lw - 36, lh=1.45)

    panel(ax, rx, y0, rw, top, fill=PANEL)
    label(ax, rx + 14, y0 + 10, "outcome breakdown")
    yy = y0 + 38
    for col, n, name, note in [
            (GREEN, "27", "Feasible", "all constraints met, inside the turn cap"),
            (RUST, "13", "Never reached FOS 1.5",
             "mean initial FOS 0.32; solved set starts at 0.70"),
            ("#D69A63", "10", "FOS met, another constraint failed",
             "buckling fixed; 3 exceeded 3× initial mass")]:
        ax.add_patch(Rectangle((rx + 14, yy + 3), 13, 13, facecolor=col, edgecolor="none", zorder=3))
        tx(ax, rx + 36, yy, n, size=FS_H, color=col, font=MONO, weight="bold")
        tx(ax, rx + 72, yy, name, size=FS_H, color=INK, maxw=riw - 58)
        yy = tx(ax, rx + 72, yy + 21, note, color=MUTED, maxw=riw - 58, lh=1.28) + 10

    half = (W - 2 * M - 14) / 2.0
    panel(ax, M, yb, half, bot, fill=PANEL2)
    label(ax, M + 14, yb + 10, "solved versus failed runs")
    tx(ax, M + 286, yb + 32, "SOLVED", color=GREEN, weight="bold", ha="right")
    tx(ax, M + 362, yb + 32, "FAILED", color=RUST, weight="bold", ha="right")
    hline(ax, M + 14, M + half - 14, yb + 50, color=RULE)
    yy = yb + 58
    for name, a, b in [("parse-and-execute rate", "0.78", "0.53"),
                       ("turns used (cap 20)", "10.4", "20.0"),
                       ("initial FOS buckling", "0.70", "0.32")]:
        tx(ax, M + 14, yy, name, maxw=250)
        tx(ax, M + 286, yy, a, color=GREEN, font=MONO, weight="bold", ha="right")
        tx(ax, M + 362, yy, b, color=RUST, font=MONO, weight="bold", ha="right")
        yy += 21

    panel(ax, M + half + 14, yb, half, bot, fill=INK)
    label(ax, M + half + 28, yb + 10, "next", color="#A7A196")
    yy = yb + 34
    for i, s_ in enumerate(["Price mass and FOS jointly in Φ",
                            "KL term that protects format",
                            "Handle degenerate solver states",
                            "Distil 30B trajectories into 14B"]):
        tx(ax, M + half + 28, yy, f"{i+1}.", color="#A7A196", font=MONO)
        yy = tx(ax, M + half + 48, yy, s_, color="#D6D1C7", maxw=half - 76, lh=1.28) + 5
    return fig


def main():
    paths = []
    for i, b in enumerate([slide1, slide2, slide3, slide4, slide5,
                           slide6, slide7, slide8, slide9], 1):
        fig = b()
        p = os.path.join(OUTDIR, f"slide_{i:02d}.png")
        fig.savefig(p, dpi=DPI, facecolor=WHITE)
        plt.close(fig)
        paths.append(p)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    for p in paths:
        s_ = prs.slides.add_slide(blank)
        s_.shapes.add_picture(p, 0, 0, width=prs.slide_width, height=prs.slide_height)
    prs.save(PPTX)
    print(f"wrote {PPTX} ({len(paths)} slides)")


if __name__ == "__main__":
    main()
