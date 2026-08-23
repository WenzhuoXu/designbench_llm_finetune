"""Build DesignBench_SFT_GRPO_Status_2.pptx — restructured, grounded, visually rich.

Narrative is framework-first: the task, the end-to-end architecture, the two
training stages (SFT warmstart for grammar compliance via selective loss masking;
GRPO for policy learning via group + tree-expanded rollout in a multi-turn FEA
environment), then grounded results. The multi-turn protocol and critical-member
observation are presented as environment-design choices, not "fixes".

Every number is grounded in docs/status_report.md, configs/rl/grpo_mt_01b.yaml,
the code (file:line in speaker notes), and results/report/deck/*.png (real runs).

Run:  python scripts/make_status_deck.py
Out:  docs/DesignBench_SFT_GRPO_Status_2.pptx   (+ _notes.md companion)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(ROOT, "results/report/deck")
OUT  = os.path.join(ROOT, "docs/DesignBench_SFT_GRPO_Status_2.pptx")
NOTESMD = os.path.join(ROOT, "docs/DesignBench_SFT_GRPO_Status_2_notes.md")

# ------------------------------------------------------------------ palette ---
NAVY   = "21304A"   # primary ink / dark slide bg
NAVY2  = "1B2233"   # deeper navy (code bg)
NAVY3  = "2C3C58"   # code accent
RED    = "EB1000"   # signal accent (titles, champion)
INK    = "2C2C2C"   # body text
SLATE  = "5A6473"   # secondary text
SLATE2 = "6E7C97"
GREEN  = "1E8449"   # positive
TEAL   = "0E7C7B"
BLUE   = "2E5A9C"
PURPLE = "7030A0"
AMBER  = "B26B00"
WHITE  = "FFFFFF"
CARD   = "FFFFFF"
CARD2  = "EEF1F6"   # light blue-grey card
CARD3  = "F4F5F7"   # light grey card
LINE   = "D9DEE6"   # hairline
GREENBG= "EAF6EE"; GREENLN="BFE0CA"
REDBG  = "FBEAE7"; REDLN ="E7C3BD"
MUTE   = "AAB2C0"
CODEDIM= "8FA0BE"; CODEHI="FF8A7A"; CODETXT="E7ECF5"

BODY = "Adobe Clean"
BOLD = "Adobe Clean ExtraBold"
MONO = "Courier New"

SW, SH = 13.333, 7.5
MX = 0.6

prs = Presentation()
prs.slide_width  = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]
NOTES = {}   # slide_index -> (title, [bullets])

def C(h): return RGBColor.from_string(h)

# ------------------------------------------------------------------ helpers ---
def slide(bg=WHITE):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = C(bg)
    return s

def _noline(shp):
    shp.line.fill.background()

def rect(s, l, t, w, h, fill=None, line=None, lw=1.0, radius=None, shadow=False):
    shp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius is not None else MSO_SHAPE.RECTANGLE,
        Inches(l), Inches(t), Inches(w), Inches(h))
    if radius is not None:
        try: shp.adjustments[0] = radius
        except Exception: pass
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = C(fill)
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = C(line); shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    return shp

def line_h(s, l, t, w, color=LINE, weight=1.0):
    ln = s.shapes.add_connector(2, Inches(l), Inches(t), Inches(l+w), Inches(t))
    ln.line.color.rgb = C(color); ln.line.width = Pt(weight)
    return ln

def text(s, l, t, w, h, runs, align="l", anchor="t", spacing=1.0, wrap=True,
         space_after=2.0):
    """runs: str | list-of-paragraphs.
    A paragraph is either a str (uses default style) or a list of run-tuples
    (txt, size, color, bold[, font, italic]). Default style on first call uses
    14/INK/BODY; override by passing run-tuples.
    """
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = wrap
    tf.margin_left=0; tf.margin_right=0; tf.margin_top=0; tf.margin_bottom=0
    tf.vertical_anchor = {"t":MSO_ANCHOR.TOP,"m":MSO_ANCHOR.MIDDLE,"b":MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(runs, str):
        runs = [runs]
    al = {"l":PP_ALIGN.LEFT,"c":PP_ALIGN.CENTER,"r":PP_ALIGN.RIGHT}[align]
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = al; p.line_spacing = spacing; p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        if isinstance(para, str):
            para = [(para, 14, INK, False)]
        for rt in para:
            txt, sz, col, bold = rt[0], rt[1], rt[2], rt[3]
            fnt = rt[4] if len(rt) > 4 else (BOLD if bold else BODY)
            ital = rt[5] if len(rt) > 5 else False
            r = p.add_run(); r.text = txt
            r.font.size = Pt(sz); r.font.bold = bold; r.font.name = fnt
            r.font.italic = ital; r.font.color.rgb = C(col)
    return tb

def image_fit(s, path, l, t, w, h, align="c", valign="m"):
    im = Image.open(path); iw, ih = im.size; ar = iw/ih; box = w/h
    if ar > box: nw, nh = w, w/ar
    else:        nh, nw = h, h*ar
    nl = l + {"l":0,"c":(w-nw)/2,"r":w-nw}[align]
    nt = t + {"t":0,"m":(h-nh)/2,"b":h-nh}[valign]
    return s.shapes.add_picture(path, Inches(nl), Inches(nt), Inches(nw), Inches(nh))

def chevron(s, l, t, color=SLATE, size=0.26):
    sh = s.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(l), Inches(t), Inches(size*1.1), Inches(size))
    sh.fill.solid(); sh.fill.fore_color.rgb = C(color); _noline(sh); sh.shadow.inherit=False
    return sh

def down_arrow(s, cx, t, color=SLATE, w=0.34, h=0.30):
    sh = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(cx-w/2), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = C(color); _noline(sh); sh.shadow.inherit=False
    return sh

def right_arrow(s, l, t, color=SLATE, w=0.42, h=0.26):
    sh = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = C(color); _noline(sh); sh.shadow.inherit=False
    return sh

def chip(s, l, t, label, fill, txtcol, w=None, size=11.5, bold=True, h=0.34):
    if w is None: w = 0.16 + 0.092*len(label)
    rect(s, l, t, w, h, fill=fill, radius=0.5)
    text(s, l, t-0.005, w, h, [[(label, size, txtcol, bold)]], align="c", anchor="m")
    return w

def notes(idx, title, bullets):
    NOTES[idx] = (title, bullets)

# ------------------------------------------------------------- composites ----
def header(s, title, dek=None, tag=None):
    # `tag` kept in signatures for readability but intentionally NOT rendered
    # (no decorative corner labels).
    text(s, MX, 0.34, 12.1, 0.7, [[(title, 27, RED, True, BOLD)]])
    if dek:
        text(s, MX, 1.06, 11.0, 0.45, [[(dek, 14.5, SLATE, False)]])
    line_h(s, MX, 1.62, SW-2*MX, color=LINE, weight=1.2)

def footer(s, n):
    # intentionally blank — no footer branding / page-number chrome
    return

def kpi_block(s, l, t, w, number, label, ncol, lcol, nsize=46, h=1.7):
    text(s, l, t, w, h*0.62, [[(number, nsize, ncol, True, BOLD)]], align="c", anchor="b")
    text(s, l, t+h*0.62, w, h*0.34, [[(label, 12, lcol, False)]], align="c", anchor="t", spacing=0.95)

def accent_card(s, l, t, w, h, accent, fill=CARD, radius=0.045):
    rect(s, l, t, w, h, fill=fill, line=LINE, lw=1.0, radius=radius)
    rect(s, l, t+0.12, 0.08, h-0.24, fill=accent)

def code_block(s, l, t, w, h, lines, bg=NAVY2, size=12.5, title=None):
    rect(s, l, t, w, h, fill=bg, radius=0.03)
    y = t + 0.18
    if title:
        text(s, l+0.28, y, w-0.5, 0.3, [[(title, 11, CODEDIM, True)]]); y += 0.38
    paras = []
    for ln in lines:
        if isinstance(ln, str): ln = [(ln, CODETXT)]
        paras.append([(seg[0], size, seg[1], False, MONO) for seg in ln])
    text(s, l+0.28, y, w-0.5, h-(y-t)-0.15, paras, spacing=1.06, space_after=1.5)

# =================================================================  SLIDES  ===
N = 0
def nxt():
    global N; N += 1; return N

# ---- 1 · TITLE --------------------------------------------------------------
s = slide(NAVY); i = nxt()
rect(s, 0, 0, 0.18, SH, fill=RED)
text(s, MX, 0.95, 10, 0.4, [[("STATUS REPORT", 13, "CADCFC", True), ("    ·    2026-06-30", 13, SLATE2, False)]])
text(s, MX, 1.55, 12.1, 1.2, [[("DesignBench: SFT + GRPO", 50, WHITE, True, BOLD)]])
text(s, MX, 2.75, 11.6, 0.6, [[("Teaching a 14B model to design feasible trusses, one edit at a time", 21, "DCE3F2", False)]])
text(s, MX, 3.45, 11.6, 0.5, [[("The limiting factor was the environment — information and protocol — not reasoning capacity or the RL algorithm.", 15, "AEB9CC", False, BODY, True)]])
# KPI strip
kpi_block(s, MX, 4.45, 1.9, "0%", "warmstart (SFT only)", MUTE, MUTE, nsize=60, h=1.5)
right_arrow(s, 2.75, 4.95, color=SLATE2, w=0.7, h=0.34)
kpi_block(s, 3.7, 4.45, 2.1, "68%", "champion policy", RED, RED, nsize=60, h=1.5)
text(s, 6.1, 4.85, 5.6, 0.6, [[("held-out feasibility on truss design problems", 14, "AEB9CC", False)]], anchor="m")
line_h(s, MX, 6.85, SW-2*MX, color=NAVY3, weight=1.0)
text(s, MX, 6.95, 12.1, 0.3, [[("Qwen3-14B · LoRA r=32 · multi-turn grammar-action policy · potential-based shaping reward · audit trail in docs/ablation_ledger.md", 10.5, SLATE2, False)]])
notes(i, "Title — opening frame", [
    "Open with the one-line thesis: this is a study of how to teach a mid-size (14B) open model to solve a hard engineering task — iterative truss design — and the headline finding is that the BOTTLENECK was the training environment, not the model's reasoning or the RL math.",
    "Anchor the audience on the single number that matters: 0% feasible after supervised warmstart → 68% feasible after the full framework. Everything else explains how and why.",
    "Set scope: Qwen3-14B with LoRA, a multi-turn grammar-action policy, trained with GRPO under a potential-based reward. Mention every claim is backed by a logged audit trail.",
    "Promise the structure: task → framework architecture → the two training stages → grounded results.",
])

# ---- 2 · EXECUTIVE SUMMARY --------------------------------------------------
s = slide(); i = nxt()
header(s, "The Story in One Slide", "What we built, what it achieved, and the one idea worth remembering.", tag="OVERVIEW")
# three stacked statement cards
accent_card(s, MX, 1.85, 7.55, 1.45, RED)
text(s, MX+0.35, 1.98, 7.1, 0.4, [[("What we built", 13, RED, True)]])
text(s, MX+0.35, 2.36, 7.1, 0.9, [[("A two-stage framework — ", 14, INK, False),
    ("SFT warmstart", 14, NAVY, True), (" for grammar, then ", 14, INK, False),
    ("GRPO", 14, NAVY, True), (" for policy — that drives a 14B model through a ", 14, INK, False),
    ("multi-turn FEA environment", 14, NAVY, True), (", repairing a truss one grammar action per turn.", 14, INK, False)]], spacing=1.02)
accent_card(s, MX, 3.45, 7.55, 1.45, GREEN)
text(s, MX+0.35, 3.58, 7.1, 0.4, [[("What it achieved", 13, GREEN, True)]])
text(s, MX+0.35, 3.96, 7.1, 0.9, [[("From ", 14, INK, False), ("0% → 68%", 14, GREEN, True),
    (" held-out feasibility. The same multi-turn loop with a strong zero-shot reasoner solves ", 14, INK, False),
    ("4 of 6", 14, GREEN, True), (" problems with no RL at all.", 14, INK, False)]], spacing=1.02)
accent_card(s, MX, 5.05, 7.55, 1.55, NAVY)
text(s, MX+0.35, 5.18, 7.1, 0.4, [[("The one idea", 13, NAVY, True)]])
text(s, MX+0.35, 5.56, 7.1, 1.0, [[("Reasoning capacity was never the limit. The leverage was ", 14, INK, False),
    ("environment design", 14, RED, True),
    (": make training match deployment, and make the observation contain the one fact the policy needs.", 14, INK, False)]], spacing=1.02)
# right rail KPIs
for (yy, num, lab, col) in [(1.95,"0%","SFT warmstart", MUTE),(3.1,"68%","champion policy", RED),
                            (4.25,"4/6","zero-shot, no RL", GREEN),(5.4,"α=5","single reward knob", NAVY)]:
    rect(s, 8.45, yy, 4.25, 1.0, fill=CARD2, radius=0.06)
    text(s, 8.65, yy+0.10, 1.7, 0.8, [[(num, 30, col, True, BOLD)]], anchor="m")
    text(s, 10.35, yy+0.10, 2.2, 0.8, [[(lab, 13, SLATE, False)]], anchor="m", spacing=0.95)
footer(s, i)
notes(i, "Executive summary", [
    "This slide is the whole talk compressed — use it to give the audience a map before any detail.",
    "What we built: a clean two-stage pipeline (SFT then GRPO) wrapped around a multi-turn environment where the model edits the truss one grammar action at a time and a physics solver scores each edit.",
    "What it achieved: 0%→68% held-out feasibility; and critically, a strong zero-shot reasoner inside the SAME loop solves 4/6 with no training — foreshadowing the punchline.",
    "The one idea: don't frame the gains as bug fixes. The reasoning was always there; the wins came from designing the environment well — train/test protocol fidelity and a complete observation.",
    "Right rail gives four numbers you'll return to: 0%, 68%, 4/6, and α=5 (the single reward hyperparameter).",
])

# ---- 3 · AGENDA -------------------------------------------------------------
s = slide(); i = nxt()
header(s, "Agenda")
items = [
    ("01","The task & the framework","Truss design as an MDP, and the end-to-end architecture"),
    ("02","Stage 1 — SFT warmstart","Grammar compliance via selective loss masking"),
    ("03","Stage 2 — GRPO","Policy learning: group + tree rollout, potential reward"),
    ("04","Results & analysis","0%→68%, training dynamics, ablations, the real bottleneck"),
]
yy = 1.95
for num, t1, t2 in items:
    rect(s, MX, yy, 12.1, 1.12, fill=CARD3, radius=0.05)
    rect(s, MX+0.28, yy+0.22, 0.7, 0.68, fill=NAVY, radius=0.18)
    text(s, MX+0.28, yy+0.18, 0.7, 0.68, [[(num, 22, WHITE, True, BOLD)]], align="c", anchor="m")
    text(s, MX+1.25, yy+0.20, 10.5, 0.45, [[(t1, 18, NAVY, True)]])
    text(s, MX+1.25, yy+0.66, 10.5, 0.4, [[(t2, 12.5, SLATE, False)]])
    yy += 1.24
footer(s, i)
notes(i, "Agenda", [
    "Four parts. Stress that the order is deliberate: the framework is presented as a designed system BEFORE any results, so the results read as validation rather than a debugging diary.",
    "Part 1 frames the problem (an MDP) and shows the architecture on one diagram.",
    "Parts 2–3 are the two training stages, each with its unique design decision (loss masking for SFT; rollout structure + reward for GRPO).",
    "Part 4 is the evidence: the progression, training dynamics, ablations, and the experiment that reframes the whole project.",
])

# =====================  PART 1 — TASK & FRAMEWORK  ===========================
def divider(num, title, sub):
    s = slide(NAVY)
    rect(s, 0, 0, 0.18, SH, fill=RED)
    text(s, MX+0.3, 2.5, 2.2, 1.4, [[(num, 96, NAVY3, True, BOLD)]], anchor="m")
    text(s, MX+0.3, 3.95, 11.5, 0.9, [[(title, 36, WHITE, True, BOLD)]])
    line_h(s, MX+0.32, 4.85, 3.2, color=RED, weight=2.5)
    text(s, MX+0.32, 5.0, 10.5, 0.6, [[(sub, 16, "AEB9CC", False)]])
    return s

s = divider("01", "The Task & the Framework", "Truss design as a sequential decision problem — and the system that learns it."); i = nxt()
notes(i, "Divider — Part 1", [
    "Transition: 'Before any results, here is the problem and the machine that solves it.'",
    "Goal of this section: by the end the audience should be able to draw the architecture from memory.",
])

# ---- 5 · THE TASK -----------------------------------------------------------
s = slide(); i = nxt()
header(s, "The Optimization Task", "Reshape a pin-jointed truss until it is structurally sound — changing one member at a time.", tag="01 · TASK")
image_fit(s, os.path.join(FIG,"truss_m4.png"), MX, 1.95, 6.1, 2.7, valign="m")
text(s, MX, 4.75, 6.1, 0.5, [[("State ", 12.5, NAVY, True),
    ("s", 12.5, NAVY, True, BODY, True),
    (" = joints, members, materials, cross-sections + the solver's verdict.", 12.5, SLATE, False)]], spacing=1.0)
# constraints
text(s, 7.05, 1.85, 5.7, 0.4, [[("Feasible = three checks pass at once", 15, NAVY, True)]])
for (yy, h1, h2, col) in [
    (2.35,"FOS_buckling ≥ 1.5","the binding check — slender members buckle first", RED),
    (3.18,"FOS_yielding ≥ 1.5","the material must not yield under load", TEAL),
    (4.01,"mass ≤ limit","and it must stay light — strength alone is not enough", NAVY)]:
    accent_card(s, 7.05, yy, 5.65, 0.74, col, fill=CARD)
    text(s, 7.32, yy+0.10, 5.3, 0.32, [[(h1, 14.5, NAVY, True)]])
    text(s, 7.32, yy+0.43, 5.3, 0.28, [[(h2, 11, SLATE, False)]])
# action card
accent_card(s, 7.05, 4.95, 5.65, 1.55, AMBER, fill=CARD2)
text(s, 7.32, 5.08, 5.2, 0.34, [[("The only move: one grammar edit per turn", 14, NAVY, True)]])
text(s, 7.32, 5.46, 5.2, 0.5, [[("SCALE_PARAM(4, r, 1.36)", 13.5, RED, True, MONO)]])
text(s, 7.32, 5.86, 5.2, 0.6, [[("— scale member 4's radius ×1.36. A physics solver (FEA) then re-scores the design. Six action types: SCALE, ADD, REMOVE, MODIFY, MOVE, SCALE_MULTI.", 11, SLATE, False)]], spacing=0.98)
footer(s, i)
notes(i, "The optimization task", [
    "Define the task concretely: take an infeasible pin-jointed truss and edit it until it satisfies all constraints. This is DesignBench's iterative truss-optimization benchmark (20 problems).",
    "State = the full structure (joints, members, materials, pipe r/t cross-sections) plus the FEA verdict. Show the truss; point out member M4 (J6–J7) is the buckling-critical member under load P — we'll return to it.",
    "Feasible means THREE constraints hold simultaneously: FOS_buckling ≥ 1.5 (usually the binding one), FOS_yielding ≥ 1.5, and mass ≤ limit. A design that fixes buckling but blows the mass budget is still a failure — this tension drives later results.",
    "Action space: exactly one grammar edit per turn, e.g. SCALE_PARAM(4, r, 1.36). Six action types total. After each action the FEA solver re-scores — that is the environment.",
    "Why one-edit-per-turn matters: it turns design into a sequential decision problem (an MDP), which is what makes RL the natural tool — set up Part 1's architecture.",
])

# ---- 6 · ARCHITECTURE -------------------------------------------------------
s = slide(); i = nxt()
header(s, "The Framework at a Glance", "One pipeline: data becomes format, format becomes policy — and the policy lives in a multi-turn FEA loop.", tag="01 · ARCHITECTURE")
stages = [
    ("Data", "5,000 DesignBench\nexpert traces", NAVY, "multi-turn <think> +\naction + FEA feedback"),
    ("SFT warmstart", "teach the response\nformat & grammar", BLUE, "selective loss masking\n(Stage 2 of talk)"),
    ("GRPO", "learn the design\npolicy", RED, "group + tree rollout,\npotential reward"),
    ("Policy", "14B grammar-action\nagent", GREEN, "deployed in the same\nmulti-turn loop"),
]
x = MX; cw = 2.78; gap = 0.42; ty = 1.95
for j,(t1,t2,col,sub) in enumerate(stages):
    accent_card(s, x, ty, cw, 1.7, col, fill=CARD)
    text(s, x+0.26, ty+0.16, cw-0.4, 0.4, [[(t1, 16, NAVY, True)]])
    text(s, x+0.26, ty+0.60, cw-0.4, 0.7, [[(t2, 12, SLATE, False)]], spacing=0.96)
    text(s, x+0.26, ty+1.24, cw-0.4, 0.4, [[(sub, 9.5, SLATE2, False, BODY, True)]], spacing=0.9)
    if j < 3:
        chevron(s, x+cw+0.06, ty+0.72, color=RED if j==1 else SLATE, size=0.30)
    x += cw + gap
# environment band
ey = 4.05
rect(s, MX, ey, 12.1, 2.05, fill=NAVY, radius=0.03)
text(s, MX+0.35, ey+0.16, 11.4, 0.4, [[("The multi-turn FEA environment", 15, WHITE, True),
    ("    — identical at training (GRPO rollout) and at test (eval)", 12.5, "AEB9CC", False)]])
loop = [("think","read the verdict, locate the fault", CODEDIM),
        ("act","emit one grammar action", "9BE6C9"),
        ("simulate","FEA re-scores the design", "9BD2E6"),
        ("observe","feedback names the critical member", CODEHI)]
lx = MX+0.35; lw = 2.72; lgap=0.30; ly=ey+0.72
for j,(t1,t2,col) in enumerate(loop):
    rect(s, lx, ly, lw, 0.95, fill=NAVY3, radius=0.06)
    text(s, lx+0.18, ly+0.13, lw-0.3, 0.34, [[(t1, 14.5, col, True)]])
    text(s, lx+0.18, ly+0.47, lw-0.3, 0.42, [[(t2, 10.5, "D7DEEC", False)]], spacing=0.92)
    if j<3: text(s, lx+lw-0.02, ly+0.18, 0.34, 0.5, [[("→", 20, "7C8AA6", True)]], align="c")
    lx += lw+lgap
footer(s, i)
notes(i, "Framework architecture", [
    "This is the slide the audience should remember. Walk it left to right: 5,000 DesignBench expert traces → SFT warmstart (format) → GRPO (policy) → a deployed 14B grammar-action agent.",
    "Emphasize the division of labour: SFT teaches the model HOW to speak (format/grammar), GRPO teaches it WHAT to do (the policy). They are not redundant — covered in Parts 2 and 3.",
    "The red chevron between SFT and GRPO marks where capability is actually acquired.",
    "The navy band is the key design principle: ONE multi-turn FEA environment, used identically during GRPO rollouts and during evaluation. think → act → simulate → observe.",
    "Foreshadow two environment-design choices that recur in results: (1) the loop is multi-turn at both train and test (protocol fidelity), and (2) the observation in 'observe' names the critical member. We do NOT call these 'fixes' — they are properties of a correct environment.",
    "Code anchors for Q&A: rollout loop multiturn_rollout.py:182–242; feedback string designbench_prompt.py:73–82.",
])

# ---- 7 · THE ENVIRONMENT ----------------------------------------------------
s = slide(); i = nxt()
header(s, "The Environment: A Faithful Multi-Turn Loop", "The model is trained exactly the way it is deployed — and it observes the one fact it needs.", tag="01 · ENVIRONMENT")
# left: loop diagram (vertical)
text(s, MX, 1.8, 5.8, 0.35, [[("One turn of the rollout", 14, NAVY, True)]])
steps = [("s_t  state","truss + FEA verdict (infeasible)", CARD2, NAVY),
         ("<think> … </think>","reason about the failing member", CARD, NAVY),
         ("<action> … </action>","one grammar edit, e.g. SCALE_PARAM(4, r, 1.36)", REDBG, RED),
         ("FEA solver","execute the edit, re-score", CARD, TEAL),
         ("[Simulation Result]","verdict + worst member(s): M4  →  s_{t+1}", GREENBG, GREEN)]
yy = 2.2
for j,(t1,t2,fill,col) in enumerate(steps):
    accent_card(s, MX, yy, 5.75, 0.70, col, fill=fill)
    text(s, MX+0.26, yy+0.08, 5.4, 0.32, [[(t1, 13.5, NAVY, True, MONO if ("<" in t1 or "[" in t1) else BODY)]])
    text(s, MX+0.26, yy+0.40, 5.4, 0.28, [[(t2, 10.5, SLATE, False)]])
    if j<4: down_arrow(s, MX+2.87, yy+0.71, color=SLATE2, w=0.30, h=0.22)
    yy += 0.92
text(s, MX, yy-0.02, 5.75, 0.4, [[("repeat ≤ 5 turns, or stop when feasible", 11, SLATE2, False, BODY, True)]], align="c")
# right: two design principles
text(s, 6.85, 1.8, 5.9, 0.35, [[("Two principles behind the loop", 14, NAVY, True)]])
accent_card(s, 6.85, 2.2, 5.85, 1.95, NAVY, fill=CARD)
text(s, 7.12, 2.34, 5.4, 0.34, [[("1 · Protocol fidelity", 14.5, NAVY, True)]])
text(s, 7.12, 2.72, 5.4, 1.35, [[("The policy makes ", 12.5, INK, False), ("one action per turn", 12.5, NAVY, True),
    (" and sees the solver's reply before the next. Training uses this exact loop, so the gradient rewards the behaviour we actually deploy — not a single-shot whole-trajectory dump.", 12.5, INK, False)]], spacing=1.02)
accent_card(s, 6.85, 4.3, 5.85, 1.95, RED, fill=CARD)
text(s, 7.12, 4.44, 5.4, 0.34, [[("2 · Complete observation", 14.5, NAVY, True)]])
text(s, 7.12, 4.82, 5.4, 1.4, [[("The FEA verdict names the ", 12.5, INK, False), ("buckling-critical member", 12.5, RED, True),
    (" (min_fos_buckling_member_id → “worst member(s): M4”). Without it the policy can reason perfectly yet edit the wrong member.", 12.5, INK, False)]], spacing=1.02)
footer(s, i)
notes(i, "The multi-turn environment", [
    "Walk the left column as one turn of the loop: state s_t (infeasible) → the model writes <think> then exactly one <action> → the FEA solver executes and re-scores → a [Simulation Result] turn returns the verdict, which becomes s_{t+1}. Loop up to 5 turns or until feasible.",
    "Frame the two boxes on the right as DESIGN PRINCIPLES, not patches. Principle 1 — protocol fidelity: the model is trained with the same one-action-per-turn loop it runs at test time, so the learning signal targets the real behaviour. (Contrast briefly: an earlier single-turn protocol asked for the whole trajectory at once and the policy could not learn — but present this as 'why fidelity matters', not as a bug story.)",
    "Principle 2 — complete observation: the solver knows which member is about to buckle; the observation must include it ('worst member(s): M4'). This is the single most consequential design choice for accuracy.",
    "These two principles are exactly what the results section will quantify, so plant them firmly here.",
    "Code anchors: max_turns=5 / multi_turn=true in grpo_mt_01b.yaml; feedback construction designbench_prompt.py:73–82.",
])

# ---- 8 · A REAL TRACE (flowchart) -------------------------------------------
s = slide(); i = nxt()
header(s, "A Real Trace, End to End", "auto_problem_000, champion policy — five of seven turns, infeasible → feasible.", tag="01 · TRACE")
nodes = [
    ("s0","FOS_b 0.44\nworst: M4", REDBG, RED),
    ("think","“M4 (J6–J7) is\ncritical → grow it”", CARD2, NAVY),
    ("action","SCALE_PARAM\n(4, r, 1.36)", CARD, RED),
    ("FEA","FOS_b 0.48\nworst: M4", REDBG, RED),
    ("action","SCALE_PARAM\n(7, r, 1.50)", CARD, RED),
    ("…","turns 3–6\ngrow critical\nmembers", CARD3, SLATE),
    ("FEASIBLE","FOS_b 1.69\nin 7 turns", GREENBG, GREEN),
]
x = MX; cw = 1.56; gap = 0.18; ty = 2.0
for j,(t1,t2,fill,col) in enumerate(nodes):
    accent_card(s, x, ty, cw, 1.5, col, fill=fill, radius=0.06)
    text(s, x+0.14, ty+0.12, cw-0.24, 0.34, [[(t1, 12.5, col, True, MONO if t1 in ("think","action","FEA") else BODY)]])
    mono = t1=="action"
    text(s, x+0.14, ty+0.52, cw-0.24, 0.9, [[(t2, 10 if not mono else 10, INK, False, MONO if mono else BODY)]], spacing=0.95)
    if j < len(nodes)-1:
        right_arrow(s, x+cw+0.0, ty+0.62, color=GREEN if j==len(nodes)-2 else SLATE2, w=0.18, h=0.22)
    x += cw + gap
# bottom insight
rect(s, MX, 3.95, 12.1, 1.1, fill=CARD2, radius=0.04)
text(s, MX+0.3, 4.08, 11.5, 0.4, [[("What to notice", 13.5, RED, True)]])
text(s, MX+0.3, 4.46, 11.5, 0.6, [[("Every turn the policy reads the named worst member from the feedback and edits ", 13, INK, False),
    ("that exact member", 13, NAVY, True),
    (". FOS_buckling climbs monotonically past the 1.5 line. The reasoning was never exotic — it was correctly grounded by one fact in the observation.", 13, INK, False)]], spacing=1.02)
# the think text sample
code_block(s, MX, 5.25, 12.1, 1.55, [
    [("<think> ", CODEDIM), ("The buckling FOS is 0.44, far below 1.5. The worst member is ", CODETXT), ("M4 (J6–J7)", CODEHI), (". Since it is critical,", CODETXT)],
    [("increasing its cross-sectional area should help most. Current r = 0.0380 m … grow the radius. ", CODETXT), ("</think>", CODEDIM)],
    [("<action>", CODEDIM), ("SCALE_PARAM(4, r, 1.36)", "9BE6C9"), ("</action>", CODEDIM)],
], title="champion · auto_problem_000 · turn 0")
footer(s, i)
notes(i, "A real trace (replaces the old 'what is a single turn' slide)", [
    "This is the slide the feedback explicitly asked for: an ACTUAL trajectory as a flowchart, not an abstract explanation of a conversation.",
    "Read the top row left→right: start infeasible (FOS_b 0.44, worst member M4) → think (locate M4) → action SCALE_PARAM(4,r,1.36) → FEA returns 0.48, still M4 → next action grows member 7 → turns 3–6 keep growing the critical members → FEASIBLE at FOS_b 1.69 in 7 turns.",
    "The 'what to notice' band is the takeaway: each turn the policy targets the member NAMED in the feedback. The competence is grounding, not raw cleverness.",
    "The code block shows the real turn-0 generation so the audience sees genuine CoT + a single grammar action — concrete, not schematic.",
    "Transition to Part 2: 'For any of this to work, the model first has to speak this format perfectly — that is what SFT does.'",
])

# =====================  PART 2 — SFT  ========================================
s = divider("02", "Stage 1 — SFT Warmstart", "Teach the model to speak the grammar — supervise what it writes, mask what it only reads."); i = nxt()
notes(i, "Divider — Part 2", [
    "Transition: the policy can only be learned if the model reliably emits the format. SFT is a lightweight warmstart for exactly that.",
    "Key message to set up: SFT teaches FORMAT, not strategy. The unique design decision is selective loss masking.",
])

# ---- 10 · SFT TRANSFORM -----------------------------------------------------
s = slide(); i = nxt()
header(s, "SFT Warmstart: Format, Not Strategy", "Reformat 5,000 expert traces into the exact inference-time conversation, then fine-tune on it.", tag="02 · SFT")
text(s, MX, 1.8, 6.0, 0.35, [[("The data transform", 14, NAVY, True)]])
trans = [
    ("Merge the setup","system[0] problem + system[1] initial state  →  one user turn", NAVY),
    ("Remap FEA feedback","every solver message  →  user role, prefixed “[Simulation Result]”", TEAL),
    ("Surface the action","pull the edit out of <think>  →  its own <action>…</action> tag", RED),
]
yy = 2.2
for t1,t2,col in trans:
    accent_card(s, MX, yy, 6.0, 0.92, col, fill=CARD)
    text(s, MX+0.26, yy+0.13, 5.6, 0.34, [[(t1, 14, NAVY, True)]])
    text(s, MX+0.26, yy+0.49, 5.6, 0.38, [[(t2, 11, SLATE, False)]], spacing=0.95)
    yy += 1.04
text(s, MX, yy+0.02, 6.0, 0.7, [[("Result: a clean ", 12, INK, False), ("system → user → assistant → user → …", 12, NAVY, True, MONO),
    (" multi-turn chat that matches the rollout exactly.", 12, INK, False)]], spacing=1.0)
# right: what it teaches / does not
accent_card(s, 6.95, 1.95, 5.75, 2.15, GREEN, fill=GREENBG)
text(s, 7.22, 2.08, 5.3, 0.34, [[("What warmstart teaches", 14, GREEN, True)]])
for k,(b) in enumerate(["the output format <think>…</think> <action>…</action>",
                        "the six-action grammar syntax and arguments",
                        "turn-taking: read [Simulation Result], then act"]):
    text(s, 7.22, 2.5+k*0.5, 5.3, 0.45, [[("✓  ", 12.5, GREEN, True),(b, 12, INK, False)]], spacing=0.95)
accent_card(s, 6.95, 4.3, 5.75, 1.95, SLATE, fill=CARD3)
text(s, 7.22, 4.43, 5.3, 0.34, [[("What it deliberately does NOT teach", 14, SLATE, True)]])
for k,(b) in enumerate(["which member to fix or how much to scale",
                        "the optimization strategy or value function"]):
    text(s, 7.22, 4.85+k*0.5, 5.3, 0.45, [[("✗  ", 12.5, RED, True),(b, 12, INK, False)]], spacing=0.95)
text(s, 7.22, 5.78, 5.3, 0.42, [[("That is GRPO's job. SFT is intentionally lightweight (lr 5e-6).", 11.5, SLATE, False, BODY, True)]], spacing=0.95)
footer(s, i)
notes(i, "SFT data transform + purpose", [
    "Explain the data first: 5,000 DesignBench expert traces (multi-turn, with <think> and step-level FEA feedback) at DesignBench/data/sft/train.jsonl.",
    "The WarmstartReasoningTarget reformats each raw trace into the exact inference-time conversation: (1) merge the problem spec + initial-state analysis into one user turn; (2) remap every FEA system message to a user turn prefixed '[Simulation Result]'; (3) extract the grammar action from inside <think> into its own <action> tag. (warmstart_transform.py)",
    "Why: training data must be token-for-token the shape the model will see at rollout. Same shape ⇒ the format learned transfers to deployment.",
    "State the division of labour explicitly: SFT teaches format/grammar/turn-taking; it does NOT teach which member to fix or how much — that is the policy, learned by GRPO. Keep SFT lightweight (low lr) so RL retains plasticity.",
    "This motivates the next slide: if SFT only supervises format, exactly which tokens get loss?",
])

# ---- 11 · LOSS MASKING ------------------------------------------------------
s = slide(); i = nxt()
header(s, "The Key SFT Design: Selective Loss Masking", "Supervise only what the model must generate. Mask everything it merely reads.", tag="02 · MASKING")
text(s, MX, 1.78, 12, 0.34, [[("One training turn, token by token", 14, NAVY, True)]])
# Row of token chips: masked (grey) then supervised (green)
def tok(s, x, y, label, kind):
    w = 0.16 + 0.085*len(label)
    if kind=="mask":
        rect(s, x, y, w, 0.42, fill=CARD3, line=LINE, lw=0.75, radius=0.18)
        text(s, x, y-0.01, w, 0.42, [[(label, 10.5, SLATE, False, MONO)]], align="c", anchor="m")
    else:
        rect(s, x, y, w, 0.42, fill=GREENBG, line=GREENLN, lw=1.0, radius=0.18)
        text(s, x, y-0.01, w, 0.42, [[(label, 10.5, GREEN, True, MONO)]], align="c", anchor="m")
    return w
xx = MX; yy = 2.25
seq = [("[system prompt]","mask"),("PROBLEM: …","mask"),("INITIAL STATE: …","mask"),
       ("[Simulation Result] …","mask"),
       ("<think>","sup"),("…reasoning…","sup"),("</think>","sup"),
       ("<action>","sup"),("SCALE_PARAM(4,r,1.36)","sup"),("</action>","sup")]
for lab,kind in seq:
    w = tok(s, xx, yy, lab, kind)
    xx += w + 0.12
    if xx > 11.6: xx = MX; yy += 0.56
# legend + braces
text(s, MX, 3.02, 12, 0.5, [
  [("■  ", 12, SLATE, True),("masked", 12, SLATE, True),("  (labels = −100, no gradient): everything the model reads — prompt, problem, every solver reply", 12, SLATE, False)],
  [("■  ", 12, GREEN, True),("supervised", 12, GREEN, True),("  (cross-entropy): everything the model writes — its own think / action / answer", 12, GREEN, True)]])
# supervision-ratio bars
text(s, MX, 3.65, 12, 0.34, [[("Share of tokens that receive loss", 14, NAVY, True)]])
def ratio_bar(s, y, name, frac, col):
    text(s, MX, y, 3.0, 0.32, [[(name, 11.5, NAVY, True)]])
    rect(s, 3.7, y, 7.6, 0.34, fill=CARD3, radius=0.12)
    rect(s, 3.7, y, 7.6*frac, 0.34, fill=col, radius=0.12)
    text(s, 3.78, y-0.005, 7.5, 0.34, [[(f"{int(frac*100)}% supervised", 11, WHITE, True)]], anchor="m")
ratio_bar(s, 4.1, "warmstart_reasoning\n(selective)", 0.64, GREEN)
ratio_bar(s, 4.62, "full_sequence\n(naive baseline)", 0.82, MUTE)
text(s, MX, 5.2, 11.8, 0.5, [[("Selective masking keeps the loss off prompt/feedback text, so the model never overfits to reciting the simulator — it preserves RL plasticity.", 11.5, SLATE, False, BODY, True)]], spacing=1.0)
# right note card on default target
accent_card(s, MX, 5.75, 12.1, 1.0, NAVY, fill=CARD2)
text(s, MX+0.3, 5.86, 11.6, 0.32, [[("Pluggable supervision target (research hook)", 13, NAVY, True)]])
text(s, MX+0.3, 6.2, 11.6, 0.5, [[("Default is ", 12, INK, False),("gold_curriculum_warmstart", 12, RED, True, MONO),
    (" — a staged curriculum that supervises only the final assistant turn's think/action/answer. Swap targets via ", 12, INK, False),
    ("data.target_fn", 12, NAVY, True, MONO),(" with no infra changes.", 12, INK, False)]], spacing=1.0)
footer(s, i)
notes(i, "Selective loss masking — the SFT centerpiece", [
    "This answers the feedback's explicit ask: 'what token is masked, what is not.'",
    "Show the token row: everything the model READS — the system prompt, the problem, the initial state, every [Simulation Result] — is masked to labels = −100 (grey, no gradient). Everything the model WRITES — <think>…</think> and <action>…</action> (and <answer> at the end) — is supervised with cross-entropy (green).",
    "The principle in one line: 'supervise what the model must generate; mask what it only reads.' This is implemented in get_loss_mask via regex spans over <think>/<action>/<answer> (targets.py).",
    "Why it matters: if you supervise the prompt/feedback (the naive full_sequence target, ~82% of tokens), the model overfits to reproducing the simulator's text and loses RL plasticity. Selective masking (~64%) keeps loss on the model's own tokens.",
    "Mention the research hook: the supervision target is pluggable (data.target_fn); default gold_curriculum_warmstart tightens supervision to the final turn for a curriculum. No training-infra change to swap.",
    "Note the ratios (64% vs 82%) are illustrative of the two targets from the loss-mask analysis figure.",
])

# ---- 12 · SFT RESULT --------------------------------------------------------
s = slide(); i = nxt()
header(s, "After SFT: Perfect Form, Zero Policy", "The warmstart does exactly its job — and nothing more. That gap is what RL must close.", tag="02 · SFT RESULT")
kpis = [("1.00","grammar success","every action parses & executes", GREEN),
        ("0%","feasibility","solves no held-out problem", RED),
        ("0.41","mean FOS_buckling","far below the 1.5 bar", NAVY)]
x = MX
for num,lab,sub,col in kpis:
    rect(s, x, 2.0, 3.9, 1.95, fill=CARD2, radius=0.05)
    text(s, x, 2.2, 3.9, 0.9, [[(num, 52, col, True, BOLD)]], align="c", anchor="m")
    text(s, x, 3.18, 3.9, 0.34, [[(lab, 14, NAVY, True)]], align="c")
    text(s, x, 3.54, 3.9, 0.34, [[(sub, 11, SLATE, False)]], align="c")
    x += 4.1
accent_card(s, MX, 4.35, 12.1, 1.05, GREEN, fill=GREENBG)
text(s, MX+0.32, 4.48, 11.5, 0.34, [[("Read it correctly", 13.5, GREEN, True)]])
text(s, MX+0.32, 4.84, 11.5, 0.5, [[("Perfect grammar with 0% feasibility is ", 13, INK, False),("success, not failure", 13, GREEN, True),
    (": SFT was asked to teach form, and it did. The model can now say anything in the grammar — it just doesn't yet know what to say.", 13, INK, False)]], spacing=1.0)
accent_card(s, MX, 5.55, 12.1, 1.1, NAVY, fill=CARD3)
text(s, MX+0.32, 5.68, 11.5, 0.34, [[("Why this is the right hand-off to GRPO", 13.5, NAVY, True)]])
text(s, MX+0.32, 6.04, 11.5, 0.55, [[("A reliable format gives RL a clean action space and a stable reference policy to anchor against (KL). The reasoning capacity is present; GRPO only has to point it in the right direction.", 13, INK, False)]], spacing=1.0)
footer(s, i)
notes(i, "SFT result — sets up GRPO", [
    "Three numbers from the warmstart eval: grammar success 1.00, feasibility 0%, mean FOS_buckling 0.41 (results/eval/sft_warmstart_mt25).",
    "Reframe 0% as success: SFT's job was format, and grammar is perfect. The model can produce any valid action; it just lacks the policy for WHICH action. Don't let the audience read 0% as the framework failing.",
    "This is the clean hand-off to GRPO: a reliable grammar gives RL a well-formed action space and a stable reference policy to regularize against (the KL anchor). Capacity is there; RL supplies direction.",
    "Transition to Part 3: 'Now the interesting part — how GRPO turns perfect form into a working policy.'",
])

# =====================  PART 3 — GRPO  =======================================
s = divider("03", "Stage 2 — GRPO Policy Learning", "Group-relative advantages, tree-expanded rollouts, and one dense potential-based reward."); i = nxt()
notes(i, "Divider — Part 3", [
    "Transition: GRPO is where the policy is actually learned. Three design pieces: how rollouts are sampled (group + tree), and how they are scored (potential reward).",
])

# ---- 14 · GRPO GROUP --------------------------------------------------------
s = slide(); i = nxt()
header(s, "GRPO: Learning From a Group of Rollouts", "No value network — the baseline is the group's own mean. Reward better-than-average trajectories.", tag="03 · GRPO")
# diagram: prompt -> K rollouts -> rewards -> advantage
rect(s, MX, 2.2, 2.3, 1.1, fill=NAVY, radius=0.06)
text(s, MX+0.2, 2.33, 1.95, 0.85, [[("problem s0", 13, WHITE, True)],[("(infeasible truss)", 10, "AEB9CC", False)]], anchor="m", spacing=0.95)
# K rollouts
rx = 3.5; ry0 = 1.85; rh=0.42; rgap=0.13
rewards = [("rollout 1","+0.8",GREEN),("rollout 2","−0.3",AMBER),("rollout 3","+1.2",GREEN),
           ("· · ·","",SLATE),("rollout 8","+0.1",GREEN)]
for j,(lab,rv,col) in enumerate(rewards):
    yy = ry0 + j*(rh+rgap)
    rect(s, rx, yy, 2.5, rh, fill=CARD if lab!="· · ·" else WHITE, line=LINE if lab!="· · ·" else None, lw=0.8, radius=0.2)
    text(s, rx+0.18, yy-0.01, 1.6, rh, [[(lab, 11.5, NAVY, True if lab!="· · ·" else False)]], anchor="m")
    if rv: text(s, rx+1.7, yy-0.01, 0.7, rh, [[(rv, 11.5, col, True)]], anchor="m", align="r")
    # connector
    ln = s.shapes.add_connector(1, Inches(MX+2.3), Inches(2.75), Inches(rx), Inches(yy+rh/2))
    ln.line.color.rgb = C(MUTE); ln.line.width = Pt(0.75)
text(s, rx, ry0-0.35, 2.5, 0.3, [[("K = 8 sampled rollouts", 11, SLATE, True)]], align="c")
# advantage box
ax0 = 6.7
rect(s, ax0, 2.2, 3.0, 1.5, fill=REDBG, line=REDLN, lw=1.2, radius=0.06)
text(s, ax0+0.22, 2.34, 2.6, 0.34, [[("group-relative advantage", 12.5, RED, True)]], spacing=0.95)
text(s, ax0+0.22, 2.78, 2.6, 0.5, [[("A = (r − μ) ⁄ σ", 17, NAVY, True, MONO)]], anchor="m")
text(s, ax0+0.22, 3.3, 2.6, 0.34, [[("baseline = group mean μ", 10.5, SLATE, False)]])
right_arrow(s, rx+2.55, 2.82, color=SLATE, w=0.3, h=0.24)
# KL anchor
rect(s, 9.95, 2.2, 2.75, 1.5, fill=CARD2, radius=0.06)
text(s, 10.15, 2.34, 2.4, 0.34, [[("KL anchor", 12.5, NAVY, True)]])
text(s, 10.15, 2.7, 2.4, 0.9, [[("β = 0.04 to the warmstart reference keeps the policy from drifting off the grammar.", 11, SLATE, False)]], spacing=0.98)
# bottom: why GRPO fits
cards = [("No critic","group mean replaces a learned value network — cheaper, stable for LLMs", TEAL),
         ("Self-normalizing","÷σ adapts to per-problem reward scale (FEA magnitudes vary widely)", BLUE),
         ("Anchored","KL to the SFT policy preserves the grammar earned in Stage 1", NAVY)]
x = MX
for t1,t2,col in cards:
    accent_card(s, x, 4.05, 3.93, 1.35, col, fill=CARD)
    text(s, x+0.24, 4.18, 3.5, 0.34, [[(t1, 13.5, NAVY, True)]])
    text(s, x+0.24, 4.55, 3.5, 0.8, [[(t2, 11.5, SLATE, False)]], spacing=0.98)
    x += 4.1
# config strip
rect(s, MX, 5.6, 12.1, 1.05, fill=NAVY, radius=0.03)
for k,(num,lab) in enumerate([("K = 8","rollouts / problem"),("β = 0.04","KL coefficient"),
                              ("lr 5e-6","from warmstart"),("100","GRPO steps")]):
    text(s, MX+0.3+k*3.0, 5.72, 2.8, 0.5, [[(num, 20, WHITE, True, BOLD)]])
    text(s, MX+0.3+k*3.0, 6.22, 2.8, 0.3, [[(lab, 11, "AEB9CC", False)]])
footer(s, i)
notes(i, "GRPO group rollout", [
    "Explain GRPO simply: for each problem, sample a GROUP of K=8 rollouts; the advantage of each is its reward minus the group mean, divided by the group std. No separate value network — the group is its own baseline.",
    "Walk the diagram: one problem → 8 rollouts, each gets a scalar reward → advantage A=(r−μ)/σ → policy-gradient update, regularized by a KL term (β=0.04) back to the SFT warmstart.",
    "Three reasons GRPO fits this setting: (1) no critic to train — cheaper and more stable for a 14B LLM; (2) ÷σ self-normalizes across problems whose FEA reward magnitudes differ a lot; (3) the KL anchor protects the grammar earned in Stage 1.",
    "Config anchors (grpo_base.yaml / grpo_mt_01b.yaml): group_size=8, kl_coef=0.04, lr 5e-6, 100 steps. grpo_trainer.py:159 sets num_generations=group_size.",
    "Next: a richer way to sample the group — tree expansion.",
])

# ---- 15 · TREE / MULTI-BRANCH ----------------------------------------------
s = slide(); i = nxt()
header(s, "Multi-Branch Rollouts: Tree-Expanded Advantage", "Optionally expand each state into a stratified action tree for a lower-variance, lookahead baseline.", tag="03 · ROLLOUT")
# tree diagram
rootx, rooty = 1.6, 3.6
rect(s, rootx-0.65, rooty-0.4, 1.3, 0.8, fill=NAVY, radius=0.1)
text(s, rootx-0.65, rooty-0.42, 1.3, 0.8, [[("state s", 12.5, WHITE, True)],[("D = 1", 10, "AEB9CC", False)]], align="c", anchor="m", spacing=0.95)
classes = [("SCALE", GREEN, "γΦ = +1.2"),("ADD", BLUE, "+0.4"),("MODIFY", TEAL, "+0.9"),
           ("REMOVE", AMBER, "−0.2"),("MOVE", PURPLE, "+0.3")]
cy0 = 1.95; chx = 4.4; ch=0.66; cgap=0.30
for j,(cl,col,val) in enumerate(classes):
    yy = cy0 + j*(ch+cgap)
    accent_card(s, chx, yy, 2.9, ch, col, fill=CARD)
    text(s, chx+0.22, yy+0.06, 2.0, 0.32, [[(cl, 12.5, NAVY, True, MONO)]])
    text(s, chx+0.22, yy+0.36, 2.0, 0.28, [[("one action class", 9.5, SLATE, False)]])
    text(s, chx+2.0, yy+0.02, 0.8, ch, [[(val, 11.5, col, True)]], anchor="m", align="r")
    ln = s.shapes.add_connector(1, Inches(rootx+0.65), Inches(rooty), Inches(chx), Inches(yy+ch/2))
    ln.line.color.rgb = C(MUTE); ln.line.width = Pt(1.0)
text(s, chx, cy0-0.32, 3.5, 0.3, [[("b = 5 branches · one per class", 10.5, SLATE, True)]])
# advantage formula
ax0 = 7.7
accent_card(s, ax0, 1.95, 5.0, 1.55, RED, fill=REDBG)
text(s, ax0+0.25, 2.08, 4.6, 0.34, [[("tree-expanded advantage", 13, RED, True)]])
text(s, ax0+0.25, 2.5, 4.6, 0.5, [[("A_k = γΦ(s_H^k) − mean_j γΦ(s_H^j)", 13.5, NAVY, True, MONO)]], anchor="m", spacing=0.95)
text(s, ax0+0.25, 3.0, 4.6, 0.45, [[("baseline = mean potential over sibling branches", 10.5, SLATE, False)]], spacing=0.95)
cards2 = [("Stratified coverage","one branch per action class ⇒ the group spans genuinely different moves, not 8 near-duplicates", BLUE),
          ("Lower variance","sibling-mean baseline is tighter than a single-rollout return", TEAL),
          ("Bounded cost","D=1, b=5 ⇒ ~20 FEA calls per state; verified in verify_tree_expansion.py", NAVY)]
yy = 3.7
for t1,t2,col in cards2:
    accent_card(s, ax0, yy, 5.0, 0.95, col, fill=CARD)
    text(s, ax0+0.25, yy+0.10, 4.6, 0.32, [[(t1, 12.5, NAVY, True)]])
    text(s, ax0+0.25, yy+0.42, 4.6, 0.5, [[(t2, 10.5, SLATE, False)]], spacing=0.95)
    yy += 1.04
text(s, MX, 6.78, 12.1, 0.35, [[("Champion run keeps it off (use_tree_expansion=false) — it is a variance-reduction option, not required for the headline result.", 11, SLATE, False, BODY, True)]])
footer(s, i)
notes(i, "Tree-expanded / multi-branch rollout", [
    "This answers the feedback's 'multi-branch rollout' ask. Beyond the flat group of 8, GRPO can expand each state into a shallow action TREE.",
    "Depth D=1, branching b=5: sample one candidate per action class (SCALE, ADD, MODIFY, REMOVE, MOVE) — stratified so the branches are genuinely different moves rather than near-duplicates.",
    "Each branch is scored by the same potential Φ; the advantage of branch k is γΦ(s_H^k) minus the MEAN potential over its sibling branches — a tighter, lookahead baseline (grpo_trainer.py:278–342; posterior/tree_expansion.py).",
    "Benefits: stratified coverage of the action space, lower-variance advantages, and bounded cost (~20 FEA calls/state, validated in verify_tree_expansion.py).",
    "Be honest: the 68% champion has tree expansion OFF (use_tree_expansion=false). Present it as an available variance-reduction mechanism in the framework, not as the source of the headline number.",
])

# ---- 16 · REWARD ------------------------------------------------------------
s = slide(); i = nxt()
header(s, "The Reward: One Dense Potential-Based Signal", "No reward zoo — a single term that is dense, physics-grounded, and provably can't change the optimum.", tag="03 · REWARD")
# formula block
code_block(s, MX, 1.9, 7.3, 2.55, [
    [("# per-step shaping reward", CODEDIM)],
    [("r_env  =  γ·Φ(s′) − Φ(s)", CODETXT)],
    [("", CODETXT)],
    [("# Lagrangian potential", CODEDIM)],
    [("Φ(s) = log(m₀/m) − α · V(s)", CODETXT)],
    [("V(s) = softplus(1.5 − FOS_b)", CODETXT)],
    [("     + softplus(1.5 − FOS_y)", CODETXT)],
    [("     + softplus(δ/0.01 − 1)", CODETXT)],
], title="rewards.py · LagrangianPotentialReward")
# telescoping
accent_card(s, MX, 4.6, 7.3, 1.0, GREEN, fill=GREENBG)
text(s, MX+0.28, 4.72, 6.8, 0.32, [[("Telescopes to the true objective", 12.5, GREEN, True)]])
text(s, MX+0.28, 5.06, 6.8, 0.5, [[("Σ γᵗ r_env  =  γᴴΦ(s_H) − Φ(s₀)", 13, NAVY, True, MONO),
    ("  — only the endpoints matter.", 11.5, SLATE, False)]], anchor="m", spacing=0.95)
# properties
props = [("Dense","every turn gets signal — no waiting for a terminal feasible/infeasible verdict", TEAL),
         ("Policy-invariant","potential shaping (Ng et al. 1999): shaping can't move the optimum", NAVY),
         ("Physics-grounded","Φ is computed straight from the FEA state the loop produces", BLUE),
         ("One knob","α = 5 prices the constraint violation; γ = 0.99 discounts", RED)]
yy = 1.95
for t1,t2,col in props:
    accent_card(s, 8.15, yy, 4.55, 1.05, col, fill=CARD)
    text(s, 8.4, yy+0.11, 4.1, 0.32, [[(t1, 13.5, NAVY, True)]])
    text(s, 8.4, yy+0.44, 4.1, 0.55, [[(t2, 11, SLATE, False)]], spacing=0.96)
    yy += 1.16
footer(s, i)
notes(i, "Potential-based reward", [
    "The framework deliberately avoids a 'reward zoo'. The champion uses ONE term: a potential-based per-step shaping reward r_env = γΦ(s′) − Φ(s).",
    "The potential Φ(s) = log(m₀/m) − α·V(s): a mass-utility term minus α times a smooth violation score V (softplus deficits on buckling FOS, yielding FOS, and deflection). (potential.py:47–68; rewards.py:388–420.)",
    "Two properties make it well-behaved: DENSE — every turn gets a gradient, no sparse terminal-only signal; and POLICY-INVARIANT — by Ng/Harada/Russell 1999, potential shaping telescopes to γᴴΦ(s_H) − Φ(s₀), so it cannot change which policy is optimal, only speed learning.",
    "It is physics-grounded (computed from the FEA state) and has effectively ONE hyperparameter: α=5 (constraint price), γ=0.99. We'll show α is non-monotone in the ablations.",
    "Config: reward_fn=composite with reward_weights.lagrangian_potential=1.0, cost_fn=null (grpo_mt_01b.yaml).",
])

# ---- 17 · ENGINEERING -------------------------------------------------------
s = slide(); i = nxt()
header(s, "Making Multi-Turn RL Tractable", "The cost lives in the rollout loop, not the loss — so the rollout is batched and turn-major.", tag="03 · SYSTEMS")
kpis = [("~511 s","per GRPO step","8×H100, measured", NAVY),
        ("4–6k","tokens / rollout","genuine multi-turn CoT", TEAL),
        ("~14 h","for 100 steps","under the 24 h wall", GREEN)]
x = MX
for num,lab,sub,col in kpis:
    rect(s, x, 1.95, 3.93, 1.5, fill=CARD2, radius=0.05)
    text(s, x, 2.1, 3.93, 0.8, [[(num, 38, col, True, BOLD)]], align="c", anchor="m")
    text(s, x, 2.86, 3.93, 0.32, [[(lab, 13, NAVY, True)]], align="c")
    text(s, x, 3.18, 3.93, 0.3, [[(sub, 10.5, SLATE, False)]], align="c")
    x += 4.1
accent_card(s, MX, 3.7, 6.0, 2.9, NAVY, fill=CARD)
text(s, MX+0.28, 3.84, 5.5, 0.34, [[("Turn-major batched rollout", 14, NAVY, True)]])
for k,b in enumerate([
    "At each turn, batch ALL still-active rollouts into one generate() call",
    "Left-padded, KV-cached, memory-chunked (gen_batch_chunk = 8)",
    "Finished rollouts drop out; the batch shrinks turn by turn",
    "max_turn_tokens = 1536 — full reasoning + one action per turn"]):
    text(s, MX+0.28, 4.26+k*0.56, 5.5, 0.5, [[("•  ", 12, RED, True),(b, 11.5, INK, False)]], spacing=0.96)
accent_card(s, 6.85, 3.7, 5.85, 2.9, AMBER, fill=CARD3)
text(s, 7.12, 3.84, 5.35, 0.34, [[("Why vLLM is off for multi-turn", 14, NAVY, True)]])
text(s, 7.12, 4.26, 5.35, 1.1, [[("vLLM generates a whole completion in one shot. The multi-turn contract is ", 12, INK, False),
    ("stateful", 12, NAVY, True),
    (" — each turn's prompt depends on the previous FEA result — so the loop uses per-turn HF generate() instead.", 12, INK, False)]], spacing=1.0)
text(s, 7.12, 5.5, 5.35, 1.0, [[("vLLM (tensor-parallel 8) still serves the single-turn path; the trade is throughput for protocol fidelity — and fidelity is what made the policy learnable.", 12, SLATE, False, BODY, True)]], spacing=1.0)
footer(s, i)
notes(i, "Engineering / systems", [
    "Set expectations: multi-turn RL is expensive because the cost is in the rollout loop (generation + FEA), not the loss. ~511 s/step on 8×H100, 4–6k tokens/rollout, ~14 h for 100 steps (under the 24 h wall).",
    "The enabling trick: a turn-major BATCHED rollout. At each turn, all still-active rollouts are batched into one generate() call (left-padded, KV-cached, chunked by memory). Finished rollouts drop out so the batch shrinks. (multiturn_rollout.py:13–15, 58–117.)",
    "Why vLLM is disabled here: vLLM emits a whole completion in one shot, but the multi-turn contract is stateful — each turn's prompt depends on the prior FEA reply — so per-turn HF generate() is required. vLLM still serves the single-turn path. (grpo_trainer.py:211–215.)",
    "Frame the trade-off as principled: we give up generation throughput to keep train/test protocol fidelity — and fidelity is exactly what made the policy learnable.",
    "This closes the framework half of the talk; next is evidence.",
])

# =====================  PART 4 — RESULTS  ====================================
s = divider("04", "Results & Analysis", "The progression, the training dynamics, the ablations — and the experiment that reframes it all."); i = nxt()
notes(i, "Divider — Part 4", [
    "Transition: with the framework defined, the results read as validation of the two environment-design principles from Part 1.",
])

# ---- 19 · PROGRESSION -------------------------------------------------------
s = slide(); i = nxt()
header(s, "From 0% to 68% Feasible", "Five configurations of the same framework. Each row isolates one design choice.", tag="04 · PROGRESSION")
image_fit(s, os.path.join(FIG,"eval_feas.png"), MX, 1.85, 6.7, 4.4, valign="t")
# right rail explanation
steps = [("Multi-turn protocol","unfreezes learning","0% → 12%", NAVY),
         ("+ critical-member observation","the decisive lever","12% → 68%", RED),
         ("+ hard mass rule","over-corrects","68% → 52%", AMBER)]
text(s, 7.55, 1.95, 5.2, 0.4, [[("What each step changed", 14, NAVY, True)]])
yy = 2.45
for t1,t2,delta,col in steps:
    accent_card(s, 7.55, yy, 5.15, 1.05, col, fill=CARD)
    text(s, 7.8, yy+0.10, 4.7, 0.34, [[(t1, 13, NAVY, True)]])
    text(s, 7.8, yy+0.44, 3.2, 0.3, [[(t2, 11, SLATE, False)]])
    text(s, 10.7, yy+0.30, 1.85, 0.45, [[(delta, 14, col, True)]], align="r", anchor="m")
    yy += 1.18
text(s, 7.55, yy+0.05, 5.15, 0.9, [[("The algorithm (GRPO) and reward are held fixed across all five. The movement comes entirely from the ", 11.5, INK, False),
    ("environment", 11.5, RED, True),(" — protocol then observation.", 11.5, INK, False)]], spacing=1.0)
footer(s, i)
notes(i, "The 0%→68% progression", [
    "This is the headline chart (real eval data, results/report/deck/eval_feas.png). Five configurations of ONE framework, GRPO and reward held fixed throughout.",
    "Read the steps on the right as an ablation of environment design, NOT as a list of bug fixes: (1) making the rollout multi-turn unfreezes learning, 0%→12%; (2) adding the critical-member to the observation is the decisive lever, 12%→68%; (3) adding a HARD mass rule over-corrects, 68%→52%.",
    "Hammer the structural point: the model, the optimizer, and the reward did not change between these bars. Only the environment did. That is the evidence for 'the bottleneck was the environment.'",
    "The +grammar bar (4%) is a preview of the ablations slide — adding a grammar reward actually hurt.",
    "Numbers are 25-problem evals (±~8%) except the champion, which has a 50-problem confirmation at 68%.",
])

# ---- 20 · TRAINING DYNAMICS -------------------------------------------------
s = slide(); i = nxt()
header(s, "Is It Learning? Watch the KL", "Reward is noisy batch-to-batch, so policy movement (KL from the reference) is the honest signal.", tag="04 · DYNAMICS")
image_fit(s, os.path.join(FIG,"kl_curves.png"), MX, 1.8, 7.4, 4.7, valign="t")
text(s, 8.05, 1.9, 4.7, 0.4, [[("Reading the curves", 14, NAVY, True)]])
rows = [("single-turn","frozen at ~0.003 for all 200 steps — the policy never moves", SLATE),
        ("multi-turn, no fb","learnable but slow", BLUE),
        ("champion","climbs to ~0.035, crosses the 0.01 bar by step ~35", RED)]
yy = 2.4
for t1,t2,col in rows:
    accent_card(s, 8.05, yy, 4.65, 1.05, col, fill=CARD)
    text(s, 8.3, yy+0.10, 4.2, 0.32, [[(t1, 13, NAVY, True)]])
    text(s, 8.3, yy+0.42, 4.2, 0.55, [[(t2, 11, SLATE, False)]], spacing=0.96)
    yy += 1.16
text(s, 8.05, yy+0.05, 4.65, 1.0, [[("≈10× more policy movement with the complete observation. More information ⇒ a sharper learning signal, not just a better final score.", 11.5, INK, False)]], spacing=1.0)
footer(s, i)
notes(i, "Training dynamics — KL", [
    "Explain why KL, not reward: r_env magnitude is problem-dependent and each batch samples different problems, so the reward curve (middle of the original fig) is too noisy to read step-to-step. KL from the reference policy is the clean 'is it moving?' signal.",
    "Real data (kl_curves.png from logs/*/metrics.jsonl). Single-turn run is frozen at ~0.003 for all 200 steps — a flat policy. Multi-turn no-feedback is learnable but slow. The champion climbs to ~0.035 and crosses the 0.01 'is it learning?' bar by ~step 35.",
    "≈10× more movement once the observation is complete — the point is that better information yields a SHARPER gradient, not merely a higher endpoint.",
    "The grammar-reward run spikes (clipped at top) — unstable; ties to the next slide's ablation.",
])

# ---- 21 · ABLATIONS ---------------------------------------------------------
s = slide(); i = nxt()
header(s, "Reward Ablations: The Simple Reward Wins", "One change per run. Nothing beat the single dense potential at α = 5.", tag="04 · ABLATIONS")
image_fit(s, os.path.join(FIG,"alpha_sweep.png"), MX, 1.9, 5.6, 4.4, valign="t")
text(s, 6.5, 1.9, 6.2, 0.4, [[("What else we tried", 14, NAVY, True)]])
tried = [("α sweep {2, 5, 10}","non-monotone — peaks at 5; α=10 destabilizes and hurts", RED),
         ("+ grammar reward","did not restore format; feasibility regressed (drift isn't reward-addressable)", AMBER),
         ("+ explicit mass rule","over-conservative — 68% → 52%", PURPLE),
         ("Verdict","the clean single-term r_env at α=5 stays best; leverage was the environment", NAVY)]
yy = 2.45
for t1,t2,col in tried:
    accent_card(s, 6.5, yy, 6.2, 0.95, col, fill=CARD if t1!="Verdict" else CARD2)
    text(s, 6.75, yy+0.10, 5.7, 0.32, [[(t1, 13, NAVY, True)]])
    text(s, 6.75, yy+0.42, 5.7, 0.5, [[(t2, 11, SLATE, False)]], spacing=0.95)
    yy += 1.04
footer(s, i)
notes(i, "Reward ablations", [
    "Methodology: change exactly one thing per run (clean ablations logged in docs/ablation_ledger.md).",
    "α sweep (no feedback): feasibility is NON-monotone — 8% at α=2, 12% at α=5, 4% at α=10. This refutes the design-doc prediction that higher α is always better; α=10 over-weights the constraint and destabilizes the FOS landscape. Peak at α=5.",
    "Adding a grammar-compliance reward did NOT recover format and slightly hurt feasibility — grammar drift is fragility-induced (tiny policy moves destroy the thin r=32 warmstart format), not something a reward can fix.",
    "Adding an explicit 'mass exceeds limit, reduce material' rule made the policy over-conservative: it hugged FOS=1.5 and traded mass-overshoot fails for FOS-undershoot fails, 68%→52%.",
    "Verdict: the clean single-term reward at α=5 is best. Combined with the progression slide, the message is consistent — the leverage was the environment, not reward engineering.",
])

# ---- 22 · EVAL DETAIL -------------------------------------------------------
s = slide(); i = nxt()
header(s, "Held-Out Evaluation: The Grammar–Strength Trade", "Feasibility is the headline, but the secondary metrics tell the rest of the story.", tag="04 · EVAL")
image_fit(s, os.path.join(FIG,"eval_grammar_fos.png"), MX, 1.85, 7.1, 4.0, valign="t")
# table-ish summary on right
text(s, 7.95, 1.9, 4.8, 0.35, [[("Champion vs warmstart", 14, NAVY, True)]])
rowdata = [("metric","warmstart","champion"),
           ("feasibility","0%","68%"),
           ("grammar success","1.00","0.67"),
           ("mean FOS_b","0.41","1.38–1.63")]
yy = 2.35
for k,(a,b,c) in enumerate(rowdata):
    hdr = k==0
    fill = NAVY if hdr else (CARD2 if k%2 else CARD)
    rect(s, 7.95, yy, 4.75, 0.5, fill=fill, line=None if hdr else LINE, lw=0.6)
    tc = WHITE if hdr else NAVY
    text(s, 8.1, yy+0.02, 2.0, 0.46, [[(a, 11.5, tc, hdr)]], anchor="m")
    text(s, 10.0, yy+0.02, 1.3, 0.46, [[(b, 11.5, SLATE if not hdr else WHITE, hdr)]], anchor="m", align="c")
    text(s, 11.3, yy+0.02, 1.3, 0.46, [[(c, 11.5, RED if (not hdr and k>0) else (WHITE if hdr else NAVY), True)]], anchor="m", align="c")
    yy += 0.52
accent_card(s, 7.95, 4.5, 4.75, 1.9, NAVY, fill=CARD3)
text(s, 8.2, 4.62, 4.3, 0.34, [[("The trade is a knob, not a law", 13, NAVY, True)]])
text(s, 8.2, 4.98, 4.3, 1.4, [[("The warmstart formats perfectly (1.00) but solves nothing. GRPO trades some grammar (→ ~0.67) for large FOS/feasibility gains. The ~33% of turns lost to grammar drift is the main recoverable inefficiency.", 11.5, INK, False)]], spacing=1.0)
footer(s, i)
notes(i, "Held-out evaluation detail", [
    "Define the metrics precisely (eval_checkpoint.py:340–350): feasibility_rate = fraction of problems where all three constraints pass (is_feasible); grammar_success = fraction of turns whose action parses & executes; mean FOS_buckling.",
    "The grouped bars (real data) show the trade: warmstart has grammar 1.00 but FOS far below the line; training pushes FOS up while grammar drifts to ~0.45–0.67.",
    "Table contrasts warmstart vs champion directly: 0%→68% feasibility, 1.00→0.67 grammar, 0.41→1.38–1.63 FOS.",
    "Key nuance: the grammar–strength trade is a KNOB, not a law. ~33% of turns are wasted on grammar drift — that's recoverable headroom (KL-to-warmstart tuning), not a fundamental ceiling.",
    "Caveat: 25-problem cells carry ±~8%; champion confirmed at 68% on a 50-problem eval.",
])

# ---- 23 · INFORMATION BOTTLENECK -------------------------------------------
s = slide(); i = nxt()
header(s, "The Experiment That Reframes Everything", "A strong reasoner that can see the failing member solves the task with no RL at all.", tag="04 · THE LEVER")
big = [("4 / 6","30B reasoner + critical-member\nobservation, zero-shot (no RL)", GREEN),
       ("0 / 6","same model, same prompt,\nwithout the member named", RED),
       ("2.12","best FOS from a short\nsearch over edits", NAVY)]
x = MX
for num,lab,col in big:
    rect(s, x, 1.95, 3.93, 2.55, fill=CARD2, radius=0.05)
    text(s, x, 2.2, 3.93, 1.1, [[(num, 58, col, True, BOLD)]], align="c", anchor="m")
    text(s, x, 3.45, 3.93, 0.9, [[(lab, 12.5, SLATE, False)]], align="c", spacing=0.98)
    x += 4.1
accent_card(s, MX, 4.8, 12.1, 1.85, RED)
text(s, MX+0.35, 4.95, 11.5, 0.4, [[("What this proves", 15, RED, True)]])
text(s, MX+0.35, 5.4, 11.5, 1.2, [[("Flipping a single field in the observation moves a frozen, untrained model from ", 14, INK, False),
    ("0/6 to 4/6", 14, GREEN, True),
    (". The reasoning capacity was always present — the bottleneck was an ", 14, INK, False),
    ("incomplete observation", 14, RED, True),
    (". This is why we call the headline a property of the environment, not of the RL algorithm. RL then amplifies and stabilizes the same competence into the trained 14B policy.", 14, INK, False)]], spacing=1.05)
footer(s, i)
notes(i, "The information bottleneck — the punchline", [
    "This is the intellectual climax. Run as a controlled experiment: take a strong 30B reasoner zero-shot in the same multi-turn loop. With the critical member named in the observation it solves 4/6; remove ONLY that field and it solves 0/6.",
    "The only variable is one field in the observation. A frozen, untrained model swings from 0/6 to 4/6. That isolates the cause: the limit was an incomplete observation, not reasoning ability or the RL algorithm.",
    "This is the justification for the talk's thesis and for framing the gains as environment design. RL's role is to amplify and stabilize this same competence into the trained 14B policy (68%).",
    "Tree-search corollary: a short search over edits reaches FOS 2.12 — the action space is rich enough; the policy just needs the right signal.",
    "Land the line: 'We didn't make the model smarter. We let it see what it needed to see.'",
])

# ---- 24 · SOLVED TRAJECTORY -------------------------------------------------
s = slide(); i = nxt()
header(s, "What Success Looks Like", "The champion solving auto_problem_000 — FOS_buckling climbs monotonically past 1.5.", tag="04 · TRAJECTORY")
image_fit(s, os.path.join(FIG,"fos_trajectory.png"), MX, 1.9, 7.7, 4.4, valign="t")
text(s, 8.2, 1.95, 4.5, 0.4, [[("The trajectory", 14, NAVY, True)]])
seqr = [("step 0","SCALE_PARAM(4, r, 1.36)","0.44→0.48"),
        ("step 1","SCALE_PARAM(7, r, 1.50)","0.48→0.51"),
        ("steps 2–6","grow critical members","climbing"),
        ("step 7","feasible","FOS_b 1.69")]
yy = 2.45
for a,b,c in seqr:
    rect(s, 8.2, yy, 4.5, 0.78, fill=CARD, line=LINE, lw=0.8, radius=0.06)
    text(s, 8.4, yy+0.08, 4.1, 0.3, [[(a, 11, SLATE, True),("    ",11,SLATE,False),(c, 11, RED, True)]])
    text(s, 8.4, yy+0.40, 4.1, 0.32, [[(b, 11.5, NAVY, True, MONO)]])
    yy += 0.88
rect(s, 8.2, yy+0.02, 4.5, 0.85, fill=GREENBG, line=GREENLN, lw=1.0, radius=0.06)
text(s, 8.4, yy+0.12, 4.1, 0.32, [[("FEASIBLE in 7 turns", 13.5, GREEN, True)]])
text(s, 8.4, yy+0.46, 4.1, 0.3, [[("teacher reasons, 14B grounds each action", 10.5, SLATE, False)]])
footer(s, i)
notes(i, "A solved trajectory", [
    "Concrete close to the results: the champion solving auto_problem_000 in 7 turns, FOS_buckling 0.44 → 1.69, crossing the 1.5 line monotonically (status_report §3).",
    "Each step targets a critical member named by the feedback: step 0 grows M4, step 1 grows M7, and so on. No wasted edits on irrelevant members.",
    "This is the behaviour the whole study was chasing: grounded reasoning that acts on the right member every turn.",
    "Use it to transition from 'what works' to 'what still breaks' — 68% means ~32% still fail.",
])

# =====================  PART 5 — CLOSING  ====================================
# ---- 25 · REMAINING 32% -----------------------------------------------------
s = slide(); i = nxt()
header(s, "The Remaining 32%", "Adding observation information has hit diminishing returns. The rest splits three ways.", tag="05 · GAPS")
# donut-ish: 68 feasible
rect(s, MX, 2.1, 3.7, 3.4, fill=CARD2, radius=0.05)
text(s, MX, 2.55, 3.7, 1.0, [[("68%", 56, GREEN, True, BOLD)]], align="c", anchor="m")
text(s, MX, 3.7, 3.7, 0.34, [[("feasible", 15, NAVY, True)]], align="c")
text(s, MX, 4.1, 3.7, 0.8, [[("the remaining 32% breaks into three failure modes →", 11.5, SLATE, False)]], align="c", spacing=0.98)
modes = [("Mass over-shoot  (3)","FOS fixed but mass budget exceeded","soft FOS↔mass reward balance", AMBER),
         ("Degenerate FEA  (4)","structurally broken / very heavy states","robust state handling", PURPLE),
         ("Grammar drift  (~33% turns)","policy forgets the warmstart format","format-protecting KL", RED)]
yy = 2.1
for t1,t2,t3,col in modes:
    accent_card(s, 4.6, yy, 8.1, 1.05, col, fill=CARD)
    text(s, 4.86, yy+0.11, 7.6, 0.32, [[(t1, 13.5, NAVY, True)]])
    text(s, 4.86, yy+0.44, 4.4, 0.3, [[(t2, 11, SLATE, False)]])
    text(s, 9.45, yy+0.30, 3.0, 0.5, [[("→ "+t3, 11.5, col, True)]], anchor="m")
    yy += 1.16
text(s, 4.6, yy+0.0, 8.1, 0.4, [[("A hard prompt rule over-corrects (the mass regression proved it) — the next gains must come from softer, structural changes.", 11, SLATE, False, BODY, True)]], spacing=0.98)
footer(s, i)
notes(i, "The remaining 32%", [
    "Be candid about the ceiling: the 'add missing information to the observation' lever produced 12%→68% but has saturated (the mass tweak regressed).",
    "The remaining failures split three ways: (1) mass over-shoot on 3 problems — FOS fixed but mass exceeded; needs SOFT reward-side balancing, not a hard rule; (2) degenerate FEA on 4 — structurally broken or very heavy intermediate states; needs robust state handling; (3) grammar drift wastes ~33% of turns — the policy forgets the warmstart format; needs a format-protecting KL constraint.",
    "The unifying lesson from the ablations: hard prompt imperatives over-correct (mass), so the next gains are softer/structural changes — sets up the roadmap.",
])

# ---- 26 · ROADMAP -----------------------------------------------------------
s = slide(); i = nxt()
header(s, "Roadmap to >90%", "Each step targets a named failure mode — the gap is a list, not a mystery.", tag="05 · NEXT")
road = [("01","Soft FOS↔mass balancing","reward-side trade-off so the policy fixes buckling without over-shooting mass", NAVY),
        ("02","Format-protecting KL","keep the policy near the warmstart grammar to reclaim the ~33% wasted turns", NAVY),
        ("03","Robust degenerate states","handle broken / very heavy intermediates so the solver returns usable signal", NAVY),
        ("★","Distillation (in reserve)","the 30B teacher already produces 67%-solving traces — distill into the 14B if the above stall", RED)]
x = MX; cw = 2.9; gap=0.17
for num,t1,t2,col in road:
    accent_card(s, x, 2.0, cw, 3.7, col, fill=CARD)
    rect(s, x+0.28, 2.3, 0.72, 0.72, fill=NAVY if num!="★" else RED, radius=0.18)
    text(s, x+0.28, 2.26, 0.72, 0.72, [[(num, 20, WHITE, True, BOLD)]], align="c", anchor="m")
    text(s, x+0.28, 3.2, cw-0.5, 0.8, [[(t1, 15, NAVY, True)]], spacing=0.96)
    text(s, x+0.28, 4.1, cw-0.5, 1.5, [[(t2, 11.5, SLATE, False)]], spacing=1.0)
    if num in ("01","02","03"): chevron(s, x+cw+0.0, 3.65, color=SLATE, size=0.26)
    x += cw+gap
text(s, MX, 6.0, 12.1, 0.5, [[("Targets in priority order; distillation is the fallback if the reward-side and KL changes plateau.", 12, SLATE, False, BODY, True)]])
footer(s, i)
notes(i, "Roadmap", [
    "Three priorities on the main path, each mapped to a failure mode from the previous slide: (1) soft FOS↔mass reward balancing; (2) a format-protecting KL constraint to stop grammar drift; (3) robust handling of degenerate FEA states.",
    "One reserve: distillation. The 30B teacher already produces trajectories that solve ~67%, so distilling those into the 14B is the fallback if the reward-side and KL changes plateau.",
    "Message: the path to >90% is concrete and prioritized — a list of named fixes, not open-ended research.",
])

# ---- 27 · TAKEAWAYS ---------------------------------------------------------
s = slide(NAVY); i = nxt()
rect(s, 0, 0, 0.18, SH, fill=RED)
text(s, MX+0.3, 0.7, 12, 0.7, [[("Key Takeaways", 30, WHITE, True, BOLD)]])
line_h(s, MX+0.32, 1.5, 3.0, color=RED, weight=2.5)
tk = [("1","The bottleneck was the environment, not the model","Same GRPO, same reward — protocol fidelity and a complete observation drove 0%→68%."),
      ("2","SFT teaches form; selective masking is how","Supervise what the model writes, mask what it reads — perfect grammar, preserved RL plasticity."),
      ("3","GRPO + one dense potential reward is enough","Group/tree-relative advantages and a policy-invariant Φ at α=5 beat every reward add-on."),
      ("4","Information completeness > raw reasoning","A zero-shot reasoner goes 0/6 → 4/6 from one observation field. RL amplifies that competence.")]
yy = 1.85
for num,t1,t2 in tk:
    rect(s, MX+0.3, yy, 11.9, 1.12, fill=NAVY3, radius=0.05)
    rect(s, MX+0.55, yy+0.25, 0.62, 0.62, fill=RED, radius=0.2)
    text(s, MX+0.55, yy+0.21, 0.62, 0.62, [[(num, 20, WHITE, True, BOLD)]], align="c", anchor="m")
    text(s, MX+1.45, yy+0.14, 10.4, 0.4, [[(t1, 16, WHITE, True)]])
    text(s, MX+1.45, yy+0.58, 10.4, 0.5, [[(t2, 12, "C4CAD4", False)]], spacing=0.96)
    yy += 1.22
text(s, MX+0.3, 6.95, 12, 0.3, [[("Qwen3-14B · DesignBench truss optimization · docs/status_report.md", 10.5, SLATE2, False)]])
notes(i, "Key takeaways", [
    "Close on the four lessons, in the order the talk built them.",
    "1 — The bottleneck was the environment: with GRPO and the reward held fixed, protocol fidelity then a complete observation drove 0%→68%.",
    "2 — SFT's contribution is form, achieved by selective loss masking (supervise written tokens, mask read tokens) — perfect grammar without sacrificing RL plasticity.",
    "3 — Simplicity won on the algorithm side: GRPO with group/tree-relative advantages and one policy-invariant potential (α=5) beat every reward add-on.",
    "4 — Information completeness beats raw reasoning capacity: the 0/6→4/6 zero-shot swing is the proof; RL amplifies that competence into the trained policy.",
    "End with the one-liner: we didn't make the model smarter — we built an environment that let a capable model show what it already knew.",
])

# ----------------------------------------------------------- write notes ------
for idx, sl in enumerate(prs.slides, start=1):
    if idx in NOTES:
        title, bullets = NOTES[idx]
        tf = sl.notes_slide.notes_text_frame
        tf.text = f"SLIDE {idx} — {title}"
        for b in bullets:
            p = tf.add_paragraph(); p.text = "• " + b

prs.save(OUT)
print("WROTE", OUT, "with", len(prs.slides.__iter__.__self__._sldIdLst), "slides")

# ----------------------------------------------------------- companion md -----
with open(NOTESMD, "w") as f:
    f.write("# DesignBench SFT + GRPO — Status Deck (v2)\n")
    f.write("## Per-slide talking points\n\n")
    f.write("_Companion to `DesignBench_SFT_GRPO_Status_2.pptx`. "
            "Use these bullets to check the narrative flows slide to slide._\n\n")
    for idx in sorted(NOTES):
        title, bullets = NOTES[idx]
        f.write(f"### Slide {idx} — {title}\n")
        for b in bullets:
            f.write(f"- {b}\n")
        f.write("\n")
print("WROTE", NOTESMD)
