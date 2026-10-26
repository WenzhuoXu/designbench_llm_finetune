"""Steady-state thermal network, v2: heat-sink placement and sizing, spreader
links and power-block placement (full typed action set).

Physics
-------
A lumped steady-state conduction network of n nodes (12..100). Node i
dissipates P_i [W] = a fixed background power + the power of the block placed
on it (if any). Fixed base links (plate / module / layer conduction) with
conductances w_ij [W/K] form the weighted graph Laplacian L_base. Every node
also loses heat to ambient by natural convection h0 [W/K] (one value per
problem). Installed spreader links (heat pipes, straps, thermal pillars) add
their conductance w_l between their two end nodes. A heat sink mounted on node
j adds its conductance g [W/K] from node j to ambient. The temperature rise
over ambient T [K] solves

    (L_base + L_links(active) + diag(h0 + G)) T = P,   G_j = g of the sink on j

The matrix is a symmetric M-matrix and positive definite for h0 > 0, so every
design has a unique, finite, non-negative solution (one dense numpy solve per
evaluation; n <= 100, ~0.1 ms). T decreases monotonically in every g and in
every installed link's conductance.

Elements (a FIXED list per problem; indices never change in a trajectory)
------------------------------------------------------------------------
  [0, K)          heat sinks, K = 2..6
      site    categorical: code 0..S-1 into the problem's list of allowed
              mounting sites (each a node with its own mounting cost
              coefficient k). Rounded to the nearest code and clamped. Setting
              a code already held by another sink SWAPS the two sinks' sites
              (each keeps its own g), so two sinks never share a node.
      g       continuous: sink conductance to ambient [W/K], bounds
              [0.02, 2.0], quantised to 0.001 W/K. Moves scale it.
  [K, K+M)        spreader links from a fixed pool, M = 4..10
      active  toggle 0/1 (v >= 0.5 -> 1). Installed link l adds conductance
              w_l [W/K] between its two end nodes and costs c_l.
  [K+M, K+M+B)    power blocks, B = 3..7
      slot    categorical: code 0..Q-1 into the problem's list of placement
              slots. Rounded/clamped. Setting a code already held by another
              block SWAPS the two blocks, so a slot holds at most one block.
get(st, i, p) returns None when p does not apply to element i.

Objective and constraints
-------------------------
    cost = sum_sinks k_site(sink) * g^1.3 + sum_installed_links c_l  (minimise)
    max_temp_rise  <= limit            (always)
    mean_temp_rise <= mean limit       (~30% of problems; the v1 constraint)
Validity envelope: every temperature rise within [0, 500] K and cost below
1.05 x the most expensive possible design. A failed solve (never expected:
the system is SPD) returns FINITE metrics beyond the envelope.

Families (x size ladder, 25 sizes in all)
-----------------------------------------
  plate  r x c plate (3x4 .. 10x10); sinks on the perimeter (cheap fins) and a
         few interior bosses (expensive); blocks on interior-biased slots;
         heat pipes from slot nodes to distant site nodes.
  board  2..6 modules (2x3 .. 3x4 node grids) in a row, joined ONLY by weak
         board links; sinks mount on some modules only, so blocks on uncooled
         modules depend on straps (the link pool) or on being moved.
  stack  2..4-layer 3-D chip stack (2x3 .. 4x4 tiles per layer), weak vertical
         bonding; sinks on the top lid (cheap) or the bottom board side
         (expensive); thermal pillars connect inner tiles to the top layer.
  mesh   random tree + cross-links (12..100 nodes, as v1 "random"); random
         site/slot subsets; heat pipes between nodes >= 3 hops apart.

Problems: a good layout is built by a seeded local search (stage A: blocks/
sinks moved to minimise the peak temperature; stage B: a cost-driven pass over
sink sites, block slots and link toggles with every sink at one common g level
found by bisection). The initial design is that layout with g jittered upward
and then PERTURBED: a hot block moved to a poorly cooled slot, a key sink
moved to a poor site, an installed link removed, and/or sinks undersized. A
perturbation that would push the nominal peak above 350 K (0.7 x the validity
cap) is undone, so every start lies inside the validity envelope. Most starts
are infeasible; reversing the perturbations restores feasibility, and many
starts cannot be repaired by g alone (even every g at 2.0 W/K stays over the
limit).

Reference (reference_class "heuristic"): the cheapest design found by a
seeded multi-start search - the problem's own stage-B layout plus 5 restarts
(random layout -> stage A -> stage B) - after sizing every sink separately
(SLSQP on the per-sink conductances, rounded up to the 0.001 W/K grid and
confirmed feasible with the exact solve) on the 3 cheapest distinct layouts.
It is a good heuristic, not an optimum; it is computed at the nominal (mean)
powers, so stochastic designs, which need margin for the chance constraint,
sit above it.

Stochastic variant: every evaluation multiplies each block power and each
node's background power by an independent mean-one lognormal factor
(family sigma 0.05..0.16), drawn from np.random.default_rng(seed) only. As in
v1 only powers are random (no ambient or sink noise).

Simplifications (explicit)
--------------------------
* Linear, temperature-independent conductances; lumped nodes; steady state.
* A sink is an ideal conductance to a fixed ambient; mounting cost is only
  the k*g^1.3 term (no fixed per-sink cost, no fan power).
* A block's whole power enters the single node of its slot; blocks and sinks
  do not change the plate's conduction.
* Moving a sink or a block, and swapping, are free (no relocation cost).
* Natural convection h0 is uniform over nodes and fixed.
* Spreader links are lumped conductances (no heat-pipe dry-out limit).
* Stochastic variant: power draws only, independent per block and per node.

numpy + scipy only; no file I/O, no prints.

Verb surface (docs/dataxp_verb_surface.md; owner's decision of 2026-10-23)
------------
The move set (candidate_moves), set(), Phi and the stored {i, p, v, v_old}
dicts are the v2 ones above and are untouched; only the spelling in prompts,
action lines and reply parsing changes (class attributes verb_surface and
verb_table; format_move / parse_move are the table-driven defaults of
dataxp.domain, grammar() is that default with the site / slot name tables
replaced by their rule). The bijection, per parameter:
  site k   (sink s)   <-> MOVE_SINK(s, s<k>)              k = site code
  g v      (sink s)   <-> SET_CONDUCTANCE(s, v)           [W/K]
  active 1 / 0 (link) <-> INSTALL_LINK(l) / REMOVE_LINK(l)
  slot k   (block b)  <-> PLACE_BLOCK(b, q<k>)            k = slot code
Judgment calls, recorded here as the surface rules ask:
  * s, l and b are the element numbers SET_PARAM used: the bracketed index
    the observation prints (sinks [0, K), links [K, K+M), blocks
    [K+M, K+M+B)), never a node number.
  * Site and slot codes are spelled as the Sites / Slots lists print them
    (s5, q3), a Categorical over the fixed name tables SITE_NAMES / SLOT_NAMES
    (s0..s63 / q0..q63; the builders give S <= 48 and Q <= 40). parse_move
    accepts any letter case; a bare code number (MOVE_SINK(2, 5)) and a node
    label (n12) are not accepted, so a sink / node mix-up cannot parse. A
    code beyond the problem's own list parses and set() clamps it to the
    last code, exactly as SET_PARAM(s, site, 100) did; legal_moves never
    offers one. grammar() prints the sJ / qJ rule in place of the 64 names
    (the only override; the table cannot state a per-problem code count).
  * A sink moved onto another sink's site, or a block onto another block's
    slot, still SWAPS the two (set() is unchanged); the verb docs say so.
  * SET_CONDUCTANCE's value is the new absolute conductance in W/K (unit in
    grammar() only), quantised to 0.001 and clamped by set() as before.
  * INSTALL_LINK / REMOVE_LINK are the family-neutral names; the grammar
    names the three link kinds (heat pipe / thermal strap / thermal pillar)
    and the observation keeps calling them by the family's own word.
  * SET_PARAM(...) text is no longer parsed on this domain: the grammar the
    policy is served lists the verbs only.
  * describe()'s "Design elements" paragraph is written in the verbs under
    verb_surface (observation text only; no generated file contains it).
    element_label (the W1 "after this change" label) is unchanged, so history
    lines read e.g. "MOVE_SINK(1, s4) (sink [1] (after this change: site
    s4 = n12 r1c2, g=0.812 W/K); was s2)"; "was 0.6" for a resize; the link
    pair needs no "was".

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
-----------------------
The same moves through the one wrapper tool, generated from the verb table
by dataxp.domain: MOVE_SINK(1, s4) <-> SET_PARAM(1, site, 4) (the stored site
code), SET_CONDUCTANCE(1, 0.812) <-> SET_PARAM(1, g, 0.812), INSTALL_LINK(7)
/ REMOVE_LINK(7) <-> SET_PARAM(7, active, 1) / SET_PARAM(7, active, 0),
PLACE_BLOCK(12, q3) <-> SET_PARAM(12, slot, 3). Judgment calls: the phrases
are "Move a heat sink", "Set a heat sink's conductance", "Install a link" /
"Remove a link", "Place a power block"; set_param_grammar is the one override
(as grammar() is under multitool): the 64-entry "code: 0 = s0, 1 = s1, ..."
lists of the site / slot codecs are replaced by the rule "the code number
from the Sites / Slots list (the number after s / q), not a node number",
since the code count is per problem; the "Design elements" paragraph reverts
to its SET_PARAM wording (the run2 text) under this spelling
(spelled_in_verbs); the history "was ..." prints the stored site / slot code.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from ..domain import (
    ArgSpec,
    Candidate,
    Categorical,
    Continuous,
    Domain,
    ParamVerb,
    Toggle,
    current_spelling,
    set_param_code_list,
    spelled_in_verbs,
    stable_seed,
    surface_grammar,
    surface_set_param_grammar,
)
from ..potential import DesignProgram, program_from_goals

_FAMILIES = ("plate", "board", "stack", "mesh")

_PLATE_SHAPES = ((3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (10, 10))
_BOARD_CONFIGS = ((2, 2, 3), (3, 2, 3), (3, 3, 3), (4, 3, 3), (5, 3, 3), (6, 3, 4))
_STACK_CONFIGS = ((2, 2, 3), (2, 3, 3), (3, 3, 3), (3, 3, 4), (4, 3, 4), (4, 4, 4))
_MESH_SIZES = (12, 20, 32, 48, 72, 100)

_LIMIT_RANGE = {"plate": (40.0, 70.0), "board": (45.0, 80.0),
                "stack": (30.0, 60.0), "mesh": (45.0, 80.0)}
_SIGMA_RANGE = {"plate": (0.05, 0.08), "board": (0.08, 0.12),
                "stack": (0.07, 0.12), "mesh": (0.10, 0.16)}
_LINK_KIND = {"plate": "heat pipe", "board": "thermal strap",
              "stack": "thermal pillar", "mesh": "heat pipe"}

G_MIN = 0.02           # [W/K] smallest sink
G_MAX = 2.0            # [W/K] largest sink that fits
G_DEC = 3              # g quantised to 10^-3 W/K
COST_EXP = 1.3         # superlinear sink cost exponent
TEMP_CAP = 500.0       # [K] validity envelope on any temperature rise
START_T_CAP = 0.7 * TEMP_CAP   # [K] perturbed starts stay below this peak
G_FACTORS = (0.5, 0.8, 0.92, 1.1, 1.3, 2.0)
N_SITE_OPTS = 5        # per sink per turn: 3 hottest sites, nearest, random
N_SLOT_OPTS = 5        # per block per turn: 3 coolest slots, nearest, random
Z90 = 1.2815515655446004

_PERTURB_WEIGHTS = (("block", 0.27), ("sink", 0.22), ("link", 0.33), ("gshort", 0.18))

# ── verb surface (docstring "Verb surface"; docs/dataxp_verb_surface.md) ─────
# Site and slot codes are spelled the way the observation's Sites / Slots
# lists print them (s5, q3). A Categorical codec needs a class-level name
# table while the number of sites / slots is per problem; by the family
# builders S <= 48 (board: 4 cooled modules x 12 nodes) and Q <= 40 (plate /
# mesh: 0.4 x 100 nodes), so a 64-entry table covers every problem and
# grammar() prints the sJ / qJ rule instead of the table. A code at or beyond
# the cap makes format_move raise (surface_value), never a silent mis-spelling.
CODE_CAP = 64
SITE_NAMES = tuple("s%d" % j for j in range(CODE_CAP))
SLOT_NAMES = tuple("q%d" % q for q in range(CODE_CAP))
_SINK_ARG = ArgSpec("sink", "int", "sink number (the number in brackets on its line under "
                                   "Heat sinks; sinks are numbered first)")
_LINK_ARG = ArgSpec("link", "int", "link number (the number in brackets on its line in the "
                                   "link list; links are numbered after the sinks)")
_BLOCK_ARG = ArgSpec("block", "int", "block number (the number in brackets on its line under "
                                     "Power blocks; blocks are numbered after the links)")
_SITE_RULE = ("a site code from the Sites list, written as that list writes it (s0, s1, ...: "
              "s followed by the code), not a node number")
_SLOT_RULE = ("a slot code from the Slots list, written as that list writes it (q0, q1, ...: "
              "q followed by the code), not a node number")
# the same rules for the set_parameter spelling, where the value is the bare code
_SITE_CODE_RULE = ("code: the site's code number from the Sites list (the number after s: "
                   "5 for s5), not a node number")
_SLOT_CODE_RULE = ("code: the slot's code number from the Slots list (the number after q: "
                   "3 for q3), not a node number")
_SITE = Categorical("site", SITE_NAMES)
_SLOT = Categorical("slot", SLOT_NAMES)
_G_CODEC = Continuous("conductance", "W/K",
                      "%g..%g, the new conductance to ambient (quantised to 0.001 W/K; a value "
                      "outside is set to the bound)" % (G_MIN, G_MAX))


# ── small helpers ─────────────────────────────────────────────────────────────
def _qg(v: float) -> float:
    """Legal sink conductance: quantised to 0.001 W/K and clamped."""
    v = float(v)
    if not math.isfinite(v):
        v = G_MIN
    return min(G_MAX, max(G_MIN, round(v, G_DEC)))


def _swap_assign(arr: np.ndarray, idx: int, code: int) -> np.ndarray:
    """arr[idx] = code; a holder of `code` takes arr[idx]'s old code (swap)."""
    out = arr.copy()
    old = int(out[idx])
    if code == old:
        return out
    holders = np.nonzero(out == code)[0]
    for h in holders.tolist():
        if h != idx:
            out[h] = old
    out[idx] = code
    return out


def _hop_dist(n: int, edges) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path
    if not edges:
        return np.zeros((n, n))
    a = np.array([e[0] for e in edges], dtype=int)
    b = np.array([e[1] for e in edges], dtype=int)
    m = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    D = shortest_path(m, method="D", directed=False, unweighted=True)
    D[~np.isfinite(D)] = 1e6
    return D


def _counts(n: int, rng: np.random.Generator) -> Tuple[int, int, int]:
    K = 2 + int(n >= 24) + int(n >= 48) + int(n >= 80) + int(rng.random() < 0.3)
    B = 3 + int(n >= 20) + int(n >= 36) + int(n >= 64) + int(rng.random() < 0.4)
    M = min(10, 4 + n // 16)
    return min(K, 6), min(B, 7), M


def _link_pool(rng, D, src, dst, m: int, min_d: int):
    """m distinct node pairs (a in src, b in dst) at hop distance >= min_d."""
    out, have = [], set()
    src = list(src)
    dst = list(dst)
    for md in (min_d, max(2, min_d - 1), 1):
        attempts = 0
        while len(out) < m and attempts < 400:
            attempts += 1
            a = int(src[int(rng.integers(len(src)))])
            b = int(dst[int(rng.integers(len(dst)))])
            key = (min(a, b), max(a, b))
            if a == b or key in have or D[a, b] < md:
                continue
            have.add(key)
            out.append((a, b))
        if len(out) >= m:
            break
    return out


# ── family builders ──────────────────────────────────────────────────────────
def _grid_edges(rows: int, cols: int, base: int = 0):
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = base + r * cols + c
            if c + 1 < cols:
                edges.append((i, i + 1))
            if r + 1 < rows:
                edges.append((i, i + cols))
    return edges


def _build_plate(rng):
    rows, cols = _PLATE_SHAPES[int(rng.integers(len(_PLATE_SHAPES)))]
    n = rows * cols
    K, B, M = _counts(n, rng)
    labels = ["r%dc%d" % divmod(i, cols) for i in range(n)]
    edges = _grid_edges(rows, cols)
    w0 = float(rng.uniform(0.08, 0.4))
    w = w0 * np.exp(rng.normal(0.0, 0.25, size=len(edges)))
    perim = [i for i in range(n) if i // cols in (0, rows - 1) or i % cols in (0, cols - 1)]
    pset = set(perim)
    interior = [i for i in range(n) if i not in pset]
    n_int = max(1, int(round(0.10 * len(interior)))) if interior else 0
    int_sites = (sorted(int(x) for x in rng.choice(interior, size=n_int, replace=False))
                 if n_int else [])
    site_nodes = sorted(perim + int_sites)
    iset = set(int_sites)
    site_k = [float(rng.uniform(1.3, 2.2)) if j in iset else float(rng.uniform(0.5, 1.0))
              for j in site_nodes]
    Q = min(n - 1, max(B + 3, int(round(0.4 * n))))
    wts = np.array([4.0 if i not in pset else 1.0 for i in range(n)])
    slot_nodes = sorted(int(x) for x in rng.choice(n, size=Q, replace=False, p=wts / wts.sum()))
    D = _hop_dist(n, edges)
    links = _link_pool(rng, D, slot_nodes, site_nodes, M, 3)
    link_w = w0 * rng.uniform(3.0, 8.0, size=len(links))
    return dict(family="plate", dims=[rows, cols], n=n, K=K, B=B, labels=labels,
                edges=edges, w=w, site_nodes=site_nodes, site_k=site_k,
                site_note={j: ("interior boss" if j in iset else "perimeter")
                           for j in site_nodes},
                slot_nodes=slot_nodes, links=links, link_w=link_w)


def _build_board(rng):
    m, mr, mc = _BOARD_CONFIGS[int(rng.integers(len(_BOARD_CONFIGS)))]
    per = mr * mc
    n = m * per
    K, B, M = _counts(n, rng)
    labels = ["m%d.r%dc%d" % (i // per, (i % per) // mc, i % mc) for i in range(n)]
    edges, w = [], []
    for j in range(m):
        e = _grid_edges(mr, mc, base=j * per)
        wm = float(rng.uniform(0.25, 0.7))
        edges += e
        w += (wm * np.exp(rng.normal(0.0, 0.2, size=len(e)))).tolist()
    for j in range(m - 1):          # weak board conduction between neighbours
        rows_used = rng.choice(mr, size=1 + int(rng.random() < 0.5 and mr > 1), replace=False)
        for r in sorted(int(x) for x in rows_used):
            edges.append((j * per + r * mc + (mc - 1), (j + 1) * per + r * mc))
            w.append(float(rng.uniform(0.01, 0.04)))
    w = np.asarray(w, dtype=float)
    # cooled modules carry the sink sites; need S >= K + 2
    n_cool = max(1, min(m - 1, int(round(m * rng.uniform(0.35, 0.6)))))
    while n_cool * per < K + 2:
        n_cool += 1
    cooled = sorted(int(x) for x in rng.choice(m, size=n_cool, replace=False))
    site_nodes = [j * per + t for j in cooled for t in range(per)]
    site_k = [float(rng.uniform(0.5, 1.4)) for _ in site_nodes]
    # a cooled module's area is mostly taken by its sink footprint: at most one
    # placement slot per cooled module, the rest on uncooled modules
    cset = set(cooled)
    unc_nodes = [i for i in range(n) if (i // per) not in cset]
    cool_slots = [j * per + int(rng.integers(per)) for j in cooled]
    Q = min(len(unc_nodes) + len(cool_slots), max(B + 3, int(round(0.45 * n))))
    n_unc = min(len(unc_nodes), Q - len(cool_slots))
    slot_nodes = sorted(cool_slots + [int(x) for x in rng.choice(unc_nodes, size=n_unc,
                                                                    replace=False)])
    if len(slot_nodes) < B + 2:          # guard: top up from cooled modules
        spare = [i for i in range(n) if i not in set(slot_nodes)]
        slot_nodes = sorted(slot_nodes + [int(x) for x in rng.choice(
            spare, size=B + 2 - len(slot_nodes), replace=False)])
    D = _hop_dist(n, edges)
    hot_src = [i for i in range(n) if (i // per) not in cset] or list(range(n))
    links = []
    have = set()
    attempts = 0
    while len(links) < M and attempts < 400:
        attempts += 1
        a = int(hot_src[int(rng.integers(len(hot_src)))]) if rng.random() < 0.7 \
            else int(rng.integers(n))
        b = int(rng.integers(n))
        if a // per == b // per:
            continue
        key = (min(a, b), max(a, b))
        if key in have:
            continue
        have.add(key)
        links.append((a, b))
    if len(links) < 2:
        links = _link_pool(rng, D, range(n), range(n), M, 2)
    link_w = rng.uniform(0.25, 0.8, size=len(links))
    return dict(family="board", dims=[m, mr, mc], n=n, K=K, B=B, labels=labels,
                edges=edges, w=w, site_nodes=site_nodes, site_k=site_k,
                site_note={j: "module m%d" % (j // per) for j in site_nodes},
                slot_nodes=slot_nodes, links=links, link_w=link_w, cooled=cooled)


def _build_stack(rng):
    Lz, r, c = _STACK_CONFIGS[int(rng.integers(len(_STACK_CONFIGS)))]
    per = r * c
    n = Lz * per
    K, B, M = _counts(n, rng)
    labels = ["L%d.r%dc%d" % (i // per, (i % per) // c, i % c) for i in range(n)]
    edges, w = [], []
    w_lat = float(rng.uniform(0.15, 0.5))
    w_v = float(rng.uniform(0.04, 0.15))
    for l in range(Lz):
        e = _grid_edges(r, c, base=l * per)
        edges += e
        w += (w_lat * np.exp(rng.normal(0.0, 0.2, size=len(e)))).tolist()
    for l in range(Lz - 1):
        for t in range(per):
            edges.append((l * per + t, (l + 1) * per + t))
            w.append(float(w_v * math.exp(rng.normal(0.0, 0.2))))
    w = np.asarray(w, dtype=float)
    top = [(Lz - 1) * per + t for t in range(per)]
    bottom = list(range(per))
    site_nodes = bottom + top
    site_k = ([float(rng.uniform(1.5, 2.5)) for _ in bottom]
              + [float(rng.uniform(0.5, 1.0)) for _ in top])
    # blocks sit in the logic layers below the top (lid) layer
    below = (Lz - 1) * per
    Q = min(below, max(B + 3, int(round(0.45 * n))))
    slot_nodes = sorted(int(x) for x in rng.choice(below, size=Q, replace=False))
    links, have, attempts = [], set(), 0
    while len(links) < M and attempts < 400:
        attempts += 1
        l = int(rng.integers(0, Lz - 1))
        t = int(rng.integers(per))
        a = l * per + t
        tr, tc = divmod(t, c)
        tr2 = min(r - 1, max(0, tr + int(rng.integers(-1, 2))))
        tc2 = min(c - 1, max(0, tc + int(rng.integers(-1, 2))))
        b = (Lz - 1) * per + tr2 * c + tc2
        key = (min(a, b), max(a, b))
        if a == b or key in have:
            continue
        have.add(key)
        links.append((a, b))
    link_w = rng.uniform(0.3, 0.9, size=len(links))
    return dict(family="stack", dims=[Lz, r, c], n=n, K=K, B=B, labels=labels,
                edges=edges, w=w, site_nodes=site_nodes, site_k=site_k,
                site_note={j: ("top lid" if j >= (Lz - 1) * per else "bottom board side")
                           for j in site_nodes},
                slot_nodes=slot_nodes, links=links, link_w=link_w,
                w_lat=w_lat, w_v=w_v)


def _build_mesh(rng):
    n = int(_MESH_SIZES[int(rng.integers(len(_MESH_SIZES)))])
    K, B, M = _counts(n, rng)
    edges = [(int(rng.integers(0, i)), i) for i in range(1, n)]
    have = {(min(u, v), max(u, v)) for u, v in edges}
    n_extra = int(round(0.4 * n))
    attempts = 0
    while n_extra > 0 and attempts < 20 * n:
        attempts += 1
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        key = (min(u, v), max(u, v))
        if u != v and key not in have:
            have.add(key)
            edges.append(key)
            n_extra -= 1
    w0 = float(rng.uniform(0.05, 0.3))
    w = w0 * np.exp(rng.normal(0.0, 0.25, size=len(edges)))
    S = max(K + 2, int(round(0.35 * n)))
    site_nodes = sorted(int(x) for x in rng.choice(n, size=S, replace=False))
    site_k = [float(rng.uniform(0.5, 2.0)) for _ in site_nodes]
    Q = min(n - 1, max(B + 3, int(round(0.4 * n))))
    slot_nodes = sorted(int(x) for x in rng.choice(n, size=Q, replace=False))
    D = _hop_dist(n, edges)
    links = _link_pool(rng, D, slot_nodes, site_nodes, M, 3)
    link_w = w0 * rng.uniform(3.0, 8.0, size=len(links))
    return dict(family="mesh", dims=[n], n=n, K=K, B=B, labels=[""] * n,
                edges=edges, w=w, site_nodes=site_nodes, site_k=site_k,
                site_note={}, slot_nodes=slot_nodes, links=links, link_w=link_w)


_BUILDERS = {"plate": _build_plate, "board": _build_board,
             "stack": _build_stack, "mesh": _build_mesh}


# ── fixed problem data + solver kernel ───────────────────────────────────────
def _fixed_from_spec(spec: dict) -> dict:
    """Immutable per-problem arrays shared by every state of the problem."""
    n = int(spec["n"])
    A0 = np.zeros((n, n), dtype=float)
    for (u, v), w in zip(spec["edges"], spec["conductances"]):
        u, v, w = int(u), int(v), float(w)
        A0[u, u] += w
        A0[v, v] += w
        A0[u, v] -= w
        A0[v, u] -= w
    h0 = float(spec["h0"])
    A0[np.arange(n), np.arange(n)] += h0
    links = spec["links"]
    return {
        "n": n,
        "family": spec["family"],
        "dims": list(spec.get("dims", [])),
        "labels": list(spec.get("node_labels", [""] * n)),
        "edges": [(int(a), int(b)) for a, b in spec["edges"]],
        "ew": np.asarray(spec["conductances"], dtype=float),
        "A0": A0,
        "h0": h0,
        "P_bg": np.asarray(spec["background_power"], dtype=float),
        "pb": np.asarray(spec["block_power"], dtype=float),
        "site_nodes": np.asarray(spec["site_nodes"], dtype=int),
        "site_k": np.asarray(spec["site_k"], dtype=float),
        "site_note": {int(k): v for k, v in (spec.get("site_note") or {}).items()},
        "slot_nodes": np.asarray(spec["slot_nodes"], dtype=int),
        "la": np.asarray([int(a) for a, _ in links], dtype=int),
        "lb": np.asarray([int(b) for _, b in links], dtype=int),
        "lw": np.asarray(spec["link_w"], dtype=float),
        "lc": np.asarray(spec["link_cost"], dtype=float),
        "link_kind": _LINK_KIND.get(spec["family"], "spreader link"),
        "sigma": float(spec.get("sigma", 0.1)),
        "K": len(spec["sink_site0"]),
        "M": len(links),
        "B": len(spec["block_power"]),
        "S": len(spec["site_nodes"]),
        "Q": len(spec["slot_nodes"]),
        "cooled": list(spec.get("cooled_modules", [])),
        "_near": None,
    }


def _system(fx: dict, site: np.ndarray, g: np.ndarray, active: np.ndarray) -> np.ndarray:
    A = fx["A0"].copy()
    on = np.nonzero(active)[0]
    if on.size:
        a = fx["la"][on]
        b = fx["lb"][on]
        w = fx["lw"][on]
        np.add.at(A, (a, a), w)
        np.add.at(A, (b, b), w)
        np.add.at(A, (a, b), -w)
        np.add.at(A, (b, a), -w)
    nodes = fx["site_nodes"][site]
    np.add.at(A, (nodes, nodes), g)
    return A


def _power(fx: dict, slot: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    pb = fx["pb"]
    P = fx["P_bg"]
    if seed is not None:
        sig = fx["sigma"]
        r = np.random.default_rng(seed)
        pb = pb * np.exp(r.normal(-0.5 * sig * sig, sig, size=pb.shape[0]))
        P = P * np.exp(r.normal(-0.5 * sig * sig, sig, size=P.shape[0]))
    P = P.copy()
    np.add.at(P, fx["slot_nodes"][slot], pb)
    return P


def _solve(fx, site, g, active, slot, seed=None) -> np.ndarray:
    return np.linalg.solve(_system(fx, site, g, active), _power(fx, slot, seed))


def _costs(fx, site, g, active) -> Tuple[float, float]:
    sink = float(np.sum(fx["site_k"][site] * np.power(g, COST_EXP)))
    link = float(np.sum(fx["lc"][np.asarray(active, dtype=bool)]))
    return sink, link


def _ratio(T: np.ndarray, lim: float, mean_lim: Optional[float]) -> float:
    r = float(np.max(T)) / lim
    if mean_lim is not None:
        r = max(r, float(np.mean(T)) / mean_lim)
    return r


def _too_hot(fx, state) -> bool:
    """A perturbed start whose nominal peak exceeds START_T_CAP is rejected so
    that initial designs stay inside the validity envelope, with room for the
    stochastic variant's high power draws."""
    site, slot, active, g = state
    try:
        T = _solve(fx, site, g, active, slot)
    except Exception:
        return True
    return (not bool(np.all(np.isfinite(T)))) or float(np.max(T)) > START_T_CAP


def _uniform_level(fx, site, active, slot, lim, mean_lim, iters: int = 22):
    """Smallest common sink level (on the 0.001 grid) meeting the limits, or
    None when even every sink at G_MAX is over."""
    K = len(site)
    P = _power(fx, slot)
    base = _system(fx, site, np.zeros(K), active)
    nodes = fx["site_nodes"][site]

    def ratio(lam):
        A = base.copy()
        A[nodes, nodes] += lam
        return _ratio(np.linalg.solve(A, P), lim, mean_lim)

    if ratio(G_MAX) > 1.0:
        return None
    if ratio(G_MIN) <= 1.0:
        return G_MIN
    lo, hi = math.log(G_MIN), math.log(G_MAX)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ratio(math.exp(mid)) > 1.0:
            lo = mid
        else:
            hi = mid
    lam = math.ceil(math.exp(hi) * 10 ** G_DEC - 1e-9) / 10 ** G_DEC
    lam = min(G_MAX, max(G_MIN, lam))
    while lam < G_MAX and ratio(lam) > 1.0:     # grid rounding safety
        lam = min(G_MAX, round(lam + 10 ** -G_DEC, G_DEC))
    return lam


def _layout_cost(fx, lay, lim, mean_lim):
    site, slot, active = lay
    lam = _uniform_level(fx, site, active, slot, lim, mean_lim)
    if lam is None:
        return float("inf"), None
    s, l = _costs(fx, site, np.full(len(site), lam), active)
    return s + l, lam


def _uniform_level_fast(fx, site, active, slot, lim, mean_lim, iters: int = 22):
    """_uniform_level via the Woodbury identity: the sinks are a rank-K
    diagonal update of the sink-free system, so each trial level costs a K x K
    solve instead of an n x n one. Used only by the reference search; its
    result is re-checked with the exact solve before it is reported."""
    K = len(site)
    P = _power(fx, slot)
    base = _system(fx, site, np.zeros(K), active)
    nodes = fx["site_nodes"][site]
    rhs = np.zeros((fx["n"], K + 1))
    rhs[:, 0] = P
    rhs[nodes, np.arange(1, K + 1)] = 1.0
    X = np.linalg.solve(base, rhs)
    z0, W = X[:, 0], X[:, 1:]
    Wn, z0n = W[nodes, :], z0[nodes]
    eye = np.eye(K)

    def ratio(lam):
        return _ratio(z0 - W @ np.linalg.solve(eye / lam + Wn, z0n), lim, mean_lim)

    if ratio(G_MAX) > 1.0:
        return None
    if ratio(G_MIN) <= 1.0:
        return G_MIN
    lo, hi = math.log(G_MIN), math.log(G_MAX)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ratio(math.exp(mid)) > 1.0:
            lo = mid
        else:
            hi = mid
    lam = math.ceil(math.exp(hi) * 10 ** G_DEC - 1e-9) / 10 ** G_DEC
    lam = min(G_MAX, max(G_MIN, lam))
    while lam < G_MAX and ratio(lam) > 1.0:
        lam = min(G_MAX, round(lam + 10 ** -G_DEC, G_DEC))
    return lam


def _layout_cost_fast(fx, lay, lim, mean_lim):
    site, slot, active = lay
    lam = _uniform_level_fast(fx, site, active, slot, lim, mean_lim)
    if lam is None:
        return float("inf"), None
    s, l = _costs(fx, site, np.full(len(site), lam), active)
    return s + l, lam


def _feasible_exact(fx, site, g, active, slot, lim, mean_lim) -> bool:
    """The constraint check evaluate() implies (deterministic, nominal power)."""
    T = _solve(fx, site, g, active, slot)
    return bool(np.all(np.isfinite(T))) and _ratio(T, lim, mean_lim) <= 1.0


def _size_sinks(fx, lay, lim, mean_lim, g0):
    """Per-sink conductances for a fixed layout: minimise sum k*g^1.3 subject
    to every temperature limit (SLSQP, analytic adjoint Jacobians), then round
    UP to the 0.001 W/K grid and confirm with the exact solve. Returns
    (cost, g) or (inf, None)."""
    from scipy.optimize import minimize
    site, slot, active = lay
    K = len(site)
    k = fx["site_k"][site]
    P = _power(fx, slot)
    base = _system(fx, site, np.zeros(K), active)
    nodes = fx["site_nodes"][site]
    n = fx["n"]
    rhs = np.zeros((n, K + 1))
    rhs[:, 0] = P
    rhs[nodes, np.arange(1, K + 1)] = 1.0
    memo = {}

    def sol(g):
        key = np.asarray(g, dtype=float).tobytes()
        if key not in memo:
            A = base.copy()
            A[nodes, nodes] += g
            X = np.linalg.solve(A, rhs)
            memo.clear()
            memo[key] = (X[:, 0], X[:, 1:])
        return memo[key]

    def cons(g):
        T, _ = sol(g)
        c = lim - T
        if mean_lim is not None:
            c = np.append(c, mean_lim - float(np.mean(T)))
        return c

    def cjac(g):
        T, Z = sol(g)
        J = Z * T[nodes][None, :]          # d(lim - T_i)/dg_s = Z[i, s] * T[node_s]
        if mean_lim is not None:
            J = np.vstack([J, J.mean(axis=0, keepdims=True)])
        return J

    g0 = np.clip(np.asarray(g0, dtype=float), G_MIN, G_MAX)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(lambda g: float(np.sum(k * np.power(np.clip(g, G_MIN, G_MAX), COST_EXP))),
                           g0, jac=lambda g: COST_EXP * k * np.power(np.clip(g, G_MIN, G_MAX),
                                                                       COST_EXP - 1.0),
                           method="SLSQP", bounds=[(G_MIN, G_MAX)] * K,
                           constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                           options={"maxiter": 100, "ftol": 1e-10})
        x = np.clip(np.asarray(res.x, dtype=float), G_MIN, G_MAX)
        if not np.all(np.isfinite(x)):
            x = g0
    except Exception:
        x = g0
    g = np.array([min(G_MAX, max(G_MIN, math.ceil(v * 10 ** G_DEC - 1e-6) / 10 ** G_DEC))
                  for v in x])
    for _ in range(400):                      # grid / tolerance repair, upward only
        if _feasible_exact(fx, site, g, active, slot, lim, mean_lim):
            return sum(_costs(fx, site, g, active)), g
        if np.all(g >= G_MAX):
            break
        g = np.array([_qg(v * 1.01 + 10 ** -G_DEC) for v in g])
    return float("inf"), None


def _reference_search(fx, base, rng, lim, mean_lim, starts: int = 5,
                      evals: int = 120, top: int = 3):
    """Heuristic best design used as the obj/ref denominator.

    base = (layout, common-level cost, level) of the problem's own stage-B
    layout. Adds `starts` seeded restarts (random layout -> stage A -> stage B,
    all with the fast level solver), then sizes every sink separately (SLSQP)
    on the `top` cheapest distinct layouts. Returns (cost, layout, g); every
    reported design is confirmed feasible with the exact solve."""
    K, M, B = len(base[0][0]), fx["M"], fx["B"]
    found = [(base[1], base[0], base[2])] if base[2] is not None else []
    for _ in range(starts):
        si = rng.choice(fx["S"], size=K, replace=False)
        sl = rng.choice(fx["Q"], size=B, replace=False)
        si, sl = _stage_a_fast(fx, si, sl, rng)
        lay, c, lam = _stage_b(fx, (si, sl, np.ones(M, dtype=int)), rng, lim, mean_lim,
                               max_evals=evals, cost_fn=_layout_cost_fast)
        if lam is not None and math.isfinite(c):
            found.append((c, lay, lam))
    found.sort(key=lambda x: x[0])
    best = (float("inf"), None, None)
    lay0, lam0 = base[0], base[2]
    if lam0 is not None:
        g = np.full(K, lam0)
        if _feasible_exact(fx, lay0[0], g, lay0[2], lay0[1], lim, mean_lim):
            best = (base[1], lay0, g)
    seen = set()
    for c, lay, lam in found:
        key = (tuple(lay[0].tolist()), tuple(lay[1].tolist()), tuple(lay[2].tolist()))
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > top:
            break
        cs, gs = _size_sinks(fx, lay, lim, mean_lim, np.full(K, lam))
        if gs is not None and cs < best[0]:
            best = (cs, lay, gs)
    return best


def _stage_a_fast(fx, site, slot, rng, sweeps: int = 3):
    """_stage_a for the reference search: the same moves and acceptance rule,
    priced from one inverse of the current system. A block move only changes
    P (one mat-vec); a sink move to a free site is a rank-2 diagonal update
    (Woodbury). Moving a sink onto another sink's site swaps two equal nominal
    sinks, which changes nothing, so it is skipped."""
    K = len(site)
    M = fx["M"]
    g_n = 0.6
    act = np.ones(M, dtype=int)
    sn = fx["site_nodes"]
    Cinv = np.array([[-1.0 / g_n, 0.0], [0.0, 1.0 / g_n]])

    def inv_of(si):
        return np.linalg.inv(_system(fx, si, np.full(K, g_n), act))

    Ainv = inv_of(site)
    T = Ainv @ _power(fx, slot)
    best = float(np.max(T))
    order_b = np.argsort(-fx["pb"], kind="stable").tolist()
    for _ in range(sweeps):
        improved = False
        for b in order_b:
            for q in rng.permutation(fx["Q"]).tolist():
                if q == slot[b]:
                    continue
                new = _swap_assign(slot, b, q)
                Tn = Ainv @ _power(fx, new)
                v = float(np.max(Tn))
                if v < best - 1e-12:
                    best, slot, T, improved = v, new, Tn, True
        for s in range(K):
            for j in rng.permutation(fx["S"]).tolist():
                if j == site[s] or np.any(site == j):
                    continue
                u, w = int(sn[site[s]]), int(sn[j])
                idx = [u, w]
                Z = Ainv[:, idx]
                Tn = T - Z @ np.linalg.solve(Cinv + Z[idx, :], T[idx])
                v = float(np.max(Tn))
                if v < best - 1e-12:
                    site = _swap_assign(site, s, j)
                    Ainv = inv_of(site)
                    T = Ainv @ _power(fx, slot)
                    best, improved = float(np.max(T)), True
        if not improved:
            break
    return site, slot


def _stage_a(fx, site, slot, rng, sweeps: int = 3):
    """Scale-free layout pass: move blocks / sinks to minimise the peak
    temperature with every sink at a nominal size and every link installed."""
    K = len(site)
    M = fx["M"]
    g_nom = np.full(K, 0.6)
    act = np.ones(M, dtype=int)
    A_cache = {}

    def peak(si, sl):
        key = tuple(si.tolist())
        A = A_cache.get(key)
        if A is None:
            A = _system(fx, si, g_nom, act)
            A_cache.clear()
            A_cache[key] = A
        return float(np.max(np.linalg.solve(A, _power(fx, sl))))

    best = peak(site, slot)
    order_b = np.argsort(-fx["pb"], kind="stable").tolist()
    for _ in range(sweeps):
        improved = False
        for b in order_b:
            for q in rng.permutation(fx["Q"]).tolist():
                if q == slot[b]:
                    continue
                new = _swap_assign(slot, b, q)
                v = peak(site, new)
                if v < best - 1e-12:
                    best, slot, improved = v, new, True
        for s in range(K):
            for j in rng.permutation(fx["S"]).tolist():
                if j == site[s]:
                    continue
                new = _swap_assign(site, s, j)
                v = peak(new, slot)
                if v < best - 1e-12:
                    best, site, improved = v, new, True
        if not improved:
            break
    return site, slot


def _stage_b(fx, lay, rng, lim, mean_lim, max_evals: int = 160, cost_fn=None):
    """Cost-driven first-improvement pass over link toggles, sink sites and
    block slots, every sink at the common level found by bisection."""
    if cost_fn is None:
        cost_fn = _layout_cost
    best, lam = cost_fn(fx, lay, lim, mean_lim)
    evals = 0
    K, M, B = len(lay[0]), fx["M"], fx["B"]
    while evals < max_evals:
        site, slot, active = lay
        g = np.full(K, lam if lam is not None else G_MAX)
        T = _solve(fx, site, g, active, slot)
        moves = [("link", l, None) for l in range(M)]
        site_T = T[fx["site_nodes"]]
        hot_sites = np.argsort(-site_T, kind="stable").tolist()
        cheap_sites = np.argsort(fx["site_k"], kind="stable").tolist()
        for s in range(K):
            opts = [j for j in hot_sites if j != site[s]][:3]
            opts += [j for j in cheap_sites if j != site[s] and j not in opts][:1]
            moves += [("site", s, j) for j in opts]
        node_T = T[fx["slot_nodes"][slot]]
        cool = np.argsort(T[fx["slot_nodes"]], kind="stable").tolist()
        for b in np.argsort(-node_T, kind="stable").tolist()[:3]:
            moves += [("slot", b, q) for q in [q for q in cool if q != slot[b]][:3]]
        order = rng.permutation(len(moves)).tolist()
        improved = False
        for k in order:
            kind, a, v = moves[k]
            if kind == "link":
                act2 = active.copy()
                act2[a] = 1 - act2[a]
                new = (site, slot, act2)
            elif kind == "site":
                new = (_swap_assign(site, a, v), slot, active)
            else:
                new = (site, _swap_assign(slot, a, v), active)
            c, lam2 = cost_fn(fx, new, lim, mean_lim)
            evals += 1
            if c < best * (1.0 - 1e-9):
                best, lam, lay = c, lam2, new
                improved = True
                break
            if evals >= max_evals:
                break
        if not improved:
            break
    return lay, best, lam


# ── the domain ───────────────────────────────────────────────────────────────
class Thermal2Domain(Domain):
    """Heat-sink placement + sizing, spreader-link toggles and power-block
    placement on a lumped steady-state thermal network."""
    # stochastic variants: phi_select compares candidates with a fresh
    # incumbent estimate (dataxp/ladder.py); no effect when deterministic
    rescore_incumbent = True

    name = "thermal2"
    params = ("site", "g", "active", "slot")
    factors = G_FACTORS
    reference_class = "heuristic"
    max_elements = 64
    max_candidates = 160

    # ── verb surface (docstring "Verb surface"; docs/dataxp_verb_surface.md) ──
    # Spelling only: every SET_PARAM move candidate_moves enumerates has one
    # verb text and parses back to the same move. format_move / parse_move are
    # the table-driven defaults of dataxp.domain; grammar() below is the
    # default with the site / slot name tables replaced by their rule.
    verb_surface = True
    verb_table = (
        ParamVerb("site", "MOVE_SINK", _SINK_ARG, _SITE,
                  doc="mount a heat sink on another allowed site; it keeps its conductance "
                      "and is re-priced at the new site's k. A site held by another sink "
                      "swaps the two sinks' sites, so two sinks never share a node.",
                  phrase="Move a heat sink"),
        ParamVerb("g", "SET_CONDUCTANCE", _SINK_ARG, _G_CODEC,
                  doc="resize a heat sink's conductance to ambient; its cost is k*g^%s at "
                      "its site's k." % ("%g" % COST_EXP),
                  phrase="Set a heat sink's conductance"),
        ParamVerb("active", "INSTALL_LINK", _LINK_ARG,
                  Toggle("REMOVE_LINK",
                         off_doc="remove an installed spreader link: its conductance between "
                                 "its two end nodes and its cost go away.",
                         off_phrase="Remove a link"),
                  doc="install a spreader link (heat pipe, thermal strap or thermal pillar by "
                      "family) from the fixed pool: it adds its conductance w between its two "
                      "end nodes and costs c.",
                  phrase="Install a link"),
        ParamVerb("slot", "PLACE_BLOCK", _BLOCK_ARG, _SLOT,
                  doc="move a power block to another placement slot (free): its power then "
                      "enters that slot's node. A slot held by another block swaps the two "
                      "blocks, so a slot holds at most one block.",
                  phrase="Place a power block"),
    )

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    def grammar(self) -> str:
        """The table-driven grammar (dataxp.domain.surface_grammar) with the
        64-entry site / slot name tables replaced by the sJ / qJ rule: the
        codes are per problem, which a class-level table cannot say. Under
        the set_parameter spelling the set_param_grammar hook below."""
        if current_spelling() == "set_parameter":
            return self.set_param_grammar()
        g = surface_grammar(self)
        for c, rule in ((_SITE, _SITE_RULE), (_SLOT, _SLOT_RULE)):
            full = "%s: one of %s" % (c.arg, " | ".join(c.names))
            if full not in g:
                raise RuntimeError("thermal2 grammar: expected %r in the table grammar" % c.arg)
            g = g.replace(full, "%s: str %s" % (c.arg, rule))
        return g

    def set_param_grammar(self) -> str:
        """The generated set_parameter grammar with the 64-entry 'code: 0 =
        s0, 1 = s1, ...' lists of the site / slot codecs replaced by the
        code-number rule (docstring "Set-parameter spelling")."""
        g = surface_set_param_grammar(self)
        for c, rule in ((_SITE, _SITE_CODE_RULE), (_SLOT, _SLOT_CODE_RULE)):
            full = set_param_code_list(c)
            if full not in g:
                raise RuntimeError("thermal2 set_param grammar: expected the %r code list"
                                   % c.arg)
            g = g.replace(full, rule)
        return g

    # ── problem generation ──────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        net = _BUILDERS[family](rng)
        n, K, B = net["n"], net["K"], net["B"]
        S, Q = len(net["site_nodes"]), len(net["slot_nodes"])
        M = len(net["links"])

        # powers (raw scale; rescaled below against the limit)
        pb = np.exp(rng.normal(0.0, 0.5, size=B))
        n_hot = 1 + int(rng.random() < 0.4)
        hot = rng.choice(B, size=n_hot, replace=False)
        pb[hot] *= rng.uniform(2.5, 6.0, size=n_hot)
        pbg = np.exp(rng.normal(0.0, 0.4, size=n))
        pbg *= rng.uniform(0.1, 0.3) * pb.sum() / pbg.sum()
        h0 = float(rng.uniform(0.002, 0.005))
        sigma = float(rng.uniform(*_SIGMA_RANGE[family]))

        spec = {
            "family": family,
            "n": int(n),
            "dims": [int(x) for x in net["dims"]],
            "node_labels": list(net["labels"]),
            "edges": [[int(a), int(b)] for a, b in net["edges"]],
            "conductances": [float(x) for x in net["w"]],
            "h0": h0,
            "background_power": [float(x) for x in pbg],
            "block_power": [float(x) for x in pb],
            "site_nodes": [int(x) for x in net["site_nodes"]],
            "site_k": [float(x) for x in net["site_k"]],
            "site_note": {str(k): v for k, v in net["site_note"].items()},
            "slot_nodes": [int(x) for x in net["slot_nodes"]],
            "links": [[int(a), int(b)] for a, b in net["links"]],
            "link_w": [float(x) for x in net["link_w"]],
            "link_cost": [0.0] * M,
            "cooled_modules": [int(x) for x in net.get("cooled", [])],
            "g_min": G_MIN,
            "g_max": G_MAX,
            "sigma": sigma,
            "sink_site0": [int(x) for x in rng.choice(S, size=K, replace=False)],
            "g0": [0.6] * K,
            "link_active0": [0] * M,
            "block_slot0": [int(x) for x in rng.choice(Q, size=B, replace=False)],
        }
        fx = _fixed_from_spec(spec)

        # stage A: a well-cooled layout (scale-free)
        site = np.asarray(spec["sink_site0"], dtype=int)
        slot = np.asarray(spec["block_slot0"], dtype=int)
        site, slot = _stage_a(fx, site, slot, rng)
        act0 = np.ones(M, dtype=int)

        # limits; scale powers so that the fully equipped layout (every sink at
        # G_MAX, every link installed) has margin. Without links it may not:
        # then some links are necessary, not just cost savers.
        limit = float(rng.uniform(*_LIMIT_RANGE[family]))
        T_full = _solve(fx, site, np.full(K, G_MAX), act0, slot)
        beta = float(rng.uniform(0.35, 0.6))
        scale = beta * limit / float(np.max(T_full))
        pb = pb * scale
        pbg = pbg * scale
        spec["block_power"] = [float(x) for x in pb]
        spec["background_power"] = [float(x) for x in pbg]
        goals = {"maximum_max_temp_rise": limit}
        if rng.random() < 0.3:
            mean_full = float(np.mean(T_full)) * scale
            goals["maximum_mean_temp_rise"] = float(
                max(limit * rng.uniform(0.35, 0.6), 1.35 * mean_full))
        mean_lim = goals.get("maximum_mean_temp_rise")
        fx = _fixed_from_spec(spec)

        # link costs relative to the sink cost of the stage-A layout
        lamA = _uniform_level(fx, site, act0, slot, limit, mean_lim)
        lamA = lamA if lamA is not None else G_MAX
        sink_cost_A = float(np.sum(fx["site_k"][site] * lamA ** COST_EXP))
        lc = (sink_cost_A / K) * rng.uniform(0.05, 0.3, size=M)
        spec["link_cost"] = [float(x) for x in lc]
        fx = _fixed_from_spec(spec)

        # stage B: cost-driven layout (prunes links that do not pay for
        # themselves) -> heuristic value for obj/ref
        lay, ref_cost, lam_ref = _stage_b(fx, (site, slot, act0), rng, limit, mean_lim)
        if lam_ref is None or not math.isfinite(ref_cost):
            lay, lam_ref = (site, slot, act0), lamA
            ref_cost = sum(_costs(fx, site, np.full(K, lam_ref), act0))
        site, slot, active = (lay[0].copy(), lay[1].copy(), lay[2].copy())

        # reference value: multi-start layout search + per-sink sizing, on its
        # own seeded stream so the main stream (initial design) is unaffected
        ref_rng = np.random.default_rng(stable_seed(
            "thermal2_ref", family, n, "%.12g" % float(np.sum(pb)), "%.12g" % h0))
        r_cost, r_lay, r_g = _reference_search(fx, (lay, ref_cost, lam_ref), ref_rng,
                                                limit, mean_lim)
        if r_lay is None:      # defensive: the base layout is feasible at its level
            r_cost, r_lay, r_g = ref_cost, lay, np.full(K, lam_ref)

        # initial design: the layout with g jittered up, then perturbed
        lo_j, hi_j = (1.15, 1.5) if self.stochastic else (1.0, 1.35)
        g = np.array([_qg(lam_ref * x) for x in rng.uniform(lo_j, hi_j, size=K)])
        u = rng.random()
        klass = "feasible" if u < 0.08 else ("near" if u < 0.40 else "deep")
        n_pert = 0 if klass == "feasible" else (
            1 if klass == "near" else 2 + int(rng.random() < 0.5) + int(rng.random() < 0.3))
        applied: List[str] = []
        state = [site, slot, active, g]
        for k in range(n_pert):
            kind = self._pick_kind(rng, state, applied)
            prev = list(state)
            self._perturb(fx, state, kind, "near" if klass == "near" else "deep", rng)
            if _too_hot(fx, state):
                state[:] = prev     # keep the start inside the validity envelope
                continue
            applied.append(kind)
        # an unneeded link left installed: costs money, never infeasibility
        if rng.random() < 0.3 and not np.all(state[2]):
            off = np.nonzero(state[2] == 0)[0]
            a2 = state[2].copy()
            a2[int(off[int(rng.integers(off.size))])] = 1
            state[2] = a2
            applied.append("extra_link")
        tries = 0
        while klass != "feasible" and tries < 3:
            T = _solve(fx, state[0], state[3], state[2], state[1])
            if _ratio(T, limit, mean_lim) > 1.0:
                break
            kind = self._pick_kind(rng, state, applied)
            prev = list(state)
            self._perturb(fx, state, kind, "deep", rng)
            if _too_hot(fx, state):
                state[:] = prev
            else:
                applied.append(kind)
            tries += 1
        site, slot, active, g = state
        T0 = _solve(fx, site, g, active, slot)
        feas0 = _ratio(T0, limit, mean_lim) <= 1.0
        if feas0:
            klass = "feasible"
        T_kmax = _solve(fx, site, np.full(K, G_MAX), active, slot)
        knob_only = _ratio(T_kmax, limit, mean_lim) <= 1.0

        spec.update({
            "sink_site0": [int(x) for x in site],
            "g0": [float(x) for x in g],
            "link_active0": [int(x) for x in active],
            "block_slot0": [int(x) for x in slot],
            "goals": {k: float(v) for k, v in goals.items()},
            "ref_cost": float(r_cost),
            "ref_layout": {"sink_site": [int(x) for x in r_lay[0]],
                           "block_slot": [int(x) for x in r_lay[1]],
                           "link_active": [int(x) for x in r_lay[2]],
                           "sink_g": [float(x) for x in r_g]},
            "start_layout_cost": float(ref_cost),
            "difficulty_class": klass,
            "perturbations": applied,
            "fixable_by_g_alone": bool(knob_only),
        })
        return spec

    @staticmethod
    def _pick_kind(rng, state, applied):
        kinds, wts = [], []
        for k, wt in _PERTURB_WEIGHTS:
            if k == "link" and not np.any(state[2]):
                continue
            if k in applied:
                wt *= 0.25
            kinds.append(k)
            wts.append(wt)
        wts = np.asarray(wts) / np.sum(wts)
        return kinds[int(rng.choice(len(kinds), p=wts))]

    @staticmethod
    def _perturb(fx, state, kind, severity, rng):
        site, slot, active, g = state
        lo_q, hi_q = (0.45, 0.8) if severity == "near" else (0.8, 1.0)

        def pick(vals):
            vals = [v for v in vals if v[0] < 0.5 * TEMP_CAP] or vals
            vals.sort()
            L = len(vals)
            a = int(lo_q * (L - 1))
            b = max(a, int(round(hi_q * (L - 1))))
            return vals[int(rng.integers(a, b + 1))][1]

        if kind == "block":
            order = np.argsort(-fx["pb"], kind="stable").tolist()
            b = order[int(rng.integers(min(2, len(order))))]
            vals = []
            for q in range(fx["Q"]):
                if q == slot[b]:
                    continue
                new = _swap_assign(slot, b, q)
                vals.append((float(np.max(_solve(fx, site, g, active, new))), q))
            if vals:
                state[1] = _swap_assign(slot, b, pick(vals))
        elif kind == "sink":
            T = _solve(fx, site, g, active, slot)
            removed = g * T[fx["site_nodes"][site]]
            order = np.argsort(-removed, kind="stable").tolist()
            s = order[int(rng.integers(min(2, len(order))))]
            vals = []
            for j in range(fx["S"]):
                if j == site[s]:
                    continue
                new = _swap_assign(site, s, j)
                vals.append((float(np.max(_solve(fx, new, g, active, slot))), j))
            if vals:
                state[0] = _swap_assign(site, s, pick(vals))
        elif kind == "link":
            # remove installed links: near -> one of the two most impactful,
            # deep -> the two most impactful (one if only one is installed)
            on = np.nonzero(active)[0].tolist()
            if on:
                vals = []
                for l in on:
                    a2 = active.copy()
                    a2[l] = 0
                    vals.append((float(np.max(_solve(fx, site, g, a2, slot))), l))
                vals.sort(key=lambda x: (-x[0], x[1]))
                if severity == "near":
                    picks = [vals[int(rng.integers(min(2, len(vals))))][1]]
                else:
                    picks = [l for _, l in vals[:2]]
                a2 = active.copy()
                a2[picks] = 0
                state[2] = a2
        else:  # gshort
            T = _solve(fx, site, g, active, slot)
            removed = g * T[fx["site_nodes"][site]]
            order = np.argsort(-removed, kind="stable").tolist()
            n_s = 1 if severity == "near" else 1 + int(rng.random() < 0.5)
            g2 = g.copy()
            for s in order[:n_s]:
                f = rng.uniform(0.45, 0.75) if severity == "near" else rng.uniform(0.08, 0.3)
                g2[s] = _qg(g2[s] * f)
            state[3] = g2

    # ── state ────────────────────────────────────────────────────────────────
    def load(self, spec: dict):
        fx = _fixed_from_spec(spec)
        return {
            "fx": fx,                                                   # shared
            "site": np.asarray(spec["sink_site0"], dtype=int).copy(),   # design
            "g": np.asarray([_qg(x) for x in spec["g0"]], dtype=float),
            "active": np.asarray(spec["link_active0"], dtype=int).copy(),
            "slot": np.asarray(spec["block_slot0"], dtype=int).copy(),
            "_T": None,                                                 # cache
        }

    def clone(self, st):
        return {"fx": st["fx"], "site": st["site"].copy(), "g": st["g"].copy(),
                "active": st["active"].copy(), "slot": st["slot"].copy(), "_T": None}

    # ── simulation ───────────────────────────────────────────────────────────
    def _fail(self, st) -> dict:
        try:
            s, l = _costs(st["fx"], st["site"], st["g"], st["active"])
        except Exception:
            s, l = 1e6, 0.0
        c = s + l if math.isfinite(s + l) else 1e6
        return {"max_temp_rise": 10.0 * TEMP_CAP, "mean_temp_rise": 10.0 * TEMP_CAP,
                "cost": float(c), "sink_cost": float(s if math.isfinite(s) else c),
                "link_cost": float(l if math.isfinite(l) else 0.0),
                "hotspot_node": -1.0, "temp_negativity": 1.0}

    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        fx = st["fx"]
        try:
            T = _solve(fx, st["site"], st["g"], st["active"], st["slot"],
                       seed=(int(seed) & 0xFFFFFFFF) if self.stochastic else None)
            if not np.all(np.isfinite(T)):
                return self._fail(st)
            s, l = _costs(fx, st["site"], st["g"], st["active"])
            if not math.isfinite(s + l):
                return self._fail(st)
            return {
                "max_temp_rise": float(np.max(T)),
                "mean_temp_rise": float(np.mean(T)),
                "cost": float(s + l),
                "sink_cost": float(s),
                "link_cost": float(l),
                "hotspot_node": float(int(np.argmax(T))),
                "temp_negativity": float(max(0.0, -float(np.min(T)))),
            }
        except Exception:
            return self._fail(st)

    # ── design variables ─────────────────────────────────────────────────────
    def n_elements(self, st) -> int:
        fx = st["fx"]
        return int(fx["K"] + fx["M"] + fx["B"])

    def _kind(self, st, i: int):
        fx = st["fx"]
        K, M, B = fx["K"], fx["M"], fx["B"]
        if 0 <= i < K:
            return "sink", i
        if K <= i < K + M:
            return "link", i - K
        if K + M <= i < K + M + B:
            return "block", i - K - M
        return None, None

    def get(self, st, i: int, p: str) -> Optional[float]:
        kind, j = self._kind(st, int(i))
        if kind == "sink":
            if p == "site":
                return float(st["site"][j])
            if p == "g":
                return float(st["g"][j])
        elif kind == "link" and p == "active":
            return float(st["active"][j])
        elif kind == "block" and p == "slot":
            return float(st["slot"][j])
        return None

    def set(self, st, i: int, p: str, v: float) -> None:
        kind, j = self._kind(st, int(i))
        fx = st["fx"]
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        if kind == "sink" and p == "g":
            st["g"][j] = _qg(v)
        elif kind == "sink" and p == "site":
            code = min(fx["S"] - 1, max(0, int(round(v))))
            st["site"] = _swap_assign(st["site"], j, code)
        elif kind == "link" and p == "active":
            st["active"][j] = 1 if v >= 0.5 else 0
        elif kind == "block" and p == "slot":
            code = min(fx["Q"] - 1, max(0, int(round(v))))
            st["slot"] = _swap_assign(st["slot"], j, code)
        else:
            return
        st["_T"] = None

    def bounds(self, st, p: str) -> Tuple[float, float]:
        fx = st["fx"]
        if p == "g":
            return (G_MIN, G_MAX)
        if p == "site":
            return (0.0, float(fx["S"] - 1))
        if p == "slot":
            return (0.0, float(fx["Q"] - 1))
        if p == "active":
            return (0.0, 1.0)
        return (0.0, 0.0)

    # ── typed move set ───────────────────────────────────────────────────────
    def _nominal(self, st) -> Optional[np.ndarray]:
        T = st.get("_T")
        if T is None:
            try:
                T = _solve(st["fx"], st["site"], st["g"], st["active"], st["slot"])
            except Exception:
                return None
            if not np.all(np.isfinite(T)):
                return None
            st["_T"] = T
        return T

    @staticmethod
    def _near_lists(fx):
        """Per site code / slot code: the other codes ordered by hop distance."""
        if fx["_near"] is None:
            D = _hop_dist(fx["n"], fx["edges"])
            out = {}
            for key, nodes in (("site", fx["site_nodes"]), ("slot", fx["slot_nodes"])):
                lists = []
                for a in range(len(nodes)):
                    others = [b for b in range(len(nodes)) if b != a]
                    others.sort(key=lambda b: (D[nodes[a], nodes[b]], b))
                    lists.append(others)
                out[key] = lists
            fx["_near"] = out
        return fx["_near"]

    def candidate_moves(self, st, rng) -> List[Candidate]:
        """Every element type and move type, every turn (<= 116 moves):
        each sink: g scaled by each factor + up to 5 site changes (the 3
        hottest sites, the nearest site, one random); each link: its toggle;
        each block: up to 5 slot changes (the 3 coolest slots, the nearest
        slot, one random)."""
        fx = st["fx"]
        K, M, B = fx["K"], fx["M"], fx["B"]
        T = self._nominal(st)
        if T is None:
            T = np.zeros(fx["n"])
        near = self._near_lists(fx)
        out: List[Candidate] = []
        for s in range(K):
            cur = float(st["g"][s])
            seen = set()
            for f in G_FACTORS:
                v = _qg(cur * f)
                if v != cur and v not in seen:
                    seen.add(v)
                    out.append(Candidate(i=s, p="g", v=v, v_old=cur))
        site_T = T[fx["site_nodes"]]
        hot = np.argsort(-site_T, kind="stable").tolist()
        for s in range(K):
            own = int(st["site"][s])
            picks = [j for j in hot if j != own][:3]
            for j in near["site"][own]:
                if j not in picks:
                    picks.append(j)
                    break
            rest = [j for j in range(fx["S"]) if j != own and j not in picks]
            if rest:
                picks.append(rest[int(rng.integers(len(rest)))])
            for j in picks[:N_SITE_OPTS]:
                out.append(Candidate(i=s, p="site", v=float(j), v_old=float(own)))
        for l in range(M):
            a = int(st["active"][l])
            out.append(Candidate(i=K + l, p="active", v=float(1 - a), v_old=float(a)))
        slot_T = T[fx["slot_nodes"]]
        cool = np.argsort(slot_T, kind="stable").tolist()
        for b in range(B):
            own = int(st["slot"][b])
            picks = [q for q in cool if q != own][:3]
            for q in near["slot"][own]:
                if q not in picks:
                    picks.append(q)
                    break
            rest = [q for q in range(fx["Q"]) if q != own and q not in picks]
            if rest:
                picks.append(rest[int(rng.integers(len(rest)))])
            for q in picks[:N_SLOT_OPTS]:
                out.append(Candidate(i=K + M + b, p="slot", v=float(q), v_old=float(own)))
        return out

    # ── program / reference ──────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec["goals"],
            objective_key="cost",
            objective_ref=self.reference(spec),
            objective_sense="min",
            domain=self.name,
        )
        K = len(spec["sink_site0"])
        ks = sorted((float(x) for x in spec["site_k"]), reverse=True)[:K]
        cost_cap = 1.05 * (sum(k * G_MAX ** COST_EXP for k in ks)
                           + float(sum(float(c) for c in spec["link_cost"])))
        return replace(prog, validity_bounds=(
            ("max_temp_rise", TEMP_CAP),
            ("mean_temp_rise", TEMP_CAP),
            ("temp_negativity", 1e-6),
            ("cost", cost_cap),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        ref = spec.get("ref_cost")
        if ref is None:       # not produced by make_problem: recompute + cache
            fx = _fixed_from_spec(spec)
            goals = spec["goals"]
            lim = float(goals["maximum_max_temp_rise"])
            mlim = goals.get("maximum_mean_temp_rise")
            mlim = float(mlim) if mlim is not None else None
            K, M = fx["K"], fx["M"]
            rl = spec.get("ref_layout") or {}
            if "sink_g" in rl:      # stored design: its exact cost, if feasible
                lay = (np.asarray(rl["sink_site"], dtype=int),
                       np.asarray(rl["block_slot"], dtype=int),
                       np.asarray(rl["link_active"], dtype=int))
                g = np.asarray([_qg(x) for x in rl["sink_g"]], dtype=float)
                if _feasible_exact(fx, lay[0], g, lay[2], lay[1], lim, mlim):
                    ref = sum(_costs(fx, lay[0], g, lay[2]))
            if ref is None:         # same procedure as make_problem, from the start layout
                rng = np.random.default_rng(stable_seed(
                    "thermal2_ref", spec.get("family"), fx["n"],
                    "%.12g" % float(np.sum(fx["pb"])), "%.12g" % fx["h0"]))
                site0 = np.asarray(spec["sink_site0"], dtype=int)
                slot0 = np.asarray(spec["block_slot0"], dtype=int)
                site, slot = _stage_a(fx, site0, slot0, rng)
                lay, c, lam = _stage_b(fx, (site, slot, np.ones(M, dtype=int)), rng, lim, mlim)
                ref, _, _ = _reference_search(fx, (lay, c, lam), rng, lim, mlim)
                if not math.isfinite(ref):
                    ref = sum(_costs(fx, site0, np.full(K, G_MAX), np.ones(M, dtype=int)))
            spec["ref_cost"] = float(ref)
        return float(ref)

    # ── depth-0 diagnostic (greedy) ──────────────────────────────────────────
    def _diag(self, st):
        """(T, hottest node, coupling per sink) under nominal power, where the
        coupling of sink s is -dT_hot/dg_s from one adjoint solve."""
        fx = st["fx"]
        T = self._nominal(st)
        if T is None:
            return None
        hot = int(np.argmax(T))
        try:
            A = _system(fx, st["site"], st["g"], st["active"])
            e = np.zeros(fx["n"])
            e[hot] = 1.0
            z = np.linalg.solve(A, e)
        except Exception:
            return None
        nodes = fx["site_nodes"][st["site"]]
        # -dT_hot/dg_s = z[node_s] * T[node_s]  (adjoint sensitivity)
        coup = z[nodes] * T[nodes]
        return T, hot, coup

    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        """The sink whose conductance most cools the hottest node (adjoint
        sensitivity) wants to grow."""
        d = self._diag(st)
        if d is None:
            return None
        _, _, coup = d
        return (int(np.argmax(coup)), "g", "up")

    def hint_move(self, st) -> Optional[Candidate]:
        """One-knob greedy: grow (x2) the sink most coupled to the hottest
        node that still has room; None when every sink is at its maximum."""
        d = self._diag(st)
        if d is None:
            return None
        _, _, coup = d
        for s in np.argsort(-coup, kind="stable").tolist():
            cur = float(st["g"][s])
            v = _qg(cur * 2.0)
            if v > cur:
                return Candidate(i=int(s), p="g", v=v, v_old=cur)
        return None

    # ── observation (read-only; deterministic) ───────────────────────────────
    def describe(self, st, spec) -> str:
        """Policy observation: problem, design and nominal results, plus the
        design's thermal-resistance data (hotspot row, driving-point values,
        per-link effective resistances) so the effect of any single change on
        the hotspot can be computed exactly. Pure function of (st, spec)."""
        fx = st["fx"]
        n, K, M, B, S, Q = fx["n"], fx["K"], fx["M"], fx["B"], fx["S"], fx["Q"]
        fam = fx["family"]
        dims = fx["dims"]
        labels = fx["labels"]
        sn, sk, qn = fx["site_nodes"], fx["site_k"], fx["slot_nodes"]
        goals = (spec or {}).get("goals") or {}
        t_lim = goals.get("maximum_max_temp_rise")
        t_lim = float(t_lim) if t_lim is not None else None
        m_lim = goals.get("maximum_mean_temp_rise")
        m_lim = float(m_lim) if m_lim is not None else None
        T = self._nominal(st)
        P = _power(fx, st["slot"])
        g = st["g"]
        kind = fx["link_kind"]
        h0 = fx["h0"]

        def nl(i):
            i = int(i)
            return "n%d" % i + ((" " + labels[i]) if labels[i] else "")

        site_of = {int(st["site"][s]): s for s in range(K)}     # site code -> sink
        slot_of = {int(st["slot"][b]): b for b in range(B)}     # slot code -> block
        sink_at = {int(sn[int(st["site"][s])]): s for s in range(K)}   # node -> sink
        block_at = {int(qn[int(st["slot"][b])]): b for b in range(B)}  # node -> block
        site_code_at = {int(sn[j]): j for j in range(S)}               # node -> site code
        slot_code_at = {int(qn[q]): q for q in range(Q)}               # node -> slot code

        out = ["PROBLEM"]
        ws = fx["ew"]
        head = ("Heat-sink placement and sizing, %s installation and power-block placement "
                "on a steady-state lumped thermal network: " % kind)
        if fam == "plate":
            rows, cols = dims
            out.append(head + (
                "a flat plate discretised as a %d x %d grid of %d nodes (node index "
                "= row*%d + col, label rXcY = row X, column Y), each node joined to "
                "its up/down/left/right neighbours by fixed conduction links (%d "
                "links, w %s to %s W/K, median %s)."
                % (rows, cols, n, cols, len(ws), _obs_num(ws.min()), _obs_num(ws.max()),
                   _obs_num(np.median(ws)))))
        elif fam == "board":
            m, mr, mc = dims
            per = mr * mc
            inner = [w for (a, b), w in zip(fx["edges"], ws) if a // per == b // per]
            board = [(a, b, w) for (a, b), w in zip(fx["edges"], ws) if a // per != b // per]
            out.append(head + (
                "a board carrying %d modules m0..m%d in a row, each a %d x %d grid of "
                "nodes (node index = module*%d + row*%d + col, label mJ.rXcY); inside "
                "a module neighbours are joined by conduction links w %s to %s W/K. "
                "Neighbouring modules are joined only through the board by weak links: %s."
                % (m, m - 1, mr, mc, per, mc, _obs_num(min(inner)), _obs_num(max(inner)),
                   ", ".join("%s-%s w=%s" % (nl(a), nl(b), _obs_num(w)) for a, b, w in board))))
            out.append("Heat sinks can be mounted only on modules %s; modules %s have "
                       "no sink sites." % (
                           ", ".join("m%d" % j for j in fx["cooled"]),
                           ", ".join("m%d" % j for j in range(m) if j not in fx["cooled"])
                           or "none"))
        elif fam == "stack":
            Lz, r, c = dims
            per = r * c
            lat = [w for (a, b), w in zip(fx["edges"], ws) if a // per == b // per]
            ver = [w for (a, b), w in zip(fx["edges"], ws) if a // per != b // per]
            out.append(head + (
                "a 3-D chip stack of %d layers (L0 = bottom, on the board side; L%d = "
                "top, under the lid), each a %d x %d grid of tiles (node index = "
                "layer*%d + row*%d + col, label Ll.rXcY). Tiles in a layer are joined "
                "by lateral links (w %s to %s W/K); each tile is joined to the same "
                "tile in the layer above by a weak bonding link (w %s to %s W/K). "
                "Sinks mount on top-layer tiles (lid) or bottom-layer tiles (board side)."
                % (Lz, Lz - 1, r, c, per, c, _obs_num(min(lat)), _obs_num(max(lat)),
                   _obs_num(min(ver)), _obs_num(max(ver)))))
        else:
            adj = [[] for _ in range(n)]
            for (a, b), w in zip(fx["edges"], ws):
                adj[a].append((b, w))
                adj[b].append((a, w))
            deg = [len(x) for x in adj]
            out.append(head + (
                "an irregular network (random tree plus cross-links) of %d nodes and "
                "%d conduction links, node degree %d-%d. Links (each listed once, on "
                "its lower-numbered end, as node:w with w in W/K to 2 digits):"
                % (n, len(ws), min(deg), max(deg))))
            for i in range(n):
                hi_n = sorted((j, w) for j, w in adj[i] if j > i)
                if hi_n:
                    out.append("  n%d: %s" % (i, " ".join("%d:%s" % (j, _obs_num(w, 2))
                                                          for j, w in hi_n)))
        out.append(
            "Every node also loses heat to ambient by natural convection, h0 = %s W/K "
            "per node. T_i = steady temperature rise of node i above ambient [K], from "
            "the heat balance at every node: P_i + sum_j w_ij*(T_j - T_i) = (h0 + G_i)*T_i, "
            "where G_i is the conductance of the heat sink mounted on node i (0 if none) "
            "and an installed %s adds its conductance between its two end nodes like "
            "any other link." % (_obs_num(h0), kind))
        pbg = fx["P_bg"]
        group = ""
        if fam in ("board", "stack"):
            per = dims[1] * dims[2]
            tag = "module" if fam == "board" else "layer"
            group = "; background per %s: %s" % (tag, ", ".join(
                "%s%d %s W" % ("m" if fam == "board" else "L", j,
                               _obs_num(pbg[j * per:(j + 1) * per].sum()))
                for j in range(dims[0])))
        out.append(
            "Power: every node dissipates a fixed background power (total %s W, %s to "
            "%s W per node%s); in addition %d power blocks (total %s W) each add their "
            "power to the node of the placement slot they occupy."
            % (_obs_num(pbg.sum()), _obs_num(pbg.min()), _obs_num(pbg.max()), group, B,
               _obs_num(fx["pb"].sum())))
        ex_s = min(7, S - 1)
        ex_q = min(3, Q - 1)
        if spelled_in_verbs(self):
            # the same facts as the SET_PARAM block below, in the verbs the
            # policy is served (docstring "Verb surface"); the SET_PARAM block
            # again under the set_parameter spelling
            out.append(
                "Design elements, addressed by the index in brackets and changed one at "
                "a time with one of MOVE_SINK(sink, site), SET_CONDUCTANCE(sink, "
                "conductance), INSTALL_LINK(link), REMOVE_LINK(link), PLACE_BLOCK(block, "
                "slot); sink, link and block are the bare bracketed numbers:")
            out.append(
                "  [0]-[%d] heat sinks. MOVE_SINK(sink, site): site is a code s0-s%d from "
                "the Sites list under RESULTS, written with its s (e.g. s%d); a site held by "
                "another sink swaps the two sinks' sites (each keeps its g). "
                "SET_CONDUCTANCE(sink, conductance): the new conductance to ambient, "
                "%s-%s W/K in steps of 0.001, with diminishing returns (sink rule under "
                "RESULTS)." % (K - 1, S - 1, ex_s, _obs_num(G_MIN), _obs_num(G_MAX)))
            out.append(
                "  [%d]-[%d] %ss from a fixed pool: INSTALL_LINK(link) installs one, "
                "REMOVE_LINK(link) removes it." % (K, K + M - 1, kind))
            out.append(
                "  [%d]-[%d] power blocks. PLACE_BLOCK(block, slot): slot is a code q0-q%d "
                "from the Slots list under RESULTS, written with its q (e.g. q%d); a slot "
                "held by another block swaps the two blocks."
                % (K + M, K + M + B - 1, Q - 1, ex_q))
        else:
            out.append(
                "Design elements, addressed by the index in brackets and changed one at "
                "a time with SET_PARAM(index, param, value):")
            out.append(
                "  [0]-[%d] heat sinks, two params each. site: categorical site code 0-%d "
                "(Sites list under RESULTS; write the bare number, e.g. %d for s%d); a site "
                "held by another sink swaps the two sinks' sites (each keeps its g). g: "
                "conductance to ambient, %s-%s W/K in steps of 0.001, with diminishing "
                "returns (sink rule under RESULTS)." % (K - 1, S - 1, ex_s, ex_s,
                                                       _obs_num(G_MIN), _obs_num(G_MAX)))
            out.append(
                "  [%d]-[%d] %ss from a fixed pool. active: write 1 to install, 0 to remove."
                % (K, K + M - 1, kind))
            out.append(
                "  [%d]-[%d] power blocks. slot: categorical slot code 0-%d (Slots list under "
                "RESULTS; write the bare number, e.g. %d for q%d); a slot held by another "
                "block swaps the two blocks." % (K + M, K + M + B - 1, Q - 1, ex_q, ex_q))
        out.append(
            "Cost (to minimise) = sum over sinks of k*g^%s (k = mounting coefficient of the "
            "sink's site; doubling g multiplies it by %s; marginal cost %s*k*g^%s per W/K) "
            "+ sum of c over installed %ss. A sink moved to a free site is re-priced at "
            "k_new*g^%s; moved onto another sink's site, the two swap and BOTH are re-priced "
            "at their new site's k (each keeps its g). Moving or swapping blocks is free."
            % (_obs_num(COST_EXP), _obs_num(2.0 ** COST_EXP), _obs_num(COST_EXP),
               _obs_num(COST_EXP - 1.0), kind, _obs_num(COST_EXP)))
        notes = fx["site_note"]
        if fam in ("plate", "stack") and notes:
            kinds = {}
            for j in range(S):
                kinds.setdefault(notes.get(int(sn[j]), "other"), []).append(j)
            out.append("Site kinds: " + "; ".join(
                "%s %s (k %s-%s)" % (nm, _code_ranges("s", codes),
                                     _obs_num(min(sk[codes])), _obs_num(max(sk[codes])))
                for nm, codes in kinds.items()) + ".")
        lim_s = _obs_num(t_lim, 5) if t_lim is not None else None
        lim_txt = []
        if t_lim is not None:
            lim_txt.append("every node T_i <= %s K" % lim_s)
        if m_lim is not None:
            lim_txt.append("average of T_i over all %d nodes <= %s K" % (n, _obs_num(m_lim, 5)))
        if lim_txt:
            out.append("Temperature limits: " + "; ".join(lim_txt) + ".")
        ks = sorted((float(x) for x in sk), reverse=True)[:K]
        cost_cap = 1.05 * (sum(k * G_MAX ** COST_EXP for k in ks) + float(np.sum(fx["lc"])))
        out.append("A design with any T_i above %s K, or with cost above %s, is outside the "
                   "model's validity and counts as failed." % (_obs_num(TEMP_CAP),
                                                               _obs_num(cost_cap)))
        hi = None
        if self.stochastic:
            sig = fx["sigma"]
            hi = float(_obs_num(math.exp(-0.5 * sig * sig + Z90 * sig)))
            out.append(
                "Power is uncertain: every evaluation multiplies each block's power and "
                "each node's background power by independent mean-one lognormal factors "
                "(sigma %s; 1-in-10 high value x%s). Each limit is judged on the 90th "
                "percentile over 4 or more such draws, fresh for every design, so it can be "
                "violated while every nominal T_i is under it. CONSTRAINT STATUS and the "
                "observed effect show that percentile (a before -> after change of a few "
                "percent can be noise); RESULTS show the nominal solve at the mean powers, "
                "hence a lower max_temp_rise. T is linear in the powers: every power x%s "
                "-> every T_i x%s."
                % (_obs_num(sig), _obs_num(hi), _obs_num(hi), _obs_num(hi)))
        out.append(
            "Metrics: max_temp_rise = max T_i; mean_temp_rise = average T_i; cost = "
            "sink_cost + link_cost; hotspot_node = index i of the hottest node n<i>%s; "
            "temp_negativity = -min T_i (always 0; ignore)."
            % (" (here averaged over the draws, can be fractional; use the nominal "
               "Hotspot line)" if self.stochastic else ""))

        # ── DESIGN ──
        out.append("")
        out.append("DESIGN")
        sink_cost, link_cost = _costs(fx, st["site"], g, st["active"])
        out.append("Heat sinks:")
        for s in range(K):
            j = int(st["site"][s])
            out.append("  [%d] sink: site s%d = %s (k=%s), g=%s W/K, cost %s" % (
                s, j, nl(sn[j]), _obs_num(sk[j]), _obs_num(g[s], 4),
                _obs_num(sk[j] * g[s] ** COST_EXP)))
        on_l = [K + l for l in range(M) if st["active"][l]]
        off_l = [K + l for l in range(M) if not st["active"][l]]
        out.append("%ss installed: %s; not installed: %s (details under RESULTS)."
                   % (kind.capitalize(),
                                        ", ".join("[%d]" % x for x in on_l) or "none",
                                        ", ".join("[%d]" % x for x in off_l) or "none"))
        out.append("Power blocks (P, slot): " + "; ".join(
            "[%d] %s W q%d" % (K + M + b, _obs_num(fx["pb"][b]), int(st["slot"][b]))
            for b in range(B)) + ".")
        out.append("Cost now %s (sinks %s, %ss %s)." % (
            _obs_num(sink_cost + link_cost, 4), _obs_num(sink_cost, 4), kind,
            _obs_num(link_cost, 4)))

        # ── RESULTS ──
        out.append("")
        Rm = None
        if T is not None:
            try:
                Rm = np.linalg.inv(_system(fx, st["site"], g, st["active"]))
                Rm = 0.5 * (Rm + Rm.T)
                if not np.all(np.isfinite(Rm)):
                    Rm = None
            except Exception:
                Rm = None
        if T is None or Rm is None:
            out.append("RESULTS: the steady-state solve failed for this design.")
            return "\n".join(out)
        out.append("RESULTS (%s steady-state solve; T = temperature rise above "
                   "ambient [K])" % ("nominal mean-power" if self.stochastic
                                     else "deterministic"))
        out.append("Temperature map:")
        out.extend(_temp_map(fam, dims, T))
        hot = int(np.argmax(T))
        rh = Rm[hot]
        rmean = Rm.mean(axis=0) if m_lim is not None else None
        order = np.argsort(-T, kind="stable").tolist()
        risk_on = hi is not None and t_lim is not None
        at_mean = " at the mean powers" if self.stochastic else ""
        if t_lim is not None:
            over = [i for i in order if T[i] > t_lim]
            if over:
                shown = over[:8]
                out.append("Over the %s K limit%s (%d node%s): %s%s." % (
                    lim_s, at_mean, len(over), "" if len(over) == 1 else "s",
                    ", ".join("%s +%s K" % (nl(i), _obs_num(T[i] - t_lim)) for i in shown),
                    ", + %d more" % (len(over) - len(shown)) if len(over) > len(shown) else ""))
            else:
                out.append("No node is over the %s K limit%s; closest: %s." % (
                    lim_s, at_mean, ", ".join(
                        "%s %s K (%s K below)" % (nl(i), _obs_num(T[i]), _obs_num(t_lim - T[i]))
                        for i in order[:3])))
            if risk_on:
                risk = [i for i in order if T[i] <= t_lim < T[i] * hi]
                if risk:
                    out.append("AT RISK (under the limit at the mean powers, over it with "
                               "every power at x%s; %d node%s): %s." % (
                                   _obs_num(hi), len(risk), "" if len(risk) == 1 else "s",
                                   ", ".join("%s %s K -> %s K" % (
                                       nl(i), _obs_num(T[i]), _obs_num(T[i] * hi))
                                       for i in risk[:6])))
        out.append("Hotspot: %s, T=%s K (hotspot_node = %d)." % (nl(hot), _obs_num(T[hot]), hot))
        total_P = float(P.sum())
        sunk = float(sum(g[s] * T[int(sn[int(st["site"][s])])] for s in range(K)))
        conv = float(h0 * T.sum())
        out.append("Heat balance: total power %s W = %s W through the sinks + %s W by "
                   "natural convection." % (_obs_num(total_P), _obs_num(sunk), _obs_num(conv)))
        if m_lim is not None:
            need = total_P - n * h0 * m_lim
            out.append("Average T over all nodes: %s K (limit %s K)." % (
                _obs_num(float(np.mean(T)), 4), _obs_num(m_lim, 5)))
            out.append(
                "Average T = (total power - heat through the sinks)/(n*h0) = (%s W - sinks)/"
                "(%d x %s W/K): the %s K average limit holds exactly when the sinks remove at "
                "least %s W (now %s W%s); each extra W through the sinks lowers the average "
                "by %s K, and a move that does not raise the sinks' total heat does not "
                "lower it."
                % (_obs_num(total_P, 4), n, _obs_num(h0), _obs_num(m_lim, 5), _obs_num(need, 4),
                   _obs_num(sunk, 4),
                   (", short by %s W" % _obs_num(need - sunk)) if need > sunk else "",
                   _obs_num(1.0 / (n * h0))))
        mean_txt = m_lim is not None
        out.append(
            "Thermal resistances of this design (every sink and installed %s as now): "
            "R_ij [K/W] = rise of T_i per extra W injected at node j; R is symmetric; "
            "Rself_j = R_jj; Rhot_j = R_ij with i = the hotspot %s%s. The Slots and Sites "
            "lists give these at every slot and site node."
            % (kind, nl(hot), "; Rmean_j = average over i of R_ij" if mean_txt else ""))
        out.append(
            "Exact effect of ONE change on any T_i (T is linear in the powers). Block of P W "
            "moved from node a to b: P*(R_ib - R_ia) (a swap moves P_this - P_other); node b "
            "itself rises by at most P*Rself_b. Sink on node j changed by dg: "
            "-dg*T_j*R_ij/(1 + dg*Rself_j) (flattens once dg*Rself_j >> 1). %s of "
            "conductance w installed between a and b: carries f = w*(T_a - T_b)/(1 + "
            "w*Reff), Reff = R_aa + R_bb - 2*R_ab, far below w*(T_a - T_b); T_i changes by "
            "-f*(R_ia - R_ib). Installed one carrying f, removed: +f*(R_ia - R_ib)/(1 - "
            "w*Reff). Sink moved from node o to a free site j: roughly the sink rule with "
            "dg = -g at o plus dg = +g at j (unreliable within 2 links).%s"
            % (kind.capitalize(), " Rmean in place of R_ij gives the change of the average "
               "T." if mean_txt else ""))
        out.append("Heat sinks (removes = g*T of its node; effects from the sink rule, every "
                   "other element unchanged):")
        for s in range(K):
            j = int(sn[int(st["site"][s])])
            q = g[s] * T[j]
            rjj = float(Rm[j, j])
            d1 = 0.1 * T[j] * rh[j] / (1.0 + 0.1 * rjj)
            txt = ("  [%d] at %s: T=%s K, removes %s W; Rself %s, Rhot %s K/W; "
                   "hotspot -%s K per +0.1 W/K" % (
                       s, nl(j), _obs_num(T[j]), _obs_num(q), _obs_num(rjj),
                       _obs_num(rh[j]), _obs_num(d1)))
            if mean_txt:
                txt += ", average -%s K" % _obs_num(0.1 * T[j] * rmean[j] / (1.0 + 0.1 * rjj))
            dg2 = G_MAX - float(g[s])
            if dg2 > 1e-9:
                Tn = T - dg2 * T[j] * Rm[:, j] / (1.0 + dg2 * rjj)
                h2 = int(np.argmax(Tn))
                txt += "; at g=%s: hotspot node %s K" % (_obs_num(G_MAX), _obs_num(Tn[hot]))
                txt += (" (still the hottest)" if h2 == hot else
                        ", hottest node then %s at %s K" % (nl(h2), _obs_num(Tn[h2])))
            else:
                txt += "; already at the %s W/K maximum" % _obs_num(G_MAX)
            out.append(txt)
        out.append("%ss (w = conductance W/K, c = cost; T of the two ends in the order "
                   "listed):" % kind.capitalize())
        for l in range(M):
            a, b = int(fx["la"][l]), int(fx["lb"][l])
            w = float(fx["lw"][l])
            reff = float(Rm[a, a] + Rm[b, b] - 2.0 * Rm[a, b])
            base = "  [%d] %s-%s w=%s c=%s" % (K + l, nl(a), nl(b), _obs_num(w),
                                               _obs_num(fx["lc"][l]))
            if st["active"][l]:
                f = w * (T[a] - T[b])
                src, dst = (a, b) if f >= 0 else (b, a)
                den = max(1.0 - w * reff, 1e-12)
                txt = ("%s INSTALLED: T %s / %s K, carries %s W n%d->n%d; Reff %s K/W; "
                       "if removed: hotspot %s K" % (
                           base, _obs_num(T[a]), _obs_num(T[b]), _obs_num(abs(f)), src, dst,
                           _obs_num(reff), _obs_signed(f * (rh[a] - rh[b]) / den)))
                if mean_txt:
                    txt += ", average %s K" % _obs_signed(f * (rmean[a] - rmean[b]) / den)
            else:
                f = w * (T[a] - T[b]) / (1.0 + w * reff)
                src, dst = (a, b) if f >= 0 else (b, a)
                ta = T[a] - f * (Rm[a, a] - Rm[a, b])
                tb = T[b] - f * (Rm[b, a] - Rm[b, b])
                txt = ("%s not installed: T %s / %s K; Reff %s K/W; if installed: carries "
                       "%s W n%d->n%d, hotspot %s K" % (
                           base, _obs_num(T[a]), _obs_num(T[b]), _obs_num(reff),
                           _obs_num(abs(f)), src, dst, _obs_signed(-f * (rh[a] - rh[b]))))
                if mean_txt:
                    txt += ", average %s K" % _obs_signed(-f * (rmean[a] - rmean[b]))
                txt += ", ends %s / %s K" % (_obs_num(ta), _obs_num(tb))
            out.append(txt)
        out.append("Power blocks:")
        for b in range(B):
            node = int(qn[int(st["slot"][b])])
            txt = "  [%d] P=%s W, slot q%d = %s: T=%s K" % (
                K + M + b, _obs_num(fx["pb"][b]), int(st["slot"][b]), nl(node),
                _obs_num(T[node]))
            if t_lim is not None:
                if T[node] > t_lim:
                    txt += " OVER +%s K" % _obs_num(T[node] - t_lim)
                else:
                    txt += " (%s K below the limit)" % _obs_num(t_lim - T[node])
            if node == hot:
                txt += " HOTSPOT"
            out.append(txt)
        rcols = "Rself Rhot" + (" Rmean" if mean_txt else "")
        # plate labels only restate row/column of the index; stack/board labels
        # carry the layer/module, so they stay in the long lists
        ll = (lambda x: "n%d" % int(x)) if fam in ("plate", "mesh") else nl

        def rvals(x):
            v = "%s %s" % (_obs_num(Rm[x, x]), _obs_num(rh[x]))
            return v + (" %s" % _obs_num(rmean[x]) if mean_txt else "")

        slot_txt = []
        for q in range(Q):
            x = int(qn[q])
            e = "q%d %s %s %s%s" % (q, ll(x), _obs_num(T[x]),
                                    ("[%d] " % (K + M + slot_of[q])) if q in slot_of else "",
                                    rvals(x))
            if x in sink_at:
                e += " +sink[%d]" % sink_at[x]
            elif x in site_code_at:
                e += " site s%d" % site_code_at[x]
            slot_txt.append(e)
        out.append("Slots (code node T [block] %s; '+sink[s]' = a sink on that node, "
                   "'site sJ' = a free sink site there): " % rcols
                   + "; ".join(slot_txt) + ".")
        site_txt = []
        for j in range(S):
            x = int(sn[j])
            e = "s%d %s %s %s %s%s" % (j, ll(x), _obs_num(sk[j]), _obs_num(T[x]),
                                       ("[%d] " % site_of[j]) if j in site_of else "",
                                       rvals(x))
            if x in block_at:
                e += " +block[%d]" % (K + M + block_at[x])
            elif x in slot_code_at:
                e += " slot q%d" % slot_code_at[x]
            site_txt.append(e)
        out.append("Sites (code node k T [sink] %s; '+block[b]' = a block on that node, "
                   "'slot qX' = a free block slot there): " % rcols
                   + "; ".join(site_txt) + ".")
        return "\n".join(out)

    def element_label(self, st, i) -> str:
        """Short role label as of the state passed in (the caller passes the
        state AFTER the change it labels; W1 fix of 2026-10), e.g. 'sink [1]
        (after this change: site s4 = n12 r1c2, g=0.812 W/K)'."""
        fx = st["fx"]
        kind, j = self._kind(st, int(i))
        labels = fx["labels"]

        def nl(x):
            x = int(x)
            return "n%d" % x + ((" " + labels[x]) if labels[x] else "")

        T = self._nominal(st)
        if kind == "sink":
            code = int(st["site"][j])
            node = int(fx["site_nodes"][code])
            bits = ["site s%d = %s" % (code, nl(node)), "g=%s W/K" % _obs_num(st["g"][j], 4)]
            return "sink [%d] (after this change: %s)" % (int(i), ", ".join(bits))
        if kind == "link":
            on = "installed" if st["active"][j] else "not installed"
            return "%s [%d] (%s-%s, %s after this change)" % (
                fx["link_kind"], int(i), nl(fx["la"][j]), nl(fx["lb"][j]), on)
        if kind == "block":
            code = int(st["slot"][j])
            node = int(fx["slot_nodes"][code])
            where = "slot q%d = %s" % (code, nl(node))
            if T is not None and int(np.argmax(T)) == node:
                where += ", the hotspot"
            return "power block [%d] (P=%s W; after this change: %s)" % (
                int(i), _obs_num(fx["pb"][j]), where)
        return "element %d" % int(i)


# ── observation helpers ──────────────────────────────────────────────────────
def _temp_map(fam: str, dims, T: np.ndarray) -> List[str]:
    """Per-node values laid out by the family's geometry (any per-node array)."""
    lines = []
    if fam == "plate":
        rows, cols = dims
        for r in range(rows):
            lines.append("  r%d: %s" % (r, " ".join(_obs_num(T[r * cols + c])
                                                   for c in range(cols))))
    elif fam in ("board", "stack"):
        m, mr, mc = dims
        per = mr * mc
        tag = "m" if fam == "board" else "L"
        for j in range(m):
            rows = []
            for r in range(mr):
                rows.append("r%d: %s" % (r, " ".join(_obs_num(T[j * per + r * mc + c])
                                                    for c in range(mc))))
            lines.append("  %s%d | %s" % (tag, j, " | ".join(rows)))
    else:
        n = T.shape[0]
        for s in range(0, n, 10):
            e = min(n, s + 10)
            lines.append("  n%d-n%d: %s" % (s, e - 1, " ".join(_obs_num(T[i])
                                                             for i in range(s, e))))
    return lines


def _code_ranges(prefix: str, codes) -> str:
    """[0,1,2,5,7,8] -> 's0-s2, s5, s7-s8' (codes ascending)."""
    codes = sorted(int(c) for c in codes)
    parts = []
    i = 0
    while i < len(codes):
        j = i
        while j + 1 < len(codes) and codes[j + 1] == codes[j] + 1:
            j += 1
        parts.append("%s%d" % (prefix, codes[i]) if i == j
                     else "%s%d-%s%d" % (prefix, codes[i], prefix, codes[j]))
        i = j + 1
    return ", ".join(parts)


def _obs_signed(x, sig: int = 3) -> str:
    """Signed compact number: '+0.31', '-2.4', '0'."""
    x = float(x)
    s = _obs_num(abs(x), sig)
    if s == "0":
        return "0"
    return ("+" if x > 0 else "-") + s


def _obs_num(x, sig: int = 3) -> str:
    """``sig`` significant figures; tiny values in compact exponent form
    (copied from the v1 thermal module so v1 can change independently)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    if x == 0.0:
        return "0"
    if abs(x) < 1e-4:
        mant, ex = ("%.*e" % (max(sig - 1, 0), x)).split("e")
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        return "%se%d" % (mant, int(ex))
    mag = int(math.floor(math.log10(abs(x))))
    dec = max(0, sig - 1 - mag)
    s = "%.*f" % (dec, x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s
