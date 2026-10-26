"""3D pin-jointed truss design with a FULL action set (v2 of frame3d):
member sizing, member material, member topology (on/off over a fixed ground
structure) and joint shape moves, all through SET_PARAM(element, param, value)
internally; the policy reads and writes them as the verbs of the "Verb
surface" section at the end of this docstring.

Physics
-------
Linear direct-stiffness analysis of a pin-jointed space truss (3 translational
DOF per node, axial members only). Member stiffness E*A/L along its direction
cosines; supports are fully pinned nodes (x, y, z fixed). Members are thin
tubes with wall t = 0.2 r, so A = 0.36 pi r^2 and I = 0.1476 pi r^4 (the v1
section, which can never invert). Checks per ACTIVE member:
  FOS_yielding = sigma_y(mat) / (|N| / A)              (all members)
  FOS_buckling = pi^2 E(mat) I / (L^2 |N|)             (compression only,
                                                        pin-ended Euler)
  mass         = sum over active members of rho(mat) * A * L
  max_deflection_m = largest nodal displacement magnitude (Euclidean norm)
  fos_shortfall = sum over active members of max(0, 1 - FOS_gov / 1.5), the
                   total relative shortfall of each member's governing FOS
A design whose free-DOF stiffness matrix is not positive definite (Cholesky
fails, a free DOF has no stiffness, or a relative pivot falls below 1e-9) is a
MECHANISM: evaluate() then returns finite, maximally bad metrics (FOS 0,
shortfall = number of active members, deflection 10 x span, mechanism = 1)
that the validity envelope marks invalid. Nothing non-finite reaches Phi.

Geometry (families, as v1, plus taper and load eccentricity)
--------
A chain of cross-sections along an axis, every node of section s braced to
three nodes of section s-1 (Henneberg attachment: the base structure is rigid).
  tower       square sections stacked along +z, base section pinned; lateral
              wind (+x) plus dead load (-z) at the top section, plus a torsion
              couple (+-y) of relative size `ecc`; sections taper linearly to
              `taper` x width at the top.
  cantilever  square sections along +x, root section pinned; tip load (-z)
              split unevenly between the two sides (1 +- ecc): bending + torsion;
              sections taper linearly to `taper` x width at the tip.
  bridge      triangular girder along +x (two deck nodes, one apex), both end
              sections pinned; vertical load on every interior node, deck side
              y- heavier by (1 + ecc) and y+ lighter by (1 - ecc); apex height
              h_s = 0.85 w (taper + (1 - taper) sin(pi s / n)), i.e. shallow
              near the supports.
Size ladder: n_bay in (2, 3, 4, 5, 6, 8) (bridges >= 4), as v1.

Elements (a FIXED list per problem; indices never change)
--------
  members 0 .. M-1     the GROUND STRUCTURE: the v1 base members first (v1
                       order, minus the members joining two supports, which
                       carry no force and add no stiffness under pinned
                       supports), then extra (optional) members:
                         sdiag2  the missing second in-section diagonal
                                 (corner 1 - corner 3) of every square section
                                 that is not fully pinned;
                         xdiag   the missing body diagonal between adjacent
                                 square sections (corner j of s-1 to corner
                                 j+2 of s), length cap 2.25 x max(pitch, width);
                         skip    (triangular girders, whose adjacent sections
                                 are already fully connected) two-bay braces
                                 from corner j of section s to a DIFFERENT
                                 corner of section s+2, same length cap.
  joints  M .. M+J-1   every free (non-support) node, in node order.

Params (class attribute `params` lists all; get() returns None where a param
does not apply to the element)
  r       member, continuous, outer tube radius [m], bounds [0.004, 0.12]
          (v1 bounds); moves scale it by (0.8, 0.9, 1.1, 1.25); set() rounds
          to 1e-5 m.
  active  member, toggle 0/1: an inactive member carries no stiffness and no
          mass. Initial: base members 1, extra members 0.
  mat     member, categorical code (integer-valued float):
            0 = 6061-T6 aluminium     E 68.9 GPa,  sigma_y 276 MPa, rho 2700
            1 = S355 structural steel E 210 GPa,   sigma_y 355 MPa, rho 7850
            2 = Ti-6Al-4V titanium    E 113.8 GPa, sigma_y 880 MPa, rho 4430
          Initial: all 0.
  dx, dy, dz
          joint, additive offset of the joint from its initial position [m];
          bounds +-0.25 b, moves +-0.05 b, where b = min(pitch, smallest
          section width) is the bay size; set() clamps and rounds to 1e-4 m.
          Loads at a loaded joint move with it (nodal forces); supports never
          move. A LOADED joint keeps its coordinate along the structure's
          axis (tower: dz, cantilever and bridge: dx): where a load acts along
          the height / reach / span is part of the brief. get() returns None
          for that param on such a joint and set() ignores it; the two
          transverse offsets stay free. (Unrestricted, the loaded tower top
          and cantilever tip slid toward the support to shorten the lever
          arm.)

Problem (make_problem)
-------
families x size ladder (3 x 6, as v1) x random geometry (width 0.8-1.6 m,
pitch 0.9-1.8 m, taper 0.55-1, load eccentricity 0-0.6) x two load classes x
two damage scenarios:
  load class   light (40%): main load 4-26 kN (v1 range);
               heavy (60%): main load scaled until the aluminium FSD sized
               WITHOUT the tube-size cap needs a largest tube of 1.03-1.25 x
               the largest radius made (0.12 m). The few most loaded members
               then cannot be made strong enough in aluminium by radius
               alone; that capacity must come from material, bracing or
               geometry. (At 0.85-0.97 of the cap, as first built,
               radius-only depth-1 selection still solved 16/18 heavy
               problems; at 1.03-1.25 it solves 4/18 and the full move set
               18/18.)
  scenario     undersized: 4-7 (heavy: 2-5) force-carrying base members
               under-built (radius x 0.45-0.92, staggered so repairing the
               worst raises the minimum to the next);
               point_load: an extra concentrated load (0.1-0.3 x main load, at
               an interior joint) the start design was never sized for, plus
               1-3 (heavy: 0-2) under-built members.
The start design is the aluminium FSD of the base structure, capped at the
largest tube (sized for the loading WITHOUT the scenario's extra point load),
with every member 4% over-built (within the cap), damaged as above; extras
inactive, offsets 0. In the heavy class the capped members start failing.
FSD design rules:
every member is sized for at least a notional 3% of the largest member force
(taken as compression) and for slenderness L / r_gyration <= 250 (r >= L/160),
so near-zero-force bracing is not sized to the 4 mm minimum. The deflection
limit is 0.70-1.0 x the deflection of the sound (undamaged) start design under
the full loading. A seeded acceptance loop redraws the damage, extra load and
limit (each retry gentler) until the start is infeasible with at most 10
failing members and deflection at most 1.35 x the limit (10 tries).
Objective: total mass (min). Constraints (goals): FOS_buckling >= 1.5,
FOS_yielding >= 1.5, fos_shortfall <= 0, max_deflection_m <= limit. The
shortfall constraint is satisfied exactly when the two FOS constraints are (it
adds no requirement), but it credits repairing ANY failing member: with only
the two minima, fixing one of several similar members leaves the minimum
unchanged, and depth-1 selection stalled on such plateaus. Validity envelope:
max_deflection_m <= 0.1 x span, mechanism <= 0.5.

Reference (reference_class = "heuristic"): the lighter FEASIBLE one of (a)
aluminium FSD and (b) FSD with per-member material selection (each member
takes the material needing the least mass for its current force among those
whose tube fits under the cap), each on the base topology and the initial
geometry under the full loading, then uniformly scaled (all radii by the same
factor, which leaves the force distribution unchanged) until the deflection
limit is met. A design that fails any check is used only if neither passes
(spec "_ref_kind" then ends in "(infeasible)"). Topology, shape,
mixed-material and targeted (non-uniform) stiffening moves can beat it.

Typed moves offered per turn (candidate_moves; 104-154, every stratum every
turn): radius x1.1 / x1.25 and "resize to FOS 1.5 at the current force" on the
lowest-FOS members and on the largest deflection contributors, x0.8 / x0.9 and
resize-down on the least-utilised members, plus seeded random ones; every other
material on the critical / deflection / least-utilised members plus random
ones; installing optional members incident to the critical members' nodes and
to the largest-deflection node plus random ones; removing low-utilisation
members whose redundancy number exceeds 0.05 (removal keeps the truss rigid);
+-1 step along every free axis (not a loaded joint's axial one) for the
joints at the critical members' ends and the largest-deflection node, plus
random joint steps.
Greedy (hint / hint_move): lowest-FOS member radius x1.25 while any FOS fails
(the v1 hint); when only the deflection limit fails, install the shortest
optional member at the largest-deflection node, else radius x1.25 on the
largest deflection contributor.

Stochastic variant: every evaluation scales all loads by a lognormal factor
(mean 1, sd 8%) and turns them about z by a normal angle (sd 0.03 rad), drawn
from np.random.default_rng(seed) only (the v1 noise model). Its resize moves
target FOS 1.5 x 1.15 (about the q90 load) instead of x 1.03. The stiffness
factorisation does not depend on the seed and is shared between replicates of
the same design, so replicates cost a few vector operations each.

Simplifications (explicit)
  * Linear elastic, small displacements; no geometric stiffness, no
    second-order effects; pin joints (no moments), loads at nodes only.
  * Buckling is pin-ended Euler buckling of each member alone (K = 1); no
    joint or overall (global) buckling check, no imperfections.
  * Self-weight is not a load; mass is only the objective.
  * Tube wall is a fixed fraction of radius (t = 0.2 r); material properties
    are nominal handbook values, no partial safety factors beyond FOS 1.5.
  * Overlapping/crossing members do not interact; joint moves do not check
    member clashes (offsets are bounded so no member can reach zero length).
  * Topology is limited to the fixed ground structure; nodes are never added.
  * Joint moves keep the loads' magnitudes and directions (nodal forces); a
    load moves with its joint across the axis (e.g. a bridge deck chord
    moved sideways carries its deck load with it), never along it.
  * One load case per problem (plus the stochastic scatter); no load
    combinations, no dynamic effects.

Python >= 3.8, numpy/scipy only. The v1 module (frame3d.py) is imported for
its geometry builder and text helpers and is not modified.

Verb surface (docs/dataxp_verb_surface.md; owner's decision of 2026-10-23)
------------
The move set (candidate_moves), set(), Phi and the stored {i, p, v, v_old}
dicts are the v2 ones above and are untouched; only the spelling in prompts,
action lines and reply parsing changes (class attributes verb_surface and
verb_table; format_move / parse_move / grammar come from the table in
dataxp.domain). The bijection, per parameter:
  active 1 / 0        <-> INSTALL_MEMBER(m) / REMOVE_MEMBER(m)
  mat k               <-> SET_MATERIAL(m, name), name = Al6061T6 / S355 / Ti6Al4V
  r v                 <-> SET_RADIUS(m, v)                        [m]
  dx / dy / dz v      <-> MOVE_JOINT(J, x / y / z, v)             [m]
Judgment calls, recorded here as the surface rules ask:
  * m and J are the element numbers SET_PARAM used (members 0..M-1; joints
    M..M+J-1, the number after J in the observation), not node numbers.
  * MOVE_JOINT's offset is the new TOTAL offset from the initial position, as
    SET_PARAM's value was; it is not a step.
  * Material names are the alloy designations written as identifiers
    (Al6061T6, S355, Ti6Al4V). parse_move also accepts the observation's row
    codes Al / St / Ti (aliases) and ignores case; format_move writes the
    canonical name only, so the surface is one-to-one on the legal moves. A
    bare code number (SET_MATERIAL(3, 2)) is not accepted.
  * SET_PARAM(...) text is no longer parsed on this domain: the grammar the
    policy is served lists the verbs only.
  * A MOVE_JOINT along a loaded joint's locked axis parses (as the SET_PARAM
    did) and apply() leaves the state unchanged, exactly as set() did;
    legal_moves never offers it. Nothing was added to or removed from the
    move set.
  * describe()'s CHANGES paragraph is written in the verbs (it is observation
    text only; no generated file contains it). The member rows keep their
    Al / St / Ti codes and the SET_MATERIAL bullet ties each name to its code.
  * History and "change you are about to make" lines print the replaced value
    ("was 0.02", "material Al6061T6 -> Ti6Al4V") for radius, material and
    joint moves; the install / remove pair needs none.

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
-----------------------
The same moves written through the one wrapper tool, generated from the
verb table by dataxp.domain (set_param_format / set_param_parse /
set_param_grammar; no override here): INSTALL_MEMBER(3) <-> SET_PARAM(3,
active, 1), REMOVE_MEMBER(3) <-> SET_PARAM(3, active, 0), SET_MATERIAL(3,
Ti6Al4V) <-> SET_PARAM(3, mat, 2) (the stored code; the grammar names the
codes), SET_RADIUS(3, 0.031) <-> SET_PARAM(3, r, 0.031), MOVE_JOINT(12, x,
0.25) <-> SET_PARAM(12, dx, 0.25). Judgment calls: the operation phrases are
"Install a member" / "Remove a member" / "Set a member's material" / "Resize
a member's tube" / "Move a joint along x|y|z"; describe()'s CHANGES paragraph
reverts to the SET_PARAM block (the run2 text) under this spelling
(spelled_in_verbs), so the two prompts differ in action syntax only; the
history "was ..." prints the stored material code, not the name.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_solve

from ..domain import (ArgSpec, Candidate, Categorical, Continuous, Domain, ParamVerb, Toggle,
                      spelled_in_verbs)
from ..potential import DesignProgram, program_from_goals
from .frame3d import (
    A_COEF,
    FOS_TARGET,
    I_COEF,
    R_MAX,
    R_MIN,
    WALL_RATIO,
    Frame3DDomain,
    _OBS_CORNER,
    _OBS_EDGE,
    _OBS_FACE_LONG,
    _obs_fos,
    _obs_num,
    _obs_runs,
)

# ── materials ────────────────────────────────────────────────────────────────
MATERIALS = (
    # name, short, E [Pa], sigma_y [Pa], rho [kg/m3]
    ("6061-T6 aluminium", "Al", 68.9e9, 276e6, 2700.0),
    ("S355 structural steel", "St", 210.0e9, 355e6, 7850.0),
    ("Ti-6Al-4V titanium", "Ti", 113.8e9, 880e6, 4430.0),
)
E_TAB = np.array([m[2] for m in MATERIALS], dtype=float)
SY_TAB = np.array([m[3] for m in MATERIALS], dtype=float)
RHO_TAB = np.array([m[4] for m in MATERIALS], dtype=float)
N_MAT = len(MATERIALS)
# verb-surface names of the material codes (docstring "Verb surface")
MAT_NAMES = ("Al6061T6", "S355", "Ti6Al4V")
_MAT_DOC = (", ".join("%s = %s (E %g GPa, yield %g MPa, %g kg/m3)"
                      % (MAT_NAMES[q], MATERIALS[q][0], E_TAB[q] / 1e9, SY_TAB[q] / 1e6,
                         RHO_TAB[q]) for q in range(N_MAT))
            + "; the member rows abbreviate them Al / St / Ti.")
_MEMBER_ARG = ArgSpec("member", "int", "member number 0..M-1 (the number after m)")
_JOINT_ARG = ArgSpec("joint", "int", "joint number (the number after J)")
_OFFSET = Continuous("offset", "m", "the joint's new total offset from its initial position "
                                    "along that axis, within the +- bound the observation "
                                    "states")
_JOINT_DOC = ("move a free joint along one axis; every member at it follows. A loaded "
              "joint's coordinate along the structure's axis is locked: such a move changes "
              "nothing.")

PARAMS = ("r", "active", "mat", "dx", "dy", "dz")
MEMBER_PARAMS = ("r", "active", "mat")
JOINT_PARAMS = ("dx", "dy", "dz")
R_FACTORS = (0.8, 0.9, 1.1, 1.25)
OFFSET_BOUND_FRAC = 0.25
OFFSET_STEP_FRAC = 0.05
PIVOT_TOL = 1e-9
FOS_CAP = 1.0e3
SAFE_REDUNDANCY = 0.05      # deactivation offered only above this
SLENDER_DIV = 160.0         # FSD minimum size r >= L / 160 (L/r_g <= 250)
NOTIONAL = 0.03             # FSD notional member force, x largest force
LOAD_SIGMA = 0.08           # stochastic variant: lognormal load-scale sd (v1)
LOAD_ROT_SD = 0.03          # stochastic variant: load rotation about z [rad]
# resize-to-FOS-1.5 move targets (x FOS 1.5): up for failing members, down for
# the least-utilised; the stochastic variant sizes for roughly the q90 load
# (offered for members below 1.5 x 1.15 there)
RESIZE_UP = {False: 1.03, True: 1.15}
RESIZE_DOWN = {False: 1.10, True: 1.20}
# heavy load class: largest tube of the UNCAPPED aluminium FSD, x R_MAX
HEAVY_CAP = (1.03, 1.25)

_FAMILIES = ("tower", "cantilever", "bridge")
_BAY_LADDER = (2, 3, 4, 5, 6, 8)


def _round_r(v: float) -> float:
    return round(min(R_MAX, max(R_MIN, float(v))), 5)


# ── geometry ─────────────────────────────────────────────────────────────────
def _ground(spec: dict, with_extra: bool = True) -> dict:
    """Immutable per-problem geometry: nodes, ground structure, loads.
    with_extra=False drops the scenario's extra point load (the loading the
    start design was originally sized for)."""
    fam = str(spec["family"])
    n_bay = int(spec["n_bay"])
    width = float(spec["width"])
    pitch = float(spec["pitch"])
    taper = float(spec.get("taper", 1.0))
    ecc = float(spec.get("ecc", 0.0))
    nodes, base, fixed = Frame3DDomain._build(fam, n_bay, width, pitch)
    nodes = np.array(nodes, dtype=float)
    k = 3 if fam == "bridge" else 4
    n_node = nodes.shape[0]
    sec = np.arange(n_node) // k
    cor = np.arange(n_node) % k

    # taper / camber of the initial geometry
    if fam == "tower":
        g = 1.0 - (1.0 - taper) * sec / max(n_bay, 1)
        nodes[:, 0] *= g
        nodes[:, 1] *= g
    elif fam == "cantilever":
        g = 1.0 - (1.0 - taper) * sec / max(n_bay, 1)
        nodes[:, 1] *= g
        nodes[:, 2] *= g
    else:
        h = 0.85 * width
        apex = cor == 2
        nodes[apex, 2] = h * (taper + (1.0 - taper)
                              * np.sin(math.pi * sec[apex] / max(n_bay, 1)))

    fixed_set = set(int(v) for v in fixed)
    base_all = [(int(a), int(b)) for a, b in np.asarray(base)]
    # A member joining two pinned supports carries no force and adds no
    # stiffness to the free DOFs: it is not part of the design (extras skip
    # such pairs too). Kept, the FSD slenderness floor gave these inert
    # members 5-8% of the design mass, which removal then "saved" for free.
    base_pairs = [(a, b) for a, b in base_all
                  if not (a in fixed_set and b in fixed_set)]
    existing = set(base_all)
    cap = 2.25 * max(pitch, width)

    def length(a, b):
        return float(np.linalg.norm(nodes[b] - nodes[a]))

    extras, ekind = [], []
    if k == 4:
        for s in range(n_bay + 1):
            a, b = s * k + 1, s * k + 3
            if a in fixed_set and b in fixed_set:
                continue
            if (a, b) not in existing:
                extras.append((a, b))
                ekind.append("sdiag2")
        for s in range(1, n_bay + 1):
            for j in range(k):
                a, b = (s - 1) * k + (j + 2) % k, s * k + j
                key = (min(a, b), max(a, b))
                if key in existing or (a in fixed_set and b in fixed_set):
                    continue
                if length(a, b) <= cap:
                    extras.append(key)
                    ekind.append("xdiag")
                    existing.add(key)
    else:
        for s in range(0, n_bay - 1):
            for ja in range(k):
                for jb in range(k):
                    if ja == jb:
                        continue
                    a, b = s * k + ja, (s + 2) * k + jb
                    if a in fixed_set and b in fixed_set:
                        continue
                    if length(a, b) <= cap:
                        extras.append((a, b))
                        ekind.append("skip")

    members = np.array(base_pairs + extras, dtype=np.int64)
    n_base = len(base_pairs)
    M = members.shape[0]
    kinds = []
    for m in range(n_base):
        a, b = base_pairs[m]
        sa, sb, ja, jb = a // k, b // k, a % k, b % k
        if sa == sb:
            kinds.append("sdiag" if (k == 4 and {ja, jb} == {0, 2}) else "ring")
        elif ja == jb:
            kinds.append("leg" if fam == "tower" else "chord")
        else:
            kinds.append("diag")
    kinds += ekind

    free = np.ones(3 * n_node, dtype=bool)
    for nd in fixed_set:
        free[3 * nd:3 * nd + 3] = False
    fmap = -np.ones(3 * n_node, dtype=np.int64)
    fmap[free] = np.arange(int(free.sum()))
    joints = [n for n in range(n_node) if n not in fixed_set]

    # loads (nodal forces; they ride with their node)
    P = np.zeros((n_node, 3), dtype=float)
    total = float(spec["load_n"])
    if fam == "tower":
        top = list(range(n_node - k, n_node))
        for nd in top:
            P[nd, 0] += total / k
            P[nd, 2] -= 0.35 * total / k
            P[nd, 1] += ecc * total / k * (1.0 if nodes[nd, 0] > 0 else -1.0)
    elif fam == "cantilever":
        tip = list(range(n_node - k, n_node))
        for nd in tip:
            side = 1.0 + ecc if nodes[nd, 1] < 0 else 1.0 - ecc
            P[nd, 2] -= total / k * side
    else:
        interior = [nd for nd in range(k, n_node - k)] or list(range(k, n_node))
        w = []
        for nd in interior:
            j = nd % k
            w.append(1.0 + ecc if j == 0 else (1.0 - ecc if j == 1 else 1.0))
        w = np.asarray(w, dtype=float)
        w = w / w.sum()
        for nd, wt in zip(interior, w):
            P[nd, 2] -= total * wt
    xl = spec.get("extra_load") if with_extra else None
    if xl:
        P[int(xl["node"])] += np.asarray(xl["f"], dtype=float)
    # A loaded joint keeps its coordinate along the structure's axis (tower
    # height z, cantilever reach / bridge span x): where a load acts along
    # the structure is part of the brief. Free to move, the loaded tower top
    # and cantilever tip slid toward the support and shortened the lever arm
    # (half of all gold joint moves). Moves across the axis stay allowed.
    axis = 2 if fam == "tower" else 0
    locked = np.zeros((n_node, 3), dtype=bool)
    locked[np.any(P != 0.0, axis=1), axis] = True
    locked[sorted(fixed_set)] = True
    # load decomposition for the rotation noise: P = Pxy + Pz, Pperp = Rz90 Pxy
    Pxy = P.copy()
    Pxy[:, 2] = 0.0
    Pperp = np.zeros_like(P)
    Pperp[:, 0] = -P[:, 1]
    Pperp[:, 1] = P[:, 0]
    Pz = np.zeros_like(P)
    Pz[:, 2] = P[:, 2]
    rhs = np.stack([Pxy.reshape(-1)[free], Pperp.reshape(-1)[free],
                    Pz.reshape(-1)[free]], axis=1)

    span = float(np.ptp(nodes, axis=0).max())
    sec_w = []
    for s in range(n_bay + 1):
        ids = list(range(s * k, (s + 1) * k))
        pts = nodes[ids]
        d = [np.linalg.norm(pts[a] - pts[b]) for a in range(k) for b in range(a + 1, k)]
        sec_w.append(min(d))
    bay = min(pitch, min(sec_w))
    off_bound = round(OFFSET_BOUND_FRAC * bay, 4)
    off_step = round(OFFSET_STEP_FRAC * bay, 4)

    inc: Dict[int, List[int]] = {}
    for m in range(M):
        a, b = int(members[m, 0]), int(members[m, 1])
        inc.setdefault(a, []).append(m)
        inc.setdefault(b, []).append(m)

    return {
        "family": fam, "k": k, "n_bay": n_bay, "nodes0": nodes,
        "members": members, "n_base": n_base, "kinds": tuple(kinds),
        "fixed": sorted(fixed_set), "free": free, "fmap": fmap,
        "n_free": int(free.sum()), "n_node": n_node, "joints": joints,
        "P": P, "rhs": rhs, "span": max(span, 1e-6), "bay": bay,
        "off_bound": off_bound, "off_step": off_step, "inc": inc,
        "sec": sec, "cor": cor, "axis": axis, "locked": locked,
    }


# ── analysis core ────────────────────────────────────────────────────────────
def _analyse(geo: dict, r: np.ndarray, act: np.ndarray, mat: np.ndarray,
             off: np.ndarray) -> dict:
    """Factorise the free-DOF stiffness of the active members and solve the
    three load components. Returns a dict; {"mech": True, ...} for a
    mechanism. Pure function of its inputs."""
    nodes = geo["nodes0"] + off
    act_idx = np.flatnonzero(act > 0.5)
    members = geo["members"]
    # lengths of ALL ground members (mass / description of inactive ones)
    vec_all = nodes[members[:, 1]] - nodes[members[:, 0]]
    L_all = np.maximum(np.sqrt(np.einsum("ij,ij->i", vec_all, vec_all)), 1e-6)
    out = {"act_idx": act_idx, "L_all": L_all, "nodes": nodes}
    rr = r[act_idx]
    mt = mat[act_idx].astype(np.int64)
    A = A_COEF * rr * rr
    out["mass"] = float(np.sum(RHO_TAB[mt] * A * L_all[act_idx]))
    if act_idx.size == 0:
        out["mech"] = True
        return out
    mem = members[act_idx]
    L = L_all[act_idx]
    c = vec_all[act_idx] / L[:, None]
    E = E_TAB[mt]
    k = E * A / L
    nf = geo["n_free"]
    na = act_idx.size
    fmap = geo["fmap"]
    B = np.zeros((na, nf), dtype=float)
    rows = np.repeat(np.arange(na), 3)
    di = (3 * mem[:, 0:1] + np.arange(3)).reshape(-1)
    dj = (3 * mem[:, 1:2] + np.arange(3)).reshape(-1)
    cv = c.reshape(-1)
    fi, fj = fmap[di], fmap[dj]
    mj = fj >= 0
    B[rows[mj], fj[mj]] += cv[mj]
    mi = fi >= 0
    B[rows[mi], fi[mi]] -= cv[mi]
    K = B.T @ (k[:, None] * B)
    dK = np.diag(K).copy()
    out.update({"L": L, "c": c, "A": A, "I": I_COEF * rr ** 4, "E": E,
                "sy": SY_TAB[mt], "k": k, "B": B, "mt": mt})
    if nf == 0 or np.any(dK <= 0.0) or not np.all(np.isfinite(K)):
        out["mech"] = True
        return out
    try:
        Lc = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        out["mech"] = True
        return out
    piv = np.diag(Lc) ** 2 / dK
    if not np.all(np.isfinite(piv)) or float(piv.min()) < PIVOT_TOL:
        out["mech"] = True
        return out
    U = cho_solve((Lc, True), geo["rhs"])
    if not np.all(np.isfinite(U)):
        out["mech"] = True
        return out
    out["mech"] = False
    out["Lc"] = Lc
    out["U"] = U
    return out


def _response(geo: dict, sol: dict, scale: float, rot: float):
    """Free-DOF displacement and member forces for one load realisation."""
    U = sol["U"]
    uf = scale * (math.cos(rot) * U[:, 0] + math.sin(rot) * U[:, 1] + U[:, 2])
    N = sol["k"] * (sol["B"] @ uf)
    u = np.zeros(3 * geo["n_node"], dtype=float)
    u[geo["free"]] = uf
    return uf, u.reshape(-1, 3), N


def _member_fos(sol: dict, N: np.ndarray):
    absN = np.abs(N)
    fos_y = sol["sy"] / np.maximum(absN / np.maximum(sol["A"], 1e-15), 1e-9)
    comp = N < -1e-9
    p_cr = (math.pi ** 2) * sol["E"] * sol["I"] / (sol["L"] ** 2)
    fos_b = np.where(comp, p_cr / np.maximum(absN, 1e-12), np.inf)
    return np.minimum(fos_y, FOS_CAP), np.minimum(fos_b, FOS_CAP), comp


def _failure_metrics(geo: dict, mass: float, n_active: int) -> dict:
    return {
        "mass": float(mass) if math.isfinite(mass) else 0.0,
        "fos_buckling": 0.0,
        "fos_yielding": 0.0,
        "fos_shortfall": float(max(n_active, 1)),
        "max_deflection_m": float(10.0 * geo["span"]),
        "mechanism": 1.0,
    }


def _shortfall(fos_y: np.ndarray, fos_b: np.ndarray, comp: np.ndarray) -> float:
    """Sum over members of the relative shortfall of the governing FOS below
    the target: 0 exactly when every member passes both checks."""
    gov = np.where(comp, np.minimum(fos_y, fos_b), fos_y)
    return float(np.sum(np.maximum(0.0, 1.0 - gov / FOS_TARGET)))


class Frame3D2Domain(Domain):
    """3D truss with sizing + material + topology + shape moves
    (variants: frame3d2, frame3d2_stoch)."""
    # stochastic variants: phi_select compares candidates with a fresh
    # incumbent estimate (dataxp/ladder.py); no effect when deterministic
    rescore_incumbent = True

    name = "frame3d2"
    params = PARAMS
    factors = R_FACTORS
    reference_class = "heuristic"
    max_candidates = 160

    # ── verb surface (docstring "Verb surface"; docs/dataxp_verb_surface.md) ──
    # Spelling only: every SET_PARAM move the table above enumerates has one
    # verb text and parses back to the same move. format_move / parse_move /
    # grammar are the table-driven defaults of dataxp.domain.
    verb_surface = True
    verb_table = (
        ParamVerb("active", "INSTALL_MEMBER", _MEMBER_ARG,
                  Toggle("REMOVE_MEMBER",
                         off_doc="remove an installed member: it then carries no stiffness "
                                 "and no mass (offered only where the truss stays rigid).",
                         off_phrase="Remove a member"),
                  doc="install an optional (or removed) member, with its listed radius and "
                      "material.",
                  phrase="Install a member"),
        ParamVerb("mat", "SET_MATERIAL", _MEMBER_ARG,
                  Categorical("material", MAT_NAMES,
                              aliases=(("Al", 0), ("St", 1), ("Ti", 2)), doc=_MAT_DOC),
                  doc="set a member's material; its radius is kept.",
                  phrase="Set a member's material"),
        ParamVerb("r", "SET_RADIUS", _MEMBER_ARG,
                  Continuous("radius", "m", "%g..%g, the new outer tube radius (a value "
                                            "outside is set to the bound)" % (R_MIN, R_MAX)),
                  doc="resize a member's tube.",
                  phrase="Resize a member's tube"),
        ParamVerb("dx", "MOVE_JOINT", _JOINT_ARG, _OFFSET, select=("axis", "x"), doc=_JOINT_DOC,
                  phrase="Move a joint along x"),
        ParamVerb("dy", "MOVE_JOINT", _JOINT_ARG, _OFFSET, select=("axis", "y"), doc=_JOINT_DOC,
                  phrase="Move a joint along y"),
        ParamVerb("dz", "MOVE_JOINT", _JOINT_ARG, _OFFSET, select=("axis", "z"), doc=_JOINT_DOC,
                  phrase="Move a joint along z"),
    )

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # ── problems ──────────────────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        n_bay = int(_BAY_LADDER[(index // len(_FAMILIES)) % len(_BAY_LADDER)])
        if family == "bridge":
            n_bay = max(n_bay, 4)
        width = float(rng.uniform(0.8, 1.6))
        pitch = float(rng.uniform(0.9, 1.8))
        load = float(rng.uniform(4.0e3, 2.6e4))
        taper = float(rng.uniform(0.55, 1.0))
        ecc = float(rng.uniform(0.0, 0.6))
        scenario = "point_load" if rng.random() < 0.5 else "undersized"
        heavy = bool(rng.random() < 0.6)
        cap_frac = float(rng.uniform(HEAVY_CAP[0], HEAVY_CAP[1]))
        base = {
            "problem_id": "frame3d2_%s_%04d" % (family, index),
            "family": family,
            "n_bay": n_bay,
            "width": round(width, 4),
            "pitch": round(pitch, 4),
            "taper": round(taper, 4),
            "ecc": round(ecc, 4),
            "load_n": round(load, 2),
            "sound_frac": 1.04,
            "scenario": scenario,
            "load_class": "heavy" if heavy else "light",
        }
        geo0 = _ground(base, with_extra=False)
        nb = geo0["n_base"]
        zeros = np.zeros(nb, dtype=np.int64)
        r_fsd, _ = self._fsd(geo0, zeros, select=False)
        if heavy:
            # Scale the load until the aluminium FSD sized WITHOUT the
            # tube-size cap needs a largest tube of cap_frac (> 1) x the
            # largest radius made. The few most loaded members then cannot be
            # made strong enough in aluminium by radius alone: that capacity
            # has to come from material, bracing or geometry. (With the
            # largest tube at 0.85-0.97 of the cap, radius-only depth-1
            # selection still solved 16/18 heavy problems.) The start design
            # is the capped FSD, so those members start failing.
            target = cap_frac * R_MAX
            c = 1.0
            r_unc, _ = self._fsd(geo0, zeros, select=False, r_cap=10.0 * R_MAX)
            for _ in range(8):
                ratio = target / max(float(r_unc.max()), 1e-9)
                if abs(ratio - 1.0) < 0.02:
                    break
                c *= ratio ** 3
                base["load_n"] = round(load * c, 2)
                geo0 = _ground(base, with_extra=False)
                r_unc, _ = self._fsd(geo0, zeros, select=False, r_cap=10.0 * R_MAX)
            r_fsd, _ = self._fsd(geo0, zeros, select=False)
            load = float(base["load_n"])
        sol0 = self._base_solve(geo0, r_fsd, zeros)
        absN = np.zeros(nb)
        if not sol0["mech"]:
            _, _, N = _response(geo0, sol0, 1.0, 0.0)
            absN[sol0["act_idx"]] = np.abs(N)
        live = np.flatnonzero(absN > 0.01 * max(float(absN.max()), 1e-9))
        k = geo0["k"]
        spec = None
        # Seeded acceptance: redraw the damage / extra load / limit until the
        # start is infeasible but repairable in a turn budget (at most 10
        # failing members, deflection at most 1.35 x the limit).
        for attempt in range(10):
            spec = dict(base)
            spec["goals"] = {
                "minimum_fos_buckling": FOS_TARGET,
                "minimum_fos_yielding": FOS_TARGET,
                "maximum_fos_shortfall": 0.0,
            }
            # each retry is gentler (milder damage, looser limit, smaller
            # extra load)
            spec["start_frac"] = round(float(rng.uniform(0.45 + 0.03 * attempt, 0.8)), 4)
            spec["defl_factor"] = round(float(rng.uniform(0.70 + 0.02 * attempt, 1.0)), 4)
            if scenario == "point_load":
                # an extra concentrated load the start design was never sized
                # for (equipment / cable attachment at an interior joint)
                hi_s = max(1, n_bay - 1)
                pool = [n for n in geo0["joints"] if 1 <= n // k <= hi_s]
                node = int(pool[int(rng.integers(len(pool)))])
                mag = float(rng.uniform(0.1, 0.3)) * load * 0.7 ** attempt
                ang = float(rng.uniform(0.0, 2.0 * math.pi))
                if family == "tower":
                    f = (mag * math.cos(ang), mag * math.sin(ang), 0.0)
                else:
                    f = (0.0, mag * math.cos(ang), -abs(mag * math.sin(ang)))
                spec["extra_load"] = {"node": node,
                                      "f": [round(v, 2) for v in f]}
                n_deg = int(rng.integers(1, 4)) - (1 if heavy else 0)
            else:
                n_deg = int(rng.integers(4, 8)) - (2 if heavy else 0)
            spec["_fsd_radii"] = [round(float(v), 6) for v in r_fsd]
            # degrade force-carrying base members, staggered severities (the
            # heavy class degrades fewer: its over-cap members fail already)
            n_deg = max(0, min(n_deg, live.size))
            idx = sorted(int(v) for v in rng.choice(live, size=n_deg, replace=False))
            sev = np.linspace(float(spec["start_frac"]), 0.92, n_deg)
            rng.shuffle(sev)
            spec["degraded"] = idx
            spec["degraded_frac"] = [round(float(v), 4) for v in sev]
            # extras' default radius: median FSD radius of the loaded base
            spec["extra_r"] = _round_r(float(np.median(r_fsd[live])) if live.size
                                       else 0.01)
            # deflection limit relative to the sound (4% over-built) FSD
            # design under the full loading
            geo = _ground(spec)
            r_sound = np.array([_round_r(v * spec["sound_frac"]) for v in r_fsd])
            d_sound = self._defl(geo, self._base_solve(geo, r_sound, zeros))
            spec["defl_sound_m"] = round(d_sound, 7)
            spec["goals"]["maximum_max_deflection_m"] = round(
                d_sound * float(spec["defl_factor"]), 7)
            st = self.load(spec)
            nom = self._nominal(st)
            if nom["mech"]:
                continue
            gov = np.where(nom["tiny"], np.inf, np.nan_to_num(nom["gov"], nan=np.inf))
            n_fail = int(np.sum(gov < FOS_TARGET))
            ratio = nom["defl"] / spec["goals"]["maximum_max_deflection_m"]
            infeasible = n_fail > 0 or ratio > 1.0
            if infeasible and n_fail <= 10 and ratio <= 1.35:
                break
        spec["accept_attempts"] = attempt + 1
        self.reference(spec)
        return spec

    # ── helpers on the base topology (reference / generator) ─────────────────
    def _base_solve(self, geo, r_base, mat_base):
        M = geo["members"].shape[0]
        nb = geo["n_base"]
        r = np.full(M, R_MIN)
        r[:nb] = r_base
        act = np.zeros(M)
        act[:nb] = 1.0
        mat = np.zeros(M, dtype=np.int64)
        mat[:nb] = mat_base
        return _analyse(geo, r, act, mat, np.zeros((geo["n_node"], 3)))

    @staticmethod
    def _defl(geo, sol) -> float:
        if sol["mech"]:
            return float(10.0 * geo["span"])
        _, u, _ = _response(geo, sol, 1.0, 0.0)
        return float(np.sqrt((u * u).sum(axis=1)).max())

    def _required(self, geo, sol, N, mats, r_cap: float = R_MAX):
        """Per base member, per material: radius meeting both FOS targets,
        clipped to [R_MIN, r_cap]. Design rules of the FSD: every member is
        sized for at least a notional NOTIONAL x (largest member force),
        taken as compression (bracing members are designed for a notional
        force, not for zero), and for the minimum-size (slenderness) rule."""
        L = sol["L"]
        absN = np.abs(N)
        nmin = NOTIONAL * float(absN.max(initial=0.0))
        eff = np.maximum(absN, nmin)
        compress = (N < -1e-9) | (absN < nmin)
        out = np.zeros((N.size, N_MAT))
        for q in range(N_MAT):
            r_y = np.sqrt(FOS_TARGET * eff / SY_TAB[q] / A_COEF)
            i_req = FOS_TARGET * eff * L ** 2 / (math.pi ** 2 * E_TAB[q])
            r_b = np.where(compress, (i_req / I_COEF) ** 0.25, 0.0)
            out[:, q] = np.clip(np.maximum(np.maximum(r_y, r_b),
                                           self._floor(sol["L"])), R_MIN, r_cap)
        return out

    @staticmethod
    def _floor(L: np.ndarray) -> np.ndarray:
        """FSD minimum-size rule: slenderness L / r_gyration <= 250, with
        r_gyration = 0.64 r for this tube, i.e. r >= L / 160."""
        return np.clip(np.asarray(L, dtype=float) / SLENDER_DIV, R_MIN, R_MAX)

    def _fsd(self, geo, mats, select: bool, iters: int = 30, repair: int = 40,
             r_cap: float = R_MAX):
        """Fully stressed design of the base topology (radii capped at r_cap;
        r_cap > R_MAX gives the uncapped sizing the heavy class is scaled
        by). select=True picks per member the material needing the least
        mass among those whose tube fits under R_MAX (if none fits, the one
        needing the smallest tube)."""
        nb = geo["n_base"]
        r = np.full(nb, 0.02)
        mats = np.array(mats, dtype=np.int64)
        for it in range(iters):
            sol = self._base_solve(geo, r, mats)
            if sol["mech"]:
                r = np.minimum(r * 1.5, r_cap)
                continue
            _, _, N = _response(geo, sol, 1.0, 0.0)
            req = self._required(geo, sol, N, mats, r_cap=r_cap)
            if select and it < iters - 8:
                # judge each material on its UNCAPPED requirement: a capped
                # one looked cheapest exactly when it could not carry the load
                unc = self._required(geo, sol, N, mats, r_cap=np.inf)
                over = unc > R_MAX * (1.0 + 1e-9)
                mass_q = np.where(over, np.inf,
                                  RHO_TAB[None, :] * unc ** 2 * sol["L"][:, None])
                pick = np.argmin(mass_q + 1e-12 * np.arange(N_MAT)[None, :], axis=1)
                none = np.all(over, axis=1)
                pick[none] = np.argmin(unc[none], axis=1)
                mats = pick.astype(np.int64)
            r = np.clip(0.5 * r + 0.5 * req[np.arange(nb), mats], R_MIN, r_cap)
        for _ in range(repair):
            sol = self._base_solve(geo, r, mats)
            if sol["mech"]:
                r = np.minimum(r * 1.5, r_cap)
                continue
            _, _, N = _response(geo, sol, 1.0, 0.0)
            req = self._required(geo, sol, N, mats, r_cap=r_cap)[np.arange(nb), mats]
            if np.all(req <= r + 1e-12):
                break
            r = np.clip(np.maximum(r, req * 1.0005), R_MIN, r_cap)
        return r, mats

    def _scaled_mass(self, geo, r, mats, limit):
        """Scale all radii by one factor until the deflection limit is met;
        returns (mass, feasible) of the scaled design under the full
        loading (feasible: every FOS check and the limit met)."""
        sol = self._base_solve(geo, r, mats)
        d = self._defl(geo, sol)
        s = 1.0
        if d > limit > 0:
            s = math.sqrt(d / limit) * 1.0005
        r2 = np.clip(r * s, R_MIN, R_MAX)
        sol2 = self._base_solve(geo, r2, mats)
        L = sol2["L_all"][:geo["n_base"]]
        mass = float(np.sum(RHO_TAB[mats] * A_COEF * r2 * r2 * L))
        ok = False
        if not sol2["mech"]:
            _, u, N = _response(geo, sol2, 1.0, 0.0)
            fy, fb, comp = _member_fos(sol2, N)
            defl = float(np.sqrt((u * u).sum(axis=1)).max())
            ok = _shortfall(fy, fb, comp) <= 0.0 and defl <= limit * (1.0 + 1e-9)
        return mass, ok

    # ── state ─────────────────────────────────────────────────────────────────
    def load(self, spec: dict) -> dict:
        geo = _ground(spec)
        M = geo["members"].shape[0]
        nb = geo["n_base"]
        r = np.full(M, float(spec.get("extra_r", 0.01)))
        fsd = spec.get("_fsd_radii")
        if fsd is None or len(fsd) != nb:
            # same loading and rounding as make_problem (no extra point load)
            fsd, _ = self._fsd(_ground(spec, with_extra=False),
                               np.zeros(nb, dtype=np.int64), select=False)
            fsd = [round(float(v), 6) for v in fsd]
        fsd = np.asarray(fsd, dtype=float)
        r[:nb] = fsd * float(spec.get("sound_frac", 1.04))
        for i, f in zip(spec.get("degraded") or [], spec.get("degraded_frac") or []):
            if 0 <= int(i) < nb:
                r[int(i)] = fsd[int(i)] * float(f)
        r = np.array([_round_r(v) for v in r])
        act = np.zeros(M)
        act[:nb] = 1.0
        return {
            "spec": spec,
            "_geo": geo,
            "r": r,
            "act": act,
            "mat": np.zeros(M, dtype=np.int64),
            "off": np.zeros((geo["n_node"], 3)),
            "_sh": {},
        }

    def clone(self, st) -> dict:
        return {
            "spec": st["spec"],
            "_geo": st["_geo"],
            "r": st["r"].copy(),
            "act": st["act"].copy(),
            "mat": st["mat"].copy(),
            "off": st["off"].copy(),
            # analysis cache of THIS design, shared until a set() replaces it
            "_sh": st["_sh"],
        }

    def _sol(self, st) -> dict:
        sh = st["_sh"]
        sol = sh.get("sol")
        if sol is None:
            sol = _analyse(st["_geo"], st["r"], st["act"], st["mat"], st["off"])
            sh["sol"] = sol
        return sol

    # ── simulation ────────────────────────────────────────────────────────────
    def evaluate(self, st, seed: int = 0) -> dict:
        geo = st["_geo"]
        scale, rot = 1.0, 0.0
        if self.stochastic:
            rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
            sigma = LOAD_SIGMA
            scale = float(rng.lognormal(0.0, sigma) / math.exp(0.5 * sigma * sigma))
            rot = float(rng.normal(0.0, LOAD_ROT_SD))
        else:
            cached = st["_sh"].get("metrics")
            if cached is not None:
                return dict(cached)
        try:
            sol = self._sol(st)
            n_act = int(np.sum(st["act"] > 0.5))
            if sol["mech"]:
                metrics = _failure_metrics(geo, sol.get("mass", 0.0), n_act)
            else:
                _, u, N = _response(geo, sol, scale, rot)
                fos_y, fos_b, comp = _member_fos(sol, N)
                defl = float(np.sqrt((u * u).sum(axis=1)).max())
                metrics = {
                    "mass": float(sol["mass"]),
                    "fos_buckling": float(fos_b.min()) if np.any(comp) else FOS_CAP,
                    "fos_yielding": float(fos_y.min()),
                    "fos_shortfall": _shortfall(fos_y, fos_b, comp),
                    "max_deflection_m": defl,
                    "mechanism": 0.0,
                }
                if not all(math.isfinite(v) for v in metrics.values()):
                    metrics = _failure_metrics(geo, sol.get("mass", 0.0), n_act)
        except Exception:
            metrics = _failure_metrics(geo, 0.0, 1)
        if not self.stochastic:
            st["_sh"]["metrics"] = dict(metrics)
        return metrics

    # ── design variables ──────────────────────────────────────────────────────
    def _n_mem(self, st) -> int:
        return int(st["_geo"]["members"].shape[0])

    def n_elements(self, st) -> int:
        return self._n_mem(st) + len(st["_geo"]["joints"])

    def _joint_node(self, st, i: int) -> Optional[int]:
        M = self._n_mem(st)
        j = int(i) - M
        joints = st["_geo"]["joints"]
        if 0 <= j < len(joints):
            return int(joints[j])
        return None

    def get(self, st, i: int, p: str) -> Optional[float]:
        i = int(i)
        M = self._n_mem(st)
        if 0 <= i < M:
            if p == "r":
                return float(st["r"][i])
            if p == "active":
                return float(st["act"][i])
            if p == "mat":
                return float(st["mat"][i])
            return None
        nd = self._joint_node(st, i)
        if nd is not None and p in JOINT_PARAMS:
            ax = JOINT_PARAMS.index(p)
            if st["_geo"]["locked"][nd, ax]:
                return None             # loaded joint: axial position fixed
            return float(st["off"][nd, ax])
        return None

    def _legal(self, st, i: int, p: str, v: float) -> float:
        v = float(v)
        if not math.isfinite(v):
            v = 0.0
        if p == "r":
            return _round_r(v)
        if p == "active":
            return 1.0 if v >= 0.5 else 0.0
        if p == "mat":
            return float(int(round(min(N_MAT - 1, max(0.0, v)))))
        b = st["_geo"]["off_bound"]
        return round(min(b, max(-b, v)), 4)

    def set(self, st, i: int, p: str, v: float) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return                      # a non-number changes nothing
        i = int(i)
        M = self._n_mem(st)
        if 0 <= i < M and p in MEMBER_PARAMS:
            v = self._legal(st, i, p, v)
            if p == "r":
                st["r"][i] = v
            elif p == "active":
                st["act"][i] = v
            else:
                st["mat"][i] = int(v)
        else:
            nd = self._joint_node(st, i)
            if nd is None or p not in JOINT_PARAMS:
                return
            ax = JOINT_PARAMS.index(p)
            if st["_geo"]["locked"][nd, ax]:
                return                  # does not apply: nothing changes
            st["off"][nd, ax] = self._legal(st, i, p, v)
        st["_sh"] = {}

    def bounds(self, st, p: str) -> Tuple[float, float]:
        if p == "r":
            return (R_MIN, R_MAX)
        if p == "active":
            return (0.0, 1.0)
        if p == "mat":
            return (0.0, float(N_MAT - 1))
        b = float(st["_geo"]["off_bound"])
        return (-b, b)

    # ── program / reference ───────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        ref = self.reference(spec)
        prog = program_from_goals(
            spec.get("goals", {}),
            state_keys=("mass", "fos_buckling", "fos_yielding", "fos_shortfall",
                        "max_deflection_m", "mechanism"),
            objective_key="mass",
            objective_ref=ref,
            objective_sense="min",
            # the shortfall limit is 0: measure it in its own units (one
            # member fully at zero FOS = 1), not relative to the limit
            scales={"fos_shortfall": 1.0},
            domain=self.name,
        )
        geo = _ground(spec)
        return dataclasses.replace(prog, validity_bounds=(
            ("max_deflection_m", 0.1 * geo["span"]),
            ("mechanism", 0.5),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        cached = spec.get("_ref_mass")
        if cached is not None:
            try:
                c = float(cached)
                if math.isfinite(c) and c > 0.0:
                    return c
            except (TypeError, ValueError):
                pass
        geo = _ground(spec)
        nb = geo["n_base"]
        limit = float(spec.get("goals", {}).get("maximum_max_deflection_m", 0.0))
        best = None
        for select in (False, True):
            r, mats = self._fsd(geo, np.zeros(nb, dtype=np.int64), select=select)
            mass, ok = self._scaled_mass(geo, r, mats, limit)
            # a design that meets every check beats any that does not (the
            # lighter-of-two rule alone quoted an aluminium design that missed
            # the deflection limit on 8/30 problems)
            key = (0 if ok else 1, mass)
            if math.isfinite(mass) and mass > 0 and (best is None or key < best[2]):
                best = (mass, select, key)
        mass = best[0] if best else 1.0
        spec["_ref_mass"] = mass
        spec["_ref_kind"] = (("FSD with material selection" if best and best[1]
                              else "aluminium FSD")
                             + ("" if best and best[2][0] == 0 else " (infeasible)"))
        return mass

    # ── nominal analysis (hints, typed moves, observation) ───────────────────
    def _nominal(self, st) -> dict:
        """Per-ground-member forces and checks at the nominal load, the
        largest-deflection node with each member's share of it (unit-load
        virtual work), and each member's redundancy number (1 - k b'K^-1 b; 0
        means removing the member leaves a mechanism). Read-only on the
        design; cached with the design's analysis."""
        sh = st["_sh"]
        nom = sh.get("nom")
        if nom is not None:
            return nom
        geo = st["_geo"]
        sol = self._sol(st)
        M = self._n_mem(st)
        nom = {"mech": bool(sol["mech"]), "L_all": sol["L_all"],
               "nodes": sol["nodes"], "mass": sol.get("mass", 0.0)}
        if not sol["mech"]:
            act_idx = sol["act_idx"]
            uf, u, N = _response(geo, sol, 1.0, 0.0)
            fos_y, fos_b, comp = _member_fos(sol, N)

            def full(fill):
                return np.full(M, fill, dtype=float)

            N_all, fy_all, fb_all = full(np.nan), full(np.nan), full(np.nan)
            N_all[act_idx] = N
            fy_all[act_idx] = fos_y
            fb_all[act_idx] = np.where(comp, fos_b, np.inf)
            gov = np.minimum(fy_all, fb_all)
            absN = np.abs(N)
            tiny_a = absN <= max(1e-3, 1e-9 * float(absN.max(initial=0.0)))
            tiny = np.zeros(M, dtype=bool)
            tiny[act_idx] = tiny_a
            util = np.where(np.isfinite(gov),
                            FOS_TARGET / np.maximum(np.nan_to_num(gov, nan=1.0), 1e-12),
                            np.nan)
            util[tiny] = 0.0
            dn = np.sqrt((u * u).sum(axis=1))
            dnode = int(np.argmax(dn))
            dval = float(dn[dnode])
            share = full(np.nan)
            if dval > 0.0:
                q = np.zeros(3 * geo["n_node"])
                q[3 * dnode:3 * dnode + 3] = u[dnode] / dval
                v = cho_solve((sol["Lc"], True), q[geo["free"]])
                n_v = sol["k"] * (sol["B"] @ v)
                share[act_idx] = (N * n_v / sol["k"]) / dval
            S = cho_solve((sol["Lc"], True), sol["B"].T)
            s = sol["k"] * np.einsum("ij,ji->i", sol["B"], S)
            red = full(np.nan)
            red[act_idx] = np.clip(1.0 - s, 0.0, 1.0)
            comp_all = np.zeros(M, dtype=bool)
            comp_all[act_idx] = comp & ~tiny_a
            nom.update({"N": N_all, "fos_y": fy_all, "fos_b": fb_all,
                        "gov": gov, "util": util, "tiny": tiny, "u": u,
                        "dn": dn, "dnode": dnode, "defl": dval,
                        "share": share, "red": red, "comp": comp_all})
        sh["nom"] = nom
        return nom

    def _limits(self, st) -> dict:
        g = st["spec"].get("goals", {})
        return {"fos_b": float(g.get("minimum_fos_buckling", FOS_TARGET)),
                "fos_y": float(g.get("minimum_fos_yielding", FOS_TARGET)),
                "defl": float(g.get("maximum_max_deflection_m", float("inf")))}

    # ── typed move set ────────────────────────────────────────────────────────
    def candidate_moves(self, st, rng: np.random.Generator) -> List[Candidate]:
        """Stratified typed moves, every stratum present every turn:
        sizing (r), material (mat), topology (active on/off) and shape
        (joint dx/dy/dz). Each stratum takes the members/joints the current
        analysis points at (lowest FOS, largest deflection share, lowest
        utilisation) plus a seeded random sample. Deactivation is offered only
        for members whose removal keeps the truss rigid (redundancy > 0.05)."""
        geo = st["_geo"]
        M = self._n_mem(st)
        nom = self._nominal(st)
        act = st["act"] > 0.5
        out: List[Candidate] = []
        seen = set()

        def add(i, p, v):
            cur = self.get(st, i, p)
            if cur is None:
                return
            v = self._legal(st, i, p, v)
            if abs(v - cur) <= 1e-12 * max(1.0, abs(cur)):
                return
            key = (int(i), p, v)
            if key in seen:
                return
            seen.add(key)
            out.append(Candidate(i=int(i), p=p, v=float(v), v_old=float(cur)))

        def pick(pool, n):
            pool = [int(x) for x in pool]
            if len(pool) <= n:
                return pool
            sel = rng.choice(len(pool), size=n, replace=False)
            return [pool[int(s)] for s in sorted(sel.tolist())]

        act_ids = [int(m) for m in np.flatnonzero(act)]
        inact_ids = [int(m) for m in np.flatnonzero(~act)]
        joint_el = {nd: M + j for j, nd in enumerate(geo["joints"])}
        step = float(geo["off_step"])
        locked = geo["locked"]

        def rand_axis(nd):
            free_ax = [ax for ax in range(3) if not locked[nd, ax]]
            return int(free_ax[int(rng.integers(len(free_ax)))])

        if nom["mech"]:
            # repair stratum first: put members back
            for m in pick([m for m in inact_ids if m < geo["n_base"]], 24):
                add(m, "active", 1.0)
            for m in pick(inact_ids, 12):
                add(m, "active", 1.0)
            for m in pick(act_ids, 12):
                add(m, "r", st["r"][m] * float(rng.choice(R_FACTORS)))
                add(m, "mat", float((int(st["mat"][m]) + 1 + int(rng.integers(2))) % N_MAT))
            for nd in pick(list(joint_el), 6):
                ax = rand_axis(nd)
                add(joint_el[nd], JOINT_PARAMS[ax],
                    st["off"][nd, ax] + step * (1 if rng.random() < 0.5 else -1))
            order = rng.permutation(len(out))
            return [out[int(k)] for k in order]

        util = nom["util"]
        lim = self._limits(st)
        by_util = sorted(act_ids, key=lambda m: (-np.nan_to_num(util[m], nan=-1.0), m))
        crit = by_util[:8]
        dviol = nom["defl"] > 0.9 * lim["defl"]
        share = nom["share"]
        dcrit = []
        if dviol:
            dcrit = [m for m in sorted(act_ids, key=lambda m: (-np.nan_to_num(share[m]), m))
                     if m not in crit[:4]][:5]
        low = [m for m in sorted(act_ids, key=lambda m: (np.nan_to_num(util[m]), m))
               if st["r"][m] > R_MIN * 1.0001 or st["mat"][m] != 0][:6]

        # sizing: steps, plus the radius that meets FOS 1.5 at the current
        # force (Euler: I ~ r^4; yielding: A ~ r^2) for failing members and
        # for the least-utilised ones (a resize down)
        def resize(m, margin):
            g = float(nom["gov"][m])
            if not math.isfinite(g) or g <= 0.0 or nom["tiny"][m]:
                return None
            buck = bool(nom["comp"][m]) and nom["fos_b"][m] <= nom["fos_y"][m]
            expo = 0.25 if buck else 0.5
            return st["r"][m] * (FOS_TARGET * margin / g) ** expo

        for m in crit:
            for f in (1.1, 1.25):
                add(m, "r", st["r"][m] * f)
            if nom["gov"][m] < FOS_TARGET * (RESIZE_UP[True] if self.stochastic else 1.0):
                v = resize(m, RESIZE_UP[self.stochastic])
                if v is not None:
                    add(m, "r", v)
        for m in dcrit:
            for f in (1.1, 1.25):
                add(m, "r", st["r"][m] * f)
        for m in low:
            for f in (0.8, 0.9):
                add(m, "r", st["r"][m] * f)
            v = resize(m, RESIZE_DOWN[self.stochastic])
            if v is not None and v < st["r"][m]:
                add(m, "r", v)
        for m in pick(act_ids, 6):
            add(m, "r", st["r"][m] * float(rng.choice(R_FACTORS)))
        # material
        for m in crit[:6] + dcrit[:3] + low[:3]:
            for q in range(N_MAT):
                add(m, "mat", float(q))
        for m in pick(act_ids, 6):
            add(m, "mat", float((int(st["mat"][m]) + 1 + int(rng.integers(2))) % N_MAT))
        # topology: brace near the trouble, remove what carries little
        focus = []
        for m in crit[:4] + dcrit[:2]:
            focus += [int(geo["members"][m, 0]), int(geo["members"][m, 1])]
        focus.append(int(nom["dnode"]))
        near = []
        for nd in focus:
            for m in geo["inc"].get(nd, []):
                if not act[m] and m not in near:
                    near.append(m)
        near = near[:12]
        for m in near:
            add(m, "active", 1.0)
        for m in pick([m for m in inact_ids if m not in near], 6):
            add(m, "active", 1.0)
        red = nom["red"]
        safe = [m for m in act_ids if np.nan_to_num(red[m]) > SAFE_REDUNDANCY]
        safe_low = sorted(safe, key=lambda m: (np.nan_to_num(util[m]), m))[:8]
        for m in safe_low:
            add(m, "active", 0.0)
        for m in pick([m for m in safe if m not in safe_low], 4):
            add(m, "active", 0.0)
        # shape
        jfocus = []
        for nd in focus:
            if nd in joint_el and nd not in jfocus:
                jfocus.append(nd)
        for nd in jfocus[:6]:
            for ax in range(3):
                for sgn in (1.0, -1.0):
                    add(joint_el[nd], JOINT_PARAMS[ax], st["off"][nd, ax] + sgn * step)
        for nd in pick([n for n in joint_el if n not in jfocus[:6]], 4):
            ax = rand_axis(nd)
            add(joint_el[nd], JOINT_PARAMS[ax],
                st["off"][nd, ax] + step * (1.0 if rng.random() < 0.5 else -1.0))
        order = rng.permutation(len(out))
        return [out[int(k)] for k in order]

    # ── depth-0 feedback ──────────────────────────────────────────────────────
    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        """Lowest-FOS active member, radius up."""
        nom = self._nominal(st)
        if nom["mech"]:
            return None
        gov = np.where(nom["tiny"], np.inf, np.nan_to_num(nom["gov"], nan=np.inf))
        if not np.any(np.isfinite(gov)):
            return None
        return (int(np.argmin(gov)), "r", "up")

    def hint_move(self, st) -> Optional[Candidate]:
        """Greedy move. Strength violated (or nothing violated): None, so the
        greedy policy scales the lowest-FOS member up. Only the deflection
        limit violated: switch on the shortest inactive member at the
        largest-deflection node (a brace where the structure is softest), else
        scale up the member with the largest share of that deflection."""
        nom = self._nominal(st)
        geo = st["_geo"]
        if nom["mech"]:
            off = [m for m in range(geo["n_base"]) if st["act"][m] < 0.5]
            if off:
                return Candidate(i=off[0], p="active", v=1.0, v_old=0.0)
            return None
        lim = self._limits(st)
        fos_b = np.nan_to_num(nom["fos_b"], nan=np.inf)
        fos_y = np.nan_to_num(nom["fos_y"], nan=np.inf)
        if (fos_b.min() < lim["fos_b"] or fos_y.min() < lim["fos_y"]
                or nom["defl"] <= lim["defl"]):
            return None
        dn = int(nom["dnode"])
        off = [m for m in geo["inc"].get(dn, []) if st["act"][m] < 0.5]
        if off:
            m = min(off, key=lambda m: (nom["L_all"][m], m))
            return Candidate(i=int(m), p="active", v=1.0, v_old=0.0)
        share = np.nan_to_num(nom["share"], nan=-np.inf)
        m = int(np.argmax(share))
        cur = float(st["r"][m])
        v = _round_r(cur * max(R_FACTORS))
        if v == cur:
            return None
        return Candidate(i=m, p="r", v=v, v_old=cur)

    # ── observation: describe() / element_label() ─────────────────────────────
    # Read-only on the design: everything is taken from the nominal analysis
    # (scale 1, no load rotation), which is exactly what evaluate() computes
    # for the deterministic variant.

    def _topo(self, st) -> list:
        """(node_a, node_b, kind, place, where) per ground member."""
        geo = st["_geo"]
        fam = geo["family"]
        k = geo["k"]
        corner = _OBS_CORNER.get(fam, _OBS_CORNER["tower"])
        edge = _OBS_EDGE.get(fam, _OBS_EDGE["tower"])
        out = []
        for m in range(self._n_mem(st)):
            a, b = int(geo["members"][m, 0]), int(geo["members"][m, 1])
            if a > b:
                a, b = b, a
            kind = geo["kinds"][m]
            sa, ca, sb, cb = a // k, a % k, b // k, b % k
            key = (min(ca, cb), max(ca, cb))
            if kind in ("ring", "sdiag", "sdiag2"):
                place, where = "s%d" % sa, edge.get(key, "")
            elif kind in ("leg", "chord"):
                place, where = "b%d" % sb, corner[ca % len(corner)]
            elif kind == "diag":
                place, where = "b%d" % sb, edge.get(key, "")
            elif kind == "xdiag":
                place = "b%d" % sb
                where = "%s>%s" % (corner[ca % len(corner)], corner[cb % len(corner)])
            else:                                   # skip: two-bay brace
                place, where = "b%d-b%d" % (sa + 1, sb), edge.get(key, "")
            out.append((a, b, kind, place, where))
        return out

    def _obs_extra(self, st) -> dict:
        """What-if numbers for the observation, all from the nominal linear
        model of THIS design (read-only on the design; cached with its
        analysis, which set() replaces):
          inst[m]  optional member m installed alone at its listed r and mat,
                   exact rank-1 update of the factorised stiffness:
                   (own N, own governing FOS, own mode, fos_shortfall,
                   largest deflection, mass added);
          rem[m]   installed member m removed (red >= SAFE_REDUNDANCY), exact
                   rank-1 downdate: (fos_shortfall, largest deflection);
          jsens[nd] per axis: None (locked) or the central-difference change
                   of (fos_shortfall, largest deflection, mass) per +1 offset
                   step;
          mech_nodes  for a mechanism: the nodes the singular modes move."""
        sh = st["_sh"]
        ox = sh.get("obsx")
        if ox is not None:
            return ox
        geo = st["_geo"]
        sol = self._sol(st)
        ox = {"inst": {}, "rem": {}, "jsens": {}, "mech_nodes": []}
        if sol["mech"]:
            ox["mech_nodes"] = _mech_nodes(geo, sol)
            sh["obsx"] = ox
            return ox
        nom = self._nominal(st)
        uf, _, N = _response(geo, sol, 1.0, 0.0)
        Lc = sol["Lc"]
        kB = sol["k"][:, None] * sol["B"]
        act_idx = sol["act_idx"]
        props = (sol["sy"], sol["A"], sol["E"], sol["I"], sol["L"])

        # install: rank-1 update K + k b b'
        opt = np.flatnonzero(st["act"] < 0.5)
        if opt.size:
            nodes = sol["nodes"]
            mem = geo["members"][opt]
            vec = nodes[mem[:, 1]] - nodes[mem[:, 0]]
            L = np.maximum(np.sqrt(np.einsum("ij,ij->i", vec, vec)), 1e-6)
            c = vec / L[:, None]
            fmap = geo["fmap"]
            Bo = np.zeros((opt.size, geo["n_free"]))
            rows = np.arange(opt.size)
            for d in range(3):
                fj = fmap[3 * mem[:, 1] + d]
                fi = fmap[3 * mem[:, 0] + d]
                mj, mi = fj >= 0, fi >= 0
                Bo[rows[mj], fj[mj]] += c[mj, d]
                Bo[rows[mi], fi[mi]] -= c[mi, d]
            q = st["mat"][opt].astype(np.int64)
            ro = st["r"][opt]
            A = A_COEF * ro * ro
            E = E_TAB[q]
            k = E * A / L
            W = cho_solve((Lc, True), Bo.T)
            s = np.einsum("ij,ji->i", Bo, W)
            Ne = k * (Bo @ uf) / (1.0 + k * s)
            N2 = N[:, None] - (kB @ W) * Ne[None, :]
            sf = _shortfall_cols(props, N2)
            own = (SY_TAB[q], A, E, I_COEF * ro ** 4, L)
            fy_o, fb_o, comp_o = (v[:, 0] for v in _fos_cols(own, Ne[:, None]))
            gov_o = np.where(comp_o, np.minimum(fy_o, fb_o), fy_o)
            sf = sf + np.maximum(0.0, 1.0 - gov_o / FOS_TARGET)
            d2 = _defl_cols(geo, uf[:, None] - W * Ne[None, :])
            for j, m in enumerate(opt):
                buck = bool(comp_o[j]) and fb_o[j] <= fy_o[j]
                ox["inst"][int(m)] = (float(Ne[j]), float(gov_o[j]),
                                      "b" if buck else "y", float(sf[j]),
                                      float(d2[j]),
                                      float(RHO_TAB[q[j]] * A[j] * L[j]))

        # removal: rank-1 downdate K - k b b' (N_j += C[j, e] N_e / red_e)
        red = np.nan_to_num(nom["red"][act_idx], nan=0.0)
        cols = np.flatnonzero(red >= SAFE_REDUNDANCY)
        if cols.size:
            S = cho_solve((Lc, True), sol["B"].T[:, cols])
            C = kB @ S
            t = N[cols] / red[cols]
            N2 = N[:, None] + C * t[None, :]
            keep = np.ones_like(N2)
            keep[cols, np.arange(cols.size)] = 0.0
            sf = _shortfall_cols(props, N2, keep)
            d2 = _defl_cols(geo, uf[:, None] + S * t[None, :])
            for j, e in enumerate(cols):
                ox["rem"][int(act_idx[e])] = (float(sf[j]), float(d2[j]))

        # joint moves: central differences of the full analysis
        h = 1.0e-4
        step = float(geo["off_step"])
        locked = geo["locked"]
        for nd in geo["joints"]:
            row = []
            for ax in range(3):
                if locked[nd, ax]:
                    row.append(None)
                    continue
                vals = []
                for sgn in (1.0, -1.0):
                    off = st["off"].copy()
                    off[nd, ax] += sgn * h
                    s2 = _analyse(geo, st["r"], st["act"], st["mat"], off)
                    if s2["mech"]:
                        break
                    _, u2, n2 = _response(geo, s2, 1.0, 0.0)
                    fy, fb, comp = _member_fos(s2, n2)
                    vals.append((_shortfall(fy, fb, comp),
                                 float(np.sqrt((u2 * u2).sum(axis=1)).max()),
                                 float(s2["mass"])))
                if len(vals) < 2:
                    row.append(None)
                    continue
                row.append(tuple((vals[0][i] - vals[1][i]) / (2.0 * h) * step
                                 for i in range(3)))
            ox["jsens"][int(nd)] = row
        sh["obsx"] = ox
        return ox

    def describe(self, st, spec=None) -> str:
        """Self-contained plain-text observation of the current design: the
        brief, the change syntax and the physics needed to plan a change, one
        row per installed member (design, result, redundancy, deflection
        share, removal outcome), each optional member's install outcome, the
        radius each material needs for the failing members, and per-joint
        displacement and first-order shape sensitivities. All from the
        nominal linear model; read-only on the design."""
        if spec is None:
            spec = st["spec"]
        geo = st["_geo"]
        fam = geo["family"]
        n_bay = geo["n_bay"]
        k = geo["k"]
        M = self._n_mem(st)
        joints = geo["joints"]
        jel = {nd: M + j for j, nd in enumerate(joints)}
        f = _obs_num
        nom = self._nominal(st)
        ox = self._obs_extra(st)
        topo = self._topo(st)
        nodes0 = geo["nodes0"]
        P = geo["P"]
        fixed = set(geo["fixed"])
        act = st["act"] > 0.5
        lim = self._limits(st)
        width, pitch = float(spec["width"]), float(spec["pitch"])
        taper = float(spec.get("taper", 1.0))
        ext = np.ptp(nodes0, axis=0)
        n_inst = int(act.sum())
        corner = _OBS_CORNER.get(fam, _OBS_CORNER["tower"])
        tgt = FOS_TARGET * (Q90_LOAD if self.stochastic else 1.0)
        tgt_s = f(tgt)
        ax_name = "xyz"[geo["axis"]]
        step = float(geo["off_step"])

        out = ["PROBLEM (3D truss design: member sizing, material, topology, joint "
               "positions)"]
        cdesc = ", ".join("c%d %s" % (c, corner[c]) for c in range(k))
        if fam == "tower":
            out.append("Lattice tower (pin-jointed space truss), %d bays of square "
                       "sections along +z, pitch %s m, height %s m, width tapering "
                       "linearly from %s m (s0) to %s m (s%d). %d nodes, node = 4*s + c "
                       "(section s, corner c: %s)."
                       % (n_bay, f(pitch, 4), f(ext[2], 4), f(width, 4),
                          f(width * taper, 4), n_bay, geo["n_node"], cdesc))
            out.append("Initial sections (z m, half-width m; corners at x, y = "
                       "+-half-width): %s."
                       % "; ".join("s%d %s, %s" % (s, f(nodes0[k * s, 2], 4),
                                                  f(abs(nodes0[k * s, 0]), 4))
                                   for s in range(n_bay + 1)))
            out.append("Supports: base section s0 (nodes %s) pinned."
                       % _obs_runs(sorted(fixed)))
        elif fam == "cantilever":
            out.append("Cantilever truss (pin-jointed space truss), %d bays of square "
                       "sections (y-z plane) along +x, pitch %s m, length %s m, size "
                       "tapering linearly from %s m (s0, root) to %s m (s%d, tip). %d "
                       "nodes, node = 4*s + c (section s, corner c: %s)."
                       % (n_bay, f(pitch, 4), f(ext[0], 4), f(width, 4),
                          f(width * taper, 4), n_bay, geo["n_node"], cdesc))
            out.append("Initial sections (x m, half-size m; corners at y, z = "
                       "+-half-size, bot = z-): %s."
                       % "; ".join("s%d %s, %s" % (s, f(nodes0[k * s, 0], 4),
                                                  f(abs(nodes0[k * s, 1]), 4))
                                   for s in range(n_bay + 1)))
            out.append("Supports: root section s0 (nodes %s) pinned."
                       % _obs_runs(sorted(fixed)))
        else:
            out.append("Bridge girder (pin-jointed space truss), %d bays of triangular "
                       "sections along +x, pitch %s m, span %s m; deck nodes at z = 0, "
                       "y = -+%s m, apex node at y = 0. %d nodes, node = 3*s + c "
                       "(section s, corner c: %s)."
                       % (n_bay, f(pitch, 4), f(ext[0], 4), f(0.5 * width, 4),
                          geo["n_node"], cdesc))
            out.append("Initial sections (x m, apex z m): %s."
                       % "; ".join("s%d %s, %s" % (s, f(nodes0[k * s, 0], 4),
                                                  f(nodes0[k * s + 2, 2], 4))
                                   for s in range(n_bay + 1)))
            out.append("Supports: end sections s0 and s%d (nodes %s) pinned."
                       % (n_bay, _obs_runs(sorted(fixed))))
        groups = {}
        for nd in range(geo["n_node"]):
            if np.any(P[nd] != 0.0):
                groups.setdefault(tuple(float(v) for v in P[nd]), []).append(nd)
        parts = [("node %s" if len(ns) == 1 else "nodes %s each")
                 % _obs_runs(ns) + " (%s)" % ", ".join(f(v / 1e3) for v in vec)
                 for vec, ns in groups.items()]
        tot = P.sum(axis=0) / 1e3
        out.append("Loads (Fx, Fy, Fz kN; fixed nodal forces that move with their "
                   "node): %s; resultant (%s)."
                   % ("; ".join(parts), ", ".join(f(v) for v in tot)))
        if self.stochastic:
            out.append("Load scatter: each evaluation scales all loads by a random "
                       "factor (mean 1, ~8%% sd) and turns them about z (sd 0.03 rad). "
                       "Requirements are judged at the adverse 90th percentile (about "
                       "%s x the nominal load); the results below are at the nominal "
                       "load, so aim there for every FOS >= about %s and a deflection "
                       "about %d%% below the limit."
                       % (f(Q90_LOAD), tgt_s, int(round(100.0 * (1.0 - 1.0 / Q90_LOAD)))))
        out.append("Elements: members m0-m%d (%d installed, the rest optional); free "
                   "joints J%d-J%d = nodes %s in order."
                   % (M - 1, n_inst, M, M + len(joints) - 1, _obs_runs(joints)))
        out.append("Member kinds: %s = same corner of consecutive sections; diag = "
                   "bay-face diagonal; ring = section edge%s%s. Bay b spans sections "
                   "b-1 to b. %s"
                   % ("leg" if fam == "tower" else "chord",
                      "; sdiag / sdiag2 = section diagonal c0-c2 / c1-c3; xdiag = body "
                      "diagonal from a corner of section b-1 to the opposite corner of "
                      "section b" if k == 4 else "",
                      "; skip = two-bay brace from a corner of section s to another "
                      "corner of section s+2" if k == 3 else "",
                      _V2_CODES.get(fam, "")[:1].upper() + _V2_CODES.get(fam, "")[1:]))

        out.append("")
        along = {"tower": "height", "cantilever": "reach", "bridge": "span"}.get(fam, "axis")
        if spelled_in_verbs(self):
            # the same facts as the SET_PARAM block below, in the verbs the
            # policy is served (docstring "Verb surface"); under the
            # set_parameter spelling the SET_PARAM block is served again
            # (docstring "Set-parameter spelling")
            out.append("CHANGES: one per turn, one of SET_RADIUS(m, radius), SET_MATERIAL(m, "
                       "material), INSTALL_MEMBER(m), REMOVE_MEMBER(m), MOVE_JOINT(J, axis, "
                       "offset); m and J are the bare numbers (m5 -> 5, J%d -> %d); a value "
                       "is the new setting, not a change." % (M, M))
            out.append("- SET_RADIUS(m, radius): member tube radius [m], %s-%s (the largest "
                       "tube made; values outside are set to the bound, so raising a member "
                       "at %s does nothing). Wall %s r, A = %s r^2, I = %s r^4."
                       % (f(R_MIN), f(R_MAX), f(R_MAX), f(WALL_RATIO), f(A_COEF, 4),
                          f(I_COEF, 4)))
            out.append("- SET_MATERIAL(m, material): keeps r (E GPa, yield MPa, density "
                       "kg/m3): %s."
                       % "; ".join("%s = %s, %s (%s, %s, %s)"
                                   % (MAT_NAMES[q], MATERIALS[q][1], _MAT_SHORT[q],
                                      f(E_TAB[q] / 1e9), f(SY_TAB[q] / 1e6), f(RHO_TAB[q], 4))
                                   for q in range(N_MAT)))
            out.append("- INSTALL_MEMBER(m) / REMOVE_MEMBER(m): installed = stiffness and "
                       "mass, removed = neither. An optional member goes in with its listed "
                       "r and mat; setting those first changes no result.")
            out.append("- MOVE_JOINT(J, axis, offset): joint offset from its initial position "
                       "[m] along axis x, y or z, the new total (current +- a step; typical "
                       "step %s m), within +-%s m; it moves every member at the joint. A "
                       "loaded joint's %s is locked (where a load acts along the %s is part "
                       "of the brief): moving it along %s does nothing."
                       % (f(step), f(geo["off_bound"]), ax_name, along, ax_name))
        else:
            out.append("CHANGES: one per turn, SET_PARAM(element, parameter, value); element "
                       "is the bare number (m5 -> 5, J%d -> %d); value is the new setting, "
                       "not a change." % (M, M))
            out.append("- r: member tube radius [m], %s-%s (the largest tube made; values "
                       "outside are set to the bound, so raising a member at %s does "
                       "nothing). Wall %s r, A = %s r^2, I = %s r^4."
                       % (f(R_MIN), f(R_MAX), f(R_MAX), f(WALL_RATIO), f(A_COEF, 4),
                          f(I_COEF, 4)))
            out.append("- mat: member material code, keeps r (E GPa, yield MPa, density "
                       "kg/m3): %s."
                       % "; ".join("%d = %s %s (%s, %s, %s)"
                                   % (q, MATERIALS[q][1], _MAT_SHORT[q], f(E_TAB[q] / 1e9),
                                      f(SY_TAB[q] / 1e6), f(RHO_TAB[q], 4))
                                   for q in range(N_MAT)))
            out.append("- active: 1 installed, 0 removed (no stiffness or mass). An optional "
                       "member goes in with its listed r and mat; setting those first "
                       "changes no result.")
            out.append("- dx, dy, dz: joint offset from its initial position [m], the new "
                       "total (current +- a step; typical step %s m), within +-%s m; it "
                       "moves every member at the joint. A loaded joint's %s is locked "
                       "(where a load acts along the %s is part of the brief): setting it "
                       "does nothing."
                       % (f(step), f(geo["off_bound"]), ax_name, along))
        out.append("REQUIREMENTS: every installed member FOS_yielding = yield/(|N|/A) "
                   ">= %s and, in compression, FOS_buckling = pi^2*E*I/(L^2*|N|) >= %s "
                   "(pin-ended Euler); fos_shortfall = sum of max(0, 1 - "
                   "FOS_governing/%s) = 0; largest nodal deflection <= %s mm. N > 0 "
                   "tension. Invalid: a mechanism or a deflection > %s m. Objective: "
                   "total mass (density*A*L over installed members)."
                   % (f(lim["fos_y"]), f(lim["fos_b"]), f(FOS_TARGET),
                      f(lim["defl"] * 1e3), f(0.1 * geo["span"])))
        out.append("HOW CHANGES ACT (linear model):")
        out.append("- At a fixed force FOS_yielding scales with r^2 and yield, "
                   "FOS_buckling with r^4, E and 1/L^2; mass with r^2 and L.")
        out.append("- Forces redistribute. Multiplying a member's EA/L by f (r x sqrt f, "
                   "or f x E by a material change) makes its own force exactly "
                   "N*f/(red + f*(1 - red)), red = its redundancy (0-1, listed): near 0 "
                   "the force stays, near 1 it grows with the stiffness; the gain comes "
                   "out of parallel paths, so other members shift too. The largest "
                   "deflection falls by share*(f - 1)/(1 + (f - 1)*(1 - red)) of itself "
                   "(exact along its direction; a negative share means stiffening adds "
                   "deflection); removal (f = 0) raises it by share/red. red < 0.05 = "
                   "ess: removal leaves a mechanism or a nearly singular truss.")
        out.append("- A material change at the same r scales FOS_yielding by the yield "
                   "ratio, FOS_buckling and EA by the E ratio and mass by the density "
                   "ratio (before redistribution). Per kg, yield capacity is cheapest in "
                   "Ti (0.51 x the Al mass, St 2.26 x), buckling capacity in a larger Al "
                   "tube (Ti 1.28 x, St 1.67 x); axial stiffness costs about the same "
                   "(St 0.95 x). Only a material change adds capacity to a member at "
                   "r = %s." % f(R_MAX))

        L_all = nom["L_all"]
        mass_m = RHO_TAB[st["mat"]] * A_COEF * st["r"] ** 2 * L_all
        mech = nom["mech"]
        out.append("")
        out.append("DESIGN AND RESULTS (%slinear stiffness solve)"
                   % ("nominal load; " if self.stochastic else ""))
        if not mech:
            g_all = np.nan_to_num(nom["gov"][act], nan=np.inf)
            fb_a = np.nan_to_num(nom["fos_b"][act], nan=np.inf)
            out.append("Now: mass %s kg; fos_shortfall %s; largest deflection %s mm "
                       "(limit %s); lowest FOS_yielding %s, lowest FOS_buckling %s."
                       % (f(nom["mass"], 5),
                          _sf_txt(float(np.sum(np.maximum(0.0, 1.0 - g_all / FOS_TARGET)))),
                          f(nom["defl"] * 1e3), f(lim["defl"] * 1e3),
                          _fos_txt(float(np.nanmin(nom["fos_y"][act]))),
                          _fos_txt(float(fb_a.min())) if np.isfinite(fb_a.min())
                          else "none (no member in compression)"))
        out.append("Installed members by place (sN section N, bN bay N): id nodes kind "
                   "code | mat, r m, mass kg%s:"
                   % ("" if mech else " | N kN, FOS_yielding, FOS_buckling if compressed "
                      "| red, share % of the largest deflection (unit-load virtual work, "
                      "sum 100) | fos_shortfall if removed alone"))
        if not mech:
            N, fy, fb = nom["N"], nom["fos_y"], nom["fos_b"]
            tiny, comp, red = nom["tiny"], nom["comp"], nom["red"]
            share = np.nan_to_num(nom["share"], nan=0.0)
        place_now = None
        for m in range(M):
            if not act[m]:
                continue
            a, b, kind, place, where = topo[m]
            if place != place_now:
                out.append("%s:" % place)
                place_now = place
            row = "m%d %d-%d %s%s | %s %s %s" % (
                m, a, b, kind, (" " + where) if where else "",
                MATERIALS[int(st["mat"][m])][1], _r_txt(st["r"][m]), f(mass_m[m]))
            if not mech:
                if tiny[m]:
                    res = "~0"
                elif comp[m]:
                    res = "%s %s %s" % (f(N[m] / 1e3), _fos_txt(fy[m]), _fos_txt(fb[m]))
                else:
                    res = "+%s %s" % (f(N[m] / 1e3), _fos_txt(fy[m]))
                rd = float(np.nan_to_num(red[m]))
                sh = 100.0 * float(share[m])
                rem = ox["rem"].get(m)
                row += " | %s | %s %s | %s" % (
                    res, _obs_num(rd, 2) if rd >= 0.01 else ("0" if rd <= 0 else "<0.01"),
                    "0" if abs(sh) < 0.05 else _obs_num(sh, 2),
                    _sf_txt(rem[0]) if rem else "ess")
            out.append(row)
        off_ids = [m for m in range(M) if not act[m]]
        if off_ids:
            # the listed r / mat most optional members share is stated once
            keys = [(int(st["mat"][m]), float(st["r"][m])) for m in off_ids]
            dq, dr = max(set(keys), key=lambda kv: (keys.count(kv), -kv[0], kv[1]))
            out.append("Optional members, not installed (%s, r %s unless shown; id nodes "
                       "kind place code | mass kg if installed%s):"
                       % (MATERIALS[dq][1], _r_txt(dr),
                          "" if mech else " | if installed alone: its N kN and governing "
                          "FOS, y yielding / b buckling | fos_shortfall, largest "
                          "deflection mm"))
            for m in off_ids:
                a, b, kind, place, where = topo[m]
                q = int(st["mat"][m])
                mr = ("%s r %s " % (MATERIALS[q][1], _r_txt(st["r"][m]))
                      if (q, float(st["r"][m])) != (dq, dr) else "")
                row = "m%d %d-%d %s %s%s%s | %s%s" % (
                    m, a, b, kind, place,
                    (" " + where) if where and kind != "xdiag" else "",
                    " (removed)" if m < geo["n_base"] else "", mr,
                    f(RHO_TAB[q] * A_COEF * st["r"][m] ** 2 * L_all[m]))
                ins = ox["inst"].get(m)
                if ins:
                    row += " | %s%s %s%s | %s %s" % (
                        "+" if ins[0] > 0 else "", f(ins[0] / 1e3), _fos_txt(ins[1]),
                        ins[2], _sf_txt(ins[3]), f(ins[4] * 1e3))
                out.append(row)

        if mech:
            out.append("")
            mn = ox["mech_nodes"]
            rm = [m for m in range(geo["n_base"]) if not act[m]
                  and (int(geo["members"][m, 0]) in mn or int(geo["members"][m, 1]) in mn)]
            out.append("RESULTS: the stiffness matrix is singular for this design (a "
                       "mechanism); no member forces are available.%s%s"
                       % ((" Unrestrained node(s): %s." % ", ".join("n%d" % n for n in mn))
                          if mn else "",
                          (" Base members removed at them: %s."
                           % " ".join("m%d" % m for m in rm)) if rm else ""))
            return "\n".join(out)

        # members that fail (or come closest), with the radius each material needs
        gov = np.where(tiny | ~act, np.inf, np.nan_to_num(nom["gov"], nan=np.inf))
        order = [int(m) for m in np.argsort(gov, kind="stable")]
        fail = [m for m in order if gov[m] < tgt][:10]
        lowl = fail + [m for m in order[:4] if m not in fail and math.isfinite(gov[m])]
        if lowl:
            if not fail:
                title = "Lowest-FOS members (all %s)" % (
                    "at or above FOS %s" % tgt_s if self.stochastic else "pass")
            else:
                title = ("Members below FOS %s (the scatter target), lowest first"
                         % tgt_s if self.stochastic else "Failing members, lowest FOS first")
                if len(lowl) > len(fail):
                    title += ", then the next lowest"
            out.append("%s: FOS, mode, mat, r m, L m | radius at which its own FOS "
                       "reaches %s per material (its own force re-computed for the new "
                       "stiffness; other members shift too):" % (title, tgt_s))
            for m in lowl:
                buck = bool(comp[m]) and fb[m] <= fy[m]
                need = []
                for q in range(N_MAT):
                    rq = _r_needed(st["r"][m], int(st["mat"][m]), q,
                                   float(fy[m]), float(fb[m]) if comp[m] else math.inf,
                                   float(np.nan_to_num(red[m])), tgt)
                    if not math.isfinite(rq):
                        txt = "none"
                    elif rq > R_MAX * (1.0 + 1e-9):
                        txt = "%s over max" % _r_txt(rq)
                    elif rq <= R_MIN:
                        txt = "%s(min)" % f(R_MIN)
                    else:
                        txt = _r_txt(rq)
                    need.append("%s %s" % (MATERIALS[q][1], txt))
                out.append("m%d %s %s %s %s %s | %s" % (
                    m, _fos_txt(gov[m]), "buckling" if buck else "yielding",
                    MATERIALS[int(st["mat"][m])][1], _r_txt(st["r"][m]), f(L_all[m]),
                    ", ".join(need)))

        dnode = int(nom["dnode"])
        u = nom["u"]
        dn = nom["dn"]
        dmm = nom["defl"] * 1e3
        dvec = u[dnode] / max(nom["defl"], 1e-15)
        over = [nd for nd in np.argsort(-dn, kind="stable")
                if dn[nd] >= 0.98 * lim["defl"]][:8]
        out.append("Largest deflection: node n%d (J%d), %s mm along unit vector (%s); "
                   "limit %s mm.%s"
                   % (dnode, jel.get(dnode, -1), f(dmm),
                      ", ".join(f(v) for v in dvec), f(lim["defl"] * 1e3),
                      (" Within 2%% of the limit or over: %s."
                       % ", ".join("J%d %s" % (jel.get(int(nd), -1), f(dn[nd] * 1e3))
                                   for nd in over)) if over else ""))

        dmass = [abs(g[2]) for row in ox["jsens"].values() for g in row if g is not None]
        out.append("Free joints (id node | offset dx, dy, dz m | displacement mm | per "
                   "+1 step (+%s m) of dx; dy; dz, first order: change in fos_shortfall "
                   "and largest deflection mm; past a member crossing FOS 1.5 trust the "
                   "sign more than the size. A step changes the mass by at most %s kg):"
                   % (f(step), f(max(dmass + [0.0]), 2)))
        d_tol = max(5e-3, 2e-4 * dmm)
        for nd in joints:
            o = st["off"][nd]
            sens = ox["jsens"].get(int(nd), [None, None, None])
            parts = []
            for ax in range(3):
                g = sens[ax]
                if geo["locked"][nd, ax]:
                    parts.append("d%s locked" % "xyz"[ax])
                elif g is None:
                    parts.append("d%s n/a" % "xyz"[ax])
                else:
                    parts.append("d%s %s %s" % ("xyz"[ax], _d_txt(g[0], 5e-4),
                                                _d_txt(g[1] * 1e3, d_tol)))
            out.append("J%d n%d | %s | %s | %s"
                       % (jel[nd], nd,
                          "0" if not np.any(o) else ", ".join(
                              ("%.4f" % v).rstrip("0").rstrip(".") for v in o),
                          _obs_num(dn[nd] * 1e3, 3), "; ".join(parts)))
        return "\n".join(out)

    def element_label(self, st, i) -> str:
        """Short role label from the fixed geometry only, e.g. 'member 8 (nodes
        2-6, bay-1 diagonal on x+ face)' or 'joint J53 (node 9, section 2
        corner x+y-)'. Material, installed state and force sense are left out:
        the label of a change is taken on the design before it, where they
        contradicted the change itself (e.g. 'not installed: active 0 -> 1')."""
        i = int(i)
        geo = st["_geo"]
        M = self._n_mem(st)
        fam = geo["family"]
        k = geo["k"]
        corner = _OBS_CORNER.get(fam, _OBS_CORNER["tower"])
        nd = self._joint_node(st, i)
        if nd is not None:
            return "joint J%d (node %d, section %d corner %s)" % (
                i, nd, nd // k, corner[nd % k])
        if not (0 <= i < M):
            return "element %d" % i
        a, b, kind, place, where = self._topo(st)[i]
        num = place[1:]
        face = _OBS_FACE_LONG.get(fam, {}).get(where, where)
        if kind == "leg":
            what = "bay-%s leg at corner %s" % (num, where)
        elif kind == "chord":
            what = "bay-%s %s chord" % (num, where.replace("bot", "bottom"))
        elif kind == "diag":
            what = "bay-%s diagonal on %s" % (num, face)
        elif kind == "ring":
            what = "section-%s ring edge on %s" % (num, face)
        elif kind == "sdiag":
            what = "section-%s in-section diagonal c0-c2" % num
        elif kind == "sdiag2":
            what = "section-%s in-section diagonal c1-c3" % num
        elif kind == "xdiag":
            what = "bay-%s body diagonal %s" % (num, where)
        else:
            what = "two-bay brace %s on %s" % (place, face)
        return "member %d (nodes %d-%d, %s)" % (i, a, b, what)


_V2_CODES = {
    "tower": ("Codes: legs their corner; rings and diags their face (x+, x-, y+, "
              "y-; the main wind load is along +x); xdiags their two corners."),
    "cantilever": ("Codes: chords their corner; rings and diags their face (bot = "
                   "z- face, top = z+ face, y+, y-); xdiags their two corners."),
    "bridge": ("Codes: chords their corner (bot = deck chords at z = 0, top = apex "
               "chord); rings, diags and skips their face (deck = plane of the two "
               "bottom chords; y+, y- = inclined side webs)."),
}


def _fos_txt(v) -> str:
    """FOS text: 3 significant figures, 4 near the 1.5 limit so that a value
    just below it never prints as the limit itself."""
    x = float(v)
    if math.isfinite(x) and 1.4 < x < 1.6:
        s = "%.4g" % x
        if x < FOS_TARGET <= float(s):
            s = ("%.6f" % x).rstrip("0")     # 1.49996, not 1.5
        return s
    return _obs_fos(x)


# ── observation helpers (describe / element_label only) ──────────────────────
# Load factor at the adverse 90th percentile of the stochastic variant's
# scatter (lognormal, sd LOAD_SIGMA, mean 1): the level at which the chance
# constraints are judged (PhiEvaluator q = 0.9). Text only.
Q90_LOAD = math.exp(1.2815516 * LOAD_SIGMA - 0.5 * LOAD_SIGMA * LOAD_SIGMA)
_MAT_SHORT = ("6061-T6 aluminium", "S355 steel", "Ti-6Al-4V titanium")


def _fos_cols(props, Nm):
    """_member_fos for a matrix of force columns (members x cases); props =
    (sigma_y, A, E, I, L) per member."""
    sy, A, E, I, L = (np.asarray(p, dtype=float)[:, None] for p in props)
    absN = np.abs(Nm)
    fy = sy / np.maximum(absN / np.maximum(A, 1e-15), 1e-9)
    comp = Nm < -1e-9
    p_cr = (math.pi ** 2) * E * I / (L ** 2)
    fb = np.where(comp, p_cr / np.maximum(absN, 1e-12), np.inf)
    return np.minimum(fy, FOS_CAP), np.minimum(fb, FOS_CAP), comp


def _shortfall_cols(props, Nm, keep=None):
    """_shortfall per force column; keep (0/1, same shape) drops entries."""
    fy, fb, comp = _fos_cols(props, Nm)
    gov = np.where(comp, np.minimum(fy, fb), fy)
    s = np.maximum(0.0, 1.0 - gov / FOS_TARGET)
    if keep is not None:
        s = s * keep
    return s.sum(axis=0)


def _defl_cols(geo, ufc):
    """Largest nodal displacement magnitude per free-DOF displacement column."""
    n = geo["n_node"]
    U = np.zeros((3 * n, ufc.shape[1]))
    U[geo["free"]] = ufc
    U = U.reshape(n, 3, -1)
    return np.sqrt((U * U).sum(axis=1)).max(axis=0)


def _mech_nodes(geo, sol) -> list:
    """Nodes moved by the (near-)singular modes of a mechanism's stiffness."""
    dof_node = np.flatnonzero(geo["free"]) // 3
    if geo["n_free"] == 0:
        return []
    if "B" not in sol:
        return sorted(set(int(n) for n in dof_node))
    K = sol["B"].T @ (sol["k"][:, None] * sol["B"])
    if not np.all(np.isfinite(K)):
        return []
    d = np.diag(K)
    w, V = np.linalg.eigh(K)
    null = V[:, w <= PIVOT_TOL * max(float(d.max()), 1e-30)]
    if null.shape[1] == 0:
        null = V[:, :1]
    wt = np.zeros(geo["n_node"])
    np.add.at(wt, dof_node, (null ** 2).sum(axis=1))
    np.add.at(wt, dof_node[d <= 0.0], 1.0)
    if wt.max() <= 0.0:
        return []
    return [int(n) for n in np.flatnonzero(wt >= 0.2 * wt.max())]


def _r_needed(r, q0, q, fos_y, fos_b, red, target) -> float:
    """Radius at which a member's OWN governing FOS reaches `target` in
    material q (from material q0 at radius r), with its own force
    re-computed exactly for the stiffness change: N' = N f / (red + f (1 -
    red)), f = (E_q / E_q0) x (r' / r)^2. inf when no radius reaches it."""
    e = E_TAB[q] / E_TAB[q0]
    yr = SY_TAB[q] / SY_TAB[q0]
    one = 1.0 - red
    # x = (r'/r)^2.  FOS_y' = (yr/e) fos_y (red + e x one)
    if one < 1e-9:
        x_y = 0.0 if yr / e * fos_y * red >= target else math.inf
    else:
        x_y = (target * e / (yr * fos_y) - red) / (e * one)
    # FOS_b' = fos_b x (red + e x one)
    if not math.isfinite(fos_b):
        x_b = 0.0
    elif one < 1e-9:
        x_b = target / (fos_b * red) if red > 0 else math.inf
    else:
        x_b = (-red + math.sqrt(red * red + 4.0 * e * one * target / fos_b)) / (2.0 * e * one)
    x = max(x_y, x_b, 0.0)
    return r * math.sqrt(x) if math.isfinite(x) else math.inf


def _r_txt(v) -> str:
    """Radius as set() stores it (1e-5 m), flagged at the bounds; a value
    above the largest tube prints with 4 significant figures."""
    x = float(v)
    if x > R_MAX * (1.0 + 1e-9):
        return "%.4g" % x
    if x >= R_MAX - 1e-9:
        return "%s(max)" % _obs_num(R_MAX)
    if x <= R_MIN + 1e-9:
        return "%s(min)" % _obs_num(R_MIN)
    return ("%.5f" % x).rstrip("0").rstrip(".")


def _sf_txt(v) -> str:
    """fos_shortfall, 4 significant figures (outcomes differ in the third)."""
    x = float(v)
    return "0" if x < 5e-5 else _obs_num(x, 4)


def _d_txt(v, tol) -> str:
    """Signed sensitivity, 2 significant figures; '0' below tol."""
    x = float(v)
    if not math.isfinite(x):
        return "n/a"
    if abs(x) < tol:
        return "0"
    return ("+" if x > 0 else "") + _obs_num(x, 2)
