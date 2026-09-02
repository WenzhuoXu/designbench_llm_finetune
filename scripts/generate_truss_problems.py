#!/usr/bin/env python
"""Procedural generator for NEW DesignBench truss problems.

WHY THIS EXISTS
---------------
DesignBench ships 100 ``auto_problem_*`` specs (plus ~30 hand/test specs).  With
a 3:1 train/held-out split that leaves ~34 held-out problems, and a paired
McNemar evaluation on 34 problems bottoms out near 5 percentage points of
resolution no matter how much GPU is spent.  Minimum detectable effect scales as
1/sqrt(n_problems), so more problems is the only lever.  This script mints more
of the same *kind* of problem.

WHAT THE ORIGINALS ACTUALLY ARE (measured, not assumed)
-------------------------------------------------------
Read from ``DesignBench/data/problems/auto_problem_*.json`` (n = 100) and from
the generator that made them, ``DesignBench/scripts/generate_modification_trees.py``:

  * 2-row Warren ground structure, ``num_bays in {2,3,4}`` -> nx = bays+1
    columns -> 6/8/10 joints.  Joint id == row*nx + col.
  * span ~ U(6, 14) m, height ~ U(1.5, 4.0) m, z is always 0 (planar).
  * supports: joint 0 pinned (bottom-left), joint nx-1 roller_y (bottom-right).
  * ground structure = bottom chord (bays) + top chord (bays) + all verticals
    (bays+1) + both diagonals per bay (2*bays) = 5*bays + 1 members.
    Observed max member counts 11/16/21 for bays 2/3/4 confirm this exactly.
  * material: one per problem, uniform over {6061_T6_Aluminum, A36_Steel}
    (measured 49/51 across the 100).
  * shape: Pipe everywhere; initial r=0.03, t=0.005 (aspect t/r = 1/6).  The
    measured t/r distribution over all 1455 members is p90 = 0.167 = 1/6 with a
    tail down to 0.042 -- because the optimiser scales r and t together and only
    the degradation shrinks t on its own.
  * loading: 1-3 point loads (measured 34/29/37), always on TOP-row joints,
    always pure -y, magnitude ~ U(30000, 80000) N.
  * goals: ``minimum_fos_buckling = minimum_fos_yielding = 1.5`` and
    ``maximum_mass``.  No deflection goal.  ``maximum_mass / optimal_mass ==
    1.100000`` for all 100 problems (measured; not assumed).
  * ``_metadata.degradation_steps`` in {3..7} (measured 15/26/23/20/16).

  The original ``generation_method`` string says ``ground_structure_lp`` but the
  code behind it is a damped fully-stressed-design iteration, not an LP.  This
  script implements the LP the metadata advertises (see ``lp_size``) AND keeps a
  fully-stressed refinement on top of it, because Euler buckling makes the
  ground-structure problem nonconvex and no single LP can be the true optimum.

THE PROCEDURE
-------------
  1. GROUND STRUCTURE.  Sample span/height/bays/material/loading from the
     measured ranges above and build the fully-connected 2-row Warren truss.
  2. LP SIZING.  Solve, over member areas A_i and axial forces q_i,

         min  sum_i rho_i L_i A_i
         s.t. B q = -f                      (nodal equilibrium, free DOFs)
              q_i <= (sigma_y / FOS_y) A_i          (tension / yielding)
             -q_i <= sigma_c_i A_i                  (compression)
              A_min <= A_i <= A_max

     With the pipe aspect t/r fixed at k, A = c_A r^2 and I = c_I r^4, so
     I = (c_I / c_A^2) A^2 and the Euler allowable compressive stress
     sigma_c = pi^2 E I / (FOS_b L^2 A) = beta_i A is LINEAR in A -- which makes
     the true constraint -q_i <= beta_i A_i^2 quadratic and the feasible set
     nonconvex.  We therefore solve a SEQUENTIAL LP: sigma_c_i is frozen at
     beta_i * A_i from the previous iterate and the LP is re-solved (damped)
     until the mass stops moving.  This is the standard ground-structure
     treatment of buckling and each iterate is a genuine LP (HiGHS).
  3. FEA REFINEMENT.  The LP works with a statically admissible force field; the
     benchmark's simulator (trussme) runs a linear-elastic FEA whose forces
     differ for indeterminate trusses.  So the LP sizing is handed to a
     fully-stressed-design loop driven by the REAL simulator, and the lowest-mass
     state that the real simulator certifies as FOS_b >= 1.5 and FOS_y >= 1.5 is
     kept.  That certified mass is what goes into ``_metadata.optimal_mass``.
  4. DEGRADATION.  The original's three moves, with the original's distributions:
     ``shrink_all`` (t *= U(0.75,0.90)), ``shrink_random`` (one member's
     t *= U(0.60,0.85)), ``remove_member`` (a non-base member).  Loop until the
     truss is infeasible AND at least ``degradation_steps ~ randint(3,7)`` moves
     have been made.  The degraded design becomes the problem's ``topology``.
     One deliberate change: a removal that would turn the truss into a mechanism
     is skipped (see check V3), because such problems are not solvable design
     problems -- they are simulator artefacts.
  5. RE-OPTIMISE ON THE POSED TOPOLOGY.  Degradation can delete members, so the
     ground-structure optimum is generally NOT attainable from the topology the
     agent is actually handed.  (This is a real defect of the originals: their
     ``optimal_mass`` is computed before removal, so some of them state a mass
     budget no sizing policy can reach.)  Here steps 2-3 are re-run on the
     post-degradation topology, and ``optimal_mass`` is that certified optimum.
     ``goals.maximum_mass = 1.1 * optimal_mass``, the ratio measured above.

VALIDATION -- and why each check exists
---------------------------------------
  V1  LOADS AND IS FINITE.  ``_load_truss_and_goals`` + ``analyze_truss`` must
      return finite mass / FOS / deflection.  trussme's ``analyze()`` calls
      ``numpy.linalg.solve`` on the free-DOF stiffness matrix; a singular matrix
      raises and DesignBench's wrapper silently returns ``mass = inf``.  11 of
      the 100 shipped auto problems fail exactly this way.
  V2  INITIAL DESIGN INFEASIBLE.  A problem whose start state already satisfies
      the goals has nothing to solve; it would inflate any feasibility metric.
      (All 100 originals pass this.)
  V3  NOT A MECHANISM.  Two tests.  (a) Exact: the geometric equilibrium matrix
      B (n_free_dofs x n_members) must have full row rank, with
      sigma_min/sigma_max above a tolerance -- this is area-independent and
      catches degeneracy at its source.  (b) Empirical: the simulator's
      deflection must stay under 0.1 * span, the same bound the repo's own
      ``program_from_truss_spec`` uses.  28 of the 100 originals report
      deflections of 1e10-1e14 m (or inf) while showing large FOS: they are
      kinematically unstable, and the search ladder showed ~17% of "successes"
      on the shipped set were policies exploiting exactly that.
  V4  A FEASIBLE DESIGN PROVABLY EXISTS.  We do not merely hope the problem is
      solvable: we hold a concrete design (the step-5 optimum) that the real
      simulator certifies feasible, with every parameter inside the problem's own
      declared bounds, so it is reachable by SCALE_PARAM actions.  This is a
      stronger guarantee than any heuristic search result.
  V5  A SEARCH POLICY ACTUALLY FINDS IT (optional, --require-policy-solve).
      Runs the repo's own policies from ``scripts/search_ladder.py`` -- greedy
      critical-member, fully-stressed design, random, and (only if those fail)
      potential-guided one-step lookahead -- for at most --policy-steps steps and
      requires at least one to reach feasibility.  NOTE: this gate makes the
      shipped set 100% solvable-by-some-policy by construction, which the
      originals are not; the manifest records the per-policy outcome for EVERY
      candidate so the unbiased comparison can still be made.
  V6  UNIQUE ID.  ``gen_problem_NNNN``: a distinct prefix that cannot collide
      with ``auto_problem_*`` or any of the hand-written specs.

Nothing here writes to ``DesignBench/data/problems/``.

Usage
-----
    python scripts/generate_truss_problems.py --n 300 \
        --out /ocean/projects/mch250030p/wxu7/DesignBench/data/problems_gen \
        --seed 20260827 --workers 32
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

# trussme divides by a member's force to get its FOS, so zero-force members raise
# "divide by zero" every single FEA call. The values (inf) are correct and are
# handled explicitly below; the warnings are pure noise at generation volume.
warnings.filterwarnings("ignore", category=RuntimeWarning)

DESIGNBENCH = Path("/ocean/projects/mch250030p/wxu7/DesignBench")
PROJECT = Path("/ocean/projects/mch250030p/wxu7/llm_finetune")
for _p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scipy.optimize import linprog  # noqa: E402

from llm_finetune.envs.truss_env import (  # noqa: E402
    _analyze_truss,
    _load_truss_and_goals,
)
from llm_finetune.training.rl.posterior.potential import (  # noqa: E402
    program_from_truss_spec,
)

# ── constants taken from the measured originals ──────────────────────────────

MATERIALS = ("6061_T6_Aluminum", "A36_Steel")

# The optimisation block is byte-identical across all 100 originals.
OPTIMIZATION_BLOCK = {
    "shape": "Pipe",
    "shape_params": {
        "r": {"min": 0.005, "max": 0.15, "initial": 0.03},
        "t": {"min": 0.001, "max": 0.02, "initial": 0.005},
    },
}
R_MIN, R_MAX = 0.005, 0.15
T_MIN, T_MAX = 0.001, 0.02
R0, T0 = 0.03, 0.005
ASPECT = T0 / R0                      # t/r = 1/6, the aspect the originals keep
FOS_MIN = 1.5
MASS_BUDGET_RATIO = 1.1               # measured: max_mass/optimal_mass == 1.1 exactly

SPAN_RANGE = (6.0, 14.0)
HEIGHT_RANGE = (1.5, 4.0)
BAYS_CHOICES = (2, 3, 4)
LOAD_RANGE = (30000.0, 80000.0)
DEGRADATION_STEPS_RANGE = (3, 7)

# V3(b): the repo's own mechanism bound (posterior/potential.py).
DEFLECTION_BOUND_FRAC = 0.1
# V3(a): smallest acceptable sigma_min/sigma_max of the equilibrium matrix.
RANK_COND_TOL = 1e-6

MATERIAL_PROPS = {
    "A36_Steel": {"density": 7850.0, "E": 200e9, "sigma_y": 250e6},
    "6061_T6_Aluminum": {"density": 2700.0, "E": 68.9e9, "sigma_y": 276e6},
}


# ── pipe section algebra (aspect t/r held at k) ──────────────────────────────
#
#   A(r) = pi (r^2 - (r-t)^2) = c_A r^2 ,  c_A = pi k (2 - k)
#   I(r) = (pi/4)(r^4 - (r-t)^4) = c_I r^4 ,  c_I = (pi/4)(1 - (1-k)^4)
# so I = (c_I / c_A^2) A^2 and the Euler allowable stress is linear in A.

def _cA(k: float = ASPECT) -> float:
    return math.pi * k * (2.0 - k)


def _cI(k: float = ASPECT) -> float:
    return (math.pi / 4.0) * (1.0 - (1.0 - k) ** 4)


def area_to_radius(area: float, k: float = ASPECT) -> float:
    return math.sqrt(max(area, 1e-12) / _cA(k))


def radius_to_area(r: float, t: float) -> float:
    return math.pi * (r ** 2 - (r - t) ** 2)


A_MIN = radius_to_area(R_MIN, T_MIN)
A_MAX = radius_to_area(min(R_MAX, T_MAX / ASPECT), T_MAX)


# ── 1. ground structure ──────────────────────────────────────────────────────

def build_ground_structure(span: float, height: float, num_bays: int,
                           material: str) -> tuple[dict, list[int]]:
    """The fully-connected 2-row Warren ground structure the originals use.

    Member order is chord-bottom, chord-top, verticals, diagonals -- identical to
    ``DesignBench/scripts/generate_modification_trees.py`` so member indices mean
    the same thing.  Returns (topology, base_member_indices); base members are
    the bottom chord and the two end verticals, which degradation may not remove.
    """
    joints: list[dict] = []
    bay = span / num_bays
    bottom, top = [], []
    for i in range(num_bays + 1):
        support = "pinned" if i == 0 else ("roller_y" if i == num_bays else None)
        joints.append({"id": len(joints), "position": [i * bay, 0.0, 0.0], "support": support})
        bottom.append(joints[-1]["id"])
    for i in range(num_bays + 1):
        joints.append({"id": len(joints), "position": [i * bay, height, 0.0], "support": None})
        top.append(joints[-1]["id"])

    shape = {"type": "Pipe", "r": R0, "t": T0}
    members: list[dict] = []

    def add(a: int, b: int) -> None:
        members.append({"joints": [a, b], "material": material, "shape": dict(shape)})

    for i in range(num_bays):                       # bottom chord   (base)
        add(bottom[i], bottom[i + 1])
    for i in range(num_bays):                       # top chord
        add(top[i], top[i + 1])
    for i in range(num_bays + 1):                   # verticals
        add(bottom[i], top[i])
    for i in range(num_bays):                       # both diagonals per bay
        add(bottom[i], top[i + 1])
        add(bottom[i + 1], top[i])

    base = list(range(num_bays)) + [2 * num_bays, 3 * num_bays]
    return {"joints": joints, "members": members}, base


def sample_loading(rng: random.Random, topology: dict) -> list[dict]:
    """1-3 downward point loads on distinct TOP-row joints (as measured)."""
    top_ids = [j["id"] for j in topology["joints"] if j["position"][1] > 0]
    n = rng.randint(1, min(3, len(top_ids)))
    return [
        {"joint": jid, "force": [0.0, -rng.uniform(*LOAD_RANGE), 0.0]}
        for jid in rng.sample(top_ids, n)
    ]


# ── V3(a): exact mechanism test on the geometric equilibrium matrix ──────────

def equilibrium_matrix(topology: dict) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """B with one row per free DOF and one column per member.

    trussme restrains z at every joint (``add_out_of_plane_support('z')``), so the
    free DOFs are x,y at unsupported joints and x at the roller.  B depends only
    on geometry and connectivity, never on member sizes, which is what makes it
    the right instrument for "is this thing a mechanism".
    """
    joints = topology["joints"]
    pos = {j["id"]: np.array(j["position"][:2], dtype=float) for j in joints}
    restricted = {}
    for j in joints:
        s = j.get("support")
        if s == "pinned":
            restricted[j["id"]] = (True, True)
        elif s == "roller_y":
            restricted[j["id"]] = (False, True)
        elif s == "roller_x":
            restricted[j["id"]] = (True, False)
        else:
            restricted[j["id"]] = (False, False)

    dof_index: dict[tuple[int, int], int] = {}
    dof_list: list[tuple[int, int]] = []
    for j in joints:
        for axis in (0, 1):
            if not restricted[j["id"]][axis]:
                dof_index[(j["id"], axis)] = len(dof_list)
                dof_list.append((j["id"], axis))

    members = topology["members"]
    B = np.zeros((len(dof_list), len(members)), dtype=float)
    for m_idx, m in enumerate(members):
        a, b = m["joints"]
        vec = pos[b] - pos[a]
        L = float(np.linalg.norm(vec))
        if L <= 0:
            continue
        d = vec / L
        for axis in (0, 1):
            if (a, axis) in dof_index:
                B[dof_index[(a, axis)], m_idx] += d[axis]
            if (b, axis) in dof_index:
                B[dof_index[(b, axis)], m_idx] -= d[axis]
    return B, dof_list


def stability(topology: dict) -> tuple[bool, float]:
    """(is_stable, sigma_min/sigma_max of B).

    Full row rank of B <=> every free DOF is restrained by some member <=> the
    stiffness matrix K = B diag(EA/L) B^T is nonsingular <=> trussme's solve does
    not blow up.  A tiny-but-nonzero smallest singular value is a NEAR-mechanism:
    K is then ill-conditioned and the FEA reports deflections of 1e10 m or more.
    """
    B, dofs = equilibrium_matrix(topology)
    if B.shape[1] < B.shape[0]:
        return False, 0.0
    sv = np.linalg.svd(B, compute_uv=False)
    if sv.size < B.shape[0] or sv[0] <= 0:
        return False, 0.0
    ratio = float(sv[-1] / sv[0])
    return ratio > RANK_COND_TOL, ratio


# ── 2. the LP ────────────────────────────────────────────────────────────────

def lp_size(topology: dict, loading: list[dict], *, fos_b: float = FOS_MIN,
            fos_y: float = FOS_MIN, iters: int = 20,
            damping: float = 0.5) -> Optional[tuple[np.ndarray, float]]:
    """Sequential ground-structure LP for minimum-mass member areas.

    Returns (areas, lp_mass) or None if the LP is infeasible (which happens when
    the topology cannot carry the load at all -- a mechanism in the load
    direction).  See the module docstring for the formulation and for why the
    buckling constraint forces a sequence of LPs rather than one.
    """
    B, dof_list = equilibrium_matrix(topology)
    members = topology["members"]
    n = len(members)
    if n == 0 or B.shape[0] == 0:
        return None

    joints = {j["id"]: np.array(j["position"][:2], dtype=float) for j in topology["joints"]}
    lengths = np.array([
        float(np.linalg.norm(joints[m["joints"][1]] - joints[m["joints"][0]])) for m in members
    ])
    props = [MATERIAL_PROPS[m["material"]] for m in members]
    rho = np.array([p["density"] for p in props])
    E = np.array([p["E"] for p in props])
    sy = np.array([p["sigma_y"] for p in props]) / fos_y

    # external load vector on the free DOFs
    load_by_joint: dict[int, np.ndarray] = {}
    for ld in loading:
        load_by_joint.setdefault(ld["joint"], np.zeros(2))
        load_by_joint[ld["joint"]] += np.array(ld["force"][:2], dtype=float)
    f = np.zeros(len(dof_list))
    for i, (jid, axis) in enumerate(dof_list):
        if jid in load_by_joint:
            f[i] = load_by_joint[jid][axis]

    # beta_i: Euler allowable compressive STRESS per unit area, sigma_c = beta_i * A
    beta = (math.pi ** 2) * E * (_cI() / _cA() ** 2) / (fos_b * lengths ** 2)

    c = np.concatenate([rho * lengths, np.zeros(n)])
    A_eq = np.hstack([np.zeros_like(B), B])
    b_eq = -f
    bounds = [(A_MIN, A_MAX)] * n + [(None, None)] * n

    areas = np.full(n, A_MAX)
    best: Optional[tuple[np.ndarray, float]] = None
    for _ in range(iters):
        sigma_c = np.minimum(sy, beta * areas)
        A_ub = np.vstack([
            np.hstack([-np.diag(sy), np.eye(n)]),        #  q_i - sy A_i <= 0
            np.hstack([-np.diag(sigma_c), -np.eye(n)]),  # -q_i - sc A_i <= 0
        ])
        b_ub = np.zeros(2 * n)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if not res.success:
            return None
        new_areas = np.clip(res.x[:n], A_MIN, A_MAX)
        mass = float(np.sum(rho * lengths * new_areas))
        if best is None or mass < best[1]:
            best = (new_areas.copy(), mass)
        if np.allclose(new_areas, areas, rtol=1e-4, atol=1e-9):
            areas = new_areas
            break
        areas = damping * areas + (1.0 - damping) * new_areas
    if best is None:
        return None
    # The frozen-sigma_c LP under-sizes compression members whose area shrank on
    # the final iterate; report the converged areas, not the cheapest LP value.
    return areas, float(np.sum(rho * lengths * areas))


def areas_to_shapes(areas: np.ndarray) -> list[dict]:
    """Areas -> Pipe(r, t) at the fixed aspect, clamped to the declared bounds."""
    out = []
    for a in areas:
        r = min(max(area_to_radius(float(a)), R_MIN), R_MAX)
        t = min(max(r * ASPECT, T_MIN), T_MAX)
        out.append({"type": "Pipe", "r": r, "t": t})
    return out


# ── 3. FEA-driven fully-stressed refinement (the certified optimum) ──────────

def _member_fos(truss) -> list[tuple[float, float]]:
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in truss.members:
            try:
                fb = float(m.fos_buckling)
            except Exception:
                fb = float("inf")
            try:
                fy = float(m.fos_yielding)
            except Exception:
                fy = float("inf")
            out.append((fb if math.isfinite(fb) else float("inf"),
                        fy if math.isfinite(fy) else float("inf")))
    return out


def _spec_from(topology: dict, loading: list[dict], goals: dict) -> dict:
    return {"topology": topology, "loading": loading, "goals": goals,
            "optimization": copy.deepcopy(OPTIMIZATION_BLOCK)}


def fsd_optimize(topology: dict, loading: list[dict], *, deflection_bound: float,
                 margin: float = 1.02, iters: int = 60) -> Optional[dict]:
    """Fully-stressed design under the REAL simulator; returns the best certified
    feasible sizing found, as {"shapes": [...], "mass": float, "state": {...}}.

    A pipe's yielding FOS scales with area (s^2 when r and t scale together) and
    its buckling FOS with the second moment of area (s^4), so a member needs
        s = max((target_y / fos_y)^(1/2), (target_b / fos_b)^(1/4)).
    Over-designed members get s < 1 and shrink, which is how the loop reaches a
    LOW mass rather than merely a safe one.  Every iterate is checked against the
    simulator, so what comes back is certified, not extrapolated.
    """
    from validation.truss_executor import analyze_truss, load_truss_from_problem

    goals = {"minimum_fos_buckling": FOS_MIN, "minimum_fos_yielding": FOS_MIN}
    shapes = [dict(m["shape"]) for m in topology["members"]]
    tgt_b = FOS_MIN * margin
    tgt_y = FOS_MIN * margin
    best: Optional[dict] = None

    for _ in range(iters):
        top = copy.deepcopy(topology)
        for m, sh in zip(top["members"], shapes):
            m["shape"] = dict(sh)
        try:
            truss = load_truss_from_problem(_spec_from(top, loading, goals))
            state = analyze_truss(truss, goals)
        except Exception:
            return best
        mass = state.get("mass", float("inf"))
        defl = state.get("deflection", float("inf"))
        fb, fy = state.get("fos_buckling", 0.0), state.get("fos_yielding", 0.0)
        ok = (
            all(isinstance(v, (int, float)) and math.isfinite(v) for v in (mass, defl, fb, fy))
            and mass > 0.0 and defl <= deflection_bound
            and fb >= FOS_MIN and fy >= FOS_MIN
        )
        if ok and (best is None or mass < best["mass"]):
            best = {"shapes": [dict(s) for s in shapes], "mass": float(mass),
                    "state": {k: state.get(k) for k in
                              ("mass", "fos_buckling", "fos_yielding", "deflection")}}

        fos = _member_fos(truss)
        moved = False
        for i, (fb_i, fy_i) in enumerate(fos):
            need_y = math.sqrt(tgt_y / fy_i) if math.isfinite(fy_i) and fy_i > 0 else 0.7
            need_b = (tgt_b / fb_i) ** 0.25 if math.isfinite(fb_i) and fb_i > 0 else 0.7
            s = min(2.0, max(0.7, max(need_y, need_b)))
            r = min(max(shapes[i]["r"] * s, R_MIN), R_MAX)
            t = min(max(r * ASPECT, T_MIN), T_MAX)
            if abs(r - shapes[i]["r"]) > 1e-9 or abs(t - shapes[i]["t"]) > 1e-12:
                moved = True
            shapes[i] = {"type": "Pipe", "r": r, "t": t}
        if not moved:
            break
    return best


def certified_optimum(topology: dict, loading: list[dict], deflection_bound: float
                      ) -> Optional[dict]:
    """LP sizing -> FSD refinement, plus two cheap restarts, keep the lightest.

    The restarts matter: FSD is a fixed-point iteration, not a global optimiser,
    and starting from the LP areas versus from a uniform oversized design lands
    on measurably different masses.  Taking the min over starts makes
    ``optimal_mass`` a tighter (harder) budget, which is the conservative choice.
    """
    starts: list[list[dict]] = []
    lp = lp_size(topology, loading)
    lp_mass = None
    if lp is not None:
        areas, lp_mass = lp
        starts.append(areas_to_shapes(areas))
        starts.append(areas_to_shapes(np.clip(areas * 1.6, A_MIN, A_MAX)))
    starts.append([{"type": "Pipe", "r": R0 * 2.0, "t": T0 * 2.0}
                   for _ in topology["members"]])

    best = None
    for shapes in starts:
        top = copy.deepcopy(topology)
        for m, sh in zip(top["members"], shapes):
            m["shape"] = dict(sh)
        got = fsd_optimize(top, loading, deflection_bound=deflection_bound)
        if got is not None and (best is None or got["mass"] < best["mass"]):
            best = got
    if best is not None:
        best["lp_mass"] = lp_mass
    return best


# ── 3b. FREE-ASPECT optimum: the mass reference that actually binds ──────────
#
# WHY THIS EXISTS.  The block above sizes every pipe at the fixed aspect
# t/r = 1/6 the originals ship.  That is NOT the lightest design the problem's
# own declared bounds allow: r may reach 0.15 and t may fall to 0.001, an aspect
# of 1/150.  For a thin-walled pipe A ~ 2*pi*r*t and I ~ pi*r^3*t, so
# I = A*r^2/2 -- buckling capacity per unit mass grows as r^2.  A member that is
# buckling-critical is therefore MUCH cheaper as a large-radius thin wall than as
# a 1/6-aspect stub.  Measured on 8 problems of the previous batch, the certified
# free-aspect optimum is 0.36-0.83x of the fixed-aspect ``optimal_mass``.
#
# That gap is the whole bug.  ``maximum_mass = 1.1 * optimal_mass`` with a
# fixed-aspect optimal_mass leaves the search ~27% of unusable slack (measured:
# lookahead_v2_d1 reached feasibility at mass/optimal_mass = 0.80 on the previous
# batch, i.e. 0.73x of its own budget).  With that much slack the mass constraint
# never binds, so "scale every member up until FOS clears" always works -- one
# global SCALE_MULTI_PARAM -- and there is no design decision left in the problem.
#
# So the mass reference is computed here against the free aspect (down to an
# ``aspect_min`` floor, because an infinitely thin wall is not a pipe), and the
# posed budget is ``1.1 * optimal_mass`` with
# ``optimal_mass = certified_mass * mass_slack``.  ``mass_slack >= 1`` is the
# difficulty dial: slack 1.0 poses the budget at the attainable optimum, slack
# 2.0 poses a comfortable one.  The 1.1 invariant every original satisfies is
# preserved exactly, and V4 still holds by construction because a design
# certified at ``certified_mass <= maximum_mass`` is held in hand.

def pipe_area(r: float, t: float) -> float:
    return math.pi * t * (2.0 * r - t)


def pipe_inertia(r: float, t: float) -> float:
    return (math.pi / 4.0) * (r ** 4 - (r - t) ** 4)


# log grid over the declared radius range; 60 points resolves the optimum to
# well under a percent and costs ~60 bisections per member per FSD iterate.
_R_GRID = tuple(R_MIN * (R_MAX / R_MIN) ** (i / 59.0) for i in range(60))


def min_area_section(area_req: float, inertia_req: float, aspect_min: float
                     ) -> Optional[tuple[float, float, float]]:
    """Lightest Pipe(r, t) inside the declared bounds with A >= A_req, I >= I_req.

    Both A and I are increasing in t on 0 < t <= r (dA/dt = 2*pi*(r-t) >= 0,
    dI/dt = pi*(r-t)^3 >= 0), so for each r the smallest admissible t is found by
    bisection and the outer sweep over r picks the lightest.  ``aspect_min``
    floors t/r: without it the optimiser walks to a 1 mm wall on a 150 mm pipe,
    which the linear-elastic simulator happily accepts and no real pipe survives.
    """
    best: Optional[tuple[float, float, float]] = None
    for r in _R_GRID:
        t_lo = max(T_MIN, aspect_min * r)
        t_hi = min(T_MAX, r)
        if t_lo > t_hi:
            continue
        if pipe_area(r, t_hi) < area_req or pipe_inertia(r, t_hi) < inertia_req:
            continue
        if pipe_area(r, t_lo) >= area_req and pipe_inertia(r, t_lo) >= inertia_req:
            t = t_lo
        else:
            lo, hi = t_lo, t_hi
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if pipe_area(r, mid) >= area_req and pipe_inertia(r, mid) >= inertia_req:
                    hi = mid
                else:
                    lo = mid
            t = hi
        a = pipe_area(r, t)
        if best is None or a < best[0]:
            best = (a, r, t)
    return best


def fsd_optimize_free(topology: dict, loading: list[dict], *, deflection_bound: float,
                      aspect_min: float, start: Optional[list[dict]] = None,
                      margin: float = 1.03, iters: int = 60, damping: float = 0.5
                      ) -> Optional[dict]:
    """Fully-stressed design with r and t free, certified by the REAL simulator.

    Same fixed-point structure as ``fsd_optimize`` but the per-member update is a
    genuine two-variable section choice: from the member's own utilisation take
    the required area factor (yielding) and inertia factor (buckling), damp them,
    and ask ``min_area_section`` for the lightest section meeting both.  Returns
    the lightest state the simulator certifies feasible, or None.
    """
    from validation.truss_executor import analyze_truss, load_truss_from_problem

    goals = {"minimum_fos_buckling": FOS_MIN, "minimum_fos_yielding": FOS_MIN}
    n = len(topology["members"])
    if start is not None:
        shapes = [dict(s) for s in start]
    else:
        shapes = [{"type": "Pipe", "r": R0, "t": T0} for _ in range(n)]
    best: Optional[dict] = None

    for _ in range(iters):
        top = copy.deepcopy(topology)
        for m, sh in zip(top["members"], shapes):
            m["shape"] = dict(sh)
        try:
            truss = load_truss_from_problem(_spec_from(top, loading, goals))
            state = analyze_truss(truss, goals)
        except Exception:
            return best
        mass = state.get("mass", float("inf"))
        defl = state.get("deflection", float("inf"))
        fb, fy = state.get("fos_buckling", 0.0), state.get("fos_yielding", 0.0)
        ok = (
            all(isinstance(v, (int, float)) and math.isfinite(v) for v in (mass, defl, fb, fy))
            and mass > 0.0 and defl <= deflection_bound
            and fb >= FOS_MIN and fy >= FOS_MIN
        )
        if ok and (best is None or mass < best["mass"]):
            best = {"shapes": [dict(s) for s in shapes], "mass": float(mass),
                    "state": {k: state.get(k) for k in
                              ("mass", "fos_buckling", "fos_yielding", "deflection")}}

        moved = False
        for i, (fb_i, fy_i) in enumerate(_member_fos(truss)):
            r, t = shapes[i]["r"], shapes[i]["t"]
            a_cur, i_cur = pipe_area(r, t), pipe_inertia(r, t)
            need_a = (FOS_MIN * margin / fy_i) if (math.isfinite(fy_i) and fy_i > 0) else 0.5
            need_i = (FOS_MIN * margin / fb_i) if (math.isfinite(fb_i) and fb_i > 0) else 0.5
            need_a = min(4.0, max(0.5, need_a)) ** damping
            need_i = min(4.0, max(0.5, need_i)) ** damping
            got = min_area_section(a_cur * need_a, i_cur * need_i, aspect_min)
            if got is None:
                continue
            _, nr, nt = got
            if abs(nr - r) > 1e-9 or abs(nt - t) > 1e-12:
                moved = True
            shapes[i] = {"type": "Pipe", "r": nr, "t": nt}
        if not moved:
            break
    return best


def certified_optimum_free(topology: dict, loading: list[dict], deflection_bound: float,
                           aspect_min: float, seeds: Optional[list[list[dict]]] = None
                           ) -> Optional[dict]:
    """Free-aspect optimum over several restarts; FSD is a fixed point, not a
    global optimiser, and different starts land on measurably different masses."""
    starts: list[Optional[list[dict]]] = [None]
    starts.append([{"type": "Pipe", "r": R0 * 2.0, "t": T0} for _ in topology["members"]])
    starts.append([{"type": "Pipe", "r": R_MAX * 0.6, "t": T_MIN * 2.0}
                   for _ in topology["members"]])
    for s in (seeds or []):
        starts.append(s)
    best = None
    for st in starts:
        got = fsd_optimize_free(topology, loading, deflection_bound=deflection_bound,
                                aspect_min=aspect_min, start=st)
        if got is not None and (best is None or got["mass"] < best["mass"]):
            best = got
    return best


# ── 4. degradation ───────────────────────────────────────────────────────────

def degrade(topology: dict, loading: list[dict], base_indices: list[int],
            n_steps: int, rng: random.Random) -> tuple[dict, int, dict]:
    """The originals' degradation, with mechanism-creating removals skipped.

    Move distribution and factor ranges are copied verbatim from
    ``generate_modification_trees.py::_degrade_truss``.  The loop runs until the
    truss is infeasible (min FOS < 1.5) AND at least ``n_steps`` moves are done,
    which is why ``_metadata.degradation_steps`` in the originals is never larger
    than the sampled target.
    """
    from validation.truss_executor import analyze_truss, load_truss_from_problem

    goals = {"minimum_fos_buckling": FOS_MIN, "minimum_fos_yielding": FOS_MIN}
    top = copy.deepcopy(topology)
    base = list(base_indices)
    removable = [i for i in range(len(top["members"])) if i not in base]
    counts = {"shrink_all": 0, "shrink_random": 0, "remove_member": 0, "removal_skipped": 0}
    applied = 0

    def min_fos() -> float:
        try:
            truss = load_truss_from_problem(_spec_from(top, loading, goals))
            st = analyze_truss(truss, goals)
            fb, fy = float(st["fos_buckling"]), float(st["fos_yielding"])
            if not (math.isfinite(fb) and math.isfinite(fy)):
                return 0.0
            return min(fb, fy)
        except Exception:
            return 0.0

    for step in range(n_steps + 10):
        if min_fos() < FOS_MIN and step >= n_steps:
            break
        action = rng.choice(["shrink_all", "shrink_random", "remove_member"])

        if action == "remove_member" and removable:
            # V3: only remove members whose removal leaves a kinematically stable
            # truss.  The originals do not check this and 28/100 of them are
            # mechanisms whose FEA reports 1e13 m deflections.
            order = removable[:]
            rng.shuffle(order)
            chosen = None
            for cand in order:
                trial = {"joints": top["joints"],
                         "members": [m for i, m in enumerate(top["members"]) if i != cand]}
                ok, _ = stability(trial)
                if ok:
                    chosen = cand
                    break
            if chosen is None:
                counts["removal_skipped"] += 1
                action = "shrink_all"
            else:
                del top["members"][chosen]
                removable = [i if i < chosen else i - 1 for i in removable if i != chosen]
                base = [i if i < chosen else i - 1 for i in base]
                counts["remove_member"] += 1
                applied += 1
                if min_fos() < FOS_MIN and step >= n_steps - 1:
                    break
                continue

        if action == "shrink_all":
            factor = rng.uniform(0.75, 0.90)
            for m in top["members"]:
                m["shape"]["t"] *= factor
            counts["shrink_all"] += 1
        else:  # shrink_random
            if not top["members"]:
                continue
            i = rng.randrange(len(top["members"]))
            top["members"][i]["shape"]["t"] *= rng.uniform(0.60, 0.85)
            counts["shrink_random"] += 1
        applied += 1
        if min_fos() < FOS_MIN and step >= n_steps - 1:
            break

    return top, applied, counts



# ── 4b. NON-UNIFORM degradation: material MISALLOCATION ─────────────────────
#
# WHY THE OLD DEGRADATION MADE ONE-ACTION PROBLEMS.  ``degrade`` above picks
# uniformly among {shrink_all, shrink_random, remove_member} and stops the moment
# min FOS drops under 1.5.  ``shrink_all`` multiplies EVERY member's t by the
# same factor, so the posed design is the optimum with its relative proportions
# essentially intact -- correctly proportioned, uniformly under-sized.  The exact
# inverse of that is one global scale-up, and with a non-binding mass budget
# (see 3b) nothing punishes the overshoot on members that did not need it.
# Measured on the previous batch: lookahead_v2_d1 solved 250/250, median 2 steps.
#
# THE FIX.  Degrade by MISALLOCATING material, not by removing it uniformly:
#
#   * a random subset (``frac_down``) is SHRUNK, each member by its OWN factor
#     drawn independently -- so the repair factor differs per member and a single
#     global factor must be the max over them, over-building everything else;
#   * a disjoint random subset (``frac_up``) is INFLATED, each by its own factor
#     -- these members are already strong, so a global scale-up spends its whole
#     mass budget on material that was never needed, and a global scale-DOWN
#     (the other one-action escape) kills the members in the first subset;
#   * the parameter touched is drawn per member from {r}, {t}, {r,t}, so the
#     aspect ratio varies member to member and no single parameter sweep fixes
#     the set;
#   * optional structural edits -- remove a member (stability-checked), jitter an
#     interior joint -- which change the force paths so the posed proportions are
#     not merely a rescaling of ANY optimum of the posed structure.
#
# The repair therefore has to move material FROM the inflated members TO the
# shrunk ones under a budget that binds, which is a sizing-allocation decision,
# not a scalar.


def _interior_joint_ids(topology: dict) -> list[int]:
    return [j["id"] for j in topology["joints"] if not j.get("support")]


def degrade_misallocate(topology: dict, loading: list[dict], base_indices: list[int],
                        rng: random.Random, *, frac_down: float, frac_up: float,
                        down_range: tuple[float, float], up_range: tuple[float, float],
                        n_remove: int, joint_jitter: float,
                        deflection_bound: float) -> tuple[dict, dict]:
    """Misallocating degradation.  Returns (topology, record)."""
    top = copy.deepcopy(topology)
    base = list(base_indices)
    rec: dict[str, Any] = {"removed": 0, "jittered": 0, "n_down": 0, "n_up": 0}

    # -- structural edits first: they change what "optimal proportions" means ---
    for _ in range(n_remove):
        removable = [i for i in range(len(top["members"])) if i not in base]
        rng.shuffle(removable)
        chosen = None
        for cand in removable:
            trial = {"joints": top["joints"],
                     "members": [m for i, m in enumerate(top["members"]) if i != cand]}
            ok, _ = stability(trial)
            if ok:
                chosen = cand
                break
        if chosen is None:
            break
        del top["members"][chosen]
        base = [i if i < chosen else i - 1 for i in base]
        rec["removed"] += 1

    if joint_jitter > 0.0:
        ids = _interior_joint_ids(top)
        if ids:
            jid = rng.choice(ids)
            for j in top["joints"]:
                if j["id"] == jid:
                    before = list(j["position"])
                    j["position"][0] += rng.uniform(-joint_jitter, joint_jitter)
                    j["position"][1] += rng.uniform(-joint_jitter, joint_jitter)
                    ok, _ = stability(top)
                    if not ok:
                        j["position"] = before
                    else:
                        rec["jittered"] = 1
                    break

    # -- per-member misallocation -------------------------------------------
    n = len(top["members"])
    order = list(range(n))
    rng.shuffle(order)
    n_dn = max(1, int(round(frac_down * n)))
    n_up = int(round(frac_up * n))
    dn_ids = set(order[:n_dn])
    up_ids = set(order[n_dn:n_dn + n_up])
    rec["n_down"], rec["n_up"] = len(dn_ids), len(up_ids)
    rec["down_ids"], rec["up_ids"] = sorted(dn_ids), sorted(up_ids)

    factors: list[float] = []
    for i, m in enumerate(top["members"]):
        if i in dn_ids:
            f = rng.uniform(*down_range)
        elif i in up_ids:
            f = rng.uniform(*up_range)
        else:
            f = 1.0
        factors.append(f)
        if f == 1.0:
            continue
        which = rng.choices(("r", "t", "rt"), weights=(0.35, 0.30, 0.35))[0]
        sh = m["shape"]
        if which in ("r", "rt"):
            sh["r"] = min(max(sh["r"] * f, R_MIN), R_MAX)
        if which in ("t", "rt"):
            sh["t"] = min(max(sh["t"] * f, T_MIN), T_MAX)
        # a pipe with t > r is not a pipe; the executor would report a negative area
        sh["t"] = min(sh["t"], sh["r"] * 0.95)
        sh["t"] = min(max(sh["t"], T_MIN), T_MAX)
        sh["r"] = max(sh["r"], sh["t"] / 0.95)
    rec["factors"] = factors
    return top, rec


def _fos_and_mass(topology: dict, loading: list[dict]) -> tuple[float, float, float]:
    """(min FOS, mass, deflection) under the real simulator; (0, inf, inf) on failure."""
    from validation.truss_executor import analyze_truss, load_truss_from_problem
    goals = {"minimum_fos_buckling": FOS_MIN, "minimum_fos_yielding": FOS_MIN}
    try:
        truss = load_truss_from_problem(_spec_from(topology, loading, goals))
        st = analyze_truss(truss, goals)
        fb, fy = float(st["fos_buckling"]), float(st["fos_yielding"])
        m, d = float(st["mass"]), float(st["deflection"])
        if not all(math.isfinite(v) for v in (fb, fy, m, d)):
            return 0.0, float("inf"), float("inf")
        return min(fb, fy), m, d
    except Exception:
        return 0.0, float("inf"), float("inf")


# ── difficulty strata ───────────────────────────────────────────────────────
# Each stratum is a knob setting, NOT a difficulty claim: the label written into
# _metadata.difficulty is assigned afterwards from MEASURED policy solve rates by
# scripts/label_difficulty.py.  These names are only the generator's intent.
STRATA: dict[str, dict] = {
    "s0": dict(mass_slack=2.80, frac_down=0.30, frac_up=0.20, down=(0.60, 0.90),
               up=(1.10, 1.45), n_remove=0, jitter=0.0),
    "s1": dict(mass_slack=2.10, frac_down=0.35, frac_up=0.25, down=(0.55, 0.85),
               up=(1.15, 1.60), n_remove=0, jitter=0.0),
    "s2": dict(mass_slack=1.70, frac_down=0.45, frac_up=0.30, down=(0.45, 0.80),
               up=(1.20, 1.80), n_remove=0, jitter=0.3),
    "s2b": dict(mass_slack=1.55, frac_down=0.45, frac_up=0.32, down=(0.42, 0.78),
                up=(1.22, 1.90), n_remove=1, jitter=0.3),
    "s3": dict(mass_slack=1.40, frac_down=0.50, frac_up=0.35, down=(0.40, 0.75),
               up=(1.25, 2.00), n_remove=1, jitter=0.3),
    "s4": dict(mass_slack=1.20, frac_down=0.55, frac_up=0.40, down=(0.35, 0.70),
               up=(1.30, 2.20), n_remove=1, jitter=0.5),
    "s5": dict(mass_slack=1.05, frac_down=0.60, frac_up=0.45, down=(0.30, 0.65),
               up=(1.40, 2.50), n_remove=1, jitter=0.5),
}


# ── 5. one problem, end to end ───────────────────────────────────────────────

REJECT_REASONS = [
    "lp_infeasible",
    "ground_structure_no_optimum",
    "unstable_topology",
    "no_certified_optimum",
    "fea_nonfinite",
    "initial_feasible",
    "degenerate_deflection",
    "params_out_of_bounds",
    "no_policy_solves",
    "certificate_over_budget",
    "exception",
]


def generate_one(task: tuple) -> dict:
    """Returns a record: {'ok': bool, 'reason': str, 'spec': dict|None, ...}."""
    seed, index, prefix, policy_gate, policy_steps, budget_ratio = task
    rng = random.Random(seed)
    rec: dict[str, Any] = {"index": index, "seed": seed, "ok": False,
                           "reason": "exception", "budget_ratio": budget_ratio}
    try:
        span = rng.uniform(*SPAN_RANGE)
        height = rng.uniform(*HEIGHT_RANGE)
        num_bays = rng.choice(BAYS_CHOICES)
        material = rng.choice(MATERIALS)
        rec.update(span=span, height=height, num_bays=num_bays, material=material)

        topology, base = build_ground_structure(span, height, num_bays, material)
        loading = sample_loading(rng, topology)
        rec["n_loads"] = len(loading)
        deflection_bound = DEFLECTION_BOUND_FRAC * max(span, height)

        # --- step 2+3 on the ground structure (realistic member proportions) ---
        gs_opt = certified_optimum(topology, loading, deflection_bound)
        if gs_opt is None:
            rec["reason"] = "ground_structure_no_optimum"
            return rec
        rec["gs_optimal_mass"] = gs_opt["mass"]
        rec["gs_lp_mass"] = gs_opt.get("lp_mass")
        for m, sh in zip(topology["members"], gs_opt["shapes"]):
            m["shape"] = dict(sh)

        # --- step 4: degrade ---
        n_steps = rng.randint(*DEGRADATION_STEPS_RANGE)
        degraded, applied, counts = degrade(topology, loading, base, n_steps, rng)
        rec.update(degradation_steps=applied, degradation_counts=counts,
                   n_members=len(degraded["members"]),
                   n_joints=len(degraded["joints"]))

        # --- V3(a): exact mechanism test on the posed topology ---
        stable, cond = stability(degraded)
        rec["stability_ratio"] = cond
        if not stable:
            rec["reason"] = "unstable_topology"
            return rec

        # --- step 5: certified optimum ON THE POSED TOPOLOGY (V4) ---
        opt = certified_optimum(degraded, loading, deflection_bound)
        if opt is None:
            rec["reason"] = "no_certified_optimum"
            return rec
        optimal_mass = opt["mass"]
        rec["optimal_mass"] = optimal_mass
        rec["lp_optimal_mass"] = opt.get("lp_mass")
        rec["optimum_state"] = opt["state"]

        goals = {
            "minimum_fos_buckling": FOS_MIN,
            "minimum_fos_yielding": FOS_MIN,
            "maximum_mass": optimal_mass * budget_ratio,
        }
        spec = {
            "problem_id": f"{prefix}_{index:04d}",
            "description": f"Truss optimization problem (ground structure variant {index})",
            "topology": degraded,
            "loading": loading,
            "goals": goals,
            "optimization": copy.deepcopy(OPTIMIZATION_BLOCK),
            "_metadata": {
                "optimal_mass": optimal_mass,
                "generation_method": "ground_structure_lp",
                "degradation_steps": applied,
            },
        }

        # --- V1: it loads and the simulator returns finite numbers ---
        from validation.truss_executor import analyze_truss as _raw_analyze
        truss, g = _load_truss_and_goals(spec)
        raw = _raw_analyze(truss, g)
        state = _analyze_truss(truss, g)
        vals = [raw.get(k) for k in ("mass", "fos_buckling", "fos_yielding", "deflection")]
        if any(v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v))
               for v in vals):
            rec["reason"] = "fea_nonfinite"
            return rec
        rec["initial_state"] = {k: float(raw[k]) for k in
                                ("mass", "fos_buckling", "fos_yielding", "deflection")}
        rec["initial_mass"] = float(raw["mass"])

        # --- V2: the start state must be infeasible ---
        if bool(state.get("is_feasible", False)):
            rec["reason"] = "initial_feasible"
            return rec

        # --- V3(b): deflection within a sane physical bound ---
        if float(raw["deflection"]) > deflection_bound or \
                float(opt["state"]["deflection"]) > deflection_bound:
            rec["reason"] = "degenerate_deflection"
            return rec

        # --- V4: the certified optimum is inside the declared action bounds ---
        bad = [sh for sh in opt["shapes"]
               if not (R_MIN - 1e-12 <= sh["r"] <= R_MAX + 1e-12
                       and T_MIN - 1e-12 <= sh["t"] <= T_MAX + 1e-12)]
        if bad:
            rec["reason"] = "params_out_of_bounds"
            return rec
        rec["optimal_shapes"] = opt["shapes"]

        # --- V5: a search policy actually reaches feasibility ---
        # The probe runs for EVERY candidate that got this far, whether or not it
        # gates acceptance, so the manifest records the unbiased solve rate as
        # well as the gated one.
        if policy_steps > 0:
            solved = policy_probe(spec, max_steps=policy_steps)
            rec["policy_solved"] = solved
            if policy_gate == "any" and not any(solved.values()):
                rec["reason"] = "no_policy_solves"
                return rec

        rec["ok"] = True
        rec["reason"] = "accepted"
        rec["spec"] = spec
        return rec
    except Exception as exc:  # keep the sweep alive
        rec["reason"] = "exception"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


def generate_one_hard(task: tuple) -> dict:
    """One problem under the misallocation scheme + free-aspect mass reference.

    Differences from ``generate_one`` (the legacy path, kept byte-identical):
      * the posed design is the fixed-aspect optimum with material MISALLOCATED
        (some members shrunk, others inflated, each by its own factor), not
        uniformly thinned -- so no single global factor repairs it;
      * the mass budget is referenced to the FREE-ASPECT certified optimum of the
        POSED topology, so it actually binds;
      * ``mass_slack`` (per stratum) is the difficulty dial and is recorded.
    """
    (seed, index, prefix, stratum_name, aspect_min, policy_steps) = task
    rng = random.Random(seed)
    S = STRATA[stratum_name]
    rec: dict[str, Any] = {"index": index, "seed": seed, "ok": False,
                           "reason": "exception", "stratum": stratum_name,
                           "mass_slack": S["mass_slack"], "aspect_min": aspect_min}
    try:
        span = rng.uniform(*SPAN_RANGE)
        height = rng.uniform(*HEIGHT_RANGE)
        num_bays = rng.choice(BAYS_CHOICES)
        material = rng.choice(MATERIALS)
        rec.update(span=span, height=height, num_bays=num_bays, material=material)

        topology, base = build_ground_structure(span, height, num_bays, material)
        loading = sample_loading(rng, topology)
        rec["n_loads"] = len(loading)
        deflection_bound = DEFLECTION_BOUND_FRAC * max(span, height)

        # --- realistic starting proportions: the fixed-aspect optimum ---------
        gs_opt = certified_optimum(topology, loading, deflection_bound)
        if gs_opt is None:
            rec["reason"] = "ground_structure_no_optimum"
            return rec
        rec["gs_optimal_mass"] = gs_opt["mass"]
        for m, sh in zip(topology["members"], gs_opt["shapes"]):
            m["shape"] = dict(sh)

        # --- misallocating degradation, intensified until infeasible ---------
        degraded = None
        deg_rec: dict[str, Any] = {}
        lo0, hi0 = S["down"]
        for attempt in range(6):
            k = 0.85 ** attempt          # each retry shrinks the shrunk set harder
            cand, dr = degrade_misallocate(
                topology, loading, base, rng,
                frac_down=S["frac_down"], frac_up=S["frac_up"],
                down_range=(max(0.12, lo0 * k), max(0.15, hi0 * k)),
                up_range=S["up"], n_remove=S["n_remove"],
                joint_jitter=S["jitter"], deflection_bound=deflection_bound)
            ok_stable, cond = stability(cand)
            if not ok_stable:
                continue
            min_fos, mass, defl = _fos_and_mass(cand, loading)
            if not math.isfinite(defl) or defl > deflection_bound:
                continue
            degraded, deg_rec = cand, dr
            deg_rec["attempt"] = attempt
            deg_rec["min_fos"] = min_fos
            deg_rec["degraded_mass"] = mass
            rec["stability_ratio"] = cond
            if min_fos < FOS_MIN:            # FOS already violated -> done
                break
        if degraded is None:
            rec["reason"] = "unstable_topology"
            return rec
        rec["degradation"] = deg_rec
        rec["n_members"] = len(degraded["members"])
        rec["n_joints"] = len(degraded["joints"])

        # --- mass reference on the POSED topology -----------------------------
        fixed_opt = certified_optimum(degraded, loading, deflection_bound)
        seeds = [fixed_opt["shapes"]] if fixed_opt else []
        free_opt = certified_optimum_free(degraded, loading, deflection_bound,
                                          aspect_min, seeds=seeds)
        if free_opt is None:
            rec["reason"] = "no_certified_optimum"
            return rec
        certified_mass = float(free_opt["mass"])
        rec["certified_mass"] = certified_mass
        rec["fixed_aspect_mass"] = fixed_opt["mass"] if fixed_opt else None
        rec["optimum_state"] = free_opt["state"]

        optimal_mass = certified_mass * float(S["mass_slack"])
        goals = {
            "minimum_fos_buckling": FOS_MIN,
            "minimum_fos_yielding": FOS_MIN,
            "maximum_mass": optimal_mass * MASS_BUDGET_RATIO,
        }
        rec["optimal_mass"] = optimal_mass

        spec = {
            "problem_id": f"{prefix}_{index:04d}",
            "description": f"Truss optimization problem (misallocated ground structure {index})",
            "topology": degraded,
            "loading": loading,
            "goals": goals,
            "optimization": copy.deepcopy(OPTIMIZATION_BLOCK),
            "_metadata": {
                "optimal_mass": optimal_mass,
                "generation_method": "ground_structure_lp_misallocated",
                "degradation_steps": int(deg_rec.get("n_down", 0) + deg_rec.get("n_up", 0)
                                         + deg_rec.get("removed", 0) + deg_rec.get("jittered", 0)),
                # audit trail for the mass budget
                "certified_mass": certified_mass,
                "certified_aspect_min": aspect_min,
                "fixed_aspect_mass": rec["fixed_aspect_mass"],
                "mass_slack": float(S["mass_slack"]),
                "stratum": stratum_name,
                "degradation": {k: v for k, v in deg_rec.items() if k != "factors"},
                # difficulty is written by scripts/label_difficulty.py from MEASURED
                # policy solve rates; the stratum above is only generation intent.
                "difficulty": None,
            },
        }

        # --- V1: loads, finite under the RAW simulator ------------------------
        from validation.truss_executor import analyze_truss as _raw_analyze
        truss, g = _load_truss_and_goals(spec)
        raw = _raw_analyze(truss, g)
        state = _analyze_truss(truss, g)
        vals = [raw.get(k) for k in ("mass", "fos_buckling", "fos_yielding", "deflection")]
        if any(v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v))
               for v in vals):
            rec["reason"] = "fea_nonfinite"
            return rec
        rec["initial_state"] = {k: float(raw[k]) for k in
                                ("mass", "fos_buckling", "fos_yielding", "deflection")}
        rec["initial_mass"] = float(raw["mass"])
        rec["initial_mass_over_budget"] = float(raw["mass"]) / goals["maximum_mass"]

        # --- V2: the posed design must be infeasible --------------------------
        program = program_from_truss_spec(spec, initial_mass=float(raw["mass"]))
        if program.is_feasible(state):
            rec["reason"] = "initial_feasible"
            return rec

        # --- V3(b): sane deflection, posed AND at the certified optimum -------
        if float(raw["deflection"]) > deflection_bound or \
                float(free_opt["state"]["deflection"]) > deflection_bound:
            rec["reason"] = "degenerate_deflection"
            return rec

        # --- V4: the certificate is inside the declared bounds AND under budget
        bad = [sh for sh in free_opt["shapes"]
               if not (R_MIN - 1e-12 <= sh["r"] <= R_MAX + 1e-12
                       and T_MIN - 1e-12 <= sh["t"] <= T_MAX + 1e-12
                       and sh["t"] <= sh["r"] + 1e-12)]
        if bad:
            rec["reason"] = "params_out_of_bounds"
            return rec
        if certified_mass > goals["maximum_mass"] + 1e-9:
            rec["reason"] = "certificate_over_budget"
            return rec
        rec["optimal_shapes"] = free_opt["shapes"]

        if policy_steps > 0:
            rec["policy_solved"] = policy_probe(spec, max_steps=policy_steps)

        rec["ok"] = True
        rec["reason"] = "accepted"
        rec["spec"] = spec
        return rec
    except Exception as exc:
        rec["reason"] = "exception"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


def policy_probe(spec: dict, max_steps: int = 20) -> dict[str, bool]:
    """Run the repo's own search policies (scripts/search_ladder.py) on a spec.

    Cheap policies first; the expensive potential-guided lookahead only runs if
    none of them succeeded, because it costs ~200 simulator calls per step.
    """
    import search_ladder as SL

    goals = spec.get("goals", {}) or {}
    bounds = SL.param_bounds(spec)
    base = SL.make_truss(spec)
    s0 = SL.fea(base, goals)
    cons = SL.program_from_truss_spec(spec, initial_mass=s0["mass"])

    out: dict[str, bool] = {}

    def _run(name, fn):
        try:
            _, sf, *_ = fn(copy.deepcopy(base))
            out[name] = bool(cons.is_feasible(sf))
        except Exception:
            out[name] = False

    _run("greedy_critical", lambda t: SL.run_greedy_critical(
        t, spec, goals, cons, max_steps, alpha=5.0, bounds=bounds))
    _run("greedy_fsd", lambda t: SL.run_greedy_fsd(t, spec, goals, cons, max_steps, bounds))
    _run("random", lambda t: SL.run_random(
        t, spec, goals, cons, max_steps, 5.0, bounds,
        seed=abs(hash(spec.get("problem_id", ""))) % 10000))
    if not any(out.values()):
        _run("lookahead_v2_d1", lambda t: SL.run_lookahead(
            t, spec, goals, cons, max_steps, "v2", 5.0, bounds, depth=1))
    return out


# ── driver ───────────────────────────────────────────────────────────────────

def existing_ids(*dirs: Path) -> set[str]:
    ids: set[str] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            try:
                spec = json.load(open(p))
            except Exception:
                continue
            if isinstance(spec, dict) and "problem_id" in spec:
                ids.add(str(spec["problem_id"]))
            ids.add(p.stem)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="number of ACCEPTED problems to write")
    ap.add_argument("--out", default=str(DESIGNBENCH / "data/problems_gen"))
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prefix", default="gen_problem")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="cap on candidates tried (default: 4x --n)")
    ap.add_argument("--require-policy-solve", choices=("none", "any"), default="any",
                    help="V5 gate: require >=1 search policy to reach feasibility")
    ap.add_argument("--policy-steps", type=int, default=20,
                    help="step budget for the V5 policy probe; 0 disables it")
    ap.add_argument("--budget-ratio", type=float, default=MASS_BUDGET_RATIO,
                    help="goals.maximum_mass / _metadata.optimal_mass. 1.1 is the "
                         "ratio every original uses. Because this generator's "
                         "optimal_mass is a genuinely attainable optimum while the "
                         "originals' is a measured 1.71x above theirs, 1.1 here is a "
                         "TIGHTER budget than 1.1 there; see docs/problem_generation.md "
                         "for the measured calibration curve.")
    ap.add_argument("--mode", choices=("legacy", "hard"), default="legacy",
                    help="legacy = the uniform-thinning scheme that produced "
                         "problems_gen (kept byte-identical for reproducibility); "
                         "hard = misallocating degradation + free-aspect mass "
                         "reference (see docs/problem_generation.md)")
    ap.add_argument("--strata", default="s1:1,s2:1,s3:1,s4:1,s5:1",
                    help="hard mode: comma list NAME:WEIGHT over STRATA")
    ap.add_argument("--aspect-min", type=float, default=1.0 / 25.0,
                    help="hard mode: floor on t/r when computing the certified "
                         "mass reference. 1/6 is the originals' fixed aspect and "
                         "makes the budget vacuous; 1/150 is the bound the "
                         "declared shape_params allow.")
    ap.add_argument("--manifest", default=None,
                    help="path for the per-candidate audit record "
                         "(default: <out>/manifest.jsonl)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.jsonl"
    max_attempts = args.max_attempts or max(4 * args.n, args.n + 50)

    taken = existing_ids(DESIGNBENCH / "data/problems", out_dir)
    print(f"{len(taken)} existing ids reserved (no collisions allowed)", flush=True)
    print(f"budget ratio maximum_mass/optimal_mass = {args.budget_ratio}", flush=True)

    rng = random.Random(args.seed)
    if args.mode == "hard":
        weights: list[tuple[str, float]] = []
        for part in args.strata.split(","):
            part = part.strip()
            if not part:
                continue
            name, _, w = part.partition(":")
            if name not in STRATA:
                raise SystemExit(f"unknown stratum {name!r}; known: {sorted(STRATA)}")
            weights.append((name, float(w or 1.0)))
        names = [n for n, _ in weights]
        wts = [w for _, w in weights]
        # Deterministic round-robin over the weighted strata so the accepted set
        # is balanced even when acceptance rates differ between strata.
        order: list[str] = []
        total = sum(wts)
        for i in range(max_attempts):
            x = ((i + 0.5) / max_attempts) * total
            acc = 0.0
            pick = names[-1]
            for n, w in weights:
                acc += w
                if x <= acc:
                    pick = n
                    break
            order.append(pick)
        rng.shuffle(order)
        print(f"strata mix: {dict((n, order.count(n)) for n in names)}", flush=True)
        worker = generate_one_hard
        tasks = [(rng.randrange(2 ** 31), i, args.prefix, order[i], args.aspect_min,
                  args.policy_steps) for i in range(max_attempts)]
    else:
        worker = generate_one
        tasks = [(rng.randrange(2 ** 31), i, args.prefix, args.require_policy_solve,
                  args.policy_steps, args.budget_ratio) for i in range(max_attempts)]

    t0 = time.time()
    accepted = 0
    reasons: dict[str, int] = {}
    records: list[dict] = []

    def consume(rec: dict) -> bool:
        """Write an accepted spec; returns True once --n have been written."""
        nonlocal accepted
        reasons[rec["reason"]] = reasons.get(rec["reason"], 0) + 1
        spec = rec.pop("spec", None)
        if rec["ok"] and spec is not None and accepted < args.n:
            pid = f"{args.prefix}_{accepted:04d}"
            if pid in taken:
                rec["ok"] = False
                rec["reason"] = "id_collision"
                reasons["id_collision"] = reasons.get("id_collision", 0) + 1
                records.append(rec)
                return False
            spec["problem_id"] = pid
            spec["description"] = (
                f"Truss optimization problem (ground structure variant {accepted})")
            with open(out_dir / f"{pid}.json", "w") as fh:
                json.dump(spec, fh, indent=2)
            taken.add(pid)
            rec["problem_id"] = pid
            accepted += 1
        records.append(rec)
        return accepted >= args.n

    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(worker, tasks), 1):
                done = consume(rec)
                if i % 25 == 0:
                    print(f"  tried {i}/{len(tasks)}  accepted {accepted}/{args.n}"
                          f"  ({time.time()-t0:.0f}s)", flush=True)
                if done:
                    pool.terminate()
                    break
    else:
        for i, task in enumerate(tasks, 1):
            done = consume(worker(task))
            if i % 10 == 0:
                print(f"  tried {i}  accepted {accepted}  ({time.time()-t0:.0f}s)", flush=True)
            if done:
                break

    with open(manifest_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, default=float) + "\n")

    print("\n" + "=" * 68)
    print(f"accepted {accepted} problems into {out_dir}")
    print(f"candidates tried: {len(records)}   wall: {time.time()-t0:.0f}s")
    print("outcome counts:")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   {k:28s} {v}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
