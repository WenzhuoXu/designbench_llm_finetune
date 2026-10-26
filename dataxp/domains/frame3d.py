"""3D pin-jointed truss sizing -- the size-ladder extension of the existing 2D
truss domain, with trusted physics (direct stiffness) and a fully-stressed
design reference.

Geometry is a chain of cross-sections along an axis. Each new section's nodes
are each connected to THREE nodes of the previous section, which is a Henneberg
3-node attachment: the structure is rigid by construction, so generated
problems are never mechanisms. In-section edges are added for realism and
redundancy.

  tower       4-node square section, vertical axis, base section fixed,
              lateral (wind) load at the top plus vertical dead load.
  cantilever  4-node square section, horizontal axis, root section fixed,
              vertical load at the tip.
  bridge      3-node triangular girder, horizontal axis, BOTH end sections
              supported, vertical load on the interior sections.

Members are aluminium tubes of radius r (the design variable) and wall
t = 0.2 r, so the cross-section can never invert -- the negative-area exploit
that the 2D study found (D9) is structurally impossible here. Deflection is
still bounded by the envelope because a thinned truss can still approach a
mechanism numerically.

  A = pi (r^2 - (r-t)^2) = 0.36 pi r^2
  I = (pi/4)(r^4 - (r-t)^4) = 0.1476 pi r^4

FOS_yielding = sigma_y / |N/A| over all members; FOS_buckling = P_cr/|N| over
COMPRESSION members only, with P_cr = pi^2 E I / L^2.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import numpy as np

from ..domain import Domain
from ..potential import DesignProgram, program_from_goals

# ── material / section (6061-T6 aluminium) ───────────────────────────────────
E_MOD = 68.9e9                  # Pa
SIGMA_Y = 276e6                 # Pa
RHO = 2700.0                    # kg/m3
WALL_RATIO = 0.2                # t = WALL_RATIO * r
A_COEF = math.pi * (1.0 - (1.0 - WALL_RATIO) ** 2)          # A = A_COEF r^2
I_COEF = 0.25 * math.pi * (1.0 - (1.0 - WALL_RATIO) ** 4)   # I = I_COEF r^4

R_MIN, R_MAX = 0.004, 0.12      # m
FOS_TARGET = 1.5

_FAMILIES = ("tower", "cantilever", "bridge")
_BAY_LADDER = (2, 3, 4, 5, 6, 8)

_HINT_NONE = "none"


def _failure_metrics() -> dict:
    """Maximally infeasible readout for a singular/failed solve."""
    return {
        "mass": float("inf"),
        "fos_buckling": 0.0,
        "fos_yielding": 0.0,
        "max_deflection_m": float("inf"),
    }


class Frame3DDomain(Domain):
    """3D truss sizing (variants: frame3d, frame3d_stoch)."""

    name = "frame3d"
    params = ("r",)
    factors = (0.85, 0.93, 1.08, 1.2, 1.35)
    reference_class = "heuristic"

    def __init__(self, stochastic: bool = False):
        super().__init__(stochastic)

    # ── geometry ──────────────────────────────────────────────────────────────
    @staticmethod
    def _build(family: str, n_bay: int, width: float, pitch: float):
        """Return (nodes[N,3], members[M,2], fixed_node_indices)."""
        if family == "bridge":
            # triangular girder cross-section
            h = width * 0.85
            section = [(0.0, -0.5 * width, 0.0),
                       (0.0, 0.5 * width, 0.0),
                       (0.0, 0.0, h)]
            axis = 0                      # extrude along x
        elif family == "cantilever":
            section = [(0.0, -0.5 * width, -0.5 * width),
                       (0.0, 0.5 * width, -0.5 * width),
                       (0.0, 0.5 * width, 0.5 * width),
                       (0.0, -0.5 * width, 0.5 * width)]
            axis = 0
        else:                             # tower: extrude along z
            section = [(-0.5 * width, -0.5 * width, 0.0),
                       (0.5 * width, -0.5 * width, 0.0),
                       (0.5 * width, 0.5 * width, 0.0),
                       (-0.5 * width, 0.5 * width, 0.0)]
            axis = 2

        k = len(section)
        nodes, members = [], []
        for b in range(n_bay + 1):
            off = b * pitch
            for (x, y, z) in section:
                p = [x, y, z]
                p[axis] += off
                nodes.append(p)
            base = b * k
            # in-section ring
            for j in range(k):
                members.append((base + j, base + (j + 1) % k))
            if k == 4:                    # one face diagonal stiffens the square
                members.append((base, base + 2))
            if b > 0:
                prev = (b - 1) * k
                # Henneberg: every new node attaches to THREE previous nodes
                for j in range(k):
                    members.append((prev + j, base + j))
                    members.append((prev + (j + 1) % k, base + j))
                    members.append((prev + (j - 1) % k, base + j))
        if family == "bridge":
            fixed = list(range(k)) + list(range(n_bay * k, (n_bay + 1) * k))
        else:
            fixed = list(range(k))
        # de-duplicate members (a 3-node section's ring and diagonals can repeat)
        seen, uniq = set(), []
        for a, b_ in members:
            key = (min(a, b_), max(a, b_))
            if key[0] != key[1] and key not in seen:
                seen.add(key)
                uniq.append(key)
        return (np.asarray(nodes, dtype=float),
                np.asarray(uniq, dtype=np.int64),
                sorted(set(fixed)))

    # ── problems ──────────────────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        n_bay = int(_BAY_LADDER[(index // len(_FAMILIES)) % len(_BAY_LADDER)])
        if family == "bridge":
            n_bay = max(n_bay, 4)          # both ends are supported
        width = float(rng.uniform(0.8, 1.6))
        pitch = float(rng.uniform(0.9, 1.8))
        load = float(rng.uniform(4.0e3, 2.6e4))     # N
        spec = {
            "problem_id": "frame3d_%s_%04d" % (family, index),
            "family": family,
            "n_bay": n_bay,
            "width": round(width, 4),
            "pitch": round(pitch, 4),
            "load_n": round(load, 2),
            # How the initial design is degraded. Fully-stressed design sizes
            # EVERY member to be exactly critical, so scaling all of them down
            # uniformly would leave all M members violating - unreachable with
            # a 20-turn budget of single-member moves (the action grammar, not
            # the estimator, would be what binds). Instead a small SUBSET is
            # under-built, which is what a degraded real design looks like and
            # what makes the critical-member diagnostic meaningful.
            "start_frac": round(float(rng.uniform(0.62, 0.86)), 4),
            "goals": {
                "minimum_fos_buckling": FOS_TARGET,
                "minimum_fos_yielding": FOS_TARGET,
            },
        }
        ref = self.reference(spec)         # FSD mass; caches into spec
        n_mem = len(spec.get("_ref_radii") or [])
        n_deg = int(min(max(3, round(0.09 * n_mem)), 5))
        if n_mem:
            idx = rng.choice(n_mem, size=min(n_deg, n_mem), replace=False)
            order = sorted(int(v) for v in idx)
            # STAGGERED severities. fos_buckling is a MIN over members, so if
            # every degraded member is equally under-built the minimum barely
            # responds until all of them are repaired - a plateau on which Phi
            # can only improve by shedding mass, which is what it then does
            # (observed: FOS flat at 0.52 while mass fell 15.0 -> 13.1). Giving
            # each degraded member its own severity makes repairing the worst
            # one raise the minimum to the next, so every correct move pays.
            sev = np.linspace(float(spec["start_frac"]), 0.93, len(order))
            spec["degraded"] = order
            spec["degraded_frac"] = [round(float(v), 4) for v in sev]
        else:
            spec["degraded"] = []
            spec["degraded_frac"] = []
        # headroom over FSD: the sound members sit just above critical
        spec["sound_frac"] = 1.04
        spec["goals"]["maximum_mass"] = round(float(ref) * 1.25, 6)
        return spec

    # ── state ─────────────────────────────────────────────────────────────────
    def load(self, spec: dict) -> dict:
        nodes, members, fixed = self._build(
            str(spec["family"]), int(spec["n_bay"]),
            float(spec["width"]), float(spec["pitch"]))
        n_node = nodes.shape[0]
        vec = nodes[members[:, 1]] - nodes[members[:, 0]]
        length = np.linalg.norm(vec, axis=1)
        length = np.maximum(length, 1e-9)
        cos = vec / length[:, None]                     # direction cosines

        free = np.ones(3 * n_node, dtype=bool)
        for nd in fixed:
            free[3 * nd:3 * nd + 3] = False

        # loads: family-appropriate, applied at the far / interior sections
        P = np.zeros(3 * n_node, dtype=float)
        fam = str(spec["family"])
        k = 3 if fam == "bridge" else 4
        total = float(spec["load_n"])
        if fam == "tower":
            top = range(n_node - k, n_node)
            for nd in top:                              # lateral wind + dead
                P[3 * nd] += total / k
                P[3 * nd + 2] -= 0.35 * total / k
        elif fam == "cantilever":
            tip = range(n_node - k, n_node)
            for nd in tip:
                P[3 * nd + 2] -= total / k
        else:                                           # bridge: interior decks
            interior = [nd for nd in range(k, n_node - k)]
            if not interior:
                interior = list(range(k, n_node))
            for nd in interior:
                P[3 * nd + 2] -= total / len(interior)

        span = float(np.ptp(nodes, axis=0).max())
        st = {
            "spec": spec,
            "r": np.full(members.shape[0], R_MIN * 2.0, dtype=float),
            "_geo": {
                "nodes": nodes, "members": members, "length": length,
                "cos": cos, "free": free, "P": P, "n_node": n_node,
                "span": max(span, 1e-6),
            },
            "cache": None,
            "_hint": None,
        }
        st["r"] = self._start_radii(spec, st)
        return st

    def clone(self, st) -> dict:
        return {
            "spec": st["spec"],
            "r": np.array(st["r"], dtype=float, copy=True),
            "_geo": st["_geo"],            # immutable geometry, shared
            "cache": None,
            "_hint": None,
        }

    # ── analysis ──────────────────────────────────────────────────────────────
    def _solve(self, geo: dict, r: np.ndarray, load_scale: float = 1.0,
               load_rot: float = 0.0):
        """Direct stiffness solve. Returns (N, u) or None if singular."""
        members, length, cos = geo["members"], geo["length"], geo["cos"]
        n_dof = 3 * geo["n_node"]
        area = A_COEF * r * r
        ea_l = E_MOD * area / length

        K = np.zeros((n_dof, n_dof), dtype=float)
        for m in range(members.shape[0]):
            i, j = int(members[m, 0]), int(members[m, 1])
            c = cos[m]
            T = np.outer(c, c) * ea_l[m]
            si, sj = 3 * i, 3 * j
            K[si:si + 3, si:si + 3] += T
            K[sj:sj + 3, sj:sj + 3] += T
            K[si:si + 3, sj:sj + 3] -= T
            K[sj:sj + 3, si:si + 3] -= T

        P = geo["P"] * load_scale
        if load_rot != 0.0:                # small in-plane jitter of the load
            ca, sa = math.cos(load_rot), math.sin(load_rot)
            Pv = P.reshape(-1, 3).copy()
            x, y = Pv[:, 0].copy(), Pv[:, 1].copy()
            Pv[:, 0] = ca * x - sa * y
            Pv[:, 1] = sa * x + ca * y
            P = Pv.reshape(-1)

        free = geo["free"]
        Kff = K[np.ix_(free, free)]
        try:
            uf = np.linalg.solve(Kff, P[free])
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(uf)):
            return None
        u = np.zeros(n_dof, dtype=float)
        u[free] = uf
        du = u.reshape(-1, 3)[members[:, 1]] - u.reshape(-1, 3)[members[:, 0]]
        N = E_MOD * area / length * np.sum(cos * du, axis=1)   # tension +ve
        return N, u

    def evaluate(self, st, seed: int = 0) -> dict:
        if st.get("cache") is not None:
            return st["cache"]
        geo = st["_geo"]
        r = np.clip(np.asarray(st["r"], dtype=float), R_MIN, R_MAX)
        scale, rot = 1.0, 0.0
        if self.stochastic:
            rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
            # FOS is a MINIMUM over up to 141 members, so load scatter reaches
            # Phi through an extreme-value statistic and is amplified. At
            # sigma = 0.12 the noise band exceeded the Phi gap at most states
            # even at the maximum replicate budget. 0.08 is also the more
            # honest number: structural dead/live load factors have a COV
            # around 7-10%, not 12% with 3 degrees of angular scatter on top.
            sigma = 0.08
            scale = float(rng.lognormal(0.0, sigma)
                          / math.exp(0.5 * sigma * sigma))
            rot = float(rng.normal(0.0, 0.03))
        try:
            out = self._solve(geo, r, scale, rot)
            if out is None:
                st["cache"] = _failure_metrics()
                return st["cache"]
            N, u = out
            area = A_COEF * r * r
            inertia = I_COEF * r ** 4
            mass = float(np.sum(RHO * area * geo["length"]))

            stress = np.abs(N) / np.maximum(area, 1e-12)
            fos_y = float(np.min(SIGMA_Y / np.maximum(stress, 1e-9)))

            comp = N < -1e-9
            if np.any(comp):
                p_cr = (math.pi ** 2) * E_MOD * inertia[comp] / (geo["length"][comp] ** 2)
                fos_b = float(np.min(p_cr / np.abs(N[comp])))
            else:
                fos_b = float(1e6)         # nothing in compression

            defl = float(np.max(np.abs(u)))
            metrics = {
                "mass": mass,
                "fos_buckling": min(fos_b, 1e6),
                "fos_yielding": min(fos_y, 1e6),
                "max_deflection_m": defl,
            }
            if not all(math.isfinite(v) for v in metrics.values()):
                metrics = _failure_metrics()
        except Exception:
            metrics = _failure_metrics()
        st["cache"] = metrics
        return metrics

    # ── design variables ──────────────────────────────────────────────────────
    def n_elements(self, st) -> int:
        return int(st["_geo"]["members"].shape[0])

    def get(self, st, i: int, p: str) -> Optional[float]:
        if p != "r" or not (0 <= int(i) < self.n_elements(st)):
            return None
        return float(st["r"][int(i)])

    def set(self, st, i: int, p: str, v: float) -> None:
        if p != "r" or not (0 <= int(i) < self.n_elements(st)):
            return
        st["r"][int(i)] = min(R_MAX, max(R_MIN, float(v)))
        st["cache"] = None
        st["_hint"] = None

    def bounds(self, st, p: str) -> Tuple[float, float]:
        return (R_MIN, R_MAX)

    # ── program / reference ───────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        ref = self.reference(spec)
        prog = program_from_goals(
            spec.get("goals", {}),
            state_keys=("mass", "fos_buckling", "fos_yielding",
                        "max_deflection_m"),
            objective_key="mass",
            objective_ref=ref,
            objective_sense="min",
            domain=self.name,
        )
        # Envelope: a truss deflecting by a tenth of its own span is a
        # mechanism, not a design, whatever the goals say -- the 2D study
        # scored such states FEASIBLE on 17% of its "successes" (D9).
        nodes, _, _ = self._build(str(spec["family"]), int(spec["n_bay"]),
                                  float(spec["width"]), float(spec["pitch"]))
        span = float(np.ptp(nodes, axis=0).max())
        return dataclasses.replace(prog, validity_bounds=(
            ("max_deflection_m", 0.1 * max(span, 1e-6)),
        ))

    def _required_radii(self, geo: dict, r: np.ndarray) -> Optional[np.ndarray]:
        """Radii at which every member would exactly meet both FOS targets
        under the forces of the CURRENT design."""
        out = self._solve(geo, r)
        if out is None:
            return None
        N, _ = out
        absN = np.abs(N)
        # yielding: A >= FOS |N| / sigma_y
        r_y = np.sqrt(np.maximum(FOS_TARGET * absN / SIGMA_Y, 0.0) / A_COEF)
        # buckling (compression only): I >= FOS |N| L^2 / (pi^2 E)
        i_req = FOS_TARGET * absN * geo["length"] ** 2 / (math.pi ** 2 * E_MOD)
        r_b = np.where(N < -1e-9,
                       (np.maximum(i_req, 0.0) / I_COEF) ** 0.25, 0.0)
        return np.clip(np.maximum(r_y, r_b), R_MIN, R_MAX)

    def _fsd_radii(self, spec: dict, st: dict, iters: int = 25,
                   repair: int = 25) -> np.ndarray:
        """Fully-stressed design: size every member to just meet both FOS.

        Two stages. The damped fixed-point pass finds the neighbourhood; it
        does NOT reliably land on a feasible design, because resizing
        redistributes the forces it was sized against (measured: the plain
        iteration left FOS_buckling at 1.11-1.24 on a third of problems).
        The second pass only ever GROWS members, so it is monotone and
        terminates at a design that genuinely meets the targets - which a
        reference optimum has to do to be worth quoting.
        """
        geo = st["_geo"]
        r = np.full(geo["members"].shape[0], 0.02, dtype=float)
        for _ in range(iters):
            req = self._required_radii(geo, r)
            if req is None:
                r = np.minimum(r * 1.5, R_MAX)
                continue
            r = np.clip(0.5 * r + 0.5 * req, R_MIN, R_MAX)
        for _ in range(repair):
            req = self._required_radii(geo, r)
            if req is None:
                r = np.minimum(r * 1.5, R_MAX)
                continue
            if np.all(req <= r + 1e-12):
                break
            r = np.clip(np.maximum(r, req), R_MIN, R_MAX)   # monotone growth
        return r

    def reference(self, spec: dict) -> Optional[float]:
        cached = spec.get("_ref_mass")
        if cached is not None:
            try:
                c = float(cached)
                if math.isfinite(c) and c > 0.0:
                    return c
            except (TypeError, ValueError):
                pass
        nodes, members, fixed = self._build(
            str(spec["family"]), int(spec["n_bay"]),
            float(spec["width"]), float(spec["pitch"]))
        st = self._bare_state(spec, nodes, members, fixed)
        r = self._fsd_radii(spec, st)
        mass = float(np.sum(RHO * A_COEF * r * r * st["_geo"]["length"]))
        if not math.isfinite(mass) or mass <= 0.0:
            mass = 1.0
        spec["_ref_mass"] = mass
        spec["_ref_radii"] = [round(float(v), 8) for v in r]
        return mass

    def _bare_state(self, spec, nodes, members, fixed) -> dict:
        """Geometry-only state used by reference() before load() exists."""
        n_node = nodes.shape[0]
        vec = nodes[members[:, 1]] - nodes[members[:, 0]]
        length = np.maximum(np.linalg.norm(vec, axis=1), 1e-9)
        cos = vec / length[:, None]
        free = np.ones(3 * n_node, dtype=bool)
        for nd in fixed:
            free[3 * nd:3 * nd + 3] = False
        P = np.zeros(3 * n_node, dtype=float)
        fam = str(spec["family"])
        k = 3 if fam == "bridge" else 4
        total = float(spec["load_n"])
        if fam == "tower":
            for nd in range(n_node - k, n_node):
                P[3 * nd] += total / k
                P[3 * nd + 2] -= 0.35 * total / k
        elif fam == "cantilever":
            for nd in range(n_node - k, n_node):
                P[3 * nd + 2] -= total / k
        else:
            interior = [nd for nd in range(k, n_node - k)] or list(range(k, n_node))
            for nd in interior:
                P[3 * nd + 2] -= total / len(interior)
        return {"_geo": {"nodes": nodes, "members": members, "length": length,
                         "cos": cos, "free": free, "P": P, "n_node": n_node,
                         "span": max(float(np.ptp(nodes, axis=0).max()), 1e-6)}}

    def _start_radii(self, spec: dict, st: dict) -> np.ndarray:
        self.reference(spec)
        ref_r = spec.get("_ref_radii")
        n = st["_geo"]["members"].shape[0]
        if not ref_r or len(ref_r) != n:
            return np.full(n, 0.01, dtype=float)
        ref_arr = np.asarray(ref_r, dtype=float)
        base = ref_arr * float(spec.get("sound_frac", 1.04))
        degraded = spec.get("degraded") or []
        sev = spec.get("degraded_frac") or []
        if degraded and len(sev) == len(degraded):
            for pos, i in enumerate(degraded):
                if 0 <= int(i) < n:
                    base[int(i)] = ref_arr[int(i)] * float(sev[pos])
        else:
            base = ref_arr * float(spec.get("start_frac", 0.75))
        return np.clip(base, R_MIN, R_MAX)

    # ── depth-0 feedback hint ─────────────────────────────────────────────────
    def hint(self, st) -> Optional[Tuple[int, str, str]]:
        cached = st.get("_hint")
        if cached is not None:
            return None if cached == _HINT_NONE else cached
        geo = st["_geo"]
        r = np.clip(np.asarray(st["r"], dtype=float), R_MIN, R_MAX)
        out = self._solve(geo, r)
        if out is None:
            st["_hint"] = _HINT_NONE
            return None
        N, _ = out
        area = A_COEF * r * r
        inertia = I_COEF * r ** 4
        fos_y = SIGMA_Y / np.maximum(np.abs(N) / np.maximum(area, 1e-12), 1e-9)
        p_cr = (math.pi ** 2) * E_MOD * inertia / (geo["length"] ** 2)
        fos_b = np.where(N < -1e-9, p_cr / np.maximum(np.abs(N), 1e-9), 1e6)
        worst = int(np.argmin(np.minimum(fos_y, fos_b)))
        h = (worst, "r", "up")
        st["_hint"] = h
        return h

    # ── observation: describe() / element_label() ─────────────────────────────
    # Purely additive. Per-member quantities are RECOMPUTED here with the
    # read-only _solve() at the nominal load (scale 1, no rotation), which is
    # exactly the solve evaluate(seed=0) runs for the deterministic variant.
    # The state is never written: no cache, no hint, no radii.

    def _obs_topology(self, st) -> list:
        """(node_a, node_b, role, place, where) per member, decoded from the
        build's node numbering node = k * section + corner."""
        fam = str(st["spec"]["family"])
        k = 3 if fam == "bridge" else 4
        corner = _OBS_CORNER.get(fam, _OBS_CORNER["tower"])
        edge = _OBS_EDGE.get(fam, _OBS_EDGE["tower"])
        members = st["_geo"]["members"]
        out = []
        for m in range(members.shape[0]):
            a, b = int(members[m, 0]), int(members[m, 1])
            if a > b:
                a, b = b, a
            sa, ja, sb, jb = a // k, a % k, b // k, b % k
            key = (min(ja, jb), max(ja, jb))
            if sa == sb:
                role = "sdiag" if (k == 4 and key == (0, 2)) else "ring"
                out.append((a, b, role, "s%d" % sa, edge.get(key, "")))
            elif ja == jb:
                out.append((a, b, "leg" if fam == "tower" else "chord",
                            "b%d" % sb, corner[ja % len(corner)]))
            else:
                out.append((a, b, "diag", "b%d" % sb, edge.get(key, "")))
        return out

    def _obs_results(self, st) -> dict:
        """Per-member section, force and FOS at the nominal load (no writes)."""
        geo = st["_geo"]
        r = np.clip(np.asarray(st["r"], dtype=float), R_MIN, R_MAX)
        length = np.asarray(geo["length"], dtype=float)
        area = A_COEF * r * r
        inertia = I_COEF * r ** 4
        res = {"r": r, "length": length, "area": area,
               "mass": RHO * area * length, "N": None}
        try:
            out = self._solve(geo, r)
        except Exception:
            out = None
        if out is None:
            return res
        N, u = out
        if not (np.all(np.isfinite(N)) and np.all(np.isfinite(u))):
            return res
        absN = np.abs(N)
        comp = N < -1e-9
        p_cr = (math.pi ** 2) * E_MOD * inertia / (length ** 2)
        res["N"] = np.asarray(N, dtype=float)
        res["u"] = np.asarray(u, dtype=float).reshape(-1, 3)
        res["comp"] = comp
        res["fos_y"] = SIGMA_Y / np.maximum(absN / np.maximum(area, 1e-12), 1e-9)
        res["fos_b"] = np.where(comp, p_cr / np.maximum(absN, 1e-9), np.inf)
        # |N| at or below 1 mN (or solver round-off) is reported as no force
        res["tiny"] = absN <= max(1e-3, 1e-9 * float(absN.max(initial=0.0)))
        return res

    def describe(self, st, spec) -> str:
        """Self-contained plain-text observation of the current design."""
        if spec is None:
            spec = st["spec"]
        geo = st["_geo"]
        fam = str(spec["family"])
        n_bay = int(spec["n_bay"])
        k = 3 if fam == "bridge" else 4
        width, pitch = float(spec["width"]), float(spec["pitch"])
        nodes = np.asarray(geo["nodes"], dtype=float)
        n_node = int(geo["n_node"])
        n_mem = int(geo["members"].shape[0])
        P = np.asarray(geo["P"], dtype=float).reshape(-1, 3)
        free = np.asarray(geo["free"], dtype=bool).reshape(-1, 3)
        fixed = set(n for n in range(n_node) if not free[n].any())
        loaded = [n for n in range(n_node) if np.any(P[n] != 0.0)]
        topo = self._obs_topology(st)
        res = self._obs_results(st)
        f = _obs_num
        ext = np.ptp(nodes, axis=0)
        tot = P.sum(axis=0) / 1e3

        out = ["PROBLEM"]
        if fam == "tower":
            out.append("Lattice tower (3D pin-jointed truss): %d bays of square "
                       "%s x %s m sections stacked along +z at pitch %s m, "
                       "height %s m; %d nodes, %d members."
                       % (n_bay, f(width, 4), f(width, 4), f(pitch, 4),
                          f(ext[2], 4), n_node, n_mem))
            out.append("Supports: base section s0 (nodes %s) pinned, x/y/z fixed."
                       % _obs_runs(sorted(fixed)))
            what = "Top-section load (wind along +x plus dead load)"
        elif fam == "cantilever":
            out.append("Cantilever truss (3D pin-jointed): %d bays of square "
                       "%s x %s m sections (y-z plane) extending along +x at "
                       "pitch %s m, length %s m; %d nodes, %d members."
                       % (n_bay, f(width, 4), f(width, 4), f(pitch, 4),
                          f(ext[0], 4), n_node, n_mem))
            out.append("Supports: root section s0 (nodes %s) pinned, x/y/z fixed."
                       % _obs_runs(sorted(fixed)))
            what = "Tip load on the last section"
        else:
            out.append("Bridge girder (3D pin-jointed truss): %d bays of "
                       "triangular sections (two deck nodes %s m apart at z=0, "
                       "apex node at z=%s m) along +x at pitch %s m, span %s m; "
                       "%d nodes, %d members."
                       % (n_bay, f(width, 4), f(0.85 * width, 4), f(pitch, 4),
                          f(ext[0], 4), n_node, n_mem))
            out.append("Supports: both end sections s0 and s%d (nodes %s) "
                       "pinned, x/y/z fixed, so both girder ends are restrained "
                       "(fixed-ended, not simply supported)."
                       % (n_bay, _obs_runs(sorted(fixed))))
            what = ("Vertical load on every interior node, deck and apex "
                    "(sections s1-s%d)" % (n_bay - 1))
        groups = {}
        for n in loaded:
            groups.setdefault(tuple(float(v) for v in P[n]), []).append(n)
        parts = ["nodes %s each (Fx, Fy, Fz) = (%s) kN"
                 % (_obs_runs(ns), ", ".join(f(v / 1e3) for v in vec))
                 for vec, ns in groups.items()]
        out.append("%s: %s; resultant (%s) kN."
                   % (what, "; ".join(parts), ", ".join(f(v) for v in tot)))
        if self.stochastic:
            out.append("Load scatter: each evaluation scales all loads by a "
                       "random factor (mean 1, ~8% sd) and turns them about z "
                       "by a random angle (sd 0.03 rad).")
        out.append("Each node of section s is braced to three nodes of section "
                   "s-1, so the truss is rigid and statically indeterminate: "
                   "member stiffness E*A/L grows with r^2, so resizing a member "
                   "shifts force to or from its neighbours.")
        out.append("Material 6061-T6 aluminium: E = %s GPa, yield stress %s MPa, "
                   "density %s kg/m3." % (f(E_MOD / 1e9), f(SIGMA_Y / 1e6), f(RHO, 4)))
        out.append("Design variable per member: r = outer tube radius [m], "
                   "bounds %s-%s m; wall t = %s r, so A = %s r^2 and I = %s r^4."
                   % (f(R_MIN), f(R_MAX), f(WALL_RATIO), f(A_COEF, 4), f(I_COEF, 4)))
        out.append("Member checks: FOS_yielding = %s MPa / (|N|/A); FOS_buckling "
                   "= pi^2*E*I / (L^2*|N|), compression only (pin-ended Euler); "
                   "mass = %s*A*L. N > 0 tension, N < 0 compression."
                   % (f(SIGMA_Y / 1e6), f(RHO, 4)))
        out.append("Roles: %s = longitudinal member between the same corner of "
                   "consecutive sections; diag = bay-face bracing diagonal; "
                   "ring = edge within a section%s. Bay b (b1-b%d) spans "
                   "sections b-1 to b. Location: %s"
                   % ("leg" if fam == "tower" else "chord",
                      "; sdiag = diagonal within a section (corner j0 to j2)"
                      if k == 4 else "", n_bay, _OBS_CODES.get(fam, "")))

        # every node's coordinates = its section's axial position + its corner
        axis = 2 if fam == "tower" else 0
        tr = [c for c in range(3) if c != axis]
        ax_name, tr_name = "xyz"[axis], "%s,%s" % ("xyz"[tr[0]], "xyz"[tr[1]])
        corner = _OBS_CORNER.get(fam, _OBS_CORNER["tower"])
        out.append("")
        out.append("NODES (node = %d*s + j for section s, corner j; m)" % k)
        out.append("Corners (%s): %s." % (tr_name, "; ".join(
            "j%d (%s) %s" % (j, ",".join(f(nodes[j, c]) for c in tr), corner[j])
            for j in range(k))))
        for s in range(n_bay + 1):
            ids = list(range(s * k, min((s + 1) * k, n_node)))
            if not ids:
                continue
            tag = ""
            if all(n in fixed for n in ids):
                tag = " pinned"
            elif all(n in loaded for n in ids):
                tag = " loaded"
            out.append("s%d %s=%s: n%d-n%d%s"
                       % (s, ax_name, f(nodes[ids[0], axis], 4), ids[0], ids[-1],
                          tag))

        r, length, mass = res["r"], res["length"], res["mass"]
        out.append("")
        out.append("DESIGN (per member: id, end nodes, role, bay/section, location "
                   "| L m, r m, mass kg; area A = %s r^2)" % f(A_COEF, 4))
        for m, (a, b, role, place, where) in enumerate(topo):
            out.append("m%d %d-%d %s %s%s | %s %s %s"
                       % (m, a, b, role, place, (" " + where) if where else "",
                          f(length[m]), f(r[m]), f(mass[m])))

        out.append("")
        N = res["N"]
        if N is None:
            out.append("RESULTS: the stiffness solve is singular for this design; "
                       "no member forces are available.")
            return "\n".join(out)
        out.append("RESULTS (%slinear stiffness solve at the nominal load; per "
                   "member: id, N kN, FOS_yielding, then FOS_buckling for "
                   "compression members)"
                   % ("per-member values are a nominal evaluation without load "
                      "scatter; " if self.stochastic else ""))
        fos_y, fos_b, comp, tiny = res["fos_y"], res["fos_b"], res["comp"], res["tiny"]
        for m, (a, b, _, _, _) in enumerate(topo):
            if tiny[m]:
                if a in fixed and b in fixed:
                    out.append("m%d 0 (both ends pinned)" % m)
                else:
                    out.append("m%d ~0 (near-zero force)" % m)
            elif comp[m]:
                out.append("m%d %s %s %s" % (m, f(N[m] / 1e3), _obs_fos(fos_y[m]),
                                            _obs_fos(fos_b[m])))
            else:
                out.append("m%d +%s %s" % (m, f(N[m] / 1e3), _obs_fos(fos_y[m])))
        gov = np.where(tiny, np.inf, np.minimum(fos_y, fos_b))
        order = [int(m) for m in np.argsort(gov, kind="stable")[:5]
                 if math.isfinite(float(gov[int(m)]))]
        if order:
            # load-path neighbourhood: the other members meeting at each end
            inc = {}
            for m, (a, b, _, _, _) in enumerate(topo):
                inc.setdefault(a, []).append(m)
                inc.setdefault(b, []).append(m)
            out.append("Lowest member FOS (governing mode | the other members "
                       "meeting at each end node):")
            for m in order:
                a, b = topo[m][0], topo[m][1]
                line = "m%d %s %s" % (m, _obs_fos(gov[m]),
                                      "buckling" if (comp[m] and fos_b[m] <= fos_y[m])
                                      else "yielding")
                line += "".join(" | n%d: %s" % (nd, " ".join(
                    "m%d" % o for o in inc.get(nd, []) if o != m) or "-")
                    for nd in (a, b))
                out.append(line)
        return "\n".join(out)

    def element_label(self, st, i) -> str:
        """Short role label for member i, e.g.
        'member 8 (nodes 2-6, bay-1 diagonal on x+ face, compression)'."""
        i = int(i)
        topo = self._obs_topology(st)
        if not (0 <= i < len(topo)):
            return "member %d" % i
        a, b, role, place, where = topo[i]
        fam = str(st["spec"]["family"])
        num = place[1:]
        face = _OBS_FACE_LONG.get(fam, {}).get(where, where)
        if role == "leg":
            what = "bay-%s leg at corner %s" % (num, where)
        elif role == "chord":
            what = "bay-%s %s chord" % (num, where.replace("bot", "bottom"))
        elif role == "diag":
            what = "bay-%s diagonal on %s" % (num, face)
        elif role == "ring":
            what = "section-%s ring edge on %s" % (num, face)
        else:
            what = "section-%s in-section diagonal" % num
        res = self._obs_results(st)
        N = res["N"]
        free = np.asarray(st["_geo"]["free"], dtype=bool).reshape(-1, 3)
        if not free[a].any() and not free[b].any():
            sense = "both ends pinned, no force"
        elif N is None:
            sense = ""
        elif res["tiny"][i]:
            sense = "near-zero force"
        elif res["comp"][i]:
            sense = "compression"
        else:
            sense = "tension"
        return "member %d (nodes %d-%d, %s%s)" % (
            i, a, b, what, (", " + sense) if sense else "")


# ── observation helpers (describe / element_label only) ──────────────────────
_OBS_CORNER = {
    "tower": ("x-y-", "x+y-", "x+y+", "x-y+"),
    "cantilever": ("bot y-", "bot y+", "top y+", "top y-"),
    "bridge": ("bot y-", "bot y+", "top"),
}
_OBS_EDGE = {
    "tower": {(0, 1): "y-", (1, 2): "x+", (2, 3): "y+", (0, 3): "x-", (0, 2): ""},
    "cantilever": {(0, 1): "bot", (1, 2): "y+", (2, 3): "top", (0, 3): "y-",
                   (0, 2): ""},
    "bridge": {(0, 1): "deck", (1, 2): "y+", (0, 2): "y-"},
}
_OBS_FACE_LONG = {
    "tower": {"x+": "leeward x+ face", "x-": "windward x- face",
              "y+": "y+ face", "y-": "y- face"},
    "cantilever": {"bot": "bottom face", "top": "top face",
                   "y+": "y+ side face", "y-": "y- side face"},
    "bridge": {"deck": "deck plane", "y+": "y+ side web", "y-": "y- side web"},
}
_OBS_CODES = {
    "tower": ("legs carry their corner code; rings and diags carry the side "
              "face they lie on (x+, x-, y+, y-; x- is windward, x+ leeward)."),
    "cantilever": ("chords carry their corner code; rings and diags carry "
                   "their face (bot = z- face, top = z+ face, y+, y-)."),
    "bridge": ("chords carry their corner code (bot = deck chords at z=0, "
               "top = apex chord); rings and diags carry their face (deck = "
               "plane of the two bottom chords; y+, y- = inclined side webs)."),
}


def _obs_num(v, sig: int = 3) -> str:
    """Significant-figure formatting; never renders a nonzero value as 0."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:
        return "nan"
    if x in (float("inf"), float("-inf")):
        return "inf" if x > 0 else "-inf"
    if x == 0.0:
        return "0"
    a = abs(x)
    if a < 1e-4:
        return "%.*e" % (sig - 1, x)
    s = "%.*g" % (sig, x)
    if "e" in s:                       # large magnitude: plain integer
        s = "%.0f" % x
    return s


def _obs_fos(v) -> str:
    x = float(v)
    if not math.isfinite(x) or x > 999.0:
        return ">999"
    return _obs_num(x)


def _obs_runs(ids) -> str:
    """Compact integer list: [0,1,2,3,7] -> '0-3, 7'."""
    ids = sorted(int(v) for v in ids)
    if not ids:
        return "none"
    runs, start, prev = [], ids[0], ids[0]
    for v in ids[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev))
        start = prev = v
    runs.append((start, prev))
    return ", ".join(("%d" % s) if s == e else ("%d-%d" % (s, e)) for s, e in runs)
