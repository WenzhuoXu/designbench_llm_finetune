"""Photonic inverse design on the invrs-gym / ceviche-challenges "lightweight"
challenges, with the field's own topology-editing verbs (native-verb domain,
docs/corpus_native_domains.md, contract docs/dataxp_move_contract.md).

Simulator
---------
ceviche-challenges 1.0.2 (Google, Schubert et al. 2022, "Inverse design of
photonic devices with strict foundry fabrication constraints", ACS Photonics
9, 2327; arXiv:2201.12965) on top of ceviche (Hughes et al. 2019, "Forward-mode
differentiation of Maxwell's equations", ACS Photonics 6, 3010): a 2-D
finite-difference frequency-domain (FDFD) solve of the out-of-plane electric
field Ez on a 40 nm grid with perfectly-matched-layer boundaries, modal
sources and 1-D mode monitors at the ports. Materials: relative permittivity
12.25 (silicon, n = 3.5) where there is material, 2.25 (silica, n = 1.5)
elsewhere; waveguides 400 nm wide. The five challenges, their geometry, the
40 nm "lightweight" simulation parameters, the 1270/1290 nm wavelength pair,
the 3-pixel minimum width / spacing and the transmission targets are ported
verbatim from invrs-gym 1.6.2 (invrs_gym/challenges/ceviche/defaults.py and
challenge.py). invrs-gym itself is not imported: its jax 0.12 dependency needs
numpy >= 2.2, and this repository is pinned to numpy 1.26 / scipy 1.13 for the
bit-for-bit v1/v2 regressions; ceviche-challenges + autograd run on Windows.

Families (index % 5), design region at 40 nm per pixel, port 1 excited
------------------------------------------------------------------------
  waveguide_bend   1.6 x 1.6 um (40 x 40 px). Port 1 enters at the left edge,
                   port 2 leaves at the bottom edge. Targets |S21|^2 >= -0.5 dB
                   and |S11|^2 <= -20 dB at both wavelengths.
  mode_converter   1.6 x 1.6 um (40 x 40). Fundamental mode in at the left,
                   SECOND-order mode of the same 400 nm guide out at the right.
                   |S21|^2 (second-order mode only) >= -0.5 dB, |S11|^2 <= -20 dB.
  beam_splitter    3.2 x 2.0 um (80 x 50). Four ports, two parallel guides;
                   port 1 (left, upper) splits to ports 2 (right, upper) and 3
                   (right, lower): each within [-3.5 dB, 1 - 10^-0.35], and
                   |S11|^2, |S41|^2 <= -20 dB.
  power_splitter   1.6 x 1.6 um (40 x 40). Port 1 (left, centre) to ports 2
                   and 3 (right, lower / upper) each >= -3.5 dB, |S11|^2 <= -20 dB.
  wdm              3.2 x 3.2 um (80 x 80). 1270 nm from port 1 to port 2 (right,
                   lower) and 1290 nm to port 3 (right, upper): >= -3 dB into the
                   right output, <= -20 dB into the wrong one, |S11|^2 <= -20 dB.

State, metrics, objective, constraints
--------------------------------------
State = a binary silicon/cladding pattern on the design region, pat[y, x]
(x = column 0..W-1 left to right, y = row 0..H-1 bottom to top, as the
challenge sketches draw it). Metrics: Tj1_<wl> = |S_j1|^2, the fraction of the
power launched into port 1 that leaves through port j's monitored mode at
wavelength wl (T11 = reflection); loss = 1 - mean over wavelengths of the
power delivered to the intended output port(s) (the objective, minimised;
insertion_loss_db is the same in dB); radiation_loss = power leaving through no
port; target_loss = the challenge's own smooth window loss (invrs-gym
orthotope_smooth_transmission_loss, exponents 1 and 2); fill_fraction;
n_islands; min_feature_violations. Goals are the challenge's transmission /
extinction targets as minimum_/maximum_ bounds on the Tj1_<wl> metrics, so
every target is a constraint at every wavelength; Phi's objective term is loss
normalised by the reference design's loss (floored at 0.01).

Verbs (typed, pixel units; the grammar() text is what the policy reads)
------------------------------------------------------------------------
  FILL_RECT(x0, y0, x1, y1)  deposit silicon on the inclusive rectangle
  CLEAR_RECT(x0, y0, x1, y1) etch it back to cladding
  FILL_DISK(cx, cy, r) / CLEAR_DISK(cx, cy, r)   the same on a pixel disk
  SHIFT_FEATURE(k, dx, dy)   move silicon island k rigidly (must stay inside)
  DILATE(k, px) / ERODE(k, px)   widen / narrow island k by px on every side
Islands are the 4-connected silicon components, numbered in reading order of
the printed map (topmost, then leftmost pixel first) and renumbered after
every move. A move must change at least one pixel and must leave the pattern
fabricable (below); otherwise apply() raises ValueError.

Minimum feature size (validity constraint, enforced at apply())
--------------------------------------------------------------
d = min_feature_px (3 px = 120 nm, the invrs-gym lightweight target, or 4 px
on a quarter of the problems): every silicon pixel must lie inside an
all-silicon d x d square and every cladding pixel inside an all-cladding
d x d square, judged on the design region together with a d-pixel margin of
the fixed surroundings (so a gap narrower than d to a waveguide stub counts).
This is a morphological-opening rule in the spirit of imageruler's minimum
length scale, with a square kernel so that rectangles pass exactly; it is not
imageruler itself. The metric min_feature_violations counts offending pixels
and the validity envelope requires it to be 0. Generated patterns (reference,
blob and strip starts) are made fabricable by _drc_repair: open/close with
the d x d kernel, then pixel fixes, keeping the best pattern seen; when
neither converges the pattern is snapped to d-aligned blocks, which satisfy
the rule by construction.

Problem generation (make_problem) and starts
--------------------------------------------
Per problem: family, min feature (3 or 4 px), target tightness (the
challenge's nominal dB levels on 55% of problems; floors relaxed by 0.5-1 dB
and/or isolation relaxed to -15 dB on the rest, all stated in the goals and
in the observation) and a start pattern of one of these kinds:
  near / mid / deep   the reference design with 1 / 2-3 / 4-6 harmful legal
                      moves applied (chosen by the adjoint sensitivity; each is
                      a legal verb, so the repair is in the vocabulary)
  blobs               random smoothed-noise islands
  strip               the naive engineer's start: 320-480 nm straight / L / Y
                      bars joining the ports; every dimension (bar width, the
                      bend's corner facet / fillet / overshoot, the converter's
                      bump side, height and extent, the Y's junction) is drawn
  empty, full         no silicon / solid slab, plus one or two seed features
                      (FILL_RECT / FILL_DISK on the blank region, CLEAR_RECT /
                      CLEAR_DISK on the slab, drawn uniformly from the
                      structured pool and recorded in the spec's perturbations)
  feasible            the reference itself (only when it meets the goals)
Every start is fabricable and every start depends on the problem's rng (an
earlier version returned the same empty / full / plain-bar pattern for every
problem of a kind, which repeated specs at n = 130). As a last safety net a
start that repeats one already built in the process is re-drawn (_BUILT
below). The reference, cached per (family, wavelengths,
min feature) in photonics_refs.json next to this module (and in the spec), is
a short gradient-based run (reference_class "heuristic"): Adam on a latent
density through a conic filter of radius d and a tanh projection with beta
1 -> 64, with the challenge's own window loss, 4 seeded restarts of 200 steps;
each final density is binarised at thresholds 0.4 / 0.5 / 0.6 and repaired to
the feature-size rule, and the fabricable design with the lowest window loss
is kept (5-17 min per entry single-threaded). It is not the published
leaderboard best; whether it meets the nominal targets is recorded in the
spec (ref_feasible). Shipped cache (loss = 1 - delivered power): waveguide
bend 0.080 (d=3, meets the nominal targets) / 0.59 (d=4); mode converter
0.21 / 0.28; beam splitter 0.29 / 0.15; power splitter 0.10 / 0.073 (d=4
meets the nominal targets); WDM 0.71 / 0.80 (far from its targets: the
demultiplexer is the hard challenge of the suite, and its problems may never
reach feasibility within an episode). The relaxed tightness classes are
where most problems have a feasible reference.

Candidate set (legal_moves, 32 per state; 24 when the ladder asks): every
structured move of every verb is scored by the first-order adjoint estimate
of its effect on the window loss; per-verb quotas (rectangles 6+6, disks 3+3,
shift 4, dilate 3, erode 3) keep every verb on offer whenever the state
admits it, the rest of the budget goes to the best remaining moves of any
verb and the last four are a seeded random draw. Only fabricable moves are
offered.

Simplifications and exclusions (explicit)
-----------------------------------------
* Only port 1 is excited (the invrs-gym default); the beam splitter's symmetric
  behaviour for inputs at the other ports is not part of the objective and no
  design symmetry is imposed.
* The wavelength pair is fixed at the lightweight 1270/1290 nm; distinct
  problems come from the start, the min feature and the target tightness.
* The feature-size rule is the square-kernel opening test above, not imageruler.
* Beam splitter: invrs-gym bounds each output to the window [10^-0.35,
  1 - 10^-0.35]. The upper edge is implied by the two floors and power
  conservation, so the goals carry the floors only (same feasible set); the
  reference's window loss keeps both edges as invrs-gym writes them.
* The other invrs-gym challenges (metagrating, diffractive splitter, metalens,
  photon extractor, bayer sorter, ...) are EXCLUDED: they are 3-D RCWA problems
  on fmmax/jax, not seconds-per-evaluation on CPU, and could not be installed
  beside numpy 1.26 here.
* No problem of this suite admits only parameter setting; nothing is excluded
  on that ground.

Cost: one evaluate() is one FDFD solve per wavelength with scipy's SuperLU
(pardiso/pyMKL does not load on this machine): measured 0.25-0.33 s for the
40 x 40 families, ~0.5 s for the beam splitter, ~0.6 s for the WDM; a
sensitivity map (one adjoint per wavelength) 0.6-1.6 s. Evaluations and
sensitivities are cached per pattern inside the domain instance. BLAS threads
are capped at one at import (threadpoolctl, when installed): the solve is
single-threaded SuperLU, and spinning BLAS threads on an oversubscribed box
inflated the same calls 100-fold.

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
--------------------------------------------------------------------------------
The same moves written through the one wrapper tool, by the generic encoding
of dataxp.domain from the `verbs` table (nothing overridden here): element =
the verb's first int argument, operation = the verb in lower case, value =
the remaining argument or the JSON list of the remaining arguments in table
order.

  FILL_RECT(12, 5, 20, 9)      SET_PARAM(12, fill_rect, [5, 20, 9])
  CLEAR_RECT(12, 5, 20, 9)     SET_PARAM(12, clear_rect, [5, 20, 9])
  FILL_DISK(20, 20, 3)         SET_PARAM(20, fill_disk, [20, 3])
  CLEAR_DISK(20, 20, 3)        SET_PARAM(20, clear_disk, [20, 3])
  SHIFT_FEATURE(2, 1, -1)      SET_PARAM(2, shift_feature, [1, -1])
  DILATE(2, 1)                 SET_PARAM(2, dilate, 1)
  ERODE(2, 1)                  SET_PARAM(2, erode, 1)

Judgment calls:
* A rectangle or a disk has no element of its own; the owner's rule for the
  native four (element = the operation's first integer argument) makes the
  left column x0 / the centre column cx the element and the remaining
  coordinates the value list, in the table's order. The island verbs keep
  the island number as the element.
* describe()'s "Changes are made with these verbs ..." paragraph prints the
  seven call signatures through Domain.call_form and says "operations" for
  "verbs" under set_parameter; nothing else in the observation changes, and
  the multitool text is byte for byte the previous one.
* VerbSpec.phrase heads each operation block of the generated grammar
  ("Fill a rectangle: SET_PARAM(x0, fill_rect, [y0, x1, y1])").
"""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import OrderedDict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from ..domain import ArgSpec, Domain, Move, VerbSpec, spelled_in_verbs, stable_seed
from ..potential import DesignProgram, program_from_goals

_FAMILIES = ("waveguide_bend", "mode_converter", "beam_splitter", "power_splitter", "wdm")

# invrs-gym lightweight ceviche constants (invrs_gym/challenges/ceviche/defaults.py)
GRID_NM = 40
WAVELENGTHS_NM = (1270, 1290)
WG_WIDTH_NM = 400
WG_MODE_PADDING_NM = 520
WG_LENGTH_NM = 720
PADDING_NM = 400
PORT_PML_OFFSET_NM = 40
INPUT_MONITOR_OFFSET_NM = 40
PML_PX = 20
EPS_CLAD = 2.25
EPS_SLAB = 12.25
MINFEAT_DEFAULT = 3

REF_STEPS = 200                      # Adam steps per restart
REF_RESTARTS = 4                     # seeded restarts; the best final design is kept
REF_THRESHOLDS = (0.4, 0.5, 0.6)     # binarisation thresholds tried on the final density
REF_BETAS = (1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
REF_LR = 0.1
REF_LOSS_FLOOR = 0.01

_EVAL_CACHE_MAX = 4096
_GRAD_CACHE_MAX = 128
_FIELD_CACHE_MAX = 6


def _db(x_db: float) -> float:
    return float(10.0 ** (x_db / 10.0))


def _to_db(x: float) -> float:
    return 10.0 * math.log10(max(float(x), 1e-12))


# per-family nominal targets per wavelength BAND (invrs-gym defaults.py); a
# single band applies to both wavelengths
_FAMILY = {
    "waveguide_bend": dict(
        title="90-degree waveguide bend", design_nm=(1600, 1600), n_ports=2,
        wanted=[[1]], lb=[[0.0, _db(-0.5)]], ub=[[_db(-20), 1.0]],
        goal="route all of the power entering at port 1 (left edge, travelling +x) out "
             "through port 2 (bottom edge, travelling -y)"),
    "mode_converter": dict(
        title="waveguide mode converter (fundamental to second-order mode)",
        design_nm=(1600, 1600), n_ports=2,
        wanted=[[1]], lb=[[0.0, _db(-0.5)]], ub=[[_db(-20), 1.0]],
        goal="convert the fundamental (first-order, even) mode entering at port 1 (left) "
             "into the SECOND-order (odd) mode of the identical 400 nm waveguide at port 2 "
             "(right); T21 counts only the power in the second-order mode"),
    "beam_splitter": dict(
        title="four-port 50:50 beam splitter (directional-coupler layout)",
        design_nm=(3200, 2000), n_ports=4,
        wanted=[[1, 2]], lb=[[0.0, _db(-3.5), _db(-3.5), 0.0]],
        ub=[[_db(-20), 1.0 - _db(-3.5), 1.0 - _db(-3.5), _db(-20)]],
        goal="split the power entering at port 1 (left, upper waveguide) equally between "
             "port 2 (right, upper) and port 3 (right, lower), with as little as possible "
             "reflected into port 1 or sent to port 4 (left, lower)"),
    "power_splitter": dict(
        title="1 x 2 power splitter", design_nm=(1600, 1600), n_ports=3,
        wanted=[[1, 2]], lb=[[0.0, _db(-3.5), _db(-3.5)]], ub=[[_db(-20), 1.0, 1.0]],
        goal="divide the power entering at port 1 (left, centre) between port 2 (right, "
             "lower) and port 3 (right, upper)"),
    "wdm": dict(
        title="two-channel wavelength demultiplexer", design_nm=(3200, 3200), n_ports=3,
        wanted=[[1], [2]],
        lb=[[0.0, _db(-3.0), 0.0], [0.0, 0.0, _db(-3.0)]],
        ub=[[_db(-20), 1.0, _db(-20)], [_db(-20), _db(-20), 1.0]],
        goal="send the 1270 nm light entering at port 1 (left, centre) out through port 2 "
             "(right, lower) and the 1290 nm light out through port 3 (right, upper), each "
             "channel with little leakage into the other output"),
}

# target-tightness classes: (floor relaxation dB, isolation level dB, weight)
_TIGHTNESS = (("nominal", 0.0, 20.0, 0.55), ("floor-0.5dB", 0.5, 20.0, 0.15),
              ("iso-15dB", 0.0, 15.0, 0.15), ("floor-1dB_iso-15dB", 1.0, 15.0, 0.15))

_START_KINDS = (("near", 0.28), ("mid", 0.20), ("deep", 0.10), ("blobs", 0.14),
                ("strip", 0.12), ("empty", 0.07), ("full", 0.03), ("feasible", 0.06))

_MODELS: Dict[tuple, object] = {}
_REFS: Dict[tuple, dict] = {}

# (family, min feature, tightness, design0) of every problem built in this
# process. Every start kind is seeded (see _start_pattern), which is what
# keeps the problem set distinct; as the last safety net make_problem
# re-draws its start kind and pattern when the start it drew repeats one
# already built, so one generation run has no duplicate spec (gate G1) and
# no duplicate start design. _N_REDRAWS counts how often that fired (the
# survey in the report records it: 0 at n = 130, seed 0). Problem i is
# therefore reproducible from a fresh process when problems are built in
# index order, which is what run_generation and smoke do (the convention
# grid2op.py uses); PhotonicsDomain.dedupe = False switches the net off.
_BUILT: set = set()
_N_REDRAWS = [0]


_REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photonics_refs.json")
_REF_DISK: Optional[dict] = None


def _load_ref_file() -> dict:
    global _REF_DISK
    if _REF_DISK is None:
        try:
            with open(_REF_FILE) as fh:
                _REF_DISK = json.load(fh)
        except (OSError, ValueError):
            _REF_DISK = {}
    return _REF_DISK


def _save_ref_file(d: dict) -> None:
    try:
        tmp = _REF_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
        os.replace(tmp, _REF_FILE)
    except OSError:
        pass


def _cap_blas_threads(n: int = 1) -> None:
    """The FDFD solve is scipy's single-threaded SuperLU and the port-mode
    eigenproblems are tiny, so BLAS threads gain nothing here, while their
    busy-waiting under an oversubscribed box is catastrophic (measured on a
    24-core machine at 100% load: a 1-minute reference run took 51-73 minutes
    and a first forward solve 94 s with 3 OpenBLAS threads; 0.4 s with one).
    Capped at import through threadpoolctl when it is installed, unless the
    caller set OPENBLAS_NUM_THREADS / MKL_NUM_THREADS explicitly."""
    if os.environ.get("OPENBLAS_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS"):
        return
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=n, user_api="blas")
    except Exception:
        pass
    try:
        import mkl
        if mkl.get_max_threads() > n:
            mkl.set_num_threads(n)
    except Exception:
        pass


_cap_blas_threads()


# ── simulator construction (ported from invrs_gym.challenges.ceviche) ─────────
def _backend():
    try:
        import ceviche_challenges  # noqa: F401
        import autograd  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "photonics domain needs ceviche-challenges + autograd "
            "(pip install -r dataxp/requirements_photonics.txt): %s" % e)


def _wdm_spec(design_nm: int, offset_nm: int):
    from ceviche_challenges import units as u
    from ceviche_challenges import wdm
    pml_nm = PML_PX * GRID_NM
    ei = design_nm + 2 * WG_LENGTH_NM + 2 * pml_nm
    ej = design_nm + 2 * PADDING_NM + 2 * pml_nm
    return wdm.spec.WdmSpec(
        extent_ij=u.Array([ei, ej], u.nm), input_wg_j=ej / 2 * u.nm,
        output_wgs_j=u.Array((ej / 2 - (design_nm / 2 - WG_WIDTH_NM / 2 - offset_nm),
                              ej / 2 + (design_nm / 2 - WG_WIDTH_NM / 2 - offset_nm)), u.nm),
        wg_width=WG_WIDTH_NM * u.nm, wg_mode_padding=WG_MODE_PADDING_NM * u.nm,
        input_mode_i=pml_nm * u.nm + PORT_PML_OFFSET_NM * u.nm,
        output_mode_i=(ei - pml_nm) * u.nm - PORT_PML_OFFSET_NM * u.nm,
        variable_region=(u.Array((ei / 2 - design_nm / 2, ej / 2 - design_nm / 2), u.nm),
                         u.Array((ei / 2 + design_nm / 2, ej / 2 + design_nm / 2), u.nm)),
        cladding_permittivity=EPS_CLAD, slab_permittivity=EPS_SLAB,
        input_monitor_offset=INPUT_MONITOR_OFFSET_NM * u.nm, pml_width=PML_PX)


def _model(family: str, wls: Tuple[int, ...]):
    key = (family, tuple(wls))
    m = _MODELS.get(key)
    if m is not None:
        return m
    _backend()
    from ceviche_challenges import beam_splitter, mode_converter, waveguide_bend, wdm
    from ceviche_challenges import params as ccp
    from ceviche_challenges import units as u
    sp = ccp.CevicheSimParams(resolution=GRID_NM * u.nm,
                              wavelengths=u.Array([float(w) for w in wls], u.nm))
    common = dict(cladding_permittivity=EPS_CLAD, slab_permittivity=EPS_SLAB,
                  input_monitor_offset=INPUT_MONITOR_OFFSET_NM * u.nm, pml_width=PML_PX,
                  port_pml_offset=PORT_PML_OFFSET_NM * u.nm)
    if family == "waveguide_bend":
        spec = waveguide_bend.spec.WaveguideBendSpec(
            wg_width=WG_WIDTH_NM * u.nm, wg_length=WG_LENGTH_NM * u.nm,
            wg_mode_padding=WG_MODE_PADDING_NM * u.nm, padding=PADDING_NM * u.nm,
            variable_region_size=(1600 * u.nm, 1600 * u.nm), **common)
        m = waveguide_bend.model.WaveguideBendModel(sp, spec)
    elif family == "mode_converter":
        spec = mode_converter.spec.ModeConverterSpec(
            left_wg_width=WG_WIDTH_NM * u.nm, left_wg_mode_padding=WG_MODE_PADDING_NM * u.nm,
            left_wg_mode_order=1, right_wg_width=WG_WIDTH_NM * u.nm,
            right_wg_mode_padding=WG_MODE_PADDING_NM * u.nm, right_wg_mode_order=2,
            wg_length=WG_LENGTH_NM * u.nm, padding=PADDING_NM * u.nm,
            variable_region_size=(1600 * u.nm, 1600 * u.nm), **common)
        m = mode_converter.model.ModeConverterModel(sp, spec)
    elif family == "beam_splitter":
        spec = beam_splitter.spec.BeamSplitterSpec(
            wg_width=WG_WIDTH_NM * u.nm, wg_length=WG_LENGTH_NM * u.nm,
            wg_separation=1120 * u.nm, wg_mode_padding=WG_MODE_PADDING_NM * u.nm,
            variable_region_size=(3200 * u.nm, 2000 * u.nm), design_symmetry=None, **common)
        m = beam_splitter.model.BeamSplitterModel(sp, spec)
    elif family == "power_splitter":
        m = wdm.model.WdmModel(sp, _wdm_spec(1600, 120))
    elif family == "wdm":
        m = wdm.model.WdmModel(sp, _wdm_spec(3200, 400))
    else:
        raise KeyError(family)
    _MODELS[key] = m
    return m


# ── bounds / goals ───────────────────────────────────────────────────────────
def _bounds(family: str, n_wl: int, relax_db: float, iso_db: float):
    """(lb, ub) arrays of shape (n_wl, n_ports) for this problem's targets."""
    info = _FAMILY[family]
    lb = np.asarray(info["lb"], dtype=float)
    ub = np.asarray(info["ub"], dtype=float)
    if lb.shape[0] == 1 and n_wl > 1:
        lb = np.repeat(lb, n_wl, axis=0)
        ub = np.repeat(ub, n_wl, axis=0)
    lb = lb.copy()
    ub = ub.copy()
    f = _db(-relax_db)
    iso = _db(-iso_db)
    for w in range(lb.shape[0]):
        for j in range(lb.shape[1]):
            if lb[w, j] > 0.0:
                lb[w, j] = lb[w, j] * f
                if ub[w, j] < 1.0:          # a window (beam splitter): widen symmetrically
                    ub[w, j] = 1.0 - lb[w, j]
            elif ub[w, j] < 1.0:
                ub[w, j] = iso
    return lb, ub


def _metric_key(j: int, wl: float) -> str:
    return "T%d1_%d" % (j + 1, int(round(wl)))


def _implied_upper(lb_j: float, ub_j: float) -> bool:
    """A window [lb, ub] on one output whose upper edge is 1 - lb (the beam
    splitter's 1 - 10^-0.35) is implied by the lower bounds of the two outputs
    and power conservation (T21 + T31 <= 1), so it is not emitted as a goal:
    the potential labels the min and max constraint of one metric alike and
    its violation dict would hide the lower side."""
    return lb_j > 0.0 and ub_j < 1.0 and ub_j >= 1.0 - lb_j - 1e-9


def _goals_from_bounds(lb, ub, wls) -> dict:
    goals = {}
    for w, wl in enumerate(wls):
        for j in range(lb.shape[1]):
            k = _metric_key(j, wl)
            if lb[w, j] > 0.0:
                goals["minimum_" + k] = float(lb[w, j])
            if ub[w, j] < 1.0 and not _implied_upper(lb[w, j], ub[w, j]):
                goals["maximum_" + k] = float(ub[w, j])
    return goals


# ── fixed per-problem data ───────────────────────────────────────────────────
def _fixed_from_spec(spec: dict) -> dict:
    family = spec["family"]
    wls = tuple(int(w) for w in spec.get("wavelengths_nm", WAVELENGTHS_NM))
    d = int(spec.get("min_feature_px", MINFEAT_DEFAULT))
    m = _model(family, wls)
    W, H = m.design_variable_shape                      # ceviche (i=x, j=y)
    x0, y0, x1, y1 = m.design_region_coords
    M = 2 * d
    bg = np.asarray(m.density_bg, dtype=float).copy()
    bg[x0:x1, y0:y1] = 0.0                              # WdmModel seeds its region with 1
    win = (bg[x0 - M:x1 + M, y0 - M:y1 + M].T > 0.5)   # [y, x], void/solid surroundings
    n_wl = len(wls)
    relax_db = float(spec.get("relax_db", 0.0))
    iso_db = float(spec.get("isolation_db", 20.0))
    lb, ub = _bounds(family, n_wl, relax_db, iso_db)
    info = _FAMILY[family]
    wanted = info["wanted"]
    if len(wanted) == 1 and n_wl > 1:
        wanted = wanted * n_wl
    ports = _port_table(m, x0, y0, x1, y1, W, H)
    return {
        "family": family, "wls": wls, "d": d, "M": M, "W": W, "H": H,
        "model": m, "win": win, "se": np.ones((d, d), dtype=bool),
        "lb": lb, "ub": ub, "wanted": [list(x) for x in wanted],
        "ports": ports, "n_ports": info["n_ports"],
        "key": (family, wls, d, round(relax_db, 3), round(iso_db, 3)),
        "sim_key": (family, wls),
        "tightness": spec.get("tightness", "nominal"),
        "goals": dict(spec.get("goals") or _goals_from_bounds(lb, ub, wls)),
    }


def _port_table(m, x0, y0, x1, y1, W, H) -> List[dict]:
    """Where each port's waveguide meets the design region, in design pixels."""
    bg = np.asarray(m.density_bg) > 0.5
    out = []
    for k, p in enumerate(m.ports):
        px, py = int(p.x), int(p.y)
        if px < x0:
            side, strip = "left", bg[x0 - 1, y0:y1]
        elif px >= x1:
            side, strip = "right", bg[x1, y0:y1]
        elif py < y0:
            side, strip = "bottom", bg[x0:x1, y0 - 1]
        else:
            side, strip = "top", bg[x0:x1, y1]
        centre = (py - y0) if side in ("left", "right") else (px - x0)
        idx = np.nonzero(strip)[0]
        near = idx[np.abs(idx - centre) <= 8] if idx.size else idx
        lo, hi = (int(near.min()), int(near.max())) if near.size else (centre - 5, centre + 4)
        out.append({"port": k + 1, "side": side, "lo": lo, "hi": hi, "order": int(p.order),
                    "axis": "y" if side in ("left", "right") else "x"})
    return out


# ── pattern helpers ──────────────────────────────────────────────────────────
def _rows_to_pat(rows: List[str]) -> np.ndarray:
    return np.array([[c == "1" for c in r] for r in rows], dtype=bool)


def _pat_to_rows(pat: np.ndarray) -> List[str]:
    return ["".join("1" if v else "0" for v in row) for row in pat]


def _padded(fx: dict, pat: np.ndarray) -> np.ndarray:
    M, H, W = fx["M"], fx["H"], fx["W"]
    win = fx["win"].copy()
    win[M:M + H, M:M + W] = pat
    return win


def _drc_maps(fx: dict, pat: np.ndarray):
    """(bad_solid, bad_void) boolean maps over the design region + margin d."""
    M, H, W, d = fx["M"], fx["H"], fx["W"], fx["d"]
    win = _padded(fx, pat)
    se = fx["se"]
    op = ndimage.binary_opening(win, structure=se)
    opv = ndimage.binary_opening(~win, structure=se)
    bad_s = win & ~op
    bad_v = (~win) & ~opv
    sl = (slice(M - d, M + H + d), slice(M - d, M + W + d))
    return bad_s[sl], bad_v[sl]


def _drc_count(fx: dict, pat: np.ndarray) -> int:
    s, v = _drc_maps(fx, pat)
    return int(s.sum() + v.sum())


def _block_edges(n: int, d: int) -> List[int]:
    """Block boundaries 0, d, 2d, ... with a remainder shorter than d merged
    into the last block, so every block is at least d wide."""
    e = list(range(0, n, d))
    if len(e) > 1 and n - e[-1] < d:
        e.pop()
    return e + [n]


def _quantise_blocks(fx: dict, pat: np.ndarray) -> np.ndarray:
    """Majority vote of pat on d-aligned blocks. Every solid pixel then lies
    in a solid d x d square and every cladding pixel in a cladding one, and
    the fixed surroundings (waveguide stubs 10 px wide, cladding elsewhere)
    are themselves wider than d, so the result passes the feature-size rule
    by construction. The coarse fallback of _drc_repair."""
    d, H, W = fx["d"], fx["H"], fx["W"]
    ye, xe = _block_edges(H, d), _block_edges(W, d)
    out = np.zeros_like(pat)
    for i in range(len(ye) - 1):
        for j in range(len(xe) - 1):
            blk = pat[ye[i]:ye[i + 1], xe[j]:xe[j + 1]]
            out[ye[i]:ye[i + 1], xe[j]:xe[j + 1]] = blk.mean() >= 0.5
    return out


def _drc_repair(fx: dict, pat: np.ndarray, iters: int = 12) -> np.ndarray:
    """Open then close with the d x d kernel until the rule holds, then fix
    stragglers pixel by pixel, keeping the best (fewest violations) pattern
    seen; if neither loop converges, snap that best pattern to d-aligned
    blocks (_quantise_blocks, clean by construction). An earlier version
    returned an EMPTY pattern when the two loops did not converge, which
    silently produced a blank reference design for one family."""
    se = fx["se"]
    M, H, W, d = fx["M"], fx["H"], fx["W"], fx["d"]
    cur = pat.copy()
    best, best_n = cur, _drc_count(fx, cur)
    if best_n == 0:
        return cur
    for _ in range(iters):
        win = _padded(fx, cur)
        win = ndimage.binary_opening(win, structure=se)
        win = ndimage.binary_closing(win, structure=se)
        cur = win[M:M + H, M:M + W].copy()
        n = _drc_count(fx, cur)
        if n < best_n:
            best, best_n = cur, n
        if n == 0:
            return cur
    cur = best.copy()
    for _ in range(iters):
        bs, bv = _drc_maps(fx, cur)
        cur = (cur & ~bs[d:d + H, d:d + W]) | bv[d:d + H, d:d + W]
        n = _drc_count(fx, cur)
        if n < best_n:
            best, best_n = cur, n
        if n == 0:
            return cur
    q = _quantise_blocks(fx, best)
    if _drc_count(fx, q) == 0:
        return q
    return np.zeros_like(pat)       # not reachable while d <= the stub width


def _islands(pat: np.ndarray):
    """(labels, order): 4-connected silicon components; order[k] = scipy label
    of island k, islands numbered in reading order of the printed map
    (topmost row first, then leftmost pixel)."""
    lab, n = ndimage.label(pat)
    keys = []
    for L in range(1, n + 1):
        ys, xs = np.nonzero(lab == L)
        top = ys.max()
        keys.append((-int(top), int(xs[ys == top].min()), L))
    keys.sort()
    return lab, [k[2] for k in keys]


def _island_mask(lab, order, k: int) -> np.ndarray:
    return lab == order[k]


def _edt_in(fx: dict, mask: np.ndarray, px: int, grow: bool) -> np.ndarray:
    """Dilate (grow=True) or erode island `mask` by px pixels (Euclidean
    distance, measured in the padded window).

    Dilation grows from the island ALONE: the result is every pixel of the
    design region within px of a pixel of the island, and nothing else. (An
    earlier version measured the distance from the island together with the
    fixed waveguide stubs, which also deposited a px-deep strip across every
    port mouth, far from the island named.)

    Erosion keeps the fixed surroundings as material, so an island joined to
    a waveguide stub is not eroded from the stub side: only its pixels within
    px of cladding are removed."""
    M, H, W = fx["M"], fx["H"], fx["W"]
    if not np.any(mask):
        return np.zeros((H, W), dtype=bool)
    if grow:
        only = np.zeros_like(fx["win"])
        only[M:M + H, M:M + W] = mask
        dist = ndimage.distance_transform_edt(~only)
        out = dist <= px + 1e-9
    else:
        full = fx["win"].copy()                 # stubs and cladding around the region
        full[M:M + H, M:M + W] = mask
        dist = ndimage.distance_transform_edt(full)
        out = full & (dist > px + 1e-9)
    return out[M:M + H, M:M + W]


def _disk_mask(H: int, W: int, cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


# ── the challenge's own window loss (invrs_gym.loss.transmission_loss) ───────
def _target_loss_np(T, lb, ub, npa=np):
    lo = npa.where(lb > 0.0, lb - T, -1.0)
    hi = npa.where(ub < 1.0, T - ub, -1.0)
    dist = npa.maximum(lo, hi)
    win = np.minimum(ub, 1.0) - np.maximum(lb, 0.0)
    dist = dist / float(np.min(win))
    sp = npa.logaddexp(0.0, dist)
    per_wl = npa.sum(sp ** 2, axis=1)          # (l2 norm over ports)^2
    return npa.mean(per_wl)


# ── the domain ───────────────────────────────────────────────────────────────
class PhotonicsDomain(Domain):
    """Pixel-level topology editing of ceviche-challenges photonic devices."""

    name = "photonics"
    native_moves = True
    reference_class = "heuristic"
    max_candidates = 32
    dedupe = True                        # re-draw a start that repeats one built in this process
    verbs = (
        VerbSpec("FILL_RECT", (ArgSpec("x0", "int", "0..W-1"), ArgSpec("y0", "int", "0..H-1"),
                               ArgSpec("x1", "int", ">= x0"), ArgSpec("y1", "int", ">= y0")),
                 "deposit silicon on every pixel with x0 <= x <= x1 and y0 <= y <= y1",
                 phrase="Fill a rectangle"),
        VerbSpec("CLEAR_RECT", (ArgSpec("x0", "int", "0..W-1"), ArgSpec("y0", "int", "0..H-1"),
                                ArgSpec("x1", "int", ">= x0"), ArgSpec("y1", "int", ">= y0")),
                 "etch that rectangle back to cladding", phrase="Clear a rectangle"),
        VerbSpec("FILL_DISK", (ArgSpec("cx", "int", "0..W-1"), ArgSpec("cy", "int", "0..H-1"),
                               ArgSpec("r", "int", ">= 2 pixels")),
                 "deposit silicon on every pixel with (x-cx)^2 + (y-cy)^2 <= r^2",
                 phrase="Fill a disk"),
        VerbSpec("CLEAR_DISK", (ArgSpec("cx", "int", "0..W-1"), ArgSpec("cy", "int", "0..H-1"),
                                ArgSpec("r", "int", ">= 2 pixels")),
                 "etch that disk back to cladding", phrase="Clear a disk"),
        VerbSpec("SHIFT_FEATURE", (ArgSpec("k", "int", "island 0..n_islands-1"),
                                   ArgSpec("dx", "int", "pixels, +x = right"),
                                   ArgSpec("dy", "int", "pixels, +y = up")),
                 "move silicon island k rigidly by (dx, dy); its old pixels become cladding; "
                 "it must stay inside the design region", phrase="Shift an island"),
        VerbSpec("DILATE", (ArgSpec("k", "int", "island 0..n_islands-1"),
                            ArgSpec("px", "int", "1..3")),
                 "widen island k by px pixels on every side (every cladding pixel within "
                 "px of it becomes silicon)", phrase="Dilate an island"),
        VerbSpec("ERODE", (ArgSpec("k", "int", "island 0..n_islands-1"),
                           ArgSpec("px", "int", "1..3")),
                 "narrow island k by px pixels on every side (its pixels within px of "
                 "cladding are etched; a thin island can vanish)", phrase="Erode an island"),
    )

    def __init__(self, stochastic: bool = False):
        super().__init__(False)          # the FDFD solve is deterministic
        self._eval_cache: "OrderedDict[tuple, dict]" = OrderedDict()
        self._grad_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._field_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

    # ── problem generation ───────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        family = _FAMILIES[index % len(_FAMILIES)]
        d = MINFEAT_DEFAULT if rng.random() < 0.75 else MINFEAT_DEFAULT + 1
        wls = tuple(WAVELENGTHS_NM)
        tw = np.asarray([t[3] for t in _TIGHTNESS])
        tname, relax_db, iso_db, _ = _TIGHTNESS[int(rng.choice(len(_TIGHTNESS), p=tw / tw.sum()))]
        spec = {
            "family": family,
            "wavelengths_nm": [int(w) for w in wls],
            "grid_nm": GRID_NM,
            "min_feature_px": int(d),
            "tightness": tname,
            "relax_db": float(relax_db),
            "isolation_db": float(iso_db),
        }
        fx0 = _fixed_from_spec(spec)
        spec["shape_wh"] = [int(fx0["W"]), int(fx0["H"])]
        spec["goals"] = _goals_from_bounds(fx0["lb"], fx0["ub"], wls)
        fx = _fixed_from_spec(spec)

        ref = self._reference_design(family, wls, d)
        ref_pat = _rows_to_pat(ref["rows"])
        ref_metrics = self._metrics_for(fx, ref_pat)
        prog = program_from_goals(spec["goals"], objective_key="loss", objective_sense="min")
        ref_feasible = bool(prog.is_feasible(ref_metrics))

        kw = np.asarray([w for _, w in _START_KINDS])
        for _attempt in range(6):
            kind = _START_KINDS[int(rng.choice(len(_START_KINDS), p=kw / kw.sum()))][0]
            if kind == "feasible" and not ref_feasible:
                kind = "near"
            pat, applied = self._start_pattern(fx, kind, ref_pat, rng)
            m0 = self._metrics_for(fx, pat)
            feas0 = bool(prog.is_feasible(m0))
            if kind in ("near", "mid", "deep") and feas0:
                # perturb until the start is actually infeasible (at most 3 more moves)
                for _ in range(3):
                    more = self._perturb(fx, pat, rng, 1, "deep")
                    if not more:
                        break
                    applied += more
                    m0 = self._metrics_for(fx, pat)
                    feas0 = bool(prog.is_feasible(m0))
                    if not feas0:
                        break
            cell = (family, int(d), tname, tuple(_pat_to_rows(pat)))
            if not self.dedupe or cell not in _BUILT:
                break
            _N_REDRAWS[0] += 1                   # a repeat of a start already built: draw again
        _BUILT.add(cell)
        klass = "feasible" if feas0 else ("near" if kind == "near" else
                                           "mid" if kind in ("mid", "feasible") else "deep")
        spec.update({
            "start_kind": kind,
            "design0": _pat_to_rows(pat),
            "perturbations": applied,
            "ref_design": list(ref["rows"]),
            "ref_loss": float(max(ref_metrics["loss"], REF_LOSS_FLOOR)),
            "ref_metrics": {k: float(v) for k, v in ref_metrics.items()},
            "ref_feasible": ref_feasible,
            "ref_steps": int(ref["steps"]),
            "ref_seed": int(ref["seed"]),
            "initial_metrics": {k: float(v) for k, v in m0.items()},
            "difficulty_class": klass,
        })
        return spec

    # ── starts ───────────────────────────────────────────────────────────────
    def _start_pattern(self, fx, kind, ref_pat, rng):
        """(pattern, moves applied to build it). Every kind depends on the
        rng: the empty and full starts carry one or two seed features drawn
        uniformly from the structured pool (FILL_* on the blank region,
        CLEAR_* on the solid slab; recorded in the spec's perturbations), the
        strip's dimensions are drawn, blobs are noise, and the perturbed
        starts are harmful moves on the reference."""
        H, W = fx["H"], fx["W"]
        applied: List[str] = []
        if kind in ("empty", "full"):
            pat = np.full((H, W), kind == "full", dtype=bool)
            verbs = ("FILL_RECT", "FILL_DISK") if kind == "empty" else ("CLEAR_RECT", "CLEAR_DISK")
            applied = self._random_moves(fx, pat, rng, 1 + int(rng.random() < 0.5), verbs)
            return pat, applied
        if kind == "blobs":
            return self._blobs(fx, rng), applied
        if kind == "strip":
            return self._strip(fx, rng), applied
        pat = ref_pat.copy()
        if kind == "feasible":
            return pat, applied
        n = {"near": 1, "mid": 2 + int(rng.random() < 0.5),
             "deep": 4 + int(rng.integers(0, 3))}[kind]
        applied = self._perturb(fx, pat, rng, n, "near" if kind == "near" else "deep")
        return pat, applied

    def _blobs(self, fx, rng):
        H, W, d = fx["H"], fx["W"], fx["d"]
        for _ in range(8):
            sigma = float(rng.uniform(1.5, 3.5))
            noise = ndimage.gaussian_filter(rng.normal(size=(H, W)), sigma)
            fill = float(rng.uniform(0.3, 0.6))
            thr = np.quantile(noise, 1.0 - fill)
            pat = _drc_repair(fx, noise > thr)
            if 0.05 < pat.mean() < 0.95:
                return pat
        return np.zeros((H, W), dtype=bool)

    def _random_moves(self, fx, pat, rng, n: int, verbs) -> List[str]:
        """Apply n fabricable moves of the given verbs, drawn UNIFORMLY from
        the structured pool (no sensitivity ranking, no field solve), in
        place; the seeded variation of the empty and full starts. Returns
        their formatted text."""
        applied: List[str] = []
        g0 = np.zeros((fx["H"], fx["W"]))
        for _ in range(n):
            cands = [list(c) for c in self._pool(fx, pat, g0) if c[0].verb in verbs]
            if not cands:
                break
            for i in rng.permutation(len(cands)).tolist()[:64]:
                if self._fabricable(fx, pat, cands[i]):
                    pat[...] = cands[i][2]
                    applied.append(self.format_move(cands[i][0]))
                    break
            else:
                break
        return applied

    def _strip(self, fx, rng):
        """The naive engineer's start: 320/400/480 nm bars joining the ports
        (an L for the bend, a straight guide with a widened section for the
        mode converter, a bar and diagonal branches for the splitters and the
        WDM). Every dimension is drawn from the rng, so two strip starts of
        one (family, min feature, tightness) cell do not coincide: the bar
        half-width (4-6 px); for the bend one of a plain corner, a 45-degree
        corner-mirror facet cut from the outer corner, or arms overshooting
        the corner, plus an optional inner-corner fillet; for the converter
        the widened section's side (upper bump, lower bump or symmetric),
        height and extent; for the Y the junction position and an optional
        junction disk."""
        H, W, d = fx["H"], fx["W"], fx["d"]
        half = int(rng.integers(4, 7))           # 4, 5, 6 px: 320, 400, 480 nm guides
        pat = np.zeros((H, W), dtype=bool)
        ports = fx["ports"]
        yy, xx = np.mgrid[0:H, 0:W]

        def centre(p):
            c = 0.5 * (p["lo"] + p["hi"])
            if p["side"] == "left":
                return (-1.0, c)
            if p["side"] == "right":
                return (float(W), c)
            if p["side"] == "bottom":
                return (c, -1.0)
            return (c, float(H))

        def bar(a, b):
            ax, ay = a
            bx, by = b
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            t = np.clip(((xx - ax) * vx + (yy - ay) * vy) / max(L2, 1e-9), 0.0, 1.0)
            dist = np.hypot(xx - (ax + t * vx), yy - (ay + t * vy))
            return dist <= half + 0.01

        def overshoot():
            return int(rng.integers(d, 7))

        fam = fx["family"]
        p1 = centre(ports[0])
        if fam == "waveguide_bend":
            p2 = centre(ports[1])
            cx, cy = p2[0], p1[1]             # the L's corner: in along y = port 1, down along x = port 2
            variant = ("plain", "facet", "overshoot")[int(rng.choice(3, p=(0.25, 0.45, 0.30)))]
            o1 = overshoot() if variant == "overshoot" else 0
            o2 = overshoot() if variant == "overshoot" else 0
            pat |= bar(p1, (cx + o1, cy))
            pat |= bar((cx, cy + o2), p2)
            if variant == "facet":            # 45-degree mirror: cut the outer (top-right) corner
                c = float(rng.uniform(d, 2 * half + 1))
                ox, oy = cx + half, cy + half
                pat &= ~((xx + yy) > (ox + oy - c))
            if rng.random() < 0.5:            # fillet on the inner (bottom-left) corner
                r = float(rng.integers(d, 11))
                ix, iy = cx - half, cy - half
                pat |= ((xx < ix) & (xx > ix - r) & (yy < iy) & (yy > iy - r)
                        & ((xx - (ix - r)) ** 2 + (yy - (iy - r)) ** 2 > r * r))
        elif fam == "mode_converter":
            p2 = centre(ports[1])
            core = bar(p1, p2)
            pat |= core
            ys = np.nonzero(core[:, W // 2])[0]
            top, bot = int(ys.max()), int(ys.min())
            x0 = int(rng.integers(W // 5, W // 2))
            x1 = int(rng.integers(W // 2, 4 * W // 5))
            b = int(rng.integers(2, 9))
            u = rng.random()
            if u < 0.35:                      # bump on the upper side: the asymmetric textbook converter
                pat[top + 1:min(top + 1 + b, H), x0:x1 + 1] = True
            elif u < 0.70:                    # bump on the lower side
                pat[max(bot - b, 0):bot, x0:x1 + 1] = True
            else:                             # symmetric widening: a multimode section
                pat[max(bot - b, 0):min(top + 1 + b, H), x0:x1 + 1] = True
        else:
            outs = [centre(p) for p in ports[1:]] if fam != "beam_splitter" else \
                   [centre(ports[1]), centre(ports[2])]
            xm = float(rng.uniform(0.25, 0.65)) * (W - 1)
            pat |= bar(p1, (xm, p1[1]))
            for o in outs:
                pat |= bar((xm, p1[1]), o)
            if rng.random() < 0.5:            # a disk at the junction feeding the branches
                r = int(rng.integers(half + 1, half + 7))
                pat |= (xx - xm) ** 2 + (yy - p1[1]) ** 2 <= r * r
        return _drc_repair(fx, pat)

    def _perturb(self, fx, pat, rng, n: int, severity: str) -> List[str]:
        """Apply n harmful legal moves in place (ranked by the sensitivity of
        the target loss); returns their formatted text."""
        applied: List[str] = []
        for _ in range(n):
            g = self._sensitivity(fx, pat)
            cands = [list(c) for c in self._pool(fx, pat, g)]
            if not cands:
                break
            cands.sort(key=lambda c: c[1])            # most harmful first
            top = max(4, int(len(cands) * (0.25 if severity == "near" else 0.10)))
            order = rng.permutation(top).tolist()
            done = False
            for i in order:
                move, gain, _ = cands[i]
                if gain >= 0.0:
                    continue
                if self._fabricable(fx, pat, cands[i]):
                    pat[...] = cands[i][2]
                    applied.append(self.format_move(move))
                    done = True
                    break
            if not done:
                break
        return applied

    # ── reference (short gradient run) ───────────────────────────────────────
    def _reference_design(self, family: str, wls, d: int) -> dict:
        """The cached reference for (family, wavelengths, min feature): from
        this process, else from photonics_refs.json next to the module (the
        designs shipped with the module, each the seeded gradient run below),
        else computed now and written back when the directory is writable."""
        key = (family, tuple(int(w) for w in wls), int(d))
        ref = _REFS.get(key)
        if ref is not None:
            return ref
        disk = _load_ref_file()
        skey = "%s|%s|%d" % (family, ",".join(str(w) for w in key[1]), key[2])
        ref = disk.get(skey)
        if ref is None:
            ref = self._optimise_reference(family, wls, d)
            disk[skey] = ref
            _save_ref_file(disk)
        _REFS[key] = ref
        return ref

    def _optimise_reference(self, family: str, wls, d: int) -> dict:
        _backend()
        import autograd
        import autograd.numpy as npa
        from autograd.extend import defvjp, primitive

        spec = {"family": family, "wavelengths_nm": list(wls), "min_feature_px": int(d)}
        fx = _fixed_from_spec(spec)
        m, H, W, M = fx["model"], fx["H"], fx["W"], fx["M"]
        lb, ub = fx["lb"], fx["ub"]

        r = d
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        ker = np.clip(1.0 - np.sqrt(xx ** 2 + yy ** 2) / (r + 1e-9), 0.0, None)
        ker = ker / ker.sum()
        win = fx["win"].astype(float)

        @primitive
        def filt(rho):
            x = win.copy()
            x[M:M + H, M:M + W] = rho
            return ndimage.convolve(x, ker, mode="nearest")[M:M + H, M:M + W]

        def filt_vjp(ans, rho):
            def vjp(g):
                x = np.zeros_like(win)
                x[M:M + H, M:M + W] = g
                return ndimage.convolve(x, ker, mode="constant", cval=0.0)[M:M + H, M:M + W]
            return vjp
        defvjp(filt, filt_vjp)

        def project(rho, beta, eta=0.5):
            return ((npa.tanh(beta * eta) + npa.tanh(beta * (rho - eta)))
                    / (npa.tanh(beta * eta) + npa.tanh(beta * (1.0 - eta))))

        def density(z, beta):
            return project(filt(1.0 / (1.0 + npa.exp(-z))), beta)

        def objective(z, beta):
            s, _ = m.simulate(npa.transpose(density(z, beta)))
            T = npa.abs(s[:, 0, :]) ** 2
            return _target_loss_np(T, lb, ub, npa)

        grad = autograd.grad(objective)
        per = max(1, REF_STEPS // len(REF_BETAS))
        best = None
        total_steps = 0
        for restart in range(REF_RESTARTS):
            seed = stable_seed("photonics_ref", family, *wls, d, restart)
            rng = np.random.default_rng(seed)
            z = rng.normal(0.0, 0.1, size=(H, W))
            mom = np.zeros((H, W))
            vel = np.zeros((H, W))
            k = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for beta in REF_BETAS:
                    for _ in range(per):
                        k += 1
                        g = np.asarray(grad(z, beta))
                        if not np.all(np.isfinite(g)):
                            break
                        mom = 0.9 * mom + 0.1 * g
                        vel = 0.999 * vel + 0.001 * g * g
                        z = z - REF_LR * (mom / (1 - 0.9 ** k)) / (
                            np.sqrt(vel / (1 - 0.999 ** k)) + 1e-8)
                dens = np.asarray(density(z, REF_BETAS[-1]))
            total_steps += k
            # binarise at each threshold, make it fabricable, keep the design
            # with the lowest value of the challenge's own window loss
            for thr in REF_THRESHOLDS:
                pat = _drc_repair(fx, dens > thr)
                metrics = self._metrics_for(fx, pat)
                key = (float(metrics["target_loss"]), float(metrics["loss"]))
                if best is None or key < best[0]:
                    best = (key, pat, metrics, seed, restart, thr)
        _, pat, metrics, seed, restart, thr = best
        return {"rows": _pat_to_rows(pat), "loss": float(metrics["loss"]),
                "metrics": {kk: float(v) for kk, v in metrics.items()},
                "steps": int(total_steps), "seed": int(seed),
                "restart": int(restart), "threshold": float(thr)}

    # ── state ────────────────────────────────────────────────────────────────
    def load(self, spec: dict):
        fx = _fixed_from_spec(spec)
        pat = _rows_to_pat(spec["design0"])
        assert pat.shape == (fx["H"], fx["W"]), "design0 shape mismatch"
        return {"fx": fx, "pat": pat}

    def clone(self, st):
        return {"fx": st["fx"], "pat": st["pat"].copy()}

    # ── simulation ───────────────────────────────────────────────────────────
    def _simulate(self, fx: dict, pat: np.ndarray, want_fields: bool = False):
        """(T (n_wl, n_ports), fields or None), cached per pattern."""
        key = fx["sim_key"] + (pat.tobytes(),)
        hit = self._eval_cache.get(key)
        if hit is not None and (not want_fields or key in self._field_cache):
            self._eval_cache.move_to_end(key)
            return hit, (self._field_cache.get(key) if want_fields else None)
        m = fx["model"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s, fields = m.simulate(pat.T.astype(float))
        T = np.abs(np.asarray(s)[:, 0, :]) ** 2
        self._eval_cache[key] = T
        if len(self._eval_cache) > _EVAL_CACHE_MAX:
            self._eval_cache.popitem(last=False)
        if want_fields:
            f = np.asarray(fields)[:, 0, :, :]
            self._field_cache[key] = f
            if len(self._field_cache) > _FIELD_CACHE_MAX:
                self._field_cache.popitem(last=False)
            return T, f
        return T, None

    def _metrics_for(self, fx: dict, pat: np.ndarray) -> dict:
        try:
            T, _ = self._simulate(fx, pat)
            if not np.all(np.isfinite(T)):
                return self._fail(fx, pat)
        except Exception:
            return self._fail(fx, pat)
        out = {}
        wls = fx["wls"]
        for w, wl in enumerate(wls):
            for j in range(T.shape[1]):
                out[_metric_key(j, wl)] = float(T[w, j])
        wanted = fx["wanted"]
        deliv = float(np.mean([sum(T[w, j] for j in wanted[w]) for w in range(len(wls))]))
        deliv = min(max(deliv, 0.0), 1.0 - 1e-9)
        out["loss"] = float(max(1.0 - deliv, 1e-6))
        out["insertion_loss_db"] = float(-_to_db(max(deliv, 1e-12)))
        out["radiation_loss"] = float(1.0 - np.mean(T.sum(axis=1)))
        out["target_loss"] = float(_target_loss_np(T, fx["lb"], fx["ub"]))
        out["fill_fraction"] = float(pat.mean())
        out["n_islands"] = float(ndimage.label(pat)[1])
        out["min_feature_violations"] = float(_drc_count(fx, pat))
        return out

    def _fail(self, fx: dict, pat: np.ndarray) -> dict:
        out = {}
        for wl in fx["wls"]:
            for j in range(fx["n_ports"]):
                out[_metric_key(j, wl)] = 0.0
        out.update({"loss": 1.0, "insertion_loss_db": 120.0, "radiation_loss": 1.0,
                    "target_loss": 1e6, "fill_fraction": float(pat.mean()),
                    "n_islands": 0.0, "min_feature_violations": 1e6})
        return out

    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        return self._metrics_for(st["fx"], st["pat"])

    # ── sensitivity (adjoint) ────────────────────────────────────────────────
    def _sensitivity(self, fx: dict, pat: np.ndarray) -> np.ndarray:
        """d(target_loss)/d(density) per design pixel, [y, x]; negative where
        filling helps. Cached per (targets, pattern)."""
        key = fx["key"] + (pat.tobytes(),)
        g = self._grad_cache.get(key)
        if g is not None:
            self._grad_cache.move_to_end(key)
            return g
        _backend()
        import autograd
        import autograd.numpy as npa
        m, lb, ub = fx["model"], fx["lb"], fx["ub"]

        def J(dv):
            s, _ = m.simulate(dv)
            T = npa.abs(s[:, 0, :]) ** 2
            return _target_loss_np(T, lb, ub, npa)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = np.asarray(autograd.grad(J)(pat.T.astype(float))).T
            if not np.all(np.isfinite(g)):
                g = np.zeros_like(g)
        except Exception:
            g = np.zeros((fx["H"], fx["W"]))
        self._grad_cache[key] = g
        if len(self._grad_cache) > _GRAD_CACHE_MAX:
            self._grad_cache.popitem(last=False)
        return g

    # ── candidate pool ───────────────────────────────────────────────────────
    def _rect_sizes(self, fx):
        d = fx["d"]
        return [(d, d), (2 * d, d), (d, 2 * d), (2 * d, 2 * d), (3 * d, 2 * d),
                (2 * d, 3 * d), (3 * d, 3 * d), (4 * d, d), (d, 4 * d)]

    def _pool(self, fx: dict, pat: np.ndarray, g: np.ndarray):
        """Every structured candidate with its first-order predicted gain
        (-dJ, positive = helps) and its new pattern (lazily None for rects and
        disks until materialised)."""
        H, W, d = fx["H"], fx["W"], fx["d"]
        void = ~pat
        gv = g * void                      # gain of filling = -sum gv
        gs = g * pat                       # gain of clearing = +sum gs
        Iv = np.zeros((H + 1, W + 1)); Iv[1:, 1:] = gv.cumsum(0).cumsum(1)
        Is = np.zeros((H + 1, W + 1)); Is[1:, 1:] = gs.cumsum(0).cumsum(1)
        Nv = np.zeros((H + 1, W + 1)); Nv[1:, 1:] = void.cumsum(0).cumsum(1)
        out = []
        stride = d
        for (w, h) in self._rect_sizes(fx):
            if w > W or h > H:
                continue
            xs = np.arange(0, W - w + 1, stride)
            ys = np.arange(0, H - h + 1, stride)
            if xs[-1] != W - w:
                xs = np.append(xs, W - w)
            if ys[-1] != H - h:
                ys = np.append(ys, H - h)
            X, Y = np.meshgrid(xs, ys)
            X = X.ravel(); Y = Y.ravel()

            def box(I):
                return I[Y + h, X + w] - I[Y, X + w] - I[Y + h, X] + I[Y, X]
            nv = box(Nv)
            sv = box(Iv)
            ss = box(Is)
            area = w * h
            for i in range(X.size):
                x0, y0 = int(X[i]), int(Y[i])
                if nv[i] > 0:
                    out.append((Move("FILL_RECT", {"x0": x0, "y0": y0, "x1": x0 + w - 1,
                                                   "y1": y0 + h - 1}), float(-sv[i]), None))
                if nv[i] < area:
                    out.append((Move("CLEAR_RECT", {"x0": x0, "y0": y0, "x1": x0 + w - 1,
                                                    "y1": y0 + h - 1}), float(ss[i]), None))
        radii = [2, 3, 4] if max(H, W) <= 48 else [2, 3, 4, 6]
        for r in radii:
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            disk = (xx ** 2 + yy ** 2 <= r * r).astype(float)
            cv = ndimage.convolve(gv, disk, mode="constant", cval=0.0)
            cs = ndimage.convolve(gs, disk, mode="constant", cval=0.0)
            nvd = ndimage.convolve(void.astype(float), disk, mode="constant", cval=0.0)
            nsd = ndimage.convolve(pat.astype(float), disk, mode="constant", cval=0.0)
            for cy in range(r, H - r, stride):
                for cx in range(r, W - r, stride):
                    if nvd[cy, cx] > 0.5:
                        out.append((Move("FILL_DISK", {"cx": cx, "cy": cy, "r": r}),
                                    float(-cv[cy, cx]), None))
                    if nsd[cy, cx] > 0.5:
                        out.append((Move("CLEAR_DISK", {"cx": cx, "cy": cy, "r": r}),
                                    float(cs[cy, cx]), None))
        lab, order = _islands(pat)
        areas = [(int((lab == L).sum()), k) for k, L in enumerate(order)]
        areas.sort(reverse=True)
        for _, k in areas[:12]:
            isl = _island_mask(lab, order, k)
            ys, xs = np.nonzero(isl)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (d, 0), (-d, 0), (0, d), (0, -d)):
                if xs.min() + dx < 0 or xs.max() + dx >= W or ys.min() + dy < 0 or ys.max() + dy >= H:
                    continue
                new = pat & ~isl
                new[ys + dy, xs + dx] = True
                out.append((Move("SHIFT_FEATURE", {"k": k, "dx": dx, "dy": dy}),
                            self._gain(g, pat, new), new))
            for px in (1, 2):
                new = pat | _edt_in(fx, isl, px, True)
                if new.sum() > pat.sum():
                    out.append((Move("DILATE", {"k": k, "px": px}), self._gain(g, pat, new), new))
                er = _edt_in(fx, isl, px, False)
                new = (pat & ~isl) | er
                if new.sum() < pat.sum():
                    out.append((Move("ERODE", {"k": k, "px": px}), self._gain(g, pat, new), new))
        return out

    @staticmethod
    def _gain(g, pat, new) -> float:
        added = new & ~pat
        removed = pat & ~new
        return float(-g[added].sum() + g[removed].sum())

    def _materialise(self, fx, pat, move) -> Optional[np.ndarray]:
        try:
            st = {"fx": fx, "pat": pat.copy()}
            self._apply_unchecked(st, move)
            return st["pat"]
        except ValueError:
            return None

    # share of the candidate budget reserved per verb (ranked by predicted gain
    # within the verb) so that every verb of the grammar is on offer whenever
    # the state admits it; a verb's unused share flows to the best remaining
    # moves of any verb, and the last eighth is a seeded random draw
    _VERB_QUOTA = (("FILL_RECT", 0.19), ("CLEAR_RECT", 0.19), ("FILL_DISK", 0.09),
                   ("CLEAR_DISK", 0.09), ("SHIFT_FEATURE", 0.12), ("DILATE", 0.09),
                   ("ERODE", 0.09))

    def _fabricable(self, fx, pat, entry) -> bool:
        move, _, new = entry
        if new is None:
            new = self._materialise(fx, pat, move)
            entry[2] = new
        return new is not None and bool(np.any(new != pat)) and _drc_count(fx, new) == 0

    def legal_moves(self, st, rng: np.random.Generator,
                    max_candidates: Optional[int] = None) -> List[Move]:
        """The best-predicted fabricable moves of EVERY verb (adjoint
        sensitivity, first order; per-verb quotas, see _VERB_QUOTA) topped up
        with the best remaining moves and a seeded random eighth of the pool,
        every one checked against the feature-size rule. Deterministic in
        (st, rng): the rng is consumed only by the random draw."""
        fx, pat = st["fx"], st["pat"]
        B = int(max_candidates) if max_candidates is not None else int(self.max_candidates)
        g = self._sensitivity(fx, pat)
        pool = [list(c) for c in self._pool(fx, pat, g)]
        pool.sort(key=lambda c: -c[1])
        n_rand = max(2, B // 8)
        n_rank = B - n_rand
        by_verb: Dict[str, List[int]] = {}
        for i, c in enumerate(pool):
            by_verb.setdefault(c[0].verb, []).append(i)
        chosen: List[Move] = []
        used = set()
        for verb, frac in self._VERB_QUOTA:
            q = max(1, int(round(frac * B)))
            got = 0
            for i in by_verb.get(verb, []):
                if got >= q or len(chosen) >= n_rank:
                    break
                if self._fabricable(fx, pat, pool[i]):
                    chosen.append(pool[i][0])
                    used.add(i)
                    got += 1
                else:
                    used.add(i)            # infeasible: never retried below
        for i, c in enumerate(pool):      # top-up from the global ranking
            if len(chosen) >= n_rank:
                break
            if i in used:
                continue
            used.add(i)
            if self._fabricable(fx, pat, c):
                chosen.append(c[0])
        rest = [i for i in range(len(pool)) if i not in used]
        if rest and len(chosen) < B:
            perm = rng.permutation(len(rest)).tolist()
            for pi in perm:
                if len(chosen) >= B:
                    break
                c = pool[rest[pi]]
                if self._fabricable(fx, pat, c):
                    chosen.append(c[0])
        return chosen

    def hint_move(self, st) -> Optional[Move]:
        """Greedy: the fabricable move with the largest predicted first-order
        drop of the target loss."""
        fx, pat = st["fx"], st["pat"]
        g = self._sensitivity(fx, pat)
        pool = [list(c) for c in self._pool(fx, pat, g)]
        pool.sort(key=lambda c: -c[1])
        for c in pool[:64]:
            if self._fabricable(fx, pat, c):
                return c[0]
        return None

    # ── apply ────────────────────────────────────────────────────────────────
    def _apply_unchecked(self, st, move: Move) -> None:
        """Geometry only (bounds and 'changes something'); no feature-size test."""
        fx, pat = st["fx"], st["pat"]
        H, W = fx["H"], fx["W"]
        a = move.args
        try:
            args = {k: int(v) for k, v in a.items()}
        except (TypeError, ValueError):
            raise ValueError("non-integer argument in %s" % self.format_move(move))
        v = move.verb
        new = pat.copy()
        if v in ("FILL_RECT", "CLEAR_RECT"):
            x0, y0, x1, y1 = args["x0"], args["y0"], args["x1"], args["y1"]
            if not (0 <= x0 <= x1 < W and 0 <= y0 <= y1 < H):
                raise ValueError("rectangle outside the %d x %d design region" % (W, H))
            new[y0:y1 + 1, x0:x1 + 1] = (v == "FILL_RECT")
        elif v in ("FILL_DISK", "CLEAR_DISK"):
            cx, cy, r = args["cx"], args["cy"], args["r"]
            if not (0 <= cx < W and 0 <= cy < H) or r < 1:
                raise ValueError("disk centre outside the region or radius < 1")
            new[_disk_mask(H, W, cx, cy, r)] = (v == "FILL_DISK")
        elif v in ("SHIFT_FEATURE", "DILATE", "ERODE"):
            lab, order = _islands(pat)
            k = args["k"]
            if not (0 <= k < len(order)):
                raise ValueError("no island %d (the design has %d)" % (k, len(order)))
            isl = _island_mask(lab, order, k)
            if v == "SHIFT_FEATURE":
                dx, dy = args["dx"], args["dy"]
                if dx == 0 and dy == 0:
                    raise ValueError("zero shift")
                ys, xs = np.nonzero(isl)
                if (xs.min() + dx < 0 or xs.max() + dx >= W
                        or ys.min() + dy < 0 or ys.max() + dy >= H):
                    raise ValueError("island %d would leave the design region" % k)
                new = pat & ~isl
                new[ys + dy, xs + dx] = True
            else:
                px = args["px"]
                if not (1 <= px <= 3):
                    raise ValueError("px must be 1..3")
                if v == "DILATE":
                    new = pat | _edt_in(fx, isl, px, True)
                else:
                    new = (pat & ~isl) | _edt_in(fx, isl, px, False)
        else:
            raise ValueError("domain photonics has no verb %r" % v)
        if not np.any(new != pat):
            raise ValueError("%s changes nothing" % self.format_move(move))
        st["pat"] = new

    def apply(self, st, move: Move) -> None:
        work = {"fx": st["fx"], "pat": st["pat"]}
        self._apply_unchecked(work, move)
        n_bad = _drc_count(st["fx"], work["pat"])
        if n_bad:
            raise ValueError("%s breaks the %d-pixel minimum feature rule (%d pixels)"
                             % (self.format_move(move), st["fx"]["d"], n_bad))
        st["pat"] = work["pat"]

    def move_label(self, st, move: Move) -> str:
        fx, pat = st["fx"], st["pat"]
        try:
            new = self._materialise(fx, pat, move)
        except Exception:
            new = None
        if new is None:
            return ""
        added = int((new & ~pat).sum())
        removed = int((pat & ~new).sum())
        v = move.verb
        if v in ("SHIFT_FEATURE", "DILATE", "ERODE"):
            lab, order = _islands(pat)
            k = int(move.args["k"])
            isl = _island_mask(lab, order, k)
            ys, xs = np.nonzero(isl)
            where = "island %d, %d px, x %d-%d, y %d-%d before this change" % (
                k, int(isl.sum()), xs.min(), xs.max(), ys.min(), ys.max())
            if v == "SHIFT_FEATURE":
                return where
            return "%s; %s %d px" % (where, "adds" if v == "DILATE" else "removes",
                                      added if v == "DILATE" else removed)
        if added and not removed:
            return "adds %d px of silicon" % added
        if removed and not added:
            return "removes %d px of silicon" % removed
        return "adds %d px, removes %d px" % (added, removed)

    # ── program / reference ──────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(spec["goals"], objective_key="loss",
                                  objective_ref=max(self.reference(spec), REF_LOSS_FLOOR),
                                  objective_sense="min", domain=self.name)
        return replace(prog, validity_bounds=(
            ("min_feature_violations", 0.5),
            ("loss", 1.0),
            ("radiation_loss", 1.5),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        ref = spec.get("ref_loss")
        if ref is None:
            fx = _fixed_from_spec(spec)
            rows = spec.get("ref_design")
            if rows is None:
                r = self._reference_design(spec["family"], tuple(spec.get(
                    "wavelengths_nm", WAVELENGTHS_NM)), int(spec.get("min_feature_px", 3)))
                rows = r["rows"]
                spec["ref_design"] = list(rows)
            m = self._metrics_for(fx, _rows_to_pat(rows))
            ref = max(float(m["loss"]), REF_LOSS_FLOOR)
            spec["ref_loss"] = float(ref)
        return float(ref)

    # ── observation ──────────────────────────────────────────────────────────
    def describe(self, st, spec) -> str:
        fx, pat = st["fx"], st["pat"]
        H, W, d = fx["H"], fx["W"], fx["d"]
        fam = fx["family"]
        info = _FAMILY[fam]
        wls = fx["wls"]
        goals = (spec or {}).get("goals") or fx["goals"]
        lb, ub = fx["lb"], fx["ub"]
        ports = fx["ports"]
        out = ["PROBLEM"]
        out.append(
            "Photonic inverse design of a %s on a silicon slab (ceviche-challenges / "
            "invrs-gym lightweight challenge). The device is a %.1f x %.1f um design "
            "region, a grid of %d x %d pixels of %d nm (x = column 0..%d, left to right; "
            "y = row 0..%d, bottom to top, as printed), inside which every pixel is either "
            "silicon (permittivity %.2f, n = 3.5, printed '#') or silica cladding "
            "(permittivity %.2f, n = 1.5, printed '.'). Fixed %d nm (%d px) wide silicon "
            "waveguides lead to the ports. The goal: %s."
            % (info["title"], info["design_nm"][0] / 1000.0, info["design_nm"][1] / 1000.0,
               W, H, GRID_NM, W - 1, H - 1, EPS_SLAB, EPS_CLAD, WG_WIDTH_NM,
               WG_WIDTH_NM // GRID_NM, info["goal"]))
        pl = []
        for p in ports:
            where = {"left": "left edge (x = -1), rows y %d-%d",
                     "right": "right edge (x = %d), rows y %%d-%%d" % W,
                     "bottom": "bottom edge (y = -1), columns x %d-%d",
                     "top": "top edge (y = %d), columns x %%d-%%d" % H}[p["side"]] % (p["lo"], p["hi"])
            mode = "fundamental mode" if p["order"] == 1 else "%s-order mode" % (
                {2: "second", 3: "third"}.get(p["order"], str(p["order"])))
            role = "input" if p["port"] == 1 else "output"
            pl.append("port %d (%s, %s): %s" % (p["port"], role, mode, where))
        out.append("Ports, where their waveguides meet the design region: " + "; ".join(pl) + ".")
        out.append(
            "Physics: 2-D finite-difference frequency-domain solve (ceviche) of the "
            "out-of-plane electric field Ez with absorbing (PML) boundaries; the "
            "fundamental mode is launched into port 1 and each port's 1-D mode monitor "
            "measures the power carried by its mode. Tj1_<wl> = |S_j1|^2 = fraction of the "
            "launched power leaving through port j at wavelength <wl> nm (T11 = reflected "
            "back into port 1). Evaluated at %s nm; every target holds at each wavelength "
            "separately. Light reaches an output only through silicon that guides it "
            "(index contrast 3.5 vs 1.5): a continuous silicon path of about waveguide "
            "width from the input to an output carries power, a gap of cladding across "
            "the path scatters or reflects it, silicon beside the path pulls the mode "
            "sideways, and a wider or narrower section changes the mode's shape and "
            "phase. Smooth, gradual changes of width transmit; abrupt ones reflect."
            % " and ".join(str(w) for w in wls))
        tl = []
        implied = []
        for w, wl in enumerate(wls):
            for j in range(lb.shape[1]):
                k = _metric_key(j, wl)
                if lb[w, j] > 0:
                    tl.append("%s >= %s (%.1f dB)" % (k, _obs_num(lb[w, j]), _to_db(lb[w, j])))
                if ub[w, j] < 1:
                    if _implied_upper(lb[w, j], ub[w, j]):
                        implied.append("%s <= %s" % (k, _obs_num(ub[w, j])))
                    else:
                        tl.append("%s <= %s (%.1f dB)" % (k, _obs_num(ub[w, j]), _to_db(ub[w, j])))
        out.append("Targets (the %s class of this challenge's specification): %s."
                   % (fx["tightness"], "; ".join(tl)))
        if implied:
            out.append("The specification also caps each output (%s); those caps follow from "
                       "the two floors and power conservation (the outputs sum to at most 1), "
                       "so they are not separate constraints." % "; ".join(implied))
        wanted_txt = " and ".join(
            "port %s at %d nm" % ("+".join(str(j + 1) for j in fx["wanted"][w]), wl)
            for w, wl in enumerate(wls))
        out.append(
            "Objective (minimise): loss = 1 - the mean over wavelengths of the power "
            "delivered to the intended output (%s); insertion_loss_db is the same in dB. "
            "radiation_loss = launched power leaving through no port. target_loss = the "
            "challenge's own smooth window loss (sum over ports of softplus(distance to the "
            "target window / smallest window)^2, averaged over wavelengths; near 0 when "
            "every target is met with margin, large when far off). fill_fraction = silicon share of "
            "the region; n_islands = number of silicon islands; min_feature_violations "
            "must stay 0." % wanted_txt)
        out.append(
            "Fabrication rule (minimum feature size %d px = %d nm): every silicon pixel "
            "must lie inside an all-silicon %d x %d square and every cladding pixel inside "
            "an all-cladding %d x %d square, judged together with the fixed waveguides and "
            "cladding just outside the region (a cladding gap narrower than %d px next to a "
            "waveguide end counts). A change that breaks this is rejected and the design "
            "stays as it was." % (d, d * GRID_NM, d, d, d, d, d))
        # the call signatures as the served grammar writes them in the process's
        # spelling (docstring, "Set-parameter spelling")
        call = self.call_form
        out.append(
            "Changes are made with these %s (pixel units): %s "
            "deposits silicon on every pixel with x0 <= x <= x1 and y0 <= y <= y1; "
            "%s etches that rectangle back to cladding; "
            "%s / %s do the same on the disk of pixels "
            "with (x-cx)^2 + (y-cy)^2 <= r^2 (r >= 2); %s moves "
            "silicon island k rigidly by dx pixels in x and dy in y (old pixels become "
            "cladding, it must stay inside the region); %s widens island k by "
            "px pixels (1-3) on every side; %s narrows it by px on every side (a "
            "thin island can vanish). Islands are the 4-connected silicon regions, "
            "numbered in reading order of the map (topmost, then leftmost pixel first) "
            "and renumbered after every change; silicon touching the region edge where a "
            "waveguide enters is connected to that waveguide. Every change must alter at "
            "least one pixel."
            % ("verbs" if spelled_in_verbs(self) else "operations",
               call("FILL_RECT"), call("CLEAR_RECT"), call("FILL_DISK"), call("CLEAR_DISK"),
               call("SHIFT_FEATURE"), call("DILATE"), call("ERODE")))

        # ── DESIGN ──
        out.append("")
        out.append("DESIGN")
        lab, order = _islands(pat)
        out.append("Silicon map (%d columns x %d rows; '#' silicon, '.' cladding; row y "
                   "printed with its index, the top line is y = %d; column ruler below):"
                   % (W, H, H - 1))
        side = {}
        for p in ports:
            if p["side"] in ("left", "right"):
                c = (p["lo"] + p["hi"]) // 2
                side.setdefault(c, []).append("%s port %d" % (
                    "<-" if p["side"] == "left" else "->", p["port"]))
        for y in range(H - 1, -1, -1):
            line = "y=%2d |%s|" % (y, "".join("#" if v else "." for v in pat[y]))
            if y in side:
                line += "  " + ", ".join(side[y])
            out.append(line)
        out.append("      " + "".join(str(x % 10) for x in range(W)) + "   (x, units digit)")
        out.append("      " + "".join((str(x // 10) if x % 10 == 0 else " ") for x in range(W))
                   + "   (x, tens digit)")
        bottom = [p for p in ports if p["side"] in ("bottom", "top")]
        if bottom:
            out.append("Port columns at the %s edge: %s." % (
                bottom[0]["side"], "; ".join("port %d x %d-%d" % (p["port"], p["lo"], p["hi"])
                                            for p in bottom)))
        out.append("fill_fraction %s (%d of %d px silicon); %d island%s:" % (
            _obs_num(pat.mean()), int(pat.sum()), H * W, len(order),
            "" if len(order) == 1 else "s"))
        for k, L in enumerate(order[:24]):
            isl = lab == L
            ys, xs = np.nonzero(isl)
            touch = []
            for p in ports:
                if p["side"] == "left" and np.any((xs == 0) & (ys >= p["lo"]) & (ys <= p["hi"])):
                    touch.append("port %d" % p["port"])
                if p["side"] == "right" and np.any((xs == W - 1) & (ys >= p["lo"]) & (ys <= p["hi"])):
                    touch.append("port %d" % p["port"])
                if p["side"] == "bottom" and np.any((ys == 0) & (xs >= p["lo"]) & (xs <= p["hi"])):
                    touch.append("port %d" % p["port"])
                if p["side"] == "top" and np.any((ys == H - 1) & (xs >= p["lo"]) & (xs <= p["hi"])):
                    touch.append("port %d" % p["port"])
            out.append("  island %d: %d px, x %d-%d, y %d-%d, centre (%s, %s)%s" % (
                k, int(isl.sum()), xs.min(), xs.max(), ys.min(), ys.max(),
                _obs_num(xs.mean()), _obs_num(ys.mean()),
                ("; joined to " + " and ".join(touch)) if touch else "; touches no port"))
        if len(order) > 24:
            out.append("  (+ %d smaller islands)" % (len(order) - 24))

        # ── RESULTS ──
        out.append("")
        try:
            T, fields = self._simulate(fx, pat, want_fields=True)
        except Exception:
            T, fields = None, None
        if T is None or not np.all(np.isfinite(T)):
            out.append("RESULTS: the field solve failed for this design.")
            return "\n".join(out)
        out.append("RESULTS (FDFD solve, port 1 excited)")
        for w, wl in enumerate(wls):
            parts = []
            for j in range(T.shape[1]):
                k = _metric_key(j, wl)
                tag = ""
                if lb[w, j] > 0 and T[w, j] < lb[w, j]:
                    tag = " BELOW target by %s" % _obs_num(lb[w, j] - T[w, j])
                elif ub[w, j] < 1 and T[w, j] > ub[w, j]:
                    if _implied_upper(lb[w, j], ub[w, j]):
                        tag = " (over the %s balance cap by %s: this output takes the other's share)" % (
                            _obs_num(ub[w, j]), _obs_num(T[w, j] - ub[w, j]))
                    else:
                        tag = " ABOVE limit by %s" % _obs_num(T[w, j] - ub[w, j])
                elif lb[w, j] > 0 or ub[w, j] < 1:
                    tag = " (meets target)"
                parts.append("%s = %s (%.1f dB)%s" % (k, _obs_num(T[w, j]), _to_db(T[w, j]), tag))
            out.append("  %d nm: %s; unaccounted (radiated) %s." % (
                wl, "; ".join(parts), _obs_num(1.0 - T[w].sum())))
        m = self._metrics_for(fx, pat)
        out.append("loss %s (insertion loss %s dB); radiation_loss %s; target_loss %s." % (
            _obs_num(m["loss"]), _obs_num(m["insertion_loss_db"]), _obs_num(m["radiation_loss"]),
            _obs_num(m["target_loss"], 4)))
        # field intensity maps
        if fields is not None:
            x0, y0, x1, y1 = fx["model"].design_region_coords
            bx = max(1, int(round(W / 10.0)))
            by = max(1, int(round(H / 10.0)))
            out.append(
                "Field intensity |Ez|^2 inside the region, averaged over %d x %d pixel "
                "blocks and scaled to 99 at the brightest block (rows top to bottom = "
                "high y to low y; block column c covers x %d*c..%d*c+%d, block row r "
                "covers y %d*r..%d*r+%d):" % (bx, by, bx, bx, bx - 1, by, by, by - 1))
            for w, wl in enumerate(wls):
                I = np.abs(fields[w][x0:x1, y0:y1].T) ** 2
                blocks = _block_mean(I, by, bx)
                mx = blocks.max() if blocks.max() > 0 else 1.0
                out.append("  %d nm:" % wl)
                for r in range(blocks.shape[0] - 1, -1, -1):
                    out.append("    y%2d-%2d | %s" % (
                        r * by, min(H - 1, r * by + by - 1),
                        " ".join("%2d" % int(round(99 * v / mx)) for v in blocks[r])))
        # sensitivities
        g = self._sensitivity(fx, pat)
        if np.any(g != 0):
            out.append(
                "Sensitivity of target_loss to each pixel (adjoint solve, first order): "
                "turning cladding pixel (x, y) into silicon changes target_loss by about "
                "s(x, y); turning a silicon pixel into cladding changes it by about -s(x, y). "
                "Sums of s over the same %d x %d blocks (negative = more silicon in that block "
                "lowers target_loss, positive = less silicon there lowers it):" % (bx, by))
            blocks = _block_sum(g, by, bx)
            for r in range(blocks.shape[0] - 1, -1, -1):
                out.append("    y%2d-%2d | %s" % (
                    r * by, min(H - 1, r * by + by - 1),
                    " ".join("%+5s" % _obs_num(v, 2) for v in blocks[r])))
            void = ~pat
            gf = np.where(void, g, np.inf)
            gc = np.where(pat, g, -np.inf)
            fi = np.argsort(gf, axis=None)[:6]
            ci = np.argsort(-gc, axis=None)[:6]
            ftxt = ["(%d, %d) %s" % (i % W, i // W, _obs_num(gf.flat[i], 2))
                    for i in fi if np.isfinite(gf.flat[i])]
            ctxt = ["(%d, %d) %s" % (i % W, i // W, _obs_num(-gc.flat[i], 2))
                    for i in ci if np.isfinite(gc.flat[i])]
            out.append("Single cladding pixels whose filling lowers target_loss most (x, y) "
                       "change: %s." % ("; ".join(ftxt) or "none"))
            out.append("Single silicon pixels whose emptying lowers target_loss most (x, y) "
                       "change: %s." % ("; ".join(ctxt) or "none"))
        return "\n".join(out)


# ── observation helpers ──────────────────────────────────────────────────────
def _block_mean(a: np.ndarray, by: int, bx: int) -> np.ndarray:
    H, W = a.shape
    ny, nx = int(math.ceil(H / by)), int(math.ceil(W / bx))
    out = np.zeros((ny, nx))
    for r in range(ny):
        for c in range(nx):
            out[r, c] = a[r * by:(r + 1) * by, c * bx:(c + 1) * bx].mean()
    return out


def _block_sum(a: np.ndarray, by: int, bx: int) -> np.ndarray:
    H, W = a.shape
    ny, nx = int(math.ceil(H / by)), int(math.ceil(W / bx))
    out = np.zeros((ny, nx))
    for r in range(ny):
        for c in range(nx):
            out[r, c] = a[r * by:(r + 1) * by, c * bx:(c + 1) * bx].sum()
    return out


def _obs_num(x, sig: int = 3) -> str:
    """``sig`` significant figures; tiny values in compact exponent form
    (same rule as the thermal modules)."""
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
