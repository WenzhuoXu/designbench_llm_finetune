"""PMO molecular optimisation: a molecule edited by named chemical operations
under a property oracle, a synthetic-accessibility envelope and an atom budget.

What the system is
------------------
PMO ("Sample Efficiency Matters: A Benchmark for Practical Molecular
Optimization", Gao, Fu, Sun & Coley, NeurIPS 2022 Datasets & Benchmarks) is the
standard benchmark for *oracle-budgeted* molecular design. A task is a property
oracle - a black-box function from a molecule to a score in [0, 1] - and the
benchmark's point is that the oracle is the expensive part: methods are compared
at a fixed budget of oracle calls (10,000 in the paper), not at convergence.
The oracles are served by PyTDC (`tdc.Oracle`, Huang et al., Therapeutics Data
Commons, NeurIPS 2021), and 20 of the 23 PMO oracles come from GuacaMol
(Brown, Fiscato, Segler & Vaucher, J. Chem. Inf. Model. 2019).

The design here is one molecule, held as a canonical SMILES string. Atom indices
are the indices of that canonical SMILES, so they are a pure function of the
state (they are re-derived after every edit, and an edit therefore renumbers the
molecule - the observation always prints the current numbering).

Quantities and units
--------------------
  oracle_score   the task oracle on the current molecule, dimensionless in
                 [0, 1], higher is better. Which oracle is the problem's own
                 (`spec["oracle"]`); what it measures is in ORACLES below and is
                 printed by describe().
  sa_score       synthetic accessibility (Ertl & Schuffenhauer, J. Cheminform.
                 2009): 1 = trivial to make, 10 = essentially unmakeable.
                 Dimensionless; computed from ECFP4 fragment frequencies in PubChem
                 plus ring-complexity, stereo and size penalties.
  heavy_atoms    count of non-hydrogen atoms (the objective: fewer is better).
  mol_weight     g/mol (average molecular mass of the heavy-atom skeleton + H).
  logp           Crippen octanol/water partition coefficient, log10, dimensionless.
  tpsa           topological polar surface area, A^2 (Ertl 2000).
  qed            quantitative estimate of drug-likeness (Bickerton et al., Nat.
                 Chem. 2012), dimensionless [0, 1]; reported for every problem
                 whether or not it is the task oracle.
  n_rings        number of SSSR rings; max_ring_size the largest ring's size.
  rot_bonds      rotatable single bonds (a flexibility proxy).
  invalid        1.0 when the stored SMILES does not parse/sanitise, else 0.0.

Verbs (the field's own operations; typed arguments)
---------------------------------------------------
  ADD_FRAGMENT(atom, fragment)   join a named fragment from the vocabulary below
        to `atom` by a single bond. Legal only where `atom` still carries a
        hydrogen (one H is consumed by the new bond); the fragment attaches
        through atom 0 of its own SMILES, which is the position named in its
        entry (e.g. morpholino attaches through its nitrogen).
  REMOVE_FRAGMENT(k)             cut off removable fragment k - a substituent
        hanging off the core by ONE acyclic single bond, at most MAX_FRAG_ATOMS
        heavy atoms, leaving at least MIN_CORE_ATOMS heavy atoms behind. The
        removable fragments are enumerated and printed in the observation, k is
        the index in that list (stable for a given state).
  SWAP_FRAGMENT(k, fragment)     a compound verb: cut off removable fragment k
        and attach `fragment` at the same attachment atom. One operation, because
        bioisosteric replacement is one medicinal-chemistry move, not two.
  REPLACE_ATOM(atom, element)    change the element of a single atom, keeping
        every bond (C->N in a ring is an aza-scan; C->O/S in a five-ring changes
        the heterocycle). The pool is ELEMENTS_AROMATIC (C, N, O, S) at an
        aromatic position and ELEMENTS_ALIPHATIC elsewhere, the halogens F, Cl
        and Br only at an atom with a single neighbour - a halogen on an internal
        atom has nowhere to put the bonds and does not sanitise.
        Hydrogens are reassigned to the first of {implicit,
        0 explicit, 1 explicit} that sanitises; when none does, the molecule is
        not a molecule and the move is inadmissible.
  ADD_RING_BOND(a, b)            close a ring by a single bond between two atoms
        that both still carry a hydrogen and are RING_SIZES apart along the
        shortest path (5-, 6- or 7-membered closure only: 3- and 4-rings are
        strained and macrocycles are outside the envelope).
  BREAK_BOND(a, b)               delete a RING bond, opening that ring. Only ring
        bonds: cutting an acyclic bond would split the molecule in two, which is
        REMOVE_FRAGMENT's job, and aromatic bonds cannot be cut without
        destroying the ring's aromatic system (such a move does not sanitise and
        is therefore inadmissible).

Fragment vocabulary: FRAGMENTS below, 29 named substituents (alkyl, halogen,
polar donor/acceptor, aromatic and saturated rings), each with the chemistry it
brings. grammar() prints the names; describe() prints names, SMILES, heavy-atom
counts and roles.

Validity
--------
Two separate things, as the brief requires:
  * RDKit sanitisation is the LEGALITY test. apply() performs the edit, runs
    Chem.SanitizeMol, and raises ValueError when the result is not a valid
    molecule (bad valence, a ring that cannot be kekulised, a disconnected
    graph). legal_moves() only returns moves that pass this test, so the ladder
    never applies an illegal move.
  * the validity ENVELOPE (program.validity_bounds) is synthetic accessibility
    and size: sa_score <= SA_ENVELOPE, heavy_atoms <= HA_ENVELOPE, mol_weight <=
    MW_ENVELOPE, max_ring_size <= RING_ENVELOPE, invalid < 0.5. A molecule can be
    a perfectly good molecule and still sit outside it - that is the D9 lesson
    here: an oracle maximised without a makeability bound walks straight into
    fused polycyclic sludge that no chemist can make.
    The four numeric bounds have ONE implementation, `envelope_ok(probe_tuple)`,
    and the generation search, the threshold, the reference and program() all go
    through it. They did not always: the search and the reference filter once
    tested SA and heavy atoms only, and three problems of the seed-0 120
    (molopt_0049, _0088, _0106) therefore got a reference carrying a 9-membered
    ring - a target the episode is forbidden to reach, since the same molecule
    scored as a state is invalid. _probe() now returns mol_weight and
    max_ring_size alongside, so nothing can select on a subset of the envelope.

Problems (100-150 per run; 6 families x the oracle pool)
--------------------------------------------------------
A problem is (oracle, start molecule, threshold, atom objective). The start is
built procedurally: one of START_SCAFFOLDS decorated with 0-4 fragments from the
vocabulary, drawn from the problem's own rng (so no external molecule dataset is
needed and every start is a valid, inside-the-envelope molecule).

The threshold is set from a seeded two-stage search run at generation time:
  stage A  best-first climb on the oracle score (SEARCH_STEPS_A steps, the legal
           move set capped at SEARCH_WIDTH per step);
  stage B  from stage A's best molecule, a trim pass that takes the move removing
           the most heavy atoms while keeping oracle_score at the threshold - the
           same cost-driven second stage thermal2 uses.
With score0 the start's score and best stage A's best INSIDE-THE-ENVELOPE score
(a molecule the search probed and then rejected cannot set the threshold - it
was reachable only by leaving the envelope),
    minimum_oracle_score = score0 + rho * (best - score0),  rho in [0.25, 0.98]
so the start VIOLATES it and the threshold is known to be reachable inside the
episode horizon. 8% of the 120-problem set get a threshold below score0 instead
(a feasible start whose only job is to shed atoms). 34% also get
maximum_sa_score, either just above the start's SA (bounding the search) or
between the reference's and the start's SA (binding at the start), which is what
spreads the initial violation. A start/oracle pair whose stage-A search cannot
improve the score by MIN_GAIN is rejected and redrawn (up to START_TRIES times):
an oracle that is flat around the start carries no ranking signal and would only
make a degenerate problem. Measured over the 120-problem set (seed 0): every
problem is improvable, none ends up with a vacuous (<= 1e-9) threshold, 92.5% of
starts are infeasible, and all 120 starts are inside the envelope.

Reference (reference_class "heuristic"): the fewest heavy atoms among the
molecules visited by that seeded two-stage search that satisfy the problem's
goals AND pass envelope_ok - every bound program() declares, not a subset of
them. It is a heuristic, not an optimum - phi_select beats it on some problems -
and it is recomputable from the spec alone (reference() re-runs the same search
under a stable_seed when the cached value is absent, applying the same filter,
so the cached and recomputed values agree). Measured over the seed-0 set: all
120 references satisfy their own program()'s goals and validity bounds.

Oracle budget
-------------
Oracle values are cached per (oracle, canonical SMILES) for the life of the
process and counted: `domain.budget()` reports unique calls and cache hits per
oracle, `domain.reset_budget()` zeroes them. This is PMO's own accounting - the
budget is spent on DISTINCT molecules - and it is what makes the ladder
affordable: one phi_select turn scores ~60 children, and the same children
recur across turns. Not counted: the single warm-up call _oracle() makes per
oracle per process (see _oracle for why it exists).

Determinism: every oracle here is a deterministic function of the SMILES (a
descriptor sum, a fingerprint similarity, or a frozen sklearn model), so
evaluate() ignores `seed` and this domain is registered deterministic only.

Simplifications (explicit)
--------------------------
* Topology only: a molecule is a 2-D graph. No conformers, no stereochemistry
  (all starts and edits are flat; CIP labels are never set), no tautomers, no
  protonation states, no solvent.
* Start molecules are built from the scaffold x fragment vocabulary, not sampled
  from ZINC 250k as PMO does, so the domain needs no external dataset. The
  starts are therefore simpler and less diverse than ZINC's.
* The verb set is a medicinal-chemistry subset: no ring fusion/expansion/
  contraction, no bond-order changes, no scaffold hopping in one move, no
  stereocentre inversion, no protecting-group chemistry.
* The fragment vocabulary is fixed at 29 and each fragment attaches through one
  fixed atom; a fragment cannot be attached through a different position.
* Synthetic accessibility is Ertl's SA score, a fragment-frequency heuristic, not
  a retrosynthesis search (no RAscore, no AiZynthFinder).
* The generation-time search walks only INSIDE the validity envelope: a child
  outside it is probed (so the oracle call is counted) and then discarded, and
  it can set neither the threshold nor the reference. This bounds how good the
  reference can be - a molecule that would score higher by growing a macrocycle
  is not offered as a target - and it is what makes every threshold reachable
  and every reference legal. It does NOT narrow the action space: legal_moves()
  still offers those moves at episode time, and an agent that takes one simply
  lands outside the envelope and is scored as invalid there.
* legal_moves() offers, per turn, every removable fragment, every breakable ring
  bond, up to N_RING_CLOSE ring closures, up to N_REPLACE_ATOMS x 2 element
  changes, and for attachment/swap the N_CORE_FRAGMENTS ubiquitous small
  substituents at each chosen site plus a seeded sample of the rest of the
  vocabulary (29 fragments x every site is ~150 moves, more than one Phi turn can
  pay for). Every fragment stays reachable from every state, but not all in the
  same turn.
* Two of those windows are DETERMINISTIC, not resampled: REPLACE_ATOM is offered
  only on the first N_REPLACE_ATOMS atoms of a ring-first, index-ordered sort,
  and SWAP_FRAGMENT only on the N_SWAP_FRAGS largest removable fragments. Over
  400 independent draws on CC(=O)Nc1ccc(O)cc1C(F)(F)F the atom set was always
  {4,5,6,7,9,10} (6 of 15 atoms) and the swap set always k in {0,1,2,4} (4 of 8
  fragments). Unlike the fragment vocabulary, these POSITIONS are not recovered
  by re-drawing; only the element / fragment at them is. Every atom is still
  reachable through REPLACE_ATOM after the molecule is renumbered by another
  edit.
* Which verbs a given state admits is a property of the molecule, not a cap: at
  the 120 generated starts ADD_FRAGMENT and REPLACE_ATOM are available at all
  120, ADD_RING_BOND at 106, REMOVE_FRAGMENT and SWAP_FRAGMENT at 101, and
  BREAK_BOND only at 45 (75 starts have no non-aromatic ring bond at all), so
  only 41 of 120 starts admit all six verbs at once. describe() says which are
  unavailable and why.
* The class attribute max_candidates = 72 is the domain's declared per-turn cap,
  but the ladder's top-level policies do not pass a cap (only phi_d2's lookahead
  does, at 24), and legal_moves applies none when max_candidates is None. The
  effective offer is whatever the move budgets above produce: 20-72 moves per
  state (median 52) over 640 walked states, and a state can exceed 72, so the
  declared number is a nominal ceiling, not something legal_moves enforces.
* Halogen changes are legal but rare in the offer: REPLACE_ATOM reaches F / Cl /
  Br only at TERMINAL aliphatic atoms (an internal atom cannot carry a halogen at
  this valence), and only two elements are drawn per atom, so over those 640
  states the elements offered were S 2372, N 1326, O 375, C 276, Cl 31, F 27,
  Br 23. Every element grammar() advertises does get offered; none is advertised
  and unreachable.
* The PMO 10,000-call budget is not enforced as a hard stop; calls are counted
  and reported instead, because the ladder's job here is to produce data, not to
  compete on the benchmark's own leaderboard.

Excluded PMO tasks (recorded here as the contract requires)
-----------------------------------------------------------
Of PMO's 23 oracles, 4 are not used. The measurements below are a 194-molecule
pool of molecules this action space actually reaches (4 procedural starts, each
expanded one and two moves), scored by calling the TDC oracle objects directly
(2026-10-23, PyTDC 1.1.15, scikit-learn 1.5.1):
* `gsk3b` - the oracle cannot be loaded in this environment: unpickling its
  random forest raises ValueError("node array from the pickle has an
  incompatible dtype") under scikit-learn >= 1.3, exactly as for jnk3 below.
  An environment failure, not a design choice. (An earlier note in this file
  recorded gsk3b as "constant 0.0 over the pool, no ranking signal". That
  reading was wrong, and the way it was wrong is worth recording: gsk3b loads
  its model LAZILY inside the scoring function, and tdc.Oracle.__call__ wraps
  the evaluator in a bare `except:` that returns default_property = 0.0, so the
  load error arrived as a perfect run of zeros. _oracle() now warms every
  oracle up through the unwrapped evaluator, so a model that will not load
  raises here instead of quietly becoming a constant objective.)
* `valsartan_smarts` - exactly 0.0 on all 194 (one distinct value). This one IS
  a measurement, not a masked failure: it is pure RDKit SMARTS matching with no
  model file, and it is PMO's hardest task - it requires an exact SMARTS core
  before any score is given. A constraint on a constant carries no ranking
  signal.
* `jnk3` - the oracle cannot be built at all: unpickling its random forest
  raises ValueError("node array from the pickle has an incompatible dtype")
  under scikit-learn >= 1.3. This is an environment failure, not a design
  choice; it, and gsk3b, come back for free if TDC reships the models (or under
  scikit-learn < 1.3).
* `sitagliptin_mpo` - not constant, but flat to within numerical noise: mean
  0.0091, max 0.2513, and most of the 83 distinct values are 1e-6 to 1e-3 (the
  GuacaMol product of desirabilities collapses because the pool is far from the
  C16H15F6N5O isomer term). A threshold of the form score0 + rho*(best - score0)
  then sits in that noise, so this one is excluded on the WEAKEST basis of the
  four and is the first candidate for re-inclusion. (An earlier note in this
  file claimed 97% of the pool scored exactly 0.0; that does not reproduce -
  on this pool nothing scores exactly 0.0 - and has been replaced by the
  figures above.)
The remaining 19 oracles are the pool in ORACLES.

Requirements: rdkit + PyTDC (dataxp/requirements_molopt.txt). Oracle model files
are downloaded once by PyTDC into DATAXP_MOLOPT_ORACLE_DIR (default
~/.dataxp/molopt_oracle); nothing else touches the filesystem and nothing prints.

Set-parameter spelling (docs/dataxp_spellings.md; DATAXP_SPELLING=set_parameter)
--------------------------------------------------------------------------------
The same moves written through the one wrapper tool, by the generic encoding
of dataxp.domain from the `verbs` table (nothing overridden here): element =
the verb's first int argument, operation = the verb in lower case, value =
the one remaining argument as the multitool call writes it.

  ADD_FRAGMENT(12, hydroxyl)    SET_PARAM(12, add_fragment, hydroxyl)
  REMOVE_FRAGMENT(2)            SET_PARAM(2, remove_fragment, 0)
  SWAP_FRAGMENT(2, methoxy)     SET_PARAM(2, swap_fragment, methoxy)
  REPLACE_ATOM(7, N)            SET_PARAM(7, replace_atom, N)
  ADD_RING_BOND(3, 9)           SET_PARAM(3, add_ring_bond, 9)
  BREAK_BOND(3, 4)              SET_PARAM(3, break_bond, 4)

Judgment calls:
* Names (a fragment, an element symbol) are written bare in the value slot,
  as the multitool call writes them; no fragment name contains a comma, a
  bracket or a quote, so the value token is the name itself. REMOVE_FRAGMENT
  has no argument beyond k: its value is the placeholder 0, required exactly.
* REPLACE_ATOM's second argument is stored under the key `element`, which is
  also the name of the tool's first slot. The key is part of the stored
  {verb, args} data and stays; the generated block reads "Replace an atom:
  SET_PARAM(atom, replace_atom, element)  atom: int ...; element: str C, N,
  O, S, F, Cl or Br", and the grammar() trailer says "element (the value of
  replace_atom; not the SET_PARAM element slot, which is the atom) is one
  of: ...".
* The ring verbs' element is atom a (the first int argument); atom b is the
  value.
* The grammar() wrapper's trailer and describe()'s "Operations: ..." sentence
  name the operations through Domain.op_name / call_form ("add_ring_bond
  closes rings of size 5, 6, 7 only; break_bond takes a non-aromatic RING
  bond (cutting a chain bond is remove_fragment)"); the multitool text is
  byte for byte the previous one. VerbSpec.phrase heads each operation block
  ("Attach a fragment: SET_PARAM(atom, add_fragment, fragment)").
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import threading
import types
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..domain import ArgSpec, Domain, Move, VerbSpec, spelled_in_verbs, stable_seed
from ..potential import DesignProgram, program_from_goals

# ── optional dependencies ────────────────────────────────────────────────────
# rdkit at import time (every helper needs it); tdc only when an oracle is first
# asked for, so the module imports in ~1 s and can be read without PyTDC.
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    _RDKIT_ERR: Optional[str] = None
except Exception as exc:                                    # pragma: no cover
    Chem = None                                             # type: ignore
    Descriptors = None                                      # type: ignore
    _RDKIT_ERR = "%s: %s" % (type(exc).__name__, exc)

_REQ = ("pip install -r dataxp/requirements_molopt.txt "
        "(rdkit + PyTDC; see dataxp/domains/molopt.py)")


def _need_rdkit():
    if _RDKIT_ERR is not None:
        raise ImportError("dataxp.domains.molopt needs rdkit (%s). %s"
                          % (_RDKIT_ERR, _REQ))


# ── fragment vocabulary ──────────────────────────────────────────────────────
# (name, SMILES with the attachment at atom 0, what the group brings chemically)
FRAGMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("methyl", "C", "CH3: smallest lipophilic group, fills a small pocket"),
    ("ethyl", "CC", "C2H5: one more carbon of flexible lipophilic bulk"),
    ("isopropyl", "C(C)C", "branched C3, lipophilic bulk with no new rotamer"),
    ("tert_butyl", "C(C)(C)C", "quaternary C4: large bulk, blocks metabolism there"),
    ("trifluoromethyl", "C(F)(F)F", "CF3: strongly electron-withdrawing, metabolically stable"),
    ("hydroxyl", "O", "OH: hydrogen-bond donor and acceptor, raises polarity"),
    ("methoxy", "OC", "OCH3: acceptor, mildly electron-donating, adds no donor H"),
    ("amino", "N", "NH2: donor and weak base"),
    ("methylamino", "NC", "NHCH3: one donor H, more basic than NH2"),
    ("dimethylamino", "N(C)C", "N(CH3)2: basic centre with no donor H"),
    ("nitrile", "C#N", "C#N: small, linear, strongly electron-withdrawing acceptor"),
    ("carboxyl", "C(=O)O", "COOH: acidic, ionised at pH 7, two oxygens"),
    ("carboxamide", "C(=O)N", "C(=O)NH2: donor + acceptor, the commonest drug linker"),
    ("acetyl", "C(=O)C", "C(=O)CH3: ketone acceptor"),
    ("sulfonamide", "S(=O)(=O)N", "SO2NH2: strong acceptor pair, acidic NH"),
    ("nitro", "[N+](=O)[O-]", "NO2: strongly electron-withdrawing; a structural alert"),
    ("fluoro", "F", "F: smallest halogen, blocks metabolism without adding bulk"),
    ("chloro", "Cl", "Cl: lipophilic halogen"),
    ("bromo", "Br", "Br: heavy lipophilic halogen, 80 g/mol on its own"),
    ("phenyl", "c1ccccc1", "benzene ring: 6 heavy atoms of flat aromatic bulk"),
    ("pyridinyl", "c1ccncc1", "pyridine joined at C4: aromatic acceptor, more polar than phenyl"),
    ("imidazolyl", "c1cnc[nH]1", "imidazole: small aromatic donor + acceptor"),
    ("thienyl", "c1cccs1", "thiophene: phenyl bioisostere, 5-membered"),
    ("furanyl", "c1ccco1", "furan: 5-membered aromatic ether"),
    ("cyclopropyl", "C1CC1", "strained C3 ring: rigid small bulk"),
    ("cyclohexyl", "C1CCCCC1", "saturated C6 ring: three-dimensional lipophilic bulk"),
    ("piperidinyl", "N1CCCCC1", "piperidine joined at N: saturated basic amine"),
    ("morpholino", "N1CCOCC1", "morpholine joined at N: polar, solubilising"),
    ("piperazinyl", "N1CCNCC1", "piperazine joined at N: two nitrogens, one still a donor"),
)
FRAG_SMILES: Dict[str, str] = {n: s for n, s, _ in FRAGMENTS}
FRAG_DOC: Dict[str, str] = {n: d for n, _, d in FRAGMENTS}
FRAG_NAMES: Tuple[str, ...] = tuple(n for n, _, _ in FRAGMENTS)
# offered at every attachment / swap site, every turn: the three substituents a
# chemist tries first anywhere
CORE_FRAGMENTS: Tuple[str, ...] = ("methyl", "hydroxyl", "fluoro")
N_CORE_FRAGMENTS = len(CORE_FRAGMENTS)

# REPLACE_ATOM: what an atom may become, by environment
ELEMENTS_AROMATIC: Tuple[str, ...] = ("C", "N", "O", "S")
ELEMENTS_ALIPHATIC: Tuple[str, ...] = ("C", "N", "O", "S", "F", "Cl", "Br")
HALOGENS: Tuple[str, ...] = ("F", "Cl", "Br")   # terminal positions only
ALL_ELEMENTS: Tuple[str, ...] = ("C", "N", "O", "S", "F", "Cl", "Br")

RING_SIZES: Tuple[int, ...] = (5, 6, 7)      # ADD_RING_BOND closures
MAX_FRAG_ATOMS = 8                           # biggest removable fragment
MIN_CORE_ATOMS = 5                           # heavy atoms that must remain

# validity envelope
SA_ENVELOPE = 7.0            # Ertl SA score; above this, unmakeable
HA_ENVELOPE = 60.0           # heavy atoms
MW_ENVELOPE = 800.0          # g/mol
RING_ENVELOPE = 8.0          # largest ring size (no macrocycles)

# what _probe returns for a SMILES that does not parse: outside the envelope on
# sa_score, so envelope_ok() rejects it without needing an `invalid` slot
_PROBE_FAIL: Tuple[float, float, float, float, float] = (0.0, 10.0, 1.0, 1.0, 0.0)


def envelope_ok(v: Sequence[float]) -> bool:
    """Is a _probe tuple inside the validity envelope program() declares?

    `v` is `(oracle_score, sa_score, heavy_atoms, mol_weight, max_ring_size)`,
    exactly as `MolOptDomain._probe` returns it, and the four bounds tested here
    are exactly the four numeric `validity_bounds` of `program()`. The fifth
    bound, `invalid < 0.5`, needs no test: `_probe` returns `_PROBE_FAIL` for a
    SMILES that does not parse, and that tuple fails the SA bound.

    ONE function, used by the generation search's `inside()`, by
    `make_problem`'s reference filter and by `reference()`, so that the three
    cannot drift apart from `program()`. They did drift: before this, the search
    and the reference filter tested only SA and heavy atoms, and three of the
    seed-0 120 problems (molopt_0049, _0088, _0106) got a reference with a
    9-membered ring - a reference outside the envelope the episode is scored
    against, i.e. an unreachable target.
    """
    return (float(v[1]) <= SA_ENVELOPE and float(v[2]) <= HA_ENVELOPE
            and float(v[3]) <= MW_ENVELOPE and float(v[4]) <= RING_ENVELOPE)

# legal_moves budgets (see the module docstring's simplifications)
N_ADD_SITES = 4
N_ADD_FRAGS = 6              # N_CORE_FRAGMENTS + 3 sampled
N_SWAP_FRAGS = 4             # removable fragments offered a replacement
N_SWAP_OPTS = 3              # replacements each
N_REPLACE_ATOMS = 6
N_REPLACE_OPTS = 2
N_RING_CLOSE = 6

# generation-time search
SEARCH_STEPS_A = 8
SEARCH_STEPS_B = 6
SEARCH_WIDTH = 24
MIN_GAIN = 0.015
START_TRIES = 6

# ── start scaffolds ──────────────────────────────────────────────────────────
# (SMILES, name): small ring systems a medicinal chemist would start from. All
# parse and sanitise; all sit well inside the envelope.
START_SCAFFOLDS: Tuple[Tuple[str, str], ...] = (
    ("c1ccccc1", "benzene"),
    ("c1ccncc1", "pyridine"),
    ("c1ccc2[nH]ccc2c1", "indole"),
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
    ("c1ccc2ncccc2c1", "quinoline"),
    ("c1ccc2[nH]cnc2c1", "benzimidazole"),
    ("c1cnc2ccccc2n1", "quinoxaline"),
    ("c1ccsc1", "thiophene"),
    ("c1ccoc1", "furan"),
    ("C1CCNCC1", "piperidine"),
    ("C1CCOCC1", "tetrahydropyran"),
    ("C1CCCCC1", "cyclohexane"),
    ("O=C(O)c1ccccc1", "benzoic acid"),
    ("Nc1ccccc1", "aniline"),
    ("Oc1ccccc1", "phenol"),
    ("CC(=O)Nc1ccccc1", "acetanilide"),
    ("c1ccc(Oc2ccccc2)cc1", "diphenyl ether"),
    ("c1ccc2scnc2c1", "benzothiazole"),
    ("O=S(=O)(N)c1ccccc1", "benzenesulfonamide"),
    ("c1ccc2[nH]c(=O)[nH]c2c1", "benzimidazolone"),
    ("C1CCc2ccccc2C1", "tetralin"),
    ("c1cnc2[nH]ccc2c1", "7-azaindole"),
    ("O=c1cc[nH]c(=O)[nH]1", "uracil"),
)

# ── oracle pool ──────────────────────────────────────────────────────────────
# name -> (family, short title, what it measures - written from the PMO /
# GuacaMol / TDC documentation and printed by describe())
ORACLES: Dict[str, Tuple[str, str, str]] = {
    "qed": ("drug_likeness", "QED drug-likeness",
            "Bickerton et al. (Nat. Chem. 2012): a weighted geometric mean of eight "
            "desirability functions over molecular weight, logP, hydrogen-bond donors "
            "and acceptors, polar surface area, rotatable bonds, aromatic rings and "
            "structural alerts. 1.0 is the centre of oral-drug property space."),
    "drd2": ("target_activity", "DRD2 activity",
             "Olivecrona et al. (J. Cheminform. 2017): the probability that an "
             "RBF-kernel support vector machine (sklearn SVC, probability=True, so a "
             "Platt-scaled probability), trained on ChEMBL dopamine D2 receptor actives "
             "and inactives, calls the molecule active. Its input is the FCFP6 (feature "
             "Morgan, radius 3) COUNT fingerprint summed into 2048 integer buckets by "
             "index modulo 2048 - buckets hold counts, not bits, so a repeated feature "
             "adds rather than saturating. A ligand-based activity model on "
             "pharmacophoric atom types, so it rewards the pharmacophore of known D2 "
             "ligands."),
    "albuterol_similarity": ("similarity", "albuterol similarity",
                             "GuacaMol: FCFP4 (feature Morgan, radius 2) Tanimoto similarity "
                             "to albuterol, the beta-2 agonist bronchodilator, passed through "
                             "ClippedScoreModifier(upper_x=0.75): the score is the similarity "
                             "divided by 0.75 and clipped at 1.0, so it SATURATES at Tanimoto "
                             "0.75 and a score of 1.0 means 'within 0.75 of albuterol on "
                             "pharmacophoric features', not albuterol itself. Feature "
                             "invariants (donor/acceptor/aromatic/halogen rather than element) "
                             "make it a pharmacophore match, not a substructure match."),
    "mestranol_similarity": ("similarity", "mestranol similarity",
                             "GuacaMol: atom-pair (AP) Tanimoto similarity to mestranol, a "
                             "steroidal oestrogen - RDKit atom-pair counts (atom type, atom "
                             "type, topological distance up to 10 bonds), so it rewards the "
                             "fused four-ring steroid skeleton's distance pattern rather than "
                             "local circular environments. Clipped the same way as albuterol "
                             "(upper_x=0.75): saturates at Tanimoto 0.75."),
    "celecoxib_rediscovery": ("similarity", "celecoxib rediscovery",
                              "GuacaMol: ECFP4 Tanimoto similarity to celecoxib, the COX-2 "
                              "inhibitor (a pyrazole with a sulfonamide-phenyl arm, a tolyl "
                              "arm and a CF3). RAW similarity - unlike the albuterol and "
                              "mestranol similarity tasks there is NO score modifier, so "
                              "nothing saturates and 1.0 means celecoxib itself. ECFP4 here "
                              "is rdkit's count-based, unhashed Morgan fingerprint at radius "
                              "2, not a folded bit vector."),
    "troglitazone_rediscovery": ("similarity", "troglitazone rediscovery",
                                 "GuacaMol: ECFP4 Tanimoto similarity to troglitazone, a "
                                 "thiazolidinedione with a chromanol head. RAW similarity on "
                                 "the count-based Morgan radius-2 fingerprint, with no score "
                                 "modifier: 1.0 means troglitazone itself."),
    "thiothixene_rediscovery": ("similarity", "thiothixene rediscovery",
                                "GuacaMol: ECFP4 Tanimoto similarity to thiothixene, a "
                                "thioxanthene antipsychotic carrying a dimethylsulfonamide "
                                "and an N-methylpiperazine. RAW similarity on the count-based "
                                "Morgan radius-2 fingerprint, with no score modifier: 1.0 "
                                "means thiothixene itself."),
    "osimertinib_mpo": ("mpo", "osimertinib MPO",
                        "GuacaMol multi-property objective: the geometric mean of "
                        "desirability functions asking for high FCFP4 similarity to "
                        "osimertinib (clipped at 0.8), ECFP6 similarity to it held BELOW "
                        "~0.85 (so not osimertinib itself), TPSA above ~100 A^2 and logP "
                        "below ~1 - i.e. the same pharmacophore on a different, more "
                        "polar structure."),
    "fexofenadine_mpo": ("mpo", "fexofenadine MPO",
                         "GuacaMol: the geometric mean of three terms - atom-pair (AP) "
                         "Tanimoto similarity to fexofenadine clipped at 0.8 "
                         "(ClippedScoreModifier, so it saturates there), TPSA AT OR ABOVE "
                         "~90 A^2 (MaxGaussianModifier, mu=90, sigma=10) and logP AT OR "
                         "BELOW ~4 (MinGaussianModifier, mu=4, sigma=1). Both descriptor "
                         "terms are one-sided: exactly 1.0 past mu, decaying only on the "
                         "wrong side. Fexofenadine itself is at logP 5.5, so the reference "
                         "drug scores 0.597 on its own objective - the task is the same "
                         "atom-pair skeleton made more polar."),
    "ranolazine_mpo": ("mpo", "ranolazine MPO",
                       "GuacaMol: the geometric mean of four terms - atom-pair (AP) Tanimoto "
                       "similarity to ranolazine clipped at 0.7 (ClippedScoreModifier, so it "
                       "saturates there), logP AT OR ABOVE ~7 and TPSA AT OR ABOVE ~95 A^2 "
                       "(both MaxGaussianModifier, mu=7/sigma=1 and mu=95/sigma=20: the term "
                       "is exactly 1.0 for anything at or above mu and only decays below it, "
                       "so these are floors, not targets), and about ONE fluorine (a full "
                       "Gaussian on the F count, mu=1, sigma=1, so zero and two are both "
                       "penalised to 0.61). The floors are the hard part: ranolazine itself "
                       "sits at logP 2.3, TPSA 74 A^2 and no fluorine, and scores only 0.049 "
                       "on its own objective."),
    "perindopril_mpo": ("mpo", "perindopril MPO",
                        "GuacaMol: the geometric mean of just two terms - RAW ECFP4 Tanimoto "
                        "similarity to perindopril (no score modifier, so no saturation) and "
                        "a Gaussian on the number of AROMATIC rings (mu=2, sigma=0.5, so 2 "
                        "scores 1.0 while 1 or 3 drop to 0.14 and 0 to 0.0003). Perindopril "
                        "itself has NO aromatic ring - it is a perhydroindole - so the two "
                        "terms pull against each other and the reference drug scores only "
                        "0.018 on its own objective: the target cannot be reached by "
                        "rediscovering perindopril."),
    "amlodipine_mpo": ("mpo", "amlodipine MPO",
                       "GuacaMol: the geometric mean of two terms - RAW ECFP4 Tanimoto "
                       "similarity to amlodipine (no score modifier) and a Gaussian on the "
                       "TOTAL ring count (CalcNumRings, rings of any kind, not just aromatic; "
                       "mu=3, sigma=0.5, so 3 scores 1.0 and 2 or 4 drop to 0.14). Amlodipine "
                       "itself has only two rings, so it scores 0.368 on its own objective - "
                       "the task is amlodipine's substructure plus one more ring."),
    "zaleplon_mpo": ("mpo", "zaleplon MPO",
                     "GuacaMol: the geometric mean of RAW ECFP4 Tanimoto similarity to "
                     "zaleplon (no score modifier) and the isomer score for the formula "
                     "C19H17N3O2. Zaleplon's own formula is C17H15N5O, so the formula term "
                     "does NOT describe zaleplon: it asks for two more carbons, two more "
                     "hydrogens, two fewer nitrogens and one more oxygen, and the reference "
                     "drug scores 0.467 on its own objective."),
    "median1": ("median_isomer", "median molecule 1",
                "GuacaMol: the geometric mean of ECFP4 similarity to camphor and to menthol - "
                "two dissimilar targets at once, so the score peaks on a compromise."),
    "median2": ("median_isomer", "median molecule 2",
                "GuacaMol: the geometric mean of ECFP6 Tanimoto similarity to tadalafil "
                "and to sildenafil - TWO targets, both on the same ECFP6 fingerprint and "
                "neither modified. Two large, dissimilar PDE5-era drugs, so the score "
                "peaks on a compromise between them."),
    "isomers_c7h8n2o2": ("median_isomer", "isomer C7H8N2O2",
                         "GuacaMol: how close the molecular formula is to C7H8N2O2, as the "
                         "GEOMETRIC MEAN of five terms - one Gaussian per element in the "
                         "target formula (mu = the wanted count, sigma=1: C 7, H 8, N 2, O 2, "
                         "hydrogens counted explicitly after AddHs) and one Gaussian on the "
                         "TOTAL atom count (mu=19, sigma=2). Being a geometric mean, one badly "
                         "wrong element count drags the whole score down. Note that elements "
                         "ABSENT from the target formula get no term of their own: a sulfur or "
                         "a halogen is only felt through the total-atom term. Rewards an exact "
                         "atom budget, so it fights the heavy-atom objective directly."),
    "isomers_c9h10n2o2pf2cl": ("median_isomer", "isomer C9H10N2O2PF2Cl",
                               "GuacaMol: formula closeness to C9H10N2O2PF2Cl, a deliberately "
                               "odd formula with phosphorus, two fluorines and a chlorine. "
                               "Same machinery as the C7H8N2O2 task but with EIGHT terms in "
                               "the geometric mean: one Gaussian per element (sigma=1; C 9, "
                               "H 10, N 2, O 2, P 1, F 2, Cl 1) and one on the total atom "
                               "count (mu=27, sigma=2). The three rare elements make it the "
                               "harder of the two - a molecule carrying no phosphorus scores "
                               "0.61 on that term alone, and the geometric mean multiplies "
                               "every such miss."),
    "deco_hop": ("scaffold", "decoration hop",
                 "GuacaMol: keep a fixed core while replacing its decorations, scored as the "
                 "UNWEIGHTED MEAN of four terms against the reference inhibitor "
                 "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C - (1) Gobbi 2-D "
                 "pharmacophore fingerprint Tanimoto similarity to it, clipped at 0.85; "
                 "(2) the core [#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12 (a benzo-fused "
                 "pyrimidine, i.e. quinazoline-like, carrying an O substituent) must be "
                 "PRESENT; (3) the sulfone CS([#6])(=O)=O and (4) the amino-benzothiazole "
                 "[#7]-c1ccc2ncsc2c1 - two of the reference's own decorations - must each be "
                 "ABSENT. Each indicator is 0 or 1, so a quarter of the score per term."),
    "scaffold_hop": ("scaffold", "scaffold hop",
                     "GuacaMol, the mirror image of deco_hop against the same reference and "
                     "the harder direction: the unweighted mean of a Gobbi pharmacophore "
                     "similarity (clipped at 0.75) and two SMARTS indicators that REQUIRE the "
                     "propoxy-to-aminobenzothiazole decoration and FORBID deco_hop's "
                     "quinazoline-like core - keep the decorations, replace the scaffold."),
}
EXCLUDED_ORACLES: Dict[str, str] = {
    "gsk3b": "TDC's pickled random forest will not load under scikit-learn >= 1.3 "
             "(ValueError: incompatible tree node dtype), exactly as for jnk3; an "
             "environment failure, not a design choice",
    "valsartan_smarts": "constant 0.0 over this action space, 194/194 "
                        "(needs an exact SMARTS core before it scores anything); "
                        "pure RDKit SMARTS, so this one is a real measurement",
    "jnk3": "TDC's pickled random forest will not load under scikit-learn >= 1.3 "
            "(ValueError: incompatible tree node dtype)",
    "sitagliptin_mpo": "flat to within noise on reachable molecules (mean 0.0091, "
                       "max 0.2513, most values 1e-6..1e-3); weakest of the four "
                       "exclusions and the first candidate for re-inclusion",
}
_FAMILIES: Tuple[str, ...] = ("drug_likeness", "target_activity", "similarity",
                              "mpo", "median_isomer", "scaffold")
FAMILY_ORACLES: Dict[str, Tuple[str, ...]] = {
    fam: tuple(sorted(n for n, (f, _, _) in ORACLES.items() if f == fam))
    for fam in _FAMILIES
}

# ── oracle plumbing: one object per name per process, cached + counted ───────
_ORACLE_OBJ: Dict[str, object] = {}
_ORACLE_CACHE: Dict[Tuple[str, str], float] = {}
# the molecule every oracle is warmed up on (see _oracle): valid, tiny, and
# nothing in the pool refuses it
_WARMUP_SMILES = "Oc1ccccc1"
_ORACLE_CALLS: Dict[str, int] = {}
_ORACLE_HITS: Dict[str, int] = {}
_ORACLE_LOCK = threading.Lock()


def oracle_dir() -> str:
    """Where PyTDC's oracle model files live. PyTDC hardcodes './oracle', so the
    construction below runs with the cwd set here instead of wherever the job
    happens to have been launched."""
    d = os.environ.get("DATAXP_MOLOPT_ORACLE_DIR")
    if not d:
        d = os.path.join(os.path.expanduser("~"), ".dataxp", "molopt_oracle")
    return d


def _import_tdc():
    """Import tdc.Oracle, working around two packaging facts of PyTDC 1.1.x:

    * `tdc.chem_utils.oracle.oracle` does `from rdkit.six import iteritems`;
      rdkit dropped that py2 shim, and the import is inside a bare except that
      turns the failure into "please install rdkit". iteritems is supplied here.
    * the same module imports `pkg_resources`, whose import can re-resolve an
      already-shadowed scipy to a copy built against a different numpy; binding
      scipy.stats.mstats (the symbol tdc wants) first pins the right one.
    """
    if "rdkit.six" not in sys.modules:
        shim = types.ModuleType("rdkit.six")
        shim.iteritems = lambda d, **kw: iter(d.items())   # type: ignore[attr-defined]
        sys.modules["rdkit.six"] = shim
    import scipy.stats.mstats          # noqa: F401  (pin scipy before tdc)
    from tdc import Oracle
    return Oracle


# tdc.chem_utils.oracle.oracle keeps each oracle's prepared state (reference
# fingerprints, SMARTS matchers, score modifiers) in MODULE GLOBALS, built
# lazily on the first call. deco_hop and scaffold_hop are the one pair in this
# pool that share two of those names - `pharmacophor_fp` and
# `scaffold_smarts_scoring` - and they DISAGREE about the second: deco_hop
# REQUIRES the quinazoline-like core (SMARTS_scoring(inverse=False)) while
# scaffold_hop FORBIDS it (inverse=True). In one process, whichever runs second
# overwrites the other's state, and because deco_hop's init guard only tests
# `pharmacophor_fp`, deco_hop called after scaffold_hop raises NameError on
# `deco1_smarts_scoring` - which tdc.Oracle.__call__ swallows into 0.0. Each of
# the two therefore gets a PRIVATE copy of that module: same code, own globals.
_ISOLATED_ORACLES: Tuple[str, ...] = ("deco_hop", "scaffold_hop")


class _PrivateOracle:
    """One tdc oracle function with its own copy of tdc's module globals."""

    def __init__(self, name: str) -> None:
        import importlib.util
        spec = importlib.util.find_spec("tdc.chem_utils.oracle.oracle")
        if spec is None or spec.loader is None:
            raise ImportError("cannot locate tdc.chem_utils.oracle.oracle")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)            # fresh globals, same code
        self.name = name
        self._mod = mod
        self.evaluator_func = getattr(mod, name)

    def __call__(self, smiles: str) -> float:
        return float(self.evaluator_func(smiles))


def _oracle(name: str):
    """The TDC oracle object for `name`, built once per process AND warmed up.

    Two PyTDC behaviours make the warm-up load-bearing, not a nicety:

    * the model-backed oracles (drd2, gsk3b, ...) load their pickled model
      LAZILY, inside the scoring function, from the RELATIVE path
      "oracle/<name>_current.pkl". Building the Oracle under oracle_dir() is
      therefore not enough - the load happens on the first CALL, by which time
      the cwd is back wherever the job was launched and the file is not found.
    * tdc.Oracle.__call__ wraps the evaluator in a bare `except:` and returns
      `default_property` (0.0). So a missing or unloadable model does not raise:
      every molecule silently scores exactly 0.0, which is indistinguishable
      from a real oracle that happens to be hard.

    The warm-up below calls the underlying evaluator - which does NOT swallow -
    once, while the cwd is still oracle_dir(). That both performs the lazy load
    (tdc caches the model in its own module globals, so every later call works
    from any cwd) and turns an unusable oracle into a loud error here instead of
    a constant-zero objective downstream. It costs one uncounted oracle call per
    oracle per process.
    """
    obj = _ORACLE_OBJ.get(name)
    if obj is not None:
        return obj
    _need_rdkit()
    with _ORACLE_LOCK:
        obj = _ORACLE_OBJ.get(name)
        if obj is not None:
            return obj
        try:
            Oracle = _import_tdc()
        except Exception as exc:
            raise ImportError("dataxp.domains.molopt needs PyTDC for oracle %r "
                              "(%s: %s). %s" % (name, type(exc).__name__, exc, _REQ))
        d = oracle_dir()
        os.makedirs(d, exist_ok=True)
        cwd = os.getcwd()
        buf = io.StringIO()
        try:
            os.chdir(d)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                obj = (_PrivateOracle(name) if name in _ISOLATED_ORACLES
                       else Oracle(name=name))
                fn = getattr(obj, "evaluator_func", None)
                if callable(fn):
                    fn(_WARMUP_SMILES)       # may raise: that is the point
                else:
                    obj(_WARMUP_SMILES)
        except Exception as exc:
            raise RuntimeError(
                "molopt oracle %r cannot be used in this environment "
                "(%s: %s). PyTDC would have returned 0.0 for every molecule "
                "instead of failing. %s"
                % (name, type(exc).__name__, str(exc).splitlines()[0], _REQ))
        finally:
            os.chdir(cwd)
        _ORACLE_OBJ[name] = obj
    return obj


def oracle_score(name: str, smiles: str) -> float:
    """Cached, counted oracle call. PMO's budget is spent on DISTINCT molecules,
    so a repeat of a molecule already scored is a cache hit, not a call."""
    key = (name, smiles)
    hit = _ORACLE_CACHE.get(key)
    if hit is not None:
        _ORACLE_HITS[name] = _ORACLE_HITS.get(name, 0) + 1
        return hit
    orc = _oracle(name)
    try:
        val = float(orc(smiles))
    except Exception:
        val = float("nan")
    if not math.isfinite(val):
        val = 0.0
    _ORACLE_CACHE[key] = val
    _ORACLE_CALLS[name] = _ORACLE_CALLS.get(name, 0) + 1
    return val


def budget() -> dict:
    """{oracle: {calls, hits}} since the last reset_budget(), process-wide."""
    names = set(_ORACLE_CALLS) | set(_ORACLE_HITS)
    return {n: {"calls": _ORACLE_CALLS.get(n, 0), "hits": _ORACLE_HITS.get(n, 0)}
            for n in sorted(names)}


def reset_budget(drop_cache: bool = False) -> None:
    _ORACLE_CALLS.clear()
    _ORACLE_HITS.clear()
    if drop_cache:
        _ORACLE_CACHE.clear()


# ── molecule helpers ─────────────────────────────────────────────────────────
def _mol(smiles: str):
    """Parsed, sanitised molecule, or None."""
    _need_rdkit()
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _finish(rw) -> Optional[str]:
    """Sanitise an edited RWMol and return its canonical SMILES, or None when
    the result is not a valid molecule (bad valence, unkekulisable ring,
    disconnected graph)."""
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None                 # an edit must never split the molecule
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _free_h(atom) -> int:
    try:
        return int(atom.GetTotalNumHs())
    except Exception:
        return 0


def _drop_h(atom) -> None:
    """Consume one hydrogen for a new bond. Implicit Hs are recomputed by
    sanitisation; an explicit count ([nH], an aliphatic NH written out) has to
    be decremented by hand."""
    if atom.GetNumExplicitHs() > 0:
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)


def _attach(rw, atom_idx: int, frag_smiles: str, need_h: bool = True) -> bool:
    """Join frag_smiles (through its atom 0) to atom_idx by a single bond,
    in place on the RWMol. False when atom_idx has no hydrogen to give up.

    need_h=False is for the attachment atom of a fragment that was just cut off
    in the same edit: its valence already has room, and its hydrogen count has
    not been recomputed yet (that happens at sanitisation), so the check and the
    decrement would both read a stale number."""
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return False
    host = rw.GetAtomWithIdx(int(atom_idx))
    if need_h:
        if _free_h(host) < 1:
            return False
    n = rw.GetNumAtoms()
    rw.InsertMol(frag)
    rw.AddBond(int(atom_idx), n, Chem.BondType.SINGLE)
    if need_h:
        _drop_h(rw.GetAtomWithIdx(int(atom_idx)))
    _drop_h(rw.GetAtomWithIdx(n))
    return True


def _component(mol, start: int, blocked: Tuple[int, int]) -> set:
    """Atoms reachable from `start` without using the bond `blocked`."""
    seen = {int(start)}
    stack = [int(start)]
    ba, bb = int(blocked[0]), int(blocked[1])
    while stack:
        i = stack.pop()
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if (i == ba and j == bb) or (i == bb and j == ba):
                continue
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def removable_fragments(mol) -> List[dict]:
    """Every substituent that one acyclic single bond holds on: a list of
    {attach, atoms, smiles, n_atoms, bond}, ordered by (attach, first atom,
    size) so index k is stable for a given state."""
    n = mol.GetNumAtoms()
    out: List[dict] = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, b = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        side_b = _component(mol, b, (a, b))
        if a in side_b:                 # not a cut bond after all (shouldn't happen)
            continue
        side_a = set(range(n)) - side_b
        for attach, leaving in ((a, side_b), (b, side_a)):
            if not (1 <= len(leaving) <= MAX_FRAG_ATOMS):
                continue
            if n - len(leaving) < MIN_CORE_ATOMS:
                continue
            atoms = sorted(int(x) for x in leaving)
            try:
                smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms)
            except Exception:
                smi = "?"
            out.append({"attach": int(attach), "atoms": atoms, "smiles": smi,
                        "n_atoms": len(atoms), "bond": (a, b)})
    out.sort(key=lambda d: (d["attach"], d["atoms"][0], d["n_atoms"]))
    return out


def ring_closures(mol) -> List[Tuple[int, int, int]]:
    """(a, b, ring_size) pairs whose single bond would close a ring of an
    allowed size, both ends still carrying a hydrogen. Ordered by ring size
    then indices."""
    free = [a.GetIdx() for a in mol.GetAtoms() if _free_h(a) >= 1]
    out: List[Tuple[int, int, int]] = []
    for ii, a in enumerate(free):
        for b in free[ii + 1:]:
            if mol.GetBondBetweenAtoms(int(a), int(b)) is not None:
                continue
            path = Chem.GetShortestPath(mol, int(a), int(b))
            size = len(path)
            if size in RING_SIZES:
                out.append((int(a), int(b), int(size)))
    out.sort(key=lambda t: (t[2], t[0], t[1]))
    return out


def breakable_bonds(mol) -> List[Tuple[int, int]]:
    """Ring bonds that can be cut without destroying an aromatic system:
    non-aromatic single or double ring bonds. Breaking them opens a ring and
    leaves the molecule connected."""
    out = []
    for bond in mol.GetBonds():
        if not bond.IsInRing() or bond.GetIsAromatic():
            continue
        if bond.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE):
            continue
        out.append((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))
    out.sort()
    return out


def _elements_for(atom) -> Tuple[str, ...]:
    """What this atom may become: aromatic positions keep to the four elements
    that make aromatic rings; a terminal aliphatic atom may also become one of
    the three halogens (the medicinal chemist's halogen scan F / Cl / Br), which
    an internal atom cannot be."""
    sym = atom.GetSymbol()
    if atom.GetIsAromatic():
        pool = ELEMENTS_AROMATIC
    elif atom.GetDegree() <= 1:
        pool = ELEMENTS_ALIPHATIC
    else:
        pool = tuple(e for e in ELEMENTS_ALIPHATIC if e not in HALOGENS)
    return tuple(e for e in pool if e != sym)


# ── the edits (one function per verb; each returns a SMILES or None) ─────────
def _edit_add_fragment(mol, atom: int, fragment: str) -> Optional[str]:
    if fragment not in FRAG_SMILES or not (0 <= atom < mol.GetNumAtoms()):
        return None
    rw = Chem.RWMol(mol)
    if not _attach(rw, atom, FRAG_SMILES[fragment]):
        return None
    return _finish(rw)


def _remove_atoms(mol, atoms: Sequence[int], anchor: int):
    """Delete `atoms` from a copy of mol. Returns (RWMol, the anchor atom's new
    index): deleting atoms shifts every higher index down by one, so the new
    index is the old one minus the number of deleted atoms below it."""
    dead = sorted({int(x) for x in atoms}, reverse=True)
    rw = Chem.RWMol(mol)
    for i in dead:
        rw.RemoveAtom(i)
    shift = sum(1 for i in dead if i < int(anchor))
    return rw, int(anchor) - shift


def _close_removal(rw, idx: int) -> Optional[str]:
    """Sanitise after a removal; when the attachment atom's hydrogens were
    written out explicitly (an aromatic [nH] that had been substituted), the
    hydrogen has to be given back by hand before the ring kekulises."""
    out = _finish(Chem.RWMol(rw))
    if out is not None:
        return out
    if idx < 0:
        return None
    rw2 = Chem.RWMol(rw)
    a = rw2.GetAtomWithIdx(idx)
    a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)
    a.SetNoImplicit(False)
    return _finish(rw2)


def _edit_remove_fragment(mol, k: int) -> Optional[str]:
    frags = removable_fragments(mol)
    if not (0 <= k < len(frags)):
        return None
    f = frags[int(k)]
    rw, idx = _remove_atoms(mol, f["atoms"], f["attach"])
    return _close_removal(rw, idx)


def _edit_swap_fragment(mol, k: int, fragment: str) -> Optional[str]:
    """Cut fragment k off and attach `fragment` at the same atom, as ONE edit:
    the attachment atom's new index is arithmetic, so no intermediate molecule
    has to be built and canonicalised."""
    frags = removable_fragments(mol)
    if not (0 <= k < len(frags)) or fragment not in FRAG_SMILES:
        return None
    f = frags[int(k)]
    rw, idx = _remove_atoms(mol, f["atoms"], f["attach"])
    if idx < 0 or idx >= rw.GetNumAtoms():
        return None
    if not _attach(rw, idx, FRAG_SMILES[fragment], need_h=False):
        return None
    return _finish(rw)


def _edit_replace_atom(mol, atom: int, element: str) -> Optional[str]:
    if element not in ALL_ELEMENTS or not (0 <= atom < mol.GetNumAtoms()):
        return None
    try:
        z = Chem.GetPeriodicTable().GetAtomicNumber(element)
    except Exception:
        return None
    if mol.GetAtomWithIdx(int(atom)).GetSymbol() == element:
        return None
    for explicit_h in (None, 0, 1):
        rw = Chem.RWMol(mol)
        a = rw.GetAtomWithIdx(int(atom))
        a.SetAtomicNum(int(z))
        a.SetFormalCharge(0)
        if explicit_h is None:
            a.SetNoImplicit(False)
        else:
            a.SetNumExplicitHs(int(explicit_h))
            a.SetNoImplicit(False)
        out = _finish(rw)
        if out is not None:
            return out
    return None


def _edit_add_ring_bond(mol, a: int, b: int) -> Optional[str]:
    n = mol.GetNumAtoms()
    a, b = int(a), int(b)
    if not (0 <= a < n and 0 <= b < n) or a == b:
        return None
    if mol.GetBondBetweenAtoms(a, b) is not None:
        return None
    size = len(Chem.GetShortestPath(mol, a, b))
    if size not in RING_SIZES:
        return None
    rw = Chem.RWMol(mol)
    for i in (a, b):
        if _free_h(rw.GetAtomWithIdx(i)) < 1:
            return None
    for i in (a, b):
        _drop_h(rw.GetAtomWithIdx(i))
    rw.AddBond(a, b, Chem.BondType.SINGLE)
    return _finish(rw)


def _edit_break_bond(mol, a: int, b: int) -> Optional[str]:
    a, b = int(a), int(b)
    bond = mol.GetBondBetweenAtoms(a, b) if (0 <= a < mol.GetNumAtoms()
                                             and 0 <= b < mol.GetNumAtoms()) else None
    if bond is None or not bond.IsInRing():
        return None
    rw = Chem.RWMol(mol)
    rw.RemoveBond(a, b)
    return _finish(rw)


_EDITS = {
    "ADD_FRAGMENT": lambda m, g: _edit_add_fragment(m, int(g["atom"]), str(g["fragment"])),
    "REMOVE_FRAGMENT": lambda m, g: _edit_remove_fragment(m, int(g["k"])),
    "SWAP_FRAGMENT": lambda m, g: _edit_swap_fragment(m, int(g["k"]), str(g["fragment"])),
    "REPLACE_ATOM": lambda m, g: _edit_replace_atom(m, int(g["atom"]), str(g["element"])),
    "ADD_RING_BOND": lambda m, g: _edit_add_ring_bond(m, int(g["a"]), int(g["b"])),
    "BREAK_BOND": lambda m, g: _edit_break_bond(m, int(g["a"]), int(g["b"])),
}


def apply_to_smiles(smiles: str, move: Move) -> Optional[str]:
    """The canonical SMILES after `move`, or None when the move is
    inadmissible (unknown verb, bad argument, or a result that does not
    sanitise). The single place the verbs are interpreted."""
    fn = _EDITS.get(move.verb)
    if fn is None:
        return None
    mol = _mol(smiles)
    if mol is None:
        return None
    try:
        return fn(mol, move.args)
    except KeyError:
        return None
    except Exception:
        return None


# ── descriptors ──────────────────────────────────────────────────────────────
def _descriptors(mol) -> dict:
    rings = mol.GetRingInfo().AtomRings()
    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "mol_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "n_rings": float(len(rings)),
        "max_ring_size": float(max((len(r) for r in rings), default=0)),
        "rot_bonds": float(Descriptors.NumRotatableBonds(mol)),
    }


def _formula(mol) -> str:
    try:
        from rdkit.Chem import rdMolDescriptors
        return str(rdMolDescriptors.CalcMolFormula(mol))
    except Exception:
        return "?"


# ── the domain ───────────────────────────────────────────────────────────────
class MolOptDomain(Domain):
    """PMO molecular optimisation with the field's own edit operations."""

    name = "molopt"
    native_moves = True
    reference_class = "heuristic"
    max_candidates = 72
    params = ()

    verbs: Tuple[VerbSpec, ...] = (
        VerbSpec("ADD_FRAGMENT",
                 (ArgSpec("atom", "int", "atom index 0..n_atoms-1 that still carries an H"),
                  ArgSpec("fragment", "str", "a name from the fragment vocabulary")),
                 "attach that fragment to that atom by a single bond", phrase="Attach a fragment"),
        VerbSpec("REMOVE_FRAGMENT",
                 (ArgSpec("k", "int", "index into the removable-fragment list in the observation"),),
                 "cut that substituent off and give the attachment atom its H back",
                 phrase="Remove a fragment"),
        VerbSpec("SWAP_FRAGMENT",
                 (ArgSpec("k", "int", "index into the removable-fragment list"),
                  ArgSpec("fragment", "str", "a name from the fragment vocabulary")),
                 "replace that substituent with the named fragment at the same position",
                 phrase="Swap a fragment"),
        VerbSpec("REPLACE_ATOM",
                 (ArgSpec("atom", "int", "atom index 0..n_atoms-1"),
                  ArgSpec("element", "str", "C, N, O, S, F, Cl or Br")),
                 "change that atom's element, keeping every bond it has", phrase="Replace an atom"),
        VerbSpec("ADD_RING_BOND",
                 (ArgSpec("a", "int", "atom index with a free H"),
                  ArgSpec("b", "int", "atom index with a free H, 5, 6 or 7 atoms away")),
                 "close a 5-, 6- or 7-membered ring with a single bond", phrase="Close a ring"),
        VerbSpec("BREAK_BOND",
                 (ArgSpec("a", "int", "atom index"),
                  ArgSpec("b", "int", "atom index bonded to a by a non-aromatic ring bond")),
                 "delete that ring bond, opening the ring", phrase="Open a ring"),
    )

    def __init__(self, stochastic: bool = False):
        super().__init__(False)       # every oracle here is deterministic
        if stochastic:
            raise ValueError("molopt has no stochastic variant: the PMO oracles are "
                             "deterministic functions of the SMILES")

    # ── problem generation ──────────────────────────────────────────────────
    def make_problem(self, index: int, rng: np.random.Generator) -> dict:
        _need_rdkit()
        family = _FAMILIES[int(index) % len(_FAMILIES)]
        pool = FAMILY_ORACLES[family]
        oracle = pool[int(rng.integers(len(pool)))]

        # pick the start whose neighbourhood the oracle actually grades: the
        # try with the largest stage-A gain, stopping early once a try clears
        # MIN_GAIN. An oracle that is flat around a start cannot carry a
        # threshold and would only make a degenerate problem.
        start = ""
        pool_seen: Dict[str, Tuple[float, float, float, float, float]] = {}
        score0 = best = 0.0
        for _ in range(START_TRIES):
            cand = self._random_start(rng)
            srng = np.random.default_rng(stable_seed("molopt_search", oracle, cand))
            seen = self._search(oracle, cand, srng)
            s0 = seen[cand][0]
            # only molecules INSIDE the envelope may set the threshold: `seen`
            # also holds the children the search probed and rejected, and a
            # threshold taken from one of those is not reachable by a legal
            # episode (it was reachable only by leaving the envelope).
            bst = max((v[0] for v in seen.values() if envelope_ok(v)), default=s0)
            if not pool_seen or (bst - s0) > (best - score0):
                start, pool_seen, score0, best = cand, seen, s0, bst
            if bst - s0 >= MIN_GAIN:
                break
        gain = max(0.0, best - score0)

        u = float(rng.random())
        # a feasible start needs a threshold strictly below its own score, so it
        # is only offered when that score is not already ~0
        if gain < 1e-9 or (u < 0.08 and score0 >= 0.02):
            klass = "feasible"
            thr = 0.9 * score0
        elif u < 0.45:
            klass = "near"
            thr = score0 + float(rng.uniform(0.25, 0.55)) * gain
        else:
            klass = "deep"
            thr = score0 + float(rng.uniform(0.60, 0.98)) * gain
        thr = float(min(max(thr, 0.0), 1.0))

        sa0 = pool_seen[start][1]
        ha0 = pool_seen[start][2]
        # the reference must satisfy the problem's goals AND sit inside the
        # envelope it will be scored against (envelope_ok, all four bounds)
        qualifying = [(v[2], v[1], s) for s, v in sorted(pool_seen.items())
                      if v[0] >= thr - 1e-12 and envelope_ok(v)]
        if not qualifying:            # only the start clears its own threshold
            qualifying = [(ha0, sa0, start)]
        qualifying.sort()
        ref_ha, ref_sa, ref_smiles = qualifying[0]

        goals = {"minimum_oracle_score": float(thr)}
        if float(rng.random()) < 0.40:
            if ref_sa < sa0 - 0.25 and float(rng.random()) < 0.5:
                cap = 0.5 * (ref_sa + sa0)           # binding at the start
            else:
                cap = max(sa0, ref_sa) + float(rng.uniform(0.2, 0.6))
            goals["maximum_sa_score"] = float(min(round(cap, 3), SA_ENVELOPE))
            if ref_sa > goals["maximum_sa_score"]:   # keep the reference legal
                qualifying = [q for q in qualifying if q[1] <= goals["maximum_sa_score"]]
                if qualifying:
                    ref_ha, ref_sa, ref_smiles = qualifying[0]
                else:
                    goals.pop("maximum_sa_score")

        return {
            "family": family,
            "oracle": oracle,
            "oracle_title": ORACLES[oracle][1],
            "start_smiles": start,
            "goals": goals,
            "oracle_score0": float(score0),
            "sa_score0": float(sa0),
            "heavy_atoms0": float(ha0),
            "search_best_score": float(best),
            "search_n_molecules": int(len(pool_seen)),
            "ref_heavy_atoms": float(ref_ha),
            "ref_sa_score": float(ref_sa),
            "ref_smiles": str(ref_smiles),
            "difficulty_class": klass,
            "fragment_vocabulary": list(FRAG_NAMES),
            "improvable": bool(best - score0 >= MIN_GAIN),
        }

    def _random_start(self, rng: np.random.Generator) -> str:
        """A scaffold decorated with 0-4 vocabulary fragments, inside the
        envelope. Falls back to the bare scaffold if a decoration fails."""
        smi, _ = START_SCAFFOLDS[int(rng.integers(len(START_SCAFFOLDS)))]
        for _ in range(int(rng.integers(0, 5))):
            mol = _mol(smi)
            if mol is None:
                break
            free = [int(a.GetIdx()) for a in mol.GetAtoms() if _free_h(a) >= 1]
            if not free:
                break
            atom = free[int(rng.integers(len(free)))]
            frag = FRAG_NAMES[int(rng.integers(len(FRAG_NAMES)))]
            out = _edit_add_fragment(mol, atom, frag)
            if out is None:
                continue
            m2 = _mol(out)
            if m2 is None or m2.GetNumHeavyAtoms() > 0.6 * HA_ENVELOPE:
                continue
            smi = out
        mol = _mol(smi)
        return Chem.MolToSmiles(mol) if mol is not None else smi

    def _probe(self, oracle: str, smiles: str
               ) -> Tuple[float, float, float, float, float]:
        """(oracle score, SA score, heavy atoms, mol weight, max ring size) for
        one molecule: the oracle plus every quantity the validity envelope
        bounds, so that `envelope_ok` can test the whole envelope and not just
        the two cheap bounds."""
        mol = _mol(smiles)
        if mol is None:
            return _PROBE_FAIL
        rings = mol.GetRingInfo().AtomRings()
        return (oracle_score(oracle, smiles), oracle_score("sa", smiles),
                float(mol.GetNumHeavyAtoms()), float(Descriptors.MolWt(mol)),
                float(max((len(r) for r in rings), default=0)))

    def _search(self, oracle: str, start: str,
                rng: np.random.Generator
                ) -> Dict[str, Tuple[float, float, float, float, float]]:
        """The seeded two-stage generation search. Returns every molecule it
        visited, mapped to its `_probe` tuple.

        stage A  best-first climb on the oracle score (ties to fewer atoms),
                 refusing molecules outside the validity envelope;
        stage B  from stage A's best, the move shedding the most heavy atoms
                 that keeps the score at stage A's best (the cost-driven second
                 pass).

        "Outside the validity envelope" is `envelope_ok`, i.e. all four of
        program()'s numeric bounds - SA, heavy atoms, molecular weight and
        largest ring - not just SA and heavy atoms as it was before."""
        seen: Dict[str, Tuple[float, float, float, float, float]] = {}

        def probe(s: str):
            if s not in seen:
                seen[s] = self._probe(oracle, s)
            return seen[s]

        def inside(v) -> bool:
            return envelope_ok(v)

        probe(start)
        cur = start
        for _ in range(SEARCH_STEPS_A):
            moves = self._moves_for(cur, rng, max_candidates=SEARCH_WIDTH)
            best_s, best_v = None, None
            for mv in moves:
                child = apply_to_smiles(cur, mv)
                if child is None:
                    continue
                v = probe(child)
                if not inside(v):
                    continue
                if best_v is None or (v[0], -v[2]) > (best_v[0], -best_v[2]):
                    best_s, best_v = child, v
            if best_s is None or best_v[0] <= seen[cur][0] + 1e-12:
                break
            cur = best_s
        # stage B starts from the best molecule the search is ALLOWED to hold:
        # `seen` also holds the children that were probed and then rejected by
        # inside(), and starting from one of those (or taking its score as the
        # target) would aim the trim pass at a molecule outside the envelope.
        inside_seen = [s for s in sorted(seen) if inside(seen[s])]
        if not inside_seen:
            return seen
        peak = max(inside_seen, key=lambda s: (seen[s][0], -seen[s][2]))
        target = seen[peak][0]
        cur = peak
        for _ in range(SEARCH_STEPS_B):
            moves = self._moves_for(cur, rng, max_candidates=SEARCH_WIDTH)
            best_s, best_v = None, None
            for mv in moves:
                child = apply_to_smiles(cur, mv)
                if child is None:
                    continue
                v = probe(child)
                if not inside(v) or v[0] < target - 1e-12:
                    continue
                if best_v is None or (v[2], v[1]) < (best_v[2], best_v[1]):
                    best_s, best_v = child, v
            if best_s is None or best_v[2] >= seen[cur][2]:
                break
            cur = best_s
        return seen

    # ── state ───────────────────────────────────────────────────────────────
    def load(self, spec: dict):
        _need_rdkit()
        return {"px": _fixed_from_spec(spec), "smiles": str(spec["start_smiles"]),
                "_mol": None}

    def clone(self, st):
        # the cached Mol is only ever read (every edit works on an RWMol copy),
        # so the clone can share it
        return {"px": st["px"], "smiles": st["smiles"], "_mol": st.get("_mol")}

    def mol(self, st):
        m = st.get("_mol")
        if m is None:
            m = _mol(st["smiles"])
            st["_mol"] = m
        return m

    # ── simulation ──────────────────────────────────────────────────────────
    def _fail(self) -> dict:
        """Finite metrics far outside the envelope: a broken state must look
        bad, never missing."""
        return {"oracle_score": 0.0, "sa_score": 10.0, "qed": 0.0,
                "heavy_atoms": 1.0, "mol_weight": 1.0, "logp": 0.0, "tpsa": 0.0,
                "n_rings": 0.0, "max_ring_size": 0.0, "rot_bonds": 0.0,
                "invalid": 1.0}

    def evaluate(self, st, seed: int = 0) -> dict:
        self.calls += 1
        mol = self.mol(st)
        if mol is None:
            return self._fail()
        smi = st["smiles"]
        px = st["px"]
        try:
            out = _descriptors(mol)
            out["oracle_score"] = oracle_score(px["oracle"], smi)
            out["sa_score"] = oracle_score("sa", smi)
            out["qed"] = (out["oracle_score"] if px["oracle"] == "qed"
                          else oracle_score("qed", smi))
            out["invalid"] = 0.0
        except ImportError:
            raise
        except Exception:
            return self._fail()
        for k, v in out.items():
            if not math.isfinite(float(v)):
                return self._fail()
        return {k: float(v) for k, v in out.items()}

    # ── design variables: this domain has none ──────────────────────────────
    def n_elements(self, st) -> int:
        mol = self.mol(st)
        return int(mol.GetNumAtoms()) if mol is not None else 0

    def get(self, st, i: int, p: str) -> Optional[float]:
        return None

    def set(self, st, i: int, p: str, v: float) -> None:
        raise NotImplementedError(
            "molopt has no (element, parameter) addressing: its actions are the "
            "chemical verbs in self.verbs, applied through apply(st, move)")

    def bounds(self, st, p: str) -> Tuple[float, float]:
        raise NotImplementedError("molopt has no tunable parameters")

    # ── moves ───────────────────────────────────────────────────────────────
    def _moves_for(self, smiles: str, rng: np.random.Generator,
                   max_candidates: Optional[int] = None) -> List[Move]:
        """legal_moves on a bare SMILES (used by the generation search too)."""
        mol = _mol(smiles)
        if mol is None:
            return []
        out: List[Move] = []

        # ADD_FRAGMENT: the attachment sites a chemist reaches for first
        # (aromatic CH, then aromatic NH/other aromatic, then aliphatic ring,
        # then chain), three of them plus one drawn from the rest; at each,
        # the core substituents plus a sample of the vocabulary.
        def site_rank(a) -> Tuple[int, int]:
            if a.GetIsAromatic():
                r = 0 if a.GetSymbol() == "C" else 1
            elif a.IsInRing():
                r = 2
            else:
                r = 3
            return (r, int(a.GetIdx()))

        free = sorted((a for a in mol.GetAtoms() if _free_h(a) >= 1), key=site_rank)
        sites = [int(a.GetIdx()) for a in free[:N_ADD_SITES - 1]]
        rest = [int(a.GetIdx()) for a in free[N_ADD_SITES - 1:]]
        if rest:
            sites.append(rest[int(rng.integers(len(rest)))])
        extra_pool = [f for f in FRAG_NAMES if f not in CORE_FRAGMENTS]
        n_extra = max(0, N_ADD_FRAGS - N_CORE_FRAGMENTS)
        for atom in sites:
            picks = list(CORE_FRAGMENTS)
            if n_extra and extra_pool:
                take = rng.choice(len(extra_pool), size=min(n_extra, len(extra_pool)),
                                  replace=False)
                picks += [extra_pool[int(k)] for k in sorted(take.tolist())]
            for frag in picks:
                out.append(Move("ADD_FRAGMENT", {"atom": int(atom), "fragment": frag}))

        # REMOVE_FRAGMENT: all of them; SWAP_FRAGMENT: the biggest few, each
        # offered one core replacement and a sample of the vocabulary
        frags = removable_fragments(mol)
        for k in range(len(frags)):
            out.append(Move("REMOVE_FRAGMENT", {"k": int(k)}))
        order = sorted(range(len(frags)), key=lambda k: (-frags[k]["n_atoms"], k))
        for k in order[:N_SWAP_FRAGS]:
            picks = [CORE_FRAGMENTS[0]]
            pool = [f for f in FRAG_NAMES
                    if f != CORE_FRAGMENTS[0] and FRAG_SMILES[f] != frags[k]["smiles"]]
            if pool:
                take = rng.choice(len(pool), size=min(N_SWAP_OPTS - 1, len(pool)),
                                  replace=False)
                picks += [pool[int(j)] for j in sorted(take.tolist())]
            for frag in picks:
                out.append(Move("SWAP_FRAGMENT", {"k": int(k), "fragment": frag}))

        # REPLACE_ATOM: ring atoms first (an aza-scan is the standard move),
        # two elements each
        atoms = sorted(mol.GetAtoms(),
                       key=lambda a: (0 if a.IsInRing() else 1, int(a.GetIdx())))
        for a in atoms[:N_REPLACE_ATOMS]:
            pool = _elements_for(a)
            if not pool:
                continue
            take = rng.choice(len(pool), size=min(N_REPLACE_OPTS, len(pool)),
                              replace=False)
            for j in sorted(take.tolist()):
                out.append(Move("REPLACE_ATOM", {"atom": int(a.GetIdx()),
                                                 "element": pool[int(j)]}))

        # ADD_RING_BOND / BREAK_BOND
        closures = ring_closures(mol)
        if len(closures) > N_RING_CLOSE:
            keep = rng.choice(len(closures), size=N_RING_CLOSE, replace=False)
            closures = [closures[int(k)] for k in sorted(keep.tolist())]
        for a, b, _size in closures:
            out.append(Move("ADD_RING_BOND", {"a": int(a), "b": int(b)}))
        for a, b in breakable_bonds(mol):
            out.append(Move("BREAK_BOND", {"a": int(a), "b": int(b)}))

        # legality: RDKit sanitisation, and the edit must change the molecule
        legal = []
        for mv in out:
            child = apply_to_smiles(smiles, mv)
            if child is not None and child != smiles:
                legal.append(mv)
        return self.subsample_moves(legal, rng, max_candidates)

    def legal_moves(self, st, rng: np.random.Generator,
                    max_candidates: Optional[int] = None) -> List[Move]:
        return self._moves_for(st["smiles"], rng, max_candidates)

    def apply(self, st, move: Move) -> None:
        out = apply_to_smiles(st["smiles"], move)
        if out is None:
            raise ValueError("inadmissible move %s on %s"
                             % (self.format_move(move), st["smiles"]))
        st["smiles"] = out
        st["_mol"] = None

    def grammar(self) -> str:
        lines = [Domain.grammar(self).rstrip(), "",
                 "Atom indices are the indices of the current molecule printed in the "
                 "observation; they are renumbered after every change.",
                 "fragment is one of these %d names (attachment position in brackets):"
                 % len(FRAG_NAMES)]
        for name, smi, _doc in FRAGMENTS:
            lines.append("    %-16s %s" % (name, smi))
        # the wrapper prose names the operations as the process's spelling
        # does (docstring, "Set-parameter spelling"); under set_parameter the
        # argument `element` of replace_atom is told apart from the tool's
        # element slot
        op = self.op_name
        if spelled_in_verbs(self):
            el_head = "element is one of: %s" % ", ".join(ALL_ELEMENTS)
        else:
            el_head = ("element (the value of %s; not the SET_PARAM element slot, which is the "
                       "atom) is one of: %s" % (op("REPLACE_ATOM"), ", ".join(ALL_ELEMENTS)))
        lines += [el_head + " (aromatic positions accept %s only; "
                  "%s only at an atom with one neighbour)"
                  % (", ".join(ELEMENTS_AROMATIC), ", ".join(HALOGENS)),
                  "%s closes rings of size %s only; %s takes a "
                  "non-aromatic RING bond (cutting a chain bond is %s)."
                  % (op("ADD_RING_BOND"), ", ".join(str(s) for s in RING_SIZES),
                     op("BREAK_BOND"), op("REMOVE_FRAGMENT"))]
        return "\n".join(lines)

    # ── program / reference ─────────────────────────────────────────────────
    def program(self, spec: dict) -> DesignProgram:
        prog = program_from_goals(
            spec["goals"],
            objective_key="heavy_atoms",
            objective_ref=self.reference(spec),
            objective_sense="min",
            domain=self.name,
        )
        return replace(prog, validity_bounds=(
            ("sa_score", SA_ENVELOPE),
            ("heavy_atoms", HA_ENVELOPE),
            ("mol_weight", MW_ENVELOPE),
            ("max_ring_size", RING_ENVELOPE),
            ("invalid", 0.5),
        ))

    def reference(self, spec: dict) -> Optional[float]:
        """Fewest heavy atoms among the searched molecules that meet the goals
        AND pass `envelope_ok` - the same filter `make_problem` applies, and the
        same bounds `program()` declares, so a cached reference and a recomputed
        one agree and neither can sit outside the envelope."""
        ref = spec.get("ref_heavy_atoms")
        if ref is None:
            oracle = str(spec["oracle"])
            start = str(spec["start_smiles"])
            rng = np.random.default_rng(stable_seed("molopt_search", oracle, start))
            seen = self._search(oracle, start, rng)
            thr = float(spec["goals"].get("minimum_oracle_score", 0.0))
            cap = spec["goals"].get("maximum_sa_score")
            cap = float(cap) if cap is not None else SA_ENVELOPE
            ok = sorted((v[2], v[1], s) for s, v in seen.items()
                        if v[0] >= thr - 1e-12 and v[1] <= cap and envelope_ok(v))
            if ok:
                ref = ok[0][0]
            else:
                mol = _mol(start)
                ref = float(mol.GetNumHeavyAtoms()) if mol is not None else 1.0
            spec["ref_heavy_atoms"] = float(ref)
        return float(ref)

    # ── depth-0 greedy ──────────────────────────────────────────────────────
    def hint_move(self, st) -> Optional[Move]:
        """The one-move medicinal-chemistry reflex, with no oracle calls beyond
        the state's own cached properties: when the molecule is over its SA cap
        or already bigger than the reference, cut the largest substituent off;
        otherwise decorate the highest-ranked open position with a small polar
        group (the rotation is keyed to the heavy-atom count, so it is a
        function of the state). The move is checked before it is returned."""
        mol = self.mol(st)
        if mol is None:
            return None
        px = st["px"]
        ha = float(mol.GetNumHeavyAtoms())
        sa = oracle_score("sa", st["smiles"])
        cap = px["goals"].get("maximum_sa_score")
        trim = (cap is not None and sa > float(cap)) or ha > px["ref_heavy_atoms"]
        frags = removable_fragments(mol)
        order: List[Move] = []
        if trim and frags:
            for k in sorted(range(len(frags)), key=lambda k: (-frags[k]["n_atoms"], k)):
                order.append(Move("REMOVE_FRAGMENT", {"k": int(k)}))
        polar = ("hydroxyl", "amino", "nitrile", "methoxy")
        free = sorted((a for a in mol.GetAtoms() if _free_h(a) >= 1),
                      key=lambda a: (0 if (a.GetIsAromatic() and a.GetSymbol() == "C")
                                     else 1, int(a.GetIdx())))
        for a in free[:4]:
            frag = polar[int(ha) % len(polar)]
            order.append(Move("ADD_FRAGMENT", {"atom": int(a.GetIdx()), "fragment": frag}))
        if not trim and frags:
            order.append(Move("REMOVE_FRAGMENT", {"k": 0}))
        for mv in order:
            child = apply_to_smiles(st["smiles"], mv)
            if child is not None and child != st["smiles"]:
                return mv
        return None

    def move_label(self, st, move: Move) -> str:
        """Short role label printed after the move in the observation."""
        mol = self.mol(st)
        a = move.args
        try:
            if move.verb in ("ADD_FRAGMENT",):
                at = mol.GetAtomWithIdx(int(a["atom"]))
                frag = str(a["fragment"])
                return "%s (%s, %d heavy atoms) onto %s" % (
                    frag, FRAG_SMILES.get(frag, "?"),
                    _frag_atoms(frag), _atom_desc(at))
            if move.verb in ("REMOVE_FRAGMENT", "SWAP_FRAGMENT"):
                frags = removable_fragments(mol)
                k = int(a["k"])
                if not (0 <= k < len(frags)):
                    return ""
                f = frags[k]
                at = mol.GetAtomWithIdx(int(f["attach"]))
                base = "fragment %d (%s, %d heavy atoms) at %s" % (
                    k, f["smiles"], f["n_atoms"], _atom_desc(at))
                if move.verb == "SWAP_FRAGMENT":
                    frag = str(a["fragment"])
                    return "%s -> %s (%s, %d heavy atoms)" % (
                        base, frag, FRAG_SMILES.get(frag, "?"), _frag_atoms(frag))
                return base
            if move.verb == "REPLACE_ATOM":
                at = mol.GetAtomWithIdx(int(a["atom"]))
                return "%s -> %s" % (_atom_desc(at), a["element"])
            if move.verb == "ADD_RING_BOND":
                size = len(Chem.GetShortestPath(mol, int(a["a"]), int(a["b"])))
                return "%d-membered closure between %s and %s" % (
                    size, _atom_desc(mol.GetAtomWithIdx(int(a["a"]))),
                    _atom_desc(mol.GetAtomWithIdx(int(a["b"]))))
            if move.verb == "BREAK_BOND":
                return "opens the ring joining %s and %s" % (
                    _atom_desc(mol.GetAtomWithIdx(int(a["a"]))),
                    _atom_desc(mol.GetAtomWithIdx(int(a["b"]))))
        except Exception:
            return ""
        return ""

    # ── observation ─────────────────────────────────────────────────────────
    def describe(self, st, spec) -> str:
        """Policy observation: what PMO is, what this problem's oracle measures,
        the molecule atom by atom with its open positions, the operations the
        current structure admits, and the measured properties. Pure function of
        (st, spec) - the oracle values come from the per-SMILES cache."""
        px = st["px"]
        goals = (spec or {}).get("goals") or px["goals"]
        oracle = px["oracle"]
        fam, title, blurb = ORACLES.get(oracle, ("?", oracle, "an oracle"))
        thr = goals.get("minimum_oracle_score")
        cap = goals.get("maximum_sa_score")
        mol = self.mol(st)
        out = ["PROBLEM"]
        out.append(
            "Molecular optimisation in the PMO benchmark (Gao et al., NeurIPS 2022): one "
            "molecule is edited by named chemical operations to satisfy a property oracle, "
            "and the oracle is the expensive part of the loop - every distinct molecule "
            "scored costs one call of the benchmark's budget.")
        out.append("Oracle for this problem: %s (%s, family %s). %s" % (title, oracle, fam, blurb))
        out.append("It returns a dimensionless score in [0, 1]; higher is better.")
        if thr is not None:
            out.append("Requirement: oracle_score >= %s." % _num(thr))
        out.append(
            "Objective: the FEWEST heavy atoms (non-hydrogen atoms) that meets the "
            "requirement. Synthetic accessibility sa_score (Ertl & Schuffenhauer 2009, 1 = "
            "trivial to make, 10 = unmakeable) is %s, and the design is only inside the "
            "simulator's validity envelope while sa_score <= %s, heavy_atoms <= %s, "
            "mol_weight <= %s g/mol and no ring larger than %s atoms."
            % (("capped at %s for this problem" % _num(cap)) if cap is not None
               else "reported but not capped for this problem",
               _num(SA_ENVELOPE), _num(HA_ENVELOPE), _num(MW_ENVELOPE), _num(RING_ENVELOPE)))
        # the call signatures as the served grammar writes them in the process's
        # spelling (docstring, "Set-parameter spelling")
        call = self.call_form
        out.append(
            "Operations: %s bonds a named fragment to an atom "
            "that still carries a hydrogen; %s cuts substituent k off; "
            "%s replaces substituent k at the same position; "
            "%s changes one atom's element and keeps its bonds "
            "(C, N, O, S anywhere; F, Cl, Br only at an atom with one neighbour); "
            "%s closes a 5-, 6- or 7-membered ring; %s "
            "deletes a non-aromatic ring bond. An operation whose product is not a valid "
            "molecule is not available."
            % (call("ADD_FRAGMENT"), call("REMOVE_FRAGMENT"), call("SWAP_FRAGMENT"),
               call("REPLACE_ATOM"), call("ADD_RING_BOND"), call("BREAK_BOND")))

        if mol is None:
            out += ["", "DESIGN", "  the stored structure %r does not parse" % st["smiles"]]
            return "\n".join(out)

        out += ["", "DESIGN"]
        out.append("SMILES: %s" % st["smiles"])
        out.append("Formula %s; %d heavy atoms, %d rings."
                   % (_formula(mol), mol.GetNumHeavyAtoms(),
                      len(mol.GetRingInfo().AtomRings())))
        out.append("Atoms (index element aromatic/aliphatic, ring, bonds, free H - an atom "
                   "with a free H can take a fragment or close a ring):")
        rows = []
        for a in mol.GetAtoms():
            rows.append("  [%d] %s %s%s, %d bonds, %d free H"
                        % (a.GetIdx(), a.GetSymbol(),
                           "aromatic" if a.GetIsAromatic() else "aliphatic",
                           ", in a ring" if a.IsInRing() else "",
                           a.GetDegree(), _free_h(a)))
        out += rows
        rings = mol.GetRingInfo().AtomRings()
        if rings:
            out.append("Rings: " + "; ".join(
                "ring %d (%d atoms: %s)" % (i, len(r), " ".join(str(x) for x in sorted(r)))
                for i, r in enumerate(rings)))
        frags = removable_fragments(mol)
        if frags:
            out.append("Removable fragments (k: structure, heavy atoms, attachment atom):")
            for k, f in enumerate(frags):
                out.append("  k=%d: %s, %d heavy atoms, on atom [%d] %s"
                           % (k, f["smiles"], f["n_atoms"], f["attach"],
                              _atom_desc(mol.GetAtomWithIdx(int(f["attach"])))))
        else:
            out.append("Removable fragments: none (no substituent is both small enough "
                       "and detachable while leaving %d heavy atoms behind)." % MIN_CORE_ATOMS)
        brk = breakable_bonds(mol)
        out.append("Ring bonds that can be opened: "
                   + (", ".join("%d-%d" % (a, b) for a, b in brk) if brk
                      else "none (every ring bond here is aromatic)"))
        clos = ring_closures(mol)
        out.append("Ring closures available: "
                   + (", ".join("%d-%d (%d-membered)" % (a, b, s) for a, b, s in clos[:12])
                      + (" ..." if len(clos) > 12 else "")
                      if clos else "none"))

        m = self.evaluate(self.clone(st))
        out += ["", "RESULTS (last evaluation of this molecule)"]
        line = "  oracle_score %s" % _num(m["oracle_score"])
        if thr is not None:
            d = float(thr) - m["oracle_score"]
            line += (" - %s SHORT of the %s requirement" % (_num(d), _num(thr))
                     if d > 0 else " - %s above the %s requirement" % (_num(-d), _num(thr)))
        out.append(line)
        out.append("  sa_score %s%s" % (
            _num(m["sa_score"]),
            ("  (cap %s%s)" % (_num(cap), ", EXCEEDED" if m["sa_score"] > float(cap) else ""))
            if cap is not None else ""))
        out.append("  heavy_atoms %s (the objective; the reference design meets the "
                   "requirement with %s)" % (_num(m["heavy_atoms"]),
                                             _num(px["ref_heavy_atoms"])))
        out.append("  mol_weight %s g/mol, logp %s, tpsa %s A^2, qed %s"
                   % (_num(m["mol_weight"]), _num(m["logp"]), _num(m["tpsa"]),
                      _num(m["qed"])))
        out.append("  rings %s (largest %s atoms), rotatable bonds %s"
                   % (_num(m["n_rings"]), _num(m["max_ring_size"]), _num(m["rot_bonds"])))
        out.append("Starting molecule for this problem: %s (oracle_score %s, sa_score %s, "
                   "%s heavy atoms)." % (px["start_smiles"], _num(px["oracle_score0"]),
                                         _num(px["sa_score0"]), _num(px["heavy_atoms0"])))
        return "\n".join(out)

    # ── budget bookkeeping (PMO's own accounting) ───────────────────────────
    @staticmethod
    def budget() -> dict:
        return budget()

    @staticmethod
    def reset_budget(drop_cache: bool = False) -> None:
        reset_budget(drop_cache)


# ── small helpers ────────────────────────────────────────────────────────────
def _fixed_from_spec(spec: dict) -> dict:
    """The per-problem constants a state needs (shared, never mutated)."""
    return {
        "oracle": str(spec["oracle"]),
        "goals": {str(k): float(v) for k, v in (spec.get("goals") or {}).items()},
        "start_smiles": str(spec["start_smiles"]),
        "oracle_score0": float(spec.get("oracle_score0", 0.0)),
        "sa_score0": float(spec.get("sa_score0", 0.0)),
        "heavy_atoms0": float(spec.get("heavy_atoms0", 0.0)),
        "ref_heavy_atoms": float(spec.get("ref_heavy_atoms", 0.0)),
        "family": str(spec.get("family", "")),
    }


_FRAG_ATOM_COUNT: Dict[str, int] = {}


def _frag_atoms(name: str) -> int:
    if name not in _FRAG_ATOM_COUNT:
        mol = _mol(FRAG_SMILES.get(name, ""))
        _FRAG_ATOM_COUNT[name] = int(mol.GetNumHeavyAtoms()) if mol is not None else 0
    return _FRAG_ATOM_COUNT[name]


def _atom_desc(atom) -> str:
    return "atom [%d] %s %s%s" % (
        atom.GetIdx(), atom.GetSymbol(),
        "aromatic" if atom.GetIsAromatic() else "aliphatic",
        " in a ring" if atom.IsInRing() else "")


def _num(x, sig: int = 4) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(f):
        return str(f)
    if f == int(f) and abs(f) < 1e9:
        return str(int(f))
    return ("%." + str(sig) + "g") % f
