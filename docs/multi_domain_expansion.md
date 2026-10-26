# Multi-domain expansion: complexity × size × stochasticity

**Written 2026-09-21.** Scope decision: expand the Φ-guided data-generation pipeline across
*many* domains simultaneously, spanning three axes — constraint coupling (complexity),
problem size, and stochasticity — rather than onboarding one domain at a time.

This plan is constrained by three results already in the ledger and turn5:

1. **Φ_v2 transfers by convention** (battery: zero potential code, 25% → 100%), but the
   *scales* (τ, α, offset) must be derived per problem — both derivations are automatic
   (`with_violation_scale`, the offset-enabling condition).
2. **The value of Φ-selection scales with constraint coupling** (truss 72% greedy vs battery
   25% greedy). Coupling is *measurable*: the feedback-sufficiency statistic (fraction of
   visited states where the depth-0 feedback field is a sufficient statistic for the depth-1
   argmax; 65.8% on truss).
3. **The data format that transfers a decision rule is preference pairs at Φ-ranked states,
   not solved trajectories** (T6 SFT distillation doubled regret; DPO on pairs moved ranking
   0.724 vs 0.50 ceiling). Pairs also harvest from failure states.

---

## 1. The confound problem, and the design that avoids it

If complexity, size, and stochasticity each arrive bundled inside a different new domain,
any cross-domain comparison confounds all three with domain identity. The design rule:

> **Vary an axis *within* a domain wherever the simulator permits; use *new* domains only
> for what cannot be varied within an existing one.**

Concretely:

- **Stochasticity is always a paired variant of a deterministic domain**, never a new domain
  on its own. Truss with sampled load ensembles vs the same truss deterministic; circuit
  sizing with Monte-Carlo device mismatch vs nominal; SEIR with binomial-chain transmission
  vs the ODE. The noise effect is then measured paired, on identical problems.
- **Size is a ladder inside domains that have a natural one**: grid dispatch on the IEEE
  14 → 30 → 118 → 300 bus cases; trusses/frames by member count; epidemic models by number
  of age/region compartments.
- **Coupling is what new domains are for** — and every domain gets *placed* on the axis by
  running the feedback-sufficiency probe, not by intuition.

The headline analysis this enables (pre-registered, ledger Turn 11 style): regress
**Φ-selection uplift over the greedy/depth-0 heuristic** against measured coupling, size,
and noise level across all domain-variants. The existing prediction is that uplift rises
with coupling and simulator cost; the K-sweep law's *shape* held before (Turn 12), so state
the functional form in advance.

---

## 2. Domain portfolio

Selection criteria: Python-drivable simulator; goals expressible in the `minimum_`/`maximum_`
convention; a computable reference optimum (needed for obj/ref and regret); a derivable
validity envelope; cost per eval compatible with candidate enumeration.

| # | Domain | Simulator | Coupling (expected) | Size ladder | Stoch. variant | Cost/eval | Reference optimum |
|---|--------|-----------|--------------------|-------------|----------------|-----------|-------------------|
| 0 | Truss 2D (existing) | in-house FEA | low (0.658 measured) | 4–40 members | **S: load ensembles** | 4 ms | LP optimum (exists) |
| 1 | 3D frame | PyNite | low–med | 20–200 members | via loads | ~10–100 ms | FSD-analog sizing |
| 2 | Battery (existing) | PyBaMM DFN | high | 9 params (fixed) | **S: parameter spread** (aging/manufacturing) | 3.7 s | best-known per family |
| 3 | Analog circuit sizing | ngspice via PySpice | high | 5–50 devices | **S: MC mismatch** (industry-standard) | 10–100 ms | published reference designs |
| 4 | Thermal network / heat sink | in-house lumped RC (~200 lines) | med | 10–100 nodes | via ambient/load draws | ms | analytic bounds |
| 5 | Epidemic control (SEIR, age-structured) | scipy ODE | high | 2–50 compartments | **S: binomial chain / Gillespie** | ms | optimal-control / LP relaxation |
| 6 | Traffic signal timing | in-house cell-transmission model | med–high | 4–100 intersections | **S: demand noise** | ms | Webster's / max-pressure |
| 7 | Grid dispatch | pandapower (IEEE cases) | high | 14–300 buses | **S: renewable forecast error** | 10–100 ms | OPF solver optimum (exact) |
| 8 | Inventory / queueing network | in-house discrete-event | med | 2–50 echelons | **intrinsically stochastic** | ms | base-stock / known policies |

Notes on the picks:

- **#5, #6, #8 carry the "social systems" claim honestly**: they are social *decision
  problems* with mechanistic simulators. Open-ended agent-based social simulation stays out
  of scope — no derivable validity envelope, contested ground truth, and the search's proven
  talent for simulator exploits makes its output data about the ABM, not about decisions.
- **#7 (grid) is the cleanest size ladder in existence** (standard IEEE cases) *and* has an
  exact reference optimum from the OPF solver — regret is computable exactly.
- **#3 (circuits) is the cleanest stochasticity story**: Monte-Carlo mismatch is how the
  field actually designs, so the chance-constrained formulation is native, not bolted on.
- **#8 is intrinsically stochastic** — the one domain with no deterministic twin — included
  because classic OR results give known-optimal policies to validate E[Φ] machinery against.

Goals-dict sketches (the convention parser handles all of these unchanged):
`maximum_peak_infections`, `minimum_vaccination_coverage` (epidemic);
`maximum_line_loading`, `minimum_voltage_pu`, `maximum_cost` (grid);
`minimum_gain_db`, `maximum_power_mw`, `minimum_phase_margin` (circuit);
`maximum_mean_delay`, `minimum_throughput` (traffic);
`maximum_stockout_prob`, `maximum_holding_cost` (inventory).
Equity/fairness criteria enter the same way (`maximum_group_disparity ≤ limit`).

---

## 3. Shared infrastructure (build once, all domains inherit)

### 3.1 Stochastic Φ — the one real extension to `potential.py`

Deterministic Φ reads one metrics dict. Under noise, Φ(s) is a distribution. Extension:

- `Domain.evaluate(st, seed)` — seeded single draw; a `StochasticDomain` mixin aggregates
  `n_rep` draws into per-metric samples.
- **Chance constraints**: `Constraint.violation` evaluated on the empirical q-quantile of
  the metric (default q = 0.9), so `minimum_fos_buckling: 1.5` under load ensembles means
  "FOS ≥ 1.5 at the 90th-percentile worst load." One field on `Constraint`; deterministic
  domains are the q-irrelevant special case.
- **The reliability gate becomes the repeat scheduler.** `lookahead_probe`'s condition
  (top-two margin vs noise band σ) already exists as a diagnostic; under stochasticity it
  decides `n_rep` adaptively — sample until the margin clears the band or the budget caps
  out (n_rep_min=3, n_rep_max=32). Turn5 §7 anticipated exactly this ("simulator cost turns
  the gate from a logged diagnostic into a scheduling rule").
- **Preference-pair hygiene**: a pair is emitted only when the Φ gap exceeds the noise band
  at the chosen n_rep. Discarded-pair rate is itself a per-domain noise statistic worth
  logging.

### 3.2 Domain registry

`da_domain.REGISTRY` grows one `Domain` subclass per row of the table. The contract is
already minimal (load/clone/evaluate/get/set/bounds/margins/budget_ratio); battery proved
the no-per-element-attribution path works (`element_margins → None`, tools degrade).

### 3.3 Validity-envelope red-team protocol (the D9 lesson, made a gate)

Every new domain, before any data harvest: run unguarded Φ-argmax search with **no**
envelope, inspect the top-Φ states for physical/semantic nonsense, and write the envelope
from what the search finds. On trusses this failure was 17% of "successes"; assume every
simulator has an equivalent exploit until the audit says otherwise. A domain without a
defensible envelope does not ship.

### 3.4 Problem generation, splits, power

- ~**100–150 problems per domain-variant**, generated in families × difficulty terciles,
  split with the stratified-seeded discipline of `make_splits.py`. The battery generality
  test failed to be a measurement at n=24 (Turns 19/21) — that mistake is not repeated.
- **Solvability audit per domain**: verify the ladder solves each generated problem or mark
  it degenerate (the truss set carries 9 unsolvable mechanisms where Φ is constant; every
  domain will have an analog, and generators must not silently produce them).

### 3.5 Acceptance test: the ladder runs first, the LLM never runs first

Each domain-variant's go/no-go is the non-LLM search ladder (`search_ladder.py` /
`battery_ladder.py` pattern): random / greedy depth-0 / blind enumeration / Φ-selection /
Φ depth-2, on CPU, deterministic (or seed-averaged). This yields, per domain, before any
GPU-hour is spent: the coupling placement (feedback-sufficiency), the Φ-uplift over greedy,
the noise band, and the reference-relative objective. A domain where Φ-uplift is inside the
noise band still contributes size/noise cells to the regression but earns no headroom claim
and lower data-harvest priority.

---

## 4. Data harvest (the point of all this)

Per accepted domain-variant, the pipeline that already exists runs unchanged:

1. **Preference pairs** at Φ-ranked states (`distill_preference_pairs.py` pattern): truss
   yielded ~60 pairs/problem incl. failure states. At ~1,000 problems across the portfolio:
   **O(50–60k) pairs** spanning nine qualitatively different decision structures.
2. **Replay-verified expert trajectories** (design-agent pattern, 140 on trusses):
   **O(1–2k) trajectories** portfolio-wide for SFT format-grounding (not for decision-rule
   transfer — T6 settled that).
3. **Per-state search justifications** (which constraint binds, candidate count, margin) as
   the reasoning scaffold, as in `distill_search_traces.py`, with the phrasing-diversity
   fix noted in turn5 §9 (temperature brittleness).

The portfolio also answers the standing proposer question: on trusses the LLM proposer never
beat uniform random into the same search (Turn 14/17 lineage). Nine domains of varying
coupling is the test of whether a learned proposer earns its place *anywhere* — a
pre-registerable prediction: proposer value should appear where coupling is high and local
moves are non-obvious, if it appears at all.

---

## 5. Waves and dependencies

**Wave 1 — machinery on trusted oracles (target: first paired D-vs-S measurement).**
- Stochastic-load truss variant (§3.1 built here, validated against the deterministic twin
  on identical problems).
- Epidemic SEIR-ODE domain (deterministic; cheapest new simulator; carries the "social"
  framing; generator + ladder + envelope audit).
- Grid dispatch (pandapower; exact regret; size ladder for free).

**Wave 2 — coupling and size coverage.**
- Circuit sizing (deterministic nominal + MC-mismatch variant).
- 3D frames (size axis extension of the strongest existing domain).
- Battery parameter-spread variant (reuses PyBaMM; the 3.7 s cost × n_rep is the stress
  test for the adaptive repeat scheduler).

**Wave 3 — intrinsically stochastic and social-mechanistic.**
- Inventory/queueing (validates E[Φ] against known-optimal policies).
- Traffic CTM (+ demand-noise variant).
- Stochastic epidemic (Gillespie twin of the Wave-1 ODE).

Each wave ends with the ladder table for its domains appended to the ledger and the
cross-domain uplift-vs-coupling regression re-fit. Kill criteria are per-domain (ladder
uplift inside noise → demote), not per-wave.

---

## 5b. Implementation status (2026-09-21)

Built in `dataxp/` (see its README for the layout). Seven domains, thirteen
variants, all passing the contract smoke test on both the local box
(Python 3.12 / numpy 1.26) and trace (3.10.9 / 1.23.5):

| domain | variants | size ladder | coupling anchor | reference class |
|---|---|---|---|---|
| SEIR epidemic control | `seir`, `seir_stoch` | 2–16 groups | peak-infection share | heuristic (uniform-reduction bisection) |
| Grid dispatch | `grid`, `grid_stoch` | 3–12 generators | PTDF sensitivity | **exact** (DC-OPF LP) |
| Thermal network | `thermal`, `thermal_stoch` | 10–100 nodes | hottest node | heuristic (uniform-g bisection) |
| Traffic signals | `traffic`, `traffic_stoch` | 4–48 intersections | worst approach | heuristic (proportional split) |
| Inventory | `inventory` | 2–14 stages | stockout / holding stage | heuristic (newsvendor) |
| Circuit sizing | `circuit`, `circuit_stoch` | 5 devices (none) | most-violated spec | heuristic (random search) |
| 3D truss | `frame3d`, `frame3d_stoch` | 39–141 members | min-FOS member | heuristic (fully-stressed design) |

### Defects found and fixed, with the evidence

Each of these was caught by measurement, not inspection, and each is an
instance of a failure mode this project has already paid for once:

1. **The G1 gate measured a constant.** `initial_violation` was read from the
   program *after* `with_violation_scale`, which divides by that same initial
   violation — so it was ≈1.0 for every problem by construction, and the
   nonzero CVs were an artefact of a τ mismatch (0.0 vs 0.05). Grid's apparent
   "zero variance" was this bug, not the domain: the raw statistic gives 0.88.
2. **Traffic's queue constraint was a phantom (the D3 pattern).**
   `max_queue_frac` measured only in-network CTM cells, which saturate at a
   level fixed by the green split, while oversaturated demand piled up
   invisibly in the boundary source queue. The metric read exactly 0.775
   regardless of demand — scaling demand 2.4× did not move it. Now the queue
   includes the source queue, normalised by approach storage.
3. **Traffic started feasible on 100% of problems** — no restoration data at
   all. Demands are now calibrated from the capacity algebra: with
   ρ = (d_main+d_cross)/USABLE and the heavy direction's share s, a problem is
   solvable iff ρ ≤ 1 and starts infeasible iff s·ρ > 0.5. Drawing inside that
   band makes every problem *provably* both. Verified: 9/9 solvable by a
   uniform green split; φ-select 6/6 against greedy 2/6.
4. **Grid started at 482% line overload**, which no budget of single-generator
   multiplicative moves could undo (φ ground 4.82 → 2.42 over 20 turns and
   solved 0/6 even at 40). The initial dispatch now bisects a blend between the
   greedy and spread dispatches to hit a *targeted* initial overload
   (1.15–2.30×), which is also the difficulty axis. Result: φ-select 0.83 vs
   greedy 0.00 — the portfolio's strongest uplift, as predicted for a
   tightly-coupled network.
5. **frame3d sat on a plateau.** Fully-stressed design makes every member
   exactly critical, so uniform under-sizing left all 39–141 members violating
   while `fos_buckling` (a **min**) barely responded to any single repair.
   Φ correctly spent its turns shedding mass instead (FOS flat at 0.52 while
   mass fell 15.0 → 13.1) — the objective-competes-with-violation case of
   turn5 §8, which the lexicographic offset cannot fix because `tanh`
   saturates at V ≫ τ. Fixed at the generator: a small subset of members is
   degraded with *staggered* severities, so repairing the worst raises the
   minimum to the next and every correct move pays.
6. **The FSD reference was not feasible** on a third of problems (FOS 1.11–1.24
   against a 1.5 target): resizing redistributes the forces it was sized
   against, so the damped fixed point does not land on the constraint. A
   monotone growth pass now follows it, and a reference optimum that actually
   meets the targets is the only kind worth quoting. 9/9 FSD-OK.
7. **Measurability was confounded by symmetry.** The top-2 Φ margin reads zero
   on a 4-fold symmetric tower because the top two candidates are *equivalent*
   — a free decision, not an unrankable state. G2 now asks whether the top
   beats a *typical* (median) candidate; the top-2 margin is retained as the
   statistic gating pair emission, where a tie genuinely carries no preference.
8. **Stochastic states were judged at the minimum replicate budget**, so the
   standard errors swamped every gap: frame3d_stoch reported 0.00 measurable
   states and emitted no data at all. "Unmeasurable" must mean undecidable at
   the *maximum* affordable budget, so the top and median candidates are now
   refined before a state is written off. Measured effect: frame3d_stoch
   0.00 → 0.39, thermal_stoch 0.17 → 0.77, grid_stoch 0.17 → 0.53.

### Run 1 results (trace, 2026-09-21): 13/13 variants accepted

`/trace/home/wxu2/dataxp_out/run1/` (reports mirrored locally in
`analysis/dataxp_run1/`; intermediates on scratch at
`/trace/scratch/biosimmlab/wxu2/dataxp_tmp/run1/`). n=130 problems per variant,
20 turns, five ladder policies, all three gates.

**69,887 preference pairs, 1,085 replay-verified trajectories, 1,586 problems
across 1,581 distinct problem ids; 0 malformed pairs.**

| variant | rnd/grd/bld/phi/d2 | uplift | pairs/traj/probs |
|---|---|---:|---|
| circuit | 0.15/0.26/0.17/**0.88**/0.90 | **+0.62** | 7791/115/130 |
| circuit_stoch | 0.18/0.32/0.22/**0.90**/0.90 | **+0.58** | 7517/117/130 |
| grid | 0.03/0.02/0.38/**0.62**/0.78 | **+0.60** | 5400/81/130 |
| grid_stoch | 0.01/0.02/0.05/**0.14**/0.17 | +0.12 | 2276/18/105 |
| traffic | 0.02/0.25/0.10/**0.82**/0.78 | **+0.57** | 7588/107/130 |
| traffic_stoch | 0.00/0.08/0.03/**0.29**/0.28 | +0.22 | 3572/37/113 |
| inventory | 0.23/0.45/0.37/**0.69**/0.57 | +0.25 | 2059/80/114 |
| seir | 0.17/**0.83**/0.25/0.81/0.80 | −0.02 | 7232/105/130 |
| seir_stoch | 0.19/**0.90**/0.34/0.64/0.50 | −0.26 | 3314/72/112 |
| thermal | 0.58/**1.00**/0.75/0.98/0.97 | −0.02 | 7800/127/130 |
| thermal_stoch | 0.56/**1.00**/0.70/0.95/0.93 | −0.05 | 4349/116/121 |
| frame3d | 0.05/**1.00**/0.12/0.56/0.78 | −0.44 | 7838/73/130 |
| frame3d_stoch | 0.02/**0.62**/0.12/0.33/0.33 | −0.28 | 3151/37/111 |

**The pre-registered relation holds, and it is sharp.** Regressing Φ's uplift
over greedy on *greedy's own feasibility* gives **r = −0.747, slope −0.70
(n=13)**, and the ordering is monotone with no exceptions:

- Where the depth-0 diagnostic fails (greedy 0.02–0.32: grid, traffic,
  circuit), Φ-ranking is worth **+0.12 to +0.62**.
- Where it succeeds (greedy 0.62–1.00: frame3d, seir, thermal), Φ-ranking is
  worth **−0.44 to −0.02** — it is actively worse.

That is the truss-vs-battery inversion of turn5 §7 reproduced across seven
independent domains: **Φ-guided search earns its cost exactly where no single
critical element exists, and nowhere else.** thermal, seir and frame3d are the
negative controls — domains whose simulator hands over an argmin (hottest node,
peak-infection group, min-FOS member) and performs the inner maximisation for
free. They are worth keeping in the portfolio for precisely that reason.

A caution on the secondary statistic: `coupling_feedback_sufficiency`
(hint/Φ-argmax agreement, r=0.860) is an *agreement rate*, not a coupling
measure — it conflates "the hint is right" with "Φ is right", so the greedy
column is the one to read as the cause.

**Envelope red-team**: 1 exploit caught (seir_stoch), 0 in the other twelve.
Read as *unproven rather than safe*: some parameterisations cannot express the
exploit by construction (frame3d ties wall thickness to radius, so the
negative-area state of D9 is unreachable), and a 12-problem red-team may simply
be too small. This is the weakest part of the evidence.

### Gate thresholds differ for stochastic variants, deliberately

Recorded in every report under `gate_thresholds`. For a deterministic domain an
unmeasurable state means Φ is *flat* across candidates — genuinely degenerate,
the truss-mechanism case. Under noise it means the replicate budget could not
separate the top candidate from a typical one: that state yields no preference
pair, but it corrupts nothing and the remaining states are still clean. The
same logic applies to pair yield — the noise-band filter is *designed* to
reject tied pairs, so a low yield under noise is the filter working, and the
floor exists to catch mis-calibration (NaNs, envelope failures) rather than to
demand that noise be absent. Bars: G2 state/problem 0.60/0.80 deterministic
against 0.35/0.70 stochastic; G3 pair yield 0.70 against 0.15.

One scale change was made to a domain's physics rather than to a threshold:
frame3d's stochastic load scatter went from σ=0.12 with 3° of angular jitter to
σ=0.08 with 1.7°, because FOS is a minimum over up to 141 members and therefore
reaches Φ through an extreme-value statistic. 7–10% is also the more honest COV
for structural dead and live load.

## 6. Risks, named

- **Envelope debt.** Nine envelopes is nine audits; this is the least automatable step and
  the one that bites silently (D9). Budgeted explicitly per domain in §3.3.
- **Repeat-cost blowup.** Stochastic variants multiply simulator calls by n_rep; the
  adaptive gate caps it, but PyBaMM-class costs may force the battery-S variant to a
  smaller problem count. Acceptable — it is one cell of the regression, not the claim.
- **Scale calibration under noise.** `with_violation_scale` divides by the *initial* state's
  violation; under stochasticity that is a random variable. Use the q-quantile initial
  violation, same q as the constraints — one convention, stated once.
- **Reference optima vary in strength** (exact for grid, best-known for battery, analytic
  bound for thermal). Regret is only cross-domain comparable within reference class; report
  the class alongside the number.
