# dataxp v2 (run2): full action sets

**Status (2026-10-02).** All 13 variants are generated and gated, and every gate passes.

- **Data:** `/ocean/projects/mch250030p/shared/dataxp/run2/<variant>/`, with `accepted/{problems,trajectories,pairs}.jsonl` and `report.json`.
- **Code:** `/ocean/projects/mch250030p/shared/dataxp/code/dataxp/`. The v2 domains are `domains/<name>2.py`.
- **run1** (v1 domains) is unchanged in `shared/dataxp/run1`.

## Why v2

In every v1 domain, the only move was to scale one continuous knob per element:
- circuit: device x
- grid: p_set
- traffic: green_frac
- thermal: g_sink
- seir: contact_reduction
- inventory: base_stock
- frame3d: member radius

This came from the design-agent `da_domain.py` contract and from a candidate generator that only applied factors. It was never discussed or chosen. It made the domains far too easy: the collaborator's eval (DX-20/21) found that the un-tuned base solves 90-98% of circuit, frame3d, inventory and seir problems within 12 turns.

v2 keeps the action format `SET_PARAM(element, param, value)`, so the parser, replay and rendering are unchanged. Each element now has several typed parameters:
- continuous, scaled;
- additive or circular, shifted;
- categorical, set to any other code;
- toggles, which also cover topology changes. Toggles act over a fixed pool of allowed elements, so element indices never change within a trajectory.

## Action sets (summary; full tables in the appendix)

| domain | moves beyond v1's single knob |
|---|---|
| frame3d2 | install/remove members from a ground structure; material (Al / S355 steel / Ti-6Al-4V); joint offsets dx, dy, dz; radius |
| circuit2 | W and L per device; mirror ratio B; Cc; Ib; nulling resistor Rz with on/off toggle; stage-1 cascode toggle; compensation scheme. A swing constraint makes the cascode a real trade-off |
| grid2 | meshed networks; dispatch with pmin; unit on/off; line switching (with a switching cost); phase-shifter angle |
| traffic2 | signals simulated step by step; cycle {60..120 s}; green split; circular offset; phase order; main-road lanes 1-3 |
| thermal2 | movable sinks (site, conductance); spreader links on/off; power-block placement (moving onto an occupied slot swaps the two) |
| seir2 | contact reduction; intervention start day and duration; testing rate; vaccine share; school policy; mask mandate |
| inventory2 | base stock; review period {1, 2, 4}; stocking point on/off; expediting on/off |

## run2 results (130 problems per variant, 20 turns)

| variant | problems / traj / turns (repair) | made feasible: random / one-knob greedy / blind / Φ-select | Φ-select obj/ref | how the gold moves split (% of gold actions) |
|---|---|---|---|---|
| circuit2 | 130 / 125 / 2316 (686) | 0.18 / 0.60 / 0.23 / 0.96 | 1.05 | Cc 29, W 23, L 20, B 11, Ib 9, comp 4, cascode 3, Rz 2 |
| circuit2_stoch | 130 / 122 / 2342 (806) | 0.12 / 0.62 / 0.22 / 0.94 | 1.01 | similar |
| frame3d2 | 130 / 119 / 2373 (913) | 0.01 / 0.35 / 0.05 / 0.92 | 0.89 | r 35, mat 25, active 24, joints 17 |
| frame3d2_stoch | 92 / 48 / 851 (666) | 0.00 / 0.29 / 0.03 / 0.53 | 0.96 | r 45, mat 27, active 16, joints 12 |
| grid2 | 130 / 116 / 698 (426) | 0.02 / 0.12 / 0.23 / 0.89 | 1.06 | shift 40, p_set 39, closed 14, on 7 |
| grid2_stoch | 124 / 100 / 1513 (923) | 0.03 / 0.07 / 0.07 / 0.81 | 1.10 | p_set 61, shift 25, closed 9, on 5 |
| inventory2 | 125 / 95 / 1467 (817) | 0.02 / 0.05 / 0.13 / 0.77 | 0.88 | base_stock 49, expedite 30, review 15, active 7 |
| seir2 | 130 / 130 / 2593 (670) | 0.07 / 0.52 / 0.22 / 1.00 | 1.29 | vax 25, duration 24, start 18, contact 13, test 10, school 7, mask 3 |
| seir2_stoch | 115 / 115 / 2161 (471) | 0.26 / 0.75 / 0.47 / 1.00 | 1.31 | similar |
| thermal2 | 130 / 130 / 1268 (376) | 0.18 / 0.61 / 0.55 / 1.00 | 1.20 | g 58, slot 22, site 11, active 9 |
| thermal2_stoch | 130 / 128 / 1980 (672) | 0.21 / 0.55 / 0.59 / 0.98 | 1.25 | similar |
| traffic2 | 130 / 127 / 2234 (392) | 0.05 / 0.06 / 0.29 / 0.98 | 1.12 | offset 37, green 35, phase order 11, cycle 10, lanes 7 |
| traffic2_stoch | 117 / 117 / 1915 (622) | 0.10 / 0.19 / 0.38 / 1.00 | 1.34 | similar |
| **total** | **1,613 / 1,472 / 23,711 (8,440 repair)** | | | 75,147 preference pairs |

Problems is the accepted count; some stochastic variants drop problems that fail the reliability checks. Across all variants, one-knob greedy is now far below Φ-select, and non-knob moves make up 25-70% of gold actions.

## Is the agent given enough information? (checked)

Each domain went through two checks:

1. **Audit, move type by move type.** For each move type, the audit lists what is needed to choose the element, the direction and the magnitude, and the move's cost, and checks that the observation provides each item.
2. **Blind-choice probe.** A different, stronger model (Claude) sees only the observation for sampled repair-phase states and picks one move. Each pick is scored by where its Φ ranks among all legal moves (random = 50, gold ≈ 99).

The `describe()` fixes from the audit change only the observation text; the generated data is byte-identical.

| domain | blind pick percentile (after fixes) | main additions |
|---|---|---|
| frame3d2 | 99 (round 1 scored 0: the action syntax was unclear) | bare element index; displacement vectors; per-joint sensitivities; the force and FOS an optional member would carry if installed; members at the radius cap flagged |
| circuit2 | 99 | operating point per device, pole/zero map, effect of each knob |
| grid2 | 99 | room up and down per generator, line sensitivities, switching effects |
| traffic2 | 98 | green times per phase, travel times between intersections, queues per phase |
| thermal2 | 100 | site, link and slot tables with their effects |
| seir2 | 98 | infections by setting and by source group, weekly timeline, ICU peak day, R_t per period |
| inventory2 | 90 | cover and fill rule, re-planning previews, cost per move type |

The training prompt (`render_cot.policy_observation`) is exactly the text the probe used. It contains:
- the domain header;
- `describe()`;
- constraint status, with every violation marked;
- the changes made so far, each with its element index and labelled as the element is after the change;
- the observed effect of the last change on every metric (before -> after).

Without the last-change block, state-only reasoning oscillated in the collaborator's runs. Values print with up to 6 significant digits, so action values are exact.

## Code changes outside the v2 modules

All are backward compatible, and the v1 smoke tests still pass.
- `domain.enumerate_candidates`: optional `candidate_values(st, i, p, cur)` and `candidate_moves(st, rng)` hooks, plus class-level `max_elements` and `max_candidates`. The v1 path is unchanged.
- `ladder.run_episode`:
  - optional `hint_move(st)` for the greedy policy;
  - with `rescore_incumbent = True` (v2 domains), stochastic Φ-select compares candidates against a fresh estimate of the current design. Before this, one lucky draw could stall the search for many turns. Replay is unchanged.
- Feasibility offset: `--offset-rule infeasible`. This fix came from another session; run2 used it.
- `render_cot`: v2 domain headers, the shared `policy_observation`, exact value formatting (`_fmtv`), history labels taken after the change, and the observed-effect block.
- `probe_info.py`: the information probe.

## Known limitations

- **frame3d2_stoch:** Φ-select solves only 53% of problems (48 trajectories).
- **inventory2:** Φ-select solves 77%.
- These two are hard by design, and they are where the search-based gold labels are thinnest.
- **`--quick` generation:** it uses 6 problems, so it can never pass gate G1, which needs at least 9. Use `--n 12` or more for checks.
- **Observation length:** 7-16k characters (about 2-4k tokens). A 20-turn episode cannot carry every past observation, so each turn's prompt is self-contained: it includes the history list and the effect of the last change.

---

# Appendix: per-domain parameter tables (from the build and verification reports)

### frame3d2

| param | kind | applies to | meaning |
|---|---|---|---|
| `r` | continuous (scaled) | member (every ground-structure member, installed or not) | Outer tube radius in m, bounds [0.004, 0.12]; 0.12 is described to the agent as the largest tube made. Wall t = 0.2 r, A = 1.131 r^2, I = 0.4637 r^4 (checked by hand). Moves: x0.8, x0.9, x1.1, x1.25, or a resize to FOS 1.5 at the current force (margin x1.03; x1.15 in the stochastic variant), or a resize down for under-used me... |
| `active` | toggle 0/1 | member | 1 = installed, 0 = removed (no stiffness, no mass). set() rounds at the 0.5 threshold. Initially base members are 1 and extras are 0. Extras are sdiag2, xdiag and the bridge two-bay skip braces, with a length cap of 2.25 x max(pitch, width). Removal is offered only when the redundancy number exceeds 0.05; describe() tags the ... |
| `mat` | categorical (integer-valued float) | member | Codes, all listed in describe() with E, yield and density: 0 = 6061-T6 Al (68.9 GPa, 276 MPa, 2700); 1 = S355 steel (210 GPa, 355 MPa, 7850); 2 = Ti-6Al-4V (113.8 GPa, 880 MPa, 4430). r is kept on a change. set() rounds and clamps: 1.6 -> 2, 1.4 -> 1, -5 -> 0, 7 -> 2. Initially all 0. |
| `dx` | additive offset (bounded, not circular) | free joint (elements M..M+J-1); does not apply on the axial coordinate of a loaded joint | x offset from the initial position in m, bounds +-0.25 b, step 0.05 b, where b = min(pitch, smallest section width). set() clamps and rounds to 1e-4. NEW: a joint that carries a load keeps its coordinate along the structure's axis. For the cantilever tip and every bridge joint (all bridge joints are deck-loaded) this axis is ... |
| `dy` | additive offset (bounded) | free joint | y offset; same bounds, step and legalisation as dx. Transverse for every family, so it applies to every joint. A load at a moved joint rides with it. |
| `dz` | additive offset (bounded) | free joint; does not apply to loaded tower joints | z offset; same bounds, step and legalisation. For towers z is the axis, so the loaded top section and the point-load joint cannot move in z. For cantilevers and bridges dz is transverse and always applies. |

**Simplifications (stated in the module docstring):** All of these are stated in the module docstring: - Linear-elastic, small-displacement direct-stiffness analysis of a pin-jointed space truss: axial members, nodal loads, one load case. - Pin-ended Euler buckling (K = 1) per member only. No global or joint buckling, no imperfections, no self-weight load. - Tube wall fixed at t = 0.2 r; nominal handbook material values. - Member clashes are not checked; offset bounds keep nodes apart. - Topology is a fixed ground structure; nodes are never added. - Loads keep their magnitude and direction. A load moves with its joint across the axis and never along it (new). - Members joining two supports are not part of the structure (new); under pinned supports they carry no force and add no stiffness. - Mechanisms (Cholesky failure, a zero-stiffness DOF, or relative pivot < 1e-9) return finite worst-case metrics: FOS 0, shortfall = number of active members, deflection 10 x span, mechanism = 1. The envelope (deflection <= 0.1 span, mechanism <= 0.5) marks them invalid. - Deflection is the Euclidean nodal norm. v1 used the largest component, so v2 values are 0-7.5% higher on the same design. - The reference is a heuristic: FSD plus uniform radius scaling for deflection. It is not a bound; phi_select beats it by about 12% at the median. - Generator: 3 families x 6 bay sizes x taper/eccentricity x light (40%) / heavy (60%) x {undersized, point_load}. The deflection limit is 0.70-1.0 x the sound design's deflection. - Seeded acceptance loop: the start must be infeasible with at most 10 failing members and deflection at most 1.35 x the limit. Al

### circuit2

| param | kind | applies to | meaning |
|---|---|---|---|
| `W` | continuous, scaled (x0.5,0.7,0.85,0.93,1.07,1.2,1.4,2.0) | M1 (0), M2 (1) | Channel width, bounds [0.5, 500] um. gm = sqrt(2k(W/L)Id), Vov = 2Id/gm. Area is 2*W*L for M1 (two devices) and W*L for M2. M2's gate capacitance (2/3)Cox*W*L loads node 1. Legal grid: 4 significant digits and at most 3 decimals. |
| `L` | continuous, scaled (same factors) | M1 (0), M2 (1) | Channel length, bounds [Lmin, 12 Lmin] um (Lmin 0.18/0.25/0.35 by node). lambda = lambda0*Lmin/L, so ro and gain rise with L, while W/L (hence gm and Vov) changes too, and area and gate capacitance rise. Mirror-load and cascode devices use length L1, so a long L1 lowers the mirror pole and raises the cascode gain. |
| `B` | continuous, scaled | MB (2) | Stage-2 to tail current ratio, bounds [0.25, 30]; I2 = B*Ib. MB is a 1 um-long device of fixed per-problem width. Area (2+B)*W_MB*L_MB; stage-2 power B*Ib*Vdd. |
| `Cc` | continuous, scaled | CC (3) | Compensation capacitance, bounds [0.05, 20] pF, area 200 um^2/pF. Sets the dominant pole (GBW about gm1/(2 pi Cc)) and the internal slew limb Ib/Cc. |
| `comp` | categorical, integer codes {0,1,2}; both other codes are offered every turn | CC (3) | Compensation scheme. - 0 = direct Miller: right-half-plane zero gm2/Cc unless the nulling resistor is on. - 1 = voltage-buffered Miller: follower at Ib/2, gm_f = Ib/0.1 V. No feed-forward zero; left-half-plane zero gm_f/Cc. Costs Ib/2 of supply current, +15 um^2 plus devices, and output low limit Vth+0.2 V. - 2 = current-buff... |
| `Ib` | continuous, scaled | IB (4) | Tail current, bounds [0.5, 300] uA; Id1 = Ib/2. Power (2Ib + I2 + buffer bias)*Vdd. Vov(MB) grows as sqrt(Ib), reducing input range and output swing. |
| `Rz` | continuous, scaled (x0.5,0.7,1.4,2.0, plus 1/gm2 and (1/gm2)(1+CL/Cc)); offered only while rz_on=1 and comp=0 | RZ (5) | Nulling resistor, bounds [0.1, 200] kOhm. In the signal path only with rz_on=1 and comp=0: the zero moves to 1/(Cc(1/gm2 - Rz)) (left half plane once Rz > 1/gm2) and a pole appears near 1/(Rz C1). Area 4 um^2/kOhm + 10 um^2 while on. |
| `rz_on` | toggle 0/1 (offered every turn) | RZ (5) | Inserts the nulling resistor in series with Cc. It costs area under every scheme but acts only with comp=0. |
| `cas_on` | toggle 0/1 (offered every turn) | CAS (6) | Telescopic stage-1 cascode. Multiplies R1 by gm_cas*ro_cas = 2/(0.1 V * lambda(L1)): 44-111x (+33 to +41 dB) at L1 = Lmin, up to about 530-1330x at 12 Lmin. The builder's '30-130' was wrong. Costs 0.2 V of input common-mode range, cascode area + 25 um^2, and a cascode-node pole gm_cas/Cx. |

**Simplifications (stated in the module docstring):** The builder's list is accurate and is stated in the module docstring. In short: - Square law everywhere: no velocity saturation, body effect or subthreshold. The Vov window [0.075 V, max(0.4, 0.35 Vdd)] enforces this on M1, M2 and MB. - One k for the signal devices; PMOS auxiliaries use k/2.5; |Vthp| = Vthn. - Auxiliary devices are auto-sized to Vov 0.1 V. Mirror loads and cascodes use length L1; buffer devices 2 Lmin. MB width is fixed per problem. - The cascode costs input common-mode range only, not output swing. - Two-node model plus the compensation branch. Device Cgd, the stage-2 load pole, buffer-node parasitics and slewing, noise, offset and CMRR/PSRR are not modelled. Mirror and cascode-node poles are separate first-order factors. Scheme 2 always uses its own common-gate device. - Slew = min(Ib/Cc, I2/CL). GBW is the first unity crossing; PM is the worst over all crossings; no gain-margin or Nyquist analysis. - Area uses nominal sizes. The stochastic variant draws k, bias, mirror-Vth, Rz (5-10%) and Cc (2-4%) errors.  Added by me: continuous values are legalised to 4 significant digits and at most 3 decimal places.  Hand and spot checks: - x0 of problem 0: icmr by hand 2.285 V, module 2.28497. - The transfer function matches a direct complex nodal solve for every comp x rz_on x cas_on combination. - On 4198 valid random designs, the module's GBW lies inside the dense-grid (1000/decade) bracket of the first unity crossing in every case. Its PM is never optimistic by more than 0.5 deg, so no narrow resonant crossings are missed.

### grid2

| param | kind | applies to | meaning |
|---|---|---|---|
| `p_set` | continuous (scaled moves), MW | generators, elements 0..G-1 (G = 6-24), only while the unit is on | Real-power setpoint, clamped to [p_min, p_max] while the unit is on. Cost: c $/MWh. - get() returns None while the unit is off, and set() is then a no-op. - Offered values: x0.6, 0.8, 0.9, 0.97, 1.03, 1.1, 1.25 and 1.5; p_min; p_max; 'take up the current slack mismatch'; and 'bring the worst overloaded line to 98% via this un... |
| `on` | toggle 0/1 (>= 0.5 means 1) | generators, elements 0..G-1 | Unit commitment. - 0->1 starts the unit at exactly p_min. It then pays no-load $/h, plus start-up $ if it was off in the initial design (a state function, refunded on shutdown). - 1->0 sets output to 0. - Every unit's flip is offered every turn. - describe() prints room up and room down to p_min and states that going below p_... |
| `closed` | toggle 0/1 | switchable lines, elements G..G+S-1 (S = 4-16; a fixed pool of breaker lines, optional radial-spur feeders, and 2-6 normally-open ties) | 1 = in service. Every flip is offered every turn. - Islanding is PENALISED LOAD SHED: de-energised buses shed load (constraint load_shed_mw <= 0, billed at VOLL) and their units trip. Only the energised component is solved, so the solve is never singular. - NEW: a switching cost (spec['switch_cost'] = 2.5 x pst_cost_per_deg, ... |
| `shift` | additive, degrees, step 5, bounds [-30, +30] | phase-shifting transformers, elements G+S..G+S+P-1 (P = 2-4) | Line flow = b*(theta_from - theta_to + shift). - set() rounds to the nearest 5 and clamps. Verified: 2.4->0, 2.5->5, -2.6->-5, 12.6->15, +-1e9->+-30. - Moves offered every turn: +-5, +-10, +-15 deg. - Operating cost: pst_cost_per_deg x /shift/. - After my fix a 5 deg step moves a median 6.4% of the line's own limit (p10 2.3%,... |

**Simplifications (stated in the module docstring):** These are unchanged from the builder: - Lossless single-snapshot DC power flow: no reactive power, voltage, ramping or min up/down times, and no N-1 security. - Bus 0 is a slack tie with import and export prices and limits. - Islanded buses are de-energised; there is no island balancing. - Renewables cannot be curtailed. - Thermal limits are synthetic, calibrated to a hidden feasible design. - Start-up cost is a state function. - There are no categorical parameters. - An overload_mw constraint is added, and constraint scales are tight. - Validity envelope: line_loading <= 10, slack import/export <= total load, load shed <= 50% of load.  My changes to the model, all stated in the module docstring:  1. Susceptance scale x4 (_B_SCALE = 4.0; b = 4 x per-bus load size x U(1.5, 6) MW/rad).    - Flows driven by injections do not depend on a uniform b scale; only the PST effect does.    - Before: the best single PST move cut the worst line's loading by a median 0.046, against 0.6-0.7 for dispatch, commitment or switching moves. All PSTs at +30 deg left the worst loading almost unchanged in 3 of 4 probed problems, and the docstring's '5-15% of a main-line flow' claim was false.    - After: the best PST move cuts it by a median 0.18.  2. New switching cost, described above.  3. make_problem fallback: after 60 rejected draws the difficulty is halved; after 120 the size is redrawn. Before, an unlucky (size, difficulty) draw raised RuntimeError, which kills run_generation. This happened at one probed index, and 2 of 300 problems at the new scale needed more than 60 draws.  The stochasti

### traffic2

| param | kind | applies to | meaning |
|---|---|---|---|
| `cycle` | categorical, codes 60, 75, 90, 105, 120 (cycle length in s) | every intersection | Cycle length. set() rounds to the nearest code and wraps the offset modulo the new cycle; junk input (NaN, inf, strings) is ignored. All 4 other codes are offered every turn. Usable flow-ratio sum is 1 - 14/C. Mixed cycles beat against each other (see issues). |
| `green_frac` | continuous [0.2, 0.8], scaled x0.8 / x0.9 / x1.1 / x1.25 | every intersection | MAIN share of the cycle. Effective greens are g*C - 7 and (1-g)*C - 7 s; the 4 s floor is never reached within the bounds. set() clamps. This is the v1 knob. |
| `offset` | circular additive, seconds in [0, cycle), shifts of k*C/8 for k in {-2,-1,1,2,4}, wraps | every intersection | Network-clock start of the first phase. set() wraps modulo the cycle (checked: -1234.567 maps into [0, C)). All shift values are exact multiples of 1/8 s, so the % wrap is exact and replay is exact. An offset shift identical to the phase flip is removed from the move set; the dedup was checked to be exact. |
| `phase_order` | toggle 0/1 | every intersection | 0 = MAIN first, 1 = CROSS first; set() thresholds at 0.5. Flipping it is not a no-op: it shifts the MAIN green window by (1-g)*C. |
| `lanes_main` | categorical/ordinal {1, 2, 3}, moves +/-1 | every intersection (the MAIN link into it) | Multiplies the link's saturation flow and its storage. Discharge is capped by the next MAIN link's lanes divided by its through share. Each lane counts 1 against maximum_lanes_used (constraint scale 1 lane per unit). set() rounds and clamps to [1, 3]. |

**Simplifications (stated in the module docstring):** These are as stated in the module docstring, and I checked them against the code: - No turns into the network; grids are one-way and acyclic, so there are no merges and n <= N_jam always holds (max_density_err stayed at 0 in every test). - Two-phase fixed-time signals with 7 s lost time per phase. - Lanes act as a capacity and storage multiplier. - Critical intersections are modelled as a saturation factor of 0.55-1. - The stochastic variant draws one lognormal level per entry (sigma 0.06). - Corridors go up to 24 intersections; grid and downtown go up to 60. The ladder is 4 families x 9 sizes against v1's 3 x 9.  What I changed: throughput_frac is now measured from the growth of the RESIDUAL queue. That is the per-own-cycle minimum of the stopped stock, i.e. what remains when an approach's green ends, compared between two windows of whole own cycles 300 s apart. It replaces the window mean of the stock. The docstring and the describe() text were updated to match.

### thermal2

| param | kind | applies to | meaning |
|---|---|---|---|
| `site` | categorical (integer-valued float code) | heat sinks [0,K), K = 2..6 | Mounting-site code 0..S-1 into the problem's list of allowed sink sites. Each site is a node with its own cost coefficient k: plate perimeter cheap, interior boss dear; stack lid cheap, bottom board side dear; board family only on cooled modules. set() rounds and clamps. Setting a code held by another sink swaps the two sinks... |
| `g` | continuous (scaled) | heat sinks [0,K) | Sink conductance to ambient [W/K], bounds [0.02, 2.0], quantised to 0.001 in set(). Verified: 99 -> 2.0, -5 -> 0.02, 0.12345 -> 0.123, NaN ignored. Cost k*g^1.3 (doubling g multiplies the cost by 2.46). Moves scale g by (0.5, 0.8, 0.92, 1.1, 1.3, 2.0) and are pre-quantised, so set/get is exact for every candidate. |
| `active` | toggle (0/1) | spreader links [K,K+M), M = 4..10 (heat pipe / thermal strap / thermal pillar by family) | 1 = installed: adds conductance w_l [W/K] between two fixed end nodes and costs c_l (5-30% of an average sink's cost). set() maps v >= 0.5 to 1, else 0. Every link's flip is offered every turn. The pool, w, c and current state are listed in describe(). |
| `slot` | categorical (integer-valued float code) | power blocks [K+M,K+M+B), B = 3..7 | Placement-slot code 0..Q-1. The block's power enters that slot's node. set() rounds and clamps (2.6 -> 3). Setting a code held by another block swaps the two blocks, so a slot holds at most one block (uniqueness verified after swaps). Free of cost. Every code is listed in describe() with its node, current T and occupant. Offe... |

**Simplifications (stated in the module docstring):** As stated in the module docstring (unchanged from the builder, plus two additions): - Linear, temperature-independent conductances; lumped nodes; steady state. - A sink is an ideal conductance to a fixed ambient. Its cost is only k_site*g^1.3: no fixed per-sink or fan cost. - A block's whole power enters one node. Blocks and sinks do not change conduction. - Moving or swapping sinks and blocks is free. - h0 is uniform and fixed. - Spreader links are plain lumped conductances (no dry-out limit). - Stochastic variant: only powers are random, as in v1. Each block power and each node's background power gets an independent mean-one lognormal factor (sigma 0.05-0.16) from default_rng(seed). - Islands and singular solves cannot occur because h0 > 0. Badly cooled designs get very hot instead.  Added: - reference() is a heuristic computed at nominal power (multi-start layout search plus SLSQP per-sink sizing), not an optimum. Stochastic designs, which need margin for the chance constraint, sit above it. - Perturbed starts are kept below a 350 K nominal peak, inside the 500 K validity envelope.

### seir2

| param | kind | applies to | meaning |
|---|---|---|---|
| `contact_reduction` | continuous (scaled x0.5/0.8/1.3/2.0; switch-on 0.05/0.1 near 0; 0 offered when <= 0.03) | group elements 0..K-1 | Share of the group's work and community contacts removed while its window runs. A g-h link keeps (1-cr_g)(1-cr_h). Bounds [0, 0.9], stored to 4 decimals. Cost per window day = w_c * (N_g/N) * cr^2. set() clamps it and ignores NaN, inf and non-numeric input (verified). |
| `start_day` | additive integer days (shifts -21/-7/+7/+21) | groups, schools (K), mask mandate (K+1) | Start of the element's window, bounds [0, 179], rounded to an integer. The window is [start, min(start+duration, 180)). It is offered only where the window has an effect and only if the shift changes the effective window. |
| `duration` | additive integer days (shifts -21/-7/+14/+42) | groups, schools, mask mandate | Window length, bounds [0, 180], clipped at day 180. Moves that would not change the effective window are filtered out. Cost scales with the effective number of window days. |
| `test_rate` | continuous (scaled; switch-on 0.02/0.05) | group elements | Per-day rate at which the group's infectious members are detected and isolated inside its window; the detected share is test/(test+gamma). Isolated cases transmit only at home, at half the household rate. Bounds [0, 0.5]. Cost per window day = w_t * (N_g/N) * test. |
| `vax_share` | continuous (scaled; switch-on 0.05/0.15; 0 offered when small) | group elements | The group's share of a fixed vaccine supply, delivered at supply/rollout doses per day from vax_day, with deliveries ending at day 180. Normalised as share_g / max(1, sum of shares); I verified that shares summing to 1 and shares summing to K give identical metrics. Doses go to susceptibles only, and unusable doses are not pa... |
| `school_policy` | categorical integer codes 0-3 (every other code offered every turn) | element K (schools) | 0 open (school contacts x1), 1 small cohorts (x0.6), 2 hybrid half attendance (x0.35), 3 closed/remote (x0.05), applied during the school window. Cost per day = w_s * school_share * (0, 0.12, 0.45, 1.0)[code]. set() rounds and clamps: 2.6 becomes 3, 9 becomes 3, -3 becomes 0. |
| `mask_mandate` | toggle 0/1 (flip offered every turn) | element K+1 (mask mandate) | When on, multiplies school, work and community transmission by (1 - mask_efficacy), with efficacy 0.15-0.30, during its window. Cost w_m per day in force. set() maps values >= 0.5 to 1 and anything lower to 0. |

**Simplifications (stated in the module docstring):** These are stated in the module docstring: - One-day discrete chain. Beta is now calibrated to that chain's own next-generation matrix (see issues, fix 1). - Synthetic contact structure built from participation profiles, not survey data. - Contact reduction acts only on work and community contacts. Home contacts are never reduced; school contacts change only through school_policy. - One mask efficacy for all non-home settings. - Isolation is perfect outside the home and leaks at half the household rate. - Detection applies to I only (not E), at a constant rate inside the window. - Vaccine efficacy is folded into the dose count (one dose immunises one susceptible), and doses reach susceptibles only. - The ICU census is accounting only; ICU capacity does not feed back on deaths. - The final-size projection ignores waning and vaccination after day 270 and adds no peak or ICU for the projected wave. I checked that this is not exploited: on the 30 final gold designs, a 900-day tail breaches no peak or ICU cap. - No importation within the 270 simulated days. - Stochastic mode uses an integer binomial chain with one stacked binomial draw per step and deterministic integer vaccine deliveries. Stochastic problems seed 3x the day-0 cases. - No structurally invalid designs exist in this domain. The numerical-failure path returns finite worst-case metrics that fall outside the validity envelope; I verified they are invalid and infeasible.

### inventory2

| param | kind | applies to | meaning |
|---|---|---|---|
| `base_stock` | continuous | every stage; offered only while the stage is active (the stored value of an inactive stage is kept and has no effect) | Order-up-to level S in units, bounds [0, base_stock_max], rounded to 0.1. It has no direct cost and drives on-hand stock (holding h $/u/period) and stockouts. Move ladder: x0.8/1.25/0.9/1.1/0.65/1.5/0.95/1.05/0.5/2 and +/-0.5/1/2 sd of demand over the protection interval, trimmed per stage to keep 40-160 moves per turn (at le... |
| `review` | categorical (integer-valued float) | every active stage | Review period, codes 1/2/4 (every period, every 2nd, every 4th). Other values round as <1.5 -> 1, <3 -> 2, else 4 (verified: 2.9 -> 2, 3.0 -> 4, -7 -> 1). Ordering costs K/R per period. The protection interval is route transit + R - 1 (+1 for non-customer stages). Review phase: (t + j) mod R. Both other codes are offered ever... |
| `active` | toggle 0/1 | only the fixed pool of optional non-customer stocking points; get() returns None elsewhere and set() is a no-op there | 0 = cross-dock pass-through: no stock, no orders, no holding or ordering cost. The stages it supplies order from the nearest active ancestor (or the external source) over the summed legs. Every on/off combination is a valid network. All-off and all-on were checked to give finite, valid metrics. Offered every turn for every po... |
| `expedite` | toggle 0/1 | every stage's inbound leg, including pass-through stages | Leg transit L -> ceil(L/2) (L is 2-6, so expediting always shortens the leg), at a premium of e $/unit on the analytic mean flow through the leg. Offered every turn for every stage. Hand checks: constant-demand chains with expedited legs, including an expedited leg on a pass-through, give service 1.0 at zero safety stock and ... |

**Simplifications (stated in the module docstring):** I checked the builder's physics independently against v1 and by hand, and it is correct.  Constant-demand hand checks (demand generator patched): - Zero-safety (R,S) chains give service 1.0 and holding 0. - +3 units of safety stock at h=2 gives holding exactly 6. - R=2 gives cycle stock (R-1)/2 x mu = 2.5. - Pass-through route sums are correct (lead 2+3 = 5; with the middle leg expedited, 2+2 = 4). - A stage that is one period short of cover gives service 0 and backorder cost b x mu. - A DC that is 4 units short rations 2.5/1.5 between stores with demand 5/3, giving fill 0.5/0.5.  The builder's stated simplifications all still apply: - no pipeline holding cost - order cost charged as K/R per review epoch - expedite premium charged on the analytic mean flow - proportional rationing - no capacities; fractional shipments - fixed review phases - each evaluation pools 20 paths x 160 periods (32 warm-up) and starts from an approximate steady state - the planner (re-base) rule on structural moves - backorder cost is b per unit not served on arrival (v1 actually charged per unit-period of backlog)  What I changed is problem calibration only: the simulator, the moves, set/get, describe and hints are untouched.  (1) Calibration now judges designs the way the evaluator does. The builder judged each design by the MEAN of 2 seeds with service >= target + 0.01. As a result, the heuristic design was infeasible under the evaluator's chance constraint (24 replicates) on about 7 of 30 problems. Now each design is judged by its worst service and worst holding over 4 fixed seeds, with service 

