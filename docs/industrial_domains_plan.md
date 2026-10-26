# Industrial-scale domain plan (DREAMPlace-level)

*Generated 2026-10-01 from a per-domain research sweep (7 researchers + 7 skeptic verifiers + synthesis). Per-domain verified proposals, candidates and sources are in the appendix.*

# Portfolio plan: rebuilding all seven design domains at DREAMPlace-level complexity

**Where each domain stands against the bar.**
- **Meets it:** placement.
- **Meets it if one gate passes:** traffic. The simulator (MOSS) is unmaintained and its benchmark data is gone.
- **Meets it only in part:** grid, epidemic and thermal Tier A. In grid, a single solve reaches DREAMPlace size only on the 78k-bus tier. In epidemic, the per-district detail is filled in by rules, not optimised. Thermal Tier A has about 100 levers over at most 2.6e5 via cells, not millions of connected objects.
- **No open tool reaches the bar:**
  - **Supply chain.** No public simulator handles 100K+ stages, and real networks stop at 2,025 stages. We would need our own engine plus synthetic networks.
  - **Structures.** No open GPU solver at 1e6 elements has a usable licence, and the CPU route is unaffordable. We would need to write our own port, about 3–6 weeks.

**How the agent and solver split the work, in every domain.** The agent changes one of roughly 30–100 addressable handles per turn: a region, group, weight, policy rule or budget. A solver then recomputes 1e5–1e7 variables, and a full evaluation measures the result. That matches DREAMPlace, where an agent would set macro positions, region constraints and net weights while the placer handles the bulk.

## 1. Domain table

Costs are taken from the verified proposals. "Unmeasured" means the number has no measurement on H100 or Blackwell yet.

| Domain | Simulator (licence) | Benchmark scale | What one turn changes | Full eval cost | Harvest cost, ~300 problems | Bar |
|---|---|---|---|---|---|---|
| **placement** (replaces circuit) | DREAMPlace 4.3.x at a pinned commit (BSD-3), apptainer image with CUDA 12.8 for sm_90 and sm_120. Xplace as fallback. OpenROAD GRT+STA as auditor. | About 45–55 public netlists with 0.1–2.5M cells: ISPD05/06 movable-macro variants (16), ICCAD15 superblue (8, macros freed), DAC12 (10), ISPD15 designs over 100K cells (about 12), TILOS and ChiPBench LEF/DEF. 6–8 variants each. | One handle out of 25–50: a macro group's slot, spread or flip (K=12–24 groups); a cell cluster's net weight or fence scale; target_density, halo or routability inflation; or a critical-path weight. Each evaluation re-places all standard cells from scratch (global placement + legalization). | Designs of 0.1–0.65M cells: 8–16 s. Designs of 0.68–2.5M cells: 35–190 s. These are V100/A100 figures; H100 is unmeasured. | 175–660 H100-h. Per turn, 20 clustered-netlist proxies are screened and the top 5 get a full evaluation; mix is 225 small + 75 large. | **Meets** |
| **grid** | ExaModelsPower (MIT). Ipopt+MA27 on CPU is the evaluation of record; MadNLP+cuDSS on GPU only for networks of 19k buses and up. lightsim2grid 1.1 (MPL-2) for N-1 AC contingencies. PowerModels as auditor. | PGLib v23.07: 45 networks with 1k+ buses, each in 3 regimes (typ/api/sad). Tiers at 1.3–5k, 5.6–10.5k, 19–30k and 78k buses. Test networks held out: 13659_pegase, 24464_goc, GO C3 23,643. | One preventive-security handle out of 24–40 curated: a branch rating fraction, a generator Pmax fraction, a shunt, or a phase-shifter angle. Each evaluation re-solves the full AC-OPF (11k–675k variables) and 100–300 post-contingency AC power flows. | 1.5 s at 1–3k buses, 8–14 s at 7–10k, 20–55 s at 19–30k. At 78k: 210 s on CPU or 20–25 s on GPU. | About 1.3k core-h at the default ladder, roughly half with a candidate cap. Plus 100–200 core-h to build the problem bank and about 90 H100-h (180 SU) for the 78k tier. | **Partial.** A single solve reaches DREAMPlace size only at 78k buses (5% of problems). The full N-1 problem (about 1.4e9 variables) is beyond what monolithic solvers can handle. |
| **traffic** | MOSS python-moss 1.4.0 (MIT, GPU), subject to a week-1 go/no-go gate. SUMO 1.27.1 for validation and as fallback. | BeST Berlin: 71.7k edges, 2,249 signalised junctions, 2.25M trips. mosstool's Beijing map. 4–8 cities built from OpenStreetMap with at least 5k junctions and 200k trips each. The moss-benchmark archive returns 404. | One element of a focus district (K=24–64): a junction's cycle, split, offset or adaptive toggle; a corridor's progression speed, cycle or bias; or a road's lane plan, speed limit or closure. The whole city runs 7:00–8:15 from a warm checkpoint. | About 10–30 s on H100 (unmeasured). | 100–350 H100-h with a 900-step screen and the full 4,500 steps for the top 4. Without screening: 350–1,000. SUMO fallback: 200–400 core-h per problem, so only a reduced set. | **Conditional on the gate** |
| **thermal** | Our own finite-volume solver (CHOLMOD/Pardiso/PCG on CPU) plus an adjoint via planner. 3D-ICE 4.0 (GPL) as reference. Tier B: Open3DBench memory-on-logic designs with DREAMPlace. | 128²×10 grid (164K unknowns) for training, 256² and 512² for evaluation. FloorSet-Lite instances with 21–120 blocks over 2–3 tiers. ATPlace_pub cases for 2.5D. Tier B: 8 netlists of 32K–1.08M cells. | One of: a block swap, tier move or DVFS step; a via-region keep-out or fill cap; or the global via budget, HTC or TIM. The planner re-places per-cell via density (1.6e4–2.6e5 cells per tier) and then solves for temperature. | About 105 core-s per turn in total. | About 175 core-h with screen-then-verify (670 without), plus about 180 core-h of audit. Tier B: 30–50 H100-h for 30–50 problems. | **Below the bar for Tier A.** Only Tier B reaches it, and it duplicates placement. |
| **epidemic** (replaces seir) | MEmilio 2.4.1 (Apache-2.0) graph-ODE SECIRVVS model. MEmilio ABM on the Brunswick twin as auditor. Optional Covasim (MIT) city of 1M agents. | Germany: 400 districts, 64.8k node states, about 53k commuter edges, real RKI/DIVI data. Unions of adjacent Länder add topologies. | One cluster lever (6–10 clusters plus a national element): NPI trigger threshold or strength, commuter testing, vaccine priority, NPI duration, or elderly-first share. DynamicNPIs and a dose allocator expand these into per-district, per-day schedules. | 30–150 core-s (unmeasured). | 1.2–6.0k core-h with the cap at 24 candidates (6.8–34k at the default 160). ABM audit 1.0–3.9k core-h. Optional Covasim city 2.1–4.2k core-h. | **Partial.** The state is the right size, but the bulk the agent drives is expanded by rules, not optimised. |
| **supply chain** (replaces inventory) | Our own batched PyTorch daily simulator, which has 4 known bugs to fix, plus our own sequential-LP GSM over the full network on HiGHS. | Family A: the 38 real Willems chains, 8–2,025 stages. Family B: synthetic networks of 20K–200K+ stages and up to 1.6M arcs, made by composing Willems chains and adding a FreshRetailNet distribution tier. | One of: a group's safety factor z (including echelon-wide moves); a hub's service-time pin or expedite; or a full GSM re-solve. Each evaluation simulates 465 days with common random numbers. | Family A: 0.5 s per batched turn (21 candidates × 8 replicates, RTX 3090). Family B: 22–44 s per turn. Warm LP re-solve: 26 s per pin candidate at 202K stages. | Family A: about 1 GPU-h + 2 CPU-h. Family B (about 120 problems): 15–30 GPU-h + about 140 CPU-h. | **No open tool.** Real data is 100–1,000× below the bar. |
| **structures** (replaces frame3d) | Our own GPU matrix-free hex8 SIMP solver, re-implemented from the Traff 2023 and Aage 2015 papers (the DTU GPU code has no licence). TopOpt_in_PETSc (LGPL) as an interim CPU solver for the pilot and for cross-checks. | 5 procedurally generated families (bridge girder, bracing, roof support, bracket, cantilever/MBB/torsion), 0.5–2M elements at native resolution, 4–8M for final verification. 30 held-out problems at 2M. | One of: a zone's material cap or minimum member size (16–64 zones); a load-case weight (1–6 cases); a persistent pad's radius (up to 8 slots); or the base filter radius. The solver re-optimises all 0.5–2M densities. | 12–27 s on H100 at 0.5M elements and 3 load cases (estimate). | 350–600 H100-h plus 30%. The CPU route would be about 120k core-h, so it is used only for a 10-problem pilot (about 4k core-h). | **No open tool.** The scale meets the bar; a 3–6 week port is needed. |

**Totals:** about 0.9–2.0k H100-h (1.7–3.9k SU on Bridges-2 GPU-shared) plus about 8–17k CPU core-h, before pilots.

## 2. Shared infrastructure

### a. Contract changes in `dataxp/`

All seven proposals need these.

1. **Move hook.** Add `Domain.candidate_moves(st, rng) -> List[Candidate]`. The generic `enumerate_candidates` (`domain.py:115`) has three problems at this scale:
   - It applies only multiplicative factors.
   - It picks 24 elements at random.
   - It defaults to `max_candidates=160`.

   It cannot move a categorical, zero-valued, circular or toggle parameter. With the hook, each turn offers 20–24 moves: 12–16 ranked by sensitivity or hotspot, 4–8 random distractors (so the median stays informative for G2), and 1 null move where drift QA is needed. `Candidate.v` must allow typed values (index or label) together with the operation kind.
2. **Candidate cap.** `ladder.py` calls the enumerator with its defaults, 160 candidates. That inflates cost 6–8× in grid, traffic and epidemic. Pass the cap per domain.
3. **Batched evaluation.** Add `Domain.evaluate_batch(states, seeds)` and route `PhiEvaluator._sample` (`domain.py:183`) through it; today it loops one call at a time. Supply chain (candidates × replicates), structures (19 evaluations per GPU batch) and multi-engine traffic all need this.
4. **Screening tier.** Add `Domain.screen_batch(states, seeds)` for a low-fidelity score. Every score records which fidelity produced it.
5. **Problem bank.** `make_problem` runs one problem at a time in the parent process (`run_generation.py:147`). Acceptance loops that call the reference solver take minutes per draw, so build a bank with a parallel SLURM array and make `make_problem` a lookup into it.
6. **Resident worker state.** `_init_worker` loads the heavy objects once per process: PlaceDB, the Julia runtime, the MOSS engine and its checkpoint, the factorisation cache, the GPU multigrid hierarchy. All evaluations of one problem stay on the same worker and the same GPU. The state itself is only a handle vector plus the problem id, because `clone()` runs once per replicate.
7. **Determinism and replay.** `replay_trajectory` (`harvest.py:92`) rebuilds every state from scratch. So warm starts may come only from a fixed per-problem anchor (grid, traffic checkpoint) or from a field that is serialised in the state (thermal via density, structures anchor density). They may never come from hidden parent-solver state.
8. **Ladder trimming at scale.**
   - phi_d2 adds 6×24 evaluations per turn on the first 40 problems (`run_generation.py:131`). Cap it at about 10 problems, from the cheapest tier only.
   - redteam: reduce from 12 problems to 6.
   - blind: allow at most k full evaluations per turn, in random order.
   - Cache the initial-state sample once per problem; today each policy's PhiEvaluator recomputes it.
   - Cache evaluations by (problem, handle hash, seed, fidelity).
9. **Replicates.** `n_rep_max=24` is unaffordable except where replicates are batched (supply chain with common random numbers, structures with load superposition). Make traffic, grid, epidemic, thermal and placement deterministic (`stochastic=False`), measure seed or multi-start noise offline, and use it as the G2 noise floor.

### b. Change to the gold-label procedure (needs owner approval)

**Current rule:** gold is the argmax of full-fidelity Phi over all enumerated candidates.

**Proposed rule (screen-then-verify):**
- Screen all N = 20–24 candidates with the low-fidelity tier.
- Fully evaluate the top k = 4–5, plus the candidate ranked median by the screen, plus the null move where the domain uses one.
- Gold is the argmax of full-fidelity Phi within that verified set.
- Preference pairs and the G2 statistics (top vs median, top-1 vs top-2) are computed only from full-fidelity scores.

**Acceptance pilot per domain,** with all candidates fully evaluated:
- Spearman correlation between screen and full ranks ≥ 0.7.
- The full-evaluation argmax is in the screened top k at least 80% of the time.
- If either fails, raise k.

| Domain | Screening tier |
|---|---|
| placement | Global placement on a netlist clustered to 2–5% of instances |
| grid | None on tiers below 19k buses. For 19k+ and 78k: an untested DC screen (DC-OPF plus DC N-1 contingencies). |
| traffic | 900-step horizon |
| thermal | Via field frozen: back-solves or warm PCG |
| epidemic | Pruning of moves that cannot change anything. The GNN surrogate is dropped (10–27% error). |
| supply chain | 180-day horizon, 2–4 replicates |
| structures | 0.26M elements with 10 iterations |

### c. Compute, containers and data

**GPU.** Bridges-2 GPU-shared H100 (`--gres=gpu:h100-80:1`, 2 SU per GPU-h) for placement, traffic, structures, supply-chain Family B, the 78k grid tier and thermal Tier B. Keep the drc RTX PRO 6000 Blackwell for FP32 pilots: DREAMPlace in float32, MOSS, the supply-chain simulator. Do not use it for FP64 OPF; its FP64 rate is 1:64. Pin one GPU architecture per problem so labels replay exactly.

**CPU (trace cluster).**
- Use the `biosimmlab` partition with `--qos=normal` for pilots; it is capped at 128 cores per account.
- Use `cpuonly` with `--qos=normal` for arrays above 128 cores. Pass the QoS explicitly, because the MCP can inject `cpuonly-debug-qos`, which caps the job at 24 cores.
- Write to `/trace/scratch/biosimmlab/wxu2`; the group directory is full.
- Set `OMP_NUM_THREADS=1`.
- Check that trace's glibc is 2.27 or newer for the memilio and lightsim2grid wheels.

**Apptainer images:**
1. EDA: DREAMPlace (pinned), Xplace, Mt-KaHyPar, OpenTimer; CUDA 12.8.
2. ORFS for the OpenROAD audit.
3. julia-power: ExaModelsPower (pinned), MadNLP/cuDSS, a PackageCompiler sysimage, juliacall and lightsim2grid. HSL MA27 needs a manual academic-licence download.
4. traffic: two conda environments, because mosstool needs numpy ≥ 2 and python-moss needs numpy < 2. Plus SUMO 1.27.1.
5. cpu-sci: scipy, pyamg, scikit-sparse, pypardiso, memilio-simulation, covasim, and a 3D-ICE build.
6. torch-gpu: PyTorch and CuPy with CUDA 12.x, for the supply-chain simulator and the structures port.

**Data.** Mirror all datasets on `/ocean` and trace scratch:
- ISPD/ICCAD benchmarks via BBOPlace-Bench's Google Drive mirror or ispd.cc; the cerc.utexas.edu host returns 403.
- PGLib and GO C3.
- BeST Berlin.
- RKI/DIVI Zenodo snapshots and the Brunswick trip chains (Zenodo 13318436).
- FreshRetailNet-50K and the Willems chains.
- FloorSet-Lite and CircuitNet textures.
- SimJEB.
- **TOP3D_XL voxel domains: download now; the link expires in November 2026.**

### d. Fidelity audits for G3

One independent tool per domain, run on sampled gold states:

| Domain | Audit |
|---|---|
| placement | OpenROAD global routing + OpenSTA on the LEF/DEF designs |
| grid | PowerModels `compute_ac_pf` with PV/PQ switching on 1% of contingencies |
| traffic | SUMO re-scores 30 BeST problems |
| thermal | 3D-ICE at 128² (and 256² on 10%) |
| epidemic | MEmilio ABM on the Brunswick twin |
| supply chain | stockpyl, InventoryManagement.jl and ReplenishmentEnv on chains of 100 stages or fewer |
| structures | TopOpt_in_PETSc on 10% of problems |

Data-quality checks use a stronger model, never the model being tuned.

## 3. Observations and action grammar at this scale

### describe()

- Each observation is self-contained, about 1.5–3.5K tokens, and never dumps per-object data. Sections, in a fixed order:
  1. A header with the instance's scale facts.
  2. A constraint dashboard: value, limit, gap, status, plus the noise σ so changes inside the noise are visible.
  3. Coarse maps.
  4. Top-k lists.
  5. A table of the addressable elements only, up to about 45 rows with the rest aggregated.
  6. The last 3 actions with their deltas, including requested vs realised values (for example legalised displacement, or how many via cells were re-planned).
- Element IDs stay stable for the whole trajectory: groups, clusters and zones are fixed per problem, and pad slots are persistent.
- Each lever is shown as live or dead (a trigger that never fires, a hub that is already stocked), so no-op moves are visible to the agent.
- If `hint_match_frac` exceeds 0.8, coarsen the sensitivity columns so the data does not teach "copy the hint".
- SFT context: 20 turns × 3K tokens is about 60K tokens if every observation is kept. History should therefore be windowed: the current observation plus the last-3 action deltas.

| Domain | Maps | Top-k lists |
|---|---|---|
| placement | Density and RUDY maps on the action slot grid (G×G), macro groups drawn as letters | Hotspots with the groups and clusters responsible; worst paths |
| grid | Zone aggregates (8–12 METIS zones) | Worst contingency→branch pairs; voltage outliers; handle table with sensitivity rank |
| traffic | District table | Queues per phase group at focus junctions; spillback lanes; corridor arrivals on green |
| thermal | 8×8 maps per tier: temperature, via density, keep-outs | Hotspots with the owning block; swap partners; whitespace holes |
| epidemic | Cluster table with peak vs threshold | Top districts by ICU excess; weekly national series |
| supply chain | Group table; fill-rate decile histogram | Worst demand stages with their upstream bottleneck; hub pin states |
| structures | 3 ASCII max-projections with zone borders | Zones by strain-energy share with shadow prices; pad table; open hotspots |

### Action grammar

This replaces the single SCALE operation.

| Operation | Use | Examples |
|---|---|---|
| `SCALE(e,p,f)` | Continuous ratios | Net weight, rate_frac, cycle, z ×1.25, vf_cap |
| `SHIFT(e,p,±d)` | Additive or circular | target_density ±0.03, offset ±0.25 (wraps around), shifter ±5°, shunt ±0.25 Qcap, z ±0.25 |
| `SET(e,p,v)` | Categorical | Macro slot "5,6", flip FN, lane_plan 2, DVFS 3, tier 1, hub pin, HTC level |
| `TOGGLE(e,p)` | Binary | Adaptive signal, keep-out, fence on/off, lane closed |
| `SWAP(a,b)` | Thermal blocks | Swap two blocks |

- Group and echelon targets, such as `SHIFT(echelon:demand, z, +0.25)` or corridor macro-elements, are needed so that 20 turns can close the initial violations.
- Element names carry a type prefix: G (macro group), C (cell cluster), P (path group), br (branch), g (generator), j (junction), r (road), c (corridor), B (block), R (via region), Z (zone), L (load case), pad.
- Each domain's `set()` legalises the action and reports what actually happened.
- The `<action>` SFT tag, the parser and the harvest JSON all need the extended grammar.

## 4. Build order

**Track 0, weeks 0–2: shared contract changes (section 2a), tested on the existing toy domains.** Regression check: with screen = full, the G1–G3 numbers must not change. Run these cheap week-1 gates in parallel:
- DREAMPlace builds with CUDA 12 and reproduces the README HPWL table within 5% on adaptec1 and bigblue4.
- The MOSS gate (a)–(d).
- One MEmilio SECIRVVS run: time and memory at 90 and 180 days.
- The Julia stack and HSL install.
- Fragile data downloads.

Then build the domains in this order:

1. **placement.** The tooling is the most mature: BSD-3, actively maintained, real benchmarks. The decision layer has published precedents (Circuit Training, MaskRegulate, VeoPlace). It exercises the whole GPU path (resident worker, screening tier, batching), which thermal Tier B and structures reuse. Pilot for 20–40 GPU-h on adaptec1 (movable macros) and superblue18. Go if the proxy-vs-full Spearman is at least 0.7, top-5 recall at least 0.8, and the G2 gap is at least 2σ_seed in at least 70% of turns.
2. **grid,** in parallel because it runs on CPU and competes for nothing. The solvers are mature and the networks are real. Pilot of about 80 core-h. Go if, on a 3-start re-solve, the top-vs-median gap exceeds 10× the noise floor in at least 80% of states, and base-case flows agree with the OPF to within 1% of rating.
3. **supply chain.** Cheapest overall (Family A costs about 1 GPU-h). It can start only after the simulator bugs are fixed and validation passes against stockpyl and InventoryManagement.jl. Report the top-1 vs top-2 margin.
4. **thermal Tier A.** CPU-cheap, and probe scripts already exist. Gates: the solver agrees with 3D-ICE to MaxAE ≤ 0.5 C, and screen vs verified top-1 agreement is at least 80%. Decide on Tier B once placement is running.
5. **epidemic.** Wheels install with pip and the API is checked against the bindings. Run on CPU once the timing and memory pilot is in.
6. **traffic,** only if the week-1 gate passed. Otherwise fall back to SUMO on a reduced problem set.
7. **structures,** the long pole. Start the GPU port in week 2 and run the G1/G2 pilot on TopOpt_in_PETSc on trace in the meantime. Port acceptance:
   - compliance within 1% of PETSc;
   - multigrid converges in at most 200 iterations on 20 nearly black-and-white designs;
   - metrics reproduce bit for bit;
   - the G2 gap exceeds 3× the self-consistency noise in at least 80% of turns.

## 5. Decisions you need to make

1. **What counts as "DREAMPlace level".** Is a problem that is harder overall acceptable where a single solve is smaller (grid N-1, epidemic rule-expanded bulk, thermal Tier A)? Or must every turn solve at least about 0.2M decision objects? The second option forces: grid mostly on the 78k tier plus GO C3; epidemic on the Covasim 1M-agent city; thermal on Tier B only; supply chain on Family B only.
2. **Approve screen-then-verify gold labels** (section 2b), its thresholds, and the trimmed ladder (phi_d2, redteam, blind). The trimming changes the baseline comparison that would be reported.
3. **Thermal Tier B.** It overlaps placement (both use DREAMPlace). Keep it in thermal, fold it into placement, or drop it?
4. **Synthetic instances.** Real data is sub-scale or missing in several places: supply-chain Family B, all structures instances, FloorSet blocks in thermal, and grid security definitions (PGLib has no contingency lists or credible emergency ratings). Is that acceptable?
5. **Licences and release policy.**
   - Closed binaries: exclude HeteroSTA and NCTUgr?
   - Licensed or copyleft tools: accept the HSL academic licence, and run GPL tools (3D-ICE, CUTh) only as separate processes?
   - Data terms: the Willems solutions article is CC BY-NC; OSM maps are ODbL; ISPD contest redistribution is unverified; BBOPlace-Bench, Open3DBench and ATPlace_pub have no licence file. Can derived problem files be released, or are they internal only?
6. **Budget.** About 0.9–2.0k H100-h (1.7–3.9k SU) plus 8–17k core-h. Confirm the Bridges-2 allocation and the caps: large placement designs at 25% or less of placement problems, the 78k grid tier at about 5%, structures at 300 problems or fewer.
7. **Traffic fallback rule** if MOSS fails its gate: SUMO with about 30–60 problems, UXsim (below the fidelity bar), or drop the domain.
8. **Structures.** Fund the 3–6 week GPU port, or keep frame3d / OpenSees, which are 2–4 orders of magnitude below the bar.
9. **Turn budget.** Keep 20 turns, with problems calibrated to 3–8 perturbed groups or 2–8 overloaded corridors and coarse group moves? Or raise the number of turns?
10. **SFT context.** Windowed history vs full history, given 2–3.5K-token observations.
11. **Hardware of record for labels.** H100 only, or Blackwell as well? Results may differ across GPU architectures.
12. **Epidemic.** Is a single country (Germany plus Länder subgraphs) enough variety for G1? Should the Covasim city become an eighth domain?

The `dataxp` code locations cited above are in `C:/Users/xuwen/designbench_llm_finetune/dataxp/`: `domain.py` (lines 115–138 and 183), `ladder.py` (lines 179–232), `run_generation.py` (lines 38, 131–132 and 147), and `harvest.py` (line 92).

---

# Appendix: verified per-domain proposals

## vlsi_circuit

**Domain:** circuit -> industrial-scale electronic design: mixed-size VLSI placement on public contest netlists (about 0.1M-2.5M cells), with DREAMPlace placing all standard cells each turn. The analog sizing track (ngspice + SKY130) does not meet the bar and stays dropped.

**simulator.** DREAMPlace 4.3.x at a pinned commit, built from source in an apptainer image (CUDA 12.x; sm_90 for Bridges-2 H100, sm_120 for the RTX PRO 6000), float32, deterministic_flag=1. Xplace is built in parallel as the fallback.

A persistent Python worker parses each base design into PlaceDB once, which removes the 4-45 s IO per evaluation. Every evaluation is a cold-start run:
- All macros sit FIXED at the agent's legalized layout.
- Std-cell GP + LG run from random centre initialisation with random_seed=seed.
- So evaluate() is a pure function of the visible decision variables.

DREAMPlace builds its operators, including the fixed-macro density map, in BasicPlace.__init__. Each evaluation therefore constructs a fresh NonLinearPlace from the resident PlaceDB with edited arrays: node_x/y for macros, net_weights, regions and node2fence_region_map.

What stays out of the screening loop:
- Warm start from the parent placement is only an optional accelerator, allowed if a pilot shows Spearman >= 0.9 against cold start. DREAMPlace has no incremental mode: GP runs more than 100 iterations regardless (Lgamma_step > 100) and re-initialises the density weight. Hidden cell positions would also make outcomes depend on history the agent cannot see.
- Timing-driven GP (about 5x cost) and NCTUgr routing (57-1,215 s per call).

Metrics per evaluation:
- DREAMPlace HPWL.
- Post-LG density overflow.
- RUDY.
- On timing designs, one post-LG STA with OpenTimer. HeteroSTA is optional (closed .so).

OpenROAD (apptainer, trace-cluster CPUs) runs GRT + OpenSTA on sampled gold states of the LEF/DEF designs as the G3 audit.

**benchmarks.** Base pool of about 45-55 netlists, all public. The original data mostly has FIXED macros, so macro-group decisions use movable-macro (MMS) or freed-macro variants.
- MMS, 16 configs in DREAMPlace test/mms: ISPD2005 adaptec1-4 and bigblue1-4 (211K-2.18M cells, 543-1,329 macros on the 6 macro-rich designs), plus ISPD2006 adaptec5 (843K), newblue1-4 (330K-646K) and newblue5-7 (sizes not re-verified, about 1.2-2.5M). ISPD2006 brings native density targets of 0.5-0.9.
- ICCAD2015 superblue, 8 designs: 768K-1.93M cells, with .lib/.sdc (DREAMPlace test/iccad2015.ot). Macros are freed as the 512 largest modules, as in BBOPlace-Bench. The die is shrunk because the contest floorplans are loose (MaskRegulate).
- DAC2012 superblue, 10 designs: 523K-1,365K nodes with routing resources, so RUDY demand/capacity is physical (DREAMPlace test/dac2012).
- ISPD2015 designs of 100K+ cells, about 12: mgc_des_perf_1/a/b (108-113K), edit_dist_a (127K), matrix_mult_1/a/b(/2/c) (146-155K), superblue11_a (926K, 4 fences), superblue12 (1.29M), superblue16_a (680K, 2 fences), superblue14/19. They have LEF/DEF fences and density limits; macros are fixed and few on the mgc_* designs.
- LEF/DEF designs with real libraries: TILOS BlackParrot (827K instances), MemPool Group (2.53M) and Ariane133/136 (about 170K), plus ChiPBench designs of 150K+ cells (bp 301K, isa_npu, subrisc).

These come from only about 5 netlist families: IBM ISPD05/06, IBM superblue (which appears in the DAC2012, ICCAD2015 and ISPD2015 versions), Intel ISPD2013-derived mgc, and open RTL. Split train/dev/test by family.

Data access: the DREAMPlace download host (cerc.utexas.edu/~zixuan) returned HTTP 403 on 2026-10-01. Use the BBOPlace-Bench Google Drive mirrors or ispd.cc and keep a copy on /ocean. Redistribution terms are not verified, so keep derived problem files internal.

**decision layer.** One agent turn sets one (element, param) handle. DREAMPlace then re-places all 0.1-2.5M std cells (cold GP + LG) and the simulator measures the result. Each problem has about 25-50 elements.

(E0) Placer knobs, kept only where they have a physical meaning:
- target_density in [max(util+0.05, 0.5), 1.0], additive steps of ±0.03/±0.06.
- Routability inflation cap max_route_opt_adjust_rate in [1.0, 2.5], with RUDY-based routability_opt_flag. Only on designs with routing capacity (DAC2012, ISPD2015, ICCAD2015, LEF/DEF).
- Global macro halo in rows, [0, 8].
- Dropped from the proposal: gamma and the initial density_weight. They are optimiser internals whose effects look like noise and cannot be explained from the observation.

(E1..EK) Macro groups, K = 12-24:
- Built by Mt-KaHyPar (MIT) clustering of the macro connectivity hypergraph. Bookshelf netlists are flat with obfuscated names, so there is no hierarchy to use.
- Param 'slot': an index on a coarse G x G die grid (G = 8-12), following MaskRegulate's grid action. This merges anchor x and y into one move, so one turn relocates one group.
- Param 'spread': [0.6, 1.6], ×0.85/×1.15.
- Param 'flip': one of {N, FN, FS, S}.
- set() packs the group's macros around the slot, then runs a deterministic macro legalizer against the other groups, fixed blocks and the die edge (DREAMPlace's macro_legalize op or a Tetris/LP pass; at most about 1.6K macros, under 1 s). Every state is therefore overlap-free, and the requested-vs-legal anchor displacement is reported.

(EK+1..EK+C) Std-cell clusters, C = 4-12, from Mt-KaHyPar:
- Intra-cluster net-weight multiplier in [1, 8], written to PlaceDB.net_weights (.wts on Bookshelf). Available on all designs.
- Fence on/off and fence-box scale in [0.6, 1.5], only on LEF/DEF designs (DEF REGIONS/GROUPS, or injected into PlaceDB.regions; untested for Bookshelf). DREAMPlace 3.0 was evaluated with at most 12 regions.

(Timing designs only) Top-8 critical-path groups: a static net-weight multiplier in [1, 10], measured by one STA call. The momentum net-weighting loop does not run.

Example turns: 'move macro group G7 to slot 5,6'; 'spread G3 ×1.15'; 'fence C2 at 0.8x'; 'path group P1 weight 4x'; 'target_density 0.90 -> 0.84'.

Contract changes:
- Categorical or additive param types (slot, flip, ±Δ) need per-param candidate generators, i.e. an overridden enumerate_candidates. Slot candidates are the 8 neighbours plus 4 slots chosen by hotspot or connectivity centroid.
- hint() blames the group or cluster with the largest HPWL share in the worst RUDY or density hotspot, or on the worst path.

Precedents: Circuit Training, VeoPlace and MaskRegulate all leave the cells to an analytical placer and give the agent the macros.

**inner solver.** DREAMPlace GP (Nesterov, electrostatic density, weighted-average wirelength) plus Greedy/Abacus LG over all movable std cells, cold-started with a fixed seed, with macros fixed at the agent's legalized layout.
- Fences enforced through multi-electrostatics and region-aware LG.
- Net weights taken from PlaceDB.
- Optional RUDY cell inflation when the routability knob is on.

Not run per candidate:
- Internal DP, which runs only on each turn's chosen state.
- Timing-driven GP.
- NCTUgr.

Pre-screen tier: the same GP runs on a clustered netlist, with std cells coarsened to about 2-5% of the instance count and macros kept exact. It ranks the 20 candidates per turn, and the top 5 get a full evaluation.

**metrics objective constraints.** Objective (minimise): hpwl_ratio = post-LG HPWL / reference HPWL (unitless). The reference is DREAMPlace's two-stage macro flow on the movable-macro version of the same base and variant, run on the same pinned build. HPWL drifts ±5-7% across versions (issue #242), so references from the README or from papers are not comparable.

Constraints, with goal names:
- maximum_density_overflow: post-LG overflow against the design's density target (ISPD2006 targets 0.5-0.9, ISPD2015 limits 27-90%, otherwise set per problem), on a fixed bin grid that does not depend on the agent's target_density knob. The proposal's 'GP overflow <= 0.10' cannot bind, because GP stops at stop_overflow = 0.1 by construction.
- maximum_rudy_top5: mean demand/capacity of the top 5% RUDY bins, unitless. It is physical only where routing capacity is given (DAC2012 .route, LEF tracks). On ISPD2005/2006 DREAMPlace assumes 1.5625/1.45 tracks per unit, so it is synthetic there and thresholds are calibrated per design. MaskRegulate reports RUDY is often inaccurate, so OpenROAD GRT audits it.
- minimum_wns and minimum_tns (ns): on ICCAD2015, TILOS and ChiPBench only, from one post-LG OpenTimer run under the per-problem clock.

Guards that must be 0 and are not used for ranking, since set() legalization and region-aware LG make them hold almost always: macro_overlap_um2 and fence_violation_cells.

Info only: average LG displacement in sites, anchor displacement, and macro regularity (MaskRegulate).

Noise: deterministic_flag=1 removes run-to-run noise (README: minor overhead). evaluate(state, seed) maps seed to random_seed, so seed replicates give the noise floor σ used by G2.

**observation summary.** About 2-3 KB of text and never per-cell data:
(1) Header: design and family, #cells/#nets/#macros, die size in µm, utilisation, density target, clock period, slot-grid size G.
(2) Metric line with constraint status and margins, e.g. 'HPWL 1.12x ref | dens-ovf 0.14 (<=0.08 VIOL) | RUDY top5 1.18 (<=1.05 VIOL) | WNS -1.2 ns (>= -0.8 VIOL) | TNS -2.1e4 ns'. It also states the design's seed-noise σ, so deltas within noise are visible.
(3) G x G (or 16x16) density and RUDY maps as digit grids, with macro groups as letters on the same slot grid the actions use.
(4) Top-5 hotspots: slot or bbox, peak value, and the overlapping groups and clusters with their HPWL share.
(5) Top-5 critical paths (timing designs): slack, source and sink cluster, groups crossed, length in mm.
(6) Element tables:
- Macro groups: id, #macros, area %, slot, spread, flip, HPWL of incident nets, displacement from the requested anchor.
- Clusters: id, #cells, fence, centroid slot, weight.
- Knob values.
(7) Last-action deltas for every metric.

**problem generator.** make_problem(index) takes base = pool[index % N] (about 45-55 netlists, sampled stratified by family) and a variant seed derived from the index. Then:
- Seeded Mt-KaHyPar grouping into K macro groups and C cell clusters.
- Start from the reference layout and perturb only 3-8 groups (moved to random far slots, centre-collapsed or edge-stacked). With one group move per turn, 20 turns can then reach feasibility; wrecking all 12-24 groups would need more turns than the budget.
- Die-area scale of 0.85-1.15. ICCAD2015 is shrunk to 0.6-0.8 because its contest die is loose.
- 0-3 inserted placement blockages.
- Fence assignments on LEF/DEF designs.
- Clock period scaled to 0.7-0.95 of what the reference meets, on timing designs.
- Thresholds set between the initial and reference metrics. Keep the problem only if the reference is feasible and the initial state violates 1-3 constraints; otherwise resample.

6-8 variants per base gives about 300-400 problems.

G1 check before generating traces: across-problem variance of initial hpwl_ratio, the mix of violated constraints, and the share of each family.
G2 check in the pilot: top-minus-median Phi gap of at least 2σ_seed in at least 70% of turns.

**cost budget.** Per full evaluation (cold GP+LG, PlaceDB resident, float32). V100-class numbers; H100 unmeasured:
- S class (0.1-0.65M cells): GP 5-9 s + LG 0.4-2 s (TCAD Table II, float64) + operator rebuild and filler setup (unmeasured, assume 2-5 s), so about 8-16 s.
- L class (0.68-2.5M: bigblue2-4, adaptec5, newblue5-7, superblue, MemPool): about 35-190 s. Basis: bigblue4 GP 25 + LG 9 s; VeoPlace 52-190 s on an A100; MemPool GP 178 s on a TITAN V.
- Timing designs add one STA per full evaluation (OpenTimer time on 1-2M cells unmeasured).
- Timing-driven GP (about 5x) and NCTUgr (274-1,746 s per run) are excluded.

Without cuts (phi_select, 20 candidates x 20 turns = 400 full evaluations): 0.9-1.8 GPU-h per S problem and 3.9-21 GPU-h per L problem.

Two-tier cut: per turn, 20 clustered-netlist proxy evaluations (assumed 1-10 s) plus full evaluations of the top 5, giving 100 full + 400 proxy per problem. That is about 0.3-0.8 GPU-h per S problem and 1.3-6.4 GPU-h per L problem.

For 300 problems (225 S + 75 L): about 75-180 + 100-480 = roughly 175-660 H100 GPU-h on GPU-shared. That is above the proposal's 100-250 GPU-h, which rested on an unmeasured warm-start speedup.

Further cuts:
- Cap L-class problems at about 25%.
- Run 2-4 workers per 80 GB GPU if profiling shows low single-worker utilisation (not verified).
- Run DP and the 3-seed finalist re-evaluation only on each turn's chosen state.
- Use the trace-cluster CPUs (DREAMPlace CPU mode is about 13x slower than a V100) only for S-class proxy evaluations and for the OpenROAD audit, a few hundred CPU-h.

Pilot first, about 20-40 GPU-h: MMS adaptec1 and superblue18, 2 problems each. Measure H100 seconds per evaluation, deterministic overhead, σ_seed, proxy-vs-full Spearman (target >= 0.7, top-5 recall >= 0.8) and G2 separation, then decide go/no-go.

**risks.** SKEPTIC: Kept: DREAMPlace as the primary placer (BSD-3, active through 2026-07), dropping analog, and OpenROAD as the offline auditor.

Corrected:
(a) ISPD2005 and ICCAD2015 macros are fixed in the original contests. Macro moves therefore need MMS (ISPD2005 + ISPD2006, 16 configs) or freed-macro variants. ICCAD2015 macro counts depend on the definition (512 largest in BBOPlace, 99-1,629 in VeoPlace), not 424-441.
(b) Runtime figures: VeoPlace rollouts take 52-190 s, not 52-83. DREAMPlace GP on MemPool Group takes 178 s, not 72. The DP times in the TCAD tables are external NTUplace3. NCTUgr routability costs 274-1,746 s per run, so both NCTUgr and timing-driven GP are out of the screening loop.
(c) Warm start is replaced by cold-start evaluation as the default. DREAMPlace has no incremental mode, and hidden cell state would make outcomes depend on history.
(d) 'Density overflow <= 0.10' is the GP stop rule and cannot bind. It is replaced by post-LG overflow at contest density targets, and fence violation and overlap become guards.
(e) Decision layer: gamma and initial density-weight knobs dropped; anchor x/y merged into one grid-slot move (MaskRegulate); only 3-8 groups perturbed so 20 turns suffice; fences limited to LEF/DEF designs.
(f) HeteroSTA is a closed .so, so OpenTimer is the default timer.
(g) The benchmark host returns HTTP 403; BBOPlace-Bench has no LICENSE file; AutoDMP has been stale since 2023-07. Ariane is about 170K cells, not millions, and the ChiPBench counts are fixed.
(h) Cost re-estimated at about 175-660 GPU-h, not 100-250.
(i) Added the MMS/ISPD2006 and DAC2012 pools and the MaskRegulate precedent.

Other risks:
(1) Build. DREAMPlace has only been tested with PyTorch <= 2.0 and Python <= 3.9. Open issues #212 (nvcc with PyTorch 2.0 + CUDA 11.8), #195 (OpenTimer doctest on new glibc) and #238 (HeteroSTA link). #242 shows 4.3.0 is off by +6.7% (adaptec1) and -5.1% (adaptec2) against the README. Pin the commit, recompute every reference on it, and keep the Xplace build ready.
(2) G2 noise. Deterministic mode fixes run-to-run results, but GP is chaotic in its inputs. Seed spread of ±0.3-1.7% (VeoPlace) can match late-trajectory candidate deltas. Measure σ_seed per base, require a top-minus-median gap of at least 2σ, and re-evaluate finalists over 3 seeds.
(3) Proxy tier. ChiPBench shows macro HPWL correlates weakly with routed wirelength. The clustered-netlist proxy must reach Spearman >= 0.7 against full evaluation, or k is raised, which raises cost.
(4) Proxy-signoff gap. RUDY is inaccurate and wirelength correlates weakly with WNS/TNS (ChiPBench, MaskRegulate). The OpenROAD GRT/STA audit on LEF/DEF designs is part of G3, and ISPD2005/2006 RUDY is synthetic.
(5) Heterogeneous constraint sets. Timing exists only on ICCAD2015/TILOS/ChiPBench, fences only on LEF/DEF, and physical routing capacity only on DAC2012/ISPD2015/ICCAD2015/LEF-DEF. Stratify problems by family.
(6) Family leakage. Superblue appears in three contests and ISPD05/06 are all IBM. Split by family.
(7) Custom code: group packing and legalizer, PlaceDB array edits for fences and net weights on Bookshelf, operator rebuild per evaluation (setup time unmeasured), and the clustered-netlist proxy.
(8) Data and licensing. The download host returns 403; contest-data redistribution terms are unverified; NCTUgr and HeteroSTA are closed binaries.
(9) Budget. L-class problems dominate the cost. H100 seconds per evaluation, workers per GPU and GPU-shared queue times are all unmeasured until the pilot.

**Candidates considered:**

| name | license | scale | runtime/eval | verdict |
|---|---|---|---|---|
| [DREAMPlace 4.3.x (limbo018)](https://github.com/limbo018/DREAMPlace) | BSD-3-Clause (verified). Two optional parts are not open source. HeteroSTA, the v4.3 GPU timer, ships as a closed .so that is free for academic use after email registration. NCTUgr, used for router-guided routability, is an external binary. The OpenTimer timing path (git submodule) is open. | Verified figures: - ISPD2005: 8 designs, 211K-2,177K cells, 221K-2,230K nets (TCAD'21 Table II). Macros are FIXED in the original contest. BBOPlace-Bench Table V gives 543-1,329 macros for adaptec1-4, bigblue1 and bigblue3. - MMS: ISPD2005 plus ISPD2006 with movable macros. DREAMPlace ships 16 configs in test/mms (adaptec1-5, bigblue1-4, newblue1-7). ISPD2006 adds adaptec5 (843K objects), newblue1 (330K), newblue2 (442K), newblue3 (494K) and newblue4 (646K), with density targets of 0.5-0.9. - ICCAD2015: 8 superblue designs, 768K-1.93M cells, 771K-1.9M nets, 2.5M-6.4M pins, with .lib/.sdc. Macros are fixed: the DREAMPlace 4.0 TCAD paper says 'Movable macros are not included in any benchmark'. The macro count depends on the definition: BBOPlace takes the 512 largest modules, and VeoPlace reports 99-1,629. - ISPD2015: 16 contest designs, plus 4 more configs in DREAMPlace test/ispd2015 (20 total). They range from 28.9K to 1.29M cells. The mgc_* designs have 0-7 fixed macros; superblue12 has 89, superblue16_a 419 and superblue11_a 1,458. Eight designs have fences (1-12 regions each). Density limits are 27-90%. - DAC2012: 10 superblue designs, 523K-1,365K nodes, with routing resources (TCAD Table V). - The 10.5M-cell industrial designs in TCAD'21 are proprietary. | TCAD'21 Table II (verified; V100, float64, ISPD2005): - GP 5 s (adaptec1) to 25 s (bigblue4). - LG 0.4-9 s. - IO 4-45 s. - DP 25-332 s, but that DP is the external NTUplace3, not DREAMPlace's own. - Total with that DP: 34-413 s. - CPU, 40 threads: GP 67 s (adaptec1) to 655 s (bigblue4), about 13x slower than V100.  Other verified points: - Routability with NCTUgr on DAC2012 (RTX 2080 Ti, float32): nonlinear placement 44-293 s plus global routing 57-1,215 s, so 274-1,746 s per run. - Timing-driven GP takes about 5x the runtime (DREAMPlace 4.0 TCAD; STA every 15 iterations after iteration 500). - VeoPlace on an A100 (DREAMPlace 4.3): 52-190 s per rollout, not 52-83. - DG-RePlAce on a TITAN V: DREAMPlace GP takes 61 s on BlackParrot and 178 s on MemPool Group (the proposal said 72 s). - v4.1 macro flow (README; GPU not stated): adaptec1 27.6 s, bigblue4 234.5 s.  H100 and Blackwell times have not been measured. | PRIMARY inner solver (confirmed). Corrections: macro decisions need the MMS or freed-macro variants. Screening runs never use NCTUgr routing or timing-driven GP. Pin the build and recalibrate every reference on it. |
| [Xplace 3.0 (CUHK)](https://github.com/cuhk-eda/Xplace) | BSD-3-Clause (verified via GitHub API). | ISPD2005, ISPD2015, ISPD2018/2019, ICCAD2015/2019, MMS, and custom LEF/DEF. | README claims about 3x faster per GP iteration than DREAMPlace. Absolute times not verified. | FALLBACK, built alongside DREAMPlace in the pilot. If the DREAMPlace CUDA 12 build fails, Xplace covers MMS, ICCAD2015 and the timing/routability problems, but not the fence elements. |
| [OpenROAD / OpenROAD-flow-scripts (RePlAce gpl, RTL-MP, GRT, OpenSTA)](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) | BSD-3-Clause for the scripts. Individual tools, PDKs and designs carry their own licenses. | From DG-RePlAce (verified): BlackParrot 827K instances (196 macros; the TILOS table lists 220 SRAMs) and MemPool Group 2,529K instances (326 macros). ChiPBench gives ariane133 as 168,551 cells and 132 macros. | DG-RePlAce on a TITAN V: GP 32 s (BlackParrot) and 122 s (MemPool Group), against DREAMPlace's 61 s and 178 s. The RePlAce CPU times (387 s / 1,896 s) and full-flow times are not re-verified. Global routing plus STA on 0.2-2.5M instances takes minutes to tens of minutes on CPU (not measured). | COMPLEMENT and G3 auditor only. On LEF/DEF designs (TILOS, ChiPBench, ISPD2015), run GRT + OpenSTA on sampled gold states to check that the HPWL/RUDY/WNS proxies track routed results. |
| [MaskRegulate / macro-regulator (LAMDA, NeurIPS'24) - added](https://github.com/lamda-bbo/macro-regulator) | BSD-3-Clause (verified via GitHub API). | ICCAD2015 superblue (8 designs); ISPD2005 in the appendix. | Not stated per step. The initial DREAMPlace placement 'takes within few minutes' (4x RTX 4090). | Closest precedent for turn-by-turn refinement of an existing placement. Adopt its grid-slot move encoding and its regularity metric. |
| [TILOS MacroPlacement testcases + flows](https://github.com/TILOS-AI-Institute/MacroPlacement) | BSD-3-Clause (verified). | Flops/macros: Ariane133 19,807/133; Ariane136 19,839/136; MemPool tile 18,278/20; MemPool group 360,724/324 (as listed); NVDLA 45,295/128; BlackParrot 214,441/220. By instance count (DG-RePlAce): BlackParrot 827K, MemPool Group 2,529K. | See the DREAMPlace and DG-RePlAce figures (DREAMPlace GP 61-178 s on a TITAN V for the two large designs). | USE as LEF/DEF base designs with real liberty files, for the timing and fence elements and for the OpenROAD audit. |
| [ChiPBench (MIRALab-USTC)](https://github.com/MIRALab-USTC/ChiPBench) | BSD-3-Clause (verified). | 20 designs with 332 to 859,382 cells and 0-136 macros. Corrected counts: ariane133 168,551 cells / 132 macros; ariane136 175,248 / 136; bp 301,030 / 24. The isa_npu and subrisc counts were not re-verified. | Per-design flow time is not reported. | USE for the G3 audit and as extra LEF/DEF bases. Keep only designs of 150K+ cells. |
| [BBOPlace-Bench (LAMDA)](https://github.com/lamda-bbo/BBOPlace-Bench) | The README says MIT, but the GitHub API detects no LICENSE file (spdx null). Treat it as unverified. | ISPD2005: 6 designs, 543-1,329 macros, 210K-1.1M cells. ICCAD2015: 8 designs, with the 512 largest modules taken as macros, 768K-1.9M cells. | Not stated. Budgets are 10,000 MP-HPWL calls and only 200 GP-HPWL calls because GP evaluations are expensive. | REFERENCE for problem encodings, the proxy/full evaluation split and the macro definition on ICCAD2015, and as a benchmark mirror. |
| [VeoPlace (VLM-guided placement, arXiv 2603.28733)](https://arxiv.org/abs/2603.28733) | Paper. Code release not confirmed. | 18 benchmarks: adaptec1-4 (58-127 macros under their definition), ibm01-04, ariane133/136, and 8 superblue designs (772K-1,938K objects, 99-1,629 macros). | 52-190 s per DREAMPlace rollout on one A100, depending on the design (the proposal's 52-83 s was the low end). 2,000 rollouts per run; 125 Gemini 2.5 Flash calls at about 230 s each. Gains of 1.3-4.3% over DREAMPlace 4.3 on 8/8 superblue designs. The spread across three seeds is about ±0.3-1.7% HPWL. | Precedent for the split where a language model picks macro regions and DREAMPlace places the rest. Its seed spread (±0.3-1.7%) is the right order of magnitude for our G2 noise floor. |
| [HeteroSTA (PKU) - added](https://heterosta.pkueda.org.cn/) | Closed source. Ships as a .so plus header. Free for academic and non-commercial use with email registration and a per-user license string. | Industrial-scale netlists (per the authors). | Not verified. | OPTIONAL accelerator only. The proposal rejected Circuit Training for its closed evaluator binary, and the same objection applies here, so OpenTimer is the default timer. |
| [Circuit Training / AlphaChip (Google)](https://github.com/google-research/circuit_training) | Apache-2.0 for the repo. The plc_wrapper_main proxy evaluator is a closed binary. | Corrected: Ariane has 133 hard macros and about 170K standard cells (ChiPBench: 168,551), clustered into soft macros. It is not 'millions of standard cells'. | Not stated. | REJECT as the simulator. Its split (agent places macros, analytical placer places cells) informs the decision layer. |
| [AutoDMP (NVlabs)](https://github.com/NVlabs/AutoDMP) | Apache-2.0 (verified). | NVDLA, Ariane and MemPool on NanGate45/ASAP7. | Not stated. | REFERENCE for knob ranges only. Do not build on it. |
| [Analog sizing: AnalogGym / AnalogGym-Opt (ngspice + SKY130)](https://github.com/CODA-Team/AnalogGym) | AnalogGym BSD-3-Clause; AnalogGym-Opt MIT (as proposed; not re-checked). | 30 topologies, each with roughly 10-40 sizing variables. | Not stated. | DROP. Far below DREAMPlace scale. |

**Sources:** https://github.com/limbo018/DREAMPlace · https://api.github.com/repos/limbo018/DREAMPlace/commits?per_page=10 · https://raw.githubusercontent.com/limbo018/DREAMPlace/master/README.md · https://raw.githubusercontent.com/limbo018/DREAMPlace/master/CMakeLists.txt · https://raw.githubusercontent.com/limbo018/DREAMPlace/master/Dockerfile · https://raw.githubusercontent.com/limbo018/DREAMPlace/master/dreamplace/params.json · https://raw.githubusercontent.com/limbo018/DREAMPlace/master/dreamplace/BasicPlace.py · https://raw.githubusercontent.com/limbo018/DREAMPlace/master/dreamplace/NonLinearPlace.py · https://raw.githubusercontent.com/limbo018/DREAMPlace/master/dreamplace/PlaceDB.py · https://raw.githubusercontent.com/limbo018/DREAMPlace/master/benchmarks/ispd2005_2015.py · https://api.github.com/repos/limbo018/DREAMPlace/contents/test · https://api.github.com/repos/limbo018/DREAMPlace/contents/test/mms · https://api.github.com/repos/limbo018/DREAMPlace/contents/test/ispd2015/lefdef · https://github.com/limbo018/DREAMPlace/issues/242 · https://github.com/limbo018/DREAMPlace/issues/212 · https://github.com/limbo018/DREAMPlace/issues/195 · https://github.com/limbo018/DREAMPlace/issues/152 · https://yibolin.com/publications/papers/PLACE_TCAD2020_Lin.pdf · https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/dreamplace4.pdf · https://yibolin.com/publications/papers/PLACE_ICCAD2020_Gu.pdf · http://www.ispd.cc/contests/15/web/benchmarks/ispd_2015_contest_benchmark_description.pdf · https://www.ispd.cc/contests/06/contest.html · http://www.ispd.cc/slides/2006/7-3.pdf · https://heterosta.pkueda.org.cn/ · https://arxiv.org/pdf/2511.11660 · https://arxiv.org/html/2603.28733 · https://arxiv.org/html/2510.23472 · https://github.com/lamda-bbo/BBOPlace-Bench · https://api.github.com/repos/lamda-bbo/BBOPlace-Bench · https://arxiv.org/pdf/2412.07167 · https://github.com/lamda-bbo/macro-regulator · https://arxiv.org/html/2404.13049 · https://arxiv.org/html/2407.15026 · https://api.github.com/repos/MIRALab-USTC/ChiPBench · https://github.com/TILOS-AI-Institute/MacroPlacement · https://github.com/cuhk-eda/Xplace · https://raw.githubusercontent.com/cuhk-eda/Xplace/main/README.md · https://api.github.com/repos/NVlabs/AutoDMP · https://api.github.com/repos/The-OpenROAD-Project/OpenROAD/commits?per_page=3 · https://api.github.com/repos/google-research/circuit_training/commits?per_page=3 · https://api.github.com/repos/kahypar/mt-kahypar · https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts · https://github.com/CODA-Team/AnalogGym

## power_grid

**Domain:** grid: large-scale AC optimal power flow with N-1 security (preventive security-constrained operation) on PGLib-OPF v23.07. Train on networks from 1k to 30k buses plus a small 78k-bus tier, the only tier at DREAMPlace-level variable counts. Held-out test: whole PGLib networks plus the GO Challenge 3 23,643-bus network.

**simulator.** Two-stage evaluator in one Julia runtime per spawned worker, driven from Python through juliacall.

State. The state is a small handle vector plus the problem id. evaluate() rebuilds the network data deterministically from the immutable per-problem base data and the handle vector. Do not clone a 10k-bus dict for each of roughly 100 candidates per turn.

Stage 1: base-case AC-OPF.
- Same model on both paths: an ExaModelsPower recipe, with the handles written into data fields (rate_a, pmax, bus shunt bs, branch shift), so the sparsity pattern never changes and nothing is rebuilt.
- Add a soft power-balance slack through the recipe's extension hook (dummy_extension(core, vars, cons) seen on main; not verified in v0.3.1), so evaluate() always returns metrics.
- Evaluation of record: Ipopt + HSL MA27 on CPU (trace cluster, Bridges-2 RM), tolerance 1e-6. It is deterministic single-threaded and solved 66 of 66 PGLib cases.
- GPU path, only for the 19k+ and 78k tiers: MadNLP + cuDSS (HybridKKT, or LiftedKKT at tolerance <= 1e-6) on Bridges-2 V100/H100. Every evaluation of one problem, including its reference, uses the same solver.
- Warm start always from that problem's fixed anchor solution (the initial state's OPF, stored in the problem bank), never from the parent state. replay_trajectory in dataxp/harvest.py re-applies all actions to a fresh state, evaluates only the final state and requires relative objective agreement within 1e-6, so path-dependent warm starts would fail G3.

Stage 2: post-contingency AC Newton power flow.
- lightsim2grid 1.1 ContingencyAnalysis on M = 100-300 monitored branch contingencies. Distributed slack, init_from_n_powerflow=True.
- nb_thread=1 when 128 workers share a node; more threads only on GPU nodes.
- Build the grid with init_from_powermodels from the same parsed data (no pandapower).
- Rating to current limit: I_lim = S_rating / (sqrt(3) x V_nom). Check both branch ends.
- Reactive limits are only detected, not enforced, so read compute_physical_violations and turn reactive-capability excess into a metric.

G3 checks:
- Base-case flows from lightsim2grid vs the OPF: max |dS| below 1% of rating.
- Spot-check 1% of contingencies against PowerModels compute_ac_pf.

**benchmarks.** PGLib-OPF v23.07: 45 networks with at least 1k buses x {typ, api, sad} regimes.

Training tiers:
(a) 1.3-5k buses, about 55%: 1354_pegase, 1888/1951/2848/2868_rte, 2000/2312/2742/3022/3970/4020/4601/4619/4837/4917_goc, 2383wp/2736sp/2737sop/2746wop/2746wp/3012wp/3120sp/3375wp_k, 2853/4661_sdet, 1803_snem, 2869_pegase.
(b) 5.6-10.5k, about 30%: 5658/7336/10192_epigrids, 6468/6470/6495/6515_rte, 8387/9241_pegase, 9591/10000/10480_goc.
(c) 19-30k, about 10%: 19402_goc, 20758_epigrids, 30000_goc.
(d) DREAMPlace-scale tier, about 5% (10-15 problems): 78484_epigrids, with 674.6k NLP variables and 1.04M constraints per OPF; 16.2 s on an A100, 198.5 s with CPU Ipopt.

Held out entirely, for test: 13659_pegase, 24464_goc and one rte network. Plus the GO C3 23,643-bus and 6,717-bus networks with their own contingency lists, which need a converter. Test episodes are cheap (about 20 evaluations each), so 78k-bus test problems are also affordable.

Scale claim, stated honestly: one inner solve in tiers (a)-(c) has 11k-209k variables, which is below DREAMPlace's ISPD2005 range (about 0.2-2.2M cells, so about 0.4-4.4M coordinates). Only tier (d) matches it per solve. The design problem itself, N-1-secure dispatch, has an extensive form of about 1.4e9 variables at 9241_pegase. Monolithic GPU SCOPF solvers stall at 500-617 buses: MadNCL needs about 3 min for 500 buses x 256 contingencies, and ExaModelsPower's GOC3 model runs out of memory at 617 buses. So the complexity exceeds DREAMPlace's, with the solver handling the bulk and the agent handling the part solvers cannot.

**decision layer.** The solver handles the bulk: every Pg, Qg, |V| and angle (about 4 x n_bus variables) comes from the base-case AC-OPF, which sees only base-case limits. The agent acts at the preventive-security layer, the way operators activate constraints and derate units.

One turn sets one handle (element i, param p, value v); the evaluator then re-solves the OPF and re-screens the monitored contingencies.

Handles: curate 24-40 per problem in the problem bank, from DC sensitivities to the initially violated contingency-branch pairs. Keep 24 or fewer, or add the hook below: the generic enumerate_candidates randomly subsamples 24 elements per turn (max_elements=24), so relevant handles would otherwise drop out at random.
(a) rate_frac on a branch or interface, in [0.5, 1.0] x rate_a, 10-14 handles. Tightens the base-case limit so the OPF pre-relieves the corridor.
(b) pmax_frac on a generator, in [0, 1], 6-10 handles. Derates units with the largest |PTDF| into the worst corridors.
(c) shunt_mvar at a weak bus, in [0, Qcap] with Qcap = 0.3-0.5 x local MVAr load, 4-8 handles.
(d) shift_deg in [-30, 30], only where the case has shifters: 66 in 9241_pegase and 16 in 6515_rte; none in the goc and epigrids cases checked. Runtime updates of the shift in lightsim2grid are not verified.
(e) Line switching: defer to phase 2. It changes the sparsity, forcing a model rebuild and a new symbolic factorization, and it is the main cause of islanding and divergence.

Move set. The default multiplicative factors (0.8, 0.9, 1.1, 1.25) work for (a) and (b) from 1.0, where the upward factors clamp to no-ops. They are stuck at 0 for (c). So add an optional Domain.candidate_moves(st) hook that returns 24-32 moves per turn: two moves (tighten and relax/undo) on the top 12-16 handles by |sensitivity| to the current worst violation, plus 4-8 random moves so the median candidate stays informative for G2. Additive steps for shunts are 0.25 x Qcap; for shifters, +/- 5 deg.

Also implement hint(): the handle with the largest linear sensitivity to the worst overload, which the greedy policy uses.

Each effect is observable: the targeted corridor's post-contingency loading, the overload total, cost and voltages all move. Expected G2 behaviour: the best tightening on a violated corridor moves the overload by tens of MVA and the cost by about 0.1-1%, while a typical non-binding move is a near no-op. top-vs-median separation is therefore physical, provided solver noise sits well below it. Tolerance 1e-4 GPU solves (0.3-4% objective bias) would not meet that.

**inner solver.** OPF: Ipopt + MA27 over the ExaModels model on CPU at tolerance 1e-6, the evaluation of record. MadNLP + cuDSS on GPU at tolerance <= 1e-6 for the >= 19k-bus and 78k tiers. One solver and tolerance per problem. Warm start from the fixed per-problem anchor. With data-only handle changes, model construction and symbolic analysis are skipped on every evaluation after the first in a worker.

Contingencies: lightsim2grid KLU Newton, M = 100-300 monitored outages, initialised from the base solution.

DC screening: lightsim2grid's sparse DC contingency analysis over all N-1, run at problem-bank build time and every 5 turns to refresh the monitored set. Never a dense LODF: 2.1 GB for 16k branches, about 10 GB for 35k, about 127 GB for 126k.

Reference (reference_class 'heuristic'): iterative preventive SCOPF. Add linearised post-contingency flow cuts for violated pairs to the base OPF, re-solve with the same solver, and repeat until AC N-1 on the monitored set is clean or 30 rounds pass. This re-implements the PowerModelsSecurityConstrained GOC1 approach.

**metrics objective constraints.** Objective: gen_cost in $/h, the PGLib quadratic cost of the OPF dispatch including the slack penalty, minimised and normalised by the reference cost.

Constraints, all flat floats:
- maximum_post_ctg_loading <= 1.0: max over monitored contingencies, branches and both ends of |S|/S_emerg. S_emerg = rate_c where 1.0 < rate_c/rate_a <= 1.5 (the epigrids cases); otherwise 1.15 x rate_a, because goc/rte have rate_c = rate_a and pegase has rate_c up to more than 3x rate_a.
- maximum_overload_mva <= 0: sum of post-contingency MVA above rating; gives a graded signal for G2.
- maximum_post_ctg_vm_violation_pu <= 0: worst excursion outside each bus's own [vmin, vmax].
- maximum_q_excess_mvar <= 1: post-contingency reactive-capability excess from lightsim2grid's physical-violation check. Needed because Q-limits are not enforced.
- maximum_n_ctg_diverged <= 0: contingencies whose power flow does not converge. Islanding contingencies are excluded from the monitored set at build time.
- maximum_unserved_mw <= 1: the OPF soft slack.
- maximum_base_loading <= 1.0.

Auxiliary: n_ctg_violating, opf_iters, opf_status, solve_s.

Validity envelope: OPF status LOCALLY_SOLVED or ACCEPTABLE; unserved_mw <= 5% of load; base loading <= 1.5.

**observation summary.** About 1.5-2.5k tokens, never raw arrays:
- Header: case, regime, buses/branches/gens, total load x stress, planned outages by branch id and zone.
- Base OPF: cost, status and iterations, binding thermal/voltage/angle limits, |V| range, unserved MW.
- Security: N-1 DC-screened count / AC-monitored M, number violating, maximum loading, total overload MVA, reactive-capability excess.
- Top-10 worst contingency to overloaded branch pairs: loading %, zone.
- Top-5 low/high-voltage buses after contingency.
- 8-12 zone aggregates: load, generation, headroom, net interchange. Use METIS or a spectral partition, because pegase and rte have a single bus area; goc and epigrids have 6-16 areas.
- Handle table: id, type, current value, bounds, and a coarse sensitivity sign/rank. Check hint_match_frac: if phi_select copies the top-sensitivity handle in more than about 80% of turns, coarsen this column so the data does not teach a trivial rule.
- Last 3 actions with metric deltas.

**problem generator.** Run make_problem(index) as a lookup into a precomputed PROBLEM BANK. Verified in dataxp/run_generation.py: make_problem runs serially in the parent process for all n problems before the worker pool starts, so an AC-reference acceptance loop inside it (minutes per draw, up to 64 retries) would serialise tens of hours.

Build the bank in a parallel SLURM array on the trace cluster. Each draw:
- network, from the training tiers;
- family (4 families, each >= 8% for G1): corridor-congested (api), angle-limited (sad), reactive/voltage-limited (typ with Q-load uplift and Qmax derates), outage-stressed (typ with k = 1-4 planned outages, with an islanding check);
- load: global stress in [1.0, 1.15] x per-load noise of +/- 15%;
- cost scaling and permutation of +/- 20%.

Acceptance tests, cheapest first:
1. The base OPF converges with unserved about 0.
2. A DC N-1 screen finds 2-8 distinct overloaded corridors. Calibrate this so roughly 20 single-handle turns can fix it; raw PGLib api cases can have far more.
3. A DC-SCOPF prefilter is feasible.
4. The AC reference reaches N-1 feasibility on the monitored set within 30 rounds.
5. Drop any monitored contingency that diverges at the reference.

Store in the spec: network id, regime, family, seeds (regenerate the vectors), outage list, monitored list, handle list, emergency-rating rule and reference cost. Store the anchor primal in a side .npz.

G1 sources of variance: about 38 training networks x 3 regimes x 4 families x stress x outages. About 10% may start feasible.

**cost budget.** Evaluations per problem, verified in dataxp/ladder.py and run_generation.py. The proposal's ~420 is wrong:
- phi_select scores EVERY enumerated candidate each turn (up to 160; with the handle set above about 50-100) for up to 20 turns, so 1,000-2,000 evaluations.
- blind scans candidates until the first feasible child.
- random and greedy add about 20 each.
- phi_d2 runs on the first 40 problems and adds 6 x 24 per turn.
- redteam repeats phi_select on the first 12.
- Average: about 1,500-3,000 evaluations per problem with the generic enumerator, or about 800-1,500 with a 24-32 move candidate_moves cap.

Seconds per evaluation, CPU path (Ipopt+MA27 OPF on an EPYC-7443-class core plus about 200 contingencies single-threaded plus about 0.3 s overhead; overhead unverified):
- 1.3-3k buses: about 1.5 s
- 4-5k: about 4 s
- 7-10.5k: about 8-14 s
- 19-30k: about 20-55 s
- 78k: about 210 s

GPU path (A100-class OPF, contingencies on host threads):
- 1-5k: about 1-2 s
- 9-14k: about 3-5 s
- 19-30k: about 6-15 s
- 78k: about 20-25 s

300 problems at about 2,000 evaluations each:
- CPU: 165 x 2,000 x 2 s, plus 90 x 2,000 x 10 s, plus 30 x 2,000 x 35 s, is about 1,270 core-hours, about 10 h wall on one 128-core node. RAM: 128 Julia workers x 2-4 GB.
- The 78k tier on GPU: 12 problems x about 1,200 evaluations (cap the candidates and skip d2/redteam) x about 22 s, roughly 90 GPU-hours, which is about 180 SU on H100.
- An all-GPU run would cost about 520 GPU-hours (about 1,040 H100 SU) and still needs CPU for contingencies. CPU is the primary path.
- Without HSL (MUMPS): 2-6x more CPU time.

Problem bank plus references: about 100-200 core-hours. Pilot before the full run, 72 problems at 8 turns: about 80 core-hours.

Ways to cut the cost:
- the candidate_moves cap (about 2x);
- skip phi_d2 and redteam on tiers (c) and (d);
- cache evaluations by handle-vector hash and drop clamped no-ops;
- reuse the anchor warm start and skip model construction and symbolic factorisation on data-only changes;
- refresh the monitored set only every 5 turns.

Unverified: the gain from IPM warm starts and lightsim2grid speed above 9.2k buses.

**risks.** SKEPTIC: Corrected, re-verified or added the following.
- The per-problem budget, using dataxp's actual ladder (about 1.5-3k evaluations, not 420).
- Solver anchors: switched from 2023 GV100 tolerance-1e-4 numbers to 2024 A100/cuDSS and EPYC Ipopt at tolerance 1e-6. That makes CPU Ipopt+MA27 on the trace cluster the primary path, with GPU only for 19k+ and 78k.
- Tolerance-1e-4 condensed GPU solves bias the objective by 0.3-4% and violate constraints by about 2e-3, so they are banned for Phi.
- Parent warm starts break replay_trajectory (G3), so a fixed anchor start is used instead.
- make_problem runs serially in the pipeline, so problems come from a parallel-built bank.
- lightsim2grid: load through init_from_powermodels, not pandapower; it needs a grid2op/LightSimBackend; Q-limits are not enforced; branch outages only. Benchmark hardware re-attributed.
- A dense full-N-1 LODF is infeasible at scale, so DC screening uses sparse solves.
- Data facts: sad is angle-limited, not voltage-limited. rate_c is unusable except in epigrids. goc/epigrids have no phase shifters, and pegase/rte have a single area. The set is 45 networks x 3 regimes, not 135. GO C3 network sizes corrected.
- Scale: only 78484 matches DREAMPlace per solve, so it moves into a small training tier; the N-1 layer is what exceeds monolithic solvers.
- Difficulty must be calibrated to 2-8 violated corridors so 20 single-handle turns can succeed.
- Line switching deferred.
Remaining risks:
1. Nonconvex AC-OPF: candidates can land in different local optima. Measure the noise floor on the pilot by re-solving each state from 3 perturbed starts; require top-minus-median Phi above 10x that floor in at least 80% of states, otherwise the deterministic-domain G2 test (gap > 1e-9) is meaningless.
2. Q-limit non-enforcement in lightsim2grid makes post-contingency voltages optimistic. The q_excess metric is a proxy. Cross-check against PowerModels with PV/PQ switching, or pypowsybl OpenLoadFlow (MATPOWER import not verified).
3. lightsim2grid is unbenchmarked at 20k-78k buses. ContingencyAnalysis also needs grid2op installed. Its wheels need glibc >= 2.27.
4. Licensing and hardware: HSL needs a free academic licence and a manual download. GPU determinism (cuDSS) is not verified, and replay needs agreement within 1e-6. RTX PRO 6000 FP64 is about 2 TFLOPS and its OPF speed is unverified. H100 costs 2 SU per GPU-hour.
5. Julia worker memory: 128 processes x 2-4 GB per node, more at 78k. Compile latency calls for a PackageCompiler sysimage.
6. ExaModelsPower data-only re-solve and the extension hook are on main and may postdate v0.3.1, so pin the commit.
7. PGLib has no contingency lists or credible emergency ratings, so the security definition is synthetic and must be documented. The GO C3 out-of-distribution test needs a GOC3-to-single-period converter.
8. If hint_match_frac exceeds 0.8, greedy suffices and the gold trajectories teach a trivial rule; coarsen the sensitivities shown.
9. Contract extension: the candidate_moves hook for additive and binary handles and for capping candidates.

**Candidates considered:**

| name | license | scale | runtime/eval | verdict |
|---|---|---|---|---|
| [PGLib-OPF v23.07 (benchmark data)](https://github.com/power-grid-lib/pglib-opf) | Data CC BY 4.0; the repo's scripts are MIT (LICENSE file checked). GitHub's SPDX field reads NOASSERTION. | Counted through the GitHub API: 66 base cases, each also in api/ (66) and sad/ (66). 45 base cases have at least 1,000 buses. That is 45 networks x 3 operating regimes, not 135 independent networks. From BASELINE.md: 9241_pegase has 9,241 buses and 16,049 branches; 30000_goc 30,000 and 35,393; 78484_epigrids 78,484 and 126,146. All typ/api/sad large cases have an AC solution in BASELINE.md. NLP size (Pacaud et al. 2024, Table 3): 1354_pegase 11.2k variables / 16.6k constraints; 9241_pegase 85.6k / 130.8k; 30000_goc 208.6k / 307.8k; 78484_epigrids 674.6k / 1.04M. Facts that matter for the design, read from the .m files: - rate_c equals rate_a on every branch of 2000_goc, 6515_rte, 10000_goc and 30000_goc, so there is no emergency rating. - In 9241_pegase, 5,347 of 16,049 branches have rate_b/rate_c different from rate_a, and 3,002 of those are more than 3x rate_a, which is implausible. - In 20758_epigrids, 24,807 of 33,368 branches have rate_c between 1.0 and 1.5x rate_a, which is usable. - Phase shifters: 66 in 9241_pegase, 16 in 6515_rte, 0 in 2000/10000/30000_goc and 20758_epigrids. - Bus areas: 9241_pegase and 6515_rte have 1, 10000_goc 6, 20758_epigrids 11, 30000_goc 16. - 9241_pegase: 1,445 gens; total Pd 312.4 GW in the typ variant. | BASELINE.md (PowerModels v0.19.9 + Ipopt with ma27 at default tolerance, dual 2.1 GHz Intel; JIT time excluded): 2000_goc 8 s, 9241_pegase 54 s, 10000_goc 55 s, 13659_pegase 68 s, 30000_goc 269 s, 78484_epigrids 1,039 s. These are stale anchors. Newer stacks at tolerance 1e-6 are 6-60x faster (see the ExaModelsPower row). | PRIMARY benchmark source. The 78484_epigrids network belongs in a small training stress tier, not test-only: it is the only network where one inner solve reaches DREAMPlace-level size (0.67M variables), and a GPU solve now takes 16 s, not 1,039 s. Hold out whole networks for test, because perturbations of a seen network leak. |
| [ExaModelsPower.jl + ExaModels.jl + MadNLP.jl (+ CUDSS.jl): one OPF model for CPU and GPU](https://github.com/madsuite-org/ExaModelsPower.jl) | MIT for all three packages (GitHub API, after following the redirect from exanauts/ to madsuite-org/). | Benchmarked on all 66 PGLib v23.07 base cases up to 78484_epigrids, which has 674.6k variables and 1.04M constraints (ExaModelsPower PSCC 2026 paper; Pacaud et al. 2024). Models available: AC-OPF (polar or rectangular), DC-OPF, multi-period OPF with storage, and a GOC3 SCOPF (goc3_model, which needs a unit-commitment solution as input). | Current anchors: Pacaud et al. 2024 (arXiv 2405.14236, Table 3), tolerance 1e-6, NVIDIA A100-80GB, MadNLP+HyKKT+cuDSS, IPM time excluding model construction: - 1354_pegase 0.56 s, 2000_goc 0.57 s, 4601_goc 1.40 s - 9241_pegase 2.76 s, 10000_goc 2.09 s, 13659_pegase 2.57 s - 19402_goc 4.23 s, 20758_epigrids 3.26 s - 30000_goc 10.46 s, 78484_epigrids 16.23 s Same table, CPU Ipopt+MA27 with ExaModels derivatives on an AMD EPYC 7443: 1354 0.47 s, 2000 0.84 s, 4601 3.81 s, 9241 7.68 s, 10000 9.17 s, 13659 9.14 s, 19402 34.56 s, 30000 50.43 s, 78484 198.54 s. Total runtime for 78484 including model construction (GV100, tolerance 1e-4): ExaModelsPower on GPU 16.1 s, ExaModelsPower on CPU 393.5 s, PowerModels on CPU 918.5 s. The paper notes that when the same model is re-solved with an unchanged sparsity pattern, construction and initialization can be skipped. The proposal's GV100 / tolerance 1e-4 / cuSOLVER figures are superseded. Its stated 5-15x GPU speedup was really 2.7-15x in that paper (24464_goc only 2.7x). | PRIMARY OPF model on both paths: the same ExaModels model solved by Ipopt+MA27 on CPU (the evaluation of record) and by MadNLP+cuDSS on GPU (only for the 19k+ and 78k tiers). Never use tolerance 1e-4 with LiftedKKT for Phi. Use tolerance <= 1e-6, and keep one solver and tolerance per problem, including its reference. |
| [PowerModels.jl (+ Ipopt)](https://github.com/lanl-ansi/PowerModels.jl) | LANL 3-clause BSD-style (LICENSE.md checked; SPDX NOASSERTION). | Any PGLib case. Formulations: AC (polar, rectangular, IV), DC, LPAC, SOC, QC, SDP. Problems: PF, OPF, OTS, TNEP. | Slower than ExaModels because of JuMP model construction and AD: 918.5 s vs 393.5 s on CPU for 78484. Ipopt itself is in BASELINE.md (see the PGLib row). | Parser and data model. Also: cross-check of the contingency power flow through compute_ac_pf for gate G3, and host of the reference iterative SCOPF. Not the per-evaluation OPF engine. |
| [lightsim2grid 1.1 (AC N-1 contingency analysis)](https://github.com/Grid2op/lightsim2grid) | MPL-2.0 | The docs benchmark grids go only up to case9241pegase. Nothing is published at 20k-78k buses. | v1.1.0 on an i7-13700H, case6515rte, 703 contingencies: 3,586 ms on 1 thread (196 power flows/s, about 5.1 ms each) and 1,072 ms on 6 threads (3.35x). Older v0.9.2 on an i7-6820HQ (2015 laptop): 20 ms per contingency at case9241pegase, and 36 ms vs 55 ms per power flow with and without factor recycling. Expect roughly 0.5-2 s per 200 contingencies at 6-14k buses. Timings at 30k and 78k are unverified. | PRIMARY post-contingency evaluator. Load the grid with init_from_powermodels from the same parsed data as the OPF (not through pandapower), and run the DC screen with its sparse DC solver, not a dense LODF. Because Q-limits are not enforced, report reactive-capability excess as an explicit constraint metric. |
| [Full AC-SCOPF solvers (MadNCL; ExaModelsPower goc3_model)](https://arxiv.org/abs/2510.13333) | MadNCL code: MIT (madsuite). goc3_model: MIT. | MadNCL solves a corrective AC-SCOPF with 500 buses and 256 contingencies in under 3 minutes on GPU; Knitro needs more than 3 hours. The ExaModelsPower GOC3 SCOPF benchmark (4 cases): only LiftedKKT on GPU solved all 4, at about 5,294 s shifted geometric mean, and every solver ran out of memory on the 617-bus cases. | Minutes to hours at 500-600 buses. Intractable at 9k-78k buses with thousands of contingencies. | Not a per-turn simulator. This is the evidence that the security layer (extensive form for 9241_pegase is about 85.6k x 16k, roughly 1.4e9 variables) is beyond monolithic solvers, which justifies having an agent act on it. |
| [ExaGO v2 (ORNL)](https://github.com/ORNL/ExaGO) | BSD-style (SPDX NOASSERTION). | OPFLOW, SCOPFLOW, TCOPFLOW, SOPFLOW. Prior GPU work used ACTIVSg up to 70k buses. | Not verified. | Do not depend on it. Optional cross-check only. |
| [PowerModelsSecurityConstrained.jl](https://github.com/lanl-ansi/PowerModelsSecurityConstrained.jl) | LANL (SPDX NOASSERTION). | GO Challenge 1 networks; sizes not re-verified. | Not verified. | Algorithm template for reference() only. Re-implement it over ExaModels rather than depending on it. |
| [ARPA-E GO Competition Challenge 3 datasets](https://data.openei.org/submissions/5997) | CC BY 4.0 | 591 public synthetic scenarios (3.6 GB) on 9 networks of 73, 617, 1,576, 2,000, 4,224, 6,049, 6,717, 8,316 and 23,643 buses. The proposal's '4200' should read 4,224, and 6,717 and 8,316 were missing. Event 4 had 669 scenarios. | The full GOC3 NLP is impractical (see the SCOPF row). | Held-out out-of-distribution test networks and realistic contingency lists. This needs a converter from GOC3 (fix the commitment, pick one period) to single-period MATPOWER-style data. That is unbudgeted engineering. |
| [grid2op / L2RPN](https://github.com/Grid2op/grid2op) | MPL-2.0 | Largest environments are 118 substations. | Milliseconds per step at 118 buses. | Not a benchmark: about 100x below the target scale. It is still a required dependency, because lightsim2grid's ContingencyAnalysis wrapper is built on a LightSimBackend. |
| [pandapower](https://github.com/e2nIEE/pandapower) | BSD-3-Clause (SPDX shows NOASSERTION). | Arbitrary. | Not needed per evaluation. | Fallback only. lightsim2grid now loads PowerModels/MATPOWER data directly, which avoids pandapower's transformer-model mismatch. |
| [gridfm-datakit](https://github.com/gridfm/gridfm-datakit) | Apache-2.0 | The abstract claims PF and OPF generation up to 10,000 buses. The proposal's 30k-bus PF figure and its 17 s per OPF (OPFData) figure were not confirmed. | Not re-verified. | Template for the Python-to-Julia bridge and the perturbation recipes. Not a simulator. |
| [MATPOWER 8.1 / PowerSimulations.jl / PyPSA](https://github.com/MATPOWER/matpower) | MATPOWER BSD-3 (case files excluded); PowerSimulations BSD-3; PyPSA MIT. | Not verified at 10k+ buses. | Not verified. | Cross-check only. |

**Sources:** https://github.com/power-grid-lib/pglib-opf · https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/BASELINE.md · https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/LICENSE · https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_case9241_pegase.m · https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_case20758_epigrids.m · https://api.github.com/repos/power-grid-lib/pglib-opf/contents/ · https://arxiv.org/abs/2307.16830 · https://arxiv.org/abs/2405.14236 · https://arxiv.org/abs/2510.12897 · https://arxiv.org/abs/2510.13333 · https://arxiv.org/abs/2410.17203 · https://github.com/madsuite-org/ExaModelsPower.jl · https://raw.githubusercontent.com/madsuite-org/ExaModelsPower.jl/main/src/opf.jl · https://github.com/madsuite-org/ExaModels.jl · https://github.com/madsuite-org/MadNLP.jl · https://github.com/exanauts/ExaPF.jl · https://github.com/lanl-ansi/PowerModels.jl · https://github.com/lanl-ansi/PowerModelsSecurityConstrained.jl · https://github.com/Grid2op/lightsim2grid · https://raw.githubusercontent.com/Grid2op/lightsim2grid/master/CHANGELOG.rst · https://raw.githubusercontent.com/Grid2op/lightsim2grid/master/docs/security_analysis.rst · https://raw.githubusercontent.com/Grid2op/lightsim2grid/master/lightsim2grid/contingencyAnalysis.py · https://lightsim2grid.readthedocs.io/en/v0.9.2/benchmarks_grid_sizes.html · https://lightsim2grid.readthedocs.io/en/latest/security_analysis.html · https://pypi.org/project/lightsim2grid/ · https://github.com/Grid2op/grid2op · https://github.com/ORNL/ExaGO · https://github.com/pnnl/ExaGO · https://github.com/spack/spack-packages/issues/6649 · https://data.openei.org/submissions/5997 · https://arxiv.org/abs/2512.14658 · https://arxiv.org/abs/2505.22825 · https://github.com/gridfm/gridfm-datakit · https://github.com/e2nIEE/pandapower · https://www.psc.edu/resources/bridges-2/user-guide/ · https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition · https://github.com/MATPOWER/matpower · C:/Users/xuwen/designbench_llm_finetune/dataxp/ladder.py · C:/Users/xuwen/designbench_llm_finetune/dataxp/run_generation.py · C:/Users/xuwen/designbench_llm_finetune/dataxp/domain.py · C:/Users/xuwen/designbench_llm_finetune/dataxp/harvest.py · C:/Users/xuwen/designbench_llm_finetune/dataxp/gates.py

## traffic

**Domain:** traffic: city-scale microscopic traffic simulation. The agent designs signal plans, corridor coordination and lane plans in a disrupted focus district while the simulator runs the whole city (MOSS on GPU, with SUMO/BeST as the validation tier and fallback)

**simulator.** MOSS python-moss 1.4.0, MIT, GPU microscopic (IDM plus randomized MOBIL).
Setup:
- Fork the repo and vendor the wheel in an Apptainer image.
- Use two conda envs: a build env with mosstool 1.3.21 and path4gmns (numpy>=2), and a sim env with python-moss (numpy<2).
- Run on Bridges-2 GPU-shared H100 through PTX JIT of the sm_70 build, or rebuild for sm_90/sm_120 with CUDA 12.8.
- Run one persistent Engine per problem process. Pass an explicit nonzero seed, because 0 means wall-clock time.
- Each problem uses one warm-up checkpoint.
WEEK-1 GO/NO-GO GATE. All of these must pass:
(a) BeST-converted Berlin and the mosstool beijing_map.pb both run 6:00-8:15 without segfaults, with less than 1% of vehicles stuck (v<0.1 for more than 600 s).
(b) Restoring and re-running the same design gives bit-identical metrics.
(c) Each evaluation takes at most 30 s on an H100.
(d) The G2 pilot in the risks section passes.
If the gate fails: switch the inner loop to SUMO 1.27.1 micro on BeST with --load-state on trace-cluster CPUs, using a reduced problem set.
Validation tier in all cases: SUMO 1.27.1 re-scores BeST-derived problems on the original network, with exact ID mapping through traci setProgramLogic.

**benchmarks.** CORRECTED. The four moss-benchmark cities cannot be downloaded (the archive bucket returns 404 and the issue is unanswered), and they were built for python-moss 0.3.3. Their published sizes (Beijing 11,953 junctions / 25,945 roads / 439,280 trips; Shanghai 6,270 / 14,837 / 436,888; Paris 6,588 / 14,411 / 202,664; New York 8,339 / 19,046 / 262,706) are now size targets only. Ask FIB-Lab to re-host the archive in parallel.
Core set:
(1) BeST Berlin (CC BY 4.0): 28,547 nodes, 71,651 edges, 2,249 signalised junctions, 2.25M car trips per 24 h with UE routes. Convert it with mosstool.map.sumo plus mosstool.trip.sumo.route. Conversion quality is unverified; check stuck vehicles, signal programs and lane connectivity.
(2) Beijing from the current-format map shipped with mosstool (81 MB pb), with demand from the gravity or AIGC OD generator and routes from mosstool.trip.gmns.STA.
(3) 4-8 OSM cities built with mosstool's Overpass builder. Target at least 5k junctions and at least 200k trips over the 3-h peak.
Against DREAMPlace: BeST has 2.25M trips and 71.7k edges, and the self-built cities have 0.2-0.44M agents. Both are in the object-count range of ISPD2005 placement (about 0.2-2.2M cells). The agent's decision layer is a district, analogous to placing macros or region constraints while a solver places millions of cells.

**decision layer.** The whole city is simulated on every evaluation. The agent edits a focus district of K = 24-64 elements next to a disruption, and each turn's candidates are drawn from the top 8-12 elements by pressure or queue.
(a) Junctions (16-48):
- cycle_s in [40,180], scale moves
- green_main_frac in [0.2,0.8], scale moves; the main/cross phase groups are derived from get_junction_phase_lanes and road headings
- offset_frac in [0,1), CIRCULAR ADDITIVE moves of ±0.1 or ±0.25
- adaptive in {0,1}, a toggle that hands the junction to MAX_PRESSURE; get() returns None for the timing parameters while adaptive=1
Implementation:
- Set the policy to MANUAL and run a Python fixed-time controller that calls set_tl_phase_batch only at switch instants, with next_step(n) between events.
- Step explicitly through the fixed_program's yellow phases, because set_tl_phase switches immediately with no yellow.
- FIXED_TIME cannot be used for the focus junctions: it does use per-phase durations from the map, but it has no offset and no runtime split API.
(b) Corridor macro-elements (2-6, recommended for G2):
- progression_speed_mps in [6,20]
- cycle_s, shared
- direction_bias in [0,1]
A green-wave rule maps these to member offsets. One move changes several junctions coherently, which gives effects that are large enough to observe.
(c) Roads (8-16):
- lane_plan: categorical, ONLY on roads whose map has next_road_lane_plans (check get_road_lane_plans; most OSM builds need them generated)
- speed_limit_mps in [8,25], scale moves
- lane_closed in {0,1}: MOSS restriction is a boolean closure, so 'bus/tidal' is not available; never close the last open lane
CONTRACT CHANGES REQUIRED in dataxp:
- A Domain move hook (for example moves(st, i, p) -> values). enumerate_candidates in C:/Users/xuwen/designbench_llm_finetune/dataxp/domain.py only multiplies by factors and clips, so it cannot move a categorical value, a zero, or a circular value.
- ladder.py calls enumerate_candidates with its defaults (max_candidates=160, max_elements=24). The cap of about 20 must be passed explicitly.
One turn is one action, for example SCALE(j7, cycle_s, 1.25), SHIFT(j7, offset_frac, +0.25), TOGGLE(j12, adaptive), SET(r3, lane_plan, 2) or SET(c1, progression_speed_mps, 12).

**inner solver.** Non-focus signals run the map fixed_program under FIXED_TIME, which uses the map's per-phase durations, or MAX_PRESSURE. The choice is made per city so that the baseline is congested but not gridlocked.
Demand: persons with PRE-COMPUTED routes, because MOSS 1.x has no in-engine router. BeST UE routes are imported. Self-built cities get static UE routes from mosstool.trip.gmns.STA (path4gmns) on trace-cluster CPUs.
Demand scale: generate persons once at the maximum scale of 1.4, then disable a seeded subset with set_person_enable before warm-up to reach 1.0-1.4.
Disruptions are unanticipated incidents at 7:00, so routes stay fixed. An optional Python re-route hook uses an igraph shortest path plus set_vehicle_route for persons whose remaining route crosses a fully blocked link.
Warm-up from 6:00 to 7:00 runs once per problem, followed by make_checkpoint, which takes a full host-RAM copy of the arena including per-person RNG state.
Each evaluation:
1. restore_checkpoint
2. apply the design
3. reset the Python controller
4. step to 8:15 (4,500 steps), sampling metrics every 60 s with one fetch_persons per sample
Common random numbers come for free, since the RNG is restored. Replicates for evaluate(st, seed) come from a seeded 2% set_person_enable thinning of persons departing after 7:00. The engine seed is fixed at construction and cannot be varied per evaluation.

**metrics objective constraints.** CORRECTED. Objective:
- focus_user_tt_veh_h (minimise): the sum of traveling_time, at 8:15, over persons departing 7:00-8:00 whose pre-computed route crosses at least one focus road, divided by 3600. Running persons count with their elapsed time. It needs one fetch_persons per evaluation and cannot be gamed by pushing queues upstream.
The proposal's focus_delay_veh_h (road waiting counts × 60 s, over focus roads plus a 1-hop upstream buffer) is kept as a reported metric only.
Constraints:
- maximum_city_att_ratio ≤ 1.01: departed ATT against the baseline. This is a weak guard: switching ALL Beijing junctions from FixedTime to MaxPressure moved ATT only 5.4% (4843 to 4580 s).
- maximum_spillback_frac ≤ 0.05: the share of focus and buffer lanes whose waiting count reaches 0.9 × length/7.5 m at any 60 s sample.
- minimum_focus_throughput_ratio ≥ 1.0: focus-route persons finished by 8:15, relative to the reference design. This REPLACES the city-wide completion ≥ 0.9 constraint, which cannot be met: benchmark 3-h throughput was 60k-145k of 203k-439k trips.
- maximum_max_queue_veh on any focus approach.
- maximum_changed_elements.
- Optional maximum_co2_kg: the sum of the cum_co2 field from fetch_persons; units unverified.
Validity envelope: maximum_stuck_frac (v<0.1 for more than 600 s) at most 0.01, which catches deadlock artefacts.
All metrics are flat floats.

**observation summary.** describe() never dumps the network. About 2-3k tokens:
- Header: city name with its actual converted junction and road counts, window 7:00-8:15, demand scale, and the disruption (e.g. 'EB lanes 2-3 closed on road 18822 at 7:00').
- Goal table: value, limit and gap for each metric.
- District table, about 8 rows: vehicles, mean speed, waiting vehicles, focus-route user travel time, each with Δ against the reference.
- Top-10 focus junctions: queue per phase group (lane waiting counts mapped through get_junction_phase_lanes), pressure, and current cycle, split, offset and adaptive.
- Corridor table: progression speed, achieved arrivals on green.
- Top-10 spillback lanes and roads: occupancy, waiting, lane_plan (or 'n/a' where the road has no plans), speed limit.
- A compact K-row parameter table.
- The last 3 actions with metric deltas.

**problem generator.** make_problem(index) cross-products:
- City: BeST, Beijing, and 4-8 OSM cities.
- Focus district: chosen by index among the top 10-20 baseline bottleneck clusters per city. Keep only clusters that are internally bottlenecked: sustained internal waiting with entry links not blocked by external spillback. K = the highest-pressure junctions and roads in the cluster, plus 2-6 corridors.
- Demand scale 1.0-1.4, set with set_person_enable on one maximum-scale person file per city.
- Disruption: partial arterial lane closure; event surge (event trips prebuilt and disabled, then enabled per problem); bridge capacity drop through speed limits; or a signal outage (policy NONE).
- Engine seed.
This gives well over 1,000 combinations.
Goal thresholds sit between the initial design (map fixed-time plus disruption) and a reference heuristic (Webster splits from the observed flows plus MAX_PRESSURE on the focus junctions; reference_class 'heuristic').
Discard a problem if any of these holds:
- the baseline is already feasible;
- the reference is infeasible;
- the G2 pilot on the first turn fails (top minus median Phi below 2× the chaos floor).
The yield is unknown, so generate 2-3x the 300 target.
G1 report: variance across problems of the initial violation, the objective/reference ratio, and the binding constraint.

**cost budget.** Per evaluation (unmeasured): restore 0.2-1 s, plus 4,500 steps at about 2-6 ms each (the floor is about 1 ms per step from the issue #7 log), plus 75 samples. That is about 10-30 s on an H100.
Setup per problem: engine build plus 1 h of warm-up, about 3-6 GPU-min. STA routing: about 0.5-3 core-h per city and demand file on the trace cluster, unmeasured.
CORRECTION on evaluation counts: phi_select as coded scores up to 160 candidates per turn (enumerate_candidates defaults: max_candidates=160, max_elements=24). If stochastic=True it uses at least 4 replicates each plus refinements. Over 20 turns:
- uncapped, deterministic: 3,200+ evaluations, 9-27 GPU-h per problem, 2.7k-8k GPU-h for 300 problems;
- uncapped, stochastic: more than 12,800 evaluations, which is far beyond budget.
Recommended: stochastic=False for phi_select, with the cap set to 20, giving about 410 evaluations or 1.1-3.4 GPU-h per problem.
Cuts:
(1) Pre-screen all 20 candidates with 900 steps and run the full 4,500 only for the top 4. That is about 8 evaluation-equivalents per turn, or about 170 per problem (0.5-1.4 GPU-h). Check the screen-versus-full rank correlation in the pilot.
(2) Pack 2-3 engine processes per H100. Memory and throughput scaling are unmeasured; the arena is unified memory and checkpoints live in host RAM, which is limited on GPU-shared.
(3) Run candidates only from the top-k congested elements.
Totals:
- with screening and packing: about 100-350 GPU-h for 300 problems, including about 15% for seed audits and the G2 pilot and about 15-30 GPU-h of setup;
- capped but without screening: about 350-1,000 GPU-h.
SUMO validation: 30 BeST problems × 2 designs × 3 seeds × about 1-2 core-h, roughly 0.2-0.4k core-h. The proposal's 1-2k core-h was an overestimate.
SUMO fallback inner loop: about 200-400 core-h per problem, roughly 60k-120k core-h for 300 problems, so it only works with a reduced set.

**risks.** SKEPTIC: MOSS stays primary but now depends on a week-1 go/no-go gate.
- Benchmark data: the moss-benchmark archive is gone (404 NoSuchBucket on 2026-10-01; issue #2 unanswered), and it targeted python-moss 0.3.3, which has no checkpoints. The core cities are now BeST Berlin converted through mosstool, mosstool's shipped Beijing map, and OSM-built cities.
- Speed anchor: 42.81 s is on the synthetic Roadnet-L with pre-1.0 MOSS, not on a city.
- Routing: MOSS 1.x requires pre-computed routes; there is no A* at departure. Use STA or BeST UE routes.
- Signals: FIXED_TIME does use the map's per-phase durations but has no offsets and no runtime split API. set_tl_phase skips yellow.
- Seeding: the RNG is restored by checkpoint, so evaluate(st, seed) cannot vary the engine seed. Replicates now use seeded set_person_enable thinning, and phi_select runs deterministic.
- Lanes: restriction is a boolean closure (bus/tidal dropped), and lane plans exist only on roads built with them.
- Install: python-moss (numpy<2) and mosstool (numpy>=2) need separate envs. Wheels are sm_70 with PTX JIT on H100 and Blackwell.
- Contract: the generic enumerator cannot move categorical, zero or circular parameters (a move hook is needed), and its default of 160 candidates per turn is 8x the assumed 400 evaluations per problem. The cap must be passed explicitly.
- Metrics: city-wide completion ≥ 0.9 cannot be met under benchmark congestion and was replaced by focus throughput against the reference. The objective changed to route-based district user travel time.
- Decision layer: added corridor macro-elements.
- Candidates: added UXsim as a CPU fallback. Corrected the LPSim URL (fork) and LibSignal's license (none). Cut the SUMO validation estimate to 0.2-0.4k core-h.
Risk list:
1. DATA. BeST-to-MOSS conversion and OSM builds are unverified. Watch for stuck vehicles, missing signal programs, lane-connectivity errors and junction-overlap deadlocks. There is an open, unanswered segfault report on 1.4.0 with self-built data (moss #7). Mitigation: the week-1 gate; request the benchmark data from FIB-Lab; fork and debug.
2. MAINTENANCE. No release since 2025-04-30 and no maintainer replies since mid-2025. Pin the version, vendor the wheel, keep the Apptainer image, and budget for C++ fixes.
3. GPU. Only sm_70 binaries are shipped. PTX JIT on sm_90 and sm_120 adds first-load compile time and is untested on Blackwell. Rebuild with CUDA 12.8 if needed. Pin one GPU type per problem, because floating-point results may differ across architectures.
4. NOISE AND G2. Determinism is not the same as a noise-free ranking: chaotic divergence from small changes sets a noise floor. Pilot on 10 problems:
(i) same design × 3 runs → bit-identical;
(ii) a null move (speed limit on a far-away road) → Δ gives the chaos floor;
(iii) 3 engine seeds → Spearman ≥ 0.7 on the candidate ranking;
(iv) top minus median Phi > 2× the floor in at least 60% of turns.
If (iv) fails, use bigger moves (cycle ×0.7/×1.4) or corridor macro-elements.
5. STATIC ROUTES. There is no within-evaluation route choice, so Braess effects are missing and full closures strand vehicles. Use partial closures, the re-route hook and the stuck-fraction envelope. Use UXsim or SUMO rerouting only to check route-choice sensitivity.
6. CONGESTION REGIME. Benchmark-like demand is gridlock-heavy: 3-h throughput was only about 27-33% of trips, and ATT 4.0-4.8k s. Choose a demand scale and districts where the bottleneck is internal; otherwise every candidate ties.
7. GUARD WEAKNESS. City ATT barely responds to district changes. The route-based objective and the buffer spillback constraint prevent harm being shifted upstream.
8. CONTRACT FIT. Without the move hook and the candidate cap, the parameters are inert or the budget grows 8x. Edits needed in C:/Users/xuwen/designbench_llm_finetune/dataxp/domain.py and dataxp/ladder.py.
9. OVERHEAD. Every metric getter calls fetch_persons, which copies data for all persons; sample every 60 s with one fetch per sample. The MANUAL controller should use next_step(n) between switch events.
10. MEMORY. Unified-memory arena plus host-RAM checkpoint copies; per-engine size and how many engines can share a GPU are unmeasured.
11. FIDELITY. IDM/MOBIL has only been validated on Shenzhen speeds. Cross-check BeST problems in SUMO: compare the improvement from initial to gold (Kendall τ against MOSS) over 3 seeds.
12. LICENSES. OSM-derived maps are under ODbL; BeST requires CC BY 4.0 attribution; LibSignal has no license (do not vendor it).
13. RUNTIME. All per-evaluation times for 1.x, H100, Blackwell and the peak SUMO window are estimates until the gate measures them.

**Candidates considered:**

| name | license | scale | runtime/eval | verdict |
|---|---|---|---|---|
| [MOSS (python-moss 1.4.0) + mosstool 1.3.21; moss-benchmark data no longer downloadable (Tsinghua FIB-Lab)](https://github.com/tsinghua-fib-lab/moss) | Code is MIT: python-moss on PyPI and GitHub, and moss-benchmark. mosstool is MIT according to its PyPI classifier, but GitHub detects no license file. Maps built from OpenStreetMap carry ODbL share-alike obligations for derived databases. | Paper cities (arXiv 2406.10661), as roads / junctions / trips over 7-10 AM, each with normal, smooth and congested variants: - Beijing: 25,945 / 11,953 / 439,280 - Shanghai: 14,837 / 6,270 / 436,888 - Paris: 14,411 / 6,588 / 202,664 - New York: 19,046 / 8,339 / 262,706 CORRECTIONS: (1) The 2,464,950-vehicle speed test used Roadnet-L, a synthetic network with 93,564 roads and 26,479 junctions. It was not a benchmark city. (2) The benchmark data archive cannot be downloaded. The archive at fiblab-neurips2024-moss-benchmark.obs.cn-north-4.myhuaweicloud.com/data.tar.gz returned HTTP 404 NoSuchBucket on 2026-10-01. moss-benchmark issue #2 (2026-06-30, 'download link ... broken') has no reply. The repo's map.bin and agents.bin files are 42-43-byte stubs. (3) The benchmark code pins python-moss==0.3.3 and mosstool==0.2.10. That is the pre-1.0 API (Engine(agent_file=...)), and it has no checkpoints, which arrived in v1.1. What can be used today: - An 81 MB current-format Beijing map in the mosstool repo (data/gravitygenerator/beijing_map.pb; junction count unverified) with a gravity-OD example. - mosstool's OSM builder (Overpass client added in 1.3.21). - mosstool's SUMO map and route converters. Scale against DREAMPlace: a 6k-12k-junction city with 0.2-0.44M agents, or BeST at 2.25M trips, is in the same object-count range as ISPD2005 placement (about 0.2-2.2M cells). | Measured anchor: Roadnet-L, 3600 steps in 42.81 s (benchmark paper) and 37.70 s (system paper), on an RTX 4090 with a Xeon 8462Y running pre-1.0 python-moss. Fixed overhead: about 1 ms per step. The issue #7 log shows about 1,000 Hz on a 190-person map, so the floor is about 4.5 s for a 75-minute window. No published timing exists for 1.x, for any benchmark city, or for H100 or Blackwell. Estimate (unmeasured): 2-6 ms per step for a 6k-12k-junction city or BeST at peak, which gives about 10-30 s per evaluation of 4,500 steps. This includes restore (0.2-1 s) and 75 metric samples. | PRIMARY, but CONDITIONAL on a one-week go/no-go gate. It is still the only GPU microscopic simulator at city scale that combines signal and lane APIs with in-memory checkpoints. It is not usable as described: - the benchmark cities must be rebuilt; - routes must be pre-computed; - the maintainers are not answering issues: moss #6 (2025-09), moss #7 (2025-11, segfault at step 3-4 on 1.4.0 with self-built data) and benchmark #2 have no replies. Fork it, vendor the wheel and be prepared to patch the C++. |
| [Eclipse SUMO 1.27.1 (micro) with the BeST Berlin scenario](https://github.com/mosaic-addons/best-scenario) | SUMO is EPL-2.0 OR GPL-2.0-or-later. The BeST README states CC BY 4.0; GitHub detects the license as NOASSERTION. | 800 km² with 28,547 nodes and 71,651 edges, including 2,249 signalised junctions. 2,246,917 car trips and 25,724 bus trips over 24 h, with an average trip of 817 s and 7.9 km. Routes were recalibrated to user equilibrium, and counts were validated against Berlin detectors. Built for SUMO 1.22.0. The download (best-scenario-v2.zip, 461 MB) was live on 2026-10-01; downloads are counted for statistics. | Verified: the full 24 h run takes 7 h on one 3.4 GHz core. Unmeasured: a peak window. A 6:00-8:15 micro run is estimated at 1-2 core-h because peak load is above the daily average. Meso mode is documented as 'up to 100x faster', but it ignores signals by default (--meso-junction-control false), models them in simplified form when enabled, and has no lane-level output. It cannot rank splits or offsets. | VALIDATION TIER, FALLBACK INNER LOOP, and the source of the core city. Convert BeST to MOSS with mosstool.map.sumo and mosstool.trip.sumo.route; the route converter imports routes, so the UE routes carry over. Because the network is the same, SUMO re-scores can map signal programs exactly through traci setProgramLogic. Conversion quality is unverified. If MOSS fails the gate, run SUMO micro on full BeST from a saved state, with 20 candidates in parallel on trace-cluster cores. That costs roughly 200-400 core-h per problem, so it is affordable only for a reduced problem set. |
| [Other public SUMO city scenarios: TAPASCologne, TuST Turin, LuST Luxembourg, MoST Monaco](https://sumo.dlr.de/docs/Data/Scenarios.html) | TAPASCologne has a non-commercial CC clause. TuST is GPL-3.0. LuST is MIT. MoST is GPL-3.0. | TAPASCologne: 400 km², about 1.2M trips/24 h, shipped at scale 0.3. TuST: 602.61 km², 32,936 nodes, 66,296 edges, 856 traffic lights, 2,202,814 departed vehicles. Meso only. LuST: 155.95 km², 203 traffic lights. MoST: 46,842 vehicles. | MoST: real-time factor 10.566 on SUMO 1.14. Not verified for the others. | Extra networks and realism cross-checks only. They are stale: TuST was last updated 2021-04 on SUMO 1.2.0, MoST 2022-09 on 1.14.0, and LuST 2023-10. Their network quality forced meso mode (TuST) or 0.3 scaling (Cologne), so do not use them as primary cities. |
| [CityFlow](https://github.com/cityflow-project/CityFlow) | Apache-2.0 | Grids up to 30x30. MPLight used a Manhattan network with 2,510 signals. | Roadnet-L (2.46M vehicles): 3806.7 s per 3600 steps (MOSS paper), about 89x slower than MOSS. | Reject. The code is unmaintained, it is far too slow at city scale, and its lane changing is simplistic. |
| [CBLab / CBEngine](https://github.com/CaradryanLiang/CityBrainLab) | MIT | Paper: 10k intersections and 100k vehicles at a 1:4 time ratio. The README claims 1M vehicles over 100k intersections at 1:1. | Only the ratios above are published. | Reject. The code has been dormant since 2023, the lane-change model is private, and it is CPU-bound. |
| [LibSignal](https://github.com/DaRL-LibSignal/LibSignal) | GitHub detects no license file, so legally all rights are reserved. Read the code; do not vendor it. | 9 datasets, the largest being Manhattan with 196 intersections. | Depends on the backend. | Not a simulator. Useful only for converters and RL baselines. |
| [MATSim Open Berlin v6.x](https://github.com/matsim-scenarios/matsim-berlin) | Code is GPL-2.0. Data is CC BY 4.0. | Synthetic population samples at 1%, 3% and 10%. | More than 4 h per iteration at 10% (arXiv 2311.14118). Calibration needs hundreds of iterations. | Reject for the inner loop: far too slow for hundreds of evaluations per problem, and signals are an add-on. |
| [LPSim (multi-GPU)](https://github.com/Xuan-1998/LPSim) | The LICENSE file is MIT; GitHub reports NOASSERTION. The proposal's URL (jayzou3773/LPSim) is a fork of Xuan-1998/LPSim. | SF Bay Area: 223K+ nodes and 540K+ edges. 2.82M trips on 1 GPU, 9.01M on 2 GPUs. | 2.82M trips in 6.28 min on 1 GPU (arXiv 2406.08496). | Reject for signal design. Signals are undocumented, and a GitHub code search of the fork returned no 'signal' hits. That is weak evidence, but nothing contradicts it. It remains a possible backend for routing or pricing variants. |
| [UXsim (added by the skeptic)](https://github.com/toruseo/UXsim) | MIT | README: 'over 60000 vehicles in a city in 30 seconds, or even 1 million vehicles in a metropolitan area in 40 seconds depending on the simulation setting'. Vehicles move in platoons, with deltan=5 by default. Performance on a real 10k-node OSM network is unverified. | Only the README figures above. Not measured on any network of interest. | CPU FALLBACK and route-choice tier only. Signals are native: per-node group green times, signal_offset, and override_signal at runtime. It also has dynamic user-optimal routing, which MOSS lacks. It is link-level mesoscopic, not lane-level, so it does not meet the DREAMPlace-level fidelity bar on its own. |

**Sources:** https://arxiv.org/html/2406.10661 · https://arxiv.org/html/2405.12520 · https://github.com/tsinghua-fib-lab/moss · https://github.com/tsinghua-fib-lab/moss/blob/main/src/CMakeLists.txt · https://github.com/tsinghua-fib-lab/moss/blob/main/src/entity/junction/trafficlight/trafficlight.cu · https://github.com/tsinghua-fib-lab/moss/blob/main/src/mem/mem.cu · https://github.com/tsinghua-fib-lab/moss/blob/main/src/python_api.cu · https://github.com/tsinghua-fib-lab/moss/blob/main/src/moss.cu · https://github.com/tsinghua-fib-lab/moss/blob/main/src/entity/person/person.cu · https://github.com/tsinghua-fib-lab/moss/blob/main/src/entity/lane/lane.cu · https://github.com/tsinghua-fib-lab/moss/blob/main/python/src/moss/engine.py · https://github.com/tsinghua-fib-lab/moss/blob/main/python/src/moss/parallel.py · https://github.com/tsinghua-fib-lab/moss/blob/main/python/before-all-manylinux.sh · https://github.com/tsinghua-fib-lab/moss/commit/950104d · https://github.com/tsinghua-fib-lab/moss/issues/7 · https://github.com/tsinghua-fib-lab/moss/issues/6 · https://github.com/tsinghua-fib-lab/moss/issues/3 · https://pypi.org/project/python-moss/ · https://python-moss.readthedocs.io/en/latest/apidocs/moss/moss.engine.html · https://python-moss.readthedocs.io/en/latest/apidocs/moss/moss.parallel.html · https://python-moss.readthedocs.io/en/latest/07-version.html · https://github.com/tsinghua-fib-lab/moss-benchmark · https://github.com/tsinghua-fib-lab/moss-benchmark/blob/main/requirements.txt · https://github.com/tsinghua-fib-lab/moss-benchmark/blob/main/TSC/run.sh · https://github.com/tsinghua-fib-lab/moss-benchmark/blob/main/Perf/common.py · https://github.com/tsinghua-fib-lab/moss-benchmark/issues/2 · https://fiblab-neurips2024-moss-benchmark.obs.cn-north-4.myhuaweicloud.com/data.tar.gz · https://github.com/tsinghua-fib-lab/mosstool · https://pypi.org/project/mosstool/ · https://github.com/tsinghua-fib-lab/mosstool/blob/main/mosstool/trip/gmns/sta.py · https://github.com/tsinghua-fib-lab/mosstool/blob/main/mosstool/trip/route/preroute.py · https://github.com/tsinghua-fib-lab/mosstool/tree/main/data/gravitygenerator · https://moss.fiblab.net/ · https://pypi.org/project/eclipse-sumo/ · https://pypi.org/project/libsumo/ · https://ghcr.io/v2/eclipse-sumo/sumo/tags/list · https://sumo.dlr.de/docs/Simulation/Meso.html · https://sumo.dlr.de/docs/Simulation/SaveAndLoad.html · https://sumo.dlr.de/docs/Tools/tls.html · https://sumo.dlr.de/docs/Data/Scenarios.html · https://github.com/mosaic-addons/best-scenario · https://www.dcaiti.tu-berlin.de/research/simulation/downloads/get/best-scenario-v2.zip · https://github.com/marcorapelli/TuSTScenario · https://github.com/lcodeca/LuSTScenario · https://github.com/lcodeca/MoSTScenario · https://github.com/cityflow-project/CityFlow · https://github.com/CaradryanLiang/CityBrainLab · https://github.com/DaRL-LibSignal/LibSignal · https://github.com/matsim-scenarios/matsim-berlin · https://arxiv.org/pdf/2311.14118 · https://github.com/Xuan-1998/LPSim · https://github.com/jayzou3773/LPSim · https://arxiv.org/abs/2406.08496 · https://github.com/toruseo/UXsim · https://pypi.org/project/uxsim/ · https://arxiv.org/pdf/2406.02126 · C:/Users/xuwen/designbench_llm_finetune/dataxp/domain.py · C:/Users/xuwen/designbench_llm_finetune/dataxp/ladder.py · C:/Users/xuwen/designbench_llm_finetune/dataxp/domains/traffic.py

## thermal

**Domain:** thermal: full-chip 3D-IC / 2.5D thermal co-design (block floorplan swaps, tier assignment, DVFS, thermal-via budget and keep-outs, cooling). Evaluated with a fine-grid compact finite-volume thermal model (0.16M unknowns at 128^2 x 10 for training, 0.65-2.6M at 256^2/512^2 for evaluation). The agent works at the block, region and global level (about 10^2 elements). The bulk is solved underneath it: a per-cell thermal-via density field placed by an adjoint optimiser (1.6e4-2.6e5 design cells per tier), plus the temperature field. Optional Tier B reaches million-object design scale: Open3DBench memory-on-logic designs with 32K-1.08M placed cells, DREAMPlace as the inner placer. Data-centre and building thermal remain rejected, since Sinergym exposes 1-2 zones.

**simulator.** Inner loop for all training data (Tier A): an in-house vectorised 7-point finite-volume assembler (numpy) covering the layer stack, per-cell vertical conductivity from the via field, and heat-sink HTC/TIM. It is solved on CPU on the trace cluster:
- One sparse factorisation per distinct conductance matrix. Prefer CHOLMOD or Pardiso; scipy SuperLU measured 33 s.
- Batched back-solves (0.03-0.05 s each, measured) for every power-only candidate.
- Warm Jacobi/AMG-PCG (2-3 s, measured) for conductance-changing candidates.
- Leakage-temperature feedback as a 2-3 step fixed point on the RHS, so it reuses the same factorisation.
GPU is used only for 256^2/512^2 held-out evaluation, via AmgX, cuDSS (nvmath-python) or CUTh-Solver as a separate GPL process.
Reference: 3D-ICE 4.0 on the same .stk/.flp, with MaxAE <= 0.5 C on a calibration set required before generation (kept from the proposal).
Liquid-cooled variant: 3D-ICE microchannels at coarse grids (kept).
Tier B (optional, million-object design): Open3DBench MoL designs with DREAMPlace 4.x re-placing cells after each agent action, scored by the same thermal solver.

**benchmarks.** Stack template (verified from CUTh Sec. V-A): the 10-layer F2F memory-on-logic stack. Layers are logic bulk 10, FEOL 2, BEOL 10, hybrid bond 2, memory BEOL 10, memory FEOL 2, memory bulk 200, TIM 100, spreader 400, heat sink 1000 um. The CUTh benchmark applies HTC 1,000 / 10,000 / 8,000 W/m^2K for SPARC, EV6 and 9-core. Extend to 2-3 tier logic-on-logic and a 2.5D interposer family.
Instance sources:
- FloorSet-LITE rectangles (not Prime), CC BY 4.0.
- ATPlace_pub Case1-10 (6-61 chiplets) for 2.5D, after licence clarification.
- Open3DBench MoL designs for Tier B.
Power: block-type densities calibrated to HotGauge and Open3DBench OpenSTA. Intra-block textures from CircuitNet (2.25 um tiles, about 300x300 per layout; N28 10,242 and N14 10,345 layouts; repo BSD-3). Textures are MANDATORY, not optional: 3D-ICE 4.0 reaches RMSE < 0.3 K with 11,760 cells on a die-level case, and CUTh Table II accuracy is flat from 128^2 to 512^2 for block-level SPARC/EV6/9-core maps. Without sub-block power, the 256^2/512^2 grids add unknowns but no information.
Sanity anchors: reproduce CUTh SPARC/EV6/9-core at 128^2 x 10 against both 3D-ICE (MaxAE 0.02-0.20 C vs COMSOL) and our assembler.

**decision layer.** One turn makes one set() on one element. The bulk solvers own everything below it.
(a) Blocks (21-120 FloorSet-Lite rectangles over K = 2-3 tiers):
- slot: swap with a block of area ratio 0.8-1.25, or move into a whitespace hole at least as large. The legaliser only shifts locally (no global re-pack), so one action displaces one block.
- tier 0..K-1: allowed if the target tier has whitespace; otherwise a swap.
- dvfs 0-4: f = 0.6-1.0x; dynamic power roughly f^2.5-3; leakage proportional to V*exp(beta*(T-T0)).
(b) Via regions, an 8x8 grid per tier (64-192 elements):
- keepout in {0,1}: routing-critical areas.
- cap_pct 0-20%: local fill cap.
These are CONSTRAINTS for the planner, replacing the proposal's per-region ttsv_pct. Reason, from a local probe at equal Cu budget: an adjoint per-cell planner free over the whole die beat uniform-in-the-4-hottest-regions by 3.0 K (0.25% budget) and 6.4 K (1%). Restricted to those same 4 regions it gained only 0.1-0.6 K. A region-pct knob plus a proportional allocator is therefore nearly the same as region-uniform, and wastes the bulk layer.
(c) Global: via_budget_pct 0-3% in 0.25% steps (BEOL/bond area), HTC level 0-5, TIM 0-2.
Total: about 90-320 elements.
About 20 candidates per turn:
- From the top-5 hotspots: DVFS -1 on the owner; swap the owner with the 2-3 coolest compatible blocks; tier move; unstack (swap the hot vertical neighbour); via budget +0.25%; lift a keep-out or raise a cap near the hotspot; HTC +1; TIM upgrade.
- Repair moves when other constraints bind: DVFS +1 on cool blocks; swap toward the net centroid; add a keep-out in an overflowing region.
- 2-3 random distractors, so the G2 median is not inflated by an all-good candidate set.
G2 evidence (local probe, 128^2 x 10, 40+16 synthetic blocks, 17 candidates): best -13.6 K (HTC +25%), median -2.1 K, worst +11.1 K. DVFS on the owner gave -10.1 K but only -0.5 to -0.9 K on the next three hottest blocks. Vias at the hotspot gave -9.6 K, about 0 elsewhere. Swaps ranged from -9.8 to +11.1 K. Outcomes are separable and depend on reading the observation correctly. Absolute rises were uncalibrated (182 K, from a 1 cm^2 sink at h = 2e4), so only relative spreads count.

**inner solver.** evaluate(state):
1) Rasterise power (block maps x CircuitNet textures x DVFS scale, plus leakage) onto 128^2 x L.
2) Via planner. Runs if the via budget, keep-outs, caps or block positions changed since the parent; otherwise the parent's field is reused (the field is part of the state, like DREAMPlace cell positions). It does 3 damped iterations of:
- J = log-sum-exp of active-layer T (beta about 2/K);
- adjoint lambda from the same SPD matrix;
- per-cell sensitivity -lambda^T (dK/drho) T;
- water-fill the budget under caps and keep-outs.
Cost: about 7 solves, about 20 s single-thread at 128^2 (measured 2.5-3.2 s per PCG solve). This is the bulk design layer: 16,384 cells per tier at 128^2, up to 262,144 at 512^2.
3) Assemble K(rho, HTC, TIM) and cache its factorisation under a hash of (rho, HTC, TIM).
4) Solve with the leakage fixed point (2-3 RHS updates on the cached factorisation).
5) Compute metrics; add RUDY congestion from FloorSet nets plus via blockage on a 32x32 grid.
phi_select screen-then-verify: score all ~20 candidates with rho frozen. Power-only candidates are back-solves; matrix-changing ones are warm PCG. Re-evaluate the top 3 with a full re-plan and commit the argmax of the verified set. Log screening-vs-verified rank agreement as a gate metric.
The proposal's per-block Green's functions are DROPPED. All power-only moves (DVFS, swap, tier, unstack) share one matrix, so they batch as multi-RHS back-solves. Precomputing up to 120 Green's responses per geometry costs more than the ~5 DVFS candidates they would replace.

**metrics objective constraints.** Objective, two families for G1:
- F1: minimum_perf_loss_pct (throughput-weighted frequency loss vs all-nominal, %), with constraint maximum_cooling_power_W.
- F2: minimum_cooling_power_W (fan/pump power from the HTC curve plus a TIM cost equivalent), with constraint maximum_perf_loss_pct.
Every lever must be costed. In the probe, HTC +25% was the single best candidate, so a free HTC in F1 (as proposed) would collapse gold traces to 'raise HTC'.
Constraints:
- maximum_peak_temp: 85-105 C over active layers, leakage included.
- maximum_thermal_gradient: 8-25 K (max-min) on any active tier.
- maximum_via_area_pct: 1-3% of tier area in BEOL/bond.
- maximum_wirelength_ratio: 1.05-1.20 vs the initial design; block-to-block plus pin-to-block HPWL in mm, with an inter-tier crossing penalty.
- maximum_routing_overflow: 1.0-1.1, as RUDY demand plus via blockage over per-region capacity. This is what makes keep-outs a real trade-off.
Diagnostics (not in Phi): hotspot_area_pct, per-tier peaks, total and leakage power (W), via cell count, overlap (0 after legalisation), leakage fixed-point residual. Units: C, K, %, W, mm.

**observation summary.** About 2.5-3.5K tokens, never raw fields:
- Header: family, die size, tiers and layers, grid (e.g. 128x128x12 = 196,608 cells), HTC level, TIM, total and leakage power.
- Constraint table: value / limit / gap / status for peak, gradient, via %, wirelength ratio, routing overflow, and the perf or cooling side-constraint; plus the objective.
- Per tier: an 8x8 map of region max temperature (integer C), an 8x8 via-density map (per-mille), keep-out flags, and the top-5 overflow regions.
- Top-8 hotspots: tier, region, T, owning block, type, W/mm^2, DVFS level, vertical neighbour and its power, local via %.
- Top-12 blocks by power density: position, tier, DVFS, area, compatible swap partners.
- Whitespace holes per tier.
- Delta from the last action: peak, gradient, objective, overflow, whether the hotspot moved, via cells re-planned.

**problem generator.** make_problem(index), seeded by index:
1) Family: MoL 2-tier, LoL 2- or 3-tier, or 2.5D. 2.5D draws from the 10 ATPlace_pub cases with jittered power and limits, so treat it as a minority family.
2) Take FloorSet-LITE sample #index (rectangles, < 5% whitespace). Min-cut it across tiers. Re-pack each tier once with a sequence-pair packer seeded from the 2D relative positions, to 80-90% utilisation. This leaves 10-20% whitespace for moves. A raw split leaves about 50% whitespace per tier; Prime leaves 0%.
3) Type the blocks by area and degree; draw per-type power densities; attach CircuitNet textures.
4) Die 8-20 mm; power scale; stack template.
5) Infeasible 'performance-first' start: all DVFS at max, hot blocks stacked vertically, via budget 0, mid HTC. Peak overshoot is 5-40 K.
6) Draw limits by bisection on a reference heuristic (greedy unstack + via budget + DVFS), as in the current thermal.py.
Before scaling, pilot 50 problems and check G1 (spread of overshoot, binding-constraint mix) and the action-type mix of gold traces. FloorSet-Lite (1M) x families x power mixes x limits is effectively unlimited. Tier B has only 8 Open3DBench MoL netlists, so its variety comes from power, stack and limit draws.

**cost budget.** Basis: measured scipy single-thread numbers at 128^2 x 10. Each turn:
- 1 factorisation: 33 s with SuperLU (CHOLMOD/Pardiso estimated at a few s, not measured).
- About 15 power-only candidates x 3 leakage back-solves x 0.035 s, about 1.6 s.
- About 5 matrix-changing candidates x warm PCG at 2.5 s, about 12.5 s.
- Verify top 3 x about 7 solves x 2.5-3 s, about 60 s.
That is about 105 core-s per turn and about 35 core-min per problem (20 turns). 300 problems is about 175 core-hours, about 1.4 node-hours on one 128-core trace node running 128 problems in parallel. Memory is about 1-1.5 GB per process (7.3e7 SuperLU fill), about 190 GB per node; check node RAM.
Without screening (full re-plan for all 20 candidates): about 2.2 h per problem and about 670 core-hours (about 5 node-hours). Still affordable, so screening is optional.
No GPU is needed at 128^2. The proposal's '1.2 s GPU solve, 30-40 GPU-h' rested on a CUTh end-to-end figure that is overhead-dominated and no faster than CPU PCG.
Held-out 256^2 x 10: about 8x per solve (4x unknowns, about 2x iterations). Run it on Bridges-2 H100 with AmgX/cuDSS/CUTh (CUTh end-to-end 1.8-2.5 s on RTX 5090), or CPU for a 30-problem subset (about 500 core-hours).
Audit: 6,000 gold states x 3D-ICE 128^2 at 12.4-19.5 s x 4 threads, about 115 core-hours. A 10% subset at 256^2 (94.7-102.1 s x 4 threads x 600) is about 67 core-hours; it fits in 32 GB, but 3D-ICE runs out of memory at 512^2.
Tier B (if built): DREAMPlace MoL full placement takes 17-454 s per design. Incremental re-place for 3 verified candidates x 20 turns at an assumed 60 s is about 1 GPU-h per problem; 30-50 problems is about 30-50 GPU-h on Bridges-2.
Probe scripts:
- C:/Users/xuwen/AppData/Local/Temp/claude/C--Users-xuwen-designbench-llm-finetune/1c3b6209-a2ab-49bb-a719-64621aa93b14/scratchpad/bench_fv.py
- .../scratchpad/g2_probe.py
- .../scratchpad/via_plan_probe.py

**risks.** SKEPTIC: Verified:
- 3D-ICE: GPL-3.0, v4.0 2025-11-19, makefile.def openblas path, Server/Client.
- CUTh Table II/III numbers (RTX 5090, w3-2423, 32 GB).
- 3D-ICE 4.0 at 2.5M grids: 146.9 s / 35.28 GB, vs PACT 949 s / 48.43 GB.
- FloorSet scale; Open3DBench Table I; HotSpot NOASSERTION.
Changed:
(1) CUTh demoted. It is not pip-installable and has no documented warm start, and its 1.2 s is overhead. The CPU inner loop measured 2-3 s per PCG solve and 0.03 s per back-solve, so the whole run is about 175-670 trace core-hours with no GPU.
(2) Power-only candidates are batched as back-solves; per-block Green's functions dropped.
(3) FloorSet-Prime is fully abutted (moves impossible), so FloorSet-Lite plus a per-tier re-pack; the dataset is CC BY 4.0.
(4) ATPlace2.5D cases ARE public (ATPlace_pub; no licence, encrypted kernel).
(5) Scale honesty. The 0.16-2.6M unknowns are state, not design. Per-region ttsv_pct is replaced by budget + keep-outs/caps with an adjoint per-cell via planner as the real bulk solver (probe: +3-6 K over region-uniform). CircuitNet textures made mandatory.
(6) The HTC lever was free in the perf family, and it was the best probe candidate, so all levers are now costed.
(7) Leakage feedback and a routing-overflow constraint added.
(8) Tier B (Open3DBench MoL + DREAMPlace, 32K-1.08M cells) added for genuine million-object design scale.
(9) IC-ThermBench checked and rejected.
Risks:
1) DREAMPlace-level bar. Tier A's agent-visible layer is about 10^2 and its bulk design field is 1.6e4-2.6e5 per tier. That is DREAMPlace-like in structure (agent sets fences, budgets and weights; solver places the bulk), but not 10^6 netlist-connected objects. Only Tier B reaches that, and it overlaps the placement domain, so the owner must decide whether Tier B lives here or there.
2) Synthetic instances. FloorSet is synthetic and block power is assigned; calibrate to HotGauge and Open3DBench. Open3DBench and ATPlace_pub have no licence file.
3) Grid resolution. Without sub-block textures, 256^2/512^2 add nothing (3D-ICE 4.0: RMSE < 0.3 K at 11,760 cells). CUTh reports 1 um maps peak up to 6.3x above 20 um maps, so set limits at the training resolution and spot-check at 256^2.
4) The probe evidence is one synthetic instance with an uncalibrated heat sink (182 K rise). The G2 spread and the 3-6 K planner gain must be re-measured on calibrated stacks over the 50-problem pilot.
5) Screen-then-verify changes phi_select slightly. Report rank agreement; fall back to full evaluation if top-1 agreement is below about 80%.
6) The via planner makes evaluate() path-dependent through rho, so rho must be serialised in the state and re-plans must be deterministic (fixed iteration count, no randomness).
7) Degeneracy. Watch the action-type mix; if DVFS or budget moves dominate, retune costs or limits.
8) Licensing. 3D-ICE, CUTh, HotSpot and CHOLMOD GPL modules run as separate processes or libraries; not vendored.
9) Liquid-cooled variants cannot use CUTh or the SPD solvers (advection makes the matrix non-symmetric). 3D-ICON data is not available, and 3D-ICE needs more than 32 GB at 512^2.
10) scipy SuperLU factorisation (33 s, 145 s with COLAMD) dominates CPU cost. If CHOLMOD/Pardiso cannot be installed on trace, use PCG-only (about 2.5 s per solve) and accept roughly 2x cost.

**Candidates considered:**

| name | license | scale | runtime/eval | verdict |
|---|---|---|---|---|
| [In-house 7-point FV assembler + sparse solver stack (scipy / SuiteSparse CHOLMOD / MKL Pardiso / pyamg on CPU; AmgX, cuDSS via nvmath-python, or cupy CG on GPU)](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html) | scipy BSD-3; pyamg MIT; SuiteSparse CHOLMOD LGPL/GPL modules (separate library); AmgX: NVIDIA open source (LICENSES folder; GitHub API reports none, not re-verified); nvmath-python: NVIDIA | 128^2 x 10 = 163,840 unknowns and 1.11M nnz (measured). 256^2 x 10 = 655K and 512^2 x 10 = 2.6M by construction. | MEASURED locally (scipy 1.13, one thread, CUTh 10-layer stack, 10 mm die): Jacobi-PCG takes 2.0-3.1 s per solve (830-890 iterations to rtol 1e-8/1e-9); warm start saves only about 20% of iterations. SuperLU factorisation takes 33-34 s with MMD_AT_PLUS_A ordering (7.3e7 fill) and 145 s with COLAMD (2.0e8 fill). After that a back-solve costs 0.03-0.05 s per RHS (12 RHS in 0.40 s). CHOLMOD/Pardiso with METIS ordering should factor about 10x faster (not measured). GPU at 164K: well under 0.1 s per solve in-process, extrapolated from CUTh Table III (5.1M unknowns in 0.5 s); not measured. | PRIMARY inner loop (replaces CUTh as primary). One factorisation per distinct conductance matrix plus batched back-solves for power-only candidates; warm Jacobi/AMG-PCG for matrix-changing ones. |
| [3D-ICE 4.0 (EPFL ESL)](https://github.com/esl-epfl/3d-ice) | GPL-3.0 (verified via GitHub API) | Verified: Fig. 6 sweeps up to 2.5M and 3.6M grids; the CUTh comparison runs at 128^2/256^2 x 10 and is out of memory at 512^2 x 10 with 32 GB. | Verified: 12.4-19.5 s at 128^2 x 10 and 94.7-102.1 s at 256^2 x 10 (4 threads, Xeon w3-2423, CUTh Table II). 2.5M grids: 146.9 s on 7 cores (0.8 s build + 146.1 s solve), 35.28 GB peak, on 2x EPYC 9734. PACT on the same case: 949.0 s and 48.43 GB. | REFERENCE / AUDIT (unchanged). Ground truth for assembler validation and gold-state audits; liquid-cooled variant only. |
| [CUTh-Solver (CUHK/EPFL, GPU Chebyshev-PCG)](https://github.com/Chenghan-Wang/CUTh-Solver) | GPL-3.0 (verified); the paper is CC BY-SA 4.0 | Verified: matrices CUTh1-7 have 5.1e6-3.5e7 unknowns and 3.4e7-2.3e8 nnz; end-to-end stacks are 128^2-512^2 x 10. | Verified (RTX 5090, 32 GB, Xeon w3-2423 host, PyTorch 2.9.0): end-to-end 1.1-1.2 s at 128^2, 1.8-2.47 s at 256^2, 3.3-3.7 s at 512^2. Time grows only about 3x for 16x more unknowns, so the 128^2 figure is fixed-overhead dominated: Table III solves the 5.1M-unknown CUTh1 in 0.5 s total (44 iterations, 1,334 SpMVs). AmgX takes 1.3-7.8 s on CUTh1-7. The 128^2 end-to-end figure is no better than our measured single-thread CPU Jacobi-PCG (2-3 s). No H100 data. | DEMOTED to optional GPU accelerator for 256^2/512^2 held-out evaluation and audits, run as a separate process because of GPL. Not needed at 128^2. AmgX or cuDSS (nvmath-python) are better-supported alternatives. |
| [HotSpot 7.0 (UVA)](https://github.com/uvahotspot/HotSpot) | Custom UVA licence (GitHub: NOASSERTION, verified) | Out of memory at 256^2 x 10 with 32 GB (verified). | 162.1-163.5 s at 128^2 x 10 (verified). | SECONDARY (unchanged): coarse microfluidic channel-layout variant only. |
| [PACT (BU PEAC Lab)](https://github.com/peaclab/PACT) | GPL-3.0 (not re-verified) | Tests use 40^2-160^2 grids. | Verified from the 3D-ICE 4.0 paper: 949.0 s on 7 cores at 2.5M grids (148.2 s build + 800.8 s solve), 48.43 GB. | REJECT for the inner loop (unchanged). |
| [MFIT (multi-fidelity chiplet thermal)](https://github.com/AlishKanani/MFIT) | GPL-3.0 (not re-verified) | 4 RC nodes per chiplet, 64-256 nodes in total. | ms (DSS) to s (RC), per the paper. | REJECT as primary (unchanged). |
| [FloorSet (Intel Labs; ICCAD 2026 Contest Problem C)](https://github.com/IntelLabs/FloorSet) | CORRECTION: repo Apache-2.0, but the dataset is CC BY 4.0 (attribution required) | Verified: 1M Prime + 1M Lite, 21-120 blocks each, 100 validation and 100 hidden test cases. Prime is about 15 GB to download (25 GB unpacked); Lite about 6.1 GB (9.5 GB). | n/a | USE FloorSet-LITE as the block and net generator. Prime has no legal slot for the proposal's 'move to coolest legal slot' move, so use it only for swap-only variants. |
| [Open3DBench v1.1 (NJU LAMDA) + DREAMPlace 4.x as Tier-B inner placer](https://github.com/lamda-bbo/Open3DBench) | Repo: no licence file (GitHub API null, verified); paper CC BY 4.0. DREAMPlace is BSD-3. | Verified (Table I): MoL macro set of 8 designs with 32K-1.08M cells and 10-220 macros (bp_quad: 1.08M cells, 1.33M nets, 220 macros; ariane133: 158K cells). LoL set of 5 designs with 14K-90K cells. | Full MoL placement takes 17-454 s per design (paper). Incremental re-placement after one macro/region action is not measured. | REALISM SOURCE (power density, MoL stack templates). Also Tier-B family for the owner's million-object requirement: the agent acts on macros, tiers, region density caps and DVFS; DREAMPlace places the 1e5-1e6 cells; our solver scores temperature. Only 8 base netlists, so G1 variety comes from power scenarios and limits. Overlaps the placement domain; coordinate before building. |
| [ATPlace2.5D public release (PKU) / ATPlace_pub](https://github.com/Brilight/ATPlace_pub) | None stated (GitHub API null); needs clarification | 10 cases: 6-61 chiplets, 2,816-8,448 nets, interposer 25-61 mm (paper). | 0.3-7.5 min per full placement (paper). | USE the cases as the 2.5D-interposer family instance source once licence is clarified; not as simulator. |
| [IC-ThermBench (arXiv 2608.23977)](https://arxiv.org/html/2608.23977) | CC BY 4.0 | 82K samples (S1 32K, S2-S4 15K each, S5 5K). | Not reported. | REJECT (new candidate checked): a surrogate-learning benchmark, 64x64, far below DREAMPlace level. |
| [TAP-2.5D (BU)](https://github.com/bu-icsg/TAP-2.5D) | Apache-2.0 (not re-verified) | Tens of chiplets. | 2.8-7.3 min per placement. | REJECT; baseline only (unchanged). |
| [3D-ICON (ICCAD 2015 Contest Problem A)](https://www.iccad-contest.org/2015/problem_A/default.html) | Contest material; none stated | Verified: 5 testcases; 101x101 cells of 100 um (10.1 mm die); 0 = silicon, 1 = fluid; MinT (Pmax constraint) or MinCE (DP x Qin, s.t. Tmax and gradient). | Not stated | Formulation only, for the liquid-cooled variant (unchanged). |
| [Sinergym / SustainDC](https://github.com/ugr-sail/sinergym) | MIT | 1-2 zones, a few setpoints. | Not verified | REJECT (unchanged). |

**Sources:** https://github.com/limbo018/DREAMPlace · https://github.com/esl-epfl/3d-ice · https://api.github.com/repos/esl-epfl/3d-ice · https://raw.githubusercontent.com/esl-epfl/3d-ice/master/makefile.def · https://github.com/esl-epfl/3d-ice/tree/master/bin · https://arxiv.org/abs/2512.05823 · https://arxiv.org/pdf/2512.05823 · https://github.com/Chenghan-Wang/CUTh-Solver · https://api.github.com/repos/Chenghan-Wang/CUTh-Solver · https://raw.githubusercontent.com/Chenghan-Wang/CUTh-Solver/main/README.md · https://arxiv.org/abs/2606.17850 · https://arxiv.org/pdf/2606.17850 · https://github.com/uvahotspot/HotSpot · https://api.github.com/repos/uvahotspot/HotSpot · https://github.com/peaclab/PACT · https://github.com/AlishKanani/MFIT · https://github.com/IntelLabs/FloorSet · https://api.github.com/repos/IntelLabs/FloorSet · https://arxiv.org/html/2405.05480 · https://github.com/lamda-bbo/Open3DBench · https://api.github.com/repos/lamda-bbo/Open3DBench · https://arxiv.org/html/2503.12946 · https://github.com/Brilight/ATPlace_pub · https://api.github.com/repos/Brilight/ATPlace_pub · https://yibolin.com/publications/papers/PLACE_ICCAD2024_Wang.pdf · https://arxiv.org/html/2511.17319 · https://arxiv.org/abs/2607.09742 · https://arxiv.org/abs/2609.13157 · https://arxiv.org/html/2608.23977 · https://github.com/bu-icsg/TAP-2.5D · https://www.iccad-contest.org/2015/problem_A/default.html · https://github.com/circuitnet/CircuitNet · https://circuitnet.github.io/feature/properties.html · https://github.com/TuftsCompArchLab/HotGauge · https://github.com/NVIDIA/AMGX · https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html · https://www.researchgate.net/publication/4204235_Thermal_via_planning_for_3-D_ICs · https://experts.umn.edu/en/publications/placement-of-thermal-vias-in-3-d-ics-using-various-thermal-object/ · https://github.com/ugr-sail/sinergym · https://github.com/HewlettPackard/dc-rl

## epidemic

**Domain:** seir -> national- and city-scale epidemic intervention design. The per-turn evaluator is MEmilio's graph-ODE SECIR-VVS metapopulation model of Germany: 400 districts, about 53k commuter edges, 83M people, real RKI/DIVI/BAA data. MEmilio's hourly ABM on the Brunswick twin (373,378 agents) is a fidelity audit, and there is an optional separate per-turn ABM city domain using Covasim at about 1M agents. The agent sets cluster-level policy rules: NPI trigger thresholds and strengths, commuter testing, vaccine priority. The simulator's DynamicNPI triggers and a dose allocator expand these into the full per-district, per-day schedule. Constraints are pooled ICU excess demand, deaths, school-closure days and death disparity.

**simulator.** MEmilio 2.4.1 (Apache-2.0; pip wheels cp38-cp313 needing glibc 2.27 or newer; CPU only).

Tier N, the per-turn evaluator: graph-ODE SECIRVVS of Germany. 400 districts x 6 ages x 27 compartments = 64.8k node states, plus about 53k commuter edges carrying up to about 8.6M commuter-state values (estimate from set_edges' 4e-5 threshold). Run deterministically, one MobilitySimulation per evaluation, with n_rep=1. Do not use GraphParameterStudy, which samples parameters.

Tier C, the fidelity audit: MEmilio ABM on the Brunswick twin (373,378 agents, Zenodo 13318436), with a small pybind extension that exposes set_home_office, set_school_closure and close_social_events.

Optional second epidemic domain, the one that is million-scale per turn: a Covasim 1M-agent city using CRN seeds, at 13-26 s per run on one core. Doing per-turn ABM in MEmilio instead would cost about 15 core-s per simulated day at 1M agents, roughly 1,350 core-s per 90-day run, which is too expensive per turn.

**benchmarks.** There is no ISPD-style epidemic suite, so instances are generated on top of versioned public data. Snapshot the data and do not hit live endpoints:
- RKI cases: Zenodo 10.5281/zenodo.13841353
- RKI vaccinations: 10.5281/zenodo.12697471
- RKI/DIVI ICU: Zenodo 10.5281/zenodo.13843100. Its Intensivregister_Landkreise_Kapazitaeten.csv (42.7 MB, CC-BY-4.0) has per-district capacity, which resolves the open data question in the proposal (check the columns).
- BAA commuters and population via memilio-epidata (licence not verified).
- Brunswick trip chains: Zenodo 10.5281/zenodo.13318436 (the proposal's 7039458 is a deleted spam record).

Reproduce first: pycode/examples/simulation/2021_vaccination_sarscov2_delta_germany.py. Verified: it sets DynamicNPIsInfectedSymptoms at graph level with thresholds 35 and 100, implementation_delay 7, duration 14 and base_value 1e5, and sets commuter_nondetection to 0.85 when testing is on. CORRECTION: it does not implement any vaccination policy. Vaccinations come from RKI data through set_nodes, so our allocator must write the DailyPartialVaccinations and DailyFullVaccinations arrays itself.

Spain at 47 provinces appears in the paper, but memilio-epidata only has national Johns Hopkins (JH) data and contact data for Spain. The province pipeline is unverified, so do not count on it.

**decision layer.** The simulator owns the bulk: the coupled ODE state, per-district DynamicNPI firing (about 400 x 4 locations x 180 days of dampings), and the per-district, per-age, per-day dose arrays. Be honest that this bulk is expanded by a rule, not optimized.

The agent edits a compact policy program. Elements: K = 6-10 district clusters (k-means on log density, ICU beds per 100k, commuter in-degree and share aged 80+, fixed per problem) plus 1 national element, so 7-11 elements, within enumerate_candidates' 24-element cap.

CORRECTED ranges and starts. The multiplicative ladder (0.5, 0.7, 1.5, 2.2) cannot move a knob from 0, and the proposal's start (strength 0, testing 0, threshold at its 400 upper bound) would make every NPI candidate a no-op on turn 1, which fails G2:
- npi_threshold: relative symptomatic infections per 100k (get_infections_relative x base_value), range [10, 3000], start 200. 400 is not 'off' for Omicron-like problems.
- npi_strength: range [0.02, 1], start 0.1. Scales a fixed damping package: school up to 0.6, work up to 0.4, other up to 0.9.
- vacc_priority: range [0.05, 1], start 0.5.
- commuter_testing (1 - commuter_nondetection): range [0.01, 0.5], start 0.02.
- National element: npi_duration [7, 28] days, start 14; elderly_first [0.05, 1], start 0.3.

One turn changes one knob by one ladder factor, e.g. 'Ruhr-urban: trigger 200 to 100 per 100k'.

CANDIDATE COUNT: dataxp/domain.py enumerate_candidates scores every candidate up to max_candidates=160, and ladder.py phi_select scores all of them. K=8 gives about 136 candidates per turn, not 20. Set max_candidates=24 in this domain and pre-prune zero-cost no-ops: a threshold move in a cluster whose incumbent-run peak prevalence stays below both the old and new threshold has no effect.

API is now verified in the binding sources: graph.get_node(i).property.parameters exposes DynamicNPIsInfectedSymptoms.set_threshold(level, [DampingSampling...]), ICUCapacity, ContactPatterns, DailyPartialVaccinations/DailyFullVaccinations and commuter_nondetection. MobilitySimulation.advance(t) exists. Mobility edges also carry DynamicNPIs, which can reduce commuting.

**inner solver.** (a) MEmilio per-node DynamicNPIs, a closed-loop trigger inside Simulation.advance with delay, duration and keep-alive renewal, turn cluster thresholds and strengths into per-district, per-day dampings.
(b) An OPEN-LOOP dose allocator, written before the run. It splits each day's national supply by cluster weight x district population. Within a district it goes oldest age group first, mixed with uniform by the elderly_first share, capped by district throughput. Its output is the cumulative (AgeGroup, SimulationDay) DailyPartialVaccinations and DailyFullVaccinations arrays, which the model differences day to day. The proposal's closed-loop 'remaining susceptible' water-filling would need day-by-day Python stepping with mid-run parameter edits; this is unverified and slower, so keep it optional.
(c) The graph integrator (adaptive RK) with instant commuter exchange.

REFERENCE (reference_class='heuristic'): bisect one uniform threshold at fixed strength 0.5 to the cheapest feasible design, as in the toy domain. MEmilio's optimal control (IPOPT, single shooting, AD, weekly piecewise-constant controls) is NOT affordable per turn on the graph. The ESID pipeline (arXiv 2608.09933) runs it on a non-spatial Germany model 'to reduce practical runtime'. It is also not in the Python bindings. At most, use it offline as a national-control reference applied uniformly to districts.

**metrics objective constraints.** Objective (min): restriction_cost, in equivalent full-lockdown days per capita. It equals the sum over districts n, days t and locations l of pop_n x w_l x d_{n,l}(t) / N_total. The damping d is read back after the run from each node's ContactPatterns via ContactMatrix.get_matrix_at / get_dampings (both bound). Add c_test x commuter_testing x commuter volume per capita as the test-cost proxy, since MEmilio does not count tests.

ICU FIX: keep ICUCapacity at its default infinity in the model. A finite value makes SECIRVVS cosine-divert critical cases to death between 90% and 100% of capacity, so occupancy never exceeds capacity and an excess metric would be identically about 0. Instead compute ICU demand (the sum of the InfectedCritical compartments) post hoc against capacity from the RKI per-district capacity CSV. Pool capacity at cluster or Land level, because ICU patients are transferred and many districts have few beds.

Constraints:
- maximum_icu_excess_bed_days_per_100k: sum over pools and days of max(0, demand - cap), using a smooth hinge.
- maximum_deaths_per_100k (national).
- maximum_school_closure_days: population-weighted days with school damping of at least 0.3.
- maximum_death_disparity: p95 / median of district deaths per 100k.

Caps are calibrated against the INITIAL weak-policy design, not a free run: deaths cap at 30-60% of its value, ICU-excess cap at 10-30%. Reported only: peak_icu_demand_frac, attack_rate, doses_used, share of person-days under NPI, and pop_conservation_err (envelope). Read everything from the node TimeSeries after interpolate_simulation_result to daily points.

**observation summary.** About 70 lines, never per-agent or per-district dumps.
- PROBLEM: pathogen family (R0, severity, immune escape), RKI start date or synthetic seeding, horizon, daily dose supply, ICU capacity scale, constraint caps, lever meanings and bounds, and a K-row cluster table (number of districts, population, ICU beds per 100k, commuter in-flow share, 3 example districts).
- DESIGN: the K x 4 lever table plus the national levers.
- RESULTS: national deaths per 100k, ICU excess bed-days, peak ICU demand/cap, restriction cost, NPI person-day share; weekly national incidence and ICU-demand series (13-26 numbers each).
- Per cluster: peak prevalence vs its threshold. This tells the agent whether the trigger is live, and moving a dead trigger is a no-op. Also per cluster: days triggered, peak ICU %, excess days, deaths per 100k, and 60+ coverage.
- Top-10 districts by ICU excess (name, cluster, peak demand/cap, peak week, triggered days); deaths by age (6 numbers); the binding constraint and its gap.

**problem generator.** make_problem(index) seeds with stable_seed and samples:
(1) The initial condition, which is the main G1 source. Either an RKI start date in 2020-03 to 2022-12, which gives real, spatially heterogeneous district states and immunity, with vaccination data from 2020-12-27; or synthetic seeding of 1-5 districts with 10-1,000 cases.
(2) Pathogen family: wildtype-, Alpha-, Delta- or Omicron-like. A flu-like family needs re-parameterised SECIRVVS and synthetic seeding only.
(3) ICU capacity scale 0.4-1.0 x RKI/DIVI.
(4) Dose supply 0-1% of population per day, starting on day 0-60.
(5) Horizon 90-180 days.
(6) K and clustering features, plus cost weights w_school > w_work > w_other.
(7) Geography. Full Germany for about 70% of problems. The rest are unions of adjacent Länder with at least 100 districts each, which gives distinct graph topologies from the same data. Spain is unverified.

The initial design is the weak national default (threshold 200, strength 0.1, testing 0.02, priority 0.5). Caps are set relative to it, so it violates the ICU and death goals. Reject any problem that the bisection reference cannot make feasible, or that starts feasible.

ABM-tier problems on Brunswick sample variant, seeding, testing capacity and ICU scale.

**cost budget.** CORRECTED. Tier N planning cost is 30-150 core-s per SECIRVVS run. 10 s (SECIR, paper Fig. 6e, hardware not stated) is a floor; about 150 core-s is the bound implied by the ESID pipeline (330 runs in 6-7 min on a 128-core EPYC 7742 node).

phi_select with max_candidates capped at 24 and no-op pruning: 24 x 20 turns = 480 runs, so 4-20 core-hours per problem, 1.2k-6.0k core-hours for 300 problems. That is about 9-47 node-hours on 128-core trace nodes, at one run per core if memory allows. Measure RSS first: 53k edges plus adaptive-step TimeSeries could be GBs.

If the default max_candidates=160 is left in place, K=8 gives about 136 candidates per turn, about 2,700 runs per problem and 6.8k-34k core-hours.

Ways to cut it:
- Pre-prune no-op threshold moves using the incumbent trajectory (free).
- Use 90-120 day horizons.
- Warm-start each turn from the earliest day any candidate can change the trajectory. This is nontrivial: edge commuter vectors and the private m_dynamic_npi state are not serialisable from Python, so restart only at day boundaries. Expect at most a 2x saving.
- CORRECTION: drop GNN pre-screening. The published surrogate (Zenodo 18507331) emulates SECIR with no vaccination layers, at 10-27% MAPE, which is larger than typical top-vs-median Phi gaps. Retraining it would cost thousands of runs.

Tier C audit: 1 chosen action x 20 turns x 2 seeds = 40 Brunswick runs per problem, at 300-1,160 core-s each. CORRECTED: the 65 core-s lower bound was 8-thread wall time. That is 1.0k-3.9k core-hours for 300 problems.

Optional Covasim 1M-agent city domain: 24 x 20 x 4 replicates x 13-26 s = 7-14 core-hours per problem, 2.1k-4.2k core-hours for 300 (about half with CRN at 2 replicates).

GPU hours: 0. Pilot first on one trace node: time and RSS for one 400-district SECIRVVS run at 90 and 180 days.

**risks.** SKEPTIC: Maintenance, licence, wheels, DynamicNPI and commuter-testing APIs, the 128M/256M ABM claim, Covasim's 7M person-days/s and Brunswick's 373,378 agents are confirmed. Changed:
(a) The 10 s graph-ODE time is for SECIR; SECIRVVS is planned at 30-150 core-s from the ESID pipeline data.
(b) The repo's phi_select scores up to 160 candidates, not 20 (K=8 gives about 136). Cap at 24, or the budget grows 6-7x.
(c) The zero-valued start knobs and threshold=400 cannot move under the multiplicative ladder, so turn 1 would be all no-ops and fail G2. New nonzero starts and bounds are given.
(d) SECIRVVS ICUCapacity diverts overflow to deaths, which would make the ICU-excess constraint about 0. Keep capacity infinite and measure demand post hoc against pooled RKI per-district capacity, which exists (Zenodo 13843100).
(e) The Brunswick data DOI was wrong; it is 13318436.
(f) ABM 0.08 s/step was on 8 threads, so audit cost is 1.0-3.9k core-hours.
(g) GNN pre-screen dropped (SECIR-only surrogate, 10-27% MAPE).
(h) The Delta example has no vaccination policy, so the allocator writes the cumulative dose arrays open-loop.
(i) Per-node parameter access and MobilitySimulation.advance confirmed in the bindings.
(j) Spain is not in memilio-epidata; G1 now comes from RKI start dates and Länder-union subgraphs.
(k) Covasim is promoted to the optional per-turn million-agent city domain.

Remaining risks:
(1) DREAMPlace parity is partial. The numerical state (about 65k node states plus up to about 8.6M edge values) and real-data scale match, but the agent-facing bulk (per-district-day NPIs and doses) is expanded by trigger and allocation rules, not optimized. Graph-level optimal control is too slow per turn (MEmilio's own pipeline runs it non-spatially).
(2) Per-run time and memory for SECIRVVS on our nodes are unmeasured. Adaptive-step TimeSeries on 400 nodes and 53k edges may limit runs per node.
(3) Threshold triggers are bang-bang per district. Aggregation over 40-70 districts per cluster should smooth Phi, but pilot G2 on 10 problems x 5 turns: fraction of exact ties, and top-vs-median gap as a share of |Phi|.
(4) Only one country, with the base mobility graph fixed. G1 relies on initial-condition and pathogen variety plus Länder subgraphs, which are smaller than the full graph.
(5) The ABM closure helpers need a C++/pybind build (conda gcc, CMake, no root).
(6) The BAA commuter licence is unverified, and the data pipeline must be frozen to the Zenodo snapshots.
(7) Reading the restriction cost back from get_matrix_at should be cross-checked against a Python re-implementation of the trigger on one run.
(8) MEmilio's ABM is likely 13-20x Covasim's cost per simulated day (hourly vs daily steps); benchmark before choosing the city-domain engine.
(9) There is no canonical benchmark; XPRIZE is a framing only.

**Candidates considered:**

| name | license | scale | runtime/eval | verdict |
|---|---|---|---|---|
| [MEmilio (DLR/HZI/FZJ)](https://github.com/SciCompMod/memilio) | VERIFIED. Code is Apache-2.0 (GitHub and PyPI metadata). The Sci Rep 2026 article (16:26950, published 2026-08-27) is CC-BY-4.0. Data assets have their own licences: the RKI Zenodo snapshots and the Brunswick trip chains are CC-BY-4.0. The BAA commuter data licence was not verified. | Graph-ODE: 400 districts x 6 age groups x 27 SECIRVVS compartments = 64,800 node states (27 counted from the bound InfectionState enum). set_edges keeps commuter links above 4e-5. A code comment says this keeps about 1/3 of the 159,600 ordered pairs and covers more than 99% of mobility, so about 53k edges. Each edge carries an age x compartment commuter vector, giving up to about 8.6M edge-state values (estimate; only mobile compartments travel). The paper also instantiates Spain at 47 mainland provinces, calibrated to ICU. ABM: 128M and 256M agents completed in the paper's benchmark; Covasim and OpenCOVID did not finish above 64M. Brunswick twin: 373,378 agents (arXiv 2410.08050). CORRECTION: its data is Zenodo 10.5281/zenodo.13318436 ('Microscopic trip chains for Brunswick (Germany) region', CC-BY-4.0, 240.5 MB CSV, 2024-08-14). Zenodo 7039458, cited in the proposal, is a deleted spam record. | CORRECTED. Graph-ODE: the paper's ~10 s per run (Fig. 6e) is for the SECIR model the GNN emulates (no vaccination layers; 30/60/90-day horizons; hardware not stated), so it is a floor. An independent SECIRVVS 400-district datum comes from arXiv 2608.09933. There, 33 scenarios x 10 runs = 330 runs took about 197 min in Kubernetes pods (8 vCPU limit) and 6-7 min on one 128-core JURECA node (2x EPYC 7742), including upload. That puts the upper bound at about 150 core-s per run (horizon not stated). Planning range: 30-150 core-s per SECIRVVS run; benchmark first. ABM: CORRECTED. 0.08 s per hourly step at 1M agents was measured with 8 OpenMP threads on JURECA EPYC 7742, averaged over a 5-day simulation; it was not a single-core laptop run. So single-core cost is at most about 0.64 s per step, or about 15 core-s per simulated day at 1M agents. Brunswick, 91 days: about 65 s wall on 8 threads, so at most about 520 core-s. The calibration bound (85,536 runs on 3,456 cores in under 8 h) gives at most 1,164 core-s per run. Use 300-1,200 core-s. Ensemble: 16,384 runs of a 2M-agent ABM in about 17 min on 16,384 cores, so about 1,000 core-s per run (horizon not stated). | PRIMARY, with corrections. It is still the only permissive, actively maintained tool that covers a real-data national metapopulation and a city ABM. The per-turn graph-ODE (1e5 node states plus up to ~1e6-1e7 edge values) is in DREAMPlace's numerical range. The decision bulk it expands (about 288k per-district-day dampings) is produced by a trigger rule, not an optimizer. Per-turn cost is 3-15x the proposal's 10 s figure. |
| [Covasim (IDM / Starsim hub)](https://github.com/starsimhub/covasim) | MIT (verified) | 64M agents completed in the MEmilio benchmark; 128M and 256M did not. The King County 2.25M people modelled as 225k agents with dynamic rescaling was NOT re-verified (not found in the text fetched). No native multi-region structure. | VERIFIED: about 7M simulated person-days per CPU-second on one i9-8950HK core, about 1 KB per agent. So 1M agents run about 13 s for 90 days and about 26 s for 180 days, before tracing overhead. MEmilio's ABM is only 1.2-1.8x faster per step but steps hourly, while Covasim's default step is 1 day. Per simulated day, MEmilio's ABM is therefore likely 13-20x more expensive (inference; benchmark locally). | PROMOTED from fallback to the optional separate per-turn ABM city domain. It is the only option where a million-agent ABM fits 20 turns x 24 candidates x 4 replicates: about 2-4k core-hours per 300 problems. Region structure must be built by hand with subtargets. |
| [Starsim](https://github.com/starsimhub/starsim) | MIT (verified) | Project claims hundreds of thousands of agents in under a minute on a laptop. The CRN paper (arXiv 2409.02086) tested 10-32,000 agents. Not re-verified. | Not verified at 1M agents. | Not primary. Copy its CRN design into the Covasim city domain (shared seeds across candidates) to cut replicates for G2. |
| [OpenABM-Covid19](https://github.com/BDI-pathogens/OpenABM-Covid19) | GPL-3.0 (verified) | 1M people by default with UK demographics; runtime is linear in population. | About 1.5 s per simulated day at 1M agents (reported; not re-verified). | No: slower than Covasim, single region, stale. |
| [EpiHiper (UVA NSSAC)](https://github.com/NSSAC/EpiHiper) | MIT (verified) | Virginia 7.7M nodes and 372M edges, up to a 288M-node US network (reported). | Not verified per run. Uses MPI/OpenMP on nodes with about 375 GB RAM. | No for data generation (cost per run). Cite as a national-scale reference. |
| [Loimos](https://github.com/loimos/loimos) | NOASSERTION on GitHub (verified); effectively unclear. | US state digital twins, including California. | About 42 s on 4,096 Perlmutter cores for California, 200 days, so about 48 core-hours per run (reported). | No: cost per run, and the licence is unclear. |
| [PanSim (PPCU)](https://github.com/khbence/pansim) | CC-BY-4.0 on GitHub (verified; unusual for code but permissive). | Szeged twin of 179,500 agents; claims up to 50-90M agents on one GPU. | Under 64 s per simulated year on a V100 (reported). | Possible secondary GPU-ABM domain (H100 / RTX PRO 6000). Single town only. |
| [AgentTorch / GradABM](https://github.com/AgentTorch/AgentTorch) | AGPL-3.0 on GitHub (verified); PyPI metadata claims MIT (not re-checked). Conflict unresolved. | 8.4M-agent NYC population. | Seconds per million agents on GPU (reported). | No as primary, because of licence ambiguity and low epidemiological fidelity. |
| [CovidSim (MRC-IDE)](https://github.com/mrc-ide/covid-sim) | GPL-3.0 (verified) | UK and US national populations. | Not verified; the UK setup needs about 20 GB RAM. | No: stale and heavy. |
| [Epiabm](https://github.com/SABS-R3-Epidemiology/epiabm) | BSD-3-Clause (verified) | Small Gibraltar benchmark only. | 8 s with the C++ backend for Gibraltar (reported). | No: not shown at scale. |
| [JUNE (Durham)](https://github.com/IDAS-Durham/JUNE) | GPL-3.0 (verified) | 53M people in England. | Not reported. | No: unknown and heavy runtime. |
| [XPRIZE Pandemic Response harness](https://github.com/cognizant-ai-labs/covid-xprize) | NOASSERTION on GitHub (verified); README says Apache-2.0. | 236 regions x 12 OxCGRT NPIs. | Not verified. | Not a simulator. Use only for the framing of a stringency-cost objective. |

**Sources:** https://github.com/SciCompMod/memilio · https://api.github.com/repos/SciCompMod/memilio/releases?per_page=6 · https://api.github.com/repos/SciCompMod/memilio/commits?per_page=3 · https://pypi.org/pypi/memilio-simulation/json · https://arxiv.org/abs/2602.11381 · https://arxiv.org/html/2602.11381 · https://pmc.ncbi.nlm.nih.gov/articles/PMC13522579/ · https://arxiv.org/pdf/2608.09933 · https://arxiv.org/abs/2411.06500 · https://zenodo.org/records/18507331 · https://arxiv.org/pdf/2410.08050 · https://zenodo.org/api/records/13318436 · https://zenodo.org/api/records/7039458 · https://zenodo.org/api/records/13843100 · https://doi.org/10.5281/zenodo.13841353 · https://doi.org/10.5281/zenodo.12697471 · https://github.com/SciCompMod/memilio/blob/main/pycode/examples/simulation/2021_vaccination_sarscov2_delta_germany.py · https://github.com/SciCompMod/memilio/blob/main/pycode/memilio-simulation/memilio/simulation/bindings/models/osecirvvs.cpp · https://github.com/SciCompMod/memilio/blob/main/pycode/memilio-simulation/memilio/simulation/bindings/mobility/metapopulation_mobility_instant.h · https://github.com/SciCompMod/memilio/blob/main/pycode/memilio-simulation/memilio/simulation/bindings/mobility/graph_simulation.h · https://github.com/SciCompMod/memilio/blob/main/pycode/memilio-simulation/memilio/simulation/bindings/epidemiology/dynamic_npis.h · https://github.com/SciCompMod/memilio/blob/main/pycode/memilio-simulation/memilio/simulation/bindings/epidemiology/contact_matrix.h · https://github.com/SciCompMod/memilio/blob/main/pycode/memilio-simulation/memilio/simulation/bindings/models/abm.cpp · https://github.com/SciCompMod/memilio/blob/main/cpp/models/ode_secirvvs/model.h · https://github.com/SciCompMod/memilio/blob/main/cpp/models/ode_secirvvs/parameters.h · https://github.com/SciCompMod/memilio/blob/main/cpp/memilio/mobility/graph.h · https://github.com/SciCompMod/memilio/blob/main/cpp/models/abm/lockdown_rules.h · https://github.com/SciCompMod/memilio-simulations/tree/main/2026_Bicker_et_al_Memilio_paper · https://github.com/starsimhub/covasim · https://pypi.org/pypi/covasim/json · https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009149 · https://github.com/starsimhub/starsim · https://arxiv.org/abs/2409.02086 · https://github.com/BDI-pathogens/OpenABM-Covid19 · https://github.com/NSSAC/EpiHiper · https://github.com/loimos/loimos · https://github.com/khbence/pansim · https://github.com/AgentTorch/AgentTorch · https://github.com/mrc-ide/covid-sim · https://github.com/SABS-R3-Epidemiology/epiabm · https://github.com/IDAS-Durham/JUNE · https://github.com/cognizant-ai-labs/covid-xprize · https://github.com/epistorm/epydemix · C:/Users/xuwen/designbench_llm_finetune/dataxp/domain.py (enumerate_candidates max_candidates=160, multiplicative factor ladder) · C:/Users/xuwen/designbench_llm_finetune/dataxp/ladder.py (phi_select scores all enumerated candidates) · C:/Users/xuwen/designbench_llm_finetune/dataxp/domains/seir.py

## supply_chain

**Domain:** supply_chain (replaces the toy 'inventory' domain): industrial-scale multi-echelon inventory optimization (MEIO). It has two problem families. Family A uses the 38 real Willems general-DAG chains (8 to 2,025 stages, up to 16,225 arcs; verified). Family B is DREAMPlace-scale: synthetic BOM/distribution networks of 20K to 200K+ stages and up to about 1.6M arcs, built the way Wan & Hong (IJOC 2023) built theirs. In both, an agent makes the strategic safety-stock decisions, an inner GSM solver handles the bulk, and a batched GPU simulator measures the result under stochastic real demand.

**simulator.** Our batched PyTorch daily periodic-review simulator, vectorised over [candidates x CRN replicates x stages/arcs] on the GPU. It models:
- BOM assembly limited by the scarcest component;
- proportional rationing and backorders;
- stochastic stage times (to be added; the prototype is deterministic);
- supplier stages with real lead time instead of unlimited instant supply;
- holding value at cumulative (value-added) stage cost.
Inner GSM: our sequential-LP (dynamic-sloping) solver on the FULL DAG using scipy-HiGHS. It replaces 'GW-DP on a spanning subtree', which drops 88% of chain 38's arcs. Measured: 0.12 s cold at 2K stages, 3 s at 20K, 100 s at 200K.
References and cross-checks:
- Data in Brief Gurobi incumbents for Family A;
- InvNet-GSM HGNA/ID-SLP as a baseline;
- stockpyl GW-DP on tree chains.
Validation gate, required before any data generation: within tolerance of InventoryManagement.jl and stockpyl.sim on chains of 100 stages or fewer, ReplenishmentEnv on serial configs, and the closed-form base-stock fill on single-stage and serial cases.

**benchmarks.** Family A (real topology): the Willems 38 chains, with verified counts.
- 12 chains under 100 stages: simulator validation and debugging.
- 21 chains of 100-999 stages: train/dev.
- 5 chains of 1,000-2,025 stages (34-38): held-out test.
- References: the Data in Brief best-known normal-demand GSM cost, with reference_class='bound'. Gaps of up to 25% mean these are not exact optima.

Family B (DREAMPlace-scale, needed because no public real MEIO network exceeds 2,025 stages): 20K-200K+ stage networks following Wan & Hong's setup (average degree about 8-10; their tests go up to 500K nodes; their real BOM exceeds 500K nodes and 4M links).
- Build them by composing several Willems chains as product families that share common Part/Manuf stages ('shared materials' per InvNet).
- Add a store -> city-DC -> central-DC distribution tier whose shape comes from FreshRetailNet-50K (898 stores, 18 cities, 863 SKUs; CC BY 4.0).
- Resample stage attributes per class from Willems empirical distributions, so the network is heterogeneous rather than uniform random.
- Split by source chain and by demand-store set to avoid train/test leakage.
- Reference for Family B: the best SLP-GSM plus uniform-z sweep solution found offline, with reference_class='bound'.

**decision layer.** One agent turn is one set(element, param, value). The inner GSM and closed-form base-stock levels then update every stage, and the GPU simulator measures the outcome.

Element types:
(1) policy groups: stageClass x depth band x demand-CV tercile; 15-45 groups in Family A and 48-96 in Family B (add region and product family). Param z in [0, 4]. Moves are x0.8 / x1.25 / +-0.25, PLUS coarse echelon-wide moves ('all demand-stage groups +0.25'). The coarse moves are required: in the probe, initial fill is about 0.77 against a 0.90-0.98 target, while a single-group x1.25 move adds only about +0.008 fill, so phi_select could not reach feasibility in 20 turns without them.
(2) hubs: the top 16 (Family A) or 32 (Family B) stages by downstream demand value x stage time. Param cst_pin in {free, force-stock (S=0), force-pass-through}. The enumerator must offer ONLY state-changing toggles: in SLP runs, 'force-stock' on the top hubs was a no-op in 7 of 8 hubs at 2K and 3 of 3 at 20K and 200K, because those hubs already held stock.
(3) optional lead-time levers at hubs: expedite with T_j -> 0.7*T_j at a $ cost, a real design lever that changes the GSM.
(4) RESOLVE knob: re-optimise all service times with per-group sigma inflation.

Exclusions and rules:
- Never offer z moves on groups made only of unlimited-supply sources. They were exact no-ops (dPhi = 0.000 +- 0.000) in the probe.
- z moves keep the current service times fixed (no GSM re-solve). Only pin, expedite and RESOLVE moves re-solve, which keeps the CPU cost bounded at 200K scale.

**inner solver.** Full-DAG SLP/dynamic-sloping GSM (scipy-HiGHS; with HiGHS directly or Gurobi it would be faster). It runs warm-started from the current service times on pin, expedite and RESOLVE candidates only, and returns S_j for all N stages. Measured warm re-solve: 0.07 s at 2,025 stages / 16,225 arcs, 1.1 s at 20K / 162K, 26 s at 202K / 1.62M. Base-stock levels are then computed vectorised as B_j = mu_j*NLT_j + z_g(j)*sigma_j*sqrt(NLT_j), with NLT_j = SI_j + T_j - S_j and sigma propagated upstream. The GPU simulator executes the policy. Intermittent real demand, rationing and stochastic stage times break GSM's bounded-demand guarantee; showing where it breaks is the agent's job and must be stated in describe().

**metrics objective constraints.** All metrics come from simulation: 365 days after a 100-day warm-up, with common random numbers across candidates.

Objective: minimise inventory_cost in $/yr = holding rate x cumulative stage value x (on-hand + component buffers + pipeline WIP if accounted). Source/supplier stock must be finite and depleting; this was the phantom-cost bug.

Constraints:
- minimum_fill_rate: demand-weighted IMMEDIATE fill at demand (sink) stages, not at 'Retail' class (13 of 38 chains have no Retail stage). Target 0.90-0.98.
- minimum_p10_fill_rate: 10th percentile across demand stages; target minus about 0.08.
- maximum_mean_wait_days: at most the SLA derived from maxServiceTime.
- Optional maximum_safety_stock_value in $ (a budget). This creates real trade-offs once the problem is feasible.

Reported only: GSM-predicted cost, number of decoupling points, $ by echelon.

Measured G2 evidence (prototype, corrected cost; 21 candidates x 8 CRN reps):
- top-vs-median Phi separation is 42-48 SE at 2K stages and 150-250 SE at 101K stages. G2 passes easily.
- BUT the top 3 candidates are often within about 1 SE of each other on a homogeneous synthetic net (top-1 minus top-3 is about 0.05 against a 2-3 Phi spread). Gold labels among the top few are near-ties unless the network is heterogeneous. Report the top-1 vs top-2 margin as an additional G2 statistic.

**observation summary.** About 80 lines (2-4K tokens) at both scales, no per-stage dump:
(a) header: chain or composition id; stages by class; arcs; max depth; number of demand stages; longest path in days; demand value per day.
(b) constraint dashboard: each metric vs target with the signed gap; cost vs reference.
(c) group table (up to 45 rows in A; in B, the 45 rows with the largest gaps or $, with the rest aggregated): z, number stocked, safety-stock $, fill mean/p10, wait.
(d) top-10 worst demand stages with the attributed upstream bottleneck.
(e) top-10 safety-stock $ locations.
(f) hub list with pin state, SI/S/T, stock $, and which toggle is currently non-trivial.
(g) decile histogram of demand-stage fill; $ by echelon.
(h) delta from the last action.
(i) a one-line note that GSM assumes bounded demand while the simulator rations.

**problem generator.** make_problem(index), with an RNG seeded by index:
- Choose the family: Family A with probability about 0.6, Family B with about 0.4.
- Family A: draw a chain (weighted toward 100+ stages) and optionally the upstream closure of 30-100% of demand stages.
- Family B: compose 3-30 Willems chains with shared parts, plus a FreshRetailNet-derived distribution tier, for 20K-200K stages.
- Demand: MABIM daily series (Family A) or latent-demand-recovered FreshRetailNet series, block-bootstrapped to 465 days (Family B), rescaled to Willems avgDemand, with a CV multiplier of 0.7-2.0.
- Other parameters: stage-time noise 0-1.5; holding rate 15-35%; fill target 0.90-0.98; SLA tightening 0-50%; grouping seed; optional $ budget.
- Initial design: SLP-GSM service times with a uniform z drawn so that the initial fill gap is 0.03-0.15. Calibrate this so phi_select reaches feasibility in about 8-12 of 20 turns; the probe's z=1.0 start (fill about 0.77) is too deep for single-group moves.

G1: there are only 26 real chains with 100+ stages, about 11 problems per chain at 300 problems, and sub-networks of one chain are correlated. Family B composition and real demand heterogeneity supply the rest of the variance. Check G1 per family.

**cost budget.** Measured on the RTX 3090 (21 candidates x 8 CRN reps in one batched call, 465 days):
- Family A (2,025 stages): 0.47-0.57 s GPU per turn, plus SLP warm re-solves of 0.07 s each for the about 8 pin/expedite candidates. Per problem (20 turns, about 420 candidate evaluations): about 10 s GPU + about 15 s CPU. 300 Family-A problems take about 1 GPU-hour + about 2 CPU-hours, far below the proposal's 30-150 CPU-hour GSM estimate.
- Family B at 101K stages / 811K arcs: 22 s GPU per turn, about 7.5 min per problem. At 202K stages / 1.62M arcs: about 44 s per turn by linear extrapolation from 6.3 s per 24 reps, about 15 min per problem. SLP warm re-solve is 26 s per pin candidate at 202K, so 8 per turn x 20 turns = about 70 CPU-min per problem, parallel across trace-cluster cores.
- About 120 Family-B problems take about 15-30 GPU-hours (3090 basis; H100 not measured, expected faster) + about 140 CPU-hours.

Ways to cut cost:
- At 100K scale the replicate SE is 0.003-0.010 against a Phi spread of 2-3, so use R=2-4 (halving GPU time).
- z moves skip the GSM re-solve.
- Screen on a 180-day horizon, then run the top 3 at 365 days.
- Track pipeline totals incrementally (the prototype's per-day pipe.sum over the ring buffer is wasteful).
- Cache SLP solutions keyed on the pin set.
- PhiEvaluator needs a batched evaluate(states[], seeds[]) API. The current per-replicate calls cannot use the GPU.

**risks.** SKEPTIC: Verified that the Willems scale numbers are exact (38 chains; 12/15/6/5 size bands; ch38 2,025 stages / 16,225 arcs, depth 3). Changes made:
- Corrected the claim that Data in Brief gives 'exact' references: gaps reach 25%, and the constraint count is 18,809.
- Corrected 'retail' metrics to demand-stage metrics, since 13 chains have no Retail stage.
- Replaced spanning-tree GW-DP local repair, which drops 88% of ch38's arcs, with a measured full-DAG SLP-GSM (0.12 s at 2K, 26-100 s at 200K).
- Found 4 bugs in the prototype simulator: a backlog-inflated fill metric; phantom source inventory that was 99.6% of reported cost; deterministic stage times; non-cumulative holding cost.
- Re-ran timing and extended it to 101K and 202K stages.
- Ran a CRN G2 probe: separation passes at 42-250 SE, but top candidates nearly tie, single-group moves are too small to reach feasibility in 20 turns, and pin and source-group candidates are often no-ops.
- Admitted that the real data is about 100-1,000x below DREAMPlace scale. Added a Family B at DREAMPlace scale (20K-200K+ stages) based on Wan & Hong (IJOC 2023, 1K-500K-node BOMs, no public code) and FreshRetailNet-50K (CC BY 4.0).
- Demoted InvNet-GSM to a baseline and fixed its citation.

Remaining risks:
(1) The engine is our own code and is not yet validated. No public simulator exists at 100K+ scale, and Wan & Hong's code is not released, so Family B fidelity rests on validation done at small scale.
(2) Family B topology is synthetic (composed from real chains). DREAMPlace's ISPD benchmarks are real netlists, and we cannot match that realism at scale.
(3) Licensing: Willems has no license, and the Data in Brief article is CC BY-NC, so do not redistribute and fetch at build time. M5 and Favorita licenses are unclear; avoid them.
(4) Mapping real series onto Willems mean/std demand, and FreshRetailNet's 90-day horizon (bootstrap needed), are our own constructions and must be documented.
(5) G2 near-ties among the top few candidates on homogeneous networks give noisy gold labels. Report the top-1 vs top-2 margin and prefer heterogeneous compositions.
(6) Turn-budget calibration: the initial gap must be closable within 20 turns. Coarse echelon moves are required.
(7) SLP is a local method, and its quality relative to Gurobi on Family A is unmeasured. Benchmark it against the Data in Brief incumbents.
(8) The 200K-scale CPU GSM cost (26 s per pin candidate) dominates wall time, so limit pin candidates per turn.
(9) H100 timings were not measured; all GPU numbers are from an RTX 3090.
(10) MABIM, HDPO and InventoryManagement.jl details beyond licenses were not re-verified.

**Candidates considered:**

| name | license | scale | runtime/eval | verdict |
|---|---|---|---|---|
| [Willems (2008) MSOM real-world multi-echelon chains dataset (38 chains)](https://doi.org/10.1287/msom.1070.0176 (mirror with per-chain degree/depth tables: https://github.com/LuisMoncayo/GuaranteedServiceTimeInventoryModel)) | No LICENSE file in the mirror. The Data in Brief article that publishes the solutions (PMC13320427) is CC BY-NC 4.0. The Zenodo record 10.5281/zenodo.8209331 says 'Other (Open)' with no further description. Use it for non-commercial research, fetch it at build time, and do not redistribute the .xls. | VERIFIED by summing the mirror's Data_In_Brief degree_depth_tables (graph_01..38.csv), and it matches Data in Brief Table 2 exactly. 38 chains, 8 to 2,025 stages. By size: 12 chains under 100 stages, 15 with 100-499, 6 with 500-999, and 5 with 1,000 or more. The five largest (stages/arcs): ch34 1,206/4,063; ch35 1,386/1,857; ch36 1,451/4,812; ch37 1,479/2,069; ch38 2,025/16,225 (820 Part, 87 Manuf, 559 Trans, 559 Retail; depth 3; 559 demand stages). The networks are shallow (maximum depth 1-9). 13 chains have NO Retail-class stage (03, 04, 06, 09, 10, 12, 13, 16, 19, 24, 33, 34, 35), so demand sits on sink stages of other classes; ch34's 53 demand stages are all Manuf. Compared with DREAMPlace, chain 38 is about 100x smaller than the smallest ISPD2005 design (adaptec1, about 211K cells) and about 1,000x smaller than bigblue4 (about 2.18M). Real topology does not reach DREAMPlace scale. | Not applicable (data). Reference solves: GNA returns in about 2 min on 28 chains; HGNA averages 88 s with a 1.1% average gap (worst 16.66%); TGNA averages 5.9 s with a 2.8% gap (Humair & Willems 2011, confirmed from the paper text). GNA took 577,190.78 s on the 2,025-node chain (InvNet README, confirmed). Data in Brief Gurobi branch-and-cut: the largest model has 4,050 variables and 18,809 constraints (not 18,890). Reported times range from 1.1 to 147.0 (units ambiguous in the extracted table; the text says 'more than 50 seconds ... over 100 seconds'). Incumbent-to-bound gaps are NON-ZERO on many chains: 18 is 25%, 33 is 25%, 32 is 24%, 28 is 20%, 29 is 17%, and 35/36 are 7%. These values are best-known upper bounds, not exact optima. | KEEP as Family A: real topology and heterogeneity, small scale. Use 'demand (sink) stage' metrics, not 'retail'. Treat the Data in Brief costs as best-known normal-demand GSM references with reference_class='bound'. |
| [Wan & Hong, 'Large-Scale Inventory Optimization: A Recurrent-Neural-Networks-Inspired Simulation Approach', INFORMS J. Computing 35(1):196-215 (2023)](https://arxiv.org/abs/2201.05868 ; https://pubsonline.informs.org/doi/10.1287/ijoc.2022.1253) | Paper only. No public code found: github.com/INFORMSJoC/2022.1253 returns 404 and the arXiv text has no code link. | This is the inventory-domain counterpart to DREAMPlace: tensorised and sparse simulation plus backprop gradients over BOM networks. The motivating real BOM (proprietary) has more than 500,000 nodes and more than 4,000,000 links. Tests use synthetic BOMs of 1K, 5K, 10K, 50K, 100K and 500K nodes with average degree 10 and 100 periods. | Table 1, sparse tensorised simulation, one replication, 2x Xeon Gold 6248R: 0.19 s at 1K nodes, 1.48 s at 10K, 13.9 s at 100K, 85.7 s at 500K. In TensorFlow, the GPU (RTX 3090) gave no advantage for sparse single-replication runs. Full two-stage optimisation took about 2 h at 50K nodes and about 1 day at 500K. | ADD as the scale and design template for Family B and as the published evidence that 10^5-10^6-node MEIO is a real industrial problem. Our batched GPU simulator gets its speed from batching candidates x replicates, which their single-replication setup did not exploit. |
| [InvNet-GSM (Huang, Yu, Yang; 'Optimizing large-scale inventory networks: An iterative decomposition approach', IISE Transactions, online 13 Oct 2025)](https://github.com/dy4om/InvNet-GSM (formerly durianh96/InvNet-GSM)) | MIT (verified via the GitHub API) | Methods: GW-DP, HGNA, PWL, DS, IMIP, SA, ID-SLP. Generators: edges_generating_by_type (trees) and edges_generating_by_shape (nodes_num, roots_num, sinks_num, maximum_relative_depth, skip_edges_num). No datasets are bundled. Instance sizes in the paper are still unverified (paywall). The volume/pages '58(8):982-996' were not confirmed. The README author order is Huang, Yu, Yang, not Yu, Huang, Yang. | Not verified. For comparison, I measured our own sequential-LP (DS-style) GSM using scipy-HiGHS on a synthetic DAG sized like chain 38 (2,025 stages / 16,225 arcs; LP with 4,050 vars and 18,250 rows): 0.04 s per LP, 3 iterations, 0.12 s cold, 0.07 s warm re-solve with a pin. At 20,250 stages / 162K arcs: 3.05 s cold, 1.10 s warm. At 202,500 stages / 1.62M arcs: 100 s cold (6 LPs, max 39 s each), 26 s warm, on one local CPU. | DEMOTE to a baseline/reference solver and to the source of the DAG generator. The inner loop does not need it: a roughly 60-line SLP on scipy-HiGHS covers the whole DAG in 0.1 s at 2K stages. |
| [stockpyl (Snyder)](https://github.com/LarrySnyder/stockpyl) | MIT (verified on PyPI) | Textbook instances. GSM DP covers serial and tree networks only. | Not measured. | Use only as a small tree/serial cross-check of the simulator and of the SLP GSM. Do NOT use it for 'local repair on a spanning subtree': on chain 38 a spanning tree keeps 2,024 of 16,225 arcs (12%), so the repair ignores 88% of the SI_j >= S_i constraints. |
| [FreshRetailNet-50K (Dingdong)](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K (paper arXiv:2505.16319)) | CC BY 4.0 | 50,000 store-product series from 898 stores in 18 cities, covering 863 perishable SKUs. About 90 days (Mar-Jun 2024), hourly sales, with hourly stockout annotations. About 4.85M rows. | Not applicable (data). | ADD as the permissively licensed real-demand source for the tens of thousands of demand nodes in Family B; the 898-store / 18-city structure maps naturally to store -> city DC -> central DC tiers. Caveat: the horizon is only about 90 days, so the 465-day simulation needs block bootstrap. |
| [MABIM / ReplenishmentEnv (MSRA)](https://github.com/VictorYXL/ReplenishmentEnv (arXiv:2306.07542)) | Apache-2.0 (verified) | 2,778 real SKUs with daily demand, configs sku50..sku2000 x single_store/2_stores/3_stores. I did not re-verify the serial-echelon and 200k-capacity details beyond the license. | Not reported. | SECONDARY: about 3 years of daily real demand for Family A demand stages (more history than FreshRetailNet) and a serial-chain validation target. |
| [M5 (Walmart) and Favorita (Kaggle) demand data](https://www.kaggle.com/competitions/m5-forecasting-accuracy ; https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting) | UNCLEAR. Both are governed by Kaggle competition rules, and the TIB registry lists no license for M5. | M5: 30,490 item-store series over 1,941 days. Favorita: about 210K store-item pairs (54 stores). | Not applicable. | Optional for internal experiments only. Do not ship derived data until the license is cleared. Prefer FreshRetailNet-50K and MABIM. |
| [HDPO Neural_inventory_control](https://github.com/MatiasAlvo/Neural_inventory_control) | MIT | Up to about 56 locations (not re-verified). | Not reported. | Design reference only. |
| [InventoryManagement.jl (Perez, CMU)](https://github.com/hdavid16/InventoryManagement.jl) | MIT | General networks with BOM; small instances. | Not reported. | Independent oracle for validating the simulator on chains of 100 stages or fewer. |
| [MARO / OR-Gym / gym-invmgmt / DynaPlex / SupplyGraph / Amazon miniSCOT](https://github.com/microsoft/maro ; https://github.com/hubbs5/or-gym ; https://github.com/r2barati/gym-invmgmt ; https://github.com/DynaPlex/DynaPlex ; https://huggingface.co/datasets/azminetoushikwasi/SupplyGraph ; https://github.com/amzn/supply-chain-simulation-environment) | MIT / MIT / MIT / proprietary engine / MIT / Apache-2.0 | All are toy-scale (9 nodes or fewer, or no inventory network at all). | n/a | Reject. Rejections unchanged. |
| [Batched PyTorch MEIO simulator prototype (ours), re-run and audited](C:/Users/xuwen/AppData/Local/Temp/claude/C--Users-xuwen-designbench-llm-finetune/1c3b6209-a2ab-49bb-a719-64621aa93b14/scratchpad/meio_sim_bench.py (+ meio_g2_probe_v2.py, gsm_slp_bench.py)) | Ours | Re-measured on the local RTX 3090 (torch 2.4.1, 465 days): - 2,025 stages / 15,914 arcs: 1.107 s per 480 reps, 0.443 s per 24. - 10,125 / 80,841: 5.62 s per 480. - NEW 101,250 / 810,932: 3.20 s per 24 reps and 12.4 s per 96. - NEW 202,500 / 1,622,197: 6.31 s per 24 reps (263 ms per replicate). That is DREAMPlace-order node counts, and it fits easily in 24 GB. | A G2 probe step (21 candidates x 8 CRN reps, one batched call) takes 0.47-0.57 s at 2,025 stages and 22.0-22.1 s at 101,250 stages (RTX 3090). H100 not measured. | Build it out as the evaluation engine after fixing bugs (a)-(d) and adding finite-capacity source/supplier stages and stochastic stage times. Validate before generating any data. |

**Sources:** https://github.com/LuisMoncayo/GuaranteedServiceTimeInventoryModel (Data_In_Brief/instances_structure/degree_depth_tables/graph_01..38.csv summed locally) · https://pmc.ncbi.nlm.nih.gov/articles/PMC13320427/ (Table 2: stages, arcs, demand stages, variables, constraints, gap %, time; CC BY-NC 4.0) · https://zenodo.org/records/8209331 · https://seanwillems.com/wp-content/uploads/2020/11/Humair_Willems_OR_v59n3_MayJun2011.pdf (local text: GNA about 2 min on 28 chains; HGNA 88 s / 1.1%; TGNA 5.9 s / 2.8%) · https://github.com/dy4om/InvNet-GSM and https://api.github.com/repos/dy4om/InvNet-GSM (MIT, pushed 2025-08-15, 577,190.78 s GNA note) · https://www.tandfonline.com/doi/abs/10.1080/24725854.2025.2561557 · https://arxiv.org/abs/2201.05868 (Wan & Hong; Tables 1-4, 7-8: 1K-500K nodes, avg degree 10, real BOM >500K nodes / >4M links) · https://pubsonline.informs.org/doi/10.1287/ijoc.2022.1253 · https://api.github.com/repos/INFORMSJoC/2022.1253 (404: no public code) · https://pypi.org/pypi/stockpyl/json (1.0.2, 2025-05-02, MIT) · https://github.com/VictorYXL/ReplenishmentEnv (Apache-2.0) · https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K (CC BY 4.0) · https://arxiv.org/abs/2505.16319 (FreshRetailNet-50K: 50,000 series, 898 stores, 18 cities, 863 SKUs) · https://service.tib.eu/ldmservice/dataset/m5-competition-dataset (no license listed) · https://arxiv.org/abs/2609.08071 (LLM inventory-policy design, lost-sales, small scale) · https://arxiv.org/abs/2502.11213 (C3 AI large-scale inventory, no public scale/code) · Local measurements (RTX 3090, torch 2.4.1, scipy 1.14.1 HiGHS): scratchpad/meio_sim_bench.py, meio_g2_probe_v2.py, g2_costsplit.py, gsm_slp_bench.py, gsm_slp_bench_s.py, willems_stats.txt in C:/Users/xuwen/AppData/Local/Temp/claude/C--Users-xuwen-designbench-llm-finetune/1c3b6209-a2ab-49bb-a719-64621aa93b14/scratchpad/

## structures

**Domain:** structures: large-scale 3D topology and zone-level material-layout design of real structural parts (bridge girder segments, bracing, roof supports, brackets) on structured voxel grids. Native evaluation is 0.5-2M elements, the ISPD2005 cell-count range; final verification is 4-8M. Code-checked building frames stay as a below-scale runner-up.

**simulator.** Large-scale 3D density-based structural design on structured voxel grids. The native evaluation resolution is 0.5-2M elements (one element is one design variable). That matches ISPD2005, whose largest case bigblue4 has 2,177,353 cells; DREAMPlace places it in about 235 s. Final verification runs at 4-8M on the final design only. The literature anchors go to 27-160M. Production engine: our own GPU matrix-free hex8 SIMP code in PyTorch or CuPy, re-implemented from the published Traff 2023 and Aage 2015 algorithms:
- node-based 81-point structured stencil, gather only, so it is deterministic and needs no atomics;
- Galerkin-coarsened geometric MG, 4-5 levels, with the coarse operators assembled;
- Chebyshev or Jacobi smoothing and a direct solve on the coarsest level;
- CG with FP32 storage and FP64 accumulation, so the FP64-weak drc RTX PRO 6000 is still usable;
- block solves over all load cases;
- variable-coefficient PDE filter, Heaviside projection, MMA with Z+1 constraints.
The port must pass three acceptance tests before production: (a) compliance within 1% of TopOpt_in_PETSc on 3 shared instances; (b) MG converges to 1e-5 in 200 iterations or fewer on 20 optimized, nearly black-and-white designs with Emin = 1e-6 (arXiv 2604.26441 shows naive FP32 GMG fails exactly here); (c) bitwise-reproducible metrics. Bridge engine: TopOpt_in_PETSc, ported to PETSc 3.26 and run on trace-cluster CPU nodes at 0.26-0.5M el, used for the G1/G2 pilot and later for cross-checks on 10% of problems. Physics: linear elasticity, multiple load cases, von Mises stress. Buckling is dropped.

**benchmarks.** Anchor instances, checked:
- TopOpt_in_PETSc paper: MBB 113.8M el, cantilever 27.6M el, Qatar roof support 3.1M el with a passive roof slab, torsion box 26.1M and 82.1M el.
- Traff 2023: cantilever 65.5M el, MBB 41M el.
- SGLDBench: cantilever 159M and 130M el; Bone 30M and 380M DOF.
- TOP3D_XL Femur, Molar and GE-bracket voxel domains (download link expires November 2026).
- SimJEB: 4 bracket load cases and 381 human designs.
These serve as templates and cross-checks. The training set is procedurally generated from them at the native resolution (see problem_generator). Unlike ISPD, there is no community-standard large TO contest suite. We publish our generated instances with seeds and keep a fixed held-out set of 30 problems at 2M el as the DREAMPlace-style benchmark.

**decision layer.** One turn is one Candidate(i,p,v) through the factor ladder. Elements have stable indices, 30-90 per problem.
(a) Z = 16-64 zones (4x2x2 to 8x4x2 boxes clipped to the design domain), each with two params:
- vf_cap in [0, 0.6], the zone's material budget. It is an inequality constraint inside the inner solver, so mass is measured, not assumed to be saturated; zones off the load path can stay under their cap.
- rmin_scale in [0.5, 3], the local minimum member size through the variable-coefficient filter.
(b) L = 1-6 load cases, each with weight w in [0.05, 20] in the weighted-compliance inner objective, moved by log factors 0.5/0.8/1.25/2. This is the analogue of net weighting.
(c) P = 8 or fewer PERSISTENT pad slots with pad_radius in [0, 6h]. An empty slot is bound to the current k-th hotspot when the turn starts. Once its radius is above 0 it keeps that xyz, so index i means the same place for the rest of the trajectory. In the original refreshed-hotspot design the meaning of index i changed every turn.
(d) One global design knob: base filter radius rmin0 in [1.5h, 4h], i.e. the minimum feature size.
Removed from the agent: projection beta, now an internal fixed continuation schedule. If beta were a lever while gray fraction was a goal, 'raise beta' would become the gold first move in nearly every trajectory, which degrades G1 and G2. The SIMP penalty stays fixed at p = 3.
Candidate set per turn: a domain-specific enumerator, because dataxp.enumerate_candidates subsamples 24 elements at random and would usually miss the critical zones. It ranks all legal moves by the first-order dPhi from the anchor's adjoints: zone shadow prices from the MMA multipliers, plus adjoint sensitivities of the p-norm stress and the deflection summed per zone. It keeps the top 12, adds 6 random legal moves for coverage and 1 null move (no change). hint() returns the zone holding the largest p-norm stress contributor with vf_cap 'up', or the load case with the worst deflection ratio with w 'up'.
Why this is one meaningful, observable decision: each move reallocates material budget, length scale or load priority between regions. The solver re-optimises all 0.5-2M densities, and the simulator measures the resulting mass, stress, deflection and connectivity. This is the same split as DREAMPlace region density targets and net weights over a global placer.

**inner solver.** evaluate(st, seed) starts from the state's anchor density, which is converged and kept on the GPU. It applies the knob and runs K = 20 MMA iterations (move limit 0.1, beta frozen at the anchor's continuation stage). It then applies a volume-preserving Heaviside threshold, choosing eta so the projected volume equals the physical volume, and computes all metrics on that design. A plain eta = 0.5 threshold would let projection artefacts change the mass.
On commit, the chosen state runs to convergence (||dx||inf < 0.01 or 60 iterations) before it becomes the next anchor. This keeps the null move's dPhi close to 0, so each candidate's dPhi is caused by its knob and not by leftover solver progress. The original proposal's anchors were not converged, which mixed the two. QA: log the null-move drift each turn, and drop a turn when |dPhi_null| exceeds 25% of the gap between the top and median candidates.
The 19 candidates of a turn are evaluated in one batched GPU call. This needs a new Domain.evaluate_batch hook, because PhiEvaluator._sample in dataxp/domain.py currently loops sequentially.
Stochastic variant: the design is optimised under nominal loads, and the seed only perturbs the evaluation loads (+-15 deg direction, 10% CoV magnitude). Those responses come from superposing 3 unit-direction solves per load point, so the 4-24 replicates that PhiEvaluator requests cost about 3L extra linear solves per state instead of 4-24 full SIMP evaluations.
The solver sets all 1e6-1e7 densities; the agent never touches voxels.

**metrics objective constraints.** Objective: minimise mass_kg = rho * h^3 * (number of solid voxels) on the volume-preserving projected design, with pads included.
Constraints, as goals:
- minimum_fos_yielding = 1.5, with FOS = f_y / sigma_PN. sigma_PN is the p-norm (P = 8) of element-centroid von Mises stress over solid elements, excluding pad, support and load-patch elements, under factored ULS loads (1.35G + 1.5Q). G is a fixed superimposed dead load. Self-weight is off by default, because design-dependent self-weight makes compliance non-monotone and needs a modified interpolation. The raw maximum and the 99.9th percentile are reported as well, but not constrained: the raw maximum on a staircase voxel boundary depends on the mesh and is dominated by single elements.
- maximum_deflection_ratio = 1.0, the maximum over load cases of u_max,l / (span/250 to span/500) under unfactored SLS loads.
- maximum_n_components = 1: solid voxels are 6-connected and every load patch and support lies in that component; computed with cupyx.scipy.ndimage.label.
Validity envelope, enforced through DesignProgram.is_valid and not as a goal:
- gray fraction (0.1 < x < 0.9) of 0.05 or less before thresholding;
- MG relative residual below 1e-5 within 200 iterations;
- compliance of the thresholded design within 10% of the continuous one.
Reported only: per-load-case compliance J (N m), u_max (mm), and zone mass shares.
Dropped: linearised buckling. Void regions produce spurious modes, the cost on a coarse grid was not verified, and a 64k-element projection cannot represent slender members.

**observation summary.** About 1.5-2.5k tokens.
- Header: family; domain size in m; grid and element count (for example 192x96x48 = 884,736); material; load-case table with patch centre, direction, magnitude and current weight; supports; keep-out boxes; goal limits.
- State: mass (t) and vf; FOS from the p-norm stress, with the raw maximum, the location and zone of the worst site, and VIOLATED flags; deflection ratio per load case; component count; the change since the previous turn, plus the null-move drift.
- Zone table: the top 10 zones by strain-energy share, each with bbox, vf_cap, used vf, rmin_scale, SE%, p-norm stress ratio, and marginal compliance value (the MMA multiplier, which an engineer would recognise as a shadow price).
- Pad table: the persistent slots with xyz, radius and local stress ratio, then the top-5 open hotspot sites.
- Three ASCII density maps, max-projected along x, y and z at about 48x12 characters, with zone boundaries marked.
- The last action and its measured effect.

**problem generator.** make_problem(index, rng) uses stable_seed(index). Family = index mod 5:
1. bridge girder segment: 3-6 lane-load cases, bearings, traffic-envelope keep-out;
2. facade or core bracing: story wind and ELF lateral patterns, passive floor slabs, window keep-outs;
3. roof support: passive roof slab, Qatar-style;
4. GE-style bracket: SimJEB 4 load cases, passive bolt bosses, TOP3D_XL bracket voxel domain;
5. classic cantilever, MBB or torsion box.
Native-resolution ladder: 0.26M (pilot only), 0.5M, 1M and 2M el, with 4-8M for final verification only. Randomised: aspect ratios, support layout, load magnitudes and directions, 0-5 keep-out boxes, material (S355, 6061-T6 or Ti-6Al-4V), and the deflection limit (span/250 to span/500).
Initial state: uniform zone caps at 0.6-0.8x the total from a coarse 64k-el uniform-budget bisection, with unequal load weights. Infeasibility is asserted at the NATIVE resolution. If the initial state is feasible, the generator tightens the limits by 0.85x until it is not.
reference() is a uniform-zone, equal-weight volume bisection (class 'heuristic'). It is solved with the SAME GPU pipeline at the SAME native resolution and with the same metric definitions, so obj/ref compares like with like. A full-resolution PETSc reference would mix discretisations. PETSc runs only as a cross-check on 10% of problems.
G1 check before scaling up: across 50 pilot problems, the coefficient of variation of the initial violations and the reference mass should exceed 0.3, and every family should appear in the top-Phi actions.

**cost budget.** Checked anchors:
- 0.21 s per SIMP iteration at 512k el and 0.38 s at 1M, single load case, fused FP32 Jacobi-PCG on an RTX 4090. Jacobi time grows faster than linearly: 3.0 s/iter at 2M.
- FP32 GMG: 0.18 s per solve at 512k on a uniform density.
- Traff: 65.5M el at about 74 s/iter on an A100.
- PETSc and OpenMP CPU: 1.4-2.3e-4 core-s per element per iteration for one load case.
Estimates, unverified until our port is benchmarked:
- One 0.5M-el eval with L = 3 load cases and K = 20 takes about 12-27 s on an H100 (0.15-0.3 s per solve x 3 loads x 20 iterations, plus filter, MMA, projection, stress and connectivity). The proposal's 5-10 s ignored the L solves per iteration.
- MG hierarchy memory is about 4-5 GB per 0.5M-el eval, not under 2 GB (8.66 GiB was checked at 1M), so 8-16 evals fit on an 80 GB H100.
- A turn of 19 evals as one batch takes about 2-4 min on one H100 if batching gains 2-3x, plus 0.5-1 min of anchor convergence.
- A problem of 20 turns takes about 0.75-1.5 H100-h. Add about 0.2 h for reference() (8 warm-started bisection steps of about 100 iterations) and about 0.15 h for one 4M-el final verification.
- Total: about 1.2-2 H100-h per problem, or 350-600 H100-h for 300 problems, plus about 30% for pilots and failures. The proposal said 100-150 H100-h.
The proposal's per-turn verification at 4M el would add about 5 min x 20 turns = 100 min per problem, so it is dropped. Instead, a coarse-to-native rank correlation is measured on 20 problems.
Further cuts:
- Two-stage screening: 0.26M el with K = 10 for all 19 candidates, then 0.5M el with K = 20 for the top 6. This is about 2.5x cheaper if the Spearman correlation between stages is 0.8 or higher.
- Reuse of the MG hierarchy across candidates, since only one zone's densities change.
- FP32 matvecs.
CPU route with TopOpt_in_PETSc: about 1 core-h per 0.5M-el eval with L = 3, so about 400 core-h per problem and about 120k core-h for 300 problems (extrapolated). The proposal said 30-40k. Use the CPU route only for the pilot of about 10 problems (about 4k core-h on the trace cluster) and for cross-checks.

**risks.** SKEPTIC: Most scale and licence claims hold. Checked from primary sources: the PETSc paper figures, the Traff 2023 timings, SGLDBench, SimJEB, and the licences and last-push dates of all repos. Corrections:
(1) Cost is 2.5-4x higher than proposed, at 350-600 H100-h for 300 problems, because the proposal omitted the per-load-case solves. Per-turn 4M-el verification is dropped, and the PETSc CPU route is about 120k core-h, not 30-40k.
(2) The GPU port is 3-6 weeks, not 1-2. The companion arXiv 2604.26441 shows FP32 GMG stalling on the high-contrast optimized densities that warm-started anchors produce, so the port needs Galerkin coarsening with FP64 accumulation, and the G2 pilot uses a PETSc bridge engine in the meantime.
(3) Beta was removed from the levers and gray fraction moved from the goals into the validity envelope; otherwise 'raise beta' would become the gold first move in nearly every trajectory.
(4) Anchors are converged at commit, and a null move each turn makes dPhi attributable to the knob.
(5) Hotspot pads are now persistent slots so element indices are stable.
(6) FOS uses the p-norm, not the mesh-dependent raw maximum.
(7) Mass uses a volume-preserving threshold.
(8) reference() runs at the native resolution with the same pipeline.
(9) Buckling and design-dependent self-weight are dropped by default.
(10) The stochastic variant uses load superposition so 4-24 replicates stay cheap.
(11) TO-Master is closed-source. The TOP3D_XL data link expires November 2026. torch-fem (MIT) and TACS (Apache-2.0, but sub-scale) were added as candidates.
Remaining risks:
(a) G2 depends on how SIMP behaves. Pilot criterion on 10 problems at 0.26-0.5M el: the gap between the top and median candidate exceeds 3x the self-consistency noise in at least 80% of turns, where the noise is |Phi(K=20) - Phi(K=40)| of the top candidate plus the null-move drift. If this fails, raise K, or switch to 'apply knob and converge' evaluations at roughly 2x cost.
(b) Coarse-to-native ranking: require a Spearman correlation of 0.7 or higher between 0.5M and 2M Phi ranks on 20 problems before training on screening-only labels.
(c) Shadow-price prefiltering may make the gold action predictable from the observation's zone table. That is acceptable for SFT but should be reported, and the 6 random candidates guard coverage.
(d) Zone caps can be traded off against each other only through the inner solver. If one family's gold trajectories reduce to 'raise the cap of the hotspot zone', G1 diversity drops, so check per-family action entropy.
(e) Stress singularities at re-entrant corners and load patches remain. Pads and pad exclusion mitigate them; report max, 99.9th percentile and p-norm.
(f) The 3D field summary loses information, and the LLM may misread zone geometry; keep the zone bounding boxes explicit.
(g) No member-level AISC or Eurocode checks. If the owner requires code checks, the OpenSees building domain is sub-scale by 2-4 orders and does not meet the DREAMPlace-level requirement on its own.
(h) The H100 hour budget on Bridges-2 GPU-shared must be confirmed before the 300-problem run.
(i) TO grids are regular, so the scale matches DREAMPlace but the combinatorial structure differs from a netlist hypergraph. The difficulty lies in allocating material across multiple load cases and constraints, and the method section should say so.

**Candidates considered:**

| name | license | scale | runtime/eval | verdict |
|---|---|---|---|---|
| [TopOpt_in_PETSc (DTU; Aage, Andreassen, Lazarov, SMO 51:565, 2015)](https://github.com/topopt/TopOpt_in_PETSc) | LGPL-2.1 (lesser.txt at the repo root; the GitHub API reports license=null, checked 2026-10-01). Running it as a separate program puts no obligation on our code or data. | Checked in the paper PDF. MBB 1008x336x336 = 113.8M elements / 343.8M DOF, 928 iterations, 1800 cores, 4h32m. Cantilever 480x240x240 = 27.6M el / 83.8M DOF, 1000 iterations on 144 cores (Xeon 5650). Qatar roof support 512x64x86 = 3.1M el on 72 cores, with a passive roof slab and Heaviside continuation. Torsion box 26.1M el on 1000 cores and 82.1M el on 2000 cores; the 82.1M run took 2000 iterations in about 4.5 h. | Checked: 30-60 s per design iteration at 27.6M el on 144 cores of 2010-era Xeons, which is about 2.3e-4 core-s per element per iteration for one load case. Extrapolated, not measured: about 1e-4 core-s on modern cores, so a 0.5M-el candidate eval (K=20 iterations, 3 load cases) costs about 1 core-hour. | Keep as the reference and cross-validation engine. Also promote it to the BRIDGE engine: run the G1/G2 pilot on it at 0.26-0.5M el on trace-cluster nodes before the GPU port exists. At about 400 core-h per problem it is too slow to produce the full dataset. |
| [topoptlib (TopOpt_in_PETSc wrapped in Python; Smit et al., SMO 2021)](https://github.com/thsmit/TopOpt_in_PETSc_wrapped_in_Python) | LGPL-2.1 (checked) | Same framework as TopOpt_in_PETSc; the wrapper's own instance sizes were not checked | Not checked; expected to match TopOpt_in_PETSc | Use for STL-domain voxelisation and quick prototyping of the bracket and roof families. Stale since 2023, so pin PETSc 3.14. Not the production engine. |
| [DTU GPU codes: TopOpt-in-OpenMP-GPU / -Multi-GPU / futtop, plus the CPU TopOpt-in-OpenMP (Traff, Rydahl, Karlsson, Sigmund, Aage; CMAME 410:116043, 2023)](https://github.com/topopt/TopOpt-in-OpenMP-GPU) | No license in TopOpt-in-OpenMP-GPU or TopOpt-in-OpenMP (GitHub API license=null, checked), so all rights are reserved. The paper itself is CC-BY-4.0, which means the algorithm can be re-implemented from the paper but the code cannot be vendored. | Checked in the paper PDF. Cantilever 65.5M el: 100 design iterations in 7380 s on one A100-80GB with Futhark (about 74 s/iter). MBB 41M el in 4 h on an A100. Nonlinear Neo-Hookean cases. Memory limits Futhark to about 65M el on an 80 GB A100. | Checked from the paper, first 20 iterations at 65.5M el, 4 MG levels: OpenMP-GPU 2300 s, Futhark 2096 s, OpenMP-CPU on 2x Xeon 8160 (48 cores) 3800 s. With 5 levels the GPU codes take 1105-1264 s. Important: below 4M elements the 48-core CPU code was faster than the GPU offload code. The CPU cost is about 1.4e-4 core-s per element per iteration, which matches the PETSc figure. | Use as the algorithm template for our port: Galerkin-coarsened geometric MG-CG with mixed precision. Do not vendor it without the authors' permission. The finding that the CPU code beat the GPU code below 4M elements shows that GPU benefit at 0.5M el depends on batching many candidates per GPU. |
| [Yang, Wang, Wang (Santa Clara U.) fused matrix-free 3D SIMP: arXiv 2604.18020 (Jacobi-PCG) and the companion arXiv 2604.26441 (matrix-free Galerkin GMG)](https://arxiv.org/abs/2604.18020) | Papers are CC BY-NC-ND 4.0; no code repository is given in either paper | Cantilever 216k-4.9M el; torsion 499,125 el; the GMG paper covers 64k-1M el (3.09M DOF) | Checked, 120 SIMP iterations with fused FP32 Jacobi-PCG: 512k el 24.8 s (0.21 s/iter), 1M 45.4 s (0.38 s/iter), 2M 358 s (3.0 s/iter), 4.9M 997 s (8.3 s/iter). Time grows faster than linearly because Jacobi needs a few hundred CG iterations per step, and in the torsion case 108-110 of 120 steps hit the 1000-iteration cap. With FP32 GMG: 0.18 s per solve at 512k on a uniform density. On optimized, heterogeneous density fields it converged within 200 iterations in only 1/9 cases at 512k (4/9 at 216k). The 1M-el MG hierarchy takes 8.66 GiB. | Not a tool. It is the strongest evidence that the hard part of our port is MG robustness on high-contrast, nearly black-and-white designs, which is exactly the state of a warm-started anchor. Kernel speed is the easier part. Use it as the speed target (about 0.2 s/iter at 0.5M) and as the failure-mode checklist. |
| [JAX-FEM (Xue et al., CPC 291:108802, 2023)](https://github.com/deepmodeling/jax-fem) | GPL-3.0 (checked). Copyleft applies only on redistribution; generating data internally is unaffected. | Checked in the abstract: 3D tensile problem with 7.7M DOF, about 10x faster than a commercial code. TO examples are small or nonlinear; TO timing at 1M el not found. | Not checked at 1M-el TO scale | Fallback for stress-constrained prototypes on unstructured STL domains, and the precedent for an LLM acting on TO knobs. Generic sparse assembly is less efficient than structured matrix-free MG at 1e6-1e7 elements. |
| [torch-fem (Nils Meyer)](https://github.com/meyer-nils/torch-fem) | MIT (checked) | Checked from its README benchmark: an elastic cube with 1.5M DOF, forward solve 1.417 s and backward 1.216 s on an RTX 5090 in FP64, 15 GB peak VRAM. It ships a 3D jet-engine-bracket TO example. | About 1.4 s per solve at 0.5M hex el (1.5M DOF). Jacobi-CG scales badly beyond that, as arXiv 2604.18020 shows. | New candidate. Permissive building block for unit tests of our port (element matrices, stress recovery) and for the STL bracket family at 0.5M el or below. Not the 1e6-1e7 engine. |
| [TOP3D_XL + SGLDBench (Wang, Bukenberger, Niedermayr, Neuhauser, Wu, Westermann 2025)](https://github.com/PSLer/TOP3D_XL) | BSD-2-Clause code (checked) | Checked in SGLDBench (arXiv 2501.03068). Cantilever 860x430x430 = 159M el; high-resolution TO at 800x400x400 = 130M el / 386M DOF. Bone 384x256x512 (30M DOF), and at high resolution 848x576x1200 (~380M DOF). The Femur, Molar and GE-bracket domains are an external download whose shared link is valid only until November 2026. | Checked: on a 6-core Xeon W2235, the 159M-el FEA solve takes about 30 min and a 130M-el TO iteration about 45 min. On the GPU (RTX 2070, 8 GB) the solve takes 91 s against 610 s on the CPU, with GPU capacity limited to about 45M el. | Source of instances and cross-validation only. Download the Femur, Molar and GE-bracket voxel domains NOW, before the link expires. |
| [SimJEB (Simulated Jet Engine Bracket dataset; Whalen et al., CGF 2021)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XFUWJG) | Open Data Commons Attribution per the paper; the original CAD is limited to non-commercial use under GrabCAD terms | Checked: 629 GrabCAD entries downloaded, 441 cleaned, 411 meshed, 381 simulated. 4 load cases (vertical, horizontal, diagonal, torsion); Ti-6Al-4V (E = 113.8 GPa); first-order tets of about 2 mm | n/a (dataset) | Real industrial load-case template for the bracket family, plus a set of human designs to compare against when reporting. |
| [OpenSees / OpenSeesPy (with archetype building models and AISC-LRFD frame benchmarks)](https://github.com/OpenSees/OpenSees) | Checked in COPYRIGHT: UC Regents; free for educational, research and non-profit noncommercial use, other entities internal use only, commercial distribution needs UC OTL permission. The Elkady SMF archetype models are GPL-3.0. | Building models have 1e2-1e4 elements; the AISC sizing benchmarks have 168-3860 members | Not checked | Runner-up only. It does not meet the DREAMPlace-scale requirement, since the variable count is 2-4 orders of magnitude below it. |
| [TACS (Toolkit for the Analysis of Composite Structures; Kennedy et al.)](https://github.com/smdogroup/tacs) | Apache-2.0 (checked) | The uCRM wingbox benchmark has 10,285 shell elements and 156,826 DOF (Brooks et al., AIAA J 2018). The TACS README also claims large 3D TO capability but gives no numbers. | Not checked | New candidate, rejected. Its licence and fidelity are good, but its released real-structure benchmarks (uCRM) are about 1e5 DOF, below DREAMPlace scale. |

**Sources:** https://github.com/limbo018/DREAMPlace · https://www.researchgate.net/publication/220915345_The_ISPD2005_placement_contest_and_benchmark_suite · https://github.com/topopt/TopOpt_in_PETSc · https://api.github.com/repos/topopt/TopOpt_in_PETSc · https://www.topopt.mek.dtu.dk/-/media/subsites/topopt/apps/dokumenter-og-filer-til-apps/petsc-1-.pdf · https://github.com/thsmit/TopOpt_in_PETSc_wrapped_in_Python · https://api.github.com/repos/thsmit/TopOpt_in_PETSc_wrapped_in_Python · https://github.com/topopt/TopOpt-in-OpenMP-GPU · https://api.github.com/repos/topopt/TopOpt-in-OpenMP-GPU · https://api.github.com/repos/topopt/TopOpt-in-OpenMP · https://github.com/topopt/TopOpt-in-OpenMP-Multi-GPU · https://backend.orbit.dtu.dk/ws/portalfiles/portal/317991078/1_s2.0_S0045782523001676_main.pdf · https://www.sciencedirect.com/science/article/pii/S0045782523001676 · https://hgpu.org/?p=28174 · https://arxiv.org/abs/2604.18020 · https://arxiv.org/html/2604.18020 · https://arxiv.org/html/2604.26441v1 · https://github.com/deepmodeling/jax-fem · https://api.github.com/repos/deepmodeling/jax-fem · https://arxiv.org/abs/2212.00964 · https://arxiv.org/html/2607.01812v1 · https://github.com/meyer-nils/torch-fem · https://github.com/PSLer/TOP3D_XL · https://api.github.com/repos/PSLer/TOP3D_XL · https://arxiv.org/html/2501.03068 · https://arxiv.org/html/2105.03534 · https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XFUWJG · https://www.nature.com/articles/nature23911 · https://anaconda.org/conda-forge/petsc · https://github.com/conda-forge/petsc-feedstock/blob/main/recipe/build.sh · https://github.com/OpenSees/OpenSees · https://raw.githubusercontent.com/OpenSees/OpenSees/master/COPYRIGHT · https://pypi.org/project/openseespy/ · https://github.com/amaelkady/OpenSEES_Models_SMF · https://arxiv.org/pdf/2205.07274 · https://github.com/smdogroup/tacs · https://arc.aiaa.org/doi/10.2514/1.J056603 · https://docs.mfem.org/4.8/ex37_8cpp_source.html · C:/Users/xuwen/designbench_llm_finetune/dataxp/domain.py
