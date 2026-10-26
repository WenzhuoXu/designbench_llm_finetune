# dataxp — multi-domain data generation with Φ gates

Implements the portfolio in `docs/multi_domain_expansion.md`: many domains
spanning constraint coupling (complexity), problem size, and stochasticity,
all sharing one potential (Φ_v2 as a `DesignProgram`) and one generation
pipeline. Self-contained: Python ≥ 3.8, numpy + scipy only (runs on trace's
`anaconda3/2023.03-1`).

## Layout

| file | role |
|---|---|
| `potential.py` | vendored Φ_v2 + chance-constraint aggregation + auto feasibility offset |
| `domain.py` | `Domain` contract, candidate enumeration, `Move` (verb + typed args; `SET_PARAM` is the Candidate, native-verb domains bring their own - `docs/dataxp_move_contract.md`; verb surfaces over v2 - `docs/dataxp_verb_surface.md`; the `DATAXP_SPELLING` switch between the `multitool` and `set_parameter` spellings of the same moves - `docs/dataxp_spellings.md`), `PhiEvaluator` (E[Φ] scoring, adaptive replicates, reliability gate) |
| `ladder.py` | random / greedy / blind / phi_select / phi_d2 policies + envelope red-team |
| `gates.py` | G1 distribution-variance, G2 phi-measurable, G3 data-quality |
| `harvest.py` | preference pairs (failure states included) + replay-verified trajectories |
| `run_generation.py` | per-variant job entrypoint |
| `smoke.py` | contract + end-to-end smoke test per variant |
| `domains/` | one module per domain; registry in `domains/__init__.py` |

## Variants (13)

`seir(_stoch)`, `grid(_stoch)`, `thermal(_stoch)`, `traffic(_stoch)`,
`inventory` (intrinsically stochastic), `circuit(_stoch)`, `frame3d(_stoch)`.

## The gates

Data enters the dataset only when all three pass (per variant), and only for
problems G2 includes:

- **G1 distribution variance** — no duplicate specs, ≥3 populated families,
  CV of initial violation ≥ 0.25, CV of initial objective ≥ 0.10, ≥50%
  infeasible starts, difficulty terciles populated.
- **G2 Φ-measurable** — per visited state: top-2 Φ margin outside the noise
  band (2σ, replicate-based for stochastic domains) and Φ non-constant across
  candidates. Problem passes at ≥60% of states; variant passes at ≥80% of
  non-degenerate problems. Φ-constant-at-start problems are degenerate
  (truss-mechanism analog) and excluded outright.
- **G3 data quality** — pair yield ≥ 70% (a pair is emitted only when its Φ
  gap clears the joint noise band and both sides are envelope-valid), zero
  NaN pairs, ≥95% of harvested trajectories replay-verified under identical
  seed streams.

## Running

```bash
python -m dataxp.smoke all
python -m dataxp.run_generation --variant seir --quick --workers 1 \
    --tmp <scratch>/dataxp_tmp --out <scratch>/dataxp_out
# full generation (one SLURM job per variant; see docs/multi_domain_expansion.md)
python -m dataxp.run_generation --variant seir --n 130 --workers 16 \
    --tmp /trace/scratch/biosimmlab/wxu2/dataxp_tmp/run1 \
    --out /trace/home/wxu2/dataxp_out/run1
```

Everything intermediate goes to `--tmp` (scratch); only `report.json` and the
gated `accepted/` dataset land in `--out`.
