# Open-rules round: operations scripts (2026-11-22)

The round under `dataxp_policy_eval_v3r.py --open-rules` (refused moves reported back; grid2op blackouts undone;
photonics fabrication-rule moves applied, design invalid until repaired). Copies of the scripts that ran it.

- `submit_collect_g2e_rev.sh` (Orchard general): grid2op + engibench collection on the training pool, back to front,
  while drc collected the same list front to back (`collect_open.sh` on drc); `coverage.py` tells when the two meet.
- `submit_sample_photonics.sh` (Orchard advanced): states of the photonics collection (OPEN_EPISODES), 8 replies per
  state at T=0.8, scoring alongside, gain rows, through `scripts/rl/orchard_screen_data.sh`.
- `sample_open.sh` (drc GPU 4): the same steps for drc's share (vLLM server, open_states, sample_states with a KV
  budget, `scripts/rl_screen/score_loop.sh` alongside, last scoring pass, build_rows).
- `check_lex.py`: the photonics ranking fix (score_states.py, commit ca66454) checked on real benchmark states.
