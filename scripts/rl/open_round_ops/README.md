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

Sampling shares (2026-11-23): fixed, disjoint shares instead of two machines meeting in the middle.

- `regrid.sh` (Orchard login node): drc's 4,440 grid2op states split at file index D=2220; the Orchard advanced job
  (OUT grid2op_drc_rev) restarted with its states file cut to [D, 4440) and a KV budget of 330,000 per server (200,000
  left 60% of the cache empty).
- `drc_stop_at.sh` + `drc_cover_upto.py` (drc, setsid): stop drc's sampler once every state below D has 8 replies, cut
  drc's states file to its share, rerun `sample_open.sh` for the last scoring pass and the rows.
- `split_g2e.py` + `resplit_g2e.sh` (Orchard login node): the general-GPU job on Orchard's own grid2op + engibench
  states restarted with SAMPLE_LIMIT=1200 (its next 1,200 states); the other 2,257 written to a states file for a
  second job (preempt, 2 GPUs).
- `after_g2r.sh` (Orchard login node): when the grid2op share job ends, that second job moves to the advanced GPUs.
- `merge_open_rows.py`: all shares' gain rows in one file, one row per state (first file wins), each row checked
  (message structure, one action line, rationale present and at most 1,000 words, the family's served system prompt
  as in the replay rows, tier).
- `drc_tail.sh` + `drc_frontier.py`: when the general GPU frees, drc's last grid2op states [D2, 2220) go to a job on
  it (back to front, OUT grid2op_drc_tail) and drc's stop watcher is restarted with D2.
- `after_rows.sh` (Orchard login node; `DRY=1` merges and mixes into `open_round/dry_merge` only): merge, mix (replay
  1 per 2 new rows, dev 1.2% of problems), training from dx38 on advanced, then the benchmark under the open rules
  (weak three families on general, the other eight on advanced, both afterok on the training).
- `photonics_rows_kind.py`: what the picked photonics moves did (rule-breaking pixels before/after); `row_shape.py`:
  the message layout of a rows file.
- `drc_eval_open.sh` (drc GPU 4, setsid): a benchmark pair under the open rules with the Orchard eval settings (same
  problem files, harness and instruction by sha1); used for a third pair of dx38 runs on the weak families during training.
- `submit_dx38_weak_b.sh`: the second pair of dx38 runs on the weak families (Orchard general, during training).
