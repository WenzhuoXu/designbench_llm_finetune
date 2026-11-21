# Operations scripts of the forecast-checked round (2026-11-20/21)

One-off helpers used to produce the round's data on Orchard job 173954 and on drc. They carry that job's id and
paths and are kept as a record of what was run, not as reusable tools.

- `node_start.sh`, `swap_run.py`, `start_tuned.sh`, `stop_tuned.sh`: the launcher's own sampler is stopped with SIGSTOP
  (the job and its servers stay up) and the same command is run as a job step with other arguments.
- `score_loop.sh`, `complete_states.py`, `start_score.sh`, `run_env.py`: scoring of finished states alongside sampling.
- `merge_append.py`, `split_states.py`, `subset_states.py`, `exclude_states.py`, `merge_restart.sh`, `submit_g1.sh`:
  replies sampled on other GPUs (drc, a one-GPU job on the general queue) merged into the main file; shares of the
  states given to each sampler.
- `clean_samples.py`, `dups.py`, `why_missed.py`, `reread.py`, `cmp_gate.py`: integrity checks. The compute node read
  the first record appended by another node as garbage (NFS page cache), so the last pass sampled two replies again;
  `clean_samples.py` wrote the file with one reply per state and sample.
- `finish_rows.sh`, `mix.sh`, `pack.sh`: last scoring pass, rows, training files, archives for the copy off the pool.
- `kill_servers.sh`, `train_eval.sh`, `start_train_eval.sh`: training, then evaluation, as a step of the same job.
- `rate.py`, `metrics_window.py`, `stats.py`, `cache_test.py`, `cache_test2.py`, `health.sh`, `health_train.sh`,
  `watch_orch.sh`, `watch_train.sh`, `peek_rows.py`, `handoff.sh`: measurements and watchers.
