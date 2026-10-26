# ops/ — cluster scripts for the datav4 corpus (copies, for backup)

These scripts ran on the clusters and existed only there; they are copied here so a cluster wipe
(Orchard `/project/community` is cleared every two weeks) cannot lose them. The live copies are:

| dir | live location | what |
|---|---|---|
| `harness_v3/` | Orchard `/project/community/wxu2/corpus/harness/scripts/`, drc `/storage/wxu/corpus/harness/scripts/` | evaluation harness generalized to native-verb and verb-surface domains (`dataxp_policy_eval_v3.py`, `dx_eval_v3.sh`, `dx_eval_orchard.sh` one vLLM server per GPU, `make_problems_v3_native.py` evaluation set = last 60 by index of each native family) |
| `orchard_sampling/` | Orchard `/project/community/wxu2/corpus/scripts/` | own-action row sampling on Orchard: `serve_one.sh` (vLLM), `sample_job*.sh` (shards, K replies per state, resumable), `shard.py` / `reshard.py` / `merge_shards.py` / `check_samples.py`. `sample_job.sh` binds fixed ports 8766+i; the per-job-port versions used from b004 on live in each batch's own scripts dir (e.g. `/project/community/wxu2/corpus/b004/scripts/sample_job_b004.sh`, drc `/home/wxu/corpus_work/b0NN/`) |
| `drc_pipeline/` | drc `/storage/wxu/corpus/` | batch-1 build: env build, state selection, scoring + tiers (`bw_score.sh`), Φ-cache repair (`fix_phi_cache.sh`), packaging (`package_all.sh`); `tools/` = flag-file generators for native evaluation overlap |

The library code they drive is `dataxp/` (Move contract, native domains, verb surfaces, spellings,
`ownrows/`, `package_datav4.py`); the design and status record is `docs/corpus_native_domains.md`,
`docs/datav4_append_plan.md`, `docs/dataxp_spellings.md`, `docs/dataxp_verb_surface.md`.
