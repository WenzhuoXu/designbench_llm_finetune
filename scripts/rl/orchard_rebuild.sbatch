#!/bin/bash
# Rebuild everything the dataxp RL pipeline needs on Orchard after a /project/community wipe
# (the wipe of 2026-11-02 removed the harness, all envs and the model snapshot). Idempotent: each
# piece is skipped when its marker exists. Runs on a compute node (they reach PyPI and Hugging Face);
# the four pieces build in parallel.
#
#   rl/envs/train    trainer env, Orchard's former d27o: torch 2.9.1+cu128, transformers 5.17.0,
#                    peft 0.21.0, trl 1.13.0, accelerate 1.15.0, flash-linear-attention 0.5.2, triton 3.8.0
#                    (+ causal-conv1d 1.7.0 if a prebuilt wheel exists; there is no nvcc on the nodes)
#   rl/envs/vllm     vLLM 0.29.0 + transformers 5.19.0 (Orchard's former vllm029)
#   rl/envs/sim      the simulator env for rollouts: drc corpus env's exact pins (scripts/rl/corpus_env_lock.txt),
#                    installed one by one with --no-deps, as Orchard's former corpus_o was
#   rl/hf            Qwen/Qwen3.8-27B at revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 (the same snapshot)
#
# Persistent inputs (code, harness copy, problem files) live in $HOME/rl and in git, not here.
#   sbatch scripts/rl/orchard_rebuild.sbatch      (from $HOME/rl/repo)
#SBATCH -J rl_rebuild
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH -A ccm
#SBATCH -t 4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /home/wxu2/rl/logs/rl_rebuild_%j.out
set -uo pipefail
RL=/project/community/wxu2/rl
REPO=${REPO:-$HOME/rl/repo}
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export PIP_CACHE_DIR=$RL/cache/pip TMPDIR=$RL/cache/tmp PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$RL/envs" "$RL/hf" "$PIP_CACHE_DIR" "$TMPDIR" "$RL/logs" "$HOME/rl/logs"
PY312=$(command -v python3.12) || { echo "ABORT: no python3.12"; exit 2; }
echo "=== $(date '+%F %T %Z') host=$(hostname) job=${SLURM_JOB_ID:-none} python3.12=$PY312"
curl -sf -o /dev/null https://pypi.org/simple/ || { echo "ABORT: no PyPI from $(hostname)"; exit 3; }

build_train() {
  local E=$RL/envs/train
  [ -f "$E/.done" ] && { echo "[train] done already"; return 0; }
  rm -rf "$E"; $PY312 -m venv "$E" && local P=$E/bin/python
  $P -m pip install -q -U pip wheel setuptools packaging ninja
  $P -m pip install -q --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1"
  $P -m pip install -q "transformers==5.17.0" "peft==0.21.0" "trl==1.13.0" "accelerate==1.15.0" \
      "flash-linear-attention==0.5.2" "fla-core==0.5.2" safetensors "omegaconf==2.3.0" "hydra-core==1.3.2" \
      pyyaml pytest numpy scipy "torch==2.9.1"
  $P -m pip install -q --no-build-isolation "causal-conv1d==1.7.0" 2>&1 | tail -2 || echo "[train] causal-conv1d: no wheel (fallback kernel)"
  # last: causal-conv1d's dependency resolution puts back torch 2.9.1's own triton 3.5.1, and fla
  # refuses Triton < 3.7.1 on Hopper (wrong gated chunk_bwd_dqkwg gradients)
  $P -m pip install -q --no-deps "triton==3.8.0"
  $P -m pip freeze > "$E/freeze.txt"
  $P -c "import torch, transformers, peft, fla, triton; print('[train]', torch.__version__, transformers.__version__, peft.__version__, triton.__version__)
assert triton.__version__ == '3.8.0', triton.__version__" \
    && touch "$E/.done"
}

build_vllm() {
  local E=$RL/envs/vllm
  [ -f "$E/.done" ] && { echo "[vllm] done already"; return 0; }
  rm -rf "$E"; $PY312 -m venv "$E" && local P=$E/bin/python
  $P -m pip install -q -U pip wheel setuptools
  $P -m pip install -q "vllm==0.29.0" "transformers==5.19.0" "huggingface_hub[cli]"
  $P -m pip freeze > "$E/freeze.txt"
  $P -c "import vllm, torch; print('[vllm]', vllm.__version__, torch.__version__)" && touch "$E/.done"
}

build_sim() {
  local E=$RL/envs/sim
  [ -f "$E/.done" ] && { echo "[sim] done already"; return 0; }
  rm -rf "$E"; $PY312 -m venv "$E" && local P=$E/bin/python
  $P -m pip install -q -U pip
  local bad=0
  while read -r pin; do
    case "$pin" in ''|\#*) continue ;; esac
    $P -m pip install -q --no-deps "$pin" >/dev/null 2>&1 || { echo "[sim] FAILED $pin"; bad=$((bad+1)); }
  done < "$REPO/scripts/rl/corpus_env_lock.txt"
  $P -m pip freeze > "$E/freeze.txt"
  $P -c "import numpy, scipy, grid2op, lightsim2grid, ceviche, ceviche_challenges, rdkit, engibench; print('[sim] numpy', numpy.__version__, 'grid2op', grid2op.__version__)" \
    && touch "$E/.done"
  echo "[sim] pins that failed: $bad"
}

fetch_model() {
  local M=$RL/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/$REV
  [ -f "$RL/hf/.done" ] && { echo "[model] done already"; return 0; }
  while [ ! -f "$RL/envs/vllm/.done" ]; do sleep 30; done          # hf CLI comes with the vLLM env
  HF_HOME=$RL/hf HF_HUB_CACHE=$RL/hf/hub HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "$RL/envs/vllm/bin/hf" download Qwen/Qwen3.8-27B --revision "$REV" > "$RL/logs/hf_download.log" 2>&1
  [ -f "$M/config.json" ] && echo "[model] $(du -shL "$M" | cut -f1) at $M" && touch "$RL/hf/.done"
}

build_train > "$RL/logs/build_train.log" 2>&1 &  P1=$!
build_vllm  > "$RL/logs/build_vllm.log"  2>&1 &  P2=$!
build_sim   > "$RL/logs/build_sim.log"   2>&1 &  P3=$!
fetch_model > "$RL/logs/fetch_model.log" 2>&1 &  P4=$!
wait $P1 $P2 $P3 $P4
for f in build_train build_vllm build_sim fetch_model; do echo "--- $f"; tail -4 "$RL/logs/$f.log"; done
ok=1; for m in envs/train/.done envs/vllm/.done envs/sim/.done hf/.done; do [ -f "$RL/$m" ] || { echo "MISSING $m"; ok=0; }; done
cp "$RL"/envs/*/freeze.txt "$HOME/rl/logs/" 2>/dev/null; for e in train vllm sim; do cp "$RL/envs/$e/freeze.txt" "$HOME/rl/logs/freeze_$e.txt" 2>/dev/null; done
echo "RL_REBUILD_DONE $(date '+%F %T') ok=$ok"
[ $ok = 1 ]
