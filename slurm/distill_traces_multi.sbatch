#!/bin/bash
#SBATCH --job-name=distill_multi
#SBATCH --partition=RM-shared
#SBATCH --account=mch250030p
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Several search traces per problem, varying the constraint price alpha.
# alpha sets how hard the potential trades mass against violation, so each
# setting traces a different point on that curve -- diversity the policy can
# learn from, rather than one deterministic path per problem.
set -euo pipefail
module load anaconda3/2024.10-1
conda activate my_env
PROJECT_DIR="/ocean/projects/mch250030p/wxu7/llm_finetune"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:/ocean/projects/mch250030p/wxu7/DesignBench:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore

for A in 2.0 5.0 10.0; do
  python scripts/distill_search_traces.py \
    --split-file data/splits/truss_v1_auto.json --split train \
    --alpha "$A" --max-steps 20 --workers 16 \
    --out "results/distill/search_alpha${A}.jsonl"
done

python - <<'PY'
import json, glob, random
rows=[]
for f in sorted(glob.glob('results/distill/search_alpha*.jsonl')):
    a=f.split('alpha')[1].replace('.jsonl','')
    for line in open(f):
        r=json.loads(line); r['trace_id']=f"{r['problem_id']}_alpha{a}"; r['strategy_type']=f"lookahead_alpha{a}"
        rows.append(r)
# de-duplicate identical trajectories (different alpha can pick the same path)
seen=set(); uniq=[]
for r in rows:
    key=(r['problem_id'], tuple(m['content'] for m in r['messages'] if m['role']=='assistant'))
    if key in seen: continue
    seen.add(key); uniq.append(r)
rng=random.Random(20260823); rng.shuffle(uniq)
by_pid={}
for r in uniq: by_pid.setdefault(r['problem_id'], []).append(r)
dev_pids=set(sorted(by_pid)[:6])          # hold out whole PROBLEMS, not traces
train=[r for r in uniq if r['problem_id'] not in dev_pids]
dev=[r for r in uniq if r['problem_id'] in dev_pids]
for name, part in (("search_policy_train_sft.jsonl", train), ("search_policy_dev_sft.jsonl", dev)):
    with open(f"results/distill/{name}","w") as fh:
        for r in part: fh.write(json.dumps(r)+"\n")
turns=lambda P: sum(len([m for m in r['messages'] if m['role']=='assistant']) for r in P)
print(f"raw={len(rows)} unique={len(uniq)}  train={len(train)} traces / {turns(train)} turns  "
      f"dev={len(dev)} traces / {turns(dev)} turns  (dev problems held out whole)")
PY
