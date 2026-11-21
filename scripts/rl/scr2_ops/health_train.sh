#!/bin/bash
# health_train.sh   (login node): one line on the training / evaluation step of job 173954.
# exit 0 fine; 10 needs a look; 12 the training ended (first time only); 13 the evaluation ended
J=173954; H=$HOME/rl/logs/fcswap
R=/project/community/wxu2/rl/runs/scr2_dx38mtR_fc_20261121
E=/home/wxu2/rl/eval/scr2_fc_20261121
ST=$(squeue -h -j $J -o %T 2>/dev/null)
STEPS=$(squeue -h -s -j $J -o %j 2>/dev/null | tr '\n' ' ')
M=$(tail -n 1 "$R/metrics.jsonl" 2>/dev/null | python3 -c 'import json,sys
try:
    m = json.loads(sys.stdin.read())
    print("step %s loss %.4f%s %s" % (m.get("step"), m.get("loss"), (" dev %.4f" % m["dev_loss"]) if "dev_loss" in m else "", m.get("time")))
except Exception as e:
    print("no metrics yet")')
DEV=$(grep -h '"dev_loss"' "$R/metrics.jsonl" 2>/dev/null | python3 -c 'import json,sys
print(" ".join("%s:%.4f" % (json.loads(l)["step"], json.loads(l)["dev_loss"]) for l in sys.stdin))')
EP=$(ls "$E"/*.episodes.jsonl 2>/dev/null | head -n 1); NE=0; [ -n "$EP" ] && NE=$(wc -l < "$EP")
echo "$(date '+%m-%d %H:%M') job=${ST:-gone} steps=[$STEPS] | $M | dev: $DEV | eval episodes $NE | $(grep -v cgroup "$H/train_eval.log" | tail -n 1 | cut -c1-90)"
grep -q "evaluation rc=" "$H/train_eval.log" && { echo "EVALUATION ENDED"; exit 13; }
if grep -q "training rc=" "$H/train_eval.log" && [ ! -e "$H/train.seen" ]; then touch "$H/train.seen"; echo "TRAINING ENDED: $(grep 'training rc=' "$H/train_eval.log" | tail -n 1)"; exit 12; fi
[ "$ST" = RUNNING ] || { echo "ATTENTION: job=${ST:-gone}"; exit 10; }
case " $STEPS " in *" fctrain "*) ;; *) echo "ATTENTION: no training/evaluation step"; exit 10 ;; esac
exit 0
