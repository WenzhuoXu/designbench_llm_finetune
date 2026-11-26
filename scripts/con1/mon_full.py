"""Training dynamics of a running con1 full-set job, read from the node's /tmp (run through srun --overlap by
mon_full.sh). Prints one summary block and FLAGS (empty when nothing looks wrong).

    python mon_full.py /tmp/con1_full_<jobid>/run [--export /project/community/wxu2/con1/full/run1/export]

Columns: loss and informative-token nll (mean of the last 20 steps against the 20 before), dev loss and dev
informative nll at each dev pass (stage 1's 120 dev rows: dev_start.json is the stage-1 adapter before any
continuation step), grad norm (median, max of the last 20), seconds per step (last 20) and the ETA of the epoch,
peak memory with the step's longest row, lr, the newest checkpoint on the node and the newest export.
Flags: non-finite loss or grad norm; grad norm of a recent step over 5x the run's median; mean loss of the last 20
steps above the 20 before by more than 10%; dev loss up at two dev passes in a row; peak memory over 76 GiB; step
time of the last 20 over 1.3x the 20 before; no new step for 20 minutes.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys
import time


def main():
    run = sys.argv[1]
    export = sys.argv[sys.argv.index("--export") + 1] if "--export" in sys.argv else None
    rows = {}
    try:
        for line in open(os.path.join(run, "metrics.jsonl")):
            try:
                r = json.loads(line)
                rows[r["step"]] = r                      # a step trained again overwrites its earlier line
            except Exception:                            # noqa: BLE001
                pass
    except FileNotFoundError:
        print("no metrics yet in", run)
    ms = [rows[k] for k in sorted(rows)]
    flags = []
    try:
        cfg = json.load(open(os.path.join(run, "config.resolved.json")))
        steps_total = cfg.get("steps")
    except Exception:                                    # noqa: BLE001
        steps_total = None
    out = [time.strftime("%F %T") + "  " + run]
    if ms:
        last = ms[-1]
        L = [m["loss"] for m in ms]
        I = [m.get("nll_info_token") for m in ms]
        G = [m["grad_norm"] for m in ms]
        T = [m.get("t_train", 0.0) for m in ms if "dev_loss" not in m]
        a, b = L[-20:], L[-40:-20]
        out.append("step %d/%s | loss last20 %.4f (prev20 %s) | info nll last20 %s | lr %.2e" % (
            last["step"] + 1, steps_total, st.mean(a), ("%.4f" % st.mean(b)) if b else "-",
            ("%.4f" % st.mean([x for x in I[-20:] if x is not None])) if any(x is not None for x in I[-20:]) else "-",
            last.get("lr", float("nan"))))
        out.append("grad norm median %.3f, last20 max %.3f | step %.0f s (last20), prev20 %s s" % (
            st.median(G), max(G[-20:]), st.mean(T[-20:]) if T else 0, ("%.0f" % st.mean(T[-40:-20])) if T[-40:-20] else "-"))
        pk = max(ms, key=lambda m: m.get("peak_mem_gib") or 0)
        out.append("peak memory max %.2f GiB at a step whose longest row is %s tokens; last step %.2f GiB / %s tokens" % (
            pk.get("peak_mem_gib") or 0, pk.get("max_row_tokens"), last.get("peak_mem_gib") or 0, last.get("max_row_tokens")))
        if steps_total and T:
            left = (steps_total - last["step"] - 1) * st.mean(T[-20:])
            out.append("ETA of the epoch: %.1f h (%s)" % (left / 3600, time.strftime("%F %H:%M", time.localtime(time.time() + left))))
        dv = [m for m in ms if "dev_loss" in m]
        try:
            d0 = json.load(open(os.path.join(run, "dev_start.json")))
            out.append("dev before the continuation: loss %.4f, info nll %s" % (d0["dev_loss"], d0.get("dev_nll_info_token")))
        except Exception:                                # noqa: BLE001
            d0 = None
        for m in dv[-6:]:
            out.append("  dev at step %d: loss %.4f, info nll %s" % (m["step"] + 1, m["dev_loss"], m.get("dev_nll_info_token")))
        if not all(math.isfinite(x) for x in L + G):
            flags.append("non-finite loss or grad norm")
        if len(G) > 20 and max(G[-20:]) > 5 * st.median(G):
            flags.append("grad norm spike: %.2f vs median %.2f" % (max(G[-20:]), st.median(G)))
        if b and st.mean(a) > 1.1 * st.mean(b):
            flags.append("training loss up >10%% over the last 20 steps (%.4f vs %.4f)" % (st.mean(a), st.mean(b)))
        if len(dv) >= 3 and dv[-1]["dev_loss"] > dv[-2]["dev_loss"] > dv[-3]["dev_loss"]:
            flags.append("dev loss up at two dev passes in a row")
        if (pk.get("peak_mem_gib") or 0) > 76:
            flags.append("peak memory %.1f GiB" % pk["peak_mem_gib"])
        if T[-40:-20] and st.mean(T[-20:]) > 1.3 * st.mean(T[-40:-20]):
            flags.append("steps slower by >30%")
        try:
            age = time.time() - os.path.getmtime(os.path.join(run, "metrics.jsonl"))
            if age > 1200:
                flags.append("no new step for %.0f min" % (age / 60))
        except OSError:
            pass
    ck = sorted(glob.glob(os.path.join(run, "ckpt", "step_*")))
    out.append("node checkpoints: %s" % ([os.path.basename(c) for c in ck][-2:]))
    if export:
        try:
            ex = json.load(open(os.path.join(export, "state.json")))
            out.append("export: step %s (%s)" % (ex["step"], time.strftime(
                "%F %T", time.localtime(os.path.getmtime(os.path.join(export, "state.json"))))))
        except Exception:                                # noqa: BLE001
            out.append("export: none yet")
    out.append("FLAGS: " + ("; ".join(flags) if flags else "none"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
