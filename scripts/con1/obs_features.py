"""con1: the first-order estimates of a move's effect that the observation itself shows, as features for the tree
baseline (owner, 11-25: give the surface-feature control the same information the observation shows).

Families whose observation carries a per-element estimate a move reads directly (checked 11-25 in dataxp/domains/):
  photonics  adjoint map s(x, y) = d target_loss / d density (block sums + top-6 pixels each way). Feature
             adj_gain = PhotonicsDomain._gain(g, pat, new): first-order DEcrease of target_loss for the pixels the
             move flips (g = _sensitivity(fx, pat), new = _materialise(fx, pat, move)); plus pixels added / removed.
  engibench  per-cell field S (strain-energy share, |grad T|^2 or optical intensity; "material added or removed where S
             is large changes the field most"; top-8 cells each way + a coarse map). Features: sum and mean of S over
             the cells the move fills, and over the cells it clears; counts.
  circuit2   what-if rows computed by the simulator: one-tunable x1.1 sensitivities (DC gain dB / crossover % / phase
             margin deg) per tunable, the other compensation schemes, the cascode and nulling-resistor toggles.
             Features: the move's tunable sensitivity triple scaled by log(new / old) / log(1.1); for a scheme or toggle
             move the numbers of its what-if text (first 8).
  Checked and not given: frame3d2, grid2, seir2, traffic2, grid2op (their observations list per-element state —
  member forces, line flows, queues — but no per-move effect estimate); held-out families get no trees.

    python obs_features.py --code C --harness V3RC --problems P [P ...] --states S [S ...] [--rows rows.jsonl]
        --out feats.jsonl [--variants photonics engibench circuit2] [--workers 12]
One line per state: {sid, variant, feats: {move text: {feature: value}}}. Resumable by sid.
"""
import argparse
import gzip
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_ENV = None
_SPECS = None


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def circuit2_feats(st):
    """Parsed from the served prompt (no simulator)."""
    u = st["user"]
    sens = {}
    m = re.search(r"Sensitivity, one tunable x1\.1[^:]*:\s*(.*)", u)
    if m:
        for part in m.group(1).split(";"):
            mm = re.match(r"\s*(\w+)\s+(%s|n/a)/(%s|n/a)/(%s|n/a)" % (NUM, NUM, NUM), part)
            if mm:
                sens[mm.group(1)] = [float(x) if x != "n/a" else float("nan") for x in mm.group(2, 3, 4)]
    schemes = {}
    m = re.search(r"Other schemes at these sizes:\s*(.*)", u)
    if m:
        for part in m.group(1).split(";"):
            mm = re.match(r"\s*comp (\d+)\s*(\w*)\s*->\s*(.*)", part)
            if mm:
                nums = [float(x) for x in re.findall(NUM, mm.group(3))][:8]
                schemes[int(mm.group(1))] = nums
                if mm.group(2):
                    schemes[mm.group(2)] = nums
    lines = {}
    for tag in ("5 RZ", "6 CAS"):
        mm = re.search(r"^\s*%s:\s*(.*)$" % tag, u, re.M)
        if mm:
            lines[tag] = [float(x) for x in re.findall(NUM, mm.group(1))][:8]
    lab = {(0, "W"): "W1", (0, "L"): "L1", (1, "W"): "W2", (1, "L"): "L2", (2, "B"): "B", (3, "Cc"): "Cc",
           (4, "Ib"): "Ib"}
    out = {}
    for m in st["moves"]:
        a = (m.get("args") or {}).get("args") or {}
        i, p = a.get("i"), a.get("p")
        f = {}
        key = lab.get((i, p))
        if key and key in sens and m["args"].get("v_old"):
            try:
                r = math.log(float(a["v"]) / float(m["args"]["v_old"])) / math.log(1.1)
                for j, nm in enumerate(("sens_gain_db", "sens_xover_pct", "sens_pm_deg")):
                    f[nm] = sens[key][j] * r
            except (ValueError, ZeroDivisionError, TypeError):
                pass
        nums = None
        if (i, p) == (3, "comp"):
            v = a.get("v")
            try:
                nums = schemes.get(int(round(float(v))))
            except (TypeError, ValueError):
                nums = schemes.get(str(v))
        elif i == 5:
            nums = lines.get("5 RZ")
        elif i == 6:
            nums = lines.get("6 CAS")
        for j, x in enumerate(nums or []):
            f["whatif_%d" % j] = x
        out[m["move"]] = f
    return out


def sim_feats(job):
    code, harness, st, spec = job
    global _ENV
    if _ENV is None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from conseq_lib import Env
        _ENV = Env(code, harness)
    import numpy as np
    from conseq_lib import rebuild
    try:
        s = rebuild(_ENV, {"variant": st["variant"], "t": st["t"], "turn_log": st["turn_log"]}, spec)
        d = s.domain
        out = {}
        if st["variant"] == "photonics":
            fx, pat = s.st["fx"], s.st["pat"]
            g = d._sensitivity(fx, pat)
            for m in st["moves"]:
                mv = s.parse(m["move"])
                new = d._materialise(fx, pat, mv) if mv is not None else None
                if new is None:
                    out[m["move"]] = {}
                    continue
                out[m["move"]] = {"adj_gain": float(d._gain(g, pat, new)), "px_added": int((new & ~pat).sum()),
                                  "px_removed": int((pat & ~new).sum())}
        elif st["variant"] == "engibench":
            S = np.asarray(d._analysis(s.st)["proxy"], float)
            rho = s.st["rho"] > 0
            for m in st["moves"]:
                mv = s.parse(m["move"])
                ch = d.clone(s.st)
                try:
                    d.apply(ch, mv)
                except Exception:                          # noqa: BLE001 (refused: no flipped cells)
                    out[m["move"]] = {}
                    continue
                new = ch["rho"] > 0
                add, rem = new & ~rho, rho & ~new
                out[m["move"]] = {"S_fill_sum": float(S[add].sum()), "S_clear_sum": float(S[rem].sum()),
                                  "S_fill_mean": float(S[add].mean()) if add.any() else 0.0,
                                  "S_clear_mean": float(S[rem].mean()) if rem.any() else 0.0,
                                  "n_fill": int(add.sum()), "n_clear": int(rem.sum())}
        return st["sid"], st["variant"], out, None
    except Exception as e:                                 # noqa: BLE001
        return st["sid"], st["variant"], {}, "%s: %s" % (type(e).__name__, str(e)[:200])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--rows", default=None, help="only the states these con1 rows use (plus every queries state)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=["photonics", "engibench", "circuit2"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    specs = {}
    for p in a.problems:
        for line in opener(p):
            r = json.loads(line)
            s = r.get("spec", r)
            specs[s["problem_id"]] = s
    keep = None
    if a.rows:
        keep = set()
        for line in opener(a.rows):
            r = json.loads(line)
            if r.get("tier") == "con1_conseq":
                keep.add(r["sid"])
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["sid"])
            except ValueError:
                pass
    jobs, n_c2 = [], 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for p in a.states:
            try:
                for line in opener(p):
                    st = json.loads(line)
                    if st["variant"] not in a.variants or st["sid"] in done:
                        continue
                    if keep is not None and st.get("split", "test") != "test" and st["sid"] not in keep:
                        continue
                    if st["variant"] == "circuit2":
                        fo.write(json.dumps({"sid": st["sid"], "variant": "circuit2", "feats": circuit2_feats(st)}) + "\n")
                        n_c2 += 1
                        continue
                    if "turn_log" not in st:
                        continue
                    jobs.append((a.code, a.harness, st, specs[st["problem_id"]]))
                    done.add(st["sid"])
            except EOFError:
                pass
        print("circuit2 states %d (parsed); simulator states to do %d" % (n_c2, len(jobs)), flush=True)
        from multiprocessing import Pool
        n = n_err = 0
        with Pool(a.workers) as pool:
            for sid, v, feats, err in pool.imap_unordered(sim_feats, jobs, chunksize=1):
                n += 1
                if err:
                    n_err += 1
                    print("!!! %s: %s" % (sid, err), flush=True)
                fo.write(json.dumps({"sid": sid, "variant": v, "feats": feats, "error": err}) + "\n")
                fo.flush()
                if n % 50 == 0:
                    print("%d/%d states, %d errors" % (n, len(jobs), n_err), flush=True)
    print("OBS_FEATURES_DONE")


if __name__ == "__main__":
    main()
