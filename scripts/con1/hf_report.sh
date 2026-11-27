#!/bin/bash
# hf_report.sh LAB : per-family and overall held-out-family numbers of LAB (zero-shot vs no-change; sequential vs its own
# zero-shot answers and the copiers, positions >= 1), each family scored on its own so it gets its own bootstrap se.
# CPU only (facets.py, no harness). Prints one table; writes full_eval/hf/report_LAB.txt.
set -uo pipefail
L=${1:?label}
C=/storage/wxu/con1; A=$C/arm; H=$C/full_eval/hf
Q=$A/main/queries_v2.jsonl
PY=/opt/anaconda3/bin/python
cd $C/repo
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p $H/fam
for f in inventory2 molopt thermal2; do
  $PY scripts/con1/facets.py --queries $Q --preds $H/preds_$L.jsonl --pass zero --groups heldout_family --variants $f \
    --boot 1000 --out $H/fam/zero_${L}_$f.txt --json $H/fam/zero_${L}_$f.json > /dev/null 2>&1 &
  $PY scripts/con1/facets.py --queries $Q --preds $H/preds_$L.jsonl $H/preds_${L}_zero_as_seq.jsonl --pass seq --min-qi 1 \
    --kshot-baselines --groups heldout_family --variants $f --boot 1000 \
    --out $H/fam/seq_${L}_$f.txt --json $H/fam/seq_${L}_$f.json > /dev/null 2>&1 &
done
wait
$PY - $H $L <<'EOF' | tee $H/report_$L.txt
import json, os, re, sys
H, L = sys.argv[1:]
M = ("regime_phys", "cpl_dir_BA", "cpl_size")
def means(path):
    """{predictor: {metric: (value, se)}} and [(a, b, {metric: (diff, se)})] from a facets txt's primary block"""
    g, p = {}, []
    txt = open(path).read()
    blk = txt.split("group means over all (primary)", 1)[1]
    gm, pd = blk.split("paired differences over all (primary)", 1)
    for line in gm.splitlines()[1:]:
        t = line.split()
        if len(t) > 2 and t[1] == "select":
            g[t[0]] = {t[k]: tuple(float(x) for x in t[k + 1].replace("+-", " ").split()) for k in range(1, len(t) - 1, 2)
                       if "+-" in t[k + 1]}
    for line in pd.split("paired differences over all")[0].splitlines()[1:]:
        m = re.match(r"\s+(\S+) - (\S+)\s+(.*)", line)
        if m:
            t = m.group(3).split()
            p.append((m.group(1), m.group(2), {t[k]: tuple(float(x) for x in t[k + 1].replace("+-", " ").split())
                                               for k in range(0, len(t) - 1, 2) if "+-" in t[k + 1]}))
    return g, p
def readable(jpath, lab):
    d = json.load(open(jpath))
    vals = [v.get("unreadable") for k, v in d.items() if k.startswith(lab + "/") and isinstance(v, dict)]
    vals = [v for v in vals if v is not None]
    return 1 - sum(vals) / len(vals) if vals else float("nan")
fmt = lambda x: "%.3f+-%.3f" % x
rows = [("all", os.path.join(H, "facets_%s_zero.txt" % L), os.path.join(H, "facets_%s_seq_vs_zero.txt" % L),
         os.path.join(H, "facets_%s_zero.json" % L))]
for f in ("inventory2", "molopt", "thermal2"):
    rows.append((f, os.path.join(H, "fam", "zero_%s_%s.txt" % (L, f)), os.path.join(H, "fam", "seq_%s_%s.txt" % (L, f)),
                 os.path.join(H, "fam", "zero_%s_%s.json" % (L, f))))
print("held-out families (queries_v2 group heldout_family): %s, facetm; se by bootstrap over states" % L)
for fam, z, s, zj in rows:
    gz, pz = means(z)
    gs, ps = means(s)
    print("== %s   (scorer-readable answers, zero-shot: %.3f)" % (fam, readable(zj, L)))
    print("  zero-shot, all positions:   " + "  ".join("%s %s" % (m, fmt(gz[L][m])) for m in M))
    print("  no-change:                  " + "  ".join("%s %s" % (m, fmt(gz["nochange"][m])) for m in M))
    d = {(a, b): v for a, b, v in pz}
    print("  zero-shot - no-change:      " + "  ".join("%s %+.3f+-%.3f" % ((m,) + d[(L, "nochange")][m]) for m in M if m in d[(L, "nochange")]))
    print("  sequential (pos >= 1):      " + "  ".join("%s %s" % (m, fmt(gs[L][m])) for m in M))
    print("  same questions, zero-shot:  " + "  ".join("%s %s" % (m, fmt(gs[L + "_zero"][m])) for m in M))
    for c in ("copy_last", "copy_mode"):
        print("  %-27s " % (c + ":") + "  ".join("%s %s" % (m, fmt(gs[c][m])) for m in M))
    d = {(a, b): v for a, b, v in ps}
    best = max(("copy_last", "copy_mode"), key=lambda c: gs[c]["select"][0])
    print("  sequential - zero-shot:     " + "  ".join("%s %+.3f+-%.3f" % ((m,) + d[(L, L + "_zero")][m]) for m in M if m in d[(L, L + "_zero")]))
    print("  sequential - best copier (%s): " % best + "  ".join("%s %+.3f+-%.3f" % ((m,) + d[(L, best)][m]) for m in M if m in d[(L, best)]))
EOF
