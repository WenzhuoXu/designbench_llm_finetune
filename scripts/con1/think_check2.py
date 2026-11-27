"""Thinking diagnostic, second independently written count (CPU; con1-2m-rows, 11-27): the PLACE_BLOCK numbers that
ea2.py / ea5.py / think_score.py report, recomputed without importing facets.py, targets.py, dirlayer.py or ea2.py.

    python think_check2.py PREDS.jsonl [QUERIES.jsonl]

From the raw records: truth direction = metrics_after vs metrics0 (same within 1e-5 relative); the printed rule =
dP * (R_new - R_old) with Rhot (max_temp_rise) or Rmean (mean_temp_rise) of the block's old and new slot from the
prompt's Slots list, dP = the block's power minus the power of a block already in the new slot (Power blocks line);
the answer read leniently ('VERDICT: applied', then any line '<metric>[:] up|down|same' after CHANGES:); the think
text 'contains the rule value' when it holds a number within 6% of |value| (|value| >= 0.05), as ea5.py defines it.
Counts per metric on applied moves with a true change: direction right among applied answers, agreement with the
rule, the rule's own accuracy; then (max_temp_rise) the share of think texts holding the rule value and the
direction right on that share (non-applied answers counted wrong, as ea5.py). Also: answers, unreadable, think_end.
"""
import collections
import json
import re
import statistics
import sys

P = sys.argv[1]
QF = sys.argv[2] if len(sys.argv) > 2 else "/storage/wxu/con1/arm/main/queries_v2.jsonl"
NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def tdir(b, a):
    return "same" if abs(a - b) <= max(1e-5 * abs(b), 1e-12) else ("up" if a > b else "down")


def sign(x):
    return "up" if x > 1e-6 else "down" if x < -1e-6 else "same"


def slots_of(user):
    m = re.search(r"Slots \([^)]*\): (.*?)\.\n", user)
    out = {}
    for it in (m.group(1).split(";") if m else []):
        w = it.split()
        if len(w) < 5 or not re.fullmatch(r"q\d+", w[0]) or not re.fullmatch(r"n\d+", w[1]):
            continue
        k = 2
        while k < len(w) and not re.fullmatch(NUM, w[k]):      # a node label after the node, if the family prints one
            k += 1
        k += 1                                                 # the node temperature T
        blk = None
        if k < len(w) and re.fullmatch(r"\[\d+\]", w[k]):
            blk, k = int(w[k][1:-1]), k + 1
        nums = []
        for x in w[k:]:
            if re.fullmatch(NUM, x):
                nums.append(float(x))
            else:
                break
        if len(nums) < 2:
            continue
        out[int(w[0][1:])] = {"blk": blk, "Rself": nums[0], "Rhot": nums[1], "Rmean": nums[2] if len(nums) > 2 else None}
    return out


def blocks_of(user):
    m = re.search(r"Power blocks \(P, slot\): (.*?)\.\n", user)
    return {int(b): (float(p), int(s)) for b, p, s in re.findall(r"\[(\d+)\] (%s) W q(\d+)" % NUM, m.group(1) if m else "")}


def read(text):
    text = text or ""
    v = re.search(r"VERDICT:\s*(\w+)", text)
    if not v:
        return ("refused" if re.search(r"refused|rejected", text, re.I) else "unreadable"), {}
    if not v.group(1).lower().startswith("appl"):
        return "refused", {}
    d = {}
    for mt, dr in re.findall(r"^\W*(max_temp_rise|mean_temp_rise)\W+(up|down|same)\b", text.split("CHANGES:", 1)[-1], re.M):
        d[mt] = dr
    return "applied", d


def near(txt, x, tol=0.06):
    if abs(x) < 0.05:
        return False
    return any(abs(abs(float(v)) - abs(x)) <= tol * abs(x) for v in re.findall(NUM, txt))


qs = {}
for line in open(QF):
    q = json.loads(line)
    if q["variant"] == "thermal2" and q["group"] == "heldout_family":
        qs[q["sid"]] = q
rows = [json.loads(l) for l in open(P)]
rows = [r for r in rows if r.get("pass") == "zero" and r["sid"] in qs]
pb = [r for r in rows if r["move"].startswith("PLACE_BLOCK(")]
c = collections.Counter()
for r in pb:
    c["answers"] += 1
    c["unreadable"] += read(r.get("text"))[0] == "unreadable"
    c["think_end " + str(r.get("think_end"))] += 1
th = [r.get("think_tokens") or 0 for r in pb]
print("%s: PLACE_BLOCK answers %d, unreadable %d, %s, think tokens median %s" % (
    P, c["answers"], c["unreadable"], {k[10:]: v for k, v in c.items() if k.startswith("think_end")},
    statistics.median(th) if th else "-"))
L = collections.defaultdict(list)
E = []
for r in pb:
    q = qs[r["sid"]]
    m = next(x for x in q["moves"] if x["move"] == r["move"])
    if m["verdict"] == "refused":
        continue
    sl, bl = slots_of(q["user"]), blocks_of(q["user"])
    a = re.match(r"PLACE_BLOCK\((\d+),\s*q(\d+)\)", r["move"])
    b, new = int(a.group(1)), int(a.group(2))
    rule = {}
    if b in bl and new in sl and bl[b][1] in sl:
        so, sn = sl[bl[b][1]], sl[new]
        dP = bl[b][0] - (bl[sn["blk"]][0] if sn["blk"] is not None and sn["blk"] in bl else 0.0)
        rule["max_temp_rise"] = dP * (sn["Rhot"] - so["Rhot"])
        if sn["Rmean"] is not None and so["Rmean"] is not None:
            rule["mean_temp_rise"] = dP * (sn["Rmean"] - so["Rmean"])
    reg, d = read(r.get("text"))
    for k in ("max_temp_rise", "mean_temp_rise"):
        td = tdir(float(q["metrics0"][k]), float(m["metrics_after"][k]))
        if td == "same":
            continue
        if k == "max_temp_rise" and "max_temp_rise" in rule:
            E.append((near(r.get("think") or "", rule[k]), reg == "applied" and d.get(k) == td))
        if reg != "applied":
            continue
        L[k].append((d.get(k) == td, sign(rule[k]) == d.get(k) if k in rule else None,
                     sign(rule[k]) == td if k in rule else None))
f = lambda xs: "%.2f (n=%d)" % (sum(xs) / len(xs), len(xs)) if xs else "-"  # noqa: E731
for k in ("max_temp_rise", "mean_temp_rise"):
    R = L[k]
    print("  %-15s direction right %s | agrees with the printed rule %s | rule right %s" % (
        k, f([x[0] for x in R]), f([x[1] for x in R if x[1] is not None]), f([x[2] for x in R if x[2] is not None])))
print("  max_temp_rise, every applied move with a true change (non-applied answers wrong): think holds the rule value %s;"
      " direction right on those %s, on the others %s" % (f([e[0] for e in E]), f([e[1] for e in E if e[0]]),
                                                          f([e[1] for e in E if not e[0]])))
print("THINK_CHECK2_DONE")
