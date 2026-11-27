"""Thinking diagnostic, counts (CPU; con1-2m-rows, 11-27): one think_client.py preds file.

    python think_score.py preds_L.jsonl

Counts only, no reading of the think text: questions answered, errors, unreadable answers (facets.pred_facets regime
'unreadable', counted apart), think length in tokens (median, quartiles, share empty, how the think block ended),
answer length. Then per verb (PLACE_BLOCK, MOVE_SINK) and metric (max_temp_rise, mean_temp_rise), on questions whose
true outcome is applied with a direction other than 'same':
  facets   direction right as ea2.py counts it (the answer read by facets.pred_facets, applied answers only)
  lines    direction from any line of the form '<metric> ... up|down|same' (the last such line per metric), every
           answer that has one: a reading that does not need the facetm layout
"""
import collections
import json
import re
import statistics
import sys

sys.path.insert(0, "/storage/wxu/con1/repo/scripts/con1")
import dirlayer as D  # noqa: E402
import facets as F  # noqa: E402

P = sys.argv[1]
qs = {}
for l in open("/storage/wxu/con1/arm/main/queries_v2.jsonl"):
    r = json.loads(l)
    if r["variant"] == "thermal2" and r["group"] == "heldout_family":
        qs[r["sid"]] = r
rows = []
for l in open(P):
    try:
        r = json.loads(l)
    except ValueError:
        continue
    if r.get("pass") == "zero" and r["sid"] in qs:
        rows.append(r)
METRICS = ("max_temp_rise", "mean_temp_rise")
LINE = re.compile(r"^\W*(%s)\b[^\n]*?\b(up|down|same)\b" % "|".join(METRICS), re.I | re.M)


def q_tile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


ok = [r for r in rows if not str(r.get("finish", "")).startswith("error")]
print("%s: %d lines, %d answered, %d errors" % (P, len(rows), len(ok), len(rows) - len(ok)))
for verb in ("all",) + ("PLACE_BLOCK", "MOVE_SINK"):
    R = [r for r in ok if verb == "all" or D.verb_of(next(m for m in qs[r["sid"]]["moves"] if m["move"] == r["move"])) == verb]
    if not R:
        continue
    th = [r.get("think_tokens") or 0 for r in R]
    ends = collections.Counter(r.get("think_end") for r in R)
    unread = sum(F.pred_facets(r.get("text"), "facetm", qs[r["sid"]])["regime"] == "unreadable" for r in R)
    empty = sum(not (r.get("think") or "").strip() for r in R)
    print("  %-12s n %3d | unreadable %3d (%.3f) | think tokens median %d (q25 %d, q75 %d), empty %d (%.3f), ended %s |"
          " answer tokens median %d, cut at the answer cap %d" % (
              verb, len(R), unread, unread / len(R), statistics.median(th), q_tile(th, .25), q_tile(th, .75), empty,
              empty / len(R), dict(ends), statistics.median([r.get("answer_tokens") or 0 for r in R]),
              sum(r.get("finish") == "length" for r in R)))
print("direction right, true outcome applied and true direction not 'same':")
print("  %-12s %-15s | %-24s | %-24s" % ("verb", "metric", "facets (applied answers)", "lines (any answer)"))
for verb in ("PLACE_BLOCK", "MOVE_SINK"):
    for k in METRICS:
        fa, li = [], []
        for r in ok:
            q = qs[r["sid"]]
            m = next(x for x in q["moves"] if x["move"] == r["move"])
            if D.verb_of(m) != verb:
                continue
            t = F.truth_facets(q, m)
            if not F.is_applied(t["regime"]) or k not in t["dir"] or t["dir"][k] == "same":
                continue
            pf = F.pred_facets(r.get("text"), "facetm", q)
            if F.is_applied(pf["regime"]):
                fa.append(pf["dir"].get(k) == t["dir"][k])
            found = [d.lower() for mk, d in LINE.findall(r.get("text") or "") if mk.lower() == k]
            if found:
                li.append(found[-1] == t["dir"][k])
        f = lambda xs: "%.2f (n=%d)" % (sum(xs) / len(xs), len(xs)) if xs else "-"  # noqa: E731
        print("  %-12s %-15s | %-24s | %-24s" % (verb, k, f(fa), f(li)))
print("THINK_SCORE_DONE")
