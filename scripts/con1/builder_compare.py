"""Compare builder_dump.py outputs (CPU; 11-28): the lib_v8 switches against lib_v6's rows and prompts.

    python builder_compare.py STATES.jsonl.gz OLD.jsonl OPTOUT.jsonl DEFAULT.jsonl ZERO.jsonl DROP.jsonl

  old == optout      every row and eval prompt byte-identical (sorted-key JSON)
  default vs old     only the first user turn differs, by exactly the definitions appended at its end (rows: the
                     row's format; eval prompts: facetm); every other message identical
  zero vs default    content identical; grid2op applied answers carry zero_spans that cover exactly the CHANGES lines of
                     the state's move-independent metrics (recomputed here); no other difference
  drop vs default    grid2op applied answers lose exactly those CHANGES lines; nothing else differs
Exit 1 on any failure.
"""
import collections
import gzip
import json
import sys

DEF_FEAS = " FEASIBLE AFTER is yes only when no constraint is violated after the move; otherwise it is no."
DEF_SAME = (" A metric's direction is 'same' when it changes by less than 0.1% of its current value; otherwise it "
            "is up or down.")


def defs(fmt):
    return DEF_FEAS + (DEF_SAME if fmt in ("facet", "facetm", "dir") else "")


def exo(st):
    app = [m for m in st["moves"] if m["verdict"] != "refused" and m.get("metrics_after")]
    out = set()
    if len(app) < 2:
        return out
    for k, b in st["metrics0"].items():
        vs = [m["metrics_after"].get(k) for m in app]
        if not isinstance(b, (int, float)) or not all(isinstance(v, (int, float)) for v in vs):
            continue
        if max(vs) - min(vs) <= 1e-9 * max(abs(vs[0]), 1e-12) and abs(vs[0] - b) > max(1e-5 * abs(b), 1e-12):
            out.add(k)
    return out


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    sp, po, pt, pd, pz, pr = sys.argv[1:7]
    states = {}
    for line in gzip.open(sp, "rt", encoding="utf-8"):
        st = json.loads(line)
        states.setdefault(st["sid"], st)
    O, T, D, Z, R = (load(p) for p in (po, pt, pd, pz, pr))
    c = collections.Counter()
    assert len(O) == len(T) == len(D) == len(Z) == len(R)
    for o, t in zip(O, T):
        c["old == optout"] += json.dumps(o, sort_keys=True) == json.dumps(t, sort_keys=True)
    for o, d in zip(O, D):
        if o["kind"] == "row":
            mo, md, fmt = o["row"]["messages"], d["row"]["messages"], o["fmt"]
        else:
            mo, md, fmt = o["msgs"], d["msgs"], "facetm"
        # every question turn (user turns 1, 3, 5, ...) carries the definitions at its end; nothing else changes
        ok = len(mo) == len(md) and \
            all((x["content"] + defs(fmt) == y["content"] and {a: b for a, b in x.items() if a != "content"} ==
                 {a: b for a, b in y.items() if a != "content"}) if x["role"] == "user" else
                json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(mo, md)) \
            and {k: v for k, v in (o.get("row") or {}).items() if k != "messages"} == {k: v for k, v in (d.get("row") or {}).items() if k != "messages"}
        c["default = old + definitions (%s)" % o["kind"]] += ok
        c["%s objects" % o["kind"]] += 1
    for d, z, r in zip(D, Z, R):
        if d["kind"] != "row":
            c["zero == default (eval)"] += d == z
            c["drop == default (eval)"] += d == r
            continue
        st = states[d["sid"]]
        ex = exo(st) if st["variant"] == "grid2op" else set()
        md, mz, mr = d["row"]["messages"], z["row"]["messages"], r["row"]["messages"]
        okz = okr = len(md) == len(mz) == len(mr)
        for k, (x, y, w) in enumerate(zip(md, mz, mr)):
            if x["role"] != "assistant" or not ex or "VERDICT: applied" not in x["content"] or d["fmt"] in ("verbatim", "masked"):
                okz = okz and x == y
                okr = okr and x == w
                continue
            lines = x["content"].split("\n")
            head = "CHANGES:" if "CHANGES:" in lines else "METRICS (before -> after):"
            h = lines.index(head)
            target = [i for i, ln in enumerate(lines) if i > h and ln.split() and ln.split()[0] in ex]
            want, pos = [], 0
            for i, ln in enumerate(lines):
                if i in target:
                    want.append((pos, pos + len(ln)))
                pos += len(ln) + 1
            yy = {kk: vv for kk, vv in y.items() if kk != "zero_spans"}
            okz = okz and yy == x and [tuple(s) for s in y.get("zero_spans", [])] == want
            c["zeroed lines"] += len(target)
            kept = [ln for i, ln in enumerate(lines) if i not in target]
            okr = okr and w["content"].split("\n") == kept and w["role"] == x["role"]
            c["dropped lines"] += len(target)
        c["zero == default + exact zero_spans (row)"] += okz
        c["drop == default - exact lines (row)"] += okr
    print(dict(c))
    bad = (c["row objects"] + c["eval objects"]) - c["old == optout"]
    bad += c["row objects"] - c["default = old + definitions (row)"] + c["eval objects"] - c["default = old + definitions (eval)"]
    bad += c["row objects"] - c["zero == default + exact zero_spans (row)"] + c["row objects"] - c["drop == default - exact lines (row)"]
    bad += c["eval objects"] - c["zero == default (eval)"] + c["eval objects"] - c["drop == default (eval)"]
    print("BUILDER_COMPARE %s (%d failures)" % ("OK" if not bad else "FAILED", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
