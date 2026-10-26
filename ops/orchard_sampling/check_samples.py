"""Statistics of sample files (dataxp.ownrows.sample output): records by (sid, sample)
(last occurrence wins), finish reasons, errors, share with an <action> tag, share whose
action parses with the domain's parse_move, completion tokens. --gate exits 3 when the
set looks broken: no records, >= 10% errors, every reply cut at max_tokens, or < 80% of
the replies that finished carrying an <action> tag. (A reply cut at max_tokens is the
model's verbosity, not a broken setup; its share is printed and sample.py redraws it on a
rerun.)

    python check_samples.py [--gate] <samples.jsonl> [more ...]
"""
import argparse
import collections
import json
import sys

from dataxp.domains import get_domain
from dataxp.ownrows.common import action_text


def load(files):
    recs, bad = {}, 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for l in fh:
                try:
                    o = json.loads(l)
                except ValueError:
                    bad += 1
                    continue
                recs[(o["sid"], int(o["sample"]))] = o
    return recs, bad


def stats(recs):
    fin, per = collections.Counter(), collections.defaultdict(collections.Counter)
    toks, domains = [], {}
    for (sid, s), o in recs.items():
        v = sid.split("|")[0]
        per[v]["n"] += 1
        if o.get("error"):
            fin["error"] += 1
            per[v]["error"] += 1
            continue
        fin[str(o.get("finish"))] += 1
        per[v][str(o.get("finish"))] += 1
        if o.get("tokens") is not None:
            toks.append(int(o["tokens"]))
        txt = action_text(o.get("answer"))
        if txt:
            per[v]["tagged"] += 1
            if o.get("finish") != "length":
                per[v]["tagged_stop"] += 1
            try:
                d = domains.get(v) or domains.setdefault(v, get_domain(v))
                mv = d.parse_move(txt)
            except Exception as e:  # noqa: BLE001
                mv = None
                per[v]["parse_exc"] += 1
                per[v]["_exc"] = str(e)[:80]
            if mv is not None:
                per[v]["parsed"] += 1
    return fin, per, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    recs, bad = load(a.files)
    n = len(recs)
    fin, per, toks = stats(recs)
    tagged = sum(c["tagged"] for c in per.values())
    tagged_stop = sum(c["tagged_stop"] for c in per.values())
    parsed = sum(c["parsed"] for c in per.values())
    err = fin.get("error", 0)
    n_len = fin.get("length", 0)
    n_stop = n - err - n_len
    print("records %d (unique (sid,sample); %d unreadable lines) finish %s" % (n, bad, dict(fin)))
    if n:
        print("errors %.1f%%  length %.1f%%  <action> tag %.1f%% (%.1f%% of the %d that finished)  parsed %.1f%%  "
              "tokens mean %.0f p50 %d p95 %d" % (
                  100.0 * err / n, 100.0 * n_len / n, 100.0 * tagged / n,
                  (100.0 * tagged_stop / n_stop) if n_stop else 0.0, n_stop, 100.0 * parsed / n,
                  (sum(toks) / len(toks)) if toks else 0, sorted(toks)[len(toks) // 2] if toks else 0,
                  sorted(toks)[int(len(toks) * 0.95)] if toks else 0))
    for v in sorted(per):
        c = per[v]
        print("  %-10s n %5d  stop %5d  length %4d  error %3d  tagged %5d  parsed %5d%s" % (
            v, c["n"], c["stop"], c["length"], c["error"], c["tagged"], c["parsed"],
            ("  parse_exc %d (%s)" % (c["parse_exc"], c["_exc"])) if c["parse_exc"] else ""))
    if a.gate:
        ok = n > 0 and err < 0.1 * n and n_stop > 0 and tagged_stop >= 0.8 * n_stop
        print("gate:", "ok" if ok else "FAILED")
        return 0 if ok else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
