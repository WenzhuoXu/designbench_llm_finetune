"""Count only (no row is built or written): grid2op rows of the stage-1 set and of the 2M set whose targets carry
move-independent metric lines (CPU; con1-2m-rows, 11-28, the parent's question).

    PYTHONDONTWRITEBYTECODE=1 python count_exo_rows.py --builder-dir REPO/scripts/con1 --states S1 S2 ... \\
        --exclude-rows R1 R2 ... --lens lens_facetm_short_pw0_nc_q36.txt --stage1-rows main/rows/train_facet.jsonl \\
        [--q 36] [--seed 0] --out REPORT.txt

Move-independent ('exogenous') metric of a state: the state has at least two applied outcomes, the metric's after-value
is the same number for every applied outcome of the state (relative difference <= 1e-9) and differs from the
before-value (relative change > 1e-5); the outcomes are every simulated outcome of the state in the states files (the
same definition as make_line_flags.py on the test set).
A row carries such a line as a target when one of its questions is an applied move of a state with an exogenous metric
(the facet / facetm answer of an applied move has a CHANGES line for every metric; checked on the stage-1 rows by
reading the answers).
  stage-1 set: the rows file itself (sid and con1.moves of every row; the answers read)
  2M set: the rows are not stored (each training job builds them on its node). Their grouping is replayed exactly: the
     builder writes rows state by state in input order (first line of a sid wins; split train only), one lens line per
     row (tokens, prompt tokens, questions, family); a state's rows are the next lens lines whose question counts add up
     to its number of outcomes left after the exclusions; its groups are the builder's own partition() with its own
     seeded random.Random, called for n_rows = ceil(n / q) ... R (the builder retries with one more row while a row
     is over the token cap, from the same generator). The replay must give the same question count per row as the lens
     file for every state, or the count stops.
"""
import argparse
import collections
import gzip
import json
import math
import random
import sys


def opener(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def exo_metrics(st):
    app = [m for m in st["moves"] if m["verdict"] != "refused" and m.get("metrics_after")]
    out = set()
    if len(app) < 2:
        return out
    for k, b in st["metrics0"].items():
        vals = [m["metrics_after"].get(k) for m in app]
        if not num(b) or not all(num(v) for v in vals):
            continue
        v0 = float(vals[0])
        if all(abs(float(v) - v0) <= 1e-9 * max(abs(v0), 1e-12) for v in vals) and \
                abs(v0 - float(b)) > max(1e-5 * abs(float(b)), 1e-12):
            out.add(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder-dir", required=True)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--exclude-rows", nargs="+", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--stage1-rows", required=True)
    ap.add_argument("--q", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default="grid2op")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, a.builder_dir)
    import build_full_rows as B  # the builder's own partition() and _kind(); nothing is built
    excl = collections.defaultdict(set)
    for p in a.exclude_rows:
        for line in opener(p):
            r = json.loads(line)
            if r.get("tier") == "con1_conseq":
                excl[r["sid"]].update(r["con1"]["moves"])
    lens = [l.split() for l in open(a.lens) if l.strip()]
    li = 0
    seen = set()
    fam_states = {}                      # sid -> (exo set, {move: applied}) for the family
    C = collections.Counter()
    by_metric = collections.Counter()
    for p in a.states:
        for line in opener(p):
            if not line.strip():
                continue
            st = json.loads(line)
            sid = st["sid"]
            if sid in seen:
                continue
            seen.add(sid)
            v = st.get("variant")
            if v == a.family:
                fam_states[sid] = (exo_metrics(st), {m["move"]: m["verdict"] != "refused" for m in st["moves"]},
                                   st.get("split"))
            if st.get("split") != "train" or not st.get("moves"):
                continue
            dup, moves = set(), []
            for m in st["moves"]:
                if m["move"] in dup:
                    continue
                dup.add(m["move"])
                if m["move"] in excl.get(sid, ()):
                    continue
                moves.append(m)
            if not moves:
                continue
            n = len(moves)
            got, k = [], 0
            while sum(got) < n:
                if li + k >= len(lens):
                    raise SystemExit("lens file ends inside state %s" % sid)
                row = lens[li + k]
                if row[3] != v:
                    raise SystemExit("lens row %d is family %s, state %s is %s" % (li + k, row[3], sid, v))
                got.append(int(row[2]))
                k += 1
            if sum(got) != n:
                raise SystemExit("question counts of state %s do not add up: %s vs %d" % (sid, got, n))
            li += k
            C["%s states" % v] += 1
            C["%s rows" % v] += k
            if v != a.family:
                continue
            rng = random.Random("con1-full|%d|%s" % (a.seed, sid))
            n0 = int(math.ceil(n / a.q))
            if k < n0:
                raise SystemExit("state %s has %d rows, fewer than ceil(n/q) = %d" % (sid, k, n0))
            for nr in range(n0, k + 1):
                groups = B.partition(moves, nr, rng)
            if [len(g) for g in groups] != got:
                raise SystemExit("replayed partition of %s gives %s, lens file %s" % (sid, [len(g) for g in groups], got))
            C["2M replay: states matched"] += 1
            exo = fam_states[sid][0]
            for g in groups:
                n_app = sum(1 for m in g if m["verdict"] != "refused")
                C["2M rows"] += 1
                C["2M questions"] += len(g)
                C["2M applied questions"] += n_app
                if exo:
                    C["2M rows of a state with an exogenous metric"] += 1
                    if n_app:
                        C["2M rows carrying exogenous target lines"] += 1
                        C["2M exogenous target lines"] += n_app * len(exo)
                        for x in exo:
                            by_metric["2M " + x] += n_app
    if li != len(lens):
        raise SystemExit("lens rows left over: %d of %d" % (len(lens) - li, len(lens)))
    C["2M replay: lens rows consumed"] = li
    # stage-1 set
    for line in opener(a.stage1_rows):
        r = json.loads(line)
        C["stage-1 rows (all families)"] += 1
        if r.get("domain") != a.family:
            continue
        sid = r["sid"]
        if sid not in fam_states:
            C["stage-1 %s rows without their state" % a.family] += 1
            continue
        exo, applied, _ = fam_states[sid]
        mv = r["con1"]["moves"]
        n_app = sum(1 for m in mv if applied.get(m))
        C["stage-1 rows"] += 1
        C["stage-1 questions"] += len(mv)
        C["stage-1 applied questions"] += n_app
        # the CHANGES lines of the row's answers that name an exogenous metric (read, not assumed)
        ans = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        present = 0
        for t in ans:
            if "CHANGES:" in t:
                body = t.split("CHANGES:", 1)[1]
                present += sum(1 for ln in body.splitlines() if ln.split() and ln.split()[0] in exo)
        if exo:
            C["stage-1 rows of a state with an exogenous metric"] += 1
            if n_app:
                C["stage-1 rows carrying exogenous target lines"] += 1
                C["stage-1 exogenous target lines (applied questions x metrics)"] += n_app * len(exo)
                C["stage-1 exogenous CHANGES lines read in the answers"] += present
                for x in exo:
                    by_metric["stage-1 " + x] += n_app
    st_exo = collections.Counter()
    for sid, (exo, _, sp) in fam_states.items():
        st_exo["%s states (%s)" % (a.family, sp)] += 1
        st_exo["%s states with an exogenous metric (%s)" % (a.family, sp)] += bool(exo)
        for x in exo:
            st_exo["  state-level: %s exogenous (%s)" % (x, sp)] += 1
    out = ["count_exo_rows: family %s; states %s; lens %s; stage-1 rows %s" % (a.family, a.states, a.lens, a.stage1_rows)]
    out += ["  %s: %d" % kv for kv in sorted(C.items())]
    out += ["  %s: %d" % kv for kv in sorted(st_exo.items())]
    out += ["  target lines by metric, %s: %d" % kv for kv in sorted(by_metric.items())]
    open(a.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("COUNT_EXO_ROWS_DONE")


if __name__ == "__main__":
    main()
