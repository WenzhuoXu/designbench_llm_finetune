"""con1 step 1: states and the simulated result of moves at each state (CPU only, no model).

  episodes  python transitions.py episodes --code C --harness V3RC --problems P [P ...] --episodes E [E ...] \\
                --out states.jsonl.gz [--turns all|one] [--split train|test] [--pair-b E_b ...] [--workers 20]
            Replays recorded episodes (rollout records; prompts optional) with the harness's bookkeeping. --turns all:
            every turn is a state (training: the chk1 collections). --turns one: one turn per episode, drawn
            uniformly with a seed from the problem id (test: pref1's benchmark pair a). Own moves at a state: the
            committed move and every check of that turn; with --pair-b, also the move another run of the same
            problem committed at the identical state (same turn log).
  scripted  python transitions.py scripted --code C --harness V3RC --problems P --variants circuit2 ... \\
                --out states.jsonl.gz [--max-moves 6] [--limit-per-variant N]
            Scripted episodes on the given problems: each move is, with probability 1/2, the best legal move by Phi
            and otherwise a uniform legal move (seeded per problem); every turn up to the first feasible design is a
            state.

Each state line: sid, variant, problem_id, split, t, turn_log (rebuilds the state: conseq_lib.rebuild), system,
user (the served prompt), phi0, bad0, feasible0, n_legal, and moves = [{move (text as the domain formats it; the
move simulated is that text parsed back, i.e. what a check of the text returns), src (own | legal | offgrid),
verdict, phi, feasible, viol, refused, result, metrics_after}]. Every enumerated legal move is simulated; verb-
surface families also get 2 off-grid moves (a parameter scaled by a log-uniform factor in [0.5, 2]); native families
2 out-of-menu moves (offmenu_moves: random arguments over the menu's argument ranges; added 11-25, states generated
before that get them from add_offmenu.py).
Resumable: episodes / problems already in --out (by key) are skipped; the file is gzip-appended per episode.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_ENV = None


def _env(code, harness):
    global _ENV
    if _ENV is None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from conseq_lib import Env
        _ENV = Env(code, harness)
    return _ENV


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def seeded(*parts) -> int:
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def split_of(pid: str) -> str:
    """1 problem in 100 is dev (by hash), the rest train (training-pool states only)."""
    return "dev" if seeded("con1-dev", pid) % 100 == 0 else "train"


def move_args(s, mv) -> dict:
    """The parsed move as features for the tree baseline: verb + arguments; verb-surface moves also carry the
    element's current value (v_old) of the parameter they set."""
    d = mv.as_dict()
    out = {"verb": d.get("verb", "SET_PARAM"), "args": dict(d.get("args", {k: v for k, v in d.items() if k != "verb"}))}
    if not s.native:
        a = out["args"]
        try:
            out["v_old"] = float(s.domain.get(s.st, int(a["i"]), a["p"]))
        except Exception:                                  # noqa: BLE001
            out["v_old"] = None
        out["verb"] = s.fmt(mv).split("(", 1)[0]          # the verb as spoken (SET_WIDTH, ...)
    return out


def offmenu_moves(s, legal, k: int, rng) -> list:
    """k native verb calls with random arguments (parent review 11-25: native families need moves from outside the
    adjoint- / heuristic-ranked menu). Each argument is drawn from its range across the whole menu of this state, not
    from the ranked list: numbers uniform over the menu's span (floats widened by a quarter of the span each way),
    names uniformly from the names the menu uses, json lists as a random non-empty subset of the items the menu's lists
    use; a rectangle's corner pairs are sorted. Some calls are refused (a rule or the physics), as arbitrary moves are."""
    import json as _json
    from dataxp.domain import Move
    pools = collections.defaultdict(lambda: collections.defaultdict(list))
    seen_verbs = set()
    for mv in legal:
        d = mv.as_dict()
        seen_verbs.add(d["verb"])
        for a, v in (d.get("args") or {}).items():
            pools[d["verb"]][a].append(v)
    specs = [vs for vs in s.domain.verbs if vs.name in seen_verbs]
    out = []
    for _ in range(6 * k):
        if len(out) >= k or not specs:
            break
        vs = specs[int(rng.integers(len(specs)))]
        args, ok = {}, True
        for a in vs.args:
            vals = pools[vs.name].get(a.name)
            if not vals:
                ok = False
                break
            if a.type in ("int", "float"):
                nums = [float(x) for x in vals]
                lo, hi = min(nums), max(nums)
                if a.type == "float":
                    lo, hi = lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo)
                x = float(rng.uniform(lo, hi)) if hi > lo else lo
                args[a.name] = int(round(x)) if a.type == "int" else round(x, 3)
            elif a.type == "json":
                items = sorted({_json.dumps(it) for v in vals for it in (v if isinstance(v, list) else [v])})
                n = int(rng.integers(1, len(items) + 1))
                args[a.name] = [_json.loads(items[i]) for i in sorted(rng.choice(len(items), n, replace=False))]
            else:
                args[a.name] = vals[int(rng.integers(len(vals)))]
        if not ok:
            continue
        for lo_k, hi_k in (("x0", "x1"), ("y0", "y1")):
            if lo_k in args and hi_k in args and args[lo_k] > args[hi_k]:
                args[lo_k], args[hi_k] = args[hi_k], args[lo_k]
        out.append(Move(vs.name, args))
    return out


def simulate_state(s, own_texts, offgrid: int, rng):
    """All moves of a state: own (texts as written), every enumerated legal move, off-grid (verb surface)."""
    from conseq_lib import finite
    moves, order = {}, []

    def add(mv, src):
        try:
            text = s.fmt(mv)
        except Exception:                                  # noqa: BLE001
            return "unformattable"
        if text in moves:
            if src not in moves[text]["src"]:
                moves[text]["src"].append(src)
            return None
        mv2 = s.parse(text)                                # simulate what a check of the TEXT returns
        if mv2 is None:
            return "unparsable"
        moves[text] = {"move": text, "src": [src], "_mv": mv2, "args": move_args(s, mv2)}
        order.append(text)
        return None

    bad = collections.Counter()
    for t in own_texts:
        mv = s.parse(t)
        if mv is None:
            bad["own unreadable"] += 1
            continue
        r = add(mv, "own")
        if r:
            bad["own " + r] += 1
    legal = s.legal()
    for mv in legal:
        r = add(mv, "legal")
        if r:
            bad["legal " + r] += 1
    if offgrid and not s.native:
        d = s.domain
        cells = [(i, p) for i in range(d.n_elements(s.st)) for p in d.params
                 if d.get(s.st, i, p) is not None and finite(d.get(s.st, i, p)) and float(d.get(s.st, i, p)) != 0.0]
        for _ in range(offgrid):
            if not cells:
                break
            i, p = cells[int(rng.integers(len(cells)))]
            cur = float(d.get(s.st, i, p))
            f = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
            r = add({"i": i, "p": p, "v": cur * f}, "offgrid")
            if r:
                bad["offgrid " + r] += 1
    if offgrid and s.native:                               # out-of-menu native moves (src offmenu)
        for mv in offmenu_moves(s, legal, offgrid, rng):
            r = add(mv, "offmenu")
            if r:
                bad["offmenu " + r] += 1
    out = []
    for text in order:
        m = moves.pop(text)
        res = s.result_of(m.pop("_mv"))
        res["move"] = text                                 # the text (result_of formats the parsed move the same way)
        res["src"] = m["src"]
        res["args"] = m["args"]
        out.append(res)
    return out, len(legal), bad


def state_record(s, key, split, own_texts, offgrid, rng):
    system, user = s.prompt()
    moves, n_legal, bad = simulate_state(s, own_texts, offgrid, rng)
    return {"sid": "%s#t%d" % (key, s.t), "key": key, "variant": s.variant, "problem_id": s.pid, "split": split,
            "t": s.t, "k": len(s.applied), "turn_log": [list(x) for x in s.turn_log], "system": system, "user": user,
            "phi0": float(s.cur.mean), "feasible0": bool(s.cur.feasible), "bad0": s.bad_now(),
            "metrics0": {k: float(v) for k, v in s.metrics.items()}, "binding": s.binding,
            "n_legal": n_legal, "n_moves": len(moves), "faults": dict(bad), "moves": moves}


# ------------------------------------------------------------------ episodes mode

def own_texts_of(rec, s):
    out = []
    for chk in rec.get("checks") or []:
        if chk.get("text"):
            out.append(chk["text"])
    mv = rec.get("action") or rec.get("move")
    if mv is not None:
        if s.native:
            out.append(s.fmt(s.env.D.Move.from_dict(mv)))
        else:
            out.append(s.fmt({"i": int(mv["i"]), "p": mv["p"], "v": float(mv["v"])}))
    return out


def tl_key(turn_log) -> str:
    return json.dumps([(k, m) for k, m, _ in turn_log], sort_keys=True)


def episode_job(job):
    code, harness, r, spec, turns_mode, split_mode, offgrid, pair_b = job
    import numpy as np
    from conseq_lib import replay_turns
    env = _env(code, harness)
    ep = r["episode"]
    turns = ep.get("turns") or []
    out = []
    t0 = time.time()
    try:
        if turns_mode == "one":
            pick = seeded("con1-test", r["problem_id"]) % len(turns)
            want = {pick}
        else:
            want = set(range(len(turns)))
        # pair b: the moves another run committed, by state (turn log up to that turn)
        b_moves = collections.defaultdict(list)
        if pair_b:
            for rec_b, s_b in replay_turns(env, r["variant"], spec, pair_b["episode"].get("turns") or []):
                mv = rec_b.get("action") or rec_b.get("move")
                if mv is not None:
                    b_moves[tl_key(s_b.turn_log)].append(mv)
        for rec, s in replay_turns(env, r["variant"], spec, turns, upto=max(want) + 1):
            if s.t not in want:
                continue
            own = own_texts_of(rec, s)
            for mv in b_moves.get(tl_key(s.turn_log), []):
                own.append(s.fmt(env.D.Move.from_dict(mv)) if s.native else
                           s.fmt({"i": int(mv["i"]), "p": mv["p"], "v": float(mv["v"])}))
            split = split_mode or split_of(r["problem_id"])
            rng = np.random.default_rng(seeded("con1-offgrid", r["key"], s.t))
            out.append(state_record(s, r["key"], split, own, offgrid, rng))
        return r["key"], out, None, time.time() - t0
    except Exception as e:                                 # noqa: BLE001
        import traceback
        return r["key"], out, "%s: %s | %s" % (type(e).__name__, str(e)[:300],
                                               traceback.format_exc().splitlines()[-3:]), time.time() - t0


# ------------------------------------------------------------------ scripted mode

def scripted_job(job):
    code, harness, variant, spec, max_moves, offgrid = job
    import numpy as np
    from conseq_lib import State
    env = _env(code, harness)
    pid = spec["problem_id"]
    key = "scripted/%s" % pid
    rng = np.random.default_rng(seeded("con1-scripted", pid))
    out = []
    t0 = time.time()
    try:
        s = State(env, variant, spec)
        s.turn_log = []
        for _ in range(max_moves):
            if s.cur.feasible:
                break
            rec = state_record(s, key, split_of(pid), [], offgrid, rng)
            out.append(rec)
            legal = [m for m in rec["moves"] if "legal" in m["src"] and m["verdict"] != "refused"]
            if not legal:
                break
            if rng.random() < 0.5:
                pick = max(legal, key=lambda m: (m["verdict"] == "better", m["phi"]))
            else:
                pick = legal[int(rng.integers(len(legal)))]
            mv = s.parse(pick["move"])
            if not s.native:
                mv = {"i": int(mv.args["i"]), "p": mv.args["p"], "v": float(mv.args["v"])}
            s.turn_log.append(("move", mv.as_dict() if s.native else dict(mv), None))
            why = s.commit(mv)
            if why is not None:
                raise RuntimeError("scripted move refused on commit: %s %s" % (pick["move"], why))
        return key, out, None, time.time() - t0
    except Exception as e:                                 # noqa: BLE001
        import traceback
        return key, out, "%s: %s | %s" % (type(e).__name__, str(e)[:300],
                                          traceback.format_exc().splitlines()[-3:]), time.time() - t0


# ------------------------------------------------------------------ driver

def done_keys(path):
    keys = set()
    if os.path.exists(path):
        try:
            for line in opener(path):
                try:
                    keys.add(json.loads(line)["key"])
                except Exception:                          # noqa: BLE001
                    continue
        except EOFError:                                   # a member cut by a killed run: what was read stands
            pass
    done = path + ".done"
    if os.path.exists(done):
        keys |= {l.strip() for l in open(done) if l.strip()}
    return keys


def read_specs(paths):
    specs = {}
    for p in paths:
        for line in opener(p):
            if line.strip():
                r = json.loads(line)
                s = r.get("spec", r)
                specs[s["problem_id"]] = (r.get("variant"), s)
    return specs


def run(jobs, fn, out, workers):
    from multiprocessing import Pool
    from conseq_lib import jdump
    t0 = time.time()
    n = n_err = n_states = 0
    by = collections.Counter()
    with Pool(workers, maxtasksperchild=50) as pool, open(out + ".done", "a") as fd:
        for key, recs, err, sec in pool.imap_unordered(fn, jobs, chunksize=1):
            n += 1
            if err:
                n_err += 1
                print("!!! %s: %s" % (key, err), flush=True)
            if recs:
                with gzip.open(out, "at", encoding="utf-8") as fo:     # one gzip member per episode
                    for r in recs:
                        fo.write(jdump(r) + "\n")
                n_states += len(recs)
                for r in recs:
                    by[r["variant"]] += 1
            if not err:
                fd.write(key + "\n")
                fd.flush()
            if n % 25 == 0 or n == len(jobs):
                print("%d/%d episodes, %d states %s, %d errors, %.0f s" % (n, len(jobs), n_states, dict(by), n_err,
                                                                         time.time() - t0), flush=True)
    print("TRANSITIONS_DONE %d episodes, %d states, %d errors" % (n, n_states, n_err), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["episodes", "scripted"])
    ap.add_argument("--code", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--episodes", nargs="*", default=[])
    ap.add_argument("--pair-b", nargs="*", default=[])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--turns", choices=["all", "one"], default="all")
    ap.add_argument("--split", default=None, help="force this split (test); default: train/dev by problem hash")
    ap.add_argument("--offgrid", type=int, default=2)
    ap.add_argument("--max-moves", type=int, default=6)
    ap.add_argument("--limit-per-variant", type=int, default=0)
    ap.add_argument("--exclude-specs", nargs="*", default=[], help="problem files whose problems must not appear")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--reverse", action="store_true", help="work the job list back to front")
    ap.add_argument("--start-frac", type=float, default=0.0, help="start at this fraction of the job list (wraps)")
    ap.add_argument("--skip-keys", default=None, help="another output file (its .done keys are skipped)")
    a = ap.parse_args(argv)
    specs = read_specs(a.problems)
    banned = set()
    for p in a.exclude_specs:
        for line in opener(p):
            if line.strip():
                r = json.loads(line)
                s = r.get("spec", r)
                banned.add(json.dumps({k: v for k, v in s.items() if k != "problem_id"}, sort_keys=True, default=str))
    canon = lambda s: json.dumps({k: v for k, v in s.items() if k != "problem_id"}, sort_keys=True, default=str)  # noqa: E731
    done = done_keys(a.out)
    jobs = []
    if a.mode == "episodes":
        b = {}
        for path in a.pair_b:
            for line in opener(path):
                r = json.loads(line)
                b[r["problem_id"]] = r
        for path in a.episodes:
            for line in opener(path):
                r = json.loads(line)
                if a.variants and r["variant"] not in a.variants:
                    continue
                ep = r["episode"]
                if ep.get("crashed") or not ep.get("turns") or r["key"] in done:
                    continue
                spec = specs[r["problem_id"]][1]
                if canon(spec) in banned:
                    raise SystemExit("%s is an excluded problem" % r["problem_id"])
                jobs.append((a.code, a.harness, r, spec, a.turns, a.split, a.offgrid, b.get(r["problem_id"])))
        fn = episode_job
    else:
        per = collections.Counter()
        for pid, (v, spec) in specs.items():
            if a.variants and v not in a.variants:
                continue
            if a.limit_per_variant and per[v] >= a.limit_per_variant:
                continue
            if canon(spec) in banned:
                raise SystemExit("%s is an excluded problem" % pid)
            per[v] += 1
            if "scripted/%s" % pid in done:
                continue
            jobs.append((a.code, a.harness, v, spec, a.max_moves, a.offgrid))
        fn = scripted_job
    if a.reverse:                                          # a second machine works the list from the other end
        jobs = jobs[::-1]
    if a.start_frac:                                       # a third machine starts in the middle
        k = int(len(jobs) * a.start_frac)
        jobs = jobs[k:] + jobs[:k]
    if a.skip_keys:
        skip = done_keys(a.skip_keys)
        jobs = [j for j in jobs if (j[2]["key"] if a.mode == "episodes" else "scripted/%s" % j[3]["problem_id"]) not in skip]
    print("%s: %d jobs to run (%d already done)" % (a.mode, len(jobs), len(done)), flush=True)
    run(jobs, fn, a.out, a.workers)


if __name__ == "__main__":
    main()
