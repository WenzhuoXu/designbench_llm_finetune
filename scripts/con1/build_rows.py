"""con1 step 2: training rows from simulated states (transitions.py) + policy replay rows.

    python build_rows.py --states native.jsonl.gz scripted.jsonl.gz --policy train_chk1.jsonl.gz \\
        --out-train train.jsonl --out-dev dev.jsonl [--per-family 2000] [--q 12] [--policy-ratio 0.33] [--seed 0]

Consequence row (docs/consequence_round_con1.md section 1): the state's served (system, user) with the move request cut
(conseq_lib.consequence_user, since 11-27) + QUESTION about one
move, the simulator's result message as the answer (empty think block), then Q-1 more questions about other moves
of the same state, each a new user turn, each answer supervised (supervise = every_assistant; llm_finetune/
rl_dataxp/conseq_format.py). Moves of a row: the policy's own moves at the state (up to 4), the 2 best legal moves by
Phi (photonics under the open rules: fewest rule-breaking pixels first), up to 2 off-grid moves (verb-surface
families), uniform legal moves for the rest; in a seeded random order. A row whose estimated length passes
--max-tokens gets fewer questions (at least --min-q), else it is dropped (counted).

Policy rows: the open-rules restated dx38 replay rows (tier ownrows_own) of the 8 training families, rationale then
action, about --policy-ratio of the consequence row count (all of them if fewer).
Dev: the states of dev problems (transitions.split_of) -> at most --dev-max consequence rows, plus --dev-policy
policy rows not used in training.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conseq_lib import consequence_user  # noqa: E402  (the prompt without the move request, owner order 11-27)
from targets import FIXED_WEIGHT, FORMATS, question, render  # noqa: E402

TRAIN_FAMILIES = ["engibench", "grid2op", "photonics", "circuit2", "frame3d2", "grid2", "seir2", "traffic2"]
HELD_OUT = {"inventory2", "molopt", "thermal2"}
CHARS_PER_TOKEN = 3.2          # conservative (measured ~3.4-3.6 on these prompts)


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def rank_key(variant, m):
    """Best-first key of an applied move (photonics: fewer rule-breaking pixels, then Phi)."""
    if variant == "photonics":
        return (-(m.get("viol") or 0.0), m.get("phi") or -1e18)
    return (0.0, m.get("phi") or -1e18)


def pick_moves(st: dict, q: int, rng: random.Random, exclude=frozenset()) -> list:
    """The row's moves: own (up to 4), the 2 best legal applied, up to 2 off-menu (verb-surface 'offgrid' or native
    'offmenu'), uniform legal for the rest; moves in `exclude` (an earlier row set's) are not drawn."""
    by_text = {m["move"]: m for m in st["moves"]}
    moves = [m for m in st["moves"] if m["move"] not in exclude]
    chosen, seen = [], set()

    def take(m):
        if m["move"] not in seen:
            seen.add(m["move"])
            chosen.append(m)

    own = [m for m in moves if "own" in m["src"]]
    rng.shuffle(own)
    for m in own[:4]:
        take(m)
    applied = [m for m in moves if "legal" in m["src"] and m["verdict"] != "refused"]
    for m in sorted(applied, key=lambda m: rank_key(st["variant"], m), reverse=True)[:2]:
        take(m)
    off = [m for m in moves if "offgrid" in m["src"] or "offmenu" in m["src"]]   # 'offmenu' was missing until 11-26
    for m in off[:2]:
        take(m)
    rest = [m for m in moves if "legal" in m["src"] and m["move"] not in seen]
    rng.shuffle(rest)
    for m in rest:
        if len(chosen) >= q:
            break
        take(m)
    chosen = chosen[:q]
    rng.shuffle(chosen)
    assert all(by_text[m["move"]] is m for m in chosen)
    return chosen


def answer(fmt: str, st: dict, m: dict) -> dict:
    content, spans = render(fmt, m, st["metrics0"])
    msg = {"role": "assistant", "content": content, "reasoning_content": ""}
    if spans is not None:
        msg["supervise_spans"] = spans
        msg["fixed_weight"] = FIXED_WEIGHT
    return msg


def conseq_row(st: dict, moves: list, fmt: str = "verbatim") -> dict:
    msgs = [{"role": "system", "content": st["system"]},
            {"role": "user", "content": consequence_user(st["user"]) + question(fmt, moves[0]["move"], True)},
            answer(fmt, st, moves[0])]
    for m in moves[1:]:
        msgs.append({"role": "user", "content": question(fmt, m["move"], False)})
        msgs.append(answer(fmt, st, m))
    return {"messages": msgs, "supervise": "every_assistant", "domain": st["variant"], "tier": "con1_conseq",
            "instance": st["problem_id"], "sid": st["sid"], "format": fmt,
            "con1": {"moves": [m["move"] for m in moves], "verdicts": [m["verdict"] for m in moves],
                     "src": [m["src"] for m in moves]}}


def est_tokens(st: dict, moves: list) -> int:
    n = len(st["system"]) + len(consequence_user(st["user"])) + len(question("verbatim", "", True))
    n += sum(len(m["result"]) + len(question("verbatim", m["move"], False)) + 40 for m in moves)
    return int(n / CHARS_PER_TOKEN)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--policy", default=None, help="rows file holding tier ownrows_own rows (e.g. train_chk1.jsonl.gz); "
                    "none: consequence rows only (ablations)")
    ap.add_argument("--target", choices=FORMATS, default="verbatim", help="answer format (targets.py)")
    ap.add_argument("--families", nargs="*", default=None, help="only these training families (ablations)")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--per-family", type=int, default=2000)
    ap.add_argument("--q", type=int, default=12)
    ap.add_argument("--min-q", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--policy-ratio", type=float, default=0.33)
    ap.add_argument("--dev-max", type=int, default=120)
    ap.add_argument("--dev-policy", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0, help="state selection (and moves unless --move-seed)")
    ap.add_argument("--move-seed", type=int, default=None, help="the move draw's own seed (a second row set)")
    ap.add_argument("--exclude-rows", nargs="*", default=None,
                    help="rows files of an earlier set: only their states are used, and their moves are not drawn again")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed if a.move_seed is None else a.move_seed)
    excl = None
    if a.exclude_rows:
        excl = collections.defaultdict(set)
        for p in a.exclude_rows:
            for line in opener(p):
                r = json.loads(line)
                if r.get("tier") == "con1_conseq":
                    excl[r["sid"]].update(r["con1"]["moves"])
        print("excluding the moves of %d states (%s)" % (len(excl), " ".join(a.exclude_rows)))
    c = collections.Counter()
    pools = collections.defaultdict(list)                  # (split, family) -> states (only what the rows need)
    def lines_of(path):
        try:
            for line in opener(path):
                yield line
        except EOFError:                                   # a file still being written: what is complete is used
            print("note: %s ends in an incomplete gzip member (still being written)" % path)

    seen_sid = set()
    for path in a.states:
        for line in lines_of(path):
            st = json.loads(line)
            if st["sid"] in seen_sid:                      # the same state from two machines' outputs: first wins
                c["duplicate states skipped"] += 1
                continue
            seen_sid.add(st["sid"])
            v = st["variant"]
            if v in HELD_OUT or v not in TRAIN_FAMILIES:
                raise SystemExit("%s: state of family %s in the training states" % (path, v))
            if st["split"] not in ("train", "dev"):
                raise SystemExit("%s: split %s in the training states" % (path, st["split"]))
            if not st["moves"]:
                c["states without moves"] += 1
                continue
            pools[(st["split"], v)].append(st)
    rows, dev = [], []
    for v in TRAIN_FAMILIES:
        if a.families and v not in a.families:
            continue
        for split, cap, out in (("train", a.per_family, rows), ("dev", None, dev)):
            sts = pools.get((split, v), [])
            # the state selection has its own rng per (split, family): it does not depend on how many moves earlier
            # states had (so a selection can be augmented with off-menu moves and rebuilt to the same states)
            sts = sorted(sts, key=lambda s: s["sid"])
            random.Random("%d-%s-%s" % (a.seed, split, v)).shuffle(sts)
            n = 0
            for st in sts:
                if cap is not None and n >= cap:
                    break
                if excl is not None and st["sid"] not in excl:
                    continue                               # a second set keeps the earlier set's states
                q = a.q
                moves = pick_moves(st, q, rng, excl[st["sid"]] if excl is not None else frozenset())
                while est_tokens(st, moves) > a.max_tokens and len(moves) > a.min_q:
                    moves = moves[:-1]
                if est_tokens(st, moves) > a.max_tokens:
                    c["dropped: too long (%s)" % v] += 1
                    continue
                if excl is not None and len(moves) < a.min_q:
                    c["dropped: under %d moves left (%s)" % (a.min_q, v)] += 1
                    continue
                out.append(conseq_row(st, moves, a.target))
                n += 1
                c["%s rows %s" % (split, v)] += 1
                c["%s questions %s" % (split, v)] += len(moves)
                for m in moves:
                    c["%s %s verdict %s" % (split, v, m["verdict"])] += 1
                    c["%s %s src %s" % (split, v, "+".join(m["src"]))] += 1
                c["%s est tokens %s" % (split, v)] += est_tokens(st, moves)
            c["%s states available %s" % (split, v)] = len(sts)
    rng.shuffle(dev)
    dev = dev[:a.dev_max]
    # policy rows
    pol = [json.loads(l) for l in opener(a.policy)] if a.policy else []
    pol = [r for r in pol if r.get("tier") == "ownrows_own"]
    for r in pol:
        if r.get("domain") not in TRAIN_FAMILIES:
            raise SystemExit("policy row of family %s" % r.get("domain"))
    rng.shuffle(pol)
    n_pol = max(0, min(len(pol) - a.dev_policy, int(round(a.policy_ratio * len(rows)))))
    dev += pol[:a.dev_policy] if pol else []
    prow = pol[a.dev_policy:a.dev_policy + n_pol]
    for r in prow:
        c["train policy rows %s" % r["domain"]] += 1
    rows += prow
    rng.shuffle(rows)
    for path, rs in ((a.out_train, rows), (a.out_dev, dev)):
        with open(path + ".tmp", "w", encoding="utf-8") as fo:
            for r in rs:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
    for k in sorted(c):
        print("%-50s %d" % (k, c[k]))
    tot = sum(v for k, v in c.items() if k.startswith("train est tokens"))
    print("train rows %d (consequence %d, policy %d), dev rows %d; consequence est. tokens %.1fM" % (
        len(rows), len(rows) - len(prow), len(prow), len(dev), tot / 1e6))
    print("BUILD_ROWS_DONE")


if __name__ == "__main__":
    main()
