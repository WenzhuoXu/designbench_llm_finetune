"""Small preparation steps for appended own-row batches (docs/datav4_append_plan.md).
Pure file work: no domain is instantiated, no model is called.

unrowed      the states of an earlier batch that have no row in ANY of the given tier
             files (pass the tiers of both spellings, so the result is one state set for
             both), with their Phi-cache lines; b002 = b001's states without a row.

    python -m dataxp.ownrows.batch_prep unrowed --states <b001 group>/states.jsonl \
        [--states-phi <b001 group>/states_phi.jsonl] --rows <tier jsonl files of both spellings> \
        --out <new group dir> [--relabel-spelling multitool]

seed-shards  write states.shard<i>.jsonl (shard.py's rule: line i mod N) and
             samples.shard<i>.jsonl holding EVERY record (finished or truncated) of the
             shard's states from the given sample files, so that a sampler run on the shard
             files redraws exactly the keys whose reply was truncated and labels them
             draw_pass 2 (dataxp.ownrows.sample counts the records already present). Existing
             samples.shard*.jsonl in the row dir are moved to --park first (never deleted).
             Use it for b002 (seeded from b001's samples) and before a redraw job whose GPU
             count differs from the first job's.

    python -m dataxp.ownrows.batch_prep seed-shards --rowdir <group dir> --shards N \
        --samples <samples.jsonl> [...] --park <dir outside the row dir>

phi-from     the second spelling's Phi cache without recomputing it: each line of
             --states (the set_parameter states.jsonl) + the Phi-only fields of the
             same sid in --phi (the multitool states_phi.jsonl). The legal set and its
             Phi do not depend on the spelling; the state record (served prompt) does,
             so the record always comes from --states (the b001 lesson: a cache copied
             whole carried stale prompts; docs/corpus_native_domains.md).

    python -m dataxp.ownrows.batch_prep phi-from --states <rows_sp grp>/states.jsonl \
        --phi <rows grp>/states_phi.jsonl --out <rows_sp grp>/states_phi.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

# b001's multitool states were selected before the spelling label existed or under its first
# names; their served text is the multitool one (package_all.sh relabelled the b001 rows the same way)
LEGACY_LABELS = (None, "verb_surface", "native")
INSTRUCTION_SHA1 = {"multitool": "1c344388a444", "set_parameter": "fe8fa8db0a90"}  # docs/dataxp_spellings.md
PHI_KEYS = ("moves", "gold_phi", "gold_pct", "gold_in_legal", "n_candidates", "n_illegal_enumerated",
            "n_nan_phi", "offset_rule")


def _lines(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line if line.endswith("\n") else line + "\n"


def _row_sid(o):
    return "%s|%s|%d" % (o["domain"], o["instance"], int(o["turn"]))


def _relabel(line: str, spelling: str):
    """(line, 1) with the state's legacy spelling label (missing, 'verb_surface',
    'native': b001's multitool states predate the label) replaced by `spelling`
    after checking that its instruction sha1 is that spelling's; (line, 0) when
    the label is already right. Any other label is refused."""
    o = json.loads(line)
    sp = o.get("spelling")
    if sp == spelling:
        return line, 0
    if sp not in LEGACY_LABELS:
        raise SystemExit("unrowed: %s is labelled %r, not %s" % (o.get("sid"), sp, spelling))
    if o.get("instruction_sha1") != INSTRUCTION_SHA1[spelling]:
        raise SystemExit("unrowed: %s carries instruction %s, not the %s one (%s); not relabelled" % (
            o.get("sid"), o.get("instruction_sha1"), spelling, INSTRUCTION_SHA1[spelling]))
    o["spelling"] = spelling
    return json.dumps(o) + "\n", 1


def cmd_unrowed(a) -> int:
    rowed = set()
    for p in a.rows:
        for line in _lines(p):
            rowed.add(_row_sid(json.loads(line)))
    os.makedirs(a.out, exist_ok=True)
    keep, n, relab = [], 0, 0
    for line in _lines(a.states):
        n += 1
        if json.loads(line)["sid"] not in rowed:
            if a.relabel_spelling:
                line, k = _relabel(line, a.relabel_spelling)
                relab += k
            keep.append(line)
    sids = [json.loads(l)["sid"] for l in keep]
    with open(os.path.join(a.out, "states.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(keep)
    msg = "unrowed: %d of %d states have no row in %d tier files -> %s" % (
        len(keep), n, len(a.rows), os.path.join(a.out, "states.jsonl"))
    if a.states_phi:
        phi = {}
        for line in _lines(a.states_phi):
            phi[json.loads(line)["sid"]] = line
        missing = [s for s in sids if s not in phi]
        if missing:
            raise SystemExit("unrowed: %d states have no Phi-cache line (%s)" % (len(missing), missing[:3]))
        out = []
        for sid in sids:
            line = phi[sid]
            if a.relabel_spelling:
                line, _ = _relabel(line, a.relabel_spelling)
            out.append(line)
        with open(os.path.join(a.out, "states_phi.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(out)
        msg += " (+ states_phi.jsonl)"
    if a.relabel_spelling:
        msg += "; %d legacy spelling labels set to %s" % (relab, a.relabel_spelling)
    print(msg, flush=True)
    return 0


def cmd_seed_shards(a) -> int:
    n = int(a.shards)
    states = list(_lines(os.path.join(a.rowdir, "states.jsonl")))
    shard_of = {}
    for i in range(n):
        part = states[i::n]
        with open(os.path.join(a.rowdir, "states.shard%d.jsonl" % i), "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(part)
        for line in part:
            shard_of[json.loads(line)["sid"]] = i
    old = sorted(f for f in os.listdir(a.rowdir) if f.startswith("samples.shard") and f.endswith(".jsonl"))
    if old:
        if not a.park:
            raise SystemExit("seed-shards: %s holds %s; pass --park <dir> to move them aside" % (a.rowdir, old))
        park = os.path.join(a.park, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(park, exist_ok=True)
        for f in old:
            shutil.move(os.path.join(a.rowdir, f), os.path.join(park, f))
        print("seed-shards: moved %d old shard files to %s" % (len(old), park), flush=True)
    outs = [open(os.path.join(a.rowdir, "samples.shard%d.jsonl" % i), "w", encoding="utf-8", newline="\n")
            for i in range(n)]
    cnt = [0] * n
    skipped = 0
    try:
        for p in a.samples:
            for line in _lines(p):
                try:
                    o = json.loads(line)
                except ValueError:
                    skipped += 1
                    continue
                i = shard_of.get(o.get("sid"))
                if i is None:
                    continue
                outs[i].write(line)
                cnt[i] += 1
    finally:
        for fh in outs:
            fh.close()
    print("seed-shards: %d states -> %d shards (%s states); %s sample records seeded%s" % (
        len(states), n, "/".join(str(len(states[i::n])) for i in range(n)), "/".join(map(str, cnt)),
        ", %d unreadable lines skipped" % skipped if skipped else ""), flush=True)
    return 0


def cmd_phi_from(a) -> int:
    cache = {}
    for line in _lines(a.phi):
        o = json.loads(line)
        cache[o["sid"]] = {k: o[k] for k in PHI_KEYS if k in o}
    out, miss = [], []
    for line in _lines(a.states):
        o = json.loads(line)
        c = cache.get(o["sid"])
        if c is None:
            miss.append(o["sid"])
            continue
        o.update(c)
        out.append(json.dumps(o) + "\n")
    if miss:
        raise SystemExit("phi-from: %d states have no cache line in %s (%s); run cand_phi for them"
                         % (len(miss), a.phi, miss[:3]))
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(out)
    print("phi-from: %d states -> %s (Phi fields from %s)" % (len(out), a.out, a.phi), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unrowed")
    u.add_argument("--states", required=True)
    u.add_argument("--states-phi")
    u.add_argument("--rows", nargs="+", required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--relabel-spelling", choices=("multitool", "set_parameter"),
                   help="set a legacy spelling label (missing / verb_surface / native) to this spelling, "
                        "after checking the state's instruction sha1 (b001's multitool states need it)")
    s = sub.add_parser("seed-shards")
    s.add_argument("--rowdir", required=True)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--samples", nargs="+", required=True)
    s.add_argument("--park")
    f = sub.add_parser("phi-from")
    f.add_argument("--states", required=True)
    f.add_argument("--phi", required=True)
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"unrowed": cmd_unrowed, "seed-shards": cmd_seed_shards, "phi-from": cmd_phi_from}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
