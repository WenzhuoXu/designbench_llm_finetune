"""Step 1: pick repair-phase states from an accepted corpus and write them as
replayable state specs.

Selection rule (sftprobe_states.py): a state is every turn of a trajectory
whose binding_constraint is not None (the repair phase, as render_cot's
iter_states stratifies). Per variant, whole problems are drawn at random
(seeded) until >= --per-domain train states, then further problems until
>= --dev-per-domain dev states; train and dev problems are disjoint.
--per-domain 0 puts every problem in train.

Each state record carries what every later step needs without the corpus:
the problem spec, the action history up to the turn (replay with
dataxp.domain.apply_action), the teacher's move at the turn, the metrics
before and after it, and the served prompt parts (system head, observation
body, instruction) built with render_cot.policy_observation.

    python -m dataxp.ownrows.select_states --data <out dir of run_generation> \
        --variants molopt circuit2 --out <dir> [--tmp <scratch of run_generation>] \
        [--per-domain 120 --dev-per-domain 12 --seed 0]

Selecting for a later batch over a corpus an earlier batch already drew from
(docs/datav4_append_plan.md):

  --exclude-problems F ...  problem ids already used by earlier batches, any
                            split: a .jsonl of states / tier rows / problem
                            specs (problem_id, instance or spec.problem_id per
                            line) or a text file with one id per line. Those
                            problems are removed before the draw, so the
                            selection takes only unused problems.
  --exclude-specs F ...     problem files (spec dicts or {variant, spec} per
                            line, e.g. problems_v3_native.jsonl,
                            problems_v2.jsonl): a corpus problem whose
                            canonical spec (json of the spec minus problem_id /
                            domain_variant / index, sorted keys - the
                            evaluation harness's canon()) equals one of them is
                            removed before the draw (evaluation problems are
                            never used for rows).
  --split-from F ...        prior states.jsonl files (split per line) or tier
                            rows (split from the file name: train_* / dev_in_*):
                            a problem found there keeps that split if this draw
                            takes it. It is taken in its turn of the seeded
                            shuffle while its split's quota is open, and skipped
                            (not forced in) once that quota is full.

None of the three given = the draw of before, state for state.
"""

from __future__ import annotations

import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import collections
import json
import math
import sys

import numpy as np

from ..domain import Move, apply_action, stable_seed
from ..domains import get_domain
from ..render_cot import _label_for
from .common import (instruction_for, is_native, load_corpus, observation_parts, sid_of,
                     spelling_of, write_jsonl)


def _spelling(domain) -> str:
    """How the domain's actions are written in this process: the
    DATAXP_SPELLING (multitool | set_parameter) for a domain that speaks
    verbs (native or verb surface; the `native` field tells which), SET_PARAM
    for a plain domain (docs/dataxp_spellings.md)."""
    return spelling_of(domain)


def _repair_idx(traj: dict):
    return [k for k, tt in enumerate(traj["turns"]) if tt.get("binding_constraint") is not None]


def _metrics_match(recorded: dict, replayed: dict, tol: float = 1e-6) -> bool:
    for k, v in recorded.items():
        if k not in replayed:
            continue
        try:
            a, b = float(v), float(replayed[k])
        except (TypeError, ValueError):
            continue
        if math.isnan(a) and math.isnan(b):
            continue
        if abs(a - b) > tol * max(1.0, abs(a)):
            return False
    return True


def states_of(domain, variant: str, spec: dict, traj: dict, split: str,
              replay_check: bool = True):
    """Every repair-phase state of one trajectory, in turn order."""
    pid = traj["problem_id"]
    turns = traj["turns"]
    rep = set(_repair_idx(traj))
    if not rep:
        return [], 0
    last = max(rep)
    instruction, isha = instruction_for(domain)
    st = domain.load(spec)
    hist = []
    out, mism = [], 0
    for k in range(last + 1):
        turn = turns[k]
        if k > 0:
            a = turns[k - 1]["action"]
            assert traj["actions"][k - 1] == a, "actions/turns misaligned in %s" % pid
            apply_action(domain, st, a)
            hist.append((_label_for(domain, st, a), a))
        if k not in rep:
            continue
        t = dict(turn)
        t["problem_id"] = pid
        prev = None
        if k > 0 and turns[k - 1]["turn"] == turn["turn"] - 1:
            prev = turns[k - 1].get("state_metrics")
            t["prev_state_metrics"] = prev
        if replay_check and not domain.stochastic:
            m = domain.evaluate(domain.clone(st), seed=0)
            if not _metrics_match(turn.get("state_metrics") or {}, m):
                mism += 1
        system, observation = observation_parts(domain, spec, st, t, hist, variant)
        gold = turn["action"]
        out.append({
            "sid": sid_of(variant, pid, turn["turn"]),
            "variant": variant, "problem_id": pid, "k": k, "turn": int(turn["turn"]),
            "split": split, "native": is_native(domain), "spelling": _spelling(domain),
            "spec": spec, "history": [a for _, a in hist],
            "gold": gold, "gold_text": domain.format_move(Move.from_dict(gold)),
            "binding_constraint": turn.get("binding_constraint"),
            "state_metrics": turn.get("state_metrics"),
            "prev_state_metrics": prev,
            "gold_outcome": turns[k + 1]["state_metrics"] if k + 1 < len(turns) else None,
            "n_prior_changes": len(hist),
            "system": system, "observation": observation,
            "instruction": instruction, "instruction_sha1": isha,
            "obs_chars": len(system) + 2 + len(observation),
        })
    return out, mism


def select(domain, variant, specs, trajs, per_domain, dev_per_domain, seed,
           exclude=frozenset(), forced=None):
    """(train problems, dev problems): the sftprobe draw - problems shuffled with
    the seed, whole problems into train until the quota, then into dev.

    `exclude`: problem ids left out before the shuffle. `forced`: {pid: split}
    of problems an earlier batch already placed; such a problem goes to that
    split while its quota is open and is skipped otherwise. Both empty = the
    draw of before."""
    forced = forced or {}
    by_pid = {t["problem_id"]: t for t in trajs if _repair_idx(t) and t["problem_id"] not in exclude}
    insts = sorted(by_pid)
    rng = np.random.default_rng(stable_seed("ownrows_select", variant, int(seed)))
    rng.shuffle(insts)
    train, dev = [], []
    n_tr = n_dev = 0
    for pid in insts:
        n = len(_repair_idx(by_pid[pid]))
        if pid in forced:
            if forced[pid] == "train" and (per_domain <= 0 or n_tr < per_domain):
                train.append(pid)
                n_tr += n
            elif forced[pid] == "dev" and n_dev < dev_per_domain:
                dev.append(pid)
                n_dev += n
            continue
        if per_domain <= 0 or n_tr < per_domain:
            train.append(pid)
            n_tr += n
        elif n_dev < dev_per_domain:
            dev.append(pid)
            n_dev += n
        else:
            break          # both quotas full: no later problem, forced or not, can be taken
    return train, dev, by_pid


def canon_spec(spec: dict) -> str:
    """The evaluation harness's canon(): the spec without its id fields, as
    sorted-key JSON (dataxp_policy_eval_v3.fresh_problems)."""
    return json.dumps({k: v for k, v in spec.items()
                       if k not in ("problem_id", "domain_variant", "index")},
                      sort_keys=True, default=str)


def _records(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _pid_of(o: dict):
    return o.get("problem_id") or o.get("instance") or (o.get("spec") or {}).get("problem_id")


def read_problem_ids(paths) -> set:
    """Problem ids from .jsonl (states, tier rows, specs) or text files."""
    out = set()
    for p in paths or []:
        if p.endswith(".jsonl") or p.endswith(".json"):
            for o in _records(p):
                pid = _pid_of(o)
                if pid:
                    out.add(pid)
        else:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        out.add(s)
    return out


def read_canon_specs(paths) -> set:
    out = set()
    for p in paths or []:
        for o in _records(p):
            spec = o["spec"] if isinstance(o.get("spec"), dict) else o
            out.add(canon_spec(spec))
    return out


def read_prior_splits(paths) -> dict:
    """{problem_id: 'train' | 'dev'} from prior states files (split field) or
    tier rows (split from the file name). A problem with two splits is an
    error."""
    out = {}
    for p in paths or []:
        base = os.path.basename(p)
        by_name = "dev" if base.startswith("dev") else ("train" if base.startswith("train") else None)
        for o in _records(p):
            pid = _pid_of(o)
            sp = o.get("split") or by_name
            if not pid or sp is None:
                continue
            if sp not in ("train", "dev"):
                raise SystemExit("%s: split %r of %s is neither train nor dev" % (p, sp, pid))
            if out.setdefault(pid, sp) != sp:
                raise SystemExit("--split-from: %s is %s in one file and %s in %s"
                                 % (pid, out[pid], sp, p))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="run_generation --out dir (<data>/<variant>/accepted)")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None,
                    help="run_generation --tmp dir: used when a variant has no accepted/ (gates failed)")
    ap.add_argument("--per-domain", type=int, default=120, help="train states per variant (0 = every problem)")
    ap.add_argument("--dev-per-domain", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-replay-check", action="store_true",
                    help="skip the recorded-vs-replayed metrics check (deterministic domains)")
    ap.add_argument("--exclude-problems", nargs="+", default=None, metavar="FILE",
                    help="problem ids already used by earlier batches (any split): .jsonl of "
                         "states / rows / specs, or one id per line")
    ap.add_argument("--exclude-specs", nargs="+", default=None, metavar="FILE",
                    help="problem files (e.g. problems_v3_native.jsonl, problems_v2.jsonl): corpus "
                         "problems with an equal canonical spec are left out")
    ap.add_argument("--split-from", nargs="+", default=None, metavar="FILE",
                    help="prior states.jsonl / tier rows: a problem found there keeps its split")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    excl_ids = read_problem_ids(a.exclude_problems)
    excl_canon = read_canon_specs(a.exclude_specs)
    forced = read_prior_splits(a.split_from)
    appending = bool(a.exclude_problems or a.exclude_specs or a.split_from)
    if appending:
        print("append selection: %d excluded ids, %d excluded specs, %d problems with a prior split"
              % (len(excl_ids), len(excl_canon), len(forced)), flush=True)

    rows, report = [], {}
    for variant in a.variants:
        domain = get_domain(variant)
        src, specs, trajs = load_corpus(a.data, variant, a.tmp)
        excl = {pid for pid in specs if pid in excl_ids}
        excl_spec = {pid for pid, s in specs.items() if excl_canon and canon_spec(s) in excl_canon}
        train, dev, by_pid = select(domain, variant, specs, trajs, a.per_domain,
                                    a.dev_per_domain, a.seed, exclude=excl_ids | excl_spec,
                                    forced=forced)
        cnt = collections.Counter()
        mism = 0
        for split, pids in (("train", train), ("dev", dev)):
            for pid in pids:
                sts, m = states_of(domain, variant, specs[pid], by_pid[pid], split,
                                   replay_check=not a.no_replay_check)
                rows += sts
                cnt[split] += len(sts)
                mism += m
        report[variant] = {"source": src, "problems_train": len(train), "problems_dev": len(dev),
                           "states_train": cnt["train"], "states_dev": cnt["dev"],
                           "replay_mismatches": mism, "native": is_native(domain),
                           "spelling": _spelling(domain),
                           "instruction_sha1": instruction_for(domain)[1]}
        if appending:
            taken = set(train) | set(dev)
            report[variant].update({
                "corpus_problems": len(specs),
                "excluded_used": len(excl), "excluded_eval_spec": len(excl_spec),
                "excluded_eval_spec_ids": sorted(excl_spec),
                "prior_split_taken": sum(1 for p in taken if p in forced),
                "prior_split_skipped": sum(1 for p in by_pid if p in forced and p not in taken),
                "unused_after": sorted(p for p in by_pid if p not in taken)})
            print("%-10s excluded: %d used, %d equal to an evaluation spec; %d prior-split problems taken"
                  % (variant, len(excl), len(excl_spec), report[variant]["prior_split_taken"]),
                  flush=True)
        print("%-10s %s: problems train %d dev %d | states train %d dev %d | replay mismatches %d"
              % (variant, _spelling(domain), len(train), len(dev),
                 cnt["train"], cnt["dev"], mism), flush=True)
    write_jsonl(os.path.join(a.out, "states.jsonl"), rows)
    args = {k: v for k, v in vars(a).items()
            if v is not None or k not in ("exclude_problems", "exclude_specs", "split_from")}
    with open(os.path.join(a.out, "select_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"args": args, "per_variant": report, "n_states": len(rows)}, fh, indent=1)
    print("wrote %d states -> %s" % (len(rows), os.path.join(a.out, "states.jsonl")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
