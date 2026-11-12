"""Step 1 of a screened supervised round: the designs a policy reached in its own episodes, as
state rows in the datav4 own-rows format (dataxp.ownrows), each with the prompt evaluation would
serve at that design.

An episode's applied moves are replayed through the harness's own `replay` policy with
record_prompts on, so the system and user text of every state are produced by the code path that
serves evaluation (observation + instruction, as the dx38 rows hold them). A state is a problem
plus the moves applied so far: the same design reached by several episodes, or held for several
turns because a move was refused, is one state. The harness stops prompting at a feasible design,
so feasible states do not appear. A state the source episode shows as ended ("the episode has
ended": a diverged power flow) is left out and counted.

    <sim python> policy_states.py --episodes FILE [FILE ...] --script HARNESS.py --out states.jsonl \\
        [--exclude-specs problems_eval.jsonl ...] [--workers 24] [--limit N]

Each FILE is a rollout / evaluation record file (one episode per line); its problem specs are read
from the request file beside it (<FILE>.request.json, or step_NNNNN.request.json for a run's
rollouts). --exclude-specs: problem files ({variant, spec} or spec per line); an episode on a
problem whose canonical spec equals one of them is refused (evaluation problems never give rows).
"""
import argparse
import collections
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENDED = "the episode has ended"


def request_of(path: str) -> str:
    cands = [path + ".request.json"]
    for suffix in (".compact.jsonl", ".jsonl"):
        if path.endswith(suffix):
            cands.append(path[:-len(suffix)] + ".request.json")
    for cand in cands:
        if os.path.exists(cand):
            return cand
    raise SystemExit("no request file beside %s" % path)


def canon(spec: dict) -> str:
    """The evaluation harness's canonical form of a problem (fresh_problems.canon / select_states)."""
    return json.dumps({k: v for k, v in spec.items() if k not in ("problem_id", "domain_variant", "index")},
                      sort_keys=True, default=str)


def applied_moves(turns: list) -> tuple[list, list, set]:
    """(moves applied in order, for every turn the number of moves applied before it, the counts at
    which the episode shows itself ended)."""
    moves, before, ended = [], [], set()
    for t in turns:
        before.append(len(moves))
        fail = t.get("fail")
        if fail is None and t.get("action") is not None:
            moves.append(t["action"])
        elif fail and ENDED in str(fail):
            ended.add(len(moves))
    return moves, before, ended


def state_id(variant: str, pid: str, history: list) -> str:
    h = hashlib.sha1(json.dumps(history, sort_keys=True).encode()).hexdigest()[:12]
    return "%s|%s|%d-%s" % (variant, pid, len(history), h)


def _job(job):
    script, argv, rec, spec = job
    sys.path.insert(0, REPO)
    from llm_finetune.rl_dataxp import harness as HG
    H = HG.load_harness(script)
    ep = rec["episode"]
    turns = ep.get("turns") or []
    moves, before, ended = applied_moves(turns)
    a = HG.harness_args(H, argv + ["--turns", str(len(moves) + 1)])
    a.policy, a.replay_actions, a.record_prompts = "replay", moves, True
    mods = H.import_dataxp(a.code)
    domain = mods[3](rec["variant"])
    out = H.run_episode(mods, rec["variant"], spec, a, None, HG.instruction_for_variant(H, a, mods[0], domain))
    rows, mismatch = [], False
    phi = out["phi_start"]
    for j, t in enumerate(out["turns"]):
        if j > len(moves):
            break
        if j not in ended and t.get("user"):
            # the policy's own turns at this state, in the source episode
            own = [x for x, b in zip(turns, before) if b == j]
            rows.append({
                "sid": state_id(rec["variant"], spec["problem_id"], moves[:j]),
                "variant": rec["variant"], "problem_id": spec["problem_id"], "k": j, "turn": j, "split": "train",
                "native": bool(out.get("native")), "spelling": os.environ.get("DATAXP_SPELLING"),
                "spec": spec, "history": moves[:j],
                "system": t["system"], "observation": t["user"], "instruction": "",
                "instruction_sha1": out.get("instruction_sha1"), "binding_constraint": t.get("binding"),
                "phi_before": float(phi), "n_prior_changes": j,
                "source": rec["key"], "source_turns": len(own),
                "source_refused": sum(1 for x in own if str(x.get("fail") or "").startswith("illegal")),
                "own_action": moves[j] if j < len(moves) else None,
            })
        if j < len(moves) and t.get("fail") is not None:       # the recorded move did not replay
            mismatch = True
            break
        phi = t["phi_after"]
    return rec["key"], rows, mismatch, len(moves)


def main(argv=None) -> int:
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--script", required=True, help="the harness script (dataxp_policy_eval_v3.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-specs", nargs="*", default=[])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="first N episodes of every file (a smoke run)")
    a = ap.parse_args(argv)

    banned = set()
    for path in a.exclude_specs:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                banned.add(canon(r.get("spec", r)))
    jobs, cnt = [], collections.Counter()
    for path in a.episodes:
        req = json.load(open(request_of(path)))
        specs = {e["key"]: e["spec"] for e in req["episodes"]}
        # --reject-note (episodes collected with the refusal note) is dropped: the states' prompts are
        # the ones evaluation serves, which carry no note
        hargv = [x for x in req["harness_argv"] if x != "--reject-note"] + [
            "--policy", "llm", "--model", "x", "--base-url", "http://x", "--temperature", "0", "--out", os.devnull]
        n = 0
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if a.variants and rec["variant"] not in a.variants:
                continue
            cnt["episodes"] += 1
            if rec["episode"].get("crashed") or rec.get("rl_error") or not rec["episode"].get("turns"):
                cnt["episodes without turns (crashed, or feasible at the start)"] += 1
                continue
            spec = specs[rec["key"]]
            if canon(spec) in banned:
                raise SystemExit("%s: %s is an excluded (evaluation) problem" % (path, spec["problem_id"]))
            jobs.append((a.script, hargv, {k: rec[k] for k in ("key", "variant", "episode")}, spec))
            n += 1
            if a.limit and n >= a.limit:
                break
    seen, n_rows = {}, 0
    tmp = a.out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with mp.get_context("spawn").Pool(a.workers) as pool, open(tmp, "w", encoding="utf-8") as fo:
        for key, rows, mismatch, n_moves in pool.imap(_job, jobs, chunksize=2):
            cnt["episodes replayed"] += 1
            cnt["applied moves"] += n_moves
            cnt["episodes whose recorded moves did not all replay"] += int(mismatch)
            for r in rows:
                cnt["states visited"] += 1
                if r["sid"] in seen:
                    seen[r["sid"]] += 1
                    continue
                seen[r["sid"]] = 1
                fo.write(json.dumps(r) + "\n")
                n_rows += 1
                cnt["states, %s" % r["variant"]] += 1
    os.replace(tmp, a.out)
    cnt["distinct states written"] = n_rows
    for k in sorted(cnt):
        print("%-58s %d" % (k, cnt[k]))
    print("POLICY_STATES_DONE %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
