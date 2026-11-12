"""First replies of a run's stored rollouts as a run directory for the gradient probe: is the
cross-entropy on a simulator-screened own reply a signal that problems share?

The 8 rollouts of a problem start from one state, so their first replies are 8 samples of the
policy at that state, each with the simulator's verdict (refused / applied, potential after). For
every problem two replies go to the probe:
    sel   the applied reply with the highest potential after it, if it raised the potential
          (what a screened supervised round would train on)
    rnd   another reply of the same problem, picked by a hash (an unscreened own sample)
A problem with no gaining reply, or fewer than two usable replies, is left out and counted.

Only every tenth step of a run keeps token ids. For the other steps the tokens are rebuilt and
checked against the recorded lengths: the prompt from the harness's own observation of the start
state (stage `prompts`, simulator env), the completion from the recorded text (stage `build`,
tokenizer env). A reply whose record is cut (reasoning 4000 chars, answer 600) or whose rebuilt
length differs is not usable. On the steps that do keep token ids the rebuilt tokens are compared
with the real ones and the agreement is printed.

    <sim python>   first_move_run.py prompts --run-dir RUN --script HARNESS.py --out DIR [--workers 16]
    <train python> first_move_run.py build --run-dir RUN --tokenizer MODEL --adapter ADAPTER --out DIR [--parts 8]

Writes DIR/prompts.jsonl, DIR/rollouts/step_0000<part>.jsonl, DIR/labels.jsonl and the link
DIR/ckpt/step_00000/adapter.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys
import zlib

END = "<|im_end|>"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def step_files(run_dir: str) -> list[tuple[int, str, bool]]:
    """(step, rollout file, has token ids) for every step of the run, the full file when kept."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "rollouts", "step_*.jsonl")):
        name = os.path.basename(f)
        step = int(name[5:10])
        full = not name.endswith(".compact.jsonl")
        if full or step not in out:
            out[step] = (step, f, full)
    return [out[s] for s in sorted(out)]


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def first_turn(rec: dict):
    """(turn record, call record) of the first reply, or None when the episode has none."""
    turns, calls = rec["episode"].get("turns") or [], rec.get("calls") or []
    if rec.get("rl_error") or rec["episode"].get("crashed") or not turns or not calls:
        return None
    return turns[0], calls[0]


def outcome(rec: dict, turn: dict) -> dict:
    phi0, phi1 = rec["episode"].get("phi_start"), turn.get("phi_after")
    legal = not turn.get("fail")
    return {"legal": bool(legal), "phi_start": phi0, "phi_after": phi1,
            "gain": bool(legal and phi0 is not None and phi1 is not None and phi1 > phi0 + 1e-12),
            "feasible_after": bool(turn.get("feasible_after")), "fail": turn.get("fail"),
            "action": turn.get("action") or turn.get("move")}


def pick(group: str, cands: list[dict]):
    """(sel, rnd, why): cands are the usable replies of one problem, each with member / gain /
    phi_after. sel = highest potential among the gaining ones (ties: lowest member); rnd = one of
    the others by a hash of the group id."""
    if len(cands) < 2:
        return None, None, "fewer_than_two_usable"
    gain = [c for c in cands if c["gain"]]
    if not gain:
        return None, None, "no_gaining_reply"
    sel = sorted(gain, key=lambda c: (-c["phi_after"], c["member"]))[0]
    rest = sorted((c for c in cands if c is not sel), key=lambda c: c["member"])
    h = int(hashlib.sha1(group.encode()).hexdigest(), 16)
    return sel, rest[h % len(rest)], None


def completion_text(reasoning: str, answer: str) -> str:
    """The completion as the server produced it: the harness keeps the reasoning stripped and the
    answer as it followed </think>."""
    return reasoning + "\n</think>" + answer + END


# ------------------------------------------------------------------ stage prompts (simulator env)

class _Captured(Exception):
    pass


def _prompt_job(job):
    script, argv, variant, spec, group = job
    from llm_finetune.rl_dataxp import harness as HG
    H = HG.load_harness(script)
    a = HG.harness_args(H, argv)
    mods = H.import_dataxp(a.code)
    domain = mods[3](variant)
    box = {}

    def fake(system, user):
        box.update(system=system, user=user)
        raise _Captured()

    try:
        H.run_episode(mods, variant, spec, a, fake, HG.instruction_for_variant(H, a, mods[0], domain))
    except _Captured:
        pass
    return {"group": group, "variant": variant, "system": box.get("system"), "user": box.get("user")}


def cmd_prompts(a) -> int:
    import multiprocessing as mp
    sys.path.insert(0, REPO)
    jobs, seen = [], set()
    for step, _path, _full in step_files(a.run_dir):
        req = json.load(open(os.path.join(a.run_dir, "rollouts", "step_%05d.request.json" % step)))
        argv = list(req["harness_argv"]) + ["--policy", "llm", "--model", "x", "--base-url", "http://x",
                                            "--temperature", "0", "--turns", "1", "--out", os.devnull]
        for e in req["episodes"]:
            if e["group"] not in seen:
                seen.add(e["group"])
                jobs.append((a.script, argv, e["variant"], e["spec"], e["group"]))
    if a.limit:
        jobs = jobs[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    n = bad = 0
    with mp.get_context("spawn").Pool(a.workers) as pool, open(os.path.join(a.out, "prompts.jsonl.tmp"), "w", encoding="utf-8") as fo:
        for r in pool.imap_unordered(_prompt_job, jobs, chunksize=1):
            n += 1
            bad += r["user"] is None
            fo.write(json.dumps(r) + "\n")
    os.replace(os.path.join(a.out, "prompts.jsonl.tmp"), os.path.join(a.out, "prompts.jsonl"))
    print("prompts: %d problems, %d without a prompt (feasible at the start)" % (n, bad), flush=True)
    return 0


# ------------------------------------------------------------------ stage build (tokenizer env)

def cmd_build(a) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prompts = {r["group"]: r for r in read_jsonl(os.path.join(a.out, "prompts.jsonl"))}
    pcache = {}

    def prompt_ids(group):
        if group not in pcache:
            p = prompts.get(group)
            ids = None
            if p and p["user"] is not None:
                ids = tok.apply_chat_template(
                    [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}],
                    tokenize=True, add_generation_prompt=True)
                ids = list(ids if isinstance(ids, list) else ids["input_ids"])   # a BatchEncoding in newer versions
            pcache[group] = ids
        return pcache[group]

    cnt = collections.Counter()
    groups = collections.defaultdict(list)
    for step, path, full in step_files(a.run_dir):
        for rec in read_jsonl(path):
            ft = first_turn(rec)
            if ft is None:
                cnt["episodes without a first reply"] += 1
                continue
            turn, call = ft
            cnt["first replies"] += 1
            rs, an = turn.get("reasoning") or "", turn.get("answer") or ""
            cut = len(rs) >= 4000 or len(an) >= 600
            rebuilt_c = None if cut else tok.encode(completion_text(rs, an), add_special_tokens=False)
            rebuilt_p = prompt_ids(rec["group"])
            if full:
                p_ids, c_ids = call["prompt_token_ids"], call["token_ids"]
                cnt["with token ids"] += 1
                cnt["with token ids: prompt rebuilt"] += rebuilt_p is not None
                cnt["with token ids: prompt rebuilt identical"] += rebuilt_p == p_ids
                if rebuilt_c is not None:
                    cnt["with token ids: completion rebuilt"] += 1
                    cnt["with token ids: completion rebuilt identical"] += rebuilt_c == c_ids
                    cnt["with token ids: completion rebuilt, same length but different"] += (
                        rebuilt_c != c_ids and len(rebuilt_c) == len(c_ids))
            else:
                cnt["without token ids"] += 1
                if cut:
                    cnt["without token ids: record cut"] += 1
                    continue
                if rebuilt_p is None or len(rebuilt_p) != call.get("n_prompt"):
                    cnt["without token ids: prompt length differs"] += 1
                    continue
                if len(rebuilt_c) != call.get("n_completion"):
                    cnt["without token ids: completion length differs"] += 1
                    continue
                p_ids, c_ids = rebuilt_p, rebuilt_c
            if not c_ids or not p_ids:
                cnt["no tokens"] += 1
                continue
            cand = {"member": rec.get("member"), "key": rec["key"], "variant": rec["variant"],
                    "problem_id": rec.get("problem_id"), "step": step, "exact": bool(full),
                    "prompt_token_ids": p_ids, "token_ids": c_ids,
                    "episode_ok": rec["episode"].get("first_feasible_turn") is not None}
            cand.update(outcome(rec, turn))
            groups[rec["group"]].append(cand)
    cnt["problems"] = len(groups)
    cnt["usable first replies"] = sum(len(g) for g in groups.values())
    os.makedirs(os.path.join(a.out, "rollouts"), exist_ok=True)
    parts = collections.defaultdict(list)
    labels = []
    for group in sorted(groups):
        cands = groups[group]
        sel, rnd, why = pick(group, cands)
        if why:
            cnt["problems left out: " + why] += 1
            continue
        cnt["problems kept"] += 1
        part = zlib.crc32(group.encode()) % a.parts
        for role, c in (("sel", sel), ("rnd", rnd)):
            key = "%s/%s" % (group, role)
            parts[part].append({"key": key, "group": group, "variant": c["variant"],
                                "calls": [{"prompt_token_ids": c["prompt_token_ids"], "token_ids": c["token_ids"]}]})
            labels.append({"key": key, "group": group, "role": role, "part": part, "usable": len(cands),
                           "gaining": sum(x["gain"] for x in cands),
                           **{k: c[k] for k in ("member", "variant", "problem_id", "step", "exact", "legal", "gain",
                                                "phi_start", "phi_after", "feasible_after", "fail", "action", "episode_ok")}})
    for part in range(a.parts):
        with open(os.path.join(a.out, "rollouts", "step_%05d.jsonl" % part), "w", encoding="utf-8") as fo:
            for r in parts.get(part, []):
                fo.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "labels.jsonl"), "w", encoding="utf-8") as fo:
        for r in labels:
            fo.write(json.dumps(r) + "\n")
    link = os.path.join(a.out, "ckpt", "step_00000", "adapter")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(a.adapter), link)
    for k in sorted(cnt):
        print("%-62s %d" % (k, cnt[k]))
    fam = collections.Counter(r["variant"] for r in labels if r["role"] == "sel")
    print("problems kept per family: %s" % " ".join("%s %d" % kv for kv in sorted(fam.items())))
    rnd = [r for r in labels if r["role"] == "rnd"]
    print("the unscreened reply: applied %d of %d, raised the potential %d" % (
        sum(r["legal"] for r in rnd), len(rnd), sum(r["gain"] for r in rnd)))
    print("FIRST_MOVE_RUN_DONE parts=%d" % a.parts)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--script", required=True, help="the harness script the run used")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="first N problems only (a smoke run)")
    b = sub.add_parser("build")
    b.add_argument("--run-dir", required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--adapter", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--parts", type=int, default=8)
    a = ap.parse_args(argv)
    return cmd_prompts(a) if a.cmd == "prompts" else cmd_build(a)


if __name__ == "__main__":
    sys.exit(main())
