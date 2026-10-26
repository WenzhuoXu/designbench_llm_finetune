"""Assemble SFT conversations from rendered dataxp CoT rows.

The renderer (shared/dataxp/code/dataxp/render_cot.py) writes one row per repair-phase turn:
`prompt` (the full observation the writer saw), `cot` (60-150 words ending in one
`<action>SET_PARAM(i, p, v)</action>` line), `action`, `accepted`, ... This turns those rows
into the conversation format our tiers use, one conversation per turn:

    system    = the problem statement: DOMAIN, GOAL, the PROBLEM text and the fixed tables
                (nodes, groups, bus loads, equations) -- everything before the DESIGN block
    user      = the observation: DESIGN / RESULTS / CONSTRAINT STATUS / CHANGES ALREADY MADE
    assistant = reasoning_content = the cot minus its action line
                content           = the action line

which is exactly what the `assistant_after_think` target supervises: the prefix is the served
prompt for that turn, and the completion is `{reasoning}\n</think>\n\n{content}<|im_end|>`.

THE CHANGE BLOCK. The writer was shown "CHANGE YOU ARE ABOUT TO MAKE" -- the action itself --
and wrote the reasoning toward it. A policy at inference is not told its move, so by default
that block is REMOVED from the user message: the model must produce the target and the action
from the state. `--include-change-block` keeps it, which trains a rationaliser (given the
move, explain it), not a policy. The handoff's §6 wording ("the prompt up to the STYLE line")
literally includes the block; whether that was intended is an open question for wxu7, and the
flag exists so both readings can be built and compared.

    python scripts/assemble_dataxp_sft.py --cot <dir-or-files> --out tier.jsonl
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import statistics

ACTION = re.compile(r"<action>\s*SET_PARAM\([^)]*\)\s*</action>")
STYLE = "\n\nSTYLE FOR THIS ONE:"
CHANGE = "\n\nCHANGE YOU ARE ABOUT TO MAKE"
# Every domain's describe() prints the fixed problem statement (setup, tech, equations,
# node/group/load tables) and then a DESIGN block with the current parameter values, followed
# by RESULTS. DESIGN is the first thing that changes turn to turn, so it starts the user turn.
DESIGN_HEADER = re.compile(r"^DESIGN\b", re.M)


def split_prompt(prompt, include_change):
    """(system, user) from the renderer's prompt; None if the layout is not recognised."""
    if STYLE not in prompt:
        return None
    obs = prompt.split(STYLE, 1)[0]
    if not include_change:
        if CHANGE not in obs:
            return None
        obs = obs.split(CHANGE, 1)[0]
    m = DESIGN_HEADER.search(obs)
    if not m:
        return None
    return obs[: m.start()].rstrip(), obs[m.start():].strip()


def convert(row, include_change, instruction=""):
    sp = split_prompt(row["prompt"], include_change)
    if sp is None:
        return None, "layout"
    system, user = sp
    # The task instruction, appended verbatim to the user turn. Without one the prompt never says
    # what to do or in what format: the un-tuned base, shown the raw observation, parsed 0/127
    # (smoke j64/j65). The same text must be appended at evaluation.
    user = user + instruction
    acts = ACTION.findall(row["cot"])
    if len(acts) != 1:
        return None, "action_count"
    reasoning = ACTION.sub("", row["cot"]).strip()
    if not reasoning:
        return None, "empty_reasoning"
    return {
        "domain": row["variant"],
        "instance": row["problem_id"],
        "turn": row["turn"],
        "turns": 1,
        "binding_constraint": row.get("binding_constraint"),
        "n_prior_changes": row.get("n_prior_changes"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "reasoning_content": reasoning, "content": acts[0].strip()},
        ],
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", nargs="+", required=True, help="rendered-row jsonl files or dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-change-block", action="store_true",
                    help="keep CHANGE YOU ARE ABOUT TO MAKE in the user turn (rationaliser)")
    ap.add_argument("--all", action="store_true", help="include rows with accepted=false too")
    ap.add_argument("--instruction-file", default=None,
                    help="text appended verbatim to every user turn (the task instruction); its "
                         "sha1 is recorded per row as instruction_sha1")
    a = ap.parse_args()
    instruction = open(a.instruction_file).read() if a.instruction_file else ""
    instr_sha = hashlib.sha1(instruction.encode()).hexdigest()[:12] if instruction else None

    files = []
    for p in a.cot:
        files += sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
    n_in = collections.Counter(); n_out = collections.Counter(); drop = collections.Counter()
    words = []
    with open(a.out, "w") as fo:
        for f in files:
            for line in open(f):
                r = json.loads(line)
                n_in[r["variant"]] += 1
                if not a.all and not r.get("accepted", False):
                    drop["not_accepted"] += 1
                    continue
                conv, why = convert(r, a.include_change_block, instruction)
                if conv is None:
                    drop[why] += 1
                    continue
                conv["instruction_sha1"] = instr_sha
                words.append(len(conv["messages"][2]["reasoning_content"].split()))
                n_out[r["variant"]] += 1
                fo.write(json.dumps(conv) + "\n")
    print("read %d rows from %d files -> wrote %d conversations to %s"
          % (sum(n_in.values()), len(files), sum(n_out.values()), a.out))
    print("  per variant: " + "  ".join("%s %d/%d" % (v, n_out[v], n_in[v]) for v in sorted(n_in)))
    print("  dropped: %s" % (dict(drop) or "none"))
    if words:
        print("  reasoning words: min %d  median %d  max %d" % (min(words), statistics.median(words), max(words)))
    print("  user turn %s the CHANGE block" % ("INCLUDES" if a.include_change_block else "excludes"))
    print("  task instruction: %s" % ("%s (sha1 %s, %d chars)" % (a.instruction_file, instr_sha, len(instruction))
                                      if instruction else "NONE"))


if __name__ == "__main__":
    main()
