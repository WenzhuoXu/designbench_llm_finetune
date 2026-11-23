"""The pair files of several shares (build_pairs.py) in one training file and one dev file, plus the replay pool,
for llm_finetune/rl_dataxp/dpo_rows.py.

    one pair per state      the first file that has a state's pair wins (give the files in the open1 merge order,
                            so a pair's chosen reply is the open1 row of that state)
    dev                     the pairs of the problems in --dev-rows (the dev rows of the supervised round: the
                            same held-out problems, so the two rounds' dev numbers are on the same problems)
    replay pool             the rows of --replay-from whose tier is --replay-tier (the replay rows the supervised
                            round trained on); dpo_rows draws replay_per_pair of them per pair
    checks, every pair      system + user messages, chosen and rejected assistant turns with a rationale of at most
                            --max-words words and no think tag, content exactly one <action>...</action>, two
                            different moves, the rejected kind one of refused / no_gain / noop

    python merge_pairs.py --out-train pairs_train.jsonl --out-dev pairs_dev.jsonl --dev-rows dev_open1.jsonl \\
        --replay-from train_open1.jsonl --out-replay replay_pool.jsonl pairs_a.jsonl pairs_b.jsonl ...
"""
import argparse
import collections
import json
import re
import sys

ACTION_ONLY = re.compile(r"^\s*<action>[^<>]*</action>\s*$")


def fault(p: dict, max_words: int):
    m = p.get("messages") or []
    if len(m) != 2 or m[0].get("role") != "system" or m[1].get("role") != "user" or not m[0].get("content") or not m[1].get("content"):
        return "messages are not system + user"
    for side in ("chosen", "rejected"):
        a = p.get(side) or {}
        r = (a.get("reasoning_content") or "").strip()
        if a.get("role") != "assistant":
            return side + " is not an assistant turn"
        if not r:
            return side + " has no rationale"
        if "<think>" in r or "</think>" in r:
            return side + " rationale has a think tag"
        if max_words and len(r.split()) > max_words:
            return side + " rationale over the word cap"
        if not ACTION_ONLY.match(a.get("content") or ""):
            return side + " content is not one action"
    if re.sub(r"\s+", "", p["chosen"]["content"]) == re.sub(r"\s+", "", p["rejected"]["content"]):
        return "chosen and rejected make the same move"
    if p.get("rejected_kind") not in ("refused", "no_gain", "noop"):
        return "rejected kind %r" % p.get("rejected_kind")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--dev-rows", required=True)
    ap.add_argument("--replay-from", required=True)
    ap.add_argument("--replay-tier", default="ownrows_own")
    ap.add_argument("--out-replay", required=True)
    ap.add_argument("--max-words", type=int, default=1000)
    a = ap.parse_args(argv)

    dev_inst = {json.loads(l)["instance"] for l in open(a.dev_rows, encoding="utf-8") if l.strip()}
    seen, kept = set(), []
    c = collections.Counter()
    for f in a.files:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            p = json.loads(line)
            c["read: %s" % f] += 1
            if p["sid"] in seen:
                c["dropped: state already paired by an earlier file"] += 1
                continue
            w = fault(p, a.max_words)
            if w:
                c["FAILED CHECK: " + w] += 1
                continue
            seen.add(p["sid"])
            kept.append(p)
    train = [p for p in kept if p["instance"] not in dev_inst]
    dev = [p for p in kept if p["instance"] in dev_inst]
    with open(a.out_train, "w", encoding="utf-8") as fo:
        for p in train:
            fo.write(json.dumps(p) + "\n")
    with open(a.out_dev, "w", encoding="utf-8") as fo:
        for p in dev:
            fo.write(json.dumps(p) + "\n")
    n_rep = 0
    with open(a.out_replay, "w", encoding="utf-8") as fo:
        for line in open(a.replay_from, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("tier") == a.replay_tier:
                    if r.get("instance") in dev_inst:
                        c["replay rows of a dev problem (dropped)"] += 1
                        continue
                    fo.write(line if line.endswith("\n") else line + "\n")
                    n_rep += 1
    for k in sorted(c):
        print("%-60s %7d" % (k, c[k]))
    for name, ps in (("train", train), ("dev", dev)):
        fam = collections.Counter(p["domain"] for p in ps)
        kind = collections.Counter(p["rejected_kind"] for p in ps)
        print("%-5s %6d pairs | %s | rejected %s" % (name, len(ps), " ".join("%s %d" % kv for kv in sorted(fam.items())),
                                                     " ".join("%s %d" % kv for kv in sorted(kind.items()))))
    print("replay pool %d rows (tier %s); dev problems %d" % (n_rep, a.replay_tier, len(dev_inst)))
    if any(k.startswith("FAILED CHECK") for k in c):
        print("MERGE_PAIRS_FAILED_CHECKS")
        return 3
    print("MERGE_PAIRS_DONE %d train, %d dev -> %s %s" % (len(train), len(dev), a.out_train, a.out_dev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
