"""merge_open_rows.py --out rows_all.jsonl --replay replay_open.jsonl <rows.jsonl> [more ...]: the open-rules round's gain
rows from every share in one file. A state sampled on two machines (drc's overshoot past its share) keeps the row of
the file given first. Every row is checked deterministically and a row failing a check is left out and counted:

  structure   messages = system, user, assistant; the assistant content is one <action>...</action> line and its
              reasoning_content (the rationale) is not empty; the rationale is at most 1,000 words
  prompt      the system message is the one the replay rows of that family use (the same served prompt); the user
              message is not empty
  identity    sid, domain, instance and action are present; the sid's family is the row's domain
  tier        the row's tier is open1_gain

Prints the counts per family and per check, and the share of rows whose rationale drafts an <action> line."""
import argparse
import collections
import json
import os
import re

ACTION = re.compile(r"^<action>[^\n]*</action>$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--tier", default="open1_gain")
    a = ap.parse_args()
    system = {}
    for line in open(a.replay, encoding="utf-8"):
        r = json.loads(line)
        system.setdefault(r["domain"], set()).add(r["messages"][0]["content"])
    seen = set()
    kept = collections.Counter()
    dup = collections.Counter()
    bad = collections.Counter()
    drafted = collections.Counter()
    per_file = []
    with open(a.out + ".tmp", "w", encoding="utf-8") as fo:
        for path in a.rows:
            n_in = n_keep = 0
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                n_in += 1
                r = json.loads(line)
                sid, fam = r.get("sid"), r.get("domain")
                if sid in seen:
                    dup[fam] += 1
                    continue
                why = None
                m = r.get("messages") or []
                if not (sid and fam and r.get("instance") and r.get("action")) or sid.split("|")[0] != fam:
                    why = "identity"
                elif [x.get("role") for x in m] != ["system", "user", "assistant"]:
                    why = "structure: roles"
                elif not ACTION.match((m[2].get("content") or "").strip()):
                    why = "structure: assistant content is not one action line"
                elif not (m[2].get("reasoning_content") or "").strip():
                    why = "structure: empty rationale"
                elif len(m[2]["reasoning_content"].split()) > 1000:
                    why = "structure: rationale over 1000 words"
                elif fam not in system or m[0].get("content") not in system[fam]:
                    why = "prompt: system message differs from the replay rows'"
                elif not (m[1].get("content") or "").strip():
                    why = "prompt: empty user message"
                elif r.get("tier") != a.tier:
                    why = "tier"
                if why:
                    bad[(fam, why)] += 1
                    continue
                seen.add(sid)
                kept[fam] += 1
                n_keep += 1
                if "<action>" in m[2]["reasoning_content"]:
                    drafted[fam] += 1
                fo.write(line if line.endswith("\n") else line + "\n")
            per_file.append((path, n_in, n_keep))
    os.replace(a.out + ".tmp", a.out)
    for path, n_in, n_keep in per_file:
        print("%-90s rows %6d kept %6d" % (path, n_in, n_keep))
    print("kept per family:", dict(sorted(kept.items())), "total", sum(kept.values()))
    print("left out as duplicate sids (kept from an earlier file):", dict(sorted(dup.items())))
    print("left out by a check:", {"%s / %s" % k: v for k, v in sorted(bad.items())} or "none")
    print("rationale drafts an <action> line:", dict(sorted(drafted.items())))
    print("MERGE_OPEN_ROWS_DONE %d -> %s" % (sum(kept.values()), a.out))


if __name__ == "__main__":
    main()
