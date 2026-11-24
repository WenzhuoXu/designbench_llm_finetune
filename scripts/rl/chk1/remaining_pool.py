"""remaining_pool.py POOL OUT EPISODES...: the lines of POOL (in their order) whose (variant, problem_id) has no episode
in any EPISODES file, for a collector that continues a pool another collector has started (chk1: the advanced-node
collector after the general-queue one). Unreadable episode lines (a partial last line) are counted and skipped."""
import json
import sys

pool, out, eps = sys.argv[1], sys.argv[2], sys.argv[3:]
done, bad = set(), 0
for p in eps:
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        ep = r.get("episode") or r
        done.add((ep.get("variant"), ep.get("problem_id")))
lines = [l if l.endswith("\n") else l + "\n" for l in open(pool, encoding="utf-8") if l.strip()]
keep = [l for l in lines if (json.loads(l)["variant"], json.loads(l)["spec"]["problem_id"]) not in done]
with open(out, "w", encoding="utf-8") as fo:
    fo.writelines(keep)
print("pool %d, episodes recorded %d (unreadable lines %d), remaining %d -> %s" % (len(lines), len(done), bad, len(keep), out))
