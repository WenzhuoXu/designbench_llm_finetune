"""check_lex.py [N]: score_states.score_state (the fixed one) on real photonics states of dx38's open-rules benchmark
episodes (weak split), with dx38's own reply at that turn as the one sample. Cases:
  break   clean design, dx38's move broke the fabrication rule      -> must not be a gain
  repair  invalid design, dx38's move lowered the violating pixels  -> must be a gain
  clean   clean design, dx38's move kept it clean                     -> gain iff phi rose (as before)
Also prints pct and pct_basis."""
import collections
import json
import sys

sys.path.insert(0, r"C:/Users/xuwen/designbench_llm_finetune/analysis/remote/rl/harness_b2/dataxp_code_v3")
sys.path.insert(0, r"C:/Users/xuwen/designbench_llm_finetune/.claude/worktrees/quirky-tesla-842de7/scripts/rl_screen")
from score_states import score_state  # noqa: E402

E = r"C:/Users/xuwen/designbench_llm_finetune/analysis/remote/rl/eval/dx38mtR_open_20261122/weak/dx38mtR.episodes.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
specs = {e["key"]: e["spec"] for e in json.load(open(E + ".request.json"))["episodes"]}
want = collections.Counter()
c = collections.Counter()
bad_lines = []
for line in open(E, encoding="utf-8"):
    r = json.loads(line)
    if r["variant"] != "photonics":
        continue
    ep, spec = r["episode"], specs[r["key"]]
    moves, bad, phi = [], 0.0, float(ep["phi_start"])
    for t in ep["turns"]:
        if not (t.get("fail") is None and t.get("action") is not None):
            continue
        nb = float(t["metrics_after"].get("min_feature_violations", 0))
        kind = "break" if bad == 0 and nb > 0 else "repair" if nb < bad else "clean" if bad == 0 and nb == 0 else None
        if kind and want[kind] < N:
            want[kind] += 1
            row = {"sid": "%s~t%d" % (r["key"], t["t"]), "problem_id": spec["problem_id"], "variant": "photonics",
                   "k": len(moves), "spec": spec, "history": list(moves), "own_action": t["action"], "open_rules": True,
                   "phi_before": phi}
            out = score_state((row, [(0, {"thinking": t.get("reasoning") or "x", "answer": t.get("answer") or "",
                                          "finish": "stop", "sha1": "own"})]))
            st = [o for o in out if o.get("state")][0]
            rep = [o for o in out if not o.get("state")]
            if st.get("error") or not rep or not rep[0].get("ok"):
                c["%s: not scored (%s)" % (kind, st.get("error") or (rep[0].get("reason") if rep else "no reply"))] += 1
            else:
                x = rep[0]
                c["%s: cases" % kind] += 1
                c["%s: gain" % kind] += x["gain"]
                c["%s: bad_after == recorded" % kind] += x["bad_after"] == int(nb)
                c["%s: bad_before == recorded" % kind] += st["bad_before"] == int(bad)
                c["%s: phi == recorded phi_after" % kind] += abs(x["phi"] - float(t["phi_after"])) < 1e-6 * max(1, abs(x["phi"]))
                c["%s: pct_basis %s" % (kind, x["pct_basis"])] += 1
                ok = {"break": not x["gain"], "repair": x["gain"], "clean": x["gain"] == (x["phi"] > st["phi_before"] + 1e-9)}[kind]
                c["%s: as expected" % kind] += ok
                if not ok:
                    bad_lines.append((kind, row["sid"], x["bad_after"], st["bad_before"], x["phi"], st["phi_before"], x["gain"]))
        moves.append(t["action"])
        bad, phi = nb, float(t["phi_after"])
    if all(want[k] >= N for k in ("break", "repair", "clean")):
        break
for k in sorted(c):
    print("%-45s %d" % (k, c[k]))
for b in bad_lines[:10]:
    print("UNEXPECTED", b)
print("CHECK_LEX_%s" % ("OK" if not bad_lines and all(c["%s: cases" % k] for k in ("break", "repair", "clean")) else "FAILED"))
