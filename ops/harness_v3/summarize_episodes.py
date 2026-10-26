"""Per-variant readout of policy-evaluation episodes files (dataxp_policy_eval_v3 `episodes` mode).

usage: python summarize_episodes.py <episodes.jsonl> [...]   (several files are pooled by variant)

Columns: episodes, feasible@12, feasible@20 (fractions), parse rate (turns whose reply yielded a move
/ turns asked), median and p90 reply tokens (completion_tokens of the asked turns), share of turns at
the token cap (finish == "length"), fail reasons, median obj/ref at first feasibility, and the LLM
token totals. Also prints the pooled row."""
import collections
import json
import statistics
import sys


def med(xs):
    return statistics.median(xs) if xs else None


def p90(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]


def row(name, eps):
    turns = [t for e in eps for t in e.get("turns", [])]
    asked = [t for t in turns if "finish" in t or "completion_tokens" in t]   # llm turns
    ct = [t["completion_tokens"] for t in asked if t.get("completion_tokens") is not None]
    cap = sum(1 for t in asked if t.get("finish") == "length")
    parsed = sum(1 for t in turns if t.get("move") is not None)
    fails = collections.Counter((t.get("fail") or "").split(":")[0] for t in turns if t.get("fail"))
    orr = [e["obj_over_ref"] for e in eps if e.get("obj_over_ref") is not None]
    crashed = sum(1 for e in eps if e.get("crashed"))
    n = len(eps)
    return {"variant": name, "episodes": n, "feasible@12": sum(bool(e.get("feasible@12")) for e in eps) / n if n else None,
            "feasible@20": sum(bool(e.get("feasible@20")) for e in eps) / n if n else None,
            "turns": len(turns), "parse_rate": parsed / len(turns) if turns else None,
            "reply_tokens_median": med(ct), "reply_tokens_p90": p90(ct),
            "at_cap_share": cap / len(asked) if asked else None, "fails": dict(fails),
            "obj_over_ref_median": med(orr), "n_obj": len(orr), "crashed": crashed,
            "prompt_tokens": None, "completion_tokens": sum(ct) if ct else None}


def main():
    by = collections.defaultdict(list)
    for p in sys.argv[1:]:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                by[r["variant"]].append(r)
    rows = [row(v, by[v]) for v in sorted(by)]
    allep = [e for v in by for e in by[v]]
    if len(by) > 1:
        rows.append(row("POOLED", allep))
    fmt = "%-10s %4s %7s %7s %6s %6s %8s %8s %7s %8s %s"
    print(fmt % ("variant", "eps", "feas@12", "feas@20", "turns", "parse", "tok_med", "tok_p90", "at_cap", "obj/ref", "fails"))
    for r in rows:
        print(fmt % (r["variant"], r["episodes"], "%.3f" % r["feasible@12"] if r["feasible@12"] is not None else "-",
                     "%.3f" % r["feasible@20"] if r["feasible@20"] is not None else "-", r["turns"],
                     "%.3f" % r["parse_rate"] if r["parse_rate"] is not None else "-",
                     r["reply_tokens_median"] if r["reply_tokens_median"] is not None else "-",
                     r["reply_tokens_p90"] if r["reply_tokens_p90"] is not None else "-",
                     "%.3f" % r["at_cap_share"] if r["at_cap_share"] is not None else "-",
                     ("%.3f (n=%d)" % (r["obj_over_ref_median"], r["n_obj"])) if r["obj_over_ref_median"] is not None else "-",
                     json.dumps(r["fails"], sort_keys=True) + (" crashed=%d" % r["crashed"] if r["crashed"] else "")))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
