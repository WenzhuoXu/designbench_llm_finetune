"""Build the training-problem file for RL rollouts, disjoint from every evaluation problem.

    python -m llm_finetune.rl_dataxp.problems \\
        --script <harness>/scripts/dataxp_policy_eval_v3.py --code <harness>/dataxp_code_v3 \\
        --corpus circuit2=<datav4_corpus>/circuit2/accepted/problems.jsonl ... \\
        --eval-problems <harness>/problems_v2.jsonl <harness>/problems_v3_native.jsonl \\
        --heldout thermal2 inventory2 molopt --out train_problems.jsonl

Input lines may be bare specs (the corpus files) or {variant, spec} (harness problem files).
A problem is dropped when:
  - its variant is held out (an error, not a drop: a held-out family must never be passed);
  - its problem_id, or its canonical spec (every key but problem_id / domain_variant, the
    harness's own fresh_problems test), equals an evaluation problem's;
  - it is feasible at the start (the harness would never ask the model; DX-37 left these out);
  - it repeats an earlier problem's canonical spec.
Output: {variant, spec} lines (the harness's --problems-file format) and OUT.summary.json with
the counts per variant and the sha1 of every input file.

Structural overlap flags of the native families (datav4_corpus/*/eval_overlap.json: shared
chronics, reference designs, base designs) are an owner-approved part of the training data
(2026-10-24) and are not filtered here.

Runs in the simulator env (needs the harness's dataxp code for the feasibility test).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys

from .harness import load_harness


def canon(spec: dict) -> str:
    """dataxp_policy_eval_v3.fresh_problems' canonical form."""
    return json.dumps({k: v for k, v in spec.items() if k not in ("problem_id", "domain_variant")},
                      sort_keys=True, default=str)


def read_specs(path: str, variant: str | None = None) -> list[tuple[str, dict]]:
    out = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if "spec" in r and "variant" in r:
            out.append((r["variant"], r["spec"]))
        else:
            v = variant or r.get("domain_variant")
            if not v:
                raise ValueError("%s: a bare spec without a variant (pass VARIANT=PATH)" % path)
            out.append((v, r))
    return out


def sha1(path: str) -> str:
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]


def build(a) -> dict:
    H = load_harness(a.script)
    D, _, _, get_domain = H.import_dataxp(a.code)
    heldout = set(a.heldout)
    eval_ids, eval_canon = set(), set()
    for p in a.eval_problems:
        for _, s in read_specs(p):
            eval_ids.add(s["problem_id"])
            eval_canon.add(canon(s))
    seen = set()
    counts = collections.defaultdict(collections.Counter)
    kept = []
    for item in a.corpus:
        v, _, path = item.partition("=")
        if not path:
            raise SystemExit("--corpus takes VARIANT=PATH, got %r" % item)
        if v in heldout:
            raise SystemExit("%s is a held-out family; it must not supply training problems" % v)
        domain = get_domain(v)
        for vv, spec in read_specs(path, v):
            if vv != v:
                raise SystemExit("%s: a %s problem in the %s corpus" % (path, vv, v))
            c = counts[v]
            c["read"] += 1
            key = canon(spec)
            if spec.get("problem_id") in eval_ids:
                c["eval_id"] += 1
                continue
            if key in eval_canon:
                c["eval_spec"] += 1
                continue
            if key in seen:
                c["duplicate"] += 1
                continue
            seen.add(key)
            ev = D.PhiEvaluator(domain, spec)
            if ev.program.is_feasible(ev.initial_agg):
                c["feasible_start"] += 1
                continue
            c["kept"] += 1
            kept.append({"variant": v, "spec": spec})
    with open(a.out, "w") as fo:
        for r in kept:
            fo.write(json.dumps(r, default=H._json_default) + "\n")
    summary = {"out": a.out, "kept": len(kept), "per_variant": {v: dict(c) for v, c in sorted(counts.items())},
               "heldout": sorted(heldout), "eval_problems": {p: sha1(p) for p in a.eval_problems},
               "corpus": {item: sha1(item.partition("=")[2]) for item in a.corpus},
               "eval_ids": len(eval_ids)}
    json.dump(summary, open(a.out + ".summary.json", "w"), indent=1)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="training problems for dataxp RL, disjoint from evaluation")
    ap.add_argument("--script", required=True, help="dataxp_policy_eval_v3.py")
    ap.add_argument("--code", required=True, help="the dataxp code copy the harness imports")
    ap.add_argument("--corpus", nargs="+", required=True, help="VARIANT=PATH (bare specs or {variant, spec})")
    ap.add_argument("--eval-problems", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="*", default=["thermal2", "inventory2", "molopt"])
    ap.add_argument("--out", required=True)
    s = build(ap.parse_args(argv))
    for v, c in s["per_variant"].items():
        print("%-10s %s" % (v, c))
    print("kept %d -> %s" % (s["kept"], s["out"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
