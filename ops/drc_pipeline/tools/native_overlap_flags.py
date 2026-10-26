"""Flag native TRAINING problems (every corpus batch) that are close to an evaluation problem, and
write datav4_corpus/<family>/eval_overlap.json for grid2op, photonics and engibench.

Per training problem: the closest evaluation problem (fewest differing spec keys), the keys that
differ, the tiers it falls in, and its rows per spelling and batch. Tiers:
  engibench: shared_base (family, source_row) - also in shared_base_with_eval.json;
             same_start_design (design0 equal)
  photonics: shared_reference (ref_design equal); same_reference_and_goals; differs_only_in_start
             (every compared key equal except design0 / initial_metrics / perturbations)
  grid2op:   shared_chronic (env + chronic); same_chronic_outage_preactions;
             differs_only_in_t0 (every compared key equal except t0 / initial_metrics)
Owner policy 2026-10-24: such problems are kept as training rows and flagged.
Usage: python native_overlap_flags.py [--dv /storage/wxu/datav4]"""
import argparse
import collections
import glob
import json
import os

EVAL = "/storage/wxu/corpus/harness/problems_v3_native.jsonl"
IGNORE = {"problem_id", "domain_variant", "index"}
START_KEYS = {"design0", "initial_metrics", "perturbations", "difficulty_class"}
T0_KEYS = {"t0", "initial_metrics", "difficulty_class"}


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def enc(v):
    return json.dumps(v, sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dv", default="/storage/wxu/datav4")
    a = ap.parse_args()
    ev = collections.defaultdict(list)
    for r in jl(EVAL):
        s = r["spec"] if isinstance(r["spec"], dict) else json.loads(r["spec"])
        ev[r["variant"]].append(s)
    rows = collections.Counter()
    for sp in ("multitool", "set_parameter"):
        root = os.path.join(a.dv, "datav4_" + sp)
        for p in glob.glob(root + "/rows/*/*_own_act.jsonl") + glob.glob(root + "/batches/*/rows/*/*_own_act.jsonl"):
            b = p.split("/batches/")[1].split("/")[0] if "/batches/" in p else "b001"
            for o in jl(p):
                rows[(sp, b, o["instance"])] += 1
    for fam in ("engibench", "photonics", "grid2op"):
        E = ev.get(fam, [])
        eids = {e["problem_id"] for e in E}
        tr = []
        for f in sorted(glob.glob(os.path.join(a.dv, "datav4_corpus", fam, "accepted*", "problems.jsonl"))):
            batch = os.path.basename(os.path.dirname(f)).replace("accepted_", "").replace("accepted", "b001")
            for s in jl(f):
                if s["problem_id"] not in eids:
                    tr.append((batch, s))
        if not E or not tr:
            print(fam, "nothing to compare"); continue
        keys = sorted(set.intersection(*[set(s) for s in E] + [set(s) for _, s in tr]) - IGNORE)
        out = []
        for batch, s in tr:
            best = None
            for e in E:
                diff = [k for k in keys if enc(s.get(k)) != enc(e.get(k))]
                if best is None or len(diff) < len(best[1]):
                    best = (e["problem_id"], diff)
            tiers = []
            same = lambda *ks: [e["problem_id"] for e in E if all(enc(s.get(k)) == enc(e.get(k)) for k in ks)]
            if fam == "engibench":
                if same("family", "source_row"): tiers.append("shared_base")
                if same("design0"): tiers.append("same_start_design")
            elif fam == "photonics":
                if same("ref_design"): tiers.append("shared_reference")
                if same("ref_design", "goals"): tiers.append("same_reference_and_goals")
                if set(best[1]) <= START_KEYS: tiers.append("differs_only_in_start")
            else:
                ck = [k for k in ("env", "env_name", "chronic") if k in keys]
                if same(*ck): tiers.append("shared_chronic")
                ok = [k for k in ("forced_outage", "pre_actions") if k in keys]
                if ok and same(*(ck + ok)): tiers.append("same_chronic_outage_preactions")
                if set(best[1]) <= T0_KEYS: tiers.append("differs_only_in_t0")
            if not tiers:
                continue
            r = {"%s/%s" % (sp, b): rows[(sp, b, s["problem_id"])] for sp in ("multitool", "set_parameter")
                 for b in sorted({k[1] for k in rows}) if rows[(sp, b, s["problem_id"])]}
            out.append({"problem_id": s["problem_id"], "corpus_batch": batch, "tiers": tiers,
                        "closest_eval": best[0], "keys_differing": best[1], "rows_own_act": r})
        summ = {t: {"problems": sum(t in o["tiers"] for o in out),
                    "rows_own_act": {sp: sum(n for o in out if t in o["tiers"] for k, n in o["rows_own_act"].items() if k.startswith(sp))
                                     for sp in ("multitool", "set_parameter")}}
                for t in sorted({t for o in out for t in o["tiers"]})}
        doc = {"what": "native %s training problems close to an evaluation problem (last 60 by index of the first 130); "
                       "owner policy 2026-10-24: kept as training rows, flagged. Drop rows whose instance is listed under a "
                       "tier if a stricter evaluation is wanted." % fam,
               "compared_keys": keys, "n_training_problems": len(tr), "summary": summ, "problems": out}
        dst = os.path.join(a.dv, "datav4_corpus", fam, "eval_overlap.json")
        json.dump(doc, open(dst, "w"), indent=1)
        print("== %s: %d training problems; %s -> %s" % (fam, len(tr), json.dumps(summ), dst))


if __name__ == "__main__":
    main()
