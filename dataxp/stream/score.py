"""Scoring of physics-CoT streams (docs/cot_stream_validation.md §2-§3): side-question answers
(normalised 80% interval score, point error, coverage, persistence), G1 grounding and G2
pre/post comparison with a state bootstrap.

    python -m dataxp.stream.score --states S.jsonl --streams W1.jsonl [W2.jsonl ...] --out R.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ..render_cot import _fmt
from .env import StreamState, element_of_text
from .protocol import parse_json_block, parse_side

ALPHA = 0.2
CAP_IS = 10.0
REVIEW_TOL = 5e-3
EL_RE = re.compile(r"^m?(\d+)\s*[._:\- ]?\s*(N_kN|N|fos_b|fos_y)$")


def _num(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def interval(v):
    """(lo, mid, hi) from [lo, mid, hi], [lo, hi] or a number; None if unusable."""
    if isinstance(v, (list, tuple)):
        xs = [_num(x) for x in v]
        if any(x is None for x in xs):
            return None
        if len(xs) == 3:
            lo, mid, hi = xs
        elif len(xs) == 2:
            lo, hi = xs
            mid = 0.5 * (lo + hi)
        elif len(xs) == 1:
            lo = mid = hi = xs[0]
        else:
            return None
        return min(lo, hi), mid, max(lo, hi)
    x = _num(v)
    return None if x is None else (x, x, x)


def flat_candidate(c: dict, variant: str) -> Dict[str, tuple]:
    out = {}
    for k, v in (c.get("metrics") or {}).items():
        iv = interval(v)
        if iv:
            out[str(k).strip()] = iv
    for el, qs in (c.get("elements") or {}).items():
        e = str(el).strip()
        if variant == "frame3d2" and e.isdigit():
            e = "m" + e
        if not isinstance(qs, dict):
            continue
        for q, v in qs.items():
            iv = interval(v)
            if iv:
                out["%s.%s" % (e, str(q).strip())] = iv
    return out


def side_candidates(ss: StreamState, H: List[str], text: str):
    """[{key: (lo, mid, hi)} or None] aligned with H; matched by position, else by action."""
    d, how = parse_side(text)
    cands = (d or {}).get("candidates") if isinstance(d, dict) else None
    out = [None] * len(H)
    if not isinstance(cands, list):
        return out, how
    if len(cands) == len(H):
        for i, c in enumerate(cands):
            if isinstance(c, dict):
                out[i] = flat_candidate(c, ss.v)
        return out, how
    for c in cands:
        if not isinstance(c, dict):
            continue
        for i, h in enumerate(H):
            if out[i] is None and ss.same_move(str(c.get("action", "")), h):
                out[i] = flat_candidate(c, ss.v)
                break
    return out, how


def score_pair(ss, key, iv, y, cur) -> dict:
    s = ss.scale(key, y, cur)
    rec = {"key": key, "y": y, "cur": cur, "s": s,
           "persist": (abs(cur - y) / s) if cur is not None else None,
           "moved": cur is not None and abs(y - cur) > 1e-3 * s}
    if iv is None:
        rec.update(missing=True, nis=CAP_IS, nis_raw=None, point=None, cover=None)
        return rec
    lo, mid, hi = iv
    is_ = (hi - lo) + (2 / ALPHA) * max(0.0, lo - y) + (2 / ALPHA) * max(0.0, y - hi)
    rec.update(missing=False, nis=min(CAP_IS, is_ / s), nis_raw=is_ / s, point=abs(mid - y) / s,
               cover=bool(lo <= y <= hi), width=(hi - lo) / s)
    return rec


def score_side(ss: StreamState, H: List[str], text: Optional[str]) -> List[dict]:
    """One row per (H index, queried key) with a simulated truth."""
    cands, how = side_candidates(ss, H, text or "")
    rows = []
    for i, h in enumerate(H):
        res = ss.check(h)
        if not res["ok"]:
            continue
        q = ss.queries(h)
        truth = ss.truth(res, q)
        own = element_of_text(ss.v, h)
        for key, (y, cur) in truth.items():
            if y is None:
                continue
            iv = (cands[i] or {}).get(key) if cands[i] is not None else None
            r = score_pair(ss, key, iv, y, cur)
            r.update(h=i, h_text=h, element=("." in key),
                     own=("." in key and key.split(".")[0] == own), parse=how)
            rows.append(r)
    return rows


def review_check(ss: StreamState, stream: dict) -> List[dict]:
    """Each <review> 'observed' entry vs the injected result of the probe just before it."""
    out = []
    msgs = stream["messages"]
    for j, m in enumerate(msgs):
        if m["role"] != "user" or "_probe" not in m:
            continue
        p = stream["probes"][m["_probe"]]
        nxt = next((x for x in msgs[j + 1:] if x["role"] == "assistant"), None)
        if nxt is None or not p.get("ok"):
            continue
        d, _ = parse_json_block(nxt["content"], "review")
        obs = (d or {}).get("observed") if isinstance(d, dict) else None
        if not isinstance(obs, dict):
            out.append({"probe": m["_probe"], "missing_block": True})
            continue
        for k, v in obs.items():
            val = _num(v)
            k2 = str(k).strip()
            inj = None
            if k2 in (p.get("metrics") or {}):
                inj = p["metrics"][k2]
            else:
                mm = EL_RE.match(k2)
                nom = p.get("nom")
                if mm and nom and not nom.get("mech"):
                    i, q = int(mm.group(1)), mm.group(2)
                    if i < len(nom["N"]):
                        inj = nom["N"][i] / 1000.0 if q in ("N_kN", "N") else nom[q][i]
            if inj is None or not math.isfinite(inj):
                out.append({"probe": m["_probe"], "key": k2, "matched": False})
                continue
            ok = val is not None and abs(val - inj) <= REVIEW_TOL * max(abs(inj), 1e-12) + 1e-12
            if not ok and val is not None:
                # the value exactly as the result displays it (e.g. 0.0116 shown as "0.012")
                shown = _num(_fmt(inj))
                ok = shown is not None and abs(val - shown) <= 1e-9 * max(1.0, abs(shown))
            if not ok and val is not None and abs(inj) < 1e-3 and abs(val) < 1e-3:
                ok = True   # a value displayed as 0
            out.append({"probe": m["_probe"], "key": k2, "matched": True, "ok": bool(ok)})
    return out


def boot_ci(diffs: np.ndarray, n: int = 10000, seed: int = 0):
    if diffs.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n, diffs.size))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def state_means(rows: List[dict], exclude_h=()) -> Optional[float]:
    v = [r["nis"] for r in rows if r["h"] not in exclude_h]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--streams", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-out", default="")
    a = ap.parse_args()
    recs = {json.loads(l)["sid"]: json.loads(l) for l in open(a.states, encoding="utf-8") if l.strip()}
    ss_cache = {}
    report, all_rows = {}, []
    for path in a.streams:
        streams = {}
        for l in open(path, encoding="utf-8"):
            o = json.loads(l)
            if o.get("status") == "error" and o["sid"] in streams:
                continue
            streams[o["sid"]] = o
        writer = next(iter(streams.values())).get("writer", path)
        g1 = defaultdict(int)
        diffs, per_family, strata = [], defaultdict(list), defaultdict(list)
        # secondary, not pre-registered (§6, 2026-10-30): pairs answered both times; missing counted apart
        diffs_both, per_family_both, miss = [], defaultdict(list), {"pre": [0, 0], "post": [0, 0]}
        parse_how = defaultdict(int)
        cov_pre, cov_post = [], []
        tok = defaultdict(float)
        for sid, st in sorted(streams.items()):
            if sid not in ss_cache:
                ss_cache[sid] = StreamState(recs[sid])
            ss = ss_cache[sid]
            g1["streams"] += 1
            g1["status_" + str(st.get("status"))] += 1
            for c in st.get("calls") or []:
                for k, v in (c.get("usage") or {}).items():
                    tok[k] += v or 0
            for side in (st.get("side") or {}).values():
                for k, v in (side.get("usage") or {}).items():
                    tok["side_" + k] += v or 0
            if st.get("status") != "done" and st.get("status") != "no_action":
                continue
            fa = st.get("final_action")
            legal = bool(fa) and ss.check(fa)["ok"]
            g1["final_legal"] += int(st.get("status") == "done" and legal)
            g1["probes"] += st.get("n_probes", 0)
            g1["probes_legal"] += sum(1 for p in st["probes"] if p.get("ok"))
            g1["cap_hit"] += int(st.get("n_probes", 0) >= st.get("cap", 6))
            g1["no_probe"] += int(st.get("n_probes", 0) == 0)
            for rv in review_check(ss, st):
                if rv.get("missing_block"):
                    g1["review_missing"] += 1
                elif not rv["matched"]:
                    g1["review_unmatched"] += 1
                else:
                    g1["review_entries"] += 1
                    g1["review_ok"] += int(rv["ok"])
            H = st.get("H") or []
            side = st.get("side") or {}
            if "pre" not in side or "post" not in side or not st.get("n_probes"):
                continue
            if any((s or {}).get("error") for s in side.values()):
                parse_how["side_api_error"] += 1  # an API failure, not the writer's answer: not scored
                continue
            probed = {i for i, h in enumerate(H) for p in st["probes"] if p.get("ok") and ss.is_probe_of(p["text"], h)}
            probed_el = {element_of_text(ss.v, p["text"]) for p in st["probes"] if p.get("ok")}
            pre = score_side(ss, H, side["pre"].get("text"))
            post = score_side(ss, H, side["post"].get("text"))
            for r, tag in [(x, "pre") for x in pre] + [(x, "post") for x in post]:
                r.update(sid=sid, writer=writer, side=tag, family=ss.v,
                         group=(recs[sid].get("pilot") or {}).get("group"), probed=r["h"] in probed,
                         probed_el=element_of_text(ss.v, r["h_text"]) in probed_el)
                all_rows.append(r)
            for tag, rs in (("pre", pre), ("post", post)):
                parse_how["%s_%s" % (tag, rs[0]["parse"] if rs else "no_rows")] += 1
            a_pre = {(r["h"], r["key"]): r for r in pre if r["h"] not in probed}
            a_post = {(r["h"], r["key"]): r for r in post if r["h"] not in probed}
            both = [k for k in a_pre if k in a_post]
            for k in both:
                for tag, a_ in (("pre", a_pre), ("post", a_post)):
                    miss[tag][0] += int(a_[k]["missing"])
                    miss[tag][1] += 1
            both = [k for k in both if not a_pre[k]["missing"] and not a_post[k]["missing"]]
            if both:
                d_ = float(np.mean([a_pre[k]["nis"] for k in both]) - np.mean([a_post[k]["nis"] for k in both]))
                diffs_both.append(d_)
                per_family_both[(recs[sid].get("pilot") or {}).get("group")].append(d_)
            m_pre, m_post = state_means(pre, probed), state_means(post, probed)
            if m_pre is None or m_post is None:
                continue
            diffs.append(m_pre - m_post)
            per_family[(recs[sid].get("pilot") or {}).get("group")].append(m_pre - m_post)
            for nm, keep in (("not_probed_element", lambda r: not r.get("probed_el")),
                             ("probed_element", lambda r: r.get("probed_el")),
                             ("element_level", lambda r: r["element"]),
                             ("metrics", lambda r: not r["element"])):
                a1 = [r["nis"] for r in pre if r["h"] not in probed and keep(r)]
                a2 = [r["nis"] for r in post if r["h"] not in probed and keep(r)]
                if a1 and a2:
                    strata[nm].append(np.mean(a1) - np.mean(a2))
            cov_pre += [r["cover"] for r in pre if r["h"] not in probed and r["cover"] is not None]
            cov_post += [r["cover"] for r in post if r["h"] not in probed and r["cover"] is not None]
        mean, lo, hi = boot_ci(np.array(diffs))
        report[writer] = {
            "G1": dict(g1),
            "G1_rates": {"review_ok": g1["review_ok"] / max(1, g1["review_entries"]),
                         "final_legal": g1["final_legal"] / max(1, g1["streams"]),
                         "probes_legal": g1["probes_legal"] / max(1, g1["probes"]),
                         "cap_hit": g1["cap_hit"] / max(1, g1["streams"])},
            "G2": {"n_states": len(diffs), "mean_pre_minus_post": mean, "ci95": [lo, hi],
                   "per_group": {g: boot_ci(np.array(v))[:3] + (len(v),) for g, v in sorted(per_family.items())},
                   "strata": {k: boot_ci(np.array(v))[:3] + (len(v),) for k, v in strata.items()},
                   "coverage_pre": float(np.mean(cov_pre)) if cov_pre else None,
                   "coverage_post": float(np.mean(cov_post)) if cov_post else None,
                   "side_parse": dict(parse_how)},
            "G2_answered_both": {
                "note": "secondary, not pre-registered: (H move, quantity) pairs parsed in both answers",
                "n_states": len(diffs_both),
                "mean_pre_minus_post": boot_ci(np.array(diffs_both))[0] if diffs_both else None,
                "ci95": list(boot_ci(np.array(diffs_both))[1:3]) if diffs_both else None,
                "per_group": {g: boot_ci(np.array(v))[:3] + (len(v),) for g, v in sorted(per_family_both.items())},
                "missing_rate_pre": miss["pre"][0] / max(1, miss["pre"][1]),
                "missing_rate_post": miss["post"][0] / max(1, miss["post"][1])},
            "tokens": dict(tok)}
    json.dump(report, open(a.out, "w"), indent=1)
    if a.rows_out:
        with open(a.rows_out, "w") as fh:
            for r in all_rows:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
