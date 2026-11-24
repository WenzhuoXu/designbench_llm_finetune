"""Build dataxp_policy_eval_v3rk.py from dataxp_policy_eval_v3r.py (sha1 4978bab2...): one opt-in flag, --best-of K.
Without the flag the file behaves as v3r. Every replacement must match exactly once.

--best-of K (2026-11-23; native variants only: photonics, grid2op, engibench): at every turn the model is asked once
for K replies (one chat completion with n = K, at --best-of-temperature, top_p 0.95 / top_k 20 as the harness samples
at T > 0); each reply's move is applied to a copy of the design under the same rules (--open-rules as given) and
scored with the seed the harness scores the next design with; the reply kept is the one whose design is best:
feasible first, then (photonics) fewer pixels breaking the fabrication rule, then the higher potential, then the
lower sample index. A reply that does not parse or whose move is refused ranks below every applied one; when none
applies, reply 0 is kept (and refused as in a plain turn). The turn then runs exactly as a plain turn with that
reply. The turn record gets best_of = {k, pick, cands: per reply parse/refusal/feasible/violations/phi}.

This measures the ceiling of a policy that always picks, by the simulator's one-step verdict, the best of K of its
own replies: what a preference round on such pairs could at most teach, turn by turn.

usage: python make_v3rk.py <v3r> <out>
"""
import hashlib
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
assert hashlib.sha1(open(src, "rb").read()).hexdigest() == "4978bab2dd8c68a8bbfedcc6cdac176df37cfab8", "not the expected v3r"


def rep(old, new):
    global s
    assert s.count(old) == 1, "%d matches for %r" % (s.count(old), old[:70])
    s = s.replace(old, new)


# 1. helpers, before the observation section
rep('''# ------------------------------------------------------------------ observation
''', '''# ------------------------------------------------------------------ best of K (--best-of)
# v3rk (2026-11-23). Without the flag this file behaves as dataxp_policy_eval_v3r.py (sha1 4978bab2).

def sample_k(a, llm, system, user):
    """K replies to one prompt (one request, n = K), each in LLM.__call__'s reply format."""
    k = int(a.best_of)
    body = {"model": a.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": float(a.best_of_temperature), "max_tokens": a.max_tokens, "n": k, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": not a.no_think}}
    req = urllib.request.Request(a.base_url.rstrip("/") + "/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                j = json.load(r)
            break
        except Exception as e:                       # noqa: BLE001
            if attempt == 3:
                return [{"err": str(e)[:300]}]
            time.sleep(5 * (attempt + 1))
    u = j.get("usage") or {}
    with LOCK:
        llm.tokens["prompt"] += u.get("prompt_tokens", 0) or 0
        llm.tokens["completion"] += u.get("completion_tokens", 0) or 0
    out = []
    for ch in sorted(j.get("choices") or [], key=lambda c: c.get("index", 0)):
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        closed = True
        if "</think>" in text:
            reasoning, text = text.rsplit("</think>", 1)
        elif not a.no_think and not reasoning:
            closed = False
        out.append({"answer": text, "reasoning": reasoning.replace("<think>", "").strip(), "think_closed": closed,
                    "finish": ch.get("finish_reason"), "completion_tokens": None})
    return out or [{"err": "no choices"}]


def pick_best(replies, C, D, domain, variant, st, ev, pid, t, opening):
    """(the reply kept, the turn's best_of record): see --best-of."""
    best, cands = None, []
    for i, r in enumerate(replies):
        mv, why = parse_move_native(C, domain, r)
        c = {"i": i, "parse": why, "finish": r.get("finish")}
        if mv is not None:
            child = domain.clone(st)
            try:
                open_apply(domain, variant, child, mv, opening)
            except Exception as e:                   # noqa: BLE001
                c["refused"] = str(e)[:160]
            else:
                sc = ev.score(child, seed0=D.stable_seed(pid, "eval", "s", t + 1))
                viol = float(sc.agg_metrics.get("min_feature_violations") or 0.0) if variant == "photonics" else 0.0
                c.update(move=domain.format_move(mv), feasible=bool(sc.feasible), viol=viol, phi=float(sc.mean))
                key = (bool(sc.feasible), -viol, float(sc.mean), -i)
                if best is None or key > best[0]:
                    best = (key, i)
        cands.append(c)
    pick = best[1] if best is not None else 0
    return replies[pick], {"k": len(replies), "pick": pick, "applied": sum(1 for c in cands if "phi" in c), "cands": cands}


# ------------------------------------------------------------------ observation
''')

# 2. the model call of a turn
rep('''                SIM.release()
                try:
                    reply = llm(system, user + note)
                finally:
                    SIM.acquire()
''', '''                bestk = int(getattr(a, "best_of", 0) or 0) > 0
                if bestk and not native:
                    raise SystemExit("--best-of takes native variants only (got %s)" % variant)
                SIM.release()
                try:
                    if bestk:
                        replies = sample_k(a, llm, system, user + note)
                    else:
                        reply = llm(system, user + note)
                finally:
                    SIM.acquire()
                if bestk:
                    reply, rec["best_of"] = pick_best(replies, C, D, domain, variant, st, ev, pid, t, opening)
''')

# 3. the flags
rep('''    ap.add_argument("--native-prompt", choices=["served", "probe"], default="served",''',
    '''    ap.add_argument("--best-of", type=int, default=0,
                    help="episodes, llm, native variants (v3rk): K replies per turn at --best-of-temperature, the one whose "
                         "design the simulator scores best is kept (feasible, fewer fabrication-rule pixels, higher potential)")
    ap.add_argument("--best-of-temperature", type=float, default=0.8)
    ap.add_argument("--native-prompt", choices=["served", "probe"], default="served",''')

open(dst, "w", encoding="utf-8", newline="").write(s)
print("wrote %s (sha1 %s)" % (dst, hashlib.sha1(open(dst, "rb").read()).hexdigest()))
