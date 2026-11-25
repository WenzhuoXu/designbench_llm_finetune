"""con1 round 2 rows (owner's P rows vs question rows, docs/consequence_round_con1.md section 11 note of 11-25 16:20).

    python build_prows.py --episodes collect_adv_p1_pref1.episodes.jsonl.gz --states S [S ...] --format F \\
        --out-dir DIR [--families engibench grid2op photonics] [--per-family 400] [--q 12] [--seed 0]

P row (owner): the state and goal as the policy prompt shows them, then the model's OWN reply: its reasoning (loss
0), its action (loss 0: the model's own pick, good or bad), then 'EXPECTED RESULT:' and the true simulator result of
that action in format F (loss on the informative tokens; fixed wording 0.05, as in every format). The reply is pref1's
first reply of the turn in the chk1 collection (pref1 at T=0 on training-pool problems; the reply it gave to the turn's
prompt before any check result), the action its last action / check line, the result the one transitions.py simulated
for that own move at that state.
Qown row: the same state, ONE question about the same own move, the same result: the same supervised outcomes as P,
only the row structure differs.
Q12 row: the question rows of round 1 (12 questions: own / best legal / off-menu / uniform legal) on a twelfth of the
same states: the same number of supervised outcomes as P (matched outcome tokens), more diverse moves.
Outputs DIR/rows_P.jsonl, rows_Qown.jsonl, rows_Q12.jsonl, sids.txt and a report.
"""
import argparse
import collections
import gzip
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_rows import conseq_row, pick_moves  # noqa: E402
from targets import FIXED_WEIGHT, render  # noqa: E402

HEADER = "EXPECTED RESULT:"
TAG = re.compile(r"<(action|check)>\s*(.*?)\s*</\1>", re.S)


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def own_reply(answer: str):
    """The answer with its last <check>/<action> tag written as an <action> line, cut after it; (text, move) or None."""
    m = list(TAG.finditer(answer or ""))
    if not m:
        return None
    last = m[-1]
    text = answer[:last.start()] + "<action>%s</action>" % " ".join(last.group(2).split())
    return text, " ".join(last.group(2).split())


def p_answer(reply_answer: str, move_rec: dict, fmt: str, metrics0: dict) -> dict:
    """The assistant message of a P row: own answer (weight 0) + header + result block in format F."""
    block, spans = render(fmt, move_rec, metrics0)
    head = reply_answer.rstrip() + "\n"
    content = head + HEADER + "\n" + block
    b0 = len(head) + len(HEADER) + 1
    sup = [(len(head), len(head) + len(HEADER))]          # the header: the decision to predict
    if spans is None:
        sup.append((b0, b0 + len(block)))
    else:
        sup += [(b0 + a, b0 + b) for a, b in spans]
    return {"role": "assistant", "content": content, "supervise_spans": sup, "zero_spans": [(0, len(head))],
            "fixed_weight": FIXED_WEIGHT, "prefix_weight": 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--format", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", nargs="*", default=["engibench", "grid2op", "photonics"])
    ap.add_argument("--per-family", type=int, default=400)
    ap.add_argument("--q", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    c = collections.Counter()
    # 1. pref1's first reply per (episode key, turn)
    replies = {}
    for p in a.episodes:
        for line in opener(p):
            r = json.loads(line)
            if r["variant"] not in a.families:
                continue
            for t in (r["episode"].get("turns") or []):
                chks = t.get("checks") or []
                src = chks[0] if chks else {"reasoning": t.get("reasoning"), "answer": t.get("answer")}
                rep = own_reply(src.get("answer"))
                c["turns"] += 1
                if rep is None or not (src.get("reasoning") or "").strip():
                    c["turns without a readable first reply"] += 1
                    continue
                replies["%s#t%d" % (r["key"], int(t["t"]))] = {"reasoning": src["reasoning"], "answer": rep[0],
                                                               "move_written": rep[1], "move": src.get("move")}
    # 2. the states (first copy of a sid wins) with the own move's simulated result
    pool = collections.defaultdict(list)
    seen = set()
    for p in a.states:
        try:
            for line in opener(p):
                try:
                    st = json.loads(line)
                except ValueError:                         # a garbled line of a file another node is appending to
                    c["unreadable state lines"] += 1
                    continue
                if st["sid"] in seen or st["variant"] not in a.families or st.get("split") != "train":
                    continue
                seen.add(st["sid"])
                rp = replies.get(st["sid"])
                if rp is None:
                    continue
                by = {m["move"]: m for m in st["moves"]}
                mrec = by.get(rp["move"]) if rp["move"] else None
                if mrec is None:
                    c["state: reply move not among its simulated moves"] += 1
                    continue
                st["_reply"], st["_own"] = rp, mrec
                pool[st["variant"]].append(st)
        except (EOFError, OSError):
            pass
    rng = random.Random(a.seed)
    rows = {"P": [], "Qown": [], "Q12": []}
    sids = []
    for v in a.families:
        sts = sorted(pool[v], key=lambda s: s["sid"])
        rng.shuffle(sts)
        sts = sts[:a.per_family]
        n12 = max(1, round(len(sts) / a.q))
        for k, st in enumerate(sts):
            sids.append(st["sid"])
            rp, own = st["_reply"], st["_own"]
            msgs = [{"role": "system", "content": st["system"]}, {"role": "user", "content": st["user"]},
                    dict(p_answer(rp["answer"], own, a.format, st["metrics0"]), reasoning_content=rp["reasoning"])]
            rows["P"].append({"messages": msgs, "supervise": "every_assistant", "domain": v, "tier": "con1_P",
                              "instance": st["problem_id"], "sid": st["sid"], "format": a.format,
                              "con1": {"moves": [own["move"]], "verdicts": [own["verdict"]], "src": [own["src"]]}})
            qr = conseq_row(st, [own], a.format)
            qr["tier"] = "con1_Qown"
            rows["Qown"].append(qr)
            if k < n12:
                q12 = conseq_row(st, pick_moves(st, a.q, rng), a.format)
                q12["tier"] = "con1_Q12"
                rows["Q12"].append(q12)
            c["P %s verdict %s" % (v, own["verdict"])] += 1
        c["states %s (available %d)" % (v, len(pool[v]))] = len(sts)
        c["Q12 states %s" % v] = n12
    for name, rs in rows.items():
        rng.shuffle(rs)
        with open(os.path.join(a.out_dir, "rows_%s.jsonl" % name), "w", encoding="utf-8") as fo:
            for r in rs:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        c["rows %s" % name] = len(rs)
        c["outcomes %s" % name] = sum(len(r["con1"]["moves"]) for r in rs)
    open(os.path.join(a.out_dir, "sids.txt"), "w").write("\n".join(sids) + "\n")
    rep = "\n".join("%-55s %d" % (k, c[k]) for k in sorted(c))
    open(os.path.join(a.out_dir, "report.txt"), "w").write(rep + "\n")
    print(rep)
    print("BUILD_PROWS_DONE")


if __name__ == "__main__":
    main()
