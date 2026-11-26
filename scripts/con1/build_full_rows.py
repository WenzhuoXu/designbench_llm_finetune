"""con1 full consequence row set: every simulated outcome of every state, partitioned per state into rows of up to Q
questions over disjoint move subsets (docs/con1_full_rows.md).

    python build_full_rows.py --states S [S ...] --exclude-rows R [R ...] --forbid-problems queries_v2.jsonl \\
        --tokenizer MODEL_DIR --q 12 --max-tokens 24000 --out-train train.jsonl --out-dev dev.jsonl \\
        --stats stats.json [--lens lens.txt] [--target facet] [--seed 0] [--workers 30]

Rows are build_rows.conseq_row(state, moves, target) unchanged (stage 1's row code and targets.py), so a row of this
set has the same layout, wording and loss weights as a stage-1 row; only which moves share a row differs:
- states: the first line of each sid across --states, in the order given (put the files whose states carry off-menu
  moves first); held-out families, test problems (--forbid-problems) and splits other than train/dev stop the build;
- moves: every simulated move of the state except the (state, move) pairs of the --exclude-rows files (stage 1's
  rows), so the set holds only outcomes stage 1 never trained on;
- partition: R = ceil(n / Q) rows of balanced size (sizes differ by at most 1); the state's own moves, then its
  off-menu / off-grid moves, are dealt round-robin over the rows (each row gets at most one of each kind before any
  gets a second), uniform moves fill the rest; then each row is shuffled. All draws use a random.Random seeded by
  (--seed, sid), so the output is the same bytes on every machine and every run (the trainer's fingerprint holds the
  rows file's sha1);
- length: every row's token count is exact (conseq_format.encode's layout with the model's tokenizer: the sum of the
  separately tokenised pieces; the rows of every 50th state are also tokenised whole and must agree, else the build
  stops); a state whose row would pass --max-tokens gets one more row (smaller rows) until every row fits. A move
  that does not fit even alone is dropped (counted).
Output order: input order of the states (file, line), the rows of a state in order; the trainer shuffles.
--lens writes one line per training row: total tokens, prompt tokens, questions, family (for the time estimate).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))        # the repo root (llm_finetune.rl_dataxp)
from build_rows import HELD_OUT, TRAIN_FAMILIES, answer, conseq_row, opener  # noqa: E402
from targets import FOLLOW_HEAD, question  # noqa: E402

SID_RE = re.compile(r'"sid":\s*"((?:[^"\\]|\\.)*)"')

_TOK = None
_EXCL: dict = {}
_ARGS = None


def _tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(_ARGS["tokenizer"], trust_remote_code=True)
    return _TOK


def _kind(m: dict) -> str:
    if "own" in m["src"]:
        return "own"
    if "offmenu" in m["src"] or "offgrid" in m["src"]:
        return "off"
    return "uni"


def _regime(m: dict) -> str:
    """The outcome's regime as the facet answer states it: refused, or applied and feasible / infeasible after."""
    if m["verdict"] == "refused":
        return "refused"
    return "applied-feasible" if m.get("feasible") else "applied-infeasible"


SHORT_TAIL = "Same answer form."


def follow_text(fmt: str, move: str) -> str:
    """A later question of a row: targets.question's wording (the full answer-format text), or with --follow short its
    first two sentences and 'Same answer form.' (the full text is in the row's first question; it is masked from the
    loss either way, so repeating it ~200 tokens per question only costs compute; 11-26, parent's change)."""
    if _ARGS and _ARGS.get("follow") == "short":
        return FOLLOW_HEAD.format(move=move) + SHORT_TAIL
    return question(fmt, move, False)


def shorten(row: dict, fmt: str) -> dict:
    """build_rows.conseq_row's row with its later questions in follow_text's wording (the first question unchanged)."""
    if not (_ARGS and _ARGS.get("follow") == "short"):
        return row
    msgs = row["messages"]
    for k, mv in zip(range(3, len(msgs), 2), row["con1"]["moves"][1:]):
        assert msgs[k]["role"] == "user" and msgs[k]["content"] == question(fmt, mv, False)
        msgs[k]["content"] = follow_text(fmt, mv)
    return row


def _pieces(tok, st: dict, m: dict, fmt: str) -> tuple:
    """(user-turn tokens, answer tokens) of a move asked as a later question: conseq_format.encode tokenises each piece
    on its own and concatenates, so a row's length is the sum of its pieces' lengths."""
    from llm_finetune.rl_dataxp.conseq_format import GLUE, answer_text
    return (len(tok.encode(GLUE % follow_text(fmt, m["move"]), add_special_tokens=False)),
            len(tok.encode(answer_text(answer(fmt, st, m)), add_special_tokens=False)))


def _head_msgs(st: dict, first: str, fmt: str) -> list:
    return [{"role": "system", "content": st["system"]}, {"role": "user", "content": st["user"] + question(fmt, first, True)}]


def _row_len(tok, row: dict) -> tuple:
    from llm_finetune.rl_dataxp.conseq_format import encode
    e = encode(tok, row["messages"])
    return len(e["prompt_ids"]) + len(e["completion_ids"]), len(e["prompt_ids"])


class Heads:
    """Exact prompt length (conseq_format._head_ids) of the state's prompt with a given first question. The rendered
    prompt is split where the question starts ('QUESTION (no move...'), after the run of newlines that ends the
    state's text: the tokenizer's pre-split never joins a newline run to a following letter, so prefix and suffix
    tokenise apart. The split is checked once per state against the exact count; if it ever differs, every row of
    the state is counted exactly."""

    def __init__(self, tok, st: dict, fmt: str, first: str):
        from llm_finetune.rl_dataxp.conseq_format import _head_ids
        self.tok, self.st, self.fmt = tok, st, fmt
        exact = len(_head_ids(tok, _head_msgs(st, first, fmt)))
        text = self._text(first)
        self.cut = text.rfind("QUESTION (no move is made now)")
        self.pre = len(tok.encode(text[:self.cut], add_special_tokens=False)) if self.cut > 0 else None
        self.additive = self.pre is not None and self.pre + self._suffix(text) == exact

    def _text(self, first: str) -> str:
        return self.tok.apply_chat_template(_head_msgs(self.st, first, self.fmt), tokenize=False, add_generation_prompt=True)

    def _suffix(self, text: str) -> int:
        return len(self.tok.encode(text[self.cut:], add_special_tokens=False))

    def __call__(self, first: str) -> int:
        if self.additive:
            text = self._text(first)
            if text.rfind("QUESTION (no move is made now)") == self.cut:
                return self.pre + self._suffix(text)
        from llm_finetune.rl_dataxp.conseq_format import _head_ids
        return len(_head_ids(self.tok, _head_msgs(self.st, first, self.fmt)))


def partition(moves: list, n_rows: int, rng: random.Random) -> list:
    """n_rows balanced groups: own, then off-menu/off-grid moves dealt round-robin, uniform moves fill; each shuffled."""
    n = len(moves)
    sizes = [n // n_rows + (1 if k < n % n_rows else 0) for k in range(n_rows)]
    groups = {"own": [], "off": [], "uni": []}
    for m in moves:
        groups[_kind(m)].append(m)
    for g in ("own", "off", "uni"):
        rng.shuffle(groups[g])
    rows = [[] for _ in range(n_rows)]
    k = 0
    for m in groups["own"] + groups["off"]:
        rows[k % n_rows].append(m)
        k += 1
    it = iter(groups["uni"])
    for r in range(n_rows):
        while len(rows[r]) < sizes[r]:
            rows[r].append(next(it))
    assert next(it, None) is None and sum(len(r) for r in rows) == n
    for r in rows:
        rng.shuffle(r)
    return rows


def do_state(st: dict, c: collections.Counter):
    """[(split, row json line, total tokens, prompt tokens, questions)] of one state."""
    a = _ARGS
    tok, fmt, q, cap = _tok(), a["target"], a["q"], a["max_tokens"]
    v, sp = st["variant"], st["split"]
    excl = _EXCL.get(st["sid"], ())
    seen, moves = set(), []
    for m in st["moves"]:
        if m["move"] in seen:
            c["%s %s duplicate move text" % (sp, v)] += 1
            continue
        seen.add(m["move"])
        if m["move"] in excl:
            c["%s %s outcomes excluded (stage 1)" % (sp, v)] += 1
            continue
        moves.append(m)
    c["%s %s outcomes simulated" % (sp, v)] += len(st["moves"])
    c["%s %s states" % (sp, v)] += 1
    if not moves:
        c["%s %s states with no outcome left" % (sp, v)] += 1
        return []
    rng = random.Random("con1-full|%d|%s" % (a["seed"], st["sid"]))
    # exact lengths: row = head(first move) + answer(first) + sum over the other moves of (user turn + answer)
    pc = {m["move"]: _pieces(tok, st, m, fmt) for m in moves}
    heads = Heads(tok, st, fmt, moves[0]["move"])
    if not heads.additive:
        c["%s %s states counted row by row (prompt split not additive)" % (sp, v)] += 1
    hcache: dict = {}

    def head(first):
        if first not in hcache:
            hcache[first] = heads(first)
        return hcache[first]

    def length(g):
        h = head(g[0]["move"])
        return h + pc[g[0]["move"]][1] + sum(pc[m["move"]][0] + pc[m["move"]][1] for m in g[1:]), h

    n = len(moves)
    n_rows = int(math.ceil(n / q))
    while True:
        groups = partition(moves, n_rows, rng)
        lens = [length(g) for g in groups]
        if all(L[0] <= cap for L in lens):
            break
        if n_rows >= n:                                    # one move per row already: drop what does not fit
            keep = [k for k, L in enumerate(lens) if L[0] <= cap]
            c["%s %s outcomes dropped: too long even alone" % (sp, v)] += len(groups) - len(keep)
            groups, lens = [groups[k] for k in keep], [lens[k] for k in keep]
            break
        n_rows += 1
    if n_rows > int(math.ceil(n / q)):
        c["%s %s states split finer for the token cap" % (sp, v)] += 1
    built = [(g, shorten(conseq_row(st, g, fmt), fmt), L[0], L[1]) for g, L in zip(groups, lens)]
    # every 50th state (by sid): its rows tokenised whole, as the trainer does, must give the same counts
    if int(hashlib.sha1(st["sid"].encode()).hexdigest()[:8], 16) % a["check_every"] == 0:
        for g, row, tot, pr in built:
            c["length checks"] += 1
            if _row_len(tok, row) != (tot, pr):
                c["length check MISMATCH"] += 1
    out = []
    for g, row, tot, pr in built:
        out.append((sp, json.dumps(row, ensure_ascii=False) + "\n", tot, pr, len(g), v))
        c["%s %s rows" % (sp, v)] += 1
        c["%s %s questions" % (sp, v)] += len(g)
        c["%s %s tokens" % (sp, v)] += tot
        c["%s %s prompt tokens" % (sp, v)] += pr
        c["%s %s rows with q=%d" % (sp, v, len(g))] += 1
        for m in g:
            c["%s %s src %s" % (sp, v, _kind(m))] += 1
            c["%s %s regime %s" % (sp, v, _regime(m))] += 1
    return out


def do_chunk(lines: list):
    c = collections.Counter()
    out = []
    for line in lines:
        st = json.loads(line)
        v, sp = st.get("variant"), st.get("split")
        if v in HELD_OUT or v not in TRAIN_FAMILIES:
            return "state %s of family %s in the states" % (st.get("sid"), v), None, None
        if sp not in ("train", "dev"):
            return "state %s of split %s in the states" % (st.get("sid"), sp), None, None
        if st.get("problem_id") in _ARGS["forbid"]:
            return "state %s is on a held-out test problem %s" % (st.get("sid"), st.get("problem_id")), None, None
        if not st.get("moves"):
            c["%s %s states without moves" % (sp, v)] += 1
            continue
        out.extend(do_state(st, c))
    return None, out, c


def _init(args, excl):
    global _ARGS, _EXCL
    _ARGS, _EXCL = args, excl
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def chunks(paths: list, size: int, c: collections.Counter, sid_src: dict):
    seen = set()
    buf = []
    for p in paths:
        n_file = 0
        try:
            for line in opener(p):
                if not line.strip():
                    continue
                mm = SID_RE.search(line[:400])
                sid = mm.group(1) if mm else json.loads(line)["sid"]
                if sid in seen:
                    c["input: duplicate states skipped"] += 1
                    continue
                seen.add(sid)
                n_file += 1
                buf.append(line)
                if len(buf) >= size:
                    yield buf
                    buf = []
        except EOFError:
            raise SystemExit("%s ends in an incomplete gzip member: the inputs must be complete files" % p)
        sid_src[p] = n_file
    if buf:
        yield buf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", nargs="+", required=True, help="first line of a sid wins: off-menu files first")
    ap.add_argument("--exclude-rows", nargs="*", default=[], help="rows files whose (sid, move) pairs are left out")
    ap.add_argument("--forbid-problems", nargs="*", default=[], help="jsonl files of held-out problems (problem_id)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--target", default="facet")
    ap.add_argument("--follow", choices=("full", "short"), default="full",
                    help="later questions of a row: the full format text (targets.question) or 'Same answer form.'")
    ap.add_argument("--q", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=24000, help="exact tokens per row (prompt + completion)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=8, help="states per task")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--lens", default=None)
    ap.add_argument("--check-every", type=int, default=50, help="tokenise the rows of 1 state in N whole (check)")
    a = ap.parse_args(argv)
    t0 = time.time()
    excl = collections.defaultdict(set)
    n_ex = 0
    for p in a.exclude_rows:
        for line in opener(p):
            r = json.loads(line)
            if r.get("tier") == "con1_conseq":
                excl[r["sid"]].update(r["con1"]["moves"])
                n_ex += len(r["con1"]["moves"])
    forbid = set()
    for p in a.forbid_problems:
        for line in opener(p):
            r = json.loads(line)
            for d in (r, r.get("spec") or {}):
                if isinstance(d, dict) and d.get("problem_id"):
                    forbid.add(d["problem_id"])
    print("excluding %d (state, move) pairs of %d states; %d forbidden problems; %.0f s" % (
        n_ex, len(excl), len(forbid), time.time() - t0), flush=True)
    args = {"tokenizer": a.tokenizer, "target": a.target, "q": a.q, "max_tokens": a.max_tokens, "seed": a.seed,
            "follow": a.follow,
            "forbid": forbid, "check_every": a.check_every}
    c = collections.Counter()
    sid_src: dict = {}
    sha = hashlib.sha1()
    n_rows = 0
    from multiprocessing import get_context
    ctx = get_context("fork")
    tmp_t, tmp_d = a.out_train + ".tmp", a.out_dev + ".tmp"
    lens = open(a.lens + ".tmp", "w") if a.lens else None
    longest = (0, None)
    with open(tmp_t, "w", encoding="utf-8") as ft, open(tmp_d, "w", encoding="utf-8") as fd, \
            ctx.Pool(a.workers, initializer=_init, initargs=(args, dict(excl))) as pool:
        for err, out, cc in pool.imap(do_chunk, chunks(a.states, a.chunk, c, sid_src), chunksize=1):
            if err:
                pool.terminate()
                raise SystemExit("ABORT: " + err)
            c.update(cc)
            for sp, line, tot, pr, nq, v in out:
                if sp == "train":
                    ft.write(line)
                    sha.update(line.encode("utf-8"))
                    n_rows += 1
                    if lens:
                        lens.write("%d %d %d %s\n" % (tot, pr, nq, v))
                    if tot > longest[0]:
                        longest = (tot, v)
                else:
                    fd.write(line)
            if n_rows and n_rows % 20000 < len(out):
                print("%d train rows, %.0f s" % (n_rows, time.time() - t0), flush=True)
    if lens:
        lens.close()
        os.replace(a.lens + ".tmp", a.lens)
    os.replace(tmp_t, a.out_train)
    os.replace(tmp_d, a.out_dev)
    fams = TRAIN_FAMILIES
    summary = {}
    for sp in ("train", "dev"):
        for v in fams:
            g = lambda k: c.get("%s %s %s" % (sp, v, k), 0)     # noqa: E731
            if not g("states"):
                continue
            summary["%s/%s" % (sp, v)] = {
                "states": g("states"), "outcomes_simulated": g("outcomes simulated"),
                "excluded_stage1": g("outcomes excluded (stage 1)"),
                "dropped_too_long": g("outcomes dropped: too long even alone"), "rows": g("rows"),
                "questions": g("questions"), "tokens": g("tokens"), "prompt_tokens": g("prompt tokens"),
                "src": {k: g("src " + k) for k in ("own", "off", "uni")},
                "regime": {k[len("%s %s regime " % (sp, v)):]: n for k, n in c.items()
                           if k.startswith("%s %s regime " % (sp, v))}}
    res = {"q": a.q, "max_tokens": a.max_tokens, "target": a.target, "follow": a.follow, "seed": a.seed, "train_rows": n_rows,
           "train_sha1": sha.hexdigest(), "longest": {"tokens": longest[0], "family": longest[1]},
           "states_from_file": sid_src, "excluded_pairs": n_ex, "seconds": round(time.time() - t0, 1),
           "workers": a.workers, "families": summary,
           "other": {k: n for k, n in c.items() if not any(k.startswith(s + " ") for s in ("train", "dev"))}}
    with open(a.stats + ".tmp", "w") as fo:
        json.dump(res, fo, indent=1, sort_keys=True)
    os.replace(a.stats + ".tmp", a.stats)
    if c.get("length check MISMATCH") or not c.get("length checks"):
        raise SystemExit("ABORT: %d of %d whole-row token counts differ from the piece sums" % (
            c.get("length check MISMATCH", 0), c.get("length checks", 0)))
    print("whole-row token counts of %d rows (1 state in %d) equal the piece sums" % (c["length checks"], a.check_every))
    tq = sum(s["questions"] for k, s in summary.items() if k.startswith("train/"))
    tt = sum(s["tokens"] for k, s in summary.items() if k.startswith("train/"))
    print("train rows %d, questions %d, tokens %.1fM (%.0f per question), longest %d (%s); sha1 %s; %.0f s" % (
        n_rows, tq, tt / 1e6, tt / max(1, tq), longest[0], longest[1], res["train_sha1"], res["seconds"]))
    print("BUILD_FULL_ROWS_DONE")


if __name__ == "__main__":
    main()
