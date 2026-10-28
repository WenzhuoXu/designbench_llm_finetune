"""Package the datav4 datasets: one self-contained directory per spelling
(``datav4_multitool``, ``datav4_set_parameter``) plus the shared,
spelling-independent corpus directory (``datav4_corpus``), ready to upload to
bridges2 under ``/ocean/projects/mch250030p/shared/dataxp/`` (the owner's
decision of 2026-10-23, docs/corpus_native_domains.md "two spellings"; the
spellings themselves: docs/dataxp_spellings.md).

Layout written under --out (one spelling)::

    datav4_<spelling>/
      README.md            what the spelling is (verbatim action lines per family), the served
                           instruction + sha1, acceptance and state-selection rules, evaluation-only
                           problems, sources, dates, row counts
      instruction.txt      the served instruction block (dataxp.ownrows.common.instruction_for
                           minus the grammar); its sha1[:12] is every row's instruction_sha1
      grammar/<family>.txt the served grammar of that family under this spelling
                           (domain.grammar() with DATAXP_SPELLING set, in a subprocess)
      rows/<family>/{train_own,train_own_act,dev_in_own,dev_in_own_act}.jsonl + manifest_own.json
      rows/_groups/<group>/{manifest_own.json,select_report.json}   the tier-group files verbatim
      CORPUS.md            where the corpora are (shared datav4_corpus/, or corpus/ here)
      corpus/              only with --corpus-mode copy
      eval/                multitool only: the base_nothink r1 episodes + NOTE.md
      manifest.json        per family: spelling, counts, states, acceptance stats, sha1 of every
                           file, job ids, instruction and grammar sha1

and, with the default ``--corpus-mode shared``, next to it::

    datav4_corpus/
      README.md, manifest.json
      <family>/accepted/{problems,trajectories,pairs}.jsonl    (batches concatenated)
      <family>/report.json | report_<batch>.json, sources.json  (every source file's sha1)

Why one corpus copy: the corpora are spelling-blind (generation never sees the
spelling; docs/dataxp_spellings.md), the rows are complete training inputs on
their own (full served prompts), and the corpora run to a few hundred MB (run2
pairs alone are ~12 MB per variant). So they are stored once and each spelling
directory names them in CORPUS.md with the sha1s it was built against;
``--check`` verifies both. ``--corpus-mode copy`` puts a copy in each.

Every row must carry ``spelling`` equal to the directory's; a row whose
spelling disagrees is refused. A row without the key (tiers built before the
switch existed) is refused too, unless ``--stamp-spelling`` is given and both
its ``instruction_sha1`` and its action line imply the directory's spelling.
Every served user text must end with instruction.txt + the packaged grammar
(the "served grammar" claim is verified per row), and every action line is
re-parsed under the spelling and compared with the stored action dict.

Usage (on drc, where the inputs live)::

    python -m dataxp.package_datav4 --spelling multitool \
        --rows-root /storage/wxu/corpus/rows --out /storage/wxu/datav4/datav4_multitool \
        --native-corpus /storage/wxu/corpus/out \
        --v2-corpus run2_supp=/storage/wxu/sftprobe/src/dataxp/run2_supp/batch1_idx0130-0259 \
        --v2-corpus-ref run2=/ocean/projects/mch250030p/shared/dataxp/run2 \
        --eval-problems-native /storage/wxu/corpus/harness/problems_v3_native.jsonl \
        --eval-problems-v2 /storage/wxu/corpus/harness/problems_v2.jsonl \
        --eval-episodes /storage/wxu/corpus/harness/eval/r1 [--stamp-spelling]
    python -m dataxp.package_datav4 --spelling set_parameter \
        --rows-root /storage/wxu/corpus/rows_sp --out /storage/wxu/datav4/datav4_set_parameter ...
    python -m dataxp.package_datav4 --check --out /storage/wxu/datav4/datav4_multitool [--parse-check]

DATAXP_SPELLING is read once per process by dataxp.domain, so everything that
needs a domain instance (grammar, instruction, format_move, parse_move) runs in
a subprocess started with the variable set; this process never imports
dataxp.domain.

Appending batches (docs/datav4_append_plan.md). The b001 directories on
bridges2 are never rewritten; a later batch is built as a directory of its own
that is copied to ``<spelling dir>/batches/b00N/`` (rows) and
``datav4_corpus/<family>/accepted_b00N/`` (corpora)::

    datav4_<spelling>/batches/b00N/
      BATCH.md            recipe, funnel per family (states -> replies -> finished ->
                          usable -> rows), checks, sources, jobs
      manifest.json       kind "batch": counts, sha1 of every file, served instruction and
                          grammar texts, prior trees checked, evaluation files, source jobs
      rows/<family>/{train_own,train_own_act,dev_in_own,dev_in_own_act}.jsonl + manifest_own.json
      rows/_groups/<group>/{manifest_own.json,select_report.json}
    datav4_corpus/<family>/accepted_b00N/{problems,trajectories,pairs}.jsonl, report.json, sources.json

    python -m dataxp.package_datav4 --append --spelling multitool --batch b002 \
        --rows-root /storage/wxu/corpus/b002/rows --out /storage/wxu/datav4_b002/multitool/b002 \
        --prior /storage/wxu/datav4/datav4_multitool \
        --eval-problems-native /storage/wxu/corpus/harness/problems_v3_native.jsonl \
        --eval-problems-v2 /storage/wxu/sftprobe/problems_v2.jsonl --draw-passes 2 \
        [--twin-states /storage/wxu/corpus/b002/rows_sp] [--source-job GROUP=TEXT ...]
    python -m dataxp.package_datav4 --corpus-batch b004 --corpus-dir <staging>/datav4_corpus \
        --corpus-src grid2op=/storage/wxu/corpus/out_b004/grid2op/grid2op ... \
        --eval-problems-native ... --eval-problems-v2 ... [--corpus-prior <dir or problems.jsonl> ...]
    python -m dataxp.package_datav4 --check --out <batch dir> [--prior ...] [--eval-problems-* ...]
    python -m dataxp.package_datav4 --check --corpus-batch b004 --corpus-dir <dir>

Append rules (each refuses the build unless noted): every row passes the
checks of the main build (spelling, instruction sha1, action-line form, served
grammar tail, own = own_act row for row, train/dev problems disjoint, re-parse
under the spelling); no row's sid (domain|instance|turn) or (instance, turn) is
in any prior tree (a spelling root given as --prior brings its rows/ and every
batches/b*/rows/); a problem present in a prior tree (or --prior-states) keeps
its split; no row of a held-out family (thermal2, inventory2, molopt); the
served instruction and grammars equal the prior spelling root's
instruction.txt / grammar/<family>.txt; rows of a problem whose canonical spec
(json of the spec minus problem_id / domain_variant / index, sorted keys)
equals an evaluation problem of either file are DROPPED and counted (not a
refusal); the sampling recipe is verified where the files record it (K per
state, --prompt probe, the model name, no thinking text, reply tokens <= the
cap, min_pct 90 and no reasoning word limit in the tier manifest, draw_pass <=
--draw-passes, no dry-run replies); every row is stamped "batch" and
"draw_passes" (the passes the batch ran: 1 = the K-reply draw, 2 = plus the one
redraw of the truncated keys) and, when the sampler recorded it, "draw_pass"
(the pass that drew the row's own reply).
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
SPELLINGS = ("multitool", "set_parameter")
V2_FAMILIES = ("frame3d2", "circuit2", "grid2", "traffic2", "thermal2", "seir2", "inventory2")
NATIVE_FAMILIES = ("grid2op", "photonics", "molopt", "engibench")
FAMILIES = V2_FAMILIES + NATIVE_FAMILIES
TIERS = ("train_own", "train_own_act", "dev_in_own", "dev_in_own_act")
CORPUS_FILES = ("problems.jsonl", "trajectories.jsonl", "pairs.jsonl")
ACTION_RE = re.compile(r"^<action>(.*)</action>$", re.S)
GRAMMAR_LEAD = "The operations in this problem are exactly:\n"
N_EVAL_NATIVE = 60
BRIDGES2_ROOT = "/ocean/projects/mch250030p/shared/dataxp"
HELD_OUT = ("thermal2", "inventory2", "molopt")       # never given rows (docs/slope_test_prereg.md)
BATCH_RE = re.compile(r"^b\d{3}$")
CANON_DROP = ("problem_id", "domain_variant", "index")
# the b001 recipe (docs/corpus_native_domains.md "datav4 delivered")
RECIPE_DEFAULT = {"k": 4, "max_tokens": 2000, "temperature": 0.6, "prompt": "probe",
                  "thinking": False, "model": "Qwen3.8-27B", "min_pct": 90.0,
                  "mode": "own_nothink"}

# Known provenance (docs/corpus_native_domains.md, docs/dataxp_v2.md,
# docs/sft_probe_rat2.md as of 2026-10-23). --provenance <json> deep-merges
# over this; anything the docs do not record says so here.
KNOWN_PROVENANCE = {
    "_source": "docs/corpus_native_domains.md, docs/dataxp_v2.md, docs/sft_probe_rat2.md (2026-10-23); "
               "override or extend with --provenance <json>",
    "rows": {
        "multitool": {
            "v2verb": "drc job bw_rows4 (the v2 five re-sampled in verb spelling after the SET_PARAM-spelled "
                      "run bw_rows2 was cancelled at ~3,000 of 11,300 replies); Qwen3.8-27B thinking off, "
                      "K=4, 2,000-token cap, 64 streams",
            "grid2op": "drc job bw_rows2 (167 states, problems 0-68); K=4, 2,000-token cap",
            "engibench": "drc job bw_rows3 (problems <= 69; 39 rows of 164 states); K=4, 2,000-token cap",
            "photonics": "drc job bw_rows3 (problems <= 69; 11 rows of 117 states); K=4, 2,000-token cap",
        },
        "set_parameter": {},
    },
    "corpus": {
        "grid2op": "drc job j1467: 130 problems, seed 0, 2026-10-23 (128 included), 94 min",
        "molopt": "drc job j1469: 130 problems, seed 0, 2026-10-23, 34 min",
        "engibench": "drc job j1465 (8 workers): 130 problems, seed 0, 2026-10-23, 4.8 h",
        "photonics": "drc job j1466 (4 workers): 130 problems, seed 0, 2026-10-23, 6.7 h",
        "run2": "bridges2 %s/run2/<variant>/ (2026-10-02; 130 problems per variant, problems 0000-0129)"
                % BRIDGES2_ROOT,
        "run2_supp": "drc /storage/wxu/sftprobe/src/dataxp/run2_supp/batch1_idx0130-0259/<dom>/accepted "
                     "(problems 0130-0259; the DX-26 / sftprobe source)",
    },
    "eval": {
        "r1": "Orchard job 160740 (2026-10-23, 1.1 GPU-h): base_nothink = un-tuned Qwen3.8-27B, thinking "
              "off, 1,024-token cap, multitool (verb-spelled) prompts, 60 evaluation problems per native "
              "family, one run; copied to drc /storage/wxu/corpus/harness/eval/r1/",
    },
}

SPELLING_BLURB = {
    "multitool": (
        "Every family's operations are its own verbs with typed arguments, written "
        "`VERB(arg, ...)`: the verb surface over the seven v2 domains (docs/dataxp_verb_surface.md; "
        "the stored actions stay `{i, p, v, v_old}`) and the native verbs of the four native domains "
        "(docs/dataxp_move_contract.md; stored actions `{verb, args}`). The action line is "
        "`<action>VERB(arg, ...)</action>`; categorical values are written by name (`S355`); the "
        "served instruction is `INSTRUCTION_NATIVE` followed by the family's verb grammar."),
    "set_parameter": (
        "One wrapper tool, `SET_PARAM(element, parameter, value)`, whose grammar documents every "
        "operation of the verb table operation by operation, so the model is told the same things as "
        "under multitool and only the action syntax differs. The seven v2 domains use their real stored "
        "parameters (`SET_PARAM(3, active, 1)` installs a member; categorical values are the stored "
        "codes, `SET_PARAM(3, mat, 1)`); the four native domains use the generic encoding "
        "`SET_PARAM(element, operation, value)` with element = the operation's first integer argument, "
        "operation = the verb in lower case, value = the remaining argument or a JSON list of them "
        "(`SET_PARAM(3, redispatch, [5, 20])`, `SET_PARAM(0, do_nothing, 0)`). The action line is "
        "`<action>SET_PARAM(element, parameter, value)</action>`; the served instruction is "
        "`INSTRUCTION_SET_PARAM` followed by the family's SET_PARAM operation list."),
}
COMMON_BLURB = (
    "Both spellings are bijections on the SAME legal moves: states, teacher, Phi, acceptance rule and "
    "the stored `action` / `gold_action` dicts are identical between `datav4_multitool` and "
    "`datav4_set_parameter`; only the served prompts (instruction, grammar, history lines) and the "
    "assistant's action line differ. The own-action rows were sampled once per spelling.")

ACCEPTANCE_RULE = (
    "At each selected repair-phase state the un-tuned Qwen3.8-27B is prompted with the training "
    "observation (thinking off), K replies are drawn, each reply's last `<action>` is parsed with "
    "`domain.parse_move` under this spelling, applied on a clone of the state and scored by its Phi "
    "percentile among `domain.legal_moves(st, rng)` (Phi via `dataxp.domain.PhiEvaluator`, offset rule "
    "`infeasible`; percentile = share of legal moves with Phi <= the pick's). Per state, among the "
    "replies that parsed, applied, are not a no-op and carry a non-empty written reasoning without a "
    "`<think>` tag, the highest-percentile reply is taken (ties: lowest "
    "sample index); the state becomes a row iff that pick is at or above the `min_pct` percentile "
    "(90) OR equals the teacher's move (`Move.matches`, rel 1e-2). A state without such a reply is "
    "dropped from both arms, so `own` and `own_act` are matched row for row. The row's assistant turn "
    "is `<action>format_move(pick)</action>`; `own` keeps the model's written reasoning as "
    "`reasoning_content`, `own_act` leaves it empty. (`dataxp/ownrows/build_tiers.py`; recipe "
    "`dataxp/ownrows/README.md`.)")

STATE_RULE = (
    "A state is every turn of an accepted trajectory whose `binding_constraint` is not None (the "
    "repair phase, as `render_cot.iter_states` stratifies). Per family the problems with at least one "
    "such turn are shuffled with `stable_seed(\"ownrows_select\", variant, seed)` and taken whole: "
    "into train until `--per-domain` states, then into dev until `--dev-per-domain` (train and dev "
    "problems disjoint). The state is replayed from `load(spec)` through `apply_action` over the "
    "previous turns' actions (deterministic domains: the replayed metrics are checked against the "
    "recorded ones). The served prompt is `render_cot.policy_observation` split as the collaborator's "
    "observation_v2 splits it: system = the DOMAIN/GOAL head, user = the rest + the instruction "
    "block + the grammar. (`dataxp/ownrows/select_states.py`.)")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_json(path: Path, obj) -> None:
    write_text(path, json.dumps(obj, indent=1, sort_keys=False) + "\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl_lines(path: Path) -> List[Tuple[str, dict]]:
    """[(raw line without its newline, parsed object)] - the raw line is what
    gets copied when the row is left untouched, so sha1s stay meaningful."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            s = line.rstrip("\r\n")
            if not s.strip():
                continue
            try:
                out.append((s, json.loads(s)))
            except ValueError as e:
                raise SystemExit("%s line %d: not JSON (%s)" % (path, n, e))
    return out


def count_jsonl(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def mtime_iso(path: Path) -> str:
    return _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def problem_index(pid: str) -> Optional[int]:
    m = re.search(r"_(\d+)$", pid)
    return int(m.group(1)) if m else None


def human_mb(n: int) -> str:
    return "%.1f MB" % (n / 1e6)


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def git_head() -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def code_sha1s() -> Dict[str, str]:
    """sha1 of the dataxp files the grammars and instructions come from."""
    out = {}
    for rel in ["dataxp/domain.py", "dataxp/ownrows/common.py", "dataxp/render_cot.py"]:
        p = REPO / rel
        if p.is_file():
            out[rel] = sha1_file(p)
    for p in sorted((REPO / "dataxp" / "domains").glob("*.py")):
        out["dataxp/domains/" + p.name] = sha1_file(p)
    return out


# ---------------------------------------------------------------------------
# the worker: everything that needs a domain instance (DATAXP_SPELLING set)
# ---------------------------------------------------------------------------
def worker_main(job_path: str, out_path: str) -> int:
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    job = read_json(Path(job_path))
    from dataxp.domain import Move, current_spelling, speaks_verbs  # noqa: E402
    from dataxp.domains import get_domain  # noqa: E402
    from dataxp.ownrows import common as C  # noqa: E402

    sp = current_spelling()
    if sp != job["spelling"]:
        raise SystemExit("worker: DATAXP_SPELLING resolved to %s, wanted %s" % (sp, job["spelling"]))
    # the set_parameter instruction as served before its VERB sentence was
    # reworded (sha1 f99b61d776f1 in docs/dataxp_spellings.md): the action
    # line replaced, the explaining sentence still saying VERB
    old_sp = C.INSTRUCTION_NATIVE.replace(C._NATIVE_ACTION_LINE,
                                          "<action>SET_PARAM(element, parameter, value)</action>")
    instructions = {
        C.INSTRUCTION_NATIVE_SHA1: {"spelling": "multitool", "name": "INSTRUCTION_NATIVE",
                                    "text": C.INSTRUCTION_NATIVE, "current": True},
        C.INSTRUCTION_SET_PARAM_SHA1: {"spelling": "set_parameter", "name": "INSTRUCTION_SET_PARAM",
                                       "text": C.INSTRUCTION_SET_PARAM, "current": True},
        sha1_text(old_sp)[:12]: {"spelling": "set_parameter", "name": "INSTRUCTION_SET_PARAM (pre-reword, "
                                 "VERB sentence still present)", "text": old_sp, "current": False},
    }
    res = {"spelling": sp, "instructions": instructions, "families": {}}
    for fam in job["families"]:
        d = get_domain(fam)
        if not speaks_verbs(d):
            raise SystemExit("worker: %s does not speak verbs; datav4 covers verb-speaking families only" % fam)
        g = d.grammar()
        instr, sha = C.instruction_for(d)
        base = instructions[sha]["text"]
        if instr != base + GRAMMAR_LEAD + g.rstrip() + "\n":
            raise SystemExit("worker: instruction_for(%s) is not base + lead + grammar" % fam)
        kind = "native" if getattr(d, "native_moves", False) else "verb_surface"
        examples = []
        for a in job.get("examples", {}).get(fam, []):
            try:
                examples.append(d.format_move(Move.from_dict(a)))
            except Exception as e:  # pragma: no cover - reported, not fatal
                examples.append("<format_move failed: %s>" % str(e)[:80])
        parse = {"n": 0, "ok": 0, "failures": []}
        for text, a in job.get("parse", {}).get(fam, []):
            parse["n"] += 1
            ref = Move.from_dict(a)
            mv = d.parse_move(text)
            if mv is None:
                why = "unparsed"
            elif not mv.matches(ref, rel=1e-5):
                why = "parsed to %s, stored %s" % (mv.as_dict(), ref.as_dict())
            else:
                parse["ok"] += 1
                continue
            if len(parse["failures"]) < 20:
                parse["failures"].append({"text": text, "why": why})
        res["families"][fam] = {"kind": kind, "grammar": g, "instruction_sha1": sha,
                                "examples": examples, "parse": parse}
    write_json(Path(out_path), res)
    return 0


def run_worker(spelling: str, job: dict) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="datav4_worker_"))
    try:
        jp, op = tmp / "job.json", tmp / "out.json"
        write_json(jp, job)
        env = dict(os.environ)
        env["DATAXP_SPELLING"] = spelling
        env.setdefault("PYTHONHASHSEED", "0")
        env["MKL_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        cmd = [sys.executable, "-m", "dataxp.package_datav4", "--_worker", str(jp), str(op)]
        r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("worker failed (DATAXP_SPELLING=%s):\n%s\n%s" % (spelling, r.stdout[-2000:],
                                                                             r.stderr[-4000:]))
        return read_json(op)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------
def discover_groups(rows_root: Path, names: Optional[List[str]]) -> dict:
    if not rows_root.is_dir():
        raise SystemExit("--rows-root %s is not a directory" % rows_root)
    cands = names or sorted(p.name for p in rows_root.iterdir()
                            if (p / "tiers" / "manifest_own.json").is_file())
    if not cands:
        raise SystemExit("no <group>/tiers/manifest_own.json under %s" % rows_root)
    groups = {}
    for g in cands:
        tiers = rows_root / g / "tiers"
        mp = tiers / "manifest_own.json"
        if not mp.is_file():
            raise SystemExit("group %s: %s missing" % (g, mp))
        rec = {"name": g, "dir": rows_root / g, "tiers_dir": tiers, "manifest": read_json(mp),
               "manifest_sha1": sha1_file(mp), "tiers": {}, "source_files": {}}
        sr = rows_root / g / "select_report.json"
        rec["select_report"] = read_json(sr) if sr.is_file() else None
        rec["select_report_path"] = sr if sr.is_file() else None
        for t in TIERS:
            p = tiers / (t + ".jsonl")
            if not p.is_file():
                raise SystemExit("group %s: %s missing" % (g, p))
            rec["tiers"][t] = read_jsonl_lines(p)
            rec["source_files"][t] = {"path": str(p), "sha1": sha1_file(p), "bytes": p.stat().st_size,
                                      "rows": len(rec["tiers"][t]), "mtime": mtime_iso(p)}
        groups[g] = rec
    return groups


def split_by_family(groups: dict) -> Tuple[dict, dict]:
    """family -> {tier: [(line, obj)]}, family -> group name. A family may come
    from one group only."""
    fam_rows: Dict[str, Dict[str, list]] = collections.defaultdict(lambda: {t: [] for t in TIERS})
    fam_group: Dict[str, str] = {}
    for g, rec in groups.items():
        for t in TIERS:
            for line, obj in rec["tiers"][t]:
                fam = obj.get("domain")
                if not isinstance(fam, str) or not fam:
                    raise SystemExit("group %s %s: a row without a domain field" % (g, t))
                if fam_group.setdefault(fam, g) != g:
                    raise SystemExit("family %s appears in groups %s and %s" % (fam, fam_group[fam], g))
                fam_rows[fam][t].append((line, obj))
    return dict(fam_rows), fam_group


def validate_family_rows(fam: str, spelling: str, tiers: Dict[str, list], accepted: dict,
                         grammar: str, eval_ids: Optional[set], stamp: bool):
    """Checks every row of one family. Returns (report, errors, warnings,
    out_lines: {tier: [line]}). `accepted` maps an instruction sha1[:12] to its
    text (None when the text is unknown) for THIS spelling."""
    errors, warnings = [], []
    rep = {"rows": {t: len(tiers[t]) for t in TIERS}, "stamped": 0,
           "instruction_sha1": collections.Counter(), "tail_checked": 0, "tail_ok": 0,
           "tail_unverifiable": 0, "served_grammar_variants": []}
    variants: Dict[str, int] = collections.OrderedDict()
    out_lines = {t: [] for t in TIERS}
    expect_tail = {sha: (txt + GRAMMAR_LEAD + grammar.rstrip() + "\n") if txt else None
                   for sha, txt in accepted.items()}

    def err(where, msg):
        if len(errors) < 50:
            errors.append("%s: %s" % (where, msg))
        elif len(errors) == 50:
            errors.append("... more errors suppressed")

    for t in TIERS:
        act_arm = t.endswith("_act")
        for idx, (line, obj) in enumerate(tiers[t]):
            where = "%s/%s.jsonl#%d" % (fam, t, idx)
            if obj.get("domain") != fam:
                err(where, "domain %r != %s" % (obj.get("domain"), fam))
            msgs = obj.get("messages") or []
            roles = [m.get("role") for m in msgs] if isinstance(msgs, list) else []
            if roles != ["system", "user", "assistant"]:
                err(where, "messages roles %s" % roles)
                out_lines[t].append(line)
                continue
            content = msgs[2].get("content") or ""
            m = ACTION_RE.match(content.strip())
            if not m or "<action>" in m.group(1):
                err(where, "assistant content is not exactly one <action>...</action> line")
                out_lines[t].append(line)
                continue
            inner = m.group(1).strip()
            is_sp_line = inner.startswith("SET_PARAM(")
            line_spelling = "set_parameter" if is_sp_line else "multitool"
            if line_spelling != spelling:
                err(where, "action line %r is not written in the %s spelling" % (inner[:60], spelling))
            sha = obj.get("instruction_sha1")
            rep["instruction_sha1"][str(sha)] += 1
            if sha not in accepted:
                err(where, "instruction_sha1 %r is not an accepted %s instruction (accepted: %s)"
                    % (sha, spelling, ", ".join(sorted(accepted))))
            sp = obj.get("spelling")
            if sp is None:
                if not stamp:
                    err(where, "row has no spelling key (refused; --stamp-spelling stamps rows whose "
                               "instruction_sha1 and action line both imply %s)" % spelling)
                elif sha in accepted and line_spelling == spelling:
                    obj["spelling"] = spelling
                    line = json.dumps(obj)
                    rep["stamped"] += 1
                else:
                    err(where, "cannot stamp: instruction_sha1 %r / action line do not imply %s"
                        % (sha, spelling))
            elif sp != spelling:
                err(where, "row spelling %r != %s" % (sp, spelling))
            rc = msgs[2].get("reasoning_content")
            if act_arm and rc not in ("", None):
                err(where, "own_act row carries reasoning_content")
            if not act_arm and not rc:
                err(where, "own row has empty reasoning_content")
            if "action" not in obj:
                err(where, "no action dict")
            exp = expect_tail.get(sha)
            if exp is None:
                rep["tail_unverifiable"] += 1
            else:
                rep["tail_checked"] += 1
                user = msgs[1].get("content") or ""
                if user.endswith(exp):
                    rep["tail_ok"] += 1
                else:
                    base = accepted[sha]
                    k = user.rfind(base)
                    served = user[k + len(base) + len(GRAMMAR_LEAD):] if k >= 0 else "<instruction text not found in user message>"
                    variants[served] = variants.get(served, 0) + 1
            if eval_ids and obj.get("instance") in eval_ids:
                err(where, "instance %s is an evaluation-only problem" % obj.get("instance"))
            out_lines[t].append(line)

    for split, pre in (("train", "train_"), ("dev", "dev_in_")):
        a_, b_ = tiers[pre + "own"], tiers[pre + "own_act"]
        if len(a_) != len(b_):
            err("%s/%s" % (fam, split), "own has %d rows, own_act %d (must match row for row)"
                % (len(a_), len(b_)))
        for i, ((_, oa), (_, ob)) in enumerate(zip(a_, b_)):
            ka = (oa.get("instance"), oa.get("turn"), (oa.get("messages") or [{}])[-1].get("content"),
                  json.dumps(oa.get("action"), sort_keys=True))
            kb = (ob.get("instance"), ob.get("turn"), (ob.get("messages") or [{}])[-1].get("content"),
                  json.dumps(ob.get("action"), sort_keys=True))
            if ka != kb:
                err("%s/%s#%d" % (fam, split, i), "own and own_act rows differ (%s vs %s)" % (ka[:2], kb[:2]))
    train_p = sorted({o.get("instance") for _, o in tiers["train_own"]})
    dev_p = sorted({o.get("instance") for _, o in tiers["dev_in_own"]})
    both = sorted(set(train_p) & set(dev_p))
    if both:
        err(fam, "problems in both train and dev: %s" % both[:10])
    rep["problems"] = {"train": train_p, "dev": dev_p}
    rep["served_grammar_variants"] = [{"rows": n, "text": txt} for txt, n in variants.items()]
    rep["instruction_sha1"] = dict(rep["instruction_sha1"])
    if variants:
        warnings.append("%s: %d of %d rows were served a grammar that differs from the packaged one "
                        "(%d distinct variant(s))" % (fam, sum(variants.values()), rep["tail_checked"],
                                                      len(variants)))
    if rep["tail_unverifiable"]:
        warnings.append("%s: %d rows carry an instruction sha1 whose text is unknown; served tail "
                        "not verified" % (fam, rep["tail_unverifiable"]))
    return rep, errors, warnings, out_lines


def family_manifest_own(gm: dict, group: str, group_manifest_rel: str, group_sha1: str, fam: str,
                        tiers: Dict[str, list]) -> dict:
    """The per-family view of a group's manifest_own.json: per_domain filtered
    to the family, the per-row statistics recomputed from the family's rows."""
    out = {"family": fam, "derived_from": {"group": group, "manifest": group_manifest_rel,
                                           "sha1": group_sha1,
                                           "note": "per_domain filtered to this family; mean_pct, "
                                                   "same_move_share and reasoning_words_median "
                                                   "recomputed from this family's rows; dropped "
                                                   "reasons are group-level (see the group manifest)"},
           "min_pct": gm.get("min_pct"), "tag": gm.get("tag")}
    for split, pre in (("train", "train_"), ("dev", "dev_in_")):
        g = gm.get(split) or {}
        per = (g.get("per_domain") or {}).get(fam, [0, 0])
        rows = [o for _, o in tiers[pre + "own"]]
        pcts = [float(o["own_pct"]) for o in rows if o.get("own_pct") is not None]
        same = [bool(o.get("own_same_move")) for o in rows]
        words = sorted(len(((o.get("messages") or [{}])[-1].get("reasoning_content") or "").split())
                       for o in rows)
        out[split] = {
            "states": int(per[1]), "kept": int(per[0]), "rows": len(rows),
            "mean_pct": (sum(pcts) / len(pcts)) if pcts else None,
            "same_move_share": (sum(same) / len(same)) if same else None,
            "reasoning_words_median": words[len(words) // 2] if words else None,
            "files": {arm: "%s%s.jsonl" % (pre, arm) for arm in ("own", "own_act")},
        }
        if rows and int(per[0]) != len(rows):
            out[split]["note"] = ("group manifest says %d kept for this family, %d rows packaged"
                                  % (int(per[0]), len(rows)))
    return out


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------
def resolve_corpus_sources(families: List[str], a) -> Tuple[dict, dict, list]:
    """family -> [{"batch", "accepted", "report"}] (copied), family -> [{"batch",
    "path"}] (referenced only), warnings."""
    src: Dict[str, list] = collections.defaultdict(list)
    refs: Dict[str, list] = collections.defaultdict(list)
    warns = []
    if a.native_corpus:
        root = Path(a.native_corpus)
        for fam in families:
            if fam not in NATIVE_FAMILIES:
                continue
            for base in (root / fam / fam, root / fam):
                if (base / "accepted" / "trajectories.jsonl").is_file():
                    src[fam].append({"batch": "main", "accepted": base / "accepted",
                                     "report": base / "report.json"})
                    break
            else:
                warns.append("native corpus of %s not found under %s (<v>/<v>/accepted or <v>/accepted)"
                             % (fam, root))
    for spec in a.v2_corpus or []:
        if "=" not in spec:
            raise SystemExit("--v2-corpus expects batch=root, got %r" % spec)
        batch, root = spec.split("=", 1)
        root = Path(root)
        for fam in families:
            if fam not in V2_FAMILIES:
                continue
            base = root / fam
            if (base / "accepted" / "trajectories.jsonl").is_file():
                src[fam].append({"batch": batch, "accepted": base / "accepted", "report": base / "report.json"})
            else:
                warns.append("v2 corpus batch %s of %s not found at %s/accepted" % (batch, fam, base))
    for spec in a.v2_corpus_ref or []:
        if "=" not in spec:
            raise SystemExit("--v2-corpus-ref expects batch=path, got %r" % spec)
        batch, path = spec.split("=", 1)
        for fam in families:
            if fam in V2_FAMILIES:
                refs[fam].append({"batch": batch, "path": path.rstrip("/") + "/" + fam})
    return dict(src), dict(refs), warns


def corpus_source_signature(sources: list) -> dict:
    sig = {}
    for s in sources:
        files = {}
        for name in CORPUS_FILES:
            p = s["accepted"] / name
            if p.is_file():
                files[name] = {"sha1": sha1_file(p), "bytes": p.stat().st_size, "mtime": mtime_iso(p)}
        rep = s["report"]
        sig[s["batch"]] = {"source_dir": str(s["accepted"]), "files": files,
                           "report": ({"path": str(rep), "sha1": sha1_file(rep), "mtime": mtime_iso(rep)}
                                      if rep.is_file() else None)}
    return sig


def package_corpus_family(fam: str, sources: list, dest: Path, force: bool, log) -> dict:
    """Writes <dest>/accepted/*.jsonl (+ report.json | report_<batch>.json,
    sources.json) and returns {"files": {rel: {sha1, bytes, rows}}, "summary":
    {...}, "reused": bool}."""
    sig = corpus_source_signature(sources)
    sj = dest / "sources.json"
    if sj.is_file():
        old = read_json(sj)
        if {b: {n: f["sha1"] for n, f in v["files"].items()} for b, v in old.get("batches", {}).items()} == \
                {b: {n: f["sha1"] for n, f in v["files"].items()} for b, v in sig.items()}:
            log("  corpus %-11s already packaged from the same sources; reused" % fam)
            return {"files": file_table(dest, dest), "summary": old.get("summary"), "reused": True}
        if not force:
            raise SystemExit("corpus %s at %s was packaged from different sources; --force to overwrite"
                             % (fam, dest))
        shutil.rmtree(dest)
    (dest / "accepted").mkdir(parents=True, exist_ok=True)
    multi = len(sources) > 1
    rows = {}
    for name in CORPUS_FILES:
        present = [s for s in sources if (s["accepted"] / name).is_file()]
        if not present:
            log("  corpus %-11s %s missing in every batch; skipped" % (fam, name))
            continue
        if len(present) != len(sources):
            log("  corpus %-11s %s present in %d of %d batches" % (fam, name, len(present), len(sources)))
        tgt = dest / "accepted" / name
        if not multi:
            shutil.copyfile(present[0]["accepted"] / name, tgt)
        else:
            with open(tgt, "wb") as out:
                for s in present:
                    with open(s["accepted"] / name, "rb") as fh:
                        data = fh.read()
                    out.write(data)
                    if data and not data.endswith(b"\n"):
                        out.write(b"\n")
        rows[name] = count_jsonl(tgt)
    for s in sources:
        if s["report"].is_file():
            shutil.copyfile(s["report"], dest / ("report.json" if not multi else "report_%s.json" % s["batch"]))
    pids = []
    with open(dest / "accepted" / "problems.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                pids.append(json.loads(line)["problem_id"])
    if len(set(pids)) != len(pids):
        raise SystemExit("corpus %s: duplicate problem ids after concatenation" % fam)
    idx = [i for i in (problem_index(p) for p in pids) if i is not None]
    summary = {"n_problems": rows.get("problems.jsonl"), "n_trajectories": rows.get("trajectories.jsonl"),
               "n_pairs": rows.get("pairs.jsonl"), "index_min": min(idx) if idx else None,
               "index_max": max(idx) if idx else None, "batches": [s["batch"] for s in sources],
               "problem_ids": pids}
    write_json(sj, {"family": fam, "written": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "layout": "accepted/{problems,trajectories,pairs}.jsonl = the batches concatenated in "
                              "this order (single batch: byte-for-byte copies)",
                    "batches": sig, "summary": {k: v for k, v in summary.items() if k != "problem_ids"}})
    return {"files": file_table(dest, dest), "summary": summary, "reused": False}


def file_table(root: Path, base: Path, exclude: Tuple[str, ...] = ()) -> Dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(base).as_posix()
            if rel in exclude:
                continue
            rec = {"sha1": sha1_file(p), "bytes": p.stat().st_size}
            if p.suffix == ".jsonl":
                rec["rows"] = count_jsonl(p)
            out[rel] = rec
    return out


# ---------------------------------------------------------------------------
# evaluation-only problems
# ---------------------------------------------------------------------------
def load_eval_ids(a, families: List[str], corpus_summaries: dict) -> Tuple[dict, list]:
    """family -> {"ids": [...], "rule": str, "source": str}; warnings."""
    out, warns = {}, []
    by_fam: Dict[str, list] = collections.defaultdict(list)
    if a.eval_problems_native:
        p = Path(a.eval_problems_native)
        for _, r in read_jsonl_lines(p):
            v = r.get("variant") or (r.get("spec") or {}).get("domain_variant")
            pid = (r.get("spec") or {}).get("problem_id") or r.get("problem_id")
            by_fam[v].append(pid)
        meta = p.with_name(p.name + ".meta.json")
        rule = (read_json(meta).get("rule") if meta.is_file() else
                "per variant, the last %d problems by index of the accepted corpus" % N_EVAL_NATIVE)
        for fam in families:
            if fam in NATIVE_FAMILIES and by_fam.get(fam):
                out[fam] = {"ids": sorted(by_fam[fam]), "rule": rule, "source": str(p)}
    if a.eval_problems_v2:
        p = Path(a.eval_problems_v2)
        for _, r in read_jsonl_lines(p):
            v = r.get("variant") or (r.get("spec") or {}).get("domain_variant")
            pid = (r.get("spec") or {}).get("problem_id") or r.get("problem_id")
            by_fam[v].append(pid)
        for fam in families:
            if fam in V2_FAMILIES and by_fam.get(fam):
                out[fam] = {"ids": sorted(by_fam[fam]),
                            "rule": "the %d problems of problems_v2.jsonl (a separate evaluation set, ids "
                                    "<family>_e1....; not part of the training corpus)" % len(by_fam[fam]),
                            "source": str(p)}
    for fam in families:
        if fam in out:
            continue
        if fam in NATIVE_FAMILIES:
            s = corpus_summaries.get(fam)
            if s and s.get("problem_ids") and len(s["problem_ids"]) > N_EVAL_NATIVE:
                ids = sorted(s["problem_ids"], key=lambda p: (problem_index(p) or -1, p))[-N_EVAL_NATIVE:]
                out[fam] = {"ids": ids, "rule": "the last %d problems by index of the packaged corpus "
                                                "(problems_v3_native.jsonl not given; rule applied here)"
                                                % N_EVAL_NATIVE, "source": "rule"}
            else:
                warns.append("%s: evaluation-only problems unknown (no --eval-problems-native and no corpus "
                             "with > %d problems); exclusion not checked" % (fam, N_EVAL_NATIVE))
                out[fam] = {"ids": [], "rule": "unknown", "source": None}
        else:
            warns.append("%s: evaluation problems unknown (no --eval-problems-v2); exclusion not checked" % fam)
            out[fam] = {"ids": [], "rule": "unknown (problems_v2.jsonl not given)", "source": None}
    return out, warns


# ---------------------------------------------------------------------------
# eval episodes (multitool only)
# ---------------------------------------------------------------------------
def package_eval(a, out: Path, spelling: str, families: List[str], mt_sha: str, prov: dict, log):
    ev = out / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    if spelling != "multitool":
        if a.eval_episodes:
            raise SystemExit("--eval-episodes: the r1 base reference was served multitool prompts; it "
                             "belongs under datav4_multitool only")
        write_text(ev / "NOTE.md",
                   "# eval/\n\nNo base reference episodes under the set_parameter spelling yet. The "
                   "base_nothink r1 reference (Orchard job 160740) was served multitool prompts and is "
                   "packaged under datav4_multitool/eval/. A set_parameter base run would need the harness "
                   "started with DATAXP_SPELLING=set_parameter.\n")
        return {"episodes": {}, "note": "none under set_parameter"}
    if not a.eval_episodes:
        write_text(ev / "NOTE.md", "# eval/\n\nNo episodes packaged (--eval-episodes not given).\n")
        return {"episodes": {}, "note": "not packaged"}
    src = Path(a.eval_episodes)
    files = sorted(src.glob("*.episodes.jsonl"))
    if not files:
        raise SystemExit("no *.episodes.jsonl under %s" % src)
    table, errors = {}, []
    for f in files:
        rows = [o for _, o in read_jsonl_lines(f)]
        variants = collections.Counter(o.get("variant") for o in rows)
        shas = collections.Counter(o.get("instruction_sha1") for o in rows)
        bad = [s for s in shas if s != mt_sha]
        if bad:
            errors.append("%s: episodes carry instruction_sha1 %s, not the multitool %s" % (f.name, bad, mt_sha))
        unknown = [v for v in variants if v not in families]
        if unknown:
            errors.append("%s: variants %s are not packaged families" % (f.name, unknown))
        shutil.copyfile(f, ev / f.name)
        n = len(rows)
        table[f.name] = {
            "variant": ", ".join(sorted(v for v in variants if v)), "episodes": n,
            "policy": ", ".join(sorted({str(o.get("policy")) for o in rows})),
            "feasible_at_12": (sum(bool(o.get("feasible@12")) for o in rows) / n) if n else None,
            "feasible_at_20": (sum(bool(o.get("feasible@20")) for o in rows) / n) if n else None,
            "problem_ids": sorted({o.get("problem_id") for o in rows}),
            "sha1": sha1_file(f), "source": str(f), "mtime": mtime_iso(f),
        }
        log("  eval %-45s %3d episodes feasible@12 %.3f" % (f.name, n, table[f.name]["feasible_at_12"] or 0))
    if errors:
        raise SystemExit("eval episodes refused:\n  " + "\n  ".join(errors))
    extra = []
    for name in ("summary_local.txt",):
        if (src / name).is_file():
            shutil.copyfile(src / name, ev / name)
            extra.append(name)
    lines = ["# eval/ — base reference on the native evaluation problems (multitool prompts)", "",
             prov.get("eval", {}).get("r1", ""), "",
             "These episodes are the un-tuned reference the slope test compares against; they were served "
             "the multitool instruction (sha1 `%s`) and the verb grammars packaged in this directory. They "
             "ran on the evaluation-only problems (the last 60 by index of each native family), never on "
             "the own-row training problems." % mt_sha, "",
             "| file | variant | policy | episodes | feasible@12 | feasible@20 | sha1 |",
             "|---|---|---|---|---|---|---|"]
    for name, t in table.items():
        lines.append("| %s | %s | %s | %d | %.3f | %.3f | %s |" % (
            name, t["variant"], t["policy"], t["episodes"], t["feasible_at_12"] or 0,
            t["feasible_at_20"] or 0, t["sha1"][:12]))
    if extra:
        lines += ["", "Also copied: " + ", ".join(extra) + "."]
    lines += ["", "Source: `%s` (mtime %s)." % (src, mtime_iso(files[0])), ""]
    write_text(ev / "NOTE.md", "\n".join(lines))
    return {"episodes": table, "extra": extra, "source": str(src)}


# ---------------------------------------------------------------------------
# README / CORPUS.md
# ---------------------------------------------------------------------------
def md_code(text: str) -> str:
    return "```\n" + text.rstrip("\n") + "\n```"


def write_readme(out: Path, spelling: str, manifest: dict, fam_info: dict, worker: dict, corpus_mode: str,
                 corpus_dir: Path, corpus_refs: dict, eval_ids: dict, prov: dict, rows_root: Optional[Path],
                 warnings: List[str]):
    ds = "datav4_" + spelling
    other = "datav4_" + ("set_parameter" if spelling == "multitool" else "multitool")
    ins = manifest["instruction"]
    L = ["# %s — own-action rows and corpora, `%s` spelling" % (ds, spelling), "",
         "Built %s by `dataxp/package_datav4.py` (repo HEAD %s). Destination on bridges2: "
         "`%s/%s/`. Its twin is `%s/` (same states, same teacher, same rule, the other action syntax)."
         % (manifest["built"], (manifest.get("repo_head") or "n/a")[:12], BRIDGES2_ROOT, ds, other), "",
         "**Every row in `rows/` carries `\"spelling\": \"%s\"`**, and every file here was served or "
         "generated under `DATAXP_SPELLING=%s`. Validate with "
         "`python -m dataxp.package_datav4 --check --out <this dir>`." % (spelling, spelling), "",
         "## What the `%s` spelling is" % spelling, "", SPELLING_BLURB[spelling], "", COMMON_BLURB, "",
         "### Action lines, verbatim, per family", "",
         "From the teacher's trajectory actions formatted under this spelling (`format_move`) and, where "
         "rows exist, from the packaged rows' assistant lines.", "",
         "| family | kind | from the corpus (teacher moves) | from the rows (the model's accepted picks) |",
         "|---|---|---|---|"]
    for fam in manifest["family_order"]:
        fi = fam_info[fam]
        ex_c = fi.get("examples_corpus") or []
        ex_r = fi.get("examples_rows") or []
        fmt = lambda xs: "<br>".join("`%s`" % x.replace("|", "\\|") for x in xs) if xs else "—"
        L.append("| %s | %s | %s | %s |" % (fam, fi["kind"], fmt(ex_c), fmt(ex_r)))
    L += ["", "## The served instruction", "",
          "`instruction.txt` is the instruction block appended to every served user message (the text of "
          "`dataxp.ownrows.common.%s`); its sha1[:12] `%s` (full `%s`) is the `instruction_sha1` every row "
          "carries. The full served tail of a user message is `instruction.txt` + "
          "`\"The operations in this problem are exactly:\\n\"` + `grammar/<family>.txt` (right-stripped) + "
          "`\"\\n\"`; the build verified this tail on every packaged row (counts in `manifest.json`, "
          "`tail_ok`)." % (ins["name"], ins["sha1_12"], ins["sha1"]), "",
          md_code(ins["text"]), ""]
    if ins.get("also_accepted"):
        L += ["Also accepted in rows (older serving of the same spelling): " + "; ".join(
            "`%s` = %s" % (s, d["name"]) for s, d in ins["also_accepted"].items()), ""]
    L += ["### Grammars", "", "| family | kind | file | sha1 | lines |", "|---|---|---|---|---|"]
    for fam in manifest["family_order"]:
        g = manifest["families"][fam]["grammar"]
        L.append("| %s | %s | `grammar/%s.txt` | %s | %d |" % (fam, fam_info[fam]["kind"], fam, g["sha1"][:12],
                                                               g["lines"]))
    L += ["", "## Rows: how a state became a row", "", "**Acceptance rule (own-action rows).** " + ACCEPTANCE_RULE,
          "", "**State selection.** " + STATE_RULE, "",
          "**Row format.** `messages` = [system, user, assistant]; the assistant turn is exactly one "
          "`<action>...</action>` line with `reasoning_content` = the model's written reasoning (`own`) or "
          "`\"\"` (`own_act`); plus `domain`, `instance`, `turn`, `binding_constraint`, `n_prior_changes`, "
          "`instruction_sha1`, `spelling`, `action` (the serialised move; `{i, p, v, v_old}` for a v2 "
          "family, `{verb, args}` for a native one — identical under both spellings), `clamped`, `tier`, "
          "`teacher`, `rationale_sample`, `sample_mode`, `sample_prompt`, `own_pct`, `own_same_move`, "
          "`own_phi`, `gold_action`, `gold_phi`, `gold_pct`, `n_candidates`. `own` and `own_act` of a split "
          "are matched row for row.", "",
          "## Row counts", "",
          "| family | group | states train / dev | rows train (own = own_act) | rows dev | mean pct | "
          "same-move share | train problems | dev problems | stamped | tail ok | parse ok |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot_tr = tot_dev = 0
    for fam in manifest["family_order"]:
        fm = manifest["families"][fam]
        r = fm.get("rows")
        if not r:
            L.append("| %s | — | — | 0 | 0 | — | — | — | — | — | — | — |" % fam)
            continue
        mo = fm["manifest_own"]
        tr, dv = mo["train"], mo["dev"]
        tot_tr += r["train_own"]
        tot_dev += r["dev_in_own"]
        pc = fm.get("parse_check") or {}
        L.append("| %s | %s | %d / %d | %d | %d | %s | %s | %d | %d | %d | %d / %d | %s |" % (
            fam, fm["group"], tr["states"], dv["states"], r["train_own"], r["dev_in_own"],
            ("%.1f" % tr["mean_pct"]) if tr["mean_pct"] is not None else "—",
            ("%.2f" % tr["same_move_share"]) if tr["same_move_share"] is not None else "—",
            len(fm["problems"]["train"]), len(fm["problems"]["dev"]), fm["checks"]["stamped"],
            fm["checks"]["tail_ok"], fm["checks"]["tail_checked"],
            ("%d / %d" % (pc["ok"], pc["n"])) if pc else "not run"))
    L += ["| **total** | | | **%d** | **%d** | | | | | | | |" % (tot_tr, tot_dev), "",
          "states = repair-phase states the sampler saw for that split (from the group's manifest_own.json); "
          "rows = states kept by the rule; stamped = rows that lacked the `spelling` key in the source tier "
          "and were stamped after their instruction sha1 and action line were checked; tail ok = rows whose "
          "served user text ends with instruction.txt + the packaged grammar; parse ok = action lines that "
          "re-parse under this spelling to the stored `action` dict.", "",
          "Families without rows here (held out or not yet sampled): " + (", ".join(
              fam for fam in manifest["family_order"] if not manifest["families"][fam].get("rows")) or "none")
          + ". Their grammar and instruction are packaged so the evaluation harness can serve them.", "",
          "## Evaluation-only problems (never among the rows' instances)", "",
          "| family | rule | n | source | ids |", "|---|---|---|---|---|"]
    for fam in manifest["family_order"]:
        e = eval_ids[fam]
        ids = e["ids"]
        show = ("%s … %s" % (ids[0], ids[-1])) if len(ids) > 4 else ", ".join(ids) if ids else "—"
        L.append("| %s | %s | %d | %s | %s |" % (fam, e["rule"], len(ids), e.get("source") or "—", show))
    L += ["", "The full id lists are in `manifest.json` (`families.<family>.eval_only_problems`).", "",
          "## Corpora", "",
          ("The spelling-independent corpora (`accepted/{problems,trajectories,pairs}.jsonl` + `report.json` "
           "per family) are stored ONCE in `%s/` (bridges2: `%s/datav4_corpus/`) and named in `CORPUS.md` "
           "with the sha1 of every corpus file this dataset was built against; `--check` verifies them. They "
           "are stored once because generation never sees the spelling (the corpora of the two datasets are "
           "byte-identical) and because they are large relative to the rows; the rows are complete training "
           "inputs on their own (full served prompts), so this directory is self-sufficient for training."
           % (corpus_dir.name, BRIDGES2_ROOT)) if corpus_mode == "shared" else
          ("The spelling-independent corpora are copied into `corpus/<family>/` here (`--corpus-mode copy`); "
           "the twin dataset carries an identical copy."), "",
          "| family | batches packaged | problems (index range) | trajectories | pairs | referenced only |",
          "|---|---|---|---|---|---|"]
    for fam in manifest["family_order"]:
        c = manifest["corpus"]["families"].get(fam)
        refs = corpus_refs.get(fam) or []
        if c:
            s = c["summary"]
            L.append("| %s | %s | %s (%s–%s) | %s | %s | %s |" % (
                fam, ", ".join(s.get("batches") or []), s.get("n_problems"), s.get("index_min"),
                s.get("index_max"), s.get("n_trajectories"), s.get("n_pairs"),
                "; ".join("%s: `%s`" % (r["batch"], r["path"]) for r in refs) or "—"))
        else:
            L.append("| %s | — | — | — | — | %s |" % (fam, "; ".join(
                "%s: `%s`" % (r["batch"], r["path"]) for r in refs) or "—"))
    L += ["", "## Sources, jobs and dates", "", "| what | where | notes |", "|---|---|---|",
          "| rows tree | `%s` | groups: %s |" % (rows_root or "—", ", ".join(
              "%s (%s)" % (g, ", ".join(sorted(f for f, gg in manifest["family_group"].items() if gg == g)))
              for g in manifest["groups"]) or "—")]
    for g, rec in manifest["groups"].items():
        for t, sf in rec["source_files"].items():
            L.append("| %s/%s | `%s` | sha1 %s, %d rows, mtime %s |" % (g, t, sf["path"], sf["sha1"][:12],
                                                                        sf["rows"], sf["mtime"]))
        L.append("| %s job | %s | — |" % (g, prov.get("rows", {}).get(spelling, {}).get(
            g, "job id not recorded in the docs; pass --provenance")))
    for fam, c in manifest["corpus"]["families"].items():
        for b, s in (c.get("sources") or {}).items():
            L.append("| corpus %s / %s | `%s` | %s; mtime %s |" % (
                fam, b, s["source_dir"], "; ".join("%s %s" % (n, f["sha1"][:12]) for n, f in s["files"].items()),
                next(iter(s["files"].values()))["mtime"] if s["files"] else "—"))
        for b in (c.get("summary") or {}).get("batches") or []:
            note = prov.get("corpus", {}).get(fam if b == "main" else b)
            if note:
                L.append("| corpus %s / %s job | %s | — |" % (fam, b, note))
    if manifest.get("eval", {}).get("source"):
        L.append("| eval episodes | `%s` | %s |" % (manifest["eval"]["source"], prov.get("eval", {}).get("r1", "")))
    L += ["", "Provenance notes come from " + prov.get("_source", "the docs") + ".", "",
          "## Layout", "", md_code(LAYOUT_TEXT % {"ds": ds, "corpus": corpus_dir.name if corpus_mode == "shared"
                                                 else "corpus/ (copied here)"}), ""]
    if warnings:
        L += ["## Build warnings", ""] + ["- " + w for w in warnings] + [""]
    L += ["## Checks the build ran", "",
          "- every row: `domain` = its family; `spelling` = `%s`; `instruction_sha1` in the accepted set; "
          "the assistant turn is exactly one `<action>...</action>` line written in this spelling; "
          "`reasoning_content` non-empty in `own`, empty in `own_act`; the served user text ends with "
          "instruction.txt + the packaged grammar; the instance is not an evaluation-only problem;" % spelling,
          "- per family: `own` and `own_act` match row for row; train and dev problems are disjoint; every "
          "action line re-parses under `DATAXP_SPELLING=%s` to the stored `action` (`Move.matches`, rel "
          "1e-5);" % spelling,
          "- grammars and the instruction come from the domains instantiated in a subprocess with "
          "`DATAXP_SPELLING=%s`; code sha1s in `manifest.json` (`code`)." % spelling, ""]
    write_text(out / "README.md", "\n".join(L))


LAYOUT_TEXT = """%(ds)s/
  README.md, manifest.json, instruction.txt, CORPUS.md
  grammar/<family>.txt
  rows/<family>/{train_own,train_own_act,dev_in_own,dev_in_own_act}.jsonl, manifest_own.json
  rows/_groups/<group>/{manifest_own.json,select_report.json}   (the tier group's files, verbatim)
  eval/   (multitool: base_nothink r1 episodes + NOTE.md; set_parameter: NOTE.md only)
%(corpus)s/<family>/accepted/{problems,trajectories,pairs}.jsonl, report[_<batch>].json, sources.json"""


def write_corpus_md(out: Path, spelling: str, corpus_mode: str, corpus_dir: Path, corpus_families: dict,
                    corpus_refs: dict):
    ds = "datav4_" + spelling
    L = ["# Corpora used by %s" % ds, ""]
    if corpus_mode == "shared":
        L += ["The spelling-independent corpora live ONCE in `%s/` next to this directory (bridges2: "
              "`%s/datav4_corpus/`), not here: generation is spelling-blind, so `datav4_multitool` and "
              "`datav4_set_parameter` would carry byte-identical copies. The rows in `rows/` are complete "
              "training inputs (full served prompts); the corpora are what the states were drawn from and "
              "what `dataxp.ownrows.select_states --data <corpus dir>` replays." % (corpus_dir.name, BRIDGES2_ROOT),
              "", "This dataset was built against these corpus files (verified by `--check`):", ""]
    else:
        L += ["The spelling-independent corpora are copied into `corpus/` here (`--corpus-mode copy`).", ""]
    L += ["| family | file | sha1 | bytes | rows |", "|---|---|---|---|---|"]
    for fam, c in corpus_families.items():
        for rel, f in c["files"].items():
            L.append("| %s | `%s/%s` | %s | %d | %s |" % (fam, fam, rel, f["sha1"], f["bytes"], f.get("rows", "")))
    if corpus_refs:
        L += ["", "Referenced but not packaged here (lives elsewhere):", ""]
        for fam, refs in corpus_refs.items():
            for r in refs:
                L.append("- %s, batch %s: `%s`" % (fam, r["batch"], r["path"]))
    write_text(out / "CORPUS.md", "\n".join(L) + "\n")


def write_corpus_readme(corpus_dir: Path, manifest: dict):
    L = ["# datav4_corpus — the spelling-independent corpora of datav4", "",
         "Written by `dataxp/package_datav4.py`, last %s. Referenced by: %s." % (
             manifest["written"], ", ".join(sorted(manifest.get("referenced_by") or [])) or "—"), "",
         "One directory per family: `accepted/{problems,trajectories,pairs}.jsonl` (the `run_generation` "
         "layout, so `dataxp.ownrows.select_states --data <this dir> --variants <family>` reads it), the "
         "generation `report.json` (`report_<batch>.json` when several batches were concatenated) and "
         "`sources.json` (every source file's path, sha1, size and mtime, and the batch order). The stored "
         "actions are spelling-blind (`{i, p, v, v_old}` for a v2 family, `{verb, args}` for a native one); "
         "`datav4_multitool` and `datav4_set_parameter` both name this directory in their CORPUS.md with "
         "the sha1s below.", "",
         "| family | batches | problems (index range) | trajectories | pairs | files |", "|---|---|---|---|---|---|"]
    for fam, c in manifest["families"].items():
        s = c["summary"]
        L.append("| %s | %s | %s (%s–%s) | %s | %s | %s |" % (
            fam, ", ".join(s.get("batches") or []), s.get("n_problems"), s.get("index_min"), s.get("index_max"),
            s.get("n_trajectories"), s.get("n_pairs"),
            "; ".join("%s %s" % (rel, f["sha1"][:12]) for rel, f in c["files"].items())))
    write_text(corpus_dir / "README.md", "\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
OUT_ENTRIES = ("README.md", "manifest.json", "instruction.txt", "CORPUS.md", "grammar", "rows", "eval", "corpus")


def clear_out(out: Path) -> None:
    """Remove what a build writes under --out (only those names)."""
    for name in OUT_ENTRIES:
        p = out / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def prepare_out(out: Path, force: bool):
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit("%s exists and is not empty; --force to rebuild it" % out)
        clear_out(out)
    out.mkdir(parents=True, exist_ok=True)


def build(a) -> int:
    if not a.spelling or not a.rows_root:
        raise SystemExit("--spelling and --rows-root are required to build (or use --check)")
    out = Path(a.out).resolve()
    prepare_out(out, a.force)
    try:
        return _build(a, out)
    except BaseException as e:  # a refused build leaves nothing half-written behind
        if not (isinstance(e, SystemExit) and e.code in (0, None)):
            clear_out(out)
            print("partial output removed from %s" % out, flush=True)
        raise


def _build(a, out: Path) -> int:
    log = lambda s: print(s, flush=True)
    spelling = a.spelling
    ds = "datav4_" + spelling
    if out.name != ds:
        log("note: --out basename is %r; the dataset is %s (bridges2 path %s/%s)" % (out.name, ds, BRIDGES2_ROOT, ds))
    corpus_mode = a.corpus_mode
    corpus_dir = (Path(a.corpus_dir).resolve() if a.corpus_dir else out.parent / "datav4_corpus") \
        if corpus_mode == "shared" else out / "corpus"
    prov = KNOWN_PROVENANCE
    if a.provenance:
        prov = deep_merge(prov, read_json(Path(a.provenance)))
    warnings: List[str] = []

    # 1. rows ---------------------------------------------------------------
    rows_root = Path(a.rows_root).resolve()
    groups = discover_groups(rows_root, a.groups)
    fam_rows, fam_group = split_by_family(groups)
    families = list(a.families or FAMILIES)
    for fam in fam_rows:
        if fam not in families:
            families.append(fam)
    log("rows: groups %s; families with rows: %s" % (", ".join(groups), ", ".join(sorted(fam_rows))))

    # 2. corpus sources -----------------------------------------------------
    corpus_src, corpus_refs, w = resolve_corpus_sources(families, a)
    warnings += w
    total_bytes = sum((s["accepted"] / n).stat().st_size for srcs in corpus_src.values() for s in srcs
                      for n in CORPUS_FILES if (s["accepted"] / n).is_file())
    log("corpus sources: %s; total %s (mode %s -> %s)" % (
        ", ".join("%s[%s]" % (f, ",".join(s["batch"] for s in ss)) for f, ss in corpus_src.items()) or "none",
        human_mb(total_bytes), corpus_mode, corpus_dir))

    # 3. worker: grammars, instruction, examples, parse check ---------------
    job = {"spelling": spelling, "families": families, "examples": {}, "parse": {}}
    for fam, srcs in corpus_src.items():
        seen, ex = set(), []
        tp = srcs[0]["accepted"] / "trajectories.jsonl"
        for _, tr in read_jsonl_lines(tp)[:60]:
            for act in tr.get("actions") or []:
                key = act.get("verb") if "verb" in act else act.get("p")
                if key in seen:
                    continue
                seen.add(key)
                ex.append(act)
                if len(ex) >= 6:
                    break
            if len(ex) >= 6:
                break
        job["examples"][fam] = ex
    if not a.no_parse_check:
        for fam, tiers in fam_rows.items():
            items = []
            for t in ("train_own_act", "dev_in_own_act"):
                for _, o in tiers[t]:
                    m = ACTION_RE.match(((o.get("messages") or [{}])[-1].get("content") or "").strip())
                    if m and "action" in o:
                        items.append([m.group(1).strip(), o["action"]])
            job["parse"][fam] = items
    log("worker: DATAXP_SPELLING=%s, %d families, parse check on %d action lines" % (
        spelling, len(families), sum(len(v) for v in job["parse"].values())))
    res = run_worker(spelling, job)
    instr_all = res["instructions"]
    current = {sha: d for sha, d in instr_all.items() if d["spelling"] == spelling and d["current"]}
    if len(current) != 1:
        raise SystemExit("expected exactly one current instruction for %s, got %s" % (spelling, list(current)))
    instr_sha, instr = next(iter(current.items()))
    accepted = {instr_sha: instr["text"]}
    also = {}
    for sha in a.accept_instruction_sha1 or []:
        d = instr_all.get(sha)
        if d and d["spelling"] != spelling:
            raise SystemExit("--accept-instruction-sha1 %s is the %s instruction; cannot accept it under %s"
                             % (sha, d["spelling"], spelling))
        accepted[sha] = d["text"] if d else None
        also[sha] = {"name": d["name"] if d else "unknown text (passed on the command line)",
                     "text": d["text"] if d else None}
        if not d:
            warnings.append("--accept-instruction-sha1 %s: text unknown; served tails of such rows not verified" % sha)
    write_text(out / "instruction.txt", instr["text"])
    for fam in families:
        write_text(out / "grammar" / (fam + ".txt"), res["families"][fam]["grammar"])
        if res["families"][fam]["instruction_sha1"] != instr_sha:
            raise SystemExit("%s: instruction_for gave sha %s, expected %s" % (
                fam, res["families"][fam]["instruction_sha1"], instr_sha))
    mt_sha = next(sha for sha, d in instr_all.items() if d["spelling"] == "multitool" and d["current"])

    # 4. corpora --------------------------------------------------------------
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_families = {}
    for fam in families:
        if fam in corpus_src:
            r = package_corpus_family(fam, corpus_src[fam], corpus_dir / fam, a.force, log)
            sig = corpus_source_signature(corpus_src[fam])
            corpus_families[fam] = {"files": r["files"], "summary": r["summary"], "sources": sig,
                                    "dir": (corpus_dir / fam).as_posix()}
            if not r["reused"]:
                log("  corpus %-11s %s problems, %s trajectories, %s pairs (%s)" % (
                    fam, r["summary"]["n_problems"], r["summary"]["n_trajectories"], r["summary"]["n_pairs"],
                    human_mb(sum(f["bytes"] for f in r["files"].values()))))
    if corpus_mode == "shared":
        cm_path = corpus_dir / "manifest.json"
        cm = read_json(cm_path) if cm_path.is_file() else {"families": {}, "referenced_by": []}
        for fam, c in corpus_families.items():
            cm["families"][fam] = {"files": c["files"], "summary": {k: v for k, v in c["summary"].items()
                                                                     if k != "problem_ids"},
                                   "sources": c["sources"]}
        cm["referenced_by"] = sorted(set(cm.get("referenced_by") or []) | {ds})
        cm["written"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cm["layout"] = "<family>/accepted/{problems,trajectories,pairs}.jsonl + report[_<batch>].json + sources.json"
        write_json(cm_path, cm)
        write_corpus_readme(corpus_dir, cm)

    # 5. evaluation-only problems ------------------------------------------
    eval_ids, w = load_eval_ids(a, families, {f: c["summary"] for f, c in corpus_families.items()})
    warnings += w

    # 6. validate and write rows ----------------------------------------------
    fam_info, man_fams, failures = {}, {}, []
    for g, rec in groups.items():
        gd = out / "rows" / "_groups" / g
        gd.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rec["tiers_dir"] / "manifest_own.json", gd / "manifest_own.json")
        if rec["select_report_path"]:
            shutil.copyfile(rec["select_report_path"], gd / "select_report.json")
    for fam in families:
        wf = res["families"][fam]
        info = {"kind": wf["kind"], "examples_corpus": wf["examples"], "examples_rows": []}
        fam_info[fam] = info
        fm = {"spelling": spelling, "kind": wf["kind"],
              "grammar": {"file": "grammar/%s.txt" % fam, "sha1": sha1_text(wf["grammar"]),
                          "lines": len(wf["grammar"].rstrip("\n").splitlines())},
              "instruction_sha1": instr_sha, "eval_only_problems": eval_ids[fam]["ids"],
              "eval_rule": eval_ids[fam]["rule"]}
        man_fams[fam] = fm
        if fam not in fam_rows:
            continue
        tiers = fam_rows[fam]
        rep, errs, warns, out_lines = validate_family_rows(fam, spelling, tiers, accepted, wf["grammar"],
                                                            set(eval_ids[fam]["ids"]), a.stamp_spelling)
        warnings += warns
        if errs:
            failures += errs
            continue
        if rep["served_grammar_variants"]:
            if not a.allow_grammar_drift:
                failures.append("%s: served grammar differs from the packaged grammar in %d rows (write the "
                                "variants and continue with --allow-grammar-drift)" % (
                                    fam, sum(v["rows"] for v in rep["served_grammar_variants"])))
                continue
            for k, v in enumerate(rep["served_grammar_variants"], 1):
                write_text(out / "grammar" / ("%s.served-%d.txt" % (fam, k)), v["text"])
        pc = wf["parse"]
        if pc["n"] and pc["ok"] != pc["n"]:
            failures.append("%s: %d of %d action lines do not re-parse to the stored action under %s; first: %s"
                            % (fam, pc["n"] - pc["ok"], pc["n"], spelling, pc["failures"][:3]))
            continue
        rd = out / "rows" / fam
        rd.mkdir(parents=True, exist_ok=True)
        for t in TIERS:
            write_text(rd / (t + ".jsonl"), "".join(l + "\n" for l in out_lines[t]))
        g = fam_group[fam]
        write_json(rd / "manifest_own.json", family_manifest_own(
            groups[g]["manifest"], g, "rows/_groups/%s/manifest_own.json" % g, groups[g]["manifest_sha1"],
            fam, tiers))
        seen, ex = set(), []
        for _, o in tiers["train_own_act"] + tiers["dev_in_own_act"]:
            m = ACTION_RE.match(((o.get("messages") or [{}])[-1].get("content") or "").strip())
            if not m:
                continue
            key = (o.get("action") or {}).get("verb") or (o.get("action") or {}).get("p")
            if key in seen:
                continue
            seen.add(key)
            ex.append(m.group(1).strip())
            if len(ex) >= 4:
                break
        info["examples_rows"] = ex
        sr = (groups[g].get("select_report") or {}).get("per_variant", {}).get(fam)
        fm.update({
            "group": g, "rows": rep["rows"], "problems": rep["problems"],
            "manifest_own": read_json(rd / "manifest_own.json"),
            "select_report": sr,
            "checks": {"stamped": rep["stamped"], "tail_checked": rep["tail_checked"], "tail_ok": rep["tail_ok"],
                       "tail_unverifiable": rep["tail_unverifiable"],
                       "served_grammar_variants": len(rep["served_grammar_variants"]),
                       "instruction_sha1_counts": rep["instruction_sha1"]},
            "parse_check": ({"n": pc["n"], "ok": pc["ok"]} if not a.no_parse_check else None),
            "job": prov.get("rows", {}).get(spelling, {}).get(g),
            "source_tiers": groups[g]["source_files"],
            "files": file_table(rd, out),
        })
        log("  rows %-11s train %4d dev %3d  stamped %d  tail %d/%d  parse %s  problems %d/%d" % (
            fam, rep["rows"]["train_own"], rep["rows"]["dev_in_own"], rep["stamped"], rep["tail_ok"],
            rep["tail_checked"], ("%d/%d" % (pc["ok"], pc["n"])) if pc["n"] else "-",
            len(rep["problems"]["train"]), len(rep["problems"]["dev"])))
    if failures:
        raise SystemExit("REFUSED - fix the inputs and rebuild:\n  " + "\n  ".join(failures))

    # 7. eval ---------------------------------------------------------------
    ev = package_eval(a, out, spelling, families, mt_sha, prov, log)

    # 8. manifest, README, CORPUS.md ------------------------------------------
    manifest = {
        "dataset": ds, "spelling": spelling, "built": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "builder": "dataxp/package_datav4.py", "repo_head": git_head(), "code": code_sha1s(),
        "bridges2_destination": "%s/%s/" % (BRIDGES2_ROOT, ds),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if not k.startswith("_")},
        "instruction": {"file": "instruction.txt", "name": instr["name"], "sha1_12": instr_sha,
                        "sha1": sha1_text(instr["text"]), "text": instr["text"],
                        "also_accepted": also, "served_tail": "instruction.txt + %r + grammar/<family>.txt"
                                                              ".rstrip() + '\\n'" % GRAMMAR_LEAD},
        "family_order": families, "family_group": fam_group,
        "groups": {g: {"dir": str(rec["dir"]), "manifest_sha1": rec["manifest_sha1"],
                       "source_files": rec["source_files"],
                       "manifest_own": rec["manifest"], "select_report": rec["select_report"]}
                   for g, rec in groups.items()},
        "families": man_fams,
        "corpus": {"mode": corpus_mode, "dir": corpus_dir.as_posix(),
                   "dir_name": corpus_dir.name if corpus_mode == "shared" else "corpus",
                   "total_source_bytes": total_bytes, "families": corpus_families, "references": corpus_refs},
        "eval": ev, "warnings": warnings, "provenance": prov,
    }
    write_corpus_md(out, spelling, corpus_mode, corpus_dir, corpus_families, corpus_refs)
    write_readme(out, spelling, manifest, fam_info, res, corpus_mode, corpus_dir, corpus_refs, eval_ids, prov,
                 rows_root, warnings)
    manifest["files"] = file_table(out, out, exclude=("manifest.json",))
    write_json(out / "manifest.json", manifest)
    log("built %s: %d families, %d with rows, %d files, %s; corpus %s (%s)%s" % (
        out, len(families), sum(1 for f in man_fams.values() if f.get("rows")), len(manifest["files"]),
        human_mb(sum(f["bytes"] for f in manifest["files"].values())), corpus_mode, corpus_dir,
        ("\nwarnings:\n  " + "\n  ".join(warnings)) if warnings else ""))
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def check(a) -> int:
    out = Path(a.out).resolve()
    mp = out / "manifest.json"
    if not mp.is_file():
        raise SystemExit("%s: no manifest.json" % out)
    m = read_json(mp)
    spelling = m["spelling"]
    errors, warnings = [], []
    log = lambda s: print(s, flush=True)
    log("check %s (%s)" % (out, spelling))
    # files
    n_ok = 0
    for rel, f in m["files"].items():
        p = out / rel
        if not p.is_file():
            errors.append("missing file %s" % rel)
        elif sha1_file(p) != f["sha1"]:
            errors.append("sha1 mismatch %s" % rel)
        else:
            n_ok += 1
    extra = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
             and p.relative_to(out).as_posix() not in m["files"] and p.name != "manifest.json"
             and not p.relative_to(out).as_posix().startswith("batches/")]
    if extra:
        warnings.append("files not in the manifest: %s" % extra[:10])
    log("  files: %d of %d sha1 ok" % (n_ok, len(m["files"])))
    if (out / "batches").is_dir():
        log("  batches/: %s (appended batches; check each with --check --out <batch dir>)" % (
            ", ".join(sorted(d.name for d in (out / "batches").iterdir() if d.is_dir())) or "empty"))
    # instruction
    ins = m["instruction"]
    text = read_text(out / "instruction.txt")
    if sha1_text(text)[:12] != ins["sha1_12"] or text != ins["text"]:
        errors.append("instruction.txt differs from the manifest")
    accepted = {ins["sha1_12"]: text}
    for sha, d in (ins.get("also_accepted") or {}).items():
        accepted[sha] = d.get("text")
    # rows
    for fam in m["family_order"]:
        fm = m["families"][fam]
        gp = out / "grammar" / (fam + ".txt")
        if not gp.is_file():
            errors.append("%s: grammar file missing" % fam)
            continue
        grammar = read_text(gp)
        if sha1_text(grammar) != fm["grammar"]["sha1"]:
            errors.append("%s: grammar sha1 differs from the manifest" % fam)
        if not fm.get("rows"):
            continue
        rd = out / "rows" / fam
        tiers = {}
        for t in TIERS:
            p = rd / (t + ".jsonl")
            if not p.is_file():
                errors.append("%s: %s missing" % (fam, p.name))
                break
            tiers[t] = read_jsonl_lines(p)
        if len(tiers) != len(TIERS):
            continue
        rep, errs, warns, _ = validate_family_rows(fam, spelling, tiers, accepted, grammar,
                                                    set(fm.get("eval_only_problems") or []), stamp=False)
        errors += errs
        warnings += warns
        for t in TIERS:
            if rep["rows"][t] != fm["rows"][t]:
                errors.append("%s: %s has %d rows, manifest says %d" % (fam, t, rep["rows"][t], fm["rows"][t]))
        if rep["problems"] != fm["problems"]:
            errors.append("%s: train/dev problem lists differ from the manifest" % fam)
        if rep["served_grammar_variants"] and not fm["checks"].get("served_grammar_variants"):
            errors.append("%s: served grammar drift not recorded at build time" % fam)
        log("  rows %-11s train %4d dev %3d  tail %d/%d  %s" % (
            fam, rep["rows"]["train_own"], rep["rows"]["dev_in_own"], rep["tail_ok"], rep["tail_checked"],
            "ok" if not errs else "%d errors" % len(errs)))
    # corpus
    c = m["corpus"]
    cdir = Path(a.corpus_dir) if a.corpus_dir else (
        out / "corpus" if c["mode"] == "copy" else
        (out.parent / c.get("dir_name", "datav4_corpus") if (out.parent / c.get("dir_name", "datav4_corpus")).is_dir()
         else Path(c["dir"])))
    if not cdir.is_dir():
        warnings.append("corpus dir %s not found; corpus not checked (--corpus-dir <path>)" % cdir)
    else:
        for fam, cf in c["families"].items():
            for rel, f in cf["files"].items():
                p = cdir / fam / rel
                if not p.is_file():
                    errors.append("corpus %s/%s missing under %s" % (fam, rel, cdir))
                elif sha1_file(p) != f["sha1"]:
                    errors.append("corpus %s/%s sha1 mismatch" % (fam, rel))
        log("  corpus: %d families checked under %s" % (len(c["families"]), cdir))
    # eval
    for name, t in (m.get("eval", {}).get("episodes") or {}).items():
        p = out / "eval" / name
        if not p.is_file():
            errors.append("eval/%s missing" % name)
            continue
        rows = [o for _, o in read_jsonl_lines(p)]
        if len(rows) != t["episodes"]:
            errors.append("eval/%s: %d episodes, manifest says %d" % (name, len(rows), t["episodes"]))
        bad = {o.get("instruction_sha1") for o in rows} - {ins["sha1_12"]}
        if bad and spelling == "multitool":
            errors.append("eval/%s: instruction_sha1 %s" % (name, bad))
    # optional parse check
    if a.parse_check:
        job = {"spelling": spelling, "families": [f for f in m["family_order"] if m["families"][f].get("rows")],
               "examples": {}, "parse": {}}
        for fam in job["families"]:
            items = []
            for t in ("train_own_act", "dev_in_own_act"):
                for _, o in read_jsonl_lines(out / "rows" / fam / (t + ".jsonl")):
                    mm = ACTION_RE.match(((o.get("messages") or [{}])[-1].get("content") or "").strip())
                    if mm:
                        items.append([mm.group(1).strip(), o["action"]])
            job["parse"][fam] = items
        res = run_worker(spelling, job)
        for fam in job["families"]:
            pc = res["families"][fam]["parse"]
            if pc["ok"] != pc["n"]:
                errors.append("%s: %d of %d action lines do not re-parse; first %s" % (
                    fam, pc["n"] - pc["ok"], pc["n"], pc["failures"][:3]))
            if sha1_text(res["families"][fam]["grammar"]) != m["families"][fam]["grammar"]["sha1"]:
                errors.append("%s: the grammar the current code generates differs from the packaged one" % fam)
            log("  parse %-11s %d/%d ok; current grammar %s" % (
                fam, pc["ok"], pc["n"], "identical" if sha1_text(res["families"][fam]["grammar"]) ==
                m["families"][fam]["grammar"]["sha1"] else "DIFFERS"))
    for w in warnings:
        log("  warning: " + w)
    if errors:
        log("CHECK FAILED (%d):\n  " % len(errors) + "\n  ".join(errors[:60]))
        return 1
    log("CHECK OK: %s" % out)
    return 0


# ---------------------------------------------------------------------------
# append: a batch directory
# ---------------------------------------------------------------------------
def canon_spec(spec: dict) -> str:
    """The evaluation harness's canon() (dataxp_policy_eval_v3.fresh_problems):
    the spec without its id fields, sorted-key JSON."""
    return json.dumps({k: v for k, v in spec.items() if k not in CANON_DROP}, sort_keys=True, default=str)


def sid_of(fam: str, instance: str, turn) -> str:
    return "%s|%s|%d" % (fam, instance, int(turn))


def load_eval_sets(paths: List[str]) -> dict:
    """{"ids": {variant: set}, "canon": {canon: problem_id}, "files": [{path, sha1, n}]}."""
    out = {"ids": collections.defaultdict(set), "canon": {}, "files": []}
    for p in paths:
        p = Path(p)
        n = 0
        for _, r in read_jsonl_lines(p):
            spec = r["spec"] if isinstance(r.get("spec"), dict) else r
            v = r.get("variant") or spec.get("domain_variant") or str(spec.get("problem_id", "")).rsplit("_", 1)[0]
            pid = spec.get("problem_id") or r.get("problem_id")
            out["ids"][v].add(pid)
            out["canon"][canon_spec(spec)] = pid
            n += 1
        out["files"].append({"path": str(p), "sha1": sha1_file(p), "problems": n})
    out["ids"] = dict(out["ids"])
    return out


def resolve_prior(path: str) -> List[dict]:
    """A --prior argument -> the rows trees it stands for: [{"rows": Path,
    "root": Path | None, "label": str}]. A spelling root brings its rows/ and
    every batches/b*/rows/."""
    p = Path(path).resolve()
    trees = []
    if (p / "rows").is_dir():
        trees.append({"rows": p / "rows", "root": p, "label": p.name})
        for b in sorted((p / "batches").glob("b*")) if (p / "batches").is_dir() else []:
            if (b / "rows").is_dir():
                trees.append({"rows": b / "rows", "root": b, "label": "%s/batches/%s" % (p.name, b.name)})
    elif p.name == "rows" and p.is_dir():
        trees.append({"rows": p, "root": p.parent if (p.parent / "manifest.json").is_file() else None,
                      "label": "%s/rows" % p.parent.name})
    elif p.is_dir() and any((d / "train_own.jsonl").is_file() for d in p.iterdir() if d.is_dir()):
        trees.append({"rows": p, "root": None, "label": p.name})
    else:
        raise SystemExit("--prior %s: not a spelling directory, a batch directory or a rows/ tree" % path)
    return trees


def read_prior_trees(paths: List[str], spelling: str, log, skip: Optional[Path] = None) -> dict:
    """sids, (instance, turn) keys, problem splits, row counts and spelling of
    every prior rows tree; spelling roots (instruction.txt + grammar/) are
    returned for the consistency check. A tree at or under `skip` (the batch
    being built or checked) is left out."""
    res = {"sids": {}, "inst_turn": {}, "split": {}, "trees": [], "roots": [], "batches": set(),
           "skipped": []}
    for path in paths:
        for t in resolve_prior(path):
            if skip is not None and (t["rows"] == skip / "rows" or skip in t["rows"].parents):
                res["skipped"].append(t["label"])
                log("  prior %-40s skipped (the batch itself)" % t["label"])
                continue
            info = {"label": t["label"], "rows_dir": str(t["rows"]), "rows": 0, "families": {},
                    "spelling": collections.Counter(), "kind": None, "batch": None}
            root = t["root"]
            if root is not None and (root / "manifest.json").is_file():
                m = read_json(root / "manifest.json")
                info["kind"] = m.get("kind", "dataset")
                info["batch"] = m.get("batch") or ("b001" if info["kind"] == "dataset" else None)
                if m.get("spelling") and m["spelling"] != spelling:
                    raise SystemExit("REFUSED: --prior %s is the %s spelling; this batch is %s" % (
                        t["label"], m["spelling"], spelling))
                if info["kind"] == "dataset" and (root / "instruction.txt").is_file():
                    res["roots"].append(root)
            if info["batch"]:
                res["batches"].add(info["batch"])
            for fd in sorted(d for d in t["rows"].iterdir() if d.is_dir() and not d.name.startswith("_")):
                n_f = 0
                for tier in TIERS:
                    fp = fd / (tier + ".jsonl")
                    if not fp.is_file():
                        continue
                    split = "dev" if tier.startswith("dev") else "train"
                    for _, o in read_jsonl_lines(fp):
                        inst, turn = o.get("instance"), o.get("turn")
                        fam = o.get("domain") or fd.name
                        if inst is None or turn is None:
                            continue
                        res["sids"].setdefault(sid_of(fam, inst, turn), t["label"])
                        res["inst_turn"].setdefault((inst, int(turn)), t["label"])
                        prev = res["split"].setdefault(inst, (split, t["label"]))
                        if prev[0] != split:
                            raise SystemExit("REFUSED: prior trees disagree on the split of %s (%s in %s, %s in %s)"
                                             % (inst, prev[0], prev[1], split, t["label"]))
                        info["spelling"][str(o.get("spelling"))] += 1
                        n_f += 1
                info["families"][fd.name] = n_f
                info["rows"] += n_f
            bad = [s for s in info["spelling"] if s != spelling]
            if bad:
                raise SystemExit("REFUSED: --prior %s holds rows of spelling %s; this batch is %s"
                                 % (t["label"], bad, spelling))
            info["spelling"] = dict(info["spelling"])
            res["trees"].append(info)
            log("  prior %-40s %6d rows (all tiers) in %d families" % (t["label"], info["rows"],
                                                                       len(info["families"])))
    return res


def read_prior_state_splits(paths: List[str]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for p in paths or []:
        for _, o in read_jsonl_lines(Path(p)):
            pid = o.get("problem_id") or o.get("instance")
            sp = o.get("split")
            if pid and sp in ("train", "dev"):
                prev = out.setdefault(pid, (sp, p))
                if prev[0] != sp:
                    raise SystemExit("--prior-states disagree on the split of %s" % pid)
    return out


def load_group_inputs(rec: dict) -> None:
    """The sampling work files of a tier group (states, samples, scored),
    for specs, the funnel and the recipe checks. Adds keys to rec."""
    d = rec["dir"]
    sp = d / "states.jsonl" if (d / "states.jsonl").is_file() else d / "states_phi.jsonl"
    if not sp.is_file():
        raise SystemExit("group %s: no states.jsonl / states_phi.jsonl in %s (the specs and the funnel need it)"
                         % (rec["name"], d))
    states = [o for _, o in read_jsonl_lines(sp)]
    rec["states_path"] = sp
    rec["states"] = {o["sid"]: {"variant": o["variant"], "problem_id": o["problem_id"], "split": o.get("split", "train"),
                                "turn": o.get("turn"), "spelling": o.get("spelling"),
                                "canon": canon_spec(o["spec"]) if isinstance(o.get("spec"), dict) else None}
                     for o in states}
    rec["work_files"] = {"states": {"path": str(sp), "sha1": sha1_file(sp), "rows": len(states)}}
    samples = collections.defaultdict(list)
    smp = d / "samples.jsonl"
    if smp.is_file():
        for o in iter_jsonl(smp):
            slim = {k: o.get(k) for k in SAMPLE_KEYS}
            slim["thinking"] = (o.get("thinking") or "").strip()[:1]     # only whether there was any
            samples[(o["sid"], int(o["sample"]))].append(slim)
        rec["work_files"]["samples"] = {"path": str(smp), "sha1": sha1_file(smp),
                                        "rows": sum(len(v) for v in samples.values())}
    rec["samples"] = dict(samples)
    scored = collections.defaultdict(list)
    scp = d / "scored.jsonl"
    if scp.is_file():
        for o in iter_jsonl(scp):
            scored[o["sid"]].append({k: o.get(k) for k in ("sid", "sample", "ok", "pct", "noop")})
        rec["work_files"]["scored"] = {"path": str(scp), "sha1": sha1_file(scp),
                                       "rows": sum(len(v) for v in scored.values())}
    rec["scored"] = dict(scored)


SAMPLE_KEYS = ("sid", "sample", "answer", "finish", "error", "tokens", "prompt", "mode", "model", "draw_pass")


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as e:
                    raise SystemExit("%s line %d: not JSON (%s)" % (path, n, e))


def _complete(o: dict) -> bool:
    return bool(o.get("answer") or o.get("thinking")) and o.get("finish") != "length" and not o.get("error")


def _final_record(recs: List[dict]) -> dict:
    """merge_shards' rule: a complete record beats an incomplete one; among
    equals the last one wins."""
    best = None
    for o in recs:
        if best is None or _complete(o) or not _complete(best):
            best = o
    return best


def _reasoning_of(answer: str) -> str:
    a = answer or ""
    i = a.rfind("<action>")
    return (a if i < 0 else a[:i]).strip()


def family_funnel(fam: str, rec: dict, allow_noop: bool = False) -> dict:
    """states -> replies -> finished -> usable -> rows for one family of a group
    (rows filled in by the caller)."""
    st = {sid: s for sid, s in rec["states"].items() if s["variant"] == fam}
    f = {"states": {"train": sum(1 for s in st.values() if s["split"] == "train"),
                    "dev": sum(1 for s in st.values() if s["split"] == "dev")},
         "problems": len({s["problem_id"] for s in st.values()}),
         "reply_keys": 0, "replies_drawn": 0, "redrawn_keys": 0, "finished": 0, "truncated": 0, "errors": 0,
         "usable": 0, "states_with_usable": 0, "states_sampled": 0, "keys_per_state": collections.Counter(),
         "draw_pass_recorded": 0}
    keys_by_sid = collections.defaultdict(list)
    for (sid, k), recs in rec["samples"].items():
        if sid in st:
            keys_by_sid[sid].append((k, recs))
    answers = {}
    for sid, ks in keys_by_sid.items():
        f["states_sampled"] += 1
        f["keys_per_state"][len(ks)] += 1
        for k, recs in ks:
            o = _final_record(recs)
            f["reply_keys"] += 1
            dp = o.get("draw_pass")
            if dp is not None:
                f["draw_pass_recorded"] += 1
            passes = int(dp) if dp is not None else len(recs)
            f["replies_drawn"] += passes
            f["redrawn_keys"] += int(passes > 1)
            if _complete(o):
                f["finished"] += 1
                answers[(sid, k)] = o.get("answer") or ""
            elif o.get("finish") == "length":
                f["truncated"] += 1
            else:
                f["errors"] += 1
    for sid in st:
        n_use = 0
        for x in rec["scored"].get(sid, []):
            if not x.get("ok") or x.get("pct") is None or (x.get("noop") and not allow_noop):
                continue
            r = _reasoning_of(answers.get((sid, int(x["sample"])), ""))
            if not r or "<think>" in r or "</think>" in r:
                continue
            n_use += 1
        f["usable"] += n_use
        f["states_with_usable"] += int(n_use > 0)
    f["keys_per_state"] = {str(k): v for k, v in sorted(f["keys_per_state"].items())}
    return f


def recipe_findings(rec: dict, recipe: dict, draw_passes: int, allow_dry_run: bool) -> Tuple[list, list, dict]:
    """(errors, warnings, observed) for one group's samples against the recipe."""
    errs, warns = [], []
    obs = {"prompt": collections.Counter(), "mode": collections.Counter(), "model": collections.Counter(),
           "max_tokens_seen": 0, "thinking_records": 0, "truncated_tokens": collections.Counter(),
           "draw_pass_max": 0, "records_without_draw_pass": 0, "k_per_state": collections.Counter()}
    per_sid = collections.defaultdict(set)
    for (sid, k), recs in rec["samples"].items():
        per_sid[sid].add(k)
        for o in recs:
            obs["prompt"][str(o.get("prompt"))] += 1
            obs["mode"][str(o.get("mode"))] += 1
            obs["model"][str(o.get("model"))] += 1
            if o.get("tokens") is not None:
                obs["max_tokens_seen"] = max(obs["max_tokens_seen"], int(o["tokens"]))
            if (o.get("thinking") or "").strip():
                obs["thinking_records"] += 1
            if o.get("finish") == "length":
                obs["truncated_tokens"][str(o.get("tokens"))] += 1
            if o.get("draw_pass") is None:
                obs["records_without_draw_pass"] += 1
            else:
                obs["draw_pass_max"] = max(obs["draw_pass_max"], int(o["draw_pass"]))
    for sid, ks in per_sid.items():
        obs["k_per_state"][len(ks)] += 1
    g = rec["name"]
    if not rec["samples"] and rec["states"]:
        warns.append("group %s: no sample records (samples.jsonl missing or empty); recipe and funnel not "
                     "verifiable from the files" % g)
    if set(obs["prompt"]) - {recipe["prompt"]}:
        errs.append("group %s: sample prompts %s, recipe says %s" % (g, dict(obs["prompt"]), recipe["prompt"]))
    modes = set(obs["mode"])
    dry = {m for m in modes if m.endswith("_dryrun")}
    if dry and not allow_dry_run:
        errs.append("group %s: %d dry-run replies (mode %s): no model was called" % (
            g, sum(obs["mode"][m] for m in dry), sorted(dry)))
    if modes - dry - {recipe["mode"]}:
        errs.append("group %s: sample modes %s, recipe says %s" % (g, sorted(modes), recipe["mode"]))
    bad_models = [m for m in obs["model"] if m != "None" and recipe["model"].lower() not in m.lower()]
    if bad_models:
        errs.append("group %s: replies from %s, recipe says %s" % (g, bad_models, recipe["model"]))
    if obs["max_tokens_seen"] > recipe["max_tokens"]:
        errs.append("group %s: a reply has %d tokens, above the %d-token cap" % (g, obs["max_tokens_seen"],
                                                                                recipe["max_tokens"]))
    off = {t: n for t, n in obs["truncated_tokens"].items() if t not in (str(recipe["max_tokens"]), "None")}
    if off:
        warns.append("group %s: truncated replies with token counts %s (cap %d)" % (g, off, recipe["max_tokens"]))
    if obs["thinking_records"] and not recipe["thinking"]:
        warns.append("group %s: %d replies carry thinking text although thinking is off" % (g, obs["thinking_records"]))
    if obs["draw_pass_max"] > draw_passes:
        errs.append("group %s: a reply was drawn in pass %d; the batch declares %d pass(es)" % (
            g, obs["draw_pass_max"], draw_passes))
    if rec["samples"] and obs["records_without_draw_pass"]:
        warns.append("group %s: %d sample records carry no draw_pass (sampled before the field existed); "
                     "their pass is counted from the records present" % (g, obs["records_without_draw_pass"]))
    ks = {int(k) for k in obs["k_per_state"]}
    if ks and ks != {recipe["k"]}:
        warns.append("group %s: replies per state %s, recipe K=%d" % (g, dict(obs["k_per_state"]), recipe["k"]))
    gm = rec["manifest"]
    if float(gm.get("min_pct", -1)) != float(recipe["min_pct"]):
        errs.append("group %s: tiers built with min_pct %s, recipe %s" % (g, gm.get("min_pct"), recipe["min_pct"]))
    if "max_words" in gm:
        errs.append("group %s: tiers built with a reasoning word limit (max_words %s); rebuild them with "
                    "the current build_tiers, which has none" % (g, gm["max_words"]))
    for key in ("prompt", "mode", "model", "truncated_tokens", "k_per_state"):
        obs[key] = dict(obs[key])
    return errs, warns, obs


def chosen_draw_pass(rec: dict, sid: str, k: int) -> Optional[int]:
    """The pass that drew the reply a row was built from: the last complete
    record of (sid, k), as build_tiers reads it."""
    last = None
    for o in rec["samples"].get((sid, k), []):
        if o.get("answer") and o.get("finish") != "length":
            last = o
    return int(last["draw_pass"]) if last is not None and last.get("draw_pass") is not None else None


def build_append(a) -> int:
    for need in ("spelling", "batch", "rows_root", "out"):
        if not getattr(a, need):
            raise SystemExit("--append needs --%s" % need.replace("_", "-"))
    if not BATCH_RE.match(a.batch) or a.batch == "b001":
        raise SystemExit("--batch must be b002, b003, ... (b001 is the top-level rows/)")
    if not a.prior:
        raise SystemExit("--append needs --prior (the b001 spelling directory or its rows/, and every earlier "
                         "batch): the duplicate and split checks run against them")
    if not (a.eval_problems_native and a.eval_problems_v2):
        raise SystemExit("--append needs both --eval-problems-native and --eval-problems-v2")
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        if not a.force:
            raise SystemExit("%s exists and is not empty; --force to rebuild it" % out)
        for name in ("rows", "manifest.json", "BATCH.md", "grammar"):
            p = out / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    try:
        return _build_append(a, out)
    except BaseException as e:
        if not (isinstance(e, SystemExit) and e.code in (0, None)):
            for name in ("rows", "manifest.json", "BATCH.md", "grammar"):
                p = out / name
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
            print("partial output removed from %s" % out, flush=True)
        raise


def _build_append(a, out: Path) -> int:
    log = lambda s: print(s, flush=True)
    spelling, batch = a.spelling, a.batch
    ds = "datav4_" + spelling
    if out.name != batch:
        log("note: --out basename is %r; it becomes %s/%s/batches/%s on bridges2" % (out.name, BRIDGES2_ROOT, ds, batch))
    recipe = dict(RECIPE_DEFAULT, k=a.recipe_k, max_tokens=a.recipe_max_tokens, temperature=a.recipe_temperature)
    prov = KNOWN_PROVENANCE
    if a.provenance:
        prov = deep_merge(prov, read_json(Path(a.provenance)))
    errors: List[str] = []
    warnings: List[str] = []

    # 1. prior trees, evaluation sets -------------------------------------------
    log("prior trees (read-only):")
    prior = read_prior_trees(a.prior, spelling, log, skip=out)
    if batch in prior["batches"]:
        raise SystemExit("REFUSED: batch %s already exists among the prior trees (%s)" % (
            batch, ", ".join(t["label"] for t in prior["trees"] if t.get("batch") == batch)))
    prior_state_split = read_prior_state_splits(a.prior_states)
    ev = load_eval_sets([a.eval_problems_native, a.eval_problems_v2])
    log("evaluation sets: %s" % ", ".join("%s (%d problems)" % (Path(f["path"]).name, f["problems"])
                                          for f in ev["files"]))

    # 2. this batch's groups --------------------------------------------------------
    rows_root = Path(a.rows_root).resolve()
    groups = discover_groups(rows_root, a.groups)
    for rec in groups.values():
        load_group_inputs(rec)
    fam_rows, fam_group = split_by_family(groups)
    families = sorted(fam_rows, key=lambda f: (FAMILIES.index(f) if f in FAMILIES else 99, f))
    log("batch %s, %s: groups %s; families with rows: %s" % (batch, spelling, ", ".join(groups), ", ".join(families)))
    for f in families:
        if f not in FAMILIES:
            errors.append("%s: unknown family" % f)
        if f in HELD_OUT:
            errors.append("%s: held-out family (thermal2, inventory2 and molopt are never given rows); %d rows"
                          % (f, sum(len(v) for v in fam_rows[f].values())))
    if errors:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(errors))

    # twin spelling: the same states
    twin = None
    if a.twin_states:
        twin_root = Path(a.twin_states).resolve()
        tw = {}
        for sp in sorted(twin_root.glob("*/states.jsonl")):
            for _, o in read_jsonl_lines(sp):
                tw[o["sid"]] = o.get("split", "train")
        mine = {sid: s["split"] for rec in groups.values() for sid, s in rec["states"].items()}
        only_me = sorted(set(mine) - set(tw))
        only_tw = sorted(set(tw) - set(mine))
        split_diff = sorted(s for s in set(mine) & set(tw) if mine[s] != tw[s])
        twin = {"root": str(twin_root), "states_here": len(mine), "states_twin": len(tw),
                "only_here": len(only_me), "only_twin": len(only_tw), "split_differs": len(split_diff)}
        log("twin states (%s): %d here, %d there, identical sets and splits: %s" % (
            twin_root, len(mine), len(tw), not (only_me or only_tw or split_diff)))
        if only_me or only_tw or split_diff:
            raise SystemExit("REFUSED (twin spelling): the states differ: %d only here (%s), %d only in the twin "
                             "(%s), %d with another split" % (len(only_me), only_me[:3], len(only_tw), only_tw[:3],
                                                               len(split_diff)))

    # recipe
    recipe_obs = {}
    for g, rec in groups.items():
        e, w, obs = recipe_findings(rec, recipe, a.draw_passes, a.allow_dry_run)
        errors += e
        warnings += w
        recipe_obs[g] = obs
    if errors:
        raise SystemExit("REFUSED (recipe):\n  " + "\n  ".join(errors))

    # 3. worker: grammar, instruction, parse ------------------------------------------
    job = {"spelling": spelling, "families": families, "examples": {}, "parse": {}}
    if not a.no_parse_check:
        for fam in families:
            items = []
            for t in ("train_own_act", "dev_in_own_act"):
                for _, o in fam_rows[fam][t]:
                    m = ACTION_RE.match(((o.get("messages") or [{}])[-1].get("content") or "").strip())
                    if m and "action" in o:
                        items.append([m.group(1).strip(), o["action"]])
            job["parse"][fam] = items
    res = run_worker(spelling, job)
    instr_all = res["instructions"]
    current = {sha: d for sha, d in instr_all.items() if d["spelling"] == spelling and d["current"]}
    instr_sha, instr = next(iter(current.items()))
    accepted = {instr_sha: instr["text"]}
    for sha in a.accept_instruction_sha1 or []:
        d = instr_all.get(sha)
        if d and d["spelling"] != spelling:
            raise SystemExit("--accept-instruction-sha1 %s is the %s instruction" % (sha, d["spelling"]))
        accepted[sha] = d["text"] if d else None
    grammars = {fam: res["families"][fam]["grammar"] for fam in families}

    # consistency with the prior spelling roots (b001's instruction.txt / grammar/)
    root_checks = []
    for root in prior["roots"]:
        itxt = read_text(root / "instruction.txt")
        ok_i = sha1_text(itxt)[:12] == instr_sha and itxt == instr["text"]
        if not ok_i:
            errors.append("the served instruction (%s) differs from %s/instruction.txt (%s)" % (
                instr_sha, root, sha1_text(itxt)[:12]))
        gdiff = []
        for fam in families:
            gp = root / "grammar" / (fam + ".txt")
            if gp.is_file() and read_text(gp) != grammars[fam]:
                gdiff.append(fam)
        if gdiff:
            msg = "grammar of %s differs from %s/grammar/" % (gdiff, root)
            if a.allow_grammar_drift:
                warnings.append(msg + " (allowed; this batch's grammars are written to grammar/)")
            else:
                errors.append(msg + " (--allow-grammar-drift writes them to the batch's grammar/)")
        root_checks.append({"root": str(root), "instruction_equal": ok_i, "grammar_differs": gdiff})
    if not prior["roots"]:
        warnings.append("no prior spelling root (instruction.txt + grammar/) among --prior; the instruction and "
                        "grammars were not compared with b001's")
    if errors:
        raise SystemExit("REFUSED (prior consistency):\n  " + "\n  ".join(errors))

    # 4. per family: eval drops, row checks, prior checks, stamping ---------------------------
    fam_out, all_canon = {}, {}
    failures = []
    for fam in families:
        g = fam_group[fam]
        rec = groups[g]
        tiers = fam_rows[fam]
        spec_canon = {s["problem_id"]: s["canon"] for s in rec["states"].values() if s["variant"] == fam}
        eval_ids = ev["ids"].get(fam, set())
        insts = sorted({o.get("instance") for t in TIERS for _, o in tiers[t]})
        missing = [i for i in insts if spec_canon.get(i) is None]
        if missing:
            failures.append("%s: no spec in %s for %d row problems (%s)" % (fam, rec["states_path"], len(missing),
                                                                             missing[:3]))
            continue
        collide = {}
        for i in insts:
            if i in eval_ids:
                collide[i] = "id"
            elif spec_canon[i] in ev["canon"]:
                collide[i] = "spec = %s" % ev["canon"][spec_canon[i]]
        dropped = {t: sum(1 for _, o in tiers[t] if o.get("instance") in collide) for t in TIERS}
        if collide:
            tiers = {t: [(l, o) for l, o in tiers[t] if o.get("instance") not in collide] for t in TIERS}
            warnings.append("%s: %d rows of %d evaluation problems dropped (%s)" % (
                fam, dropped["train_own"] + dropped["dev_in_own"], len(collide), sorted(collide)[:5]))
        rep, errs, warns, out_lines = validate_family_rows(fam, spelling, tiers, accepted, grammars[fam],
                                                           eval_ids, a.stamp_spelling)
        warnings += warns
        if rep["served_grammar_variants"] and not a.allow_grammar_drift:
            errs.append("%s: served grammar differs from the current one in %d rows" % (
                fam, sum(v["rows"] for v in rep["served_grammar_variants"])))
        pc = res["families"][fam]["parse"]
        if not a.no_parse_check:
            # the parse job held every own_act row; dropped rows are re-parsed too, which is harmless
            if pc["n"] and pc["ok"] != pc["n"]:
                errs.append("%s: %d of %d action lines do not re-parse under %s; first %s" % (
                    fam, pc["n"] - pc["ok"], pc["n"], spelling, pc["failures"][:3]))
        # duplicates within the batch, prior collisions, split consistency
        seen = collections.Counter()
        prior_hits, split_conf = [], []
        for t in TIERS:
            split = "dev" if t.startswith("dev") else "train"
            for _, o in tiers[t]:
                inst, turn = o.get("instance"), int(o.get("turn"))
                s = sid_of(fam, inst, turn)
                if t in ("train_own", "dev_in_own"):
                    seen[s] += 1
                hit = prior["sids"].get(s) or prior["inst_turn"].get((inst, turn))
                if hit:
                    prior_hits.append("%s (%s, in %s)" % (s, t, hit))
                ps = prior["split"].get(inst) or prior_state_split.get(inst)
                if ps and ps[0] != split:
                    split_conf.append("%s: %s here, %s in %s" % (inst, split, ps[0], ps[1]))
        dup = [s for s, n in seen.items() if n > 1]
        if dup:
            errs.append("%s: %d sids appear twice in this batch (%s)" % (fam, len(dup), dup[:3]))
        if prior_hits:
            errs.append("%s: %d rows repeat a state of a prior batch: %s" % (fam, len(prior_hits), prior_hits[:3]))
        split_conf = sorted(set(split_conf))
        if split_conf:
            errs.append("%s: %d problems change split against a prior batch: %s" % (fam, len(split_conf),
                                                                                    split_conf[:3]))
        # states of a problem: all train or all dev within the batch
        st_split = collections.defaultdict(set)
        for s in rec["states"].values():
            if s["variant"] == fam:
                st_split[s["problem_id"]].add(s["split"])
        mixed = sorted(p for p, ss in st_split.items() if len(ss) > 1)
        if mixed:
            errs.append("%s: problems with states in both splits: %s" % (fam, mixed[:5]))
        if errs:
            failures += errs
            continue
        # stamp
        stamped = {t: [] for t in TIERS}
        dp_counter = collections.Counter()
        for t in TIERS:
            for line in out_lines[t]:
                o = json.loads(line)
                if o.get("batch") not in (None, batch):
                    failures.append("%s: a row already says batch %s" % (fam, o.get("batch")))
                o["batch"] = batch
                o["draw_passes"] = int(a.draw_passes)
                dp = chosen_draw_pass(rec, sid_of(fam, o["instance"], o["turn"]), int(o.get("rationale_sample", -1)))
                if dp is not None:
                    o["draw_pass"] = dp
                if t == "train_own" or t == "dev_in_own":
                    dp_counter[str(dp)] += 1
                stamped[t].append(json.dumps(o))
        rd = out / "rows" / fam
        rd.mkdir(parents=True, exist_ok=True)
        for t in TIERS:
            write_text(rd / (t + ".jsonl"), "".join(l + "\n" for l in stamped[t]))
        write_json(rd / "manifest_own.json", family_manifest_own(
            rec["manifest"], g, "rows/_groups/%s/manifest_own.json" % g, rec["manifest_sha1"], fam, tiers))
        funnel = family_funnel(fam, rec)
        funnel["rows"] = {"train": rep["rows"]["train_own"], "dev": rep["rows"]["dev_in_own"]}
        funnel["dropped_eval"] = {"rows": dropped["train_own"] + dropped["dev_in_own"], "problems": sorted(collide),
                                  "why": collide}
        funnel["rows_by_draw_pass"] = dict(dp_counter)
        canon_sha = {p: sha1_text(spec_canon[p]) for p in rep["problems"]["train"] + rep["problems"]["dev"]}
        all_canon.update(canon_sha)
        fam_out[fam] = {
            "group": g, "rows": rep["rows"], "problems": rep["problems"],
            "problem_canon_sha1": canon_sha,
            "checks": {"stamped_spelling": rep["stamped"], "tail_checked": rep["tail_checked"],
                       "tail_ok": rep["tail_ok"], "served_grammar_variants": len(rep["served_grammar_variants"]),
                       "instruction_sha1_counts": rep["instruction_sha1"],
                       "parse": ({"n": pc["n"], "ok": pc["ok"]} if not a.no_parse_check else None),
                       "prior_sid_collisions": 0, "prior_split_conflicts": 0,
                       "eval_collisions_dropped": len(collide)},
            "grammar": {"sha1": sha1_text(grammars[fam]), "text": grammars[fam]},
            "funnel": funnel,
            "manifest_own": read_json(rd / "manifest_own.json"),
        }
        log("  rows %-11s train %4d dev %3d | states %d/%d replies %d finished %d usable %d | eval drops %d | "
            "prior hits 0 | tail %d/%d parse %s" % (
                fam, rep["rows"]["train_own"], rep["rows"]["dev_in_own"], funnel["states"]["train"],
                funnel["states"]["dev"], funnel["replies_drawn"], funnel["finished"], funnel["usable"],
                funnel["dropped_eval"]["rows"], rep["tail_ok"], rep["tail_checked"],
                ("%d/%d" % (pc["ok"], pc["n"])) if pc["n"] else "-"))
    if failures:
        raise SystemExit("REFUSED - nothing written; fix the inputs and rebuild:\n  " + "\n  ".join(failures[:60]))
    for g, rec in groups.items():
        gd = out / "rows" / "_groups" / g
        gd.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rec["tiers_dir"] / "manifest_own.json", gd / "manifest_own.json")
        if rec["select_report_path"]:
            shutil.copyfile(rec["select_report_path"], gd / "select_report.json")
    if a.allow_grammar_drift:
        for root in root_checks:
            for fam in root["grammar_differs"]:
                write_text(out / "grammar" / (fam + ".txt"), grammars[fam])

    # 5. manifest and BATCH.md ---------------------------------------------------------
    jobs = {}
    for spec in a.source_job or []:
        k, _, v = spec.partition("=")
        jobs.setdefault(k, []).append(v)
    manifest = {
        "kind": "batch", "batch": batch, "dataset": ds, "spelling": spelling,
        "built": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "builder": "dataxp/package_datav4.py --append",
        "repo_head": git_head(), "code": code_sha1s(),
        "bridges2_destination": "%s/%s/batches/%s/" % (BRIDGES2_ROOT, ds, batch),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if not k.startswith("_")},
        "recipe": dict(recipe, draw_passes=int(a.draw_passes), acceptance=ACCEPTANCE_RULE,
                       redraw="one pass: dataxp.ownrows.sample re-run on the same output redraws exactly the "
                              "(state, sample) keys whose reply ended at the token cap (finish == length)",
                       held_out=list(HELD_OUT)),
        "recipe_observed": recipe_obs,
        "instruction": {"name": instr["name"], "sha1_12": instr_sha, "sha1": sha1_text(instr["text"]),
                        "text": instr["text"], "accepted": sorted(accepted),
                        "accepted_texts": {s: t for s, t in accepted.items() if s != instr_sha}},
        "family_order": families, "family_group": fam_group, "families": fam_out,
        "groups": {g: {"dir": str(rec["dir"]), "manifest_sha1": rec["manifest_sha1"],
                       "source_files": rec["source_files"], "work_files": rec["work_files"],
                       "jobs": jobs.get(g, [])} for g, rec in groups.items()},
        "source_jobs": jobs,
        "prior": {"args": list(a.prior), "trees": prior["trees"], "batches": sorted(prior["batches"]),
                  "n_sids": len(prior["sids"]), "n_problems": len(prior["split"]),
                  "prior_states": list(a.prior_states or []), "roots_checked": root_checks},
        "eval_sets": ev["files"], "twin": twin, "warnings": warnings, "notes": list(a.note or []),
    }
    write_batch_md(out, manifest)
    manifest["files"] = file_table(out, out, exclude=("manifest.json",))
    write_json(out / "manifest.json", manifest)
    tot = {s: sum(f["rows"][t] for f in fam_out.values()) for s, t in (("train", "train_own"), ("dev", "dev_in_own"))}
    log("built batch %s (%s) at %s: %d families, rows train %d dev %d, %d files%s" % (
        batch, spelling, out, len(fam_out), tot["train"], tot["dev"], len(manifest["files"]),
        ("\nwarnings:\n  " + "\n  ".join(warnings)) if warnings else ""))
    return 0


def write_batch_md(out: Path, m: dict) -> None:
    r = m["recipe"]
    ds, b = m["dataset"], m["batch"]
    L = ["# %s — batch %s" % (ds, b), "",
         "Appended own-action rows, `%s` spelling, batch `%s`. Built %s by `%s` (repo HEAD %s). Lives at "
         "`%s` next to the b001 rows (`../../rows/`), which it never changes. Every row carries "
         "`\"batch\": \"%s\"`, `\"draw_passes\": %d` and, where the sampler recorded it, `\"draw_pass\"`." % (
             m["spelling"], b, m["built"], m["builder"], (m.get("repo_head") or "n/a")[:12],
             m["bridges2_destination"], b, r["draw_passes"]), "",
         "Validate: `python -m dataxp.package_datav4 --check --out <this dir> [--prior <spelling dir>] "
         "[--eval-problems-native ... --eval-problems-v2 ...] [--parse-check]`.", "",
         "## Recipe", "",
         "- teacher: the un-tuned %s, thinking off, `--prompt %s`, K = %d replies per state, temperature %s, "
         "%d-token cap; mode `%s`" % (r["model"], r["prompt"], r["k"], r["temperature"], r["max_tokens"], r["mode"]),
         "- draw passes: %d (%s)" % (r["draw_passes"], "the K-reply draw only" if r["draw_passes"] == 1 else
                                     "the K-reply draw + " + r["redraw"]),
         "- acceptance (min_pct %s): %s" % (r["min_pct"], r["acceptance"]),
         "- held out, never given rows: %s; evaluation problems never used (canonical spec checked against %s)" % (
             ", ".join(r["held_out"]), ", ".join("`%s`" % e["path"] for e in m["eval_sets"])),
         "- served instruction `%s` (%s), identical to the spelling directory's `instruction.txt`; grammars "
         "identical to its `grammar/<family>.txt` (checked: %s)" % (
             m["instruction"]["sha1_12"], m["instruction"]["name"],
             "; ".join("%s instruction %s, grammar differs %s" % (Path(c["root"]).name, c["instruction_equal"],
                                                                  c["grammar_differs"] or "none")
                       for c in m["prior"]["roots_checked"]) or "no spelling root given"), "",
         "## Funnel per family", "",
         "| family | group | states train / dev | problems | replies drawn | redrawn keys | finished | truncated "
         "| usable replies | states with a usable reply | rows train / dev | rows by draw pass | eval drops |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for fam in m["family_order"]:
        f = m["families"][fam]
        fu = f["funnel"]
        L.append("| %s | %s | %d / %d | %d | %d | %d | %d | %d | %d | %d | %d / %d | %s | %d |" % (
            fam, f["group"], fu["states"]["train"], fu["states"]["dev"], fu["problems"], fu["replies_drawn"],
            fu["redrawn_keys"], fu["finished"], fu["truncated"], fu["usable"], fu["states_with_usable"],
            fu["rows"]["train"], fu["rows"]["dev"],
            ", ".join("%s: %d" % kv for kv in sorted(fu["rows_by_draw_pass"].items())) or "—",
            fu["dropped_eval"]["rows"]))
        for k in ("replies_drawn", "redrawn_keys", "finished", "truncated", "usable", "states_with_usable"):
            tot[k] += fu[k]
        tot["st_tr"] += fu["states"]["train"]
        tot["st_dev"] += fu["states"]["dev"]
        tot["rows_tr"] += fu["rows"]["train"]
        tot["rows_dev"] += fu["rows"]["dev"]
        tot["problems"] += fu["problems"]
        tot["drops"] += fu["dropped_eval"]["rows"]
    L += ["| **total** | | %d / %d | %d | %d | %d | %d | %d | %d | %d | **%d / %d** | | %d |" % (
        tot["st_tr"], tot["st_dev"], tot["problems"], tot["replies_drawn"], tot["redrawn_keys"], tot["finished"],
        tot["truncated"], tot["usable"], tot["states_with_usable"], tot["rows_tr"], tot["rows_dev"], tot["drops"]), "",
          "states = the batch's selected repair-phase states; replies drawn = every draw over all passes (a key "
          "redrawn once counts twice); finished = keys whose final reply ended before the cap; usable = finished "
          "replies that parsed, applied, were not a no-op and carry non-empty written reasoning "
          "without a think tag; rows = states kept by the acceptance rule (own = own_act, row for row); eval "
          "drops = rows removed because their problem equals an evaluation problem.", "",
          "## Checks", "",
          "| family | rows checked (all tiers) | served tail ok | re-parse ok | prior sid / (instance, turn) "
          "collisions | split conflicts | evaluation problems dropped |", "|---|---|---|---|---|---|---|"]
    for fam in m["family_order"]:
        f = m["families"][fam]
        c = f["checks"]
        L.append("| %s | %d | %d / %d | %s | %d | %d | %d |" % (
            fam, sum(f["rows"].values()), c["tail_ok"], c["tail_checked"],
            ("%d / %d" % (c["parse"]["ok"], c["parse"]["n"])) if c.get("parse") else "not run",
            c["prior_sid_collisions"], c["prior_split_conflicts"], c["eval_collisions_dropped"]))
    pr = m["prior"]
    L += ["", "Prior trees checked (%d sids, %d problems with a split): %s." % (
        pr["n_sids"], pr["n_problems"], "; ".join("`%s` (%d rows)" % (t["label"], t["rows"]) for t in pr["trees"])),
          "No held-out family has rows; every row's spelling is `%s`." % m["spelling"]]
    if m.get("twin"):
        t = m["twin"]
        L.append("Twin spelling (`%s`): %d states here, %d there; only here %d, only there %d, split differs %d." % (
            t["root"], t["states_here"], t["states_twin"], t["only_here"], t["only_twin"], t["split_differs"]))
    L += ["", "## Sources", "", "| group | file | sha1 | records | jobs |", "|---|---|---|---|---|"]
    for g, gr in m["groups"].items():
        for name, wf in gr["work_files"].items():
            L.append("| %s | `%s` | %s | %d | %s |" % (g, wf["path"], wf["sha1"][:12], wf["rows"],
                                                      "; ".join(gr["jobs"]) or "—"))
        for t, sf in gr["source_files"].items():
            L.append("| %s | `%s` | %s | %d | |" % (g, sf["path"], sf["sha1"][:12], sf["rows"]))
    if m.get("notes"):
        L += ["", "## Notes", ""] + ["- " + n for n in m["notes"]]
    if m.get("warnings"):
        L += ["", "## Build warnings", ""] + ["- " + w for w in m["warnings"]]
    L += ["", "## Layout", "", md_code("batches/%s/\n  BATCH.md, manifest.json\n  rows/<family>/{train_own,train_own_act,"
                                        "dev_in_own,dev_in_own_act}.jsonl, manifest_own.json\n  rows/_groups/<group>/"
                                        "{manifest_own.json,select_report.json}" % b), ""]
    write_text(out / "BATCH.md", "\n".join(L))


def check_batch(a, out: Path, m: dict) -> int:
    log = lambda s: print(s, flush=True)
    errors, warnings = [], []
    spelling, batch = m["spelling"], m["batch"]
    log("check batch %s (%s) at %s" % (batch, spelling, out))
    n_ok = 0
    for rel, f in m["files"].items():
        p = out / rel
        if not p.is_file():
            errors.append("missing file %s" % rel)
        elif sha1_file(p) != f["sha1"]:
            errors.append("sha1 mismatch %s" % rel)
        else:
            n_ok += 1
    extra = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
             and p.relative_to(out).as_posix() not in m["files"] and p.name != "manifest.json"]
    if extra:
        warnings.append("files not in the manifest: %s" % extra[:10])
    log("  files: %d of %d sha1 ok" % (n_ok, len(m["files"])))
    ins = m["instruction"]
    accepted = {ins["sha1_12"]: ins["text"]}
    accepted.update(ins.get("accepted_texts") or {})
    prior = read_prior_trees(a.prior, spelling, log, skip=out) if a.prior else None
    ev = load_eval_sets([p for p in (a.eval_problems_native, a.eval_problems_v2) if p]) \
        if (a.eval_problems_native or a.eval_problems_v2) else None
    ev_canon_sha = {sha1_text(c) for c in ev["canon"]} if ev else set()
    for fam in m["family_order"]:
        fm = m["families"][fam]
        if fam in HELD_OUT:
            errors.append("%s: held-out family has rows" % fam)
        if sha1_text(fm["grammar"]["text"]) != fm["grammar"]["sha1"]:
            errors.append("%s: grammar text and sha1 disagree in the manifest" % fam)
        tiers = {}
        for t in TIERS:
            p = out / "rows" / fam / (t + ".jsonl")
            if not p.is_file():
                errors.append("%s: %s missing" % (fam, p.name))
                break
            tiers[t] = read_jsonl_lines(p)
        if len(tiers) != len(TIERS):
            continue
        eval_ids = set()
        if ev:
            eval_ids = ev["ids"].get(fam, set())
        rep, errs, warns, _ = validate_family_rows(fam, spelling, tiers, accepted, fm["grammar"]["text"],
                                                   eval_ids, stamp=False)
        errors += errs
        warnings += warns
        for t in TIERS:
            if rep["rows"][t] != fm["rows"][t]:
                errors.append("%s: %s has %d rows, manifest says %d" % (fam, t, rep["rows"][t], fm["rows"][t]))
        n_bad_stamp = sum(1 for t in TIERS for _, o in tiers[t]
                          if o.get("batch") != batch or o.get("draw_passes") != m["recipe"]["draw_passes"])
        if n_bad_stamp:
            errors.append("%s: %d rows without batch %s / draw_passes %d" % (fam, n_bad_stamp, batch,
                                                                             m["recipe"]["draw_passes"]))
        hits = sorted(p for p, h in fm.get("problem_canon_sha1", {}).items() if h in ev_canon_sha)
        if hits:
            errors.append("%s: %d problems equal an evaluation problem (canonical spec): %s" % (fam, len(hits), hits[:5]))
        if prior:
            ph, sc = [], []
            for t in TIERS:
                split = "dev" if t.startswith("dev") else "train"
                for _, o in tiers[t]:
                    s = sid_of(fam, o["instance"], o["turn"])
                    if s in prior["sids"] or (o["instance"], int(o["turn"])) in prior["inst_turn"]:
                        ph.append(s)
                    ps = prior["split"].get(o["instance"])
                    if ps and ps[0] != split:
                        sc.append(o["instance"])
            if ph:
                errors.append("%s: %d rows repeat a prior state (%s)" % (fam, len(ph), ph[:3]))
            if sc:
                errors.append("%s: %d rows change a prior problem's split (%s)" % (fam, len(sc), sorted(set(sc))[:3]))
        log("  rows %-11s train %4d dev %3d  tail %d/%d  %s" % (
            fam, rep["rows"]["train_own"], rep["rows"]["dev_in_own"], rep["tail_ok"], rep["tail_checked"],
            "ok" if not errs else "%d errors" % len(errs)))
    if a.parse_check:
        job = {"spelling": spelling, "families": list(m["family_order"]), "examples": {}, "parse": {}}
        for fam in job["families"]:
            items = []
            for t in ("train_own_act", "dev_in_own_act"):
                for _, o in read_jsonl_lines(out / "rows" / fam / (t + ".jsonl")):
                    mm = ACTION_RE.match(((o.get("messages") or [{}])[-1].get("content") or "").strip())
                    if mm:
                        items.append([mm.group(1).strip(), o["action"]])
            job["parse"][fam] = items
        res = run_worker(spelling, job)
        for fam in job["families"]:
            pc = res["families"][fam]["parse"]
            if pc["ok"] != pc["n"]:
                errors.append("%s: %d of %d action lines do not re-parse" % (fam, pc["n"] - pc["ok"], pc["n"]))
            same = res["families"][fam]["grammar"] == m["families"][fam]["grammar"]["text"]
            if not same:
                errors.append("%s: the grammar the current code generates differs from the batch's" % fam)
            log("  parse %-11s %d/%d ok; current grammar %s" % (fam, pc["ok"], pc["n"], "identical" if same else "DIFFERS"))
    for w in warnings:
        log("  warning: " + w)
    if errors:
        log("CHECK FAILED (%d):\n  " % len(errors) + "\n  ".join(errors[:60]))
        return 1
    log("CHECK OK: batch %s (%s), %d families%s%s" % (
        batch, spelling, len(m["family_order"]), ", prior trees re-checked" if prior else "",
        ", evaluation sets re-checked" if ev else ""))
    return 0


# ---------------------------------------------------------------------------
# append: a corpus batch
# ---------------------------------------------------------------------------
def _corpus_src_dirs(path: str) -> Tuple[Path, Optional[Path]]:
    p = Path(path).resolve()
    if (p / "accepted" / "problems.jsonl").is_file():
        return p / "accepted", (p / "report.json") if (p / "report.json").is_file() else None
    if (p / "problems.jsonl").is_file():        # an accepted/ dir, or run_generation's --tmp/<variant>
        rep = p / "report.json" if (p / "report.json").is_file() else p.parent / "report.json"
        return p, rep if rep.is_file() else None
    raise SystemExit("--corpus-src %s: no accepted/problems.jsonl or problems.jsonl" % path)


def _prior_corpus_problems(paths: List[str], fam: str, exclude_dir: Path) -> Dict[str, dict]:
    """problem_id -> {"canon", "file"} of the family's problems in earlier corpora."""
    files = []
    for p in paths or []:
        p = Path(p).resolve()
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += sorted(p.glob("%s/**/problems.jsonl" % fam)) + sorted(p.glob("**/%s/accepted*/problems.jsonl" % fam))
    out = {}
    for f in sorted(set(files)):
        if exclude_dir in f.parents:
            continue
        for _, s in read_jsonl_lines(f):
            v = s.get("domain_variant") or str(s.get("problem_id", "")).rsplit("_", 1)[0]
            if v == fam:
                out.setdefault(s["problem_id"], {"canon": canon_spec(s), "file": str(f)})
    return out


def build_corpus_batch(a) -> int:
    log = lambda s: print(s, flush=True)
    batch = a.corpus_batch
    if not BATCH_RE.match(batch) or batch == "b001":
        raise SystemExit("--corpus-batch must be b002, b003, ...")
    if not a.corpus_dir or not a.corpus_src:
        raise SystemExit("--corpus-batch needs --corpus-dir and --corpus-src <family>=<dir> ...")
    if not (a.eval_problems_native and a.eval_problems_v2):
        raise SystemExit("--corpus-batch needs both --eval-problems-native and --eval-problems-v2")
    cdir = Path(a.corpus_dir).resolve()
    ev = load_eval_sets([a.eval_problems_native, a.eval_problems_v2])
    plan = []
    for spec in a.corpus_src:
        fam, _, path = spec.partition("=")
        if not path:
            raise SystemExit("--corpus-src expects <family>=<dir>, got %r" % spec)
        if fam not in FAMILIES:
            raise SystemExit("--corpus-src: unknown family %s" % fam)
        acc, rep = _corpus_src_dirs(path)
        dest = cdir / fam / ("accepted_" + batch)
        if dest.exists() and any(dest.iterdir()):
            if not a.force:
                raise SystemExit("%s exists; a corpus batch is written once (--force only on a staging copy)" % dest)
            shutil.rmtree(dest)
        plan.append((fam, acc, rep, dest))
    errors = []
    for fam, acc, rep, dest in plan:
        specs = [o for _, o in read_jsonl_lines(acc / "problems.jsonl")]
        pids = [s["problem_id"] for s in specs]
        if len(set(pids)) != len(pids):
            errors.append("%s: duplicate problem ids in %s" % (fam, acc))
        prior = _prior_corpus_problems(a.corpus_prior, fam, dest)
        prior_canon = {v["canon"]: k for k, v in prior.items()}
        id_hits = sorted(p for p in pids if p in prior)
        canon_hits = sorted("%s = %s" % (s["problem_id"], prior_canon[canon_spec(s)]) for s in specs
                            if canon_spec(s) in prior_canon and s["problem_id"] not in prior)
        if id_hits:
            errors.append("%s: %d problem ids already in an earlier corpus (%s)" % (fam, len(id_hits), id_hits[:3]))
        if canon_hits:
            errors.append("%s: %d problems equal an earlier corpus problem (%s)" % (fam, len(canon_hits), canon_hits[:3]))
        if errors:
            continue
        ev_hits = {s["problem_id"]: ev["canon"][canon_spec(s)] for s in specs if canon_spec(s) in ev["canon"]}
        ev_ids = sorted(p for p in pids if p in ev["ids"].get(fam, set()))
        # not a collision (the specs differ), recorded for the owner: problems built on the same
        # dataset design as an evaluation problem (engibench: family + source_row)
        base_key = lambda s: (s.get("family"), s.get("source"), s.get("source_row")) \
            if s.get("source_row") is not None else None
        ev_base = collections.defaultdict(list)
        for _, r in [(None, x) for f in ev["files"] for _, x in read_jsonl_lines(Path(f["path"]))]:
            sp_ = r["spec"] if isinstance(r.get("spec"), dict) else r
            if base_key(sp_) is not None:
                ev_base[base_key(sp_)].append(sp_.get("problem_id"))
        shared_base = {s["problem_id"]: sorted(ev_base[base_key(s)]) for s in specs
                       if base_key(s) is not None and base_key(s) in ev_base}
        (dest).mkdir(parents=True, exist_ok=True)
        files = {}
        for name in CORPUS_FILES:
            if (acc / name).is_file():
                shutil.copyfile(acc / name, dest / name)
                files[name] = {"sha1": sha1_file(dest / name), "bytes": (dest / name).stat().st_size,
                               "rows": count_jsonl(dest / name), "source": str(acc / name)}
        if rep:
            shutil.copyfile(rep, dest / "report.json")
            files["report.json"] = {"sha1": sha1_file(dest / "report.json"), "bytes": (dest / "report.json").stat().st_size,
                                    "source": str(rep)}
        idx = [i for i in (problem_index(p) for p in pids) if i is not None]
        report = read_json(rep) if rep else {}
        src = {"family": fam, "corpus_batch": batch, "written": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "builder": "dataxp/package_datav4.py --corpus-batch", "repo_head": git_head(),
               "source_dir": str(acc), "files": files,
               "summary": {"n_problems": len(pids), "index_min": min(idx) if idx else None,
                           "index_max": max(idx) if idx else None,
                           "start_index": report.get("start_index"), "prebuilt": report.get("prebuilt"),
                           "all_gates_passed": report.get("all_gates_passed")},
               "problem_ids": pids,
               "earlier_corpora_checked": sorted({v["file"] for v in prior.values()}),
               "n_earlier_problems_checked": len(prior),
               "eval_sets": ev["files"],
               "eval_collisions": {"by_canonical_spec": ev_hits, "by_id": ev_ids,
                                   "note": "kept in the corpus (it records the generation); rows never use them: "
                                           "select_states --exclude-specs leaves them out and package_datav4 "
                                           "--append drops any row of them"},
               "shared_base_with_eval": {"by_problem": shared_base,
                                         "note": "problems built on the same dataset design (family, source, "
                                                 "source_row) as an evaluation problem; their specs differ "
                                                 "(perturbation, goals), so they are not collisions; recorded "
                                                 "for the owner's decision (docs/datav4_append_plan.md)"},
               "notes": list(a.note or [])}
        write_json(dest / "sources.json", src)
        log("  corpus %-11s %s: %d problems (index %s-%s), %s; eval collisions %d (spec) %d (id)" % (
            fam, dest, len(pids), src["summary"]["index_min"], src["summary"]["index_max"],
            ", ".join("%s %s" % (n, f.get("rows", "")) for n, f in files.items()), len(ev_hits), len(ev_ids)))
    if errors:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(errors))
    return 0


def check_corpus_batch(a) -> int:
    cdir = Path(a.corpus_dir).resolve()
    batch = a.corpus_batch
    dirs = sorted(cdir.glob("*/accepted_%s" % batch))
    if not dirs:
        raise SystemExit("no */accepted_%s under %s" % (batch, cdir))
    errors = []
    for d in dirs:
        s = read_json(d / "sources.json")
        for name, f in s["files"].items():
            p = d / name
            if not p.is_file() or sha1_file(p) != f["sha1"]:
                errors.append("%s/%s: missing or sha1 mismatch" % (d.parent.name, name))
        pids = [o["problem_id"] for _, o in read_jsonl_lines(d / "problems.jsonl")]
        if pids != s["problem_ids"]:
            errors.append("%s: problem ids differ from sources.json" % d.parent.name)
        print("  corpus %-11s accepted_%s: %d problems, %d files %s" % (
            d.parent.name, batch, len(pids), len(s["files"]), "ok" if not errors else ""), flush=True)
    if errors:
        print("CHECK FAILED:\n  " + "\n  ".join(errors), flush=True)
        return 1
    print("CHECK OK: corpus batch %s, %d families" % (batch, len(dirs)), flush=True)
    return 0


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--_worker"] and len(argv) == 3:  # the subprocess started by run_worker
        return worker_main(argv[1], argv[2])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spelling", choices=SPELLINGS, help="the spelling this directory serves")
    ap.add_argument("--rows-root", help="own-row tiers: <rows-root>/<group>/tiers/{train_own,...}.jsonl + "
                                        "manifest_own.json (drc /storage/wxu/corpus/rows for multitool, "
                                        "rows_sp for set_parameter)")
    ap.add_argument("--out", help="the dataset directory to build or check (.../datav4_<spelling>); with "
                                  "--append the batch directory (becomes <spelling dir>/batches/<batch>)")
    ap.add_argument("--check", action="store_true", help="validate a built --out (dataset or batch) or, with "
                                                         "--corpus-batch, a corpus batch, instead of building")
    ap.add_argument("--append", action="store_true", help="build a batch directory (rows only) to append")
    ap.add_argument("--batch", help="--append: the batch name, b002, b003, ...")
    ap.add_argument("--prior", nargs="+", default=[], metavar="PATH",
                    help="--append / batch --check: earlier rows, read-only: a spelling directory (its rows/ "
                         "and every batches/b*/rows/), a batch directory or a rows/ tree")
    ap.add_argument("--prior-states", nargs="+", default=[], metavar="FILE",
                    help="--append: earlier states.jsonl files whose problems keep their split even without rows")
    ap.add_argument("--draw-passes", type=int, default=2,
                    help="--append: sampling passes the batch ran (1 = the K-reply draw; 2 = plus the one "
                         "redraw of the truncated keys, the batch recipe); stamped on every row")
    ap.add_argument("--twin-states", help="--append: the other spelling's rows root; its <group>/states.jsonl "
                                          "must hold the same states with the same splits")
    ap.add_argument("--source-job", action="append", default=[], metavar="GROUP=TEXT",
                    help="--append: a job that produced a group's samples (repeatable)")
    ap.add_argument("--note", action="append", default=[], help="--append / --corpus-batch: a note for the record")
    ap.add_argument("--recipe-k", type=int, default=RECIPE_DEFAULT["k"])
    ap.add_argument("--recipe-max-tokens", type=int, default=RECIPE_DEFAULT["max_tokens"])
    ap.add_argument("--recipe-temperature", type=float, default=RECIPE_DEFAULT["temperature"])
    ap.add_argument("--allow-dry-run", action="store_true",
                    help="--append: accept dry-run replies (tests and demos only; recorded in the manifest)")
    ap.add_argument("--corpus-batch", help="write (or with --check verify) <corpus-dir>/<family>/accepted_<batch>/")
    ap.add_argument("--corpus-src", nargs="+", default=[], metavar="FAMILY=DIR",
                    help="--corpus-batch: run_generation output (<dir>/accepted + report.json) or an accepted/ dir")
    ap.add_argument("--corpus-prior", nargs="+", default=[], metavar="PATH",
                    help="--corpus-batch: earlier corpora (dirs searched for <family>/**/problems.jsonl, or "
                         "problems.jsonl files); a new problem id or canonical spec found there is refused")
    ap.add_argument("--groups", nargs="*", help="tier groups to package (default: every <group>/tiers/ found)")
    ap.add_argument("--families", nargs="*", help="families to package grammars/corpora for (default: the "
                                                  "eleven; families found in the rows are always added)")
    ap.add_argument("--corpus-mode", choices=("shared", "copy"), default="shared",
                    help="shared (default): corpora once in --corpus-dir, named by CORPUS.md; copy: under <out>/corpus")
    ap.add_argument("--corpus-dir", help="shared corpus dir (default: <out>/../datav4_corpus); with --check: where to find it")
    ap.add_argument("--native-corpus", help="drc /storage/wxu/corpus/out (<root>/<v>/<v>/accepted)")
    ap.add_argument("--v2-corpus", nargs="*", default=[], metavar="BATCH=ROOT",
                    help="v2 corpus batches to copy, <root>/<family>/accepted (e.g. run2_supp=/storage/wxu/"
                         "sftprobe/src/dataxp/run2_supp/batch1_idx0130-0259)")
    ap.add_argument("--v2-corpus-ref", nargs="*", default=[], metavar="BATCH=PATH",
                    help="v2 corpus batches recorded by reference only (e.g. run2=%s/run2)" % BRIDGES2_ROOT)
    ap.add_argument("--eval-problems-native", help="problems_v3_native.jsonl (the native evaluation-only problems)")
    ap.add_argument("--eval-problems-v2", help="problems_v2.jsonl (the v2 evaluation problems)")
    ap.add_argument("--eval-episodes", help="multitool only: drc /storage/wxu/corpus/harness/eval/r1")
    ap.add_argument("--provenance", help="JSON deep-merged over KNOWN_PROVENANCE (job ids, dates)")
    ap.add_argument("--stamp-spelling", action="store_true",
                    help="stamp the spelling into rows that lack the key, after checking that their "
                         "instruction_sha1 and action line both imply it")
    ap.add_argument("--accept-instruction-sha1", nargs="*", default=[],
                    help="additional instruction sha1[:12] accepted in rows (e.g. f99b61d776f1, the "
                         "set_parameter instruction before its VERB sentence was reworded)")
    ap.add_argument("--allow-grammar-drift", action="store_true",
                    help="package rows whose served grammar differs from the packaged one; the served "
                         "variants are written as grammar/<family>.served-<n>.txt")
    ap.add_argument("--no-parse-check", action="store_true", help="skip re-parsing every action line")
    ap.add_argument("--parse-check", action="store_true", help="with --check: also re-parse every action line")
    ap.add_argument("--force", action="store_true", help="rebuild a non-empty --out / overwrite a corpus family")
    a = ap.parse_args(argv)
    if a.corpus_batch:
        return check_corpus_batch(a) if a.check else build_corpus_batch(a)
    if not a.out:
        ap.error("--out is required")
    if a.append:
        if a.check:
            ap.error("--append builds; check a batch with --check --out <batch dir>")
        return build_append(a)
    if a.check:
        mp = Path(a.out).resolve() / "manifest.json"
        if mp.is_file():
            m = read_json(mp)
            if m.get("kind") == "batch":
                return check_batch(a, Path(a.out).resolve(), m)
        return check(a)
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
