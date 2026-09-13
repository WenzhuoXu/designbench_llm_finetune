#!/usr/bin/env python
"""Does the distilled policy reach feasibility in a handful of calls?

The teacher spends ~287 FEA analyses per truss problem. The claim under test is
that a model distilled from it reaches comparable feasibility in a handful of
model calls. This script measures that and its controls in one run, on the same
problems, through the same episode loop.

Arms, all speaking the SAME action space -- the tool vocabulary in
da_serial.tool_vocabulary, which is what the corpus supervised:

  model     the checkpoint, served over an OpenAI-compatible endpoint
  base      the un-finetuned base model, same endpoint, same prompt. This is the
            arm that says what SFT bought, as opposed to what the prompt and the
            tool vocabulary bought.
  fsd       fully-stressed design: a sizing pass every turn, the standard
            heuristic. It costs one analysis per turn and no model calls.
  search    the teacher itself, unrestricted. Not a matched arm -- it is the
            ceiling, and its analysis count is the number the claim is against.

Cost is reported in both currencies: FEA analyses (the engineering cost, counted
off the domain's own evaluate) and model calls (the serving cost). An arm that
reaches feasibility by burning analyses inside its own head is not cheap, so both
are recorded per problem, not averaged after the fact.

Problems come from problems_hard, which gen_corpus excludes from truss training
supply for exactly this reason.

Feasibility is binary per problem, so the paired exact sign test on discordant
pairs IS the exact McNemar test. Discordant counts are printed alongside p --
a tiny p over four discordant pairs is not a result.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Repo root from this file's location; DesignBench is a sibling by default
# (the Bridges-2 layout) and can be pointed elsewhere with DESIGNBENCH_ROOT.
PROJECT = Path(__file__).resolve().parents[1]
DESIGNBENCH = Path(os.environ.get("DESIGNBENCH_ROOT", PROJECT.parent / "DesignBench"))
for p in (str(PROJECT), str(DESIGNBENCH), str(PROJECT / "design_agent"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# The FEA counter has to be installed before TrussDomain binds the function, so
# this import block runs before anything from design_agent.
import llm_finetune.envs.truss_env as TE

_RAW_ANALYZE = TE._analyze_truss
_tls = threading.local()


def _counted_analyze(truss, goals):
    _tls.n = getattr(_tls, "n", 0) + 1
    return _RAW_ANALYZE(truss, goals)


TE._analyze_truss = _counted_analyze

import da_domtools  # noqa: F401  registers the truss topology tools
import da_meta      # noqa: F401  per-domain nouns and column headers
import da_serial as S
from da_search import size_pass, rollout
from da_search3 import apply_tool
from gen_corpus import HORIZON, POOL, make, run_instance, system_prompt
from probe26_presentation import sign_test

ARMS = ("model", "base", "fsd", "native", "search", "search_k")


class BudgetExhausted(BaseException):
    """BaseException on purpose: `except Exception` in the tool library must not eat it."""


def budgeted(dom, cap):
    """Cap a domain at `cap` simulations, remembering whether it ever saw a
    feasible state.

    The unrestricted search costs 137-726 analyses per problem. The claim under
    test is that a distilled model matches it in a handful of calls -- so the
    honest reference is not only the unrestricted search but the same search
    held to the same small budget. Without this arm, a distilled model that
    beats fully-stressed design looks impressive when a budget-matched search
    might already do better.

    Feasibility is read off `found` rather than the returned state, because the
    budget can expire mid-rollout on a state the search would have discarded.
    """
    raw = dom.evaluate
    dom.found = False

    def capped(st):
        if st.get("cache") is None and dom.calls >= cap:
            raise BudgetExhausted()
        out = raw(st)
        try:
            if dom.feasible(st):
                dom.found = True
        except Exception:
            pass
        return out

    dom.evaluate = capped
    return dom

# The strongest single-tool heuristic each domain actually offers.
#
# The generic `fsd` arm is a continuous sizing pass. On catalogue that is close
# to meaningless -- sections come from a discrete list, so a continuous rescale
# cannot express a legal move, and the arm scores 0.0000 for reasons that have
# nothing to do with heuristic quality. On cases, sizing to the current worst
# margin ignores that a member must clear EVERY load case at once.
#
# Each domain ships its own honest analogue: CATALOGUE_PASS picks the smallest
# section clearing the margin, SIZE_ENVELOPE sizes against the governing case.
# Those are what the search has to beat for a transfer claim to mean anything.
NATIVE_HEURISTIC = {
    "catalogue": ("CATALOGUE_PASS", {"margin": 1.05}),
    "cases": ("SIZE_ENVELOPE", {"margin": 1.05}),
}


def seed_of(x):
    return zlib.crc32(str(x).encode()) & 0xFFFFFFFF


# ------------------------------------------------------------------ model client

class Client:
    """One OpenAI-compatible endpoint. vLLM and the OpenAI API differ only in
    base_url and key, so the arms share a code path and cannot silently diverge
    in sampling settings."""

    def __init__(self, model, base_url=None, token_file=None, temperature=0.0,
                 max_tokens=512):
        from openai import OpenAI

        if token_file:
            key = Path(token_file).expanduser().read_text().strip()
        else:
            key = "EMPTY"          # vLLM ignores it
        self.cli = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    def __call__(self, messages):
        kw = dict(model=self.model, messages=messages, max_completion_tokens=self.max_tokens)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        r = None
        for attempt in range(5):
            try:
                r = self.cli.chat.completions.create(**kw)
                break
            except Exception as e:
                msg = str(e)
                # A server that rejects a parameter says so in the 400 body, not
                # by raising TypeError. vLLM and the OpenAI API disagree about
                # max_completion_tokens vs max_tokens and about whether an
                # explicit temperature is allowed, so adapt on the message and
                # retry rather than burning the attempt budget on the same call.
                if "max_completion_tokens" in msg and "max_completion_tokens" in kw:
                    kw["max_tokens"] = kw.pop("max_completion_tokens")
                    continue
                if "temperature" in msg and "temperature" in kw:
                    kw.pop("temperature")
                    continue
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        if r is None:
            raise RuntimeError("no response after retries")
        u = getattr(r, "usage", None)
        if u is not None:
            with self._lock:
                self.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                self.completion_tokens += getattr(u, "completion_tokens", 0) or 0
        return r.choices[0].message.content or ""


# ------------------------------------------------------------------ the episode

def run_episode(domain, key, arm, calls, client=None, param=None, analysis_budget=0):
    """One problem, one arm. Returns the row that every comparison is computed from.

    The loop is identical across arms; only the source of the next tool call
    differs. Budget is spent per TURN, so an arm that emits an unparseable or
    inapplicable move pays for it -- otherwise a model that babbles would get
    unlimited retries and the call budget would mean nothing.
    """
    _tls.n = 0
    t0 = time.time()
    dom, st, prm = make(domain, key)
    param = param or prm
    st0 = dom.clone(st)
    # Domain-registered tools. gen_corpus passes this into apply_tool; passing
    # None instead made every domain-specific call raise TypeError inside the
    # tool library, which the episode loop then counted as an ordinary failure.
    try:
        extra = dom.tools() or {}
    except Exception:
        extra = {}

    row = {"domain": domain, "instance": str(key), "arm": arm, "calls_used": 0,
           "parse_fail": 0, "apply_fail": 0, "feasible": False, "turns": 0}

    if arm == "search_k":
        dom = budgeted(dom, analysis_budget)
        try:
            run_instance(dom, st, param, random.Random(seed_of("%s%s" % (domain, key))),
                         steps=HORIZON, nprop=POOL)
        except BudgetExhausted:
            pass
        except Exception:
            pass
        row.update(feasible=bool(getattr(dom, "found", False)), turns=0, calls_used=0,
                   analyses=int(getattr(dom, "calls", 0)), seconds=time.time() - t0)
        return row

    if arm == "search":
        # The teacher, unrestricted: this is the cost the claim is measured against.
        turns, solved = run_instance(dom, st, param, random.Random(seed_of("%s%s" % (domain, key))),
                                     steps=HORIZON, nprop=POOL)
        row.update(feasible=bool(solved), turns=len(turns), calls_used=0,
                   analyses=int(getattr(dom, "calls", 0)), seconds=time.time() - t0)
        return row

    msgs = [{"role": "system", "content": system_prompt(dom, st0)}]
    for _ in range(calls):
        if dom.feasible(st):
            break
        row["turns"] += 1
        obs = S.render_state(dom, st)

        if arm == "fsd":
            # The heuristic control: size every element to the stress it carries.
            nxt = size_pass(dom, st, param)
            if nxt is not None:
                st = nxt
            continue

        if arm == "native":
            # The domain's own heuristic, where it has one stronger than sizing.
            spec = NATIVE_HEURISTIC.get(domain)
            if spec is None:
                nxt = size_pass(dom, st, param)
            else:
                nxt = apply_tool(dom, st, param, spec[0], dict(spec[1]), extra)
            if nxt is not None:
                st = nxt
            continue

        msgs.append({"role": "user", "content": obs})
        try:
            text = client(msgs)
        except Exception:
            row["parse_fail"] += 1
            break
        row["calls_used"] += 1
        msgs.append({"role": "assistant", "content": text})

        parsed = S.parse_tools(text)
        if not parsed:
            row["parse_fail"] += 1
            row.setdefault("unparsed_samples", []).append(text[:300])
            continue
        head, args = parsed[0]
        nxt = apply_tool(dom, st, param, head, args, extra)
        if nxt is None:
            row["apply_fail"] += 1
            continue
        st = nxt

    row["feasible"] = bool(dom.feasible(st))
    row["analyses"] = int(getattr(dom, "calls", 0))
    row["seconds"] = time.time() - t0
    try:
        row["budget_ratio"] = float(dom.budget_ratio(st))
    except Exception:
        pass
    return row


# ------------------------------------------------------------------ reporting

def summarise(rows, arm):
    r = [x for x in rows if x["arm"] == arm]
    if not r:
        return None
    n = len(r)
    feas = sum(x["feasible"] for x in r)
    an = sorted(x.get("analyses", 0) for x in r)
    # The median problem is solved by sizing alone, so the median analysis count
    # says nothing about what the search costs. The mean and the p90 are where
    # the hard tail -- the part the distilled model has to learn -- shows up.
    return {"arm": arm, "n": n, "feasible": feas, "rate": feas / n,
            "mean_analyses": sum(an) / n, "p90_analyses": an[min(n - 1, int(n * 0.9))],
            "max_analyses": an[-1],
            "mean_calls": sum(x["calls_used"] for x in r) / n,
            "parse_fail": sum(x["parse_fail"] for x in r),
            "apply_fail": sum(x["apply_fail"] for x in r)}


def paired(rows, a, b):
    """Exact paired sign test on per-problem feasibility, keyed by instance."""
    ta = {x["instance"]: x["feasible"] for x in rows if x["arm"] == a}
    tb = {x["instance"]: x["feasible"] for x in rows if x["arm"] == b}
    keys = sorted(set(ta) & set(tb))
    va = [int(ta[k]) for k in keys]
    vb = [int(tb[k]) for k in keys]
    # sign_test(x, y) counts positions where y > x as "up", so the control goes
    # first and "up" reads as a win for arm `a`.
    wins, losses, p = sign_test(vb, va)
    return wins, losses, len(keys) - wins - losses, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="truss")
    ap.add_argument("--problems-dir", default=str(DESIGNBENCH / "data/problems_hard"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--calls", type=int, default=8, help="model calls per problem")
    ap.add_argument("--analysis-budget", type=int, default=8,
                    help="simulation cap for the search_k arm")
    ap.add_argument("--arms", default="model,base,fsd,search")
    ap.add_argument("--model", default=None, help="served model name or checkpoint path")
    ap.add_argument("--base-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--openai", action="store_true",
                    help="route the model arm to the OpenAI API instead of a local server")
    ap.add_argument("--token-file", default=str(Path.home() / ".openai_token"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(PROJECT / "results/eval_distilled"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]

    if a.domain == "truss":
        keys = [str(p) for p in sorted(Path(a.problems_dir).glob("*.json"))][: a.n]
    else:
        keys = list(range(a.n))
    print("domain %s | %d problems | %d model calls per problem | arms %s"
          % (a.domain, len(keys), a.calls, ",".join(arms)), flush=True)

    clients = {}
    if "model" in arms:
        if a.openai:
            clients["model"] = Client(a.model, token_file=a.token_file,
                                      temperature=a.temperature)
        else:
            clients["model"] = Client(a.model, base_url=a.base_url,
                                      temperature=a.temperature)
    if "base" in arms:
        clients["base"] = Client(a.base_model, base_url=a.base_url,
                                 temperature=a.temperature)

    jobs = [(k, arm) for arm in arms for k in keys]
    rows = []

    def work(job):
        k, arm = job
        try:
            return run_episode(a.domain, k, arm, a.calls, clients.get(arm),
                               analysis_budget=a.analysis_budget)
        except Exception as e:
            return {"domain": a.domain, "instance": str(k), "arm": arm, "error": repr(e),
                    "feasible": False, "calls_used": 0, "parse_fail": 0, "apply_fail": 0,
                    "turns": 0, "analyses": 0}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, jobs), 1):
            rows.append(r)
            if i % 20 == 0:
                print("  %d/%d episodes" % (i, len(jobs)), flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / ("rows_%s_%s.jsonl" % (a.domain, stamp))
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    errs = [r for r in rows if "error" in r]
    if errs:
        print("\n  %d episodes errored; first: %s" % (len(errs), errs[0]["error"]), flush=True)

    print("\n  arm        n   feasible      rate   analyses mean/p90/max   mean calls   parse/apply")
    for arm in arms:
        s = summarise(rows, arm)
        if s:
            print("  %-8s %3d   %6d   %7.4f   %8.1f /%5d /%5d   %10.2f   %d/%d"
                  % (s["arm"], s["n"], s["feasible"], s["rate"], s["mean_analyses"],
                     s["p90_analyses"], s["max_analyses"], s["mean_calls"],
                     s["parse_fail"], s["apply_fail"]))

    print("\n  paired exact sign tests on per-problem feasibility")
    ref = "model" if "model" in arms else arms[0]
    for b in arms:
        if b == ref:
            continue
        w, l, tie, p = paired(rows, ref, b)
        print("    %-8s vs %-8s  %d-%d discordant (%d tied)   p = %.4g"
              % (ref, b, w, l, tie, p))

    for arm, c in clients.items():
        if c.prompt_tokens or c.completion_tokens:
            print("  %s tokens: %d prompt + %d completion = %d"
                  % (arm, c.prompt_tokens, c.completion_tokens,
                     c.prompt_tokens + c.completion_tokens))
    print("  rows written to %s" % path)
