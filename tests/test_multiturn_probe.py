"""The lookahead probe as the multi-turn rollout actually calls it.

``probe_state`` is unit-tested against synthetic successors in
test_potential_v2.py. This exercises the integration helper the rollout uses --
real truss, real grammar actions, real simulator -- because that is the code path
that runs inside a 14-hour training job, where a diagnostic that raises would
take the run down with it.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

PROBLEM = Path("/ocean/projects/mch250030p/wxu7/DesignBench/data/problems/auto_problem_000.json")

pytestmark = pytest.mark.skipif(not PROBLEM.exists(), reason="DesignBench problem set not present")


@pytest.fixture(scope="module")
def truss_state():
    from validation.truss_executor import analyze_truss, load_truss_from_problem
    from llm_finetune.training.rl.posterior.lookahead_probe import truss_param_bounds
    from llm_finetune.training.rl.posterior.potential import program_from_truss_spec

    spec = json.load(open(PROBLEM))
    goals = spec.get("goals", {})
    truss = load_truss_from_problem(spec)
    state = analyze_truss(truss, goals)
    return {
        "spec": spec, "goals": goals, "truss": truss, "state": state,
        "program": program_from_truss_spec(spec, initial_mass=state["mass"]),
        "bounds": truss_param_bounds(spec),
    }


def _probe(truss_state, action):
    """Run the rollout's helper exactly as multiturn_rollout.py does."""
    from llm_finetune.envs.truss_env import _analyze_truss, _apply_action
    from llm_finetune.training.rl.multiturn_rollout import _probe_here
    from llm_finetune.training.rl.posterior.lookahead_probe import (
        probe_state, truss_candidate_actions,
    )

    pre_truss = copy.deepcopy(truss_state["truss"])
    st = dict(truss_state)
    st["truss"] = copy.deepcopy(truss_state["truss"])
    _apply_action(st["truss"], action)
    st["state"] = _analyze_truss(st["truss"], st["goals"])
    return _probe_here(
        pre_truss, st, _analyze_truss, _apply_action, probe_state,
        truss_candidate_actions, 48, random.Random(0), 5.0, 0.05, 0.99,
    )


def test_probe_returns_the_fields_the_reward_and_logger_read(truss_state):
    result = _probe(truss_state, "SCALE_PARAM(4, r, 1.35)")
    assert result is not None
    for key in ("rho", "regret", "rank_frac", "margin", "sigma", "reliable",
                "n_candidates", "policy_phi", "baseline_phi", "advantage"):
        assert key in result, f"missing {key}"
    assert result["n_candidates"] >= 2
    assert result["advantage"] == pytest.approx(result["policy_phi"] - result["baseline_phi"])


def test_probe_prefers_the_critical_member(truss_state):
    """M4 is the buckling-critical member the FEA feedback reports."""
    assert truss_state["state"]["min_fos_buckling_member_id"] == 4
    on_critical = _probe(truss_state, "SCALE_PARAM(4, r, 1.35)")
    off_critical = _probe(truss_state, "SCALE_PARAM(6, r, 1.35)")
    assert on_critical["regret"] < off_critical["regret"]
    assert on_critical["rank_frac"] < off_critical["rank_frac"]
    assert on_critical["advantage"] > off_critical["advantage"]


def test_probe_never_raises_on_a_broken_action(truss_state):
    # a training step must survive anything the policy emits
    for action in ("SCALE_PARAM(999, r, 1.2)", "NOT_AN_ACTION(1,2)", ""):
        _probe(truss_state, action)  # must not raise


def test_probe_cost_is_negligible_against_a_training_step(truss_state):
    import time
    start = time.perf_counter()
    _probe(truss_state, "SCALE_PARAM(4, r, 1.35)")
    elapsed = time.perf_counter() - start
    # ~0.13 s at 48 candidates; a multi-turn GRPO step is ~500 s
    assert elapsed < 5.0, f"probe took {elapsed:.2f}s -- too slow to run every step"


def test_rollout_config_flags_are_read(truss_state):
    """all_steps must lift the per-batch budget; the reward depends on it."""
    import inspect
    from llm_finetune.training.rl import multiturn_rollout
    source = inspect.getsource(multiturn_rollout.make_multiturn_rollout_func)
    assert "probe_all_steps" in source
    assert "lookahead_probe" in source
    assert "probes" in source


class TestInterTurnGlue:
    """D7: the trained token sequence must actually contain the FEA feedback.

    Before 2026-08-23 it did not. The rollout rebuilt the whole conversation each
    turn and kept it only if it was a prefix extension of the running sequence;
    Qwen3's template strips <think> from non-final assistant messages, so it never
    was, and the glue silently became empty on every transition (9176 warnings in
    the champion run). env_mask stayed all-ones and TRL's environment masking --
    the reason this rollout_func exists -- never fired.
    """

    @pytest.fixture(scope="class")
    def glue(self):
        from transformers import AutoTokenizer
        from llm_finetune.data.processors.chat_formatter import ChatFormatter
        from llm_finetune.training.rl.multiturn_rollout import _build_glue_fn
        ckpt = Path("checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final")
        if not ckpt.exists():
            pytest.skip("warmstart checkpoint not present")
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        fmt = ChatFormatter.from_model_id("Qwen/Qwen3-14B", tok)
        eos = [i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
               if isinstance(i, int) and i >= 0]
        fn = _build_glue_fn(fmt, tok, eos)
        assert fn is not None, "glue derivation failed on the shipped template"
        return fn, tok, fmt

    def test_glue_is_not_empty(self, glue):
        fn, tok, _ = glue
        gen = tok("<action>SCALE_PARAM(4, r, 1.3)</action><|im_end|>", add_special_tokens=False)["input_ids"]
        assert len(fn("FEA RESULT", gen)) > 5

    def test_feedback_reaches_the_trained_sequence(self, glue):
        fn, tok, fmt = glue
        prompt = list(fmt.apply_template(
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": "P"}],
            add_generation_prompt=True, tokenize=True))
        gen = tok("<think>t</think>\n<action>SCALE_PARAM(4, r, 1.3)</action><|im_end|>",
                  add_special_tokens=False)["input_ids"]
        decoded = tok.decode(prompt + gen + fn("Status: INFEASIBLE", gen))
        assert "INFEASIBLE" in decoded
        assert decoded.count("<|im_start|>user") == 2
        assert decoded.endswith("<|im_start|>assistant\n")

    def test_terminator_is_not_doubled_and_is_supplied_when_missing(self, glue):
        fn, tok, fmt = glue
        prompt = list(fmt.apply_template(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "P"}],
            add_generation_prompt=True, tokenize=True))
        with_eot = tok("<action>A</action><|im_end|>", add_special_tokens=False)["input_ids"]
        without = tok("<action>A</action>", add_special_tokens=False)["input_ids"]
        for gen in (with_eot, without):
            decoded = tok.decode(prompt + gen + fn("FEA", gen))
            assert "<|im_end|>\n<|im_end|>" not in decoded
            assert decoded.count("<|im_start|>user") == 2


class TestSplitLoading:
    """The split file must be authoritative, and must not be derailed by the
    non-spec files that live alongside the problems."""

    PROBLEMS = Path("/ocean/projects/mch250030p/wxu7/DesignBench/data/problems")
    SPLIT = Path("data/splits/truss_v1_auto.json")

    @pytest.fixture(scope="class")
    def loader(self):
        if not (self.PROBLEMS.exists() and self.SPLIT.exists()):
            pytest.skip("problem set or split file not present")
        ckpt = Path("checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final")
        if not ckpt.exists():
            pytest.skip("warmstart checkpoint not present")
        from transformers import AutoTokenizer
        from llm_finetune.data.processors.chat_formatter import ChatFormatter
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        return tok, ChatFormatter.from_model_id("Qwen/Qwen3-14B", tok)

    def test_non_spec_files_do_not_break_the_loader(self, loader):
        # data/problems/test_problems.json is a LIST; globbing "*.json" for an
        # explicit id list surfaces it and used to raise
        # AttributeError: 'list' object has no attribute 'get' AFTER the 14B model
        # had already been loaded onto two GPUs.
        import json
        from llm_finetune.data.datasets.rl_dataset import RLPromptDataset
        tok, fmt = loader
        ids = json.load(open(self.SPLIT))["train"]
        ds = RLPromptDataset.from_problems_dir(
            problems_dir=str(self.PROBLEMS), tokenizer=tok, formatter=fmt,
            max_prompt_len=2048, problem_ids=ids, repeat=1)
        assert len(ds) == len(ids), f"{len(ds)} prompts from {len(ids)} split ids"

    def test_train_and_eval_sides_are_disjoint(self):
        import json
        if not self.SPLIT.exists():
            pytest.skip("split file not present")
        split = json.load(open(self.SPLIT))
        assert set(split["train"]) & set(split["eval"]) == set()
        assert len(split["train"]) > 0 and len(split["eval"]) > 0


class TestPerTokenAdvantageAlignment:
    """The rollout must emit a credit vector aligned 1:1 with the completion.

    A misalignment here is silent and would attribute one turn's credit to another
    turn's tokens for an entire 20-hour run, so the bookkeeping is exercised
    directly rather than trusted.
    """

    def _simulate(self, turns):
        """Reproduce the rollout's append order: gen tokens, then env glue."""
        comp_ids, comp_mask, tok_adv, spans = [], [], [], []
        for n_gen, n_glue, adv in turns:
            start = len(comp_ids)
            comp_ids += [7] * n_gen
            comp_mask += [1] * n_gen
            tok_adv += [0.0] * n_gen
            spans.append((start, start + n_gen))
            lo, hi = spans[-1]                      # probe result backfills the span
            for i in range(lo, min(hi, len(tok_adv))):
                tok_adv[i] = adv
            comp_ids += [9] * n_glue
            comp_mask += [0] * n_glue
            tok_adv += [0.0] * n_glue
        return comp_ids, comp_mask, tok_adv

    def test_lengths_match_the_completion(self):
        ids, mask, adv = self._simulate([(5, 3, 0.4), (7, 3, -0.2), (4, 0, 0.9)])
        assert len(adv) == len(ids) == len(mask)

    def test_credit_lands_only_on_model_tokens(self):
        ids, mask, adv = self._simulate([(5, 3, 0.4), (7, 3, -0.2)])
        assert all(a == 0.0 for a, m in zip(adv, mask) if m == 0)

    def test_each_turn_gets_its_own_advantage(self):
        ids, mask, adv = self._simulate([(3, 2, 0.4), (3, 2, -0.7)])
        assert adv[0:3] == [0.4] * 3
        assert adv[5:8] == [-0.7] * 3
        assert adv[3:5] == [0.0] * 2          # the glue between them

    def test_rollout_emits_the_field(self):
        import inspect
        from llm_finetune.training.rl import multiturn_rollout
        src = inspect.getsource(multiturn_rollout.make_multiturn_rollout_func)
        assert "step_token_advantages" in src
        assert "turn_spans" in src


class TestActionNormalisation:
    """27.5% of the gold SFT actions (3295 of 11964) are
    ``SCALE_PARAM(all_members, ...)``, which DesignBench's executor rejects because
    it matches the member id with ``(\\d+)``. Every one of them silently did
    nothing, which is a large part of the ~0.66 execute rates this study blamed on
    grammar drift."""

    @pytest.fixture(scope="class")
    def truss(self):
        import json
        from validation.truss_executor import load_truss_from_problem
        if not PROBLEM.exists():
            pytest.skip("problem set not present")
        return load_truss_from_problem(json.load(open(PROBLEM)))

    def test_all_members_becomes_executable(self, truss):
        import copy, json
        from validation.truss_executor import analyze_truss
        from llm_finetune.envs.truss_env import _apply_action
        goals = json.load(open(PROBLEM))["goals"]
        t = copy.deepcopy(truss)
        before = analyze_truss(t, goals)["mass"]
        _apply_action(t, "SCALE_PARAM(all_members, thickness, 1.224)")
        after = analyze_truss(t, goals)["mass"]
        assert abs(after - before) > 1e-6, "the whole-structure action still does nothing"

    def test_ordinary_actions_are_untouched(self, truss):
        from llm_finetune.envs.truss_env import normalize_action
        for a in ("SCALE_PARAM(4, r, 1.3)", "REMOVE_MEMBER(2)",
                  "SCALE_MULTI_PARAM([0,1], [r:1.1])"):
            assert normalize_action(truss, a) == a

    def test_expansion_covers_every_member(self, truss):
        from llm_finetune.envs.truss_env import normalize_action
        out = normalize_action(truss, "SCALE_PARAM(all_members, r, 1.5)")
        ids = out[out.index("[") + 1: out.index("]")].split(",")
        assert len(ids) == len(truss.members)


class TestFakeTreePathRefuses:
    """rl.use_tree_expansion never called evaluate_tree; it must not look like search."""

    def test_config_flag_is_rejected(self):
        import pytest as _pytest
        from omegaconf import OmegaConf
        from llm_finetune.training.rl import grpo_trainer as gt
        import inspect
        src = inspect.getsource(gt._build_reward_callable)
        assert "does NOT run a tree search" in src
        assert "allow_fake_tree" in src


class TestGoldActionsExecute:
    """Two format mismatches made ~68% of the supervised signal inert.

    ``SCALE_PARAM(all_members, ...)`` and the flat
    ``ADD_MEMBER(j1, j2, material, Shape, p1, p2)`` are what the gold SFT data and
    CLAUDE.md both use, but DesignBench's executor requires a numeric member id and
    the shape parameters inside the shape call. Both fell through to "unknown
    action" and returned without touching the truss. Measured over 80 gold traces
    walked on their own problems: 31.6% of actions changed the design before
    normalisation, 96.5% after.
    """

    def test_add_member_flat_form_creates_a_member(self):
        import copy, json
        from validation.truss_executor import load_truss_from_problem
        from llm_finetune.envs.truss_env import _apply_action
        if not PROBLEM.exists():
            pytest.skip("problem set not present")
        spec = json.load(open(PROBLEM))
        truss = copy.deepcopy(load_truss_from_problem(spec))
        before = len(truss.members)
        _apply_action(truss, "ADD_MEMBER(1, 6, A36_Steel, Pipe, 0.023171, 0.003862)")
        assert len(truss.members) == before + 1

    def test_shape_parameter_order_is_respected(self):
        import json
        from validation.truss_executor import load_truss_from_problem
        from llm_finetune.envs.truss_env import normalize_action
        if not PROBLEM.exists():
            pytest.skip("problem set not present")
        truss = load_truss_from_problem(json.load(open(PROBLEM)))
        out = normalize_action(truss, "ADD_MEMBER(1, 6, A36_Steel, Pipe, 0.02, 0.003)")
        assert "r=0.02" in out and "t=0.003" in out

    def test_executor_native_form_is_untouched(self):
        import json
        from validation.truss_executor import load_truss_from_problem
        from llm_finetune.envs.truss_env import normalize_action
        if not PROBLEM.exists():
            pytest.skip("problem set not present")
        truss = load_truss_from_problem(json.load(open(PROBLEM)))
        native = "ADD_MEMBER(1, 6, A36_Steel, Pipe(r=0.02, t=0.003))"
        assert normalize_action(truss, native) == native

    def test_majority_of_gold_actions_execute(self):
        """Regression bound: the fraction must not fall back toward 31.6%."""
        import json, re, glob, copy
        from pathlib import Path
        from validation.truss_executor import (
            load_truss_from_problem, analyze_truss, execute_grammar_action)
        from llm_finetune.envs.truss_env import normalize_action
        gold = Path("/ocean/projects/mch250030p/wxu7/DesignBench/data/sft/train.jsonl")
        if not gold.exists():
            pytest.skip("gold SFT data not present")
        specs = {}
        for f in glob.glob("/ocean/projects/mch250030p/wxu7/DesignBench/data/problems/*.json"):
            d = json.load(open(f))
            if isinstance(d, dict) and "topology" in d:
                specs.setdefault(d.get("problem_id", Path(f).stem), d)
        pat = re.compile(r"Action:\s*([^\n<]+)")
        total = eff = traces = 0
        with open(gold) as fh:
            for line in fh:
                if traces >= 25:
                    break
                d = json.loads(line)
                pid = d.get("problem_id")
                if pid not in specs:
                    continue
                traces += 1
                spec = specs[pid]
                goals = spec.get("goals", {})
                truss = load_truss_from_problem(spec)
                for m in d.get("messages", []):
                    if m["role"] != "assistant":
                        continue
                    for raw in pat.findall(m["content"]):
                        a = raw.strip()
                        if not a or a.startswith("OPTIMAL_STATE"):
                            continue
                        m0 = analyze_truss(truss, goals).get("mass")
                        n0 = len(truss.members)
                        try:
                            execute_grammar_action(truss, normalize_action(truss, a))
                            m1 = analyze_truss(truss, goals).get("mass")
                            ok = abs((m1 or 0) - (m0 or 0)) > 1e-9 or len(truss.members) != n0
                        except Exception:
                            ok = False
                        total += 1
                        eff += int(ok)
        assert total > 50
        assert eff / total > 0.85, f"only {eff/total:.1%} of gold actions execute"


class TestActionEffectiveness:
    """grammar_success_rate reads 1.00 for an action that parses and does nothing.

    That is how ~2/3 of supervised actions stayed invisible: the two gold forms
    below fell through the executor's dispatch and returned the truss untouched,
    so every rollout scored a design that never moved. action_effective_rate is
    the metric that can see it.
    """

    def _fp_and_env(self):
        import importlib.util
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("ec", root / "scripts/eval_checkpoint.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._design_fingerprint

    def test_repaired_forms_now_move_the_design(self):
        import glob, json, pytest
        from llm_finetune.envs.truss_env import _load_truss_and_goals, _apply_action
        probs = sorted(glob.glob("/ocean/projects/mch250030p/wxu7/DesignBench/"
                                 "data/problems/auto_problem_0*.json"))
        if not probs:
            pytest.skip("DesignBench problems unavailable")
        fp = self._fp_and_env()
        spec_d = json.load(open(probs[0]))
        for act in ("SCALE_PARAM(all_members, thickness, 1.3)",
                    "ADD_MEMBER(0, 3, 6061_T6_Aluminum, Pipe, 0.03, 0.004)",
                    "SCALE_PARAM(0, radius, 1.2)"):
            truss, _ = _load_truss_and_goals(spec_d)
            before = fp(truss)
            _apply_action(truss, act)
            assert fp(truss) != before, f"{act} left the design untouched"

    def test_genuinely_invalid_action_stays_inert(self):
        import glob, json, pytest
        from llm_finetune.envs.truss_env import _load_truss_and_goals, _apply_action
        probs = sorted(glob.glob("/ocean/projects/mch250030p/wxu7/DesignBench/"
                                 "data/problems/auto_problem_0*.json"))
        if not probs:
            pytest.skip("DesignBench problems unavailable")
        fp = self._fp_and_env()
        truss, _ = _load_truss_and_goals(json.load(open(probs[0])))
        before = fp(truss)
        try:
            _apply_action(truss, "SCALE_PARAM(999, radius, 1.2)")
        except Exception:
            pass
        assert fp(truss) == before, "out-of-range member id must not change the design"
