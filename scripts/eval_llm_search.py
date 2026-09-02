#!/usr/bin/env python
"""LLM proposes, the potential ranks: the framework's architecture, at inference.

The two halves of the system fail in opposite directions, measured on 130 truss
problems:

  * potential-guided search optimises well (median mass 0.620x reference) but
    walks out of the simulator's domain of validity when nothing stops it --
    mechanisms and negative-mass sections were 17% of its "successes";
  * the trained policy never does that (max FOS_buckling among solved designs is
    2.9 across every evaluation cell in the study) but optimises poorly (0.917).

So the policy's engineering prior is the validity constraint, and the potential
is the ranker. This script runs that combination with no retraining: at each turn
the policy proposes K actions, each is executed one step, and the successor with
the highest goal-aligned potential is kept.

K=1 reproduces the ordinary greedy evaluation exactly, so the two arms differ in
one thing only.

It also measures what training has been unable to: the policy's POTENTIAL REGRET
-- how far its first sample sits below the best of its own K, and below the best
of a procedural candidate set at the same state. That is the internalisation
diagnostic the study has been missing.

    python scripts/eval_llm_search.py \\
        --checkpoint checkpoints/grpo/mt_t3_01b_critfeedback_20260616_0900/final \\
        --model-config configs/model/qwen3_14b.yaml \\
        --split-file data/splits/truss_v1_auto.json --split eval \\
        --n-candidates 8 --out results/eval/llm_search_k8
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _sample_actions(model, tokenizer, formatter, messages, k, max_new_tokens, temperature, device):
    """K completions from one context, in a single batched generate."""
    import torch
    prompt_ids = formatter.apply_template(messages, add_generation_prompt=True, tokenize=True)
    tensor = torch.tensor([list(prompt_ids)], dtype=torch.long).to(device)
    attn = torch.ones_like(tensor)   # pad == eos for this tokenizer; be explicit
    with torch.no_grad():
        out = model.generate(
            tensor,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [tokenizer.decode(seq[tensor.shape[1]:], skip_special_tokens=True) for seq in out]


def eval_problem(spec, model, tokenizer, formatter, *, max_steps, n_candidates,
                 temperature, max_new_tokens, alpha, tau, procedural_probe, rng,
                 min_mass=None):
    from llm_finetune.data.processors import designbench_prompt
    from llm_finetune.envs.truss_env import (
        _analyze_truss, _apply_action, _load_truss_and_goals, parse_grammar_action,
    )
    from llm_finetune.training.rl.posterior.lookahead_probe import (
        truss_candidate_actions, truss_param_bounds,
    )
    from llm_finetune.training.rl.posterior.potential import (
        compute_potential_v2, program_from_truss_spec,
    )

    truss, goals = _load_truss_and_goals(spec)
    state = initial_state = _analyze_truss(truss, goals)
    program = program_from_truss_spec(spec, initial_mass=state.get("mass"))
    bounds = truss_param_bounds(spec)
    device = next(model.parameters()).device
    phi = lambda s: compute_potential_v2(s, program, alpha=alpha, tau=tau)

    action_history, parse_success = [], []
    regrets_within, regrets_procedural, ranks = [], [], []
    best = (state["mass"], state) if program.is_feasible(state) else None
    n_sim = 1

    # Whether to keep optimising after the first feasible design must NOT be tied
    # to K, or a K=1-vs-K=8 comparison confounds ranking with extra turns.
    keep_going = (n_candidates > 1) if min_mass is None else min_mass
    for _turn in range(max_steps):
        if program.is_feasible(state) and not keep_going:
            break
        messages = designbench_prompt.build_messages(spec, initial_state, action_history, formatter)
        texts = _sample_actions(model, tokenizer, formatter, messages, n_candidates,
                                max_new_tokens, temperature, device)

        scored = []
        for text in texts:
            parsed = parse_grammar_action(text)
            if parsed is None:
                continue
            trial = copy.deepcopy(truss)
            try:
                _apply_action(trial, parsed)
                nxt = _analyze_truss(trial, goals)
            except Exception:
                continue
            n_sim += 1
            if not program.is_valid(nxt):
                continue          # the policy's own prior rarely produces these
            scored.append((phi(nxt), parsed, trial, nxt, text))

        parse_success.append(bool(scored))
        if not scored:
            # See eval_checkpoint: recording unparsed text as the action poisons
            # every later turn's context.
            action_history.append({
                "action": "",
                "thinking": "No valid <action> was produced last turn. Emit exactly one "
                            "<action>...</action> block using the grammar above.",
                "fea_result": state,
            })
            continue

        order = sorted(scored, key=lambda item: item[0], reverse=True)
        chosen = order[0]
        # How far did the policy's FIRST sample sit below the best of its own K?
        first = next((s for s in scored if s[4] is texts[0]), scored[0])
        regrets_within.append(order[0][0] - first[0])
        ranks.append(sum(1 for s in scored if s[0] > first[0] + 1e-12) / len(scored))

        if procedural_probe:
            proc = truss_candidate_actions(truss, bounds, max_candidates=48, rng=rng)
            proc_best = -float("inf")
            for action in proc:
                trial = copy.deepcopy(truss)
                try:
                    _apply_action(trial, action)
                    nxt = _analyze_truss(trial, goals)
                except Exception:
                    continue
                if program.is_valid(nxt):
                    proc_best = max(proc_best, phi(nxt))
            if proc_best > -float("inf"):
                regrets_procedural.append(proc_best - first[0])

        _, parsed, truss, state, text = chosen
        thinking, _ = formatter.extract_thinking(text)
        action_history.append({"action": parsed, "thinking": thinking, "fea_result": state})
        if program.is_feasible(state) and (best is None or state["mass"] < best[0]):
            best = (state["mass"], state)

    final = best[1] if best else state
    return {
        "problem_id": spec.get("problem_id"),
        "reaches_solution": best is not None,
        "n_steps": len(action_history),
        "final_mass": final.get("mass"),
        "mass_ref": program.objective_ref,
        "mass_ratio": (final.get("mass") / program.objective_ref) if program.objective_ref else None,
        "final_fos_buckling": final.get("fos_buckling"),
        "final_fos_yielding": final.get("fos_yielding"),
        "final_deflection": final.get("deflection"),
        "initial_mass": initial_state.get("mass"),
        "grammar_success_rate": sum(parse_success) / max(len(parse_success), 1),
        "n_simulator_calls": n_sim,
        "regret_within_samples": st.mean(regrets_within) if regrets_within else None,
        "regret_vs_procedural": st.mean(regrets_procedural) if regrets_procedural else None,
        "rank_frac_of_first_sample": st.mean(ranks) if ranks else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="configs/model/qwen3_14b.yaml")
    ap.add_argument("--problems-dir", default="/ocean/projects/mch250030p/wxu7/DesignBench/data/problems")
    ap.add_argument("--split-file", default="data/splits/truss_v1_auto.json")
    ap.add_argument("--split", default="eval", choices=["train", "eval"])
    ap.add_argument("--n-candidates", type=int, default=8,
                    help="Actions proposed per turn. 1 reproduces the ordinary greedy eval.")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--min-mass", dest="min_mass", action="store_true", default=None,
                    help="Keep optimising after the first feasible design. Defaults to on for "
                         "K>1 and off for K=1; set explicitly to make the two comparable.")
    ap.add_argument("--stop-on-feasible", dest="min_mass", action="store_false",
                    help="Stop at the first feasible design.")
    ap.add_argument("--procedural-probe", action="store_true",
                    help="Also score a procedural candidate set at each state (regret diagnostic).")
    ap.add_argument("--max-problems", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from scripts.eval_checkpoint import _load_model_config
    from llm_finetune.data.processors.chat_formatter import ChatFormatter
    from llm_finetune.models.loader import load_model_and_tokenizer

    model_cfg = _load_model_config(args.model_config)
    # Never load with gradient checkpointing for inference: it forces
    # use_cache=False, so every turn re-computes the whole context and a
    # 20-turn rollout takes tens of minutes. eval_checkpoint.py:230 does the
    # same. (The disable below is belt-and-braces for adapters.)
    # A queued SLURM job carries its parameters in its environment, and SLURM
    # does not retain that environment for a PENDING job -- so a run that does
    # not print its own arguments is unauditable after the fact. Print them
    # first, before anything expensive, so the .out file identifies the arm.
    log.info("ARGS %s", json.dumps(vars(args), sort_keys=True, default=str))
    log.info("RESOLVED keep_going(min_mass)=%s  (None => n_candidates>1)",
             bool(args.min_mass) if args.min_mass is not None else (args.n_candidates > 1))

    model_cfg.use_gradient_checkpointing = False
    base_model_id = model_cfg.model_name_or_path
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    ckpt = Path(args.checkpoint)
    if (ckpt / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ckpt))
        log.info(f"loaded LoRA adapter from {ckpt}")
    model.eval()
    # The loader turns on gradient checkpointing for training, which forces
    # use_cache=False and makes generation crawl (no KV cache => quadratic
    # re-computation every turn). Inference wants the opposite.
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    base = getattr(model, "base_model", model)
    for obj in {model, base, getattr(base, "model", base)}:
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            cfg.use_cache = True
    log.info("generation: gradient checkpointing off, KV cache on")
    formatter = ChatFormatter.from_model_id(base_model_id, tokenizer)

    wanted = set(json.load(open(args.split_file))[args.split]) if args.split_file else None
    specs, seen = [], set()
    for f in sorted(Path(args.problems_dir).glob("*.json")):
        try:
            spec = json.load(open(f))
        except Exception:
            continue
        if not (isinstance(spec, dict) and "topology" in spec):
            continue
        pid = spec.setdefault("problem_id", f.stem)
        if pid in seen or (wanted is not None and pid not in wanted):
            continue
        seen.add(pid)
        specs.append(spec)
    if args.max_problems:
        specs = specs[: args.max_problems]
    log.info(f"{len(specs)} problems on split '{args.split}', K={args.n_candidates}")

    rng = random.Random(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "eval_results.json"

    def _write(results):
        """Persist after every problem: a wall-limit kill must not lose the run."""
        solved = [r for r in results if r["reaches_solution"]]
        ratios = [r["mass_ratio"] for r in solved if r["mass_ratio"]]
        def _m(key):
            vals = [r[key] for r in results if r.get(key) is not None]
            return st.mean(vals) if vals else None
        agg = {
            "n_problems": len(results),
            "n_candidates": args.n_candidates,
            "min_mass": bool(args.min_mass) if args.min_mass is not None else (args.n_candidates > 1),
            "temperature": args.temperature,
            "complete": len(results) == len(specs),
            "feasibility_rate": len(solved) / max(len(results), 1),
            "median_mass_ratio": st.median(ratios) if ratios else None,
            "mean_mass_ratio": st.mean(ratios) if ratios else None,
            "mean_grammar_success": _m("grammar_success_rate"),
            "mean_simulator_calls": _m("n_simulator_calls"),
            "mean_regret_within_samples": _m("regret_within_samples"),
            "mean_regret_vs_procedural": _m("regret_vs_procedural"),
            "mean_rank_frac_of_first_sample": _m("rank_frac_of_first_sample"),
        }
        json.dump({"aggregate": agg, "per_problem": results},
                  open(results_path, "w"), indent=2)
        return agg

    results = []
    for i, spec in enumerate(specs, 1):
        r = eval_problem(spec, model, tokenizer, formatter,
                         max_steps=args.max_steps, n_candidates=args.n_candidates,
                         temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                         alpha=args.alpha, tau=args.tau,
                         procedural_probe=args.procedural_probe, rng=rng,
                         min_mass=args.min_mass)
        results.append(r)
        _write(results)
        log.info(f"[{i}/{len(specs)}] {r['problem_id']}: feasible={r['reaches_solution']} "
                 f"mass/ref={r['mass_ratio'] and round(r['mass_ratio'],3)} "
                 f"regret={r['regret_within_samples'] and round(r['regret_within_samples'],4)}")

    agg = _write(results)
    log.info(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
