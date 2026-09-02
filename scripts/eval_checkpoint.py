"""
Checkpoint evaluation: load a fine-tuned model and run DesignBench benchmark.

Uses DesignBench's ValidationRunner to evaluate the model on all 10 benchmark
problems, computing feasibility rate, FOS improvement, mass reduction, and
grammar compliance.

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint checkpoints/sft/qwen3_sft/final \
        --model-config configs/model/qwen3_14b.yaml \
        --problems-dir /ocean/projects/mch250030p/wxu7/DesignBench/data/problems \
        --output-dir results/eval/qwen3_sft_001

    # Or evaluate base model (no checkpoint):
    python scripts/eval_checkpoint.py --model-config configs/model/qwen3_14b.yaml
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

_MT_DBG = {"n": 0}  # logs a few sample CoT turns across the eval for inspection



def _design_fingerprint(truss):
    """Cheap structural identity: member count plus every member's shape params.

    Compared before/after an action to tell "the executor applied it" from "the
    executor silently ignored it". Falls back to repr() for non-truss designs so
    this stays domain-agnostic.
    """
    try:
        members = getattr(truss, "members", None)
        if members is None:
            return repr(truss)
        out = [len(members)]
        for m in (members.values() if isinstance(members, dict) else members):
            shape = getattr(m, "shape", None)
            params = getattr(shape, "parameters", None) or getattr(shape, "params", None)
            out.append(tuple(sorted(params.items())) if isinstance(params, dict) else repr(shape))
        return tuple(out)
    except Exception:  # noqa: BLE001
        return repr(truss)


def _load_model_config(path):
    """Load a model config YAML with its Hydra ``defaults:`` list resolved.

    ``OmegaConf.load()`` alone does NOT process Hydra's ``defaults:`` list, so a
    config that inherits from ``base.yaml`` (e.g. ``qwen3_14b.yaml``) ends up
    missing every base key (``torch_dtype``, ``attn_implementation``,
    ``device_map``, ...). That is exactly what crashed the May-27 eval batch
    with ``Missing key torch_dtype``. This merges each default — resolved
    relative to the config's own directory — *under* the named config, which
    mimics Hydra composition for the common single-level ``defaults: [base]``
    case (and recurses for nested defaults). ``_self_`` and missing files are
    skipped.
    """
    from omegaconf import OmegaConf

    path = Path(path)
    cfg = OmegaConf.load(path)
    defaults = cfg.pop("defaults", []) or []
    merged = OmegaConf.create({})
    for d in defaults:
        if not isinstance(d, str) or d == "_self_":
            continue
        dep = path.parent / f"{d}.yaml"
        if dep.exists():
            merged = OmegaConf.merge(merged, _load_model_config(dep))
    merged = OmegaConf.merge(merged, cfg)
    return merged


def _extract_action_followup(model, tokenizer, formatter, messages, cot_completion, device):
    """Parse the action 'another way' for reasoning models that produce excellent
    CoT but won't emit the <action> tag inline. We keep the full CoT and ask the
    SAME model, in one focused follow-up, to commit its decision to a single
    grammar action. Returns (parsed_action_or_None, followup_text)."""
    import torch
    followup = list(messages) + [
        {"role": "assistant", "content": cot_completion},
        {"role": "user", "content":
         "Based on your reasoning above, output EXACTLY ONE grammar action and nothing else, "
         "in the form:\n<action>GRAMMAR_ACTION(...)</action>"},
    ]
    ids = formatter.apply_template(followup, add_generation_prompt=True, tokenize=True)
    t = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=1024, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    from llm_finetune.envs.truss_env import parse_grammar_action
    resp = tokenizer.decode(out[0][t.shape[1]:], skip_special_tokens=True)
    return parse_grammar_action(resp), resp


def _eval_problem_multiturn(spec, model, tokenizer, formatter, problem_text,
                            max_steps, temperature, max_new_tokens, few_shot=False,
                            extract_action=False):
    """Multi-turn rollout — the protocol the model was trained for and that the
    master doc (§3, horizon H≈9) assumes: generate ONE action, apply it via FEA,
    feed the result back as the next user turn, repeat until feasible or
    max_steps.

    This mirrors TrussRolloutEnv.run_completion's internal truss chaining
    (_load_truss_and_goals / _apply_action / _analyze_truss) but interleaves a
    fresh model generation between every action. The legacy single-turn path
    (run_completion on one big completion) forces the model to emit a whole
    trajectory in one shot — out-of-distribution for a model SFT'd one-action-
    per-turn — then guesses step boundaries by splitting on "\\n\\n", which
    yielded ~4% grammar success and ~0% feasibility in the S0 baseline.
    """
    import torch
    from llm_finetune.data.processors import designbench_prompt
    from llm_finetune.envs.truss_env import (
        _load_truss_and_goals, _apply_action, _analyze_truss, parse_grammar_action,
        normalize_action as _normalize_action,
    )

    truss, goals = _load_truss_and_goals(spec)
    initial_state = _analyze_truss(truss, goals)
    state = initial_state
    device = next(model.parameters()).device

    action_history = []
    parse_success = []
    action_effective = []
    action_rewritten = []

    for _t in range(max_steps):
        if state.get("is_feasible", False):
            break
        messages = designbench_prompt.build_messages(spec, initial_state, action_history, formatter, few_shot=few_shot)
        prompt_ids = formatter.apply_template(messages, add_generation_prompt=True, tokenize=True)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(output[0][input_tensor.shape[1]:], skip_special_tokens=True)
        thinking, _ = formatter.extract_thinking(completion)
        parsed = parse_grammar_action(completion)
        followup = ""
        if parsed is None and extract_action:  # reasoning model: extract action from its CoT
            parsed, followup = _extract_action_followup(
                model, tokenizer, formatter, messages, completion, device)
        # keep the FULL CoT as the reasoning for this turn (this is what we distill)
        thinking = completion.strip()
        if _MT_DBG["n"] < 4:  # log a few sample CoT turns for inspection
            _MT_DBG["n"] += 1
            log.info(f"[CoT SAMPLE {spec.get('problem_id')} turn={_t} ntok={len(completion.split())} "
                     f"parsed={parsed!r}]\n>>> {completion[:500]}\n... [followup-extract]: {followup[:200]}")
        if parsed is None:
            parse_success.append(False)
            # Do NOT record the unparsed text as this turn's action. It is rendered
            # back into the next prompt as an assistant turn full of garbage, so one
            # parse failure poisons the rest of the episode -- which is why zero-parse
            # problems show n_simulator_calls=1 and never recover. Give the model a
            # corrective observation instead and leave the history well-formed.
            action_history.append({
                "action": "",
                "fea_result": state,
                "thinking": "No valid <action> was produced last turn. Emit exactly one "
                            "<action>...</action> block using the grammar above.",
            })
            continue
        try:
            # Would the pre-fix executor have silently ignored this? normalize_action
            # rewrites exactly the two forms it could not dispatch, so "text changed"
            # is precisely "this action used to be a no-op".
            action_rewritten.append(_normalize_action(truss, parsed) != parsed)
            _before = _design_fingerprint(truss)
            truss = _apply_action(truss, parsed)
            state = _analyze_truss(truss, goals)
            parse_success.append(True)
            # An action that parses but leaves the design bit-identical did nothing.
            # grammar_success_rate cannot see this: it reads 1.00 while the episode
            # stands still. Two gold action forms no-oped this way until the
            # normalisation in truss_env.normalize_action.
            action_effective.append(_design_fingerprint(truss) != _before)
        except Exception as e:  # noqa: BLE001
            log.debug(f"action {parsed!r} failed: {e}")
            parse_success.append(False)
            action_effective.append(False)
        action_history.append({"action": parsed, "fea_result": state, "thinking": thinking})

    # Validity-gated metrics, so a policy's number is comparable to the search
    # ladder's. The simulator's own is_feasible accepts states outside its domain
    # of validity: on the mass-constrained family, which states no deflection
    # limit, a truss thinned into a mechanism reports huge FOS, low mass and
    # FEASIBLE with a deflection of 1e14 m. No trained policy has been observed to
    # do this, but the metric should not depend on that continuing to hold.
    # mass_ratio normalises by the problem's own reference mass.
    from llm_finetune.training.rl.posterior.potential import program_from_truss_spec
    program = program_from_truss_spec(spec, initial_mass=initial_state.get("mass"))
    mass = state.get("mass", 0) or 0.0
    return {
        "problem_id": spec.get("problem_id", ""),
        "reaches_solution": bool(state.get("is_feasible", False)),
        # Conservative by construction: the simulator's own verdict AND the
        # validity envelope. Using the program's verdict alone can be LESS strict
        # than the simulator, because TrussRolloutEnv sanitises non-finite FEA
        # output (a diverged deflection is clamped to a sentinel) and marks the
        # state infeasible, while a problem with no deflection goal has no
        # constraint for the program to catch it on. Two of 34 T6 designs landed
        # exactly there, which is how a "validity-gated" rate came out ABOVE the
        # raw one -- an impossible relationship that gave the bug away.
        "valid_feasible": bool(state.get("is_feasible", False)) and bool(program.is_valid(state)),
        "state_is_valid": bool(program.is_valid(state)),
        "n_steps": len(action_history),
        "final_fos_buckling": state.get("fos_buckling", 0),
        "final_fos_yielding": state.get("fos_yielding", 0),
        "final_mass": mass,
        "final_deflection": state.get("deflection", 0),
        "mass_ref": program.objective_ref,
        "mass_ratio": (mass / program.objective_ref) if program.objective_ref else None,
        "initial_fos_buckling": initial_state.get("fos_buckling", 0),
        "initial_mass": initial_state.get("mass", 0),
        "grammar_success_rate": sum(parse_success) / max(len(parse_success), 1),
        "action_effective_rate": sum(action_effective) / max(len(action_effective), 1),
        "action_rewritten_rate": sum(action_rewritten) / max(len(action_rewritten), 1),
    }


def _eval_problems_batched(specs, model, tokenizer, formatter, *, max_steps, temperature,
                           max_new_tokens, chunk=8):
    """Turn-major batched evaluation: every problem advances one turn together.

    The sequential path generates for one problem at a time, so a 34-problem
    evaluation spends ~6 hours with the GPU mostly idle between short decodes.
    Batching across problems is the same computation in a different order --
    identical greedy decoding, identical FEA, identical stopping rule -- and it is
    how the training rollout has always worked (multiturn_rollout.py is turn-major
    for exactly this reason).

    Sequences are LEFT-padded with an explicit attention mask so a short context
    cannot shift another problem's positions.
    """
    import torch
    from llm_finetune.data.processors import designbench_prompt
    from llm_finetune.envs.truss_env import (
        _analyze_truss, _apply_action, _load_truss_and_goals, parse_grammar_action,
        normalize_action as _normalize_action,
    )
    from llm_finetune.training.rl.posterior.potential import program_from_truss_spec

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_ids = {i for i in (tokenizer.eos_token_id,
                           tokenizer.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0}

    states = []
    for spec in specs:
        truss, goals = _load_truss_and_goals(spec)
        init = _analyze_truss(truss, goals)
        states.append({"spec": spec, "truss": truss, "goals": goals, "initial": init,
                       "state": init, "history": [], "parse": [], "effective": [],
                       "rewritten": [],
                       "done": False})

    for _turn in range(max_steps):
        active = [st for st in states if not st["done"] and not st["state"].get("is_feasible", False)]
        for st in states:
            if not st["done"] and st["state"].get("is_feasible", False):
                st["done"] = True
        if not active:
            break
        prompts = [
            list(formatter.apply_template(
                designbench_prompt.build_messages(st["spec"], st["initial"], st["history"], formatter),
                add_generation_prompt=True, tokenize=True))
            for st in active
        ]
        outputs = []
        for s0 in range(0, len(prompts), chunk):
            batch = prompts[s0: s0 + chunk]
            width = max(len(x) for x in batch)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), width), dtype=torch.long)
            for j, x in enumerate(batch):          # left pad
                ids[j, width - len(x):] = torch.tensor(x, dtype=torch.long)
                attn[j, width - len(x):] = 1
            ids, attn = ids.to(device), attn.to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=pad_id,
                )
            for j in range(len(batch)):
                gen = out[j, width:].tolist()
                cut = len(gen)
                for t, tid in enumerate(gen):
                    if tid in eos_ids:
                        cut = t + 1
                        break
                outputs.append(tokenizer.decode(gen[:cut], skip_special_tokens=True))

        for st, text in zip(active, outputs):
            thinking = text.strip()
            parsed = parse_grammar_action(text)
            if parsed is None:
                st["parse"].append(False)
                st["effective"].append(False)
                st["history"].append({"action": text.strip()[:120], "fea_result": st["state"],
                                      "thinking": thinking})
                continue
            try:
                st["rewritten"].append(_normalize_action(st["truss"], parsed) != parsed)
                _before = _design_fingerprint(st["truss"])
                st["truss"] = _apply_action(st["truss"], parsed)
                st["state"] = _analyze_truss(st["truss"], st["goals"])
                st["parse"].append(True)
                st["effective"].append(_design_fingerprint(st["truss"]) != _before)
            except Exception:  # noqa: BLE001
                st["parse"].append(False)
                st["effective"].append(False)
            st["history"].append({"action": parsed, "fea_result": st["state"], "thinking": thinking})

    results = []
    for st in states:
        state, initial, spec = st["state"], st["initial"], st["spec"]
        program = program_from_truss_spec(spec, initial_mass=initial.get("mass"))
        mass = state.get("mass", 0) or 0.0
        # Terminal potential, so a group of K rollouts can be RANKED at inference.
        # Offline this picks the feasible member of a group 98-100% of the time
        # across three independent trajectory samplers; see
        # scripts/composite_ordering_divergence.py. Shaping cannot deliver that
        # through the gradient (tests/test_shaping_invisibility.py), but selection
        # does not go through the gradient.
        try:
            from llm_finetune.training.rl.posterior.potential import compute_potential_v2
            phi_v2 = float(compute_potential_v2(state, program, alpha=5.0, tau=0.05))
        except Exception:  # noqa: BLE001
            phi_v2 = float("-inf")
        results.append({
            "phi_v2": phi_v2,
            "problem_id": spec.get("problem_id", ""),
            "reaches_solution": bool(state.get("is_feasible", False)),
            "valid_feasible": bool(state.get("is_feasible", False)) and bool(program.is_valid(state)),
            "state_is_valid": bool(program.is_valid(state)),
            "n_steps": len(st["history"]),
            "final_fos_buckling": state.get("fos_buckling", 0),
            "final_fos_yielding": state.get("fos_yielding", 0),
            "final_mass": mass,
            "final_deflection": state.get("deflection", 0),
            "mass_ref": program.objective_ref,
            "mass_ratio": (mass / program.objective_ref) if program.objective_ref else None,
            "initial_fos_buckling": initial.get("fos_buckling", 0),
            "initial_mass": initial.get("mass", 0),
            "grammar_success_rate": sum(st["parse"]) / max(len(st["parse"]), 1),
            "action_effective_rate": sum(st["effective"]) / max(len(st["effective"]), 1),
            "action_rewritten_rate": sum(st["rewritten"]) / max(len(st["rewritten"]), 1),
        })
    return results


def _eval_problems_sampled(specs, model, tokenizer, formatter, *, max_steps, temperature,
                           max_new_tokens, n_samples, chunk=8, seed=0):
    """Expected feasibility under the policy's own sampling distribution.

    Greedy single-rollout evaluation is a knife-edge estimator on this task. The
    same checkpoint scored 0.667 sequentially and 0.500 through a batch of 8 --
    batch size 1 reproduced the sequential path EXACTLY (12/12 problems, 12/12
    step counts), so there is no bug: batching merely reorders floating-point
    reductions, one greedy token flips, and a 14-turn rollout diverges from there.
    A ~17-point swing from a numerically meaningless change is larger than any
    effect these experiments are trying to resolve.

    Averaging over n_samples rollouts at the training temperature replaces that
    0/1 knife edge with a per-problem rate in {0, 1/n, ..., 1}. It is also the
    quantity GRPO actually optimises -- expected return under the policy -- rather
    than the behaviour of one arbitrarily-chosen decoding path.

    Returns one record per problem with solve_rate (the estimator), any_feasible
    (pass@n) and the best mass achieved across samples.
    """
    import statistics as _st

    replicated = []
    for i, spec in enumerate(specs):
        for k in range(n_samples):
            copy_spec = dict(spec)
            copy_spec["_sample_index"] = k
            copy_spec["_origin"] = spec.get("problem_id", str(i))
            replicated.append(copy_spec)

    torch_seed = seed
    try:
        import torch
        torch.manual_seed(torch_seed)
    except Exception:  # noqa: BLE001
        pass

    flat = _eval_problems_batched(
        replicated, model, tokenizer, formatter, max_steps=max_steps,
        temperature=temperature, max_new_tokens=max_new_tokens, chunk=chunk,
    )

    grouped: dict = {}
    for spec, rec in zip(replicated, flat):
        grouped.setdefault(spec["_origin"], []).append(rec)

    out = []
    for pid, recs in grouped.items():
        feas = [r for r in recs if r["reaches_solution"] and r.get("state_is_valid", True)]
        masses = [r["mass_ratio"] for r in feas if r.get("mass_ratio")]
        # Three selection rules over one generation pass, so the comparison is
        # paired at the rollout level: no-selection (first sample), Phi-ranked
        # (the hybrid), and pass@n (the oracle ceiling).
        _phi_pick = max(recs, key=lambda r: r.get("phi_v2", float("-inf")))
        _first = recs[0]
        out.append({
            "problem_id": pid,
            "n_samples": len(recs),
            "solve_rate": len(feas) / len(recs),
            "first_feasible": bool(_first["reaches_solution"] and _first.get("state_is_valid", True)),
            "phi_selected_feasible": bool(_phi_pick["reaches_solution"]
                                          and _phi_pick.get("state_is_valid", True)),
            "phi_selected_mass_ratio": _phi_pick.get("mass_ratio"),
            "oracle_feasible": bool(feas),
            "reaches_solution": bool(feas),          # pass@n, for continuity
            "valid_feasible": bool(feas),
            "state_is_valid": all(r.get("state_is_valid", True) for r in recs),
            "n_steps": _st.mean([r["n_steps"] for r in recs]),
            "final_fos_buckling": _st.mean([r["final_fos_buckling"] for r in recs]),
            "final_fos_yielding": _st.mean([r["final_fos_yielding"] for r in recs]),
            "final_mass": min((r["final_mass"] for r in feas), default=recs[0]["final_mass"]),
            "mass_ref": recs[0]["mass_ref"],
            "mass_ratio": min(masses) if masses else None,
            "initial_fos_buckling": recs[0]["initial_fos_buckling"],
            "initial_mass": recs[0]["initial_mass"],
            "grammar_success_rate": _st.mean([r["grammar_success_rate"] for r in recs]),
            "action_effective_rate": _st.mean([r.get("action_effective_rate", 0.0) for r in recs]),
            "action_rewritten_rate": _st.mean([r.get("action_rewritten_rate", 0.0) for r in recs]),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on DesignBench")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint directory (None = use base model)")
    parser.add_argument("--model-config", required=True,
                        help="Path to model config YAML (e.g., configs/model/qwen3_14b.yaml)")
    parser.add_argument("--problems-dir",
                        default="/ocean/projects/mch250030p/wxu7/DesignBench/data/problems",
                        help="Directory with problem spec JSON files")
    parser.add_argument("--output-dir", default="results/eval",
                        help="Directory to save evaluation results")
    parser.add_argument("--wandb-project", default="designbench-training",
                        help="W&B project for logging results")
    parser.add_argument("--wandb-run-name", default=None,
                        help="W&B run name for this eval")
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Number of GPUs for model inference")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max grammar actions per episode")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Rollouts per problem. >1 switches to the sampled estimator: "
                             "mean solve rate at --temperature, which is stable where a single "
                             "greedy rollout is not (batch size alone moved greedy feasibility "
                             "by 17 points on the same checkpoint).")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--rollout", choices=["multi", "single"], default="multi",
                        help="multi: generate→FEA→generate loop, one action/turn "
                             "(matches training/inference distribution). single: one big "
                             "completion parsed for all actions (legacy; OOD, mismatched).")
    parser.add_argument("--few-shot", action="store_true",
                        help="Append a worked example to the system prompt (format-enable a base teacher model).")
    parser.add_argument("--extract-action", action="store_true",
                        help="For reasoning models that produce CoT but no inline <action>: when "
                             "no grammar action is parseable, ask the model once more to commit its "
                             "reasoning to a single grammar action (keeps the full CoT).")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max new tokens per model turn (multi-turn rollout).")
    parser.add_argument("--batched", action="store_true",
                        help="Turn-major batched generation across problems. VERIFIED EQUIVALENT "
                             "ONLY AT --batch-size 1 (12/12 problems and step counts matched the "
                             "sequential path). At batch size 8 it diverged to 8/12 and moved "
                             "feasibility 0.667 -> 0.500 on a fixed checkpoint: batching reorders "
                             "float reductions, one greedy token flips, and a 14-turn rollout "
                             "goes elsewhere. Use --allow-unequal-batching to accept that.")
    parser.add_argument("--allow-unequal-batching", action="store_true",
                        help="Permit --batched with batch size > 1 for greedy decoding.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split-file", default=None,
                        help="Path to a data/splits/*.json produced by scripts/make_splits.py. "
                             "Without it, evaluation is NOT held out from GRPO training.")
    parser.add_argument("--split", default="eval", choices=["train", "eval"],
                        help="Which side of --split-file to evaluate.")
    parser.add_argument("--max-problems", type=int, default=0,
                        help="If >0, evaluate only the first N problems (quick baseline).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model config (resolves Hydra `defaults: [base]` — see _load_model_config)
    model_cfg = _load_model_config(args.model_config)
    base_model_id = model_cfg.model_name_or_path

    # Eval-time overrides (single-GPU inference, no DeepSpeed):
    #   device_map=auto  → place the model on the GPU. base.yaml uses null for
    #                      DeepSpeed; left as-is the model would sit on CPU and
    #                      mismatch the .cuda() input tensors below.
    #   grad-ckpt off    → gradient_checkpointing_enable() forces use_cache=False,
    #                      which makes generate() crawl. Off for eval.
    #   use_lora=False   → we attach the *trained* adapter below, not a fresh one.
    model_cfg.device_map = "auto"
    model_cfg.use_gradient_checkpointing = False
    model_cfg.use_compile = False
    model_cfg.use_lora = False

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    is_adapter = bool(ckpt and (ckpt / "adapter_config.json").exists())
    if ckpt and not is_adapter:
        # Full-weights checkpoint: load directly from the checkpoint directory.
        model_cfg.model_name_or_path = str(ckpt)

    model_id = str(ckpt) if ckpt else base_model_id
    log.info(f"Evaluating model: {model_id} (base={base_model_id}, adapter={is_adapter})")

    from llm_finetune.models.loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # Attach the trained checkpoint. The v2 checkpoints are LoRA adapters on top
    # of the base model; the previous code loaded ONLY the base model and never
    # the adapter, so it would have scored raw Qwen3-14B (a silent, meaningless
    # baseline). Apply + merge the adapter, and adopt the checkpoint's own
    # tokenizer so the chat template matches what was used during training.
    if is_adapter:
        from peft import PeftModel
        log.info(f"Loading + merging LoRA adapter from {ckpt}")
        model = PeftModel.from_pretrained(model, str(ckpt))
        model = model.merge_and_unload()
        if (ckpt / "tokenizer_config.json").exists():
            from transformers import AutoTokenizer
            ck_tok = AutoTokenizer.from_pretrained(str(ckpt))
            if ck_tok.pad_token is None and ck_tok.eos_token is not None:
                ck_tok.pad_token = ck_tok.eos_token
                ck_tok.pad_token_id = ck_tok.eos_token_id
            tokenizer = ck_tok

    model.eval()
    log.info(f"Model loaded: {model_id}")

    # NOTE: We intentionally do NOT construct DesignBench's LocalModelAdapter here.
    # Its __init__ re-loads the model from `model_name` via AutoTokenizer/AutoModel
    # (model_name="final" → HF hub → 401), and the object was never used downstream
    # anyway — the eval loop below drives `model.generate` directly. Removing it
    # avoids a wasteful second load and the crash it caused.

    # Run benchmark evaluation problem by problem
    problems_dir = Path(args.problems_dir)
    if getattr(args, "split_file", None):
        # A split file is authoritative; globbing auto_problem_* first would
        # silently drop the rest of the held-out set.
        problem_files = sorted(problems_dir.glob("*.json"))
    else:
        problem_files = sorted(problems_dir.glob("auto_problem_*.json"))
        if not problem_files:
            problem_files = sorted(problems_dir.glob("*.json"))
    log.info(f"Found {len(problem_files)} problems in {problems_dir}")

    from llm_finetune.envs.truss_env import TrussRolloutEnv
    from llm_finetune.data.processors.chat_formatter import ChatFormatter
    from llm_finetune.data.datasets.rl_dataset import _spec_to_problem_text
    import torch

    # Use the base HF id (not the checkpoint path) so thinking mode resolves via
    # MODEL_THINKING_MODE — a checkpoint path would default to NONE and silently
    # drop Qwen3 enable_thinking.
    formatter = ChatFormatter.from_model_id(base_model_id, tokenizer)
    env = TrussRolloutEnv(max_steps=args.max_steps)
    device = next(model.parameters()).device

    # Held-out discipline: --split-file restricts evaluation to the side of a
    # fixed split (scripts/make_splits.py). Without it, evaluation reads the same
    # directory GRPO trains on and no number is held out.
    if getattr(args, "split_file", None):
        with open(args.split_file) as f:
            split_payload = json.load(f)
        wanted = set(split_payload[args.split])
        kept, seen = [], set()
        for pf in problem_files:
            with open(pf) as f:
                spec = json.load(f)
            if not isinstance(spec, dict) or "topology" not in spec:
                continue          # e.g. test_problems.json is a list, not a spec
            pid = spec.get("problem_id", pf.stem)
            if pid in wanted and pid not in seen:
                seen.add(pid)
                kept.append(pf)
        log.info(
            f"Split {args.split_file} [{args.split}]: {len(kept)}/{len(problem_files)} problems kept"
        )
        problem_files = kept

    if args.max_problems and args.max_problems > 0:
        problem_files = problem_files[:args.max_problems]
        log.info(f"Limiting to first {len(problem_files)} problems (--max-problems)")
    log.info(f"Rollout mode: {args.rollout}")

    all_results = []

    if (args.batched and args.batch_size > 1 and args.n_samples <= 1
            and args.temperature == 0.0 and not args.allow_unequal_batching):
        raise SystemExit(
            "Refusing --batched with --batch-size > 1 under greedy decoding: it does not "
            "reproduce the sequential result (0.667 -> 0.500 on a fixed checkpoint). Use "
            "--batch-size 1, or --n-samples > 1 where the averaging absorbs it, or pass "
            "--allow-unequal-batching deliberately."
        )

    if args.n_samples > 1 and args.rollout == "multi":
        specs = []
        for pf in problem_files:
            with open(pf) as f:
                spec = json.load(f)
            spec.setdefault("problem_id", pf.stem)
            specs.append(spec)
        log.info(f"Sampled evaluation: {len(specs)} problems x {args.n_samples} rollouts "
                 f"at temperature {args.temperature}, seed {args.sample_seed}")
        t_start = time.time()
        all_results = _eval_problems_sampled(
            specs, model, tokenizer, formatter, max_steps=args.max_steps,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            n_samples=args.n_samples, chunk=args.batch_size, seed=args.sample_seed,
        )
        for r in all_results:
            log.info(f"  {r['problem_id']}: solve_rate={r['solve_rate']:.2f} "
                     f"({r['n_samples']} samples), gram={r['grammar_success_rate']:.2f} "
                     f"eff={r.get('action_effective_rate',0.0):.2f}")
        log.info(f"Sampled evaluation took {time.time()-t_start:.0f}s")
        problem_files = []
    elif args.batched and args.rollout == "multi":
        specs = []
        for pf in problem_files:
            with open(pf) as f:
                spec = json.load(f)
            spec.setdefault("problem_id", pf.stem)
            specs.append(spec)
        log.info(f"Batched evaluation of {len(specs)} problems, batch size {args.batch_size}")
        t_start = time.time()
        all_results = _eval_problems_batched(
            specs, model, tokenizer, formatter,
            max_steps=args.max_steps, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens, chunk=args.batch_size,
        )
        for r in all_results:
            log.info(f"  {r['problem_id']}: feasible={r['reaches_solution']}, "
                     f"fos_b={r['final_fos_buckling']:.3f}, n_steps={r['n_steps']}, "
                     f"gram={r['grammar_success_rate']:.2f} eff={r.get('action_effective_rate',0.0):.2f}")
        log.info(f"Batched evaluation took {time.time()-t_start:.0f}s")
        problem_files = []          # skip the sequential loop below

    for pf in problem_files:
        with open(pf) as f:
            spec = json.load(f)
        pid = spec.get("problem_id", pf.stem)
        log.info(f"Evaluating problem: {pid}")
        problem_text = _spec_to_problem_text(spec)

        if args.rollout == "multi":
            result = _eval_problem_multiturn(
                spec, model, tokenizer, formatter, problem_text,
                max_steps=args.max_steps, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens, few_shot=args.few_shot,
                extract_action=args.extract_action,
            )
            result["problem_id"] = pid
        else:
            # Legacy single-turn: one big completion, parse all actions out of it.
            messages = formatter.build_messages(problem_text=problem_text)
            prompt_ids = formatter.apply_template(messages, add_generation_prompt=True, tokenize=True)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    max_new_tokens=8192,
                    temperature=args.temperature if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(output[0][input_tensor.shape[1]:], skip_special_tokens=True)
            rollout = env.run_completion(spec, completion)
            result = {
                "problem_id": pid,
                "reaches_solution": rollout.reaches_solution,
                "n_steps": rollout.n_steps,
                "final_fos_buckling": rollout.final_state.get("fos_buckling", 0),
                "final_fos_yielding": rollout.final_state.get("fos_yielding", 0),
                "final_mass": rollout.final_state.get("mass", 0),
                "initial_fos_buckling": rollout.initial_state.get("fos_buckling", 0),
                "initial_mass": rollout.initial_state.get("mass", 0),
                "grammar_success_rate": sum(rollout.parse_success) / max(len(rollout.parse_success), 1),
                "action_effective_rate": getattr(rollout, "action_effective_rate", 0.0),
            }

        all_results.append(result)
        log.info(f"  {pid}: feasible={result['reaches_solution']}, "
                 f"fos_b={result['final_fos_buckling']:.3f}, n_steps={result['n_steps']}, "
                 f"gram={result['grammar_success_rate']:.2f}")

    # Aggregate metrics
    n = len(all_results)
    import statistics as _st
    _valid_solved = [r for r in all_results if r.get("valid_feasible")]
    _ratios = [r["mass_ratio"] for r in _valid_solved
               if r.get("mass_ratio") is not None and math.isfinite(r["mass_ratio"])]
    agg = {
        "n_problems": n,
        "rollout": args.rollout,
        "feasibility_rate": sum(r["reaches_solution"] for r in all_results) / n,
        # The number to quote: feasible AND inside the simulator's validity envelope.
        "valid_feasibility_rate": sum(bool(r.get("valid_feasible")) for r in all_results) / n,
        # The stable estimator when --n-samples > 1: mean over problems of the
        # fraction of rollouts that reached a valid feasible design.
        "mean_solve_rate": (sum(r["solve_rate"] for r in all_results) / n
                            if all("solve_rate" in r for r in all_results) else None),
        "n_samples_per_problem": (all_results[0].get("n_samples") if all_results else None),
        # Design quality on the solved set -- the axis where potential estimation
        # shows an advantage and feasibility (a pass/fail gate a greedy heuristic
        # saturates) cannot.
        "median_mass_ratio": _st.median(_ratios) if _ratios else None,
        "mean_mass_ratio": _st.mean(_ratios) if _ratios else None,
        "mean_grammar_success_rate": sum(r["grammar_success_rate"] for r in all_results) / n,
        "mean_action_effective_rate": sum(r.get("action_effective_rate", 0.0) for r in all_results) / n,
        "mean_action_rewritten_rate": sum(r.get("action_rewritten_rate", 0.0) for r in all_results) / n,
        "first_feasibility_rate": (sum(bool(r.get("first_feasible")) for r in all_results) / n
                                   if any("first_feasible" in r for r in all_results) else None),
        "phi_selected_feasibility_rate": (sum(bool(r.get("phi_selected_feasible")) for r in all_results) / n
                                          if any("phi_selected_feasible" in r for r in all_results) else None),
        "oracle_feasibility_rate": (sum(bool(r.get("oracle_feasible")) for r in all_results) / n
                                    if any("oracle_feasible" in r for r in all_results) else None),
        "mean_final_fos_buckling": sum(r["final_fos_buckling"] for r in all_results) / n,
        "mean_steps": sum(r["n_steps"] for r in all_results) / n,
    }
    log.info(f"\nAggregated results:\n" + json.dumps(agg, indent=2))

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump({"aggregate": agg, "per_problem": all_results}, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Log to W&B
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval_{model_id.split('/')[-1]}",
            config={"checkpoint": args.checkpoint, "model": model_id},
        )
        wandb.log({"eval/" + k: v for k, v in agg.items()})
        # Log as W&B table
        table = wandb.Table(columns=list(all_results[0].keys()))
        for r in all_results:
            table.add_data(*r.values())
        wandb.log({"eval/per_problem": table})
        run.finish()
        log.info("W&B eval run logged")
    except Exception as e:
        log.warning(f"W&B logging failed: {e}")


if __name__ == "__main__":
    main()
