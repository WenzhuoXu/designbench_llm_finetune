"""Multi-turn rollout for TRL GRPO via the ``rollout_func`` hook (batched).

Turn-0 of the ablation study found that GRPO (like the eval harness) drove a
multi-turn-trained model **single-turn** — one prompt, one big completion,
trajectory faked by splitting on "\\n\\n". The model never practiced the real
task (generate one action → see FEA feedback → generate the next), so RL could
not learn an optimisation policy. The fix (validated: kl moves, grad_norm 9× up)
is to run the real per-step generate→FEA→generate loop here. See
``docs/ablation_ledger.md`` and ``docs/multiturn_grpo_design.md``.

The model (Qwen3) does genuine multi-token reasoning per turn (the ``<action>``
comes after ``</think>``), so turns are long and CANNOT be truncated. To make a
reasoning-faithful sweep tractable this rollout is **turn-major and batched**:
at each turn we batch ALL still-active rollouts into one ``generate`` call
(left-padded, memory-chunked), instead of generating each rollout sequentially.

Contract (verified against the installed TRL 1.0.0 source, NOT the docstring):
  * ``prompts`` has length ``N = B*G`` with each problem prompt repeated
    ``num_generations`` (G) times consecutively (one GRPO group). We return one
    rollout per entry, 1:1; temperature gives in-group diversity.
  * Return ``prompt_ids``/``completion_ids``/``logprobs`` (length N, per-token
    aligned), ``env_mask`` (1 = model token, 0 = FEA/template token; TRL masks
    the loss with it), and ``rollout_result`` (extra field, merged 1:1 into the
    reward kwargs so the reward uses the *actual experienced* trajectory).
  * vLLM must be OFF (per-turn HF generate).
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# prompt_text (exact string TRL passes) -> problem_spec dict
_PROMPT_TO_SPEC: dict[str, dict] = {}


def register_prompt_specs(dataset) -> int:
    """Populate the prompt→spec registry from an RL prompt dataset.

    The rollout_func only receives the prompt strings; it needs each prompt's
    problem spec to drive FEA. We key by the exact ``prompt`` string the dataset
    exposes (what TRL forwards verbatim).
    """
    n = 0
    for i in range(len(dataset)):
        item = dataset[i]
        prompt = item.get("prompt")
        spec = item.get("problem_spec")
        if isinstance(prompt, str) and spec:
            _PROMPT_TO_SPEC[prompt] = spec
            n += 1
    log.info(f"[mt-rollout] registered {n} prompt→spec entries ({len(_PROMPT_TO_SPEC)} total)")
    return n


def _generate_batch(trainer, input_ids_list, gen_cfg, eos_ids, chunk):
    """Batched generation. ``input_ids_list``: list of 1-D token-id lists.

    Returns a list (aligned with input) of (gen_ids: list[int], logprobs:
    list[float]) for the generated tokens only (truncated at the first eos).
    Left-pads each chunk; uses the KV cache (gradient checkpointing otherwise
    forces use_cache=False → crawling generate); memory-chunked to bound the
    batch × context size.
    """
    import torch
    from trl.models import unwrap_model_for_generation

    tok = trainer.processing_class
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    device = trainer.accelerator.device
    gather = getattr(trainer.args, "ds3_gather_for_generation", True)
    eos_set = set(eos_ids)
    results: list = []

    with unwrap_model_for_generation(
        trainer.model_wrapped, trainer.accelerator, gather_deepspeed3_params=gather
    ) as unwrapped, torch.no_grad():
        prev_cache = getattr(unwrapped.config, "use_cache", None)
        unwrapped.config.use_cache = True
        try:
            for s in range(0, len(input_ids_list), chunk):
                batch = input_ids_list[s:s + chunk]
                maxlen = max(len(x) for x in batch)
                padded = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
                attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
                for j, x in enumerate(batch):  # left-pad
                    padded[j, maxlen - len(x):] = torch.tensor(x, dtype=torch.long)
                    attn[j, maxlen - len(x):] = 1
                padded, attn = padded.to(device), attn.to(device)
                out = unwrapped.generate(
                    input_ids=padded, attention_mask=attn, generation_config=gen_cfg,
                    return_dict_in_generate=True, output_scores=True,
                )
                seqs = out.sequences            # [b, maxlen + G]
                scores = out.scores             # tuple length G of [b, vocab]
                glen = len(scores)
                for j in range(len(batch)):
                    gen = seqs[j, maxlen:].tolist()
                    cut = len(gen)
                    for t, tid in enumerate(gen):     # stop at first eos (inclusive)
                        if tid in eos_set:
                            cut = t + 1
                            break
                    gen = gen[:cut]
                    logps = []
                    for t in range(min(cut, glen)):
                        lp = torch.log_softmax(scores[t][j].float(), dim=-1)
                        logps.append(float(lp[seqs[j, maxlen + t]]))
                    if len(logps) < len(gen):
                        logps += [0.0] * (len(gen) - len(logps))
                    results.append((gen, logps[:len(gen)]))
        finally:
            if prev_cache is not None:
                unwrapped.config.use_cache = prev_cache
    return results


def _build_glue_fn(formatter, tok, eos_ids):
    """Return ``glue(fea_text, gen_ids) -> list[int]``: the tokens between turns.

    The rollout used to obtain these by re-rendering the whole conversation and
    diffing against the running ids. That silently produced NOTHING. Qwen3's chat
    template strips ``<think>...</think>`` from every non-final assistant message,
    so the re-render is not a prefix extension of the running sequence, the
    ``len(next_ids) > len(run)`` guard fell through to ``delta = []``, and the
    completion became gen(turn0) ++ gen(turn1) ++ ... with **no FEA feedback and
    no chat-template glue at all**. ``env_mask`` stayed all-ones, so TRL's
    environment-token masking -- the entire reason this rollout_func exists --
    never fired, and every token of turns >= 1 was scored under a context that did
    not exist at sampling time. The champion run logged 9176 of those warnings: it
    fired on essentially every transition.

    Derive the separator ONCE from the template instead. Render a probe
    conversation whose assistant turn is followed by a user turn -- the same
    position a rollout's assistant turn is in -- so the template applies exactly
    the transition it will apply during training, and read off everything after
    the assistant's own content. Template-agnostic: no chat tokens are hardcoded.
    """
    ASSISTANT_MARK = "\x02ASSISTANT\x02"
    FEA_MARK = "\x02FEA\x02"
    probe = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

    def _fail(reason: str):
        log.warning(f"[mt-rollout] could not derive chat glue ({reason}); "
                    "inter-turn tokens would be omitted -- refusing to train blind")
        return None

    opened = formatter.apply_template(probe, add_generation_prompt=True, tokenize=False)
    with_user = formatter.apply_template(
        probe
        + [{"role": "assistant", "content": ASSISTANT_MARK},
           {"role": "user", "content": FEA_MARK}],
        add_generation_prompt=True, tokenize=False,
    )
    if not with_user.startswith(opened):
        return _fail("generation prompt is not a prefix of the next-turn render")
    tail = with_user[len(opened):]
    if ASSISTANT_MARK not in tail or FEA_MARK not in tail:
        return _fail("probe markers did not survive the template")
    glue_template = tail[tail.index(ASSISTANT_MARK) + len(ASSISTANT_MARK):]
    if not glue_template.endswith(opened[opened.rindex("<") :]) and not glue_template:
        return _fail("empty glue")
    eos_set = set(eos_ids)

    def glue(fea_text: str, gen_ids: list) -> list:
        ids = tok(glue_template.replace(FEA_MARK, fea_text),
                  add_special_tokens=False)["input_ids"]
        # The model usually emits the turn terminator itself; do not double it.
        if gen_ids and ids and gen_ids[-1] in eos_set and ids[0] == gen_ids[-1]:
            ids = ids[1:]
        return ids

    return glue


def _probe_here(pre_truss, st, analyze_fn, apply_fn, probe_state_fn, candidates_fn,
                max_candidates, rng, alpha, tau, gamma):
    """Rank the candidate actions at the pre-action state and place the policy's choice.

    ``pre_truss`` is a snapshot taken before the policy's action was applied, so
    the counterfactuals branch from the same state the policy chose from.
    Failures are swallowed: a diagnostic must never take down a training step.
    """
    try:
        cands = candidates_fn(pre_truss, st.get("bounds") or {},
                              max_candidates=max_candidates, rng=rng)
        if len(cands) < 2:
            return None
        goals = st["goals"]

        def transition(action: str):
            trial = copy.deepcopy(pre_truss)
            try:
                apply_fn(trial, action)
            except Exception:
                return None
            return analyze_fn(trial, goals)

        result = probe_state_fn(
            program=st["program"], candidates=cands, transition=transition,
            policy_next_state=st.get("state"),
            alpha=alpha, tau=tau, gamma=gamma, depth=1,
        )
        if result is None:
            return None
        return {
            "rho": result.rho, "regret": result.regret, "rank_frac": result.rank_frac,
            "margin": result.margin, "sigma": result.sigma,
            "reliable": bool(result.reliable), "n_candidates": result.n_candidates,
            "policy_phi": result.policy_phi, "baseline_phi": result.baseline_phi,
            "best_phi": result.best_phi,
            "advantage": result.policy_phi - result.baseline_phi,
        }
    except Exception as exc:  # noqa: BLE001
        log.debug(f"[mt-rollout] lookahead probe failed: {exc}")
        return None


def make_multiturn_rollout_func(
    formatter,
    rl_cfg,
    base_model_id: Optional[str] = None,
) -> Callable[[list, Any], dict]:
    """Build a TRL ``rollout_func`` running batched multi-turn truss rollouts."""
    from transformers import GenerationConfig

    from llm_finetune.data.processors import designbench_prompt
    from llm_finetune.envs.truss_env import (
        _analyze_truss, _apply_action, _load_truss_and_goals, parse_grammar_action,
    )
    from llm_finetune.training.rl.posterior.lookahead_probe import (
        probe_state, truss_candidate_actions, truss_param_bounds,
    )
    from llm_finetune.training.rl.posterior.potential import program_from_truss_spec
    from llm_finetune.training.rl.rewards import RolloutResult

    # Lookahead probe: at a bounded number of visited states per batch, rank the
    # candidate actions by the design program's potential and locate the action
    # the policy actually took in that order. This replaces the rho of
    # grpo_trainer.py:345, which is an identity under tree mode and a 1/K coin
    # flip otherwise. See posterior/lookahead_probe.py.
    probe_cfg = dict(rl_cfg.get("lookahead_probe", {}) or {})
    probe_enabled = bool(probe_cfg.get("enabled", False))
    probe_states_per_batch = int(probe_cfg.get("max_states_per_batch", 6))
    probe_all_steps = bool(probe_cfg.get("all_steps", False))
    probe_max_candidates = int(probe_cfg.get("max_candidates", 48))
    posterior_cfg = dict(rl_cfg.get("posterior", {}) or {})
    probe_alpha = float(posterior_cfg.get("alpha", 5.0))
    probe_gamma = float(posterior_cfg.get("gamma", 0.99))
    probe_tau = float(posterior_cfg.get("tau", 0.05))

    max_turns = int(rl_cfg.get("max_turns", rl_cfg.get("max_steps", 20)))
    max_new_tokens = int(rl_cfg.get("max_turn_tokens", 1536))
    temperature = float(rl_cfg.get("temperature", 0.8))
    top_p = float(rl_cfg.get("top_p", 0.95))
    gen_chunk = int(rl_cfg.get("gen_batch_chunk", 8))  # cap batch×context memory

    def rollout_func(prompts: list, trainer) -> dict:
        import torch  # noqa: F401  (ensures torch loaded in this process)
        tok = trainer.processing_class
        eos_ids = [t for t in [tok.eos_token_id] if t is not None]
        imend = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(imend, int) and imend >= 0 and imend != tok.unk_token_id and imend not in eos_ids:
            eos_ids.append(imend)
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None, top_p=top_p,
            pad_token_id=tok.pad_token_id, eos_token_id=eos_ids or tok.eos_token_id,
        )

        def apply_tpl(msgs):
            return list(formatter.apply_template(msgs, add_generation_prompt=True, tokenize=True))

        glue_fn = _build_glue_fn(formatter, tok, eos_ids)

        # ── init one rollout state per prompt entry (1:1 with N=B*G) ─────────
        R: list[dict] = []
        for prompt in prompts:
            spec = _PROMPT_TO_SPEC.get(prompt)
            st: dict = {"spec": spec, "done": spec is None, "ok": spec is not None}
            if spec is None:
                log.warning("[mt-rollout] prompt not in registry; empty rollout")
            else:
                try:
                    truss, goals = _load_truss_and_goals(spec)
                    init = _analyze_truss(truss, goals)
                    st.update(truss=truss, goals=goals, initial_state=init, state=init,
                              action_history=[], state_history=[init], action_sequence=[],
                              raw_outputs=[], token_counts=[], parse_success=[], n_fea=0,
                              comp_ids=[], comp_logps=[], comp_mask=[], probes=[],
                              tok_adv=[], turn_spans=[])
                    if probe_enabled:
                        st["program"] = program_from_truss_spec(spec, initial_mass=init.get("mass"))
                        st["bounds"] = truss_param_bounds(spec)
                    st["running_ids"] = apply_tpl(
                        designbench_prompt.build_messages(spec, init, [], formatter))
                    st["prompt_ids"] = list(st["running_ids"])
                except Exception as e:  # noqa: BLE001
                    log.warning(f"[mt-rollout] reset failed: {e}")
                    st["done"] = True
                    st["ok"] = False
            R.append(st)

        _dbg = {"logged": False}
        # One shared probe budget for the whole batch keeps the added simulator
        # cost independent of batch size (~0.13 s per measured state at 64
        # candidates, against a ~500 s training step).
        probe_budget = [
            (10 ** 9 if probe_all_steps else probe_states_per_batch) if probe_enabled else 0
        ]
        probe_rng = random.Random(1234)

        # ── turn-major batched loop ──────────────────────────────────────────
        for turn in range(max_turns):
            active = [i for i, st in enumerate(R) if not st["done"] and st["ok"]]
            if not active:
                break
            gens = _generate_batch(trainer, [R[i]["running_ids"] for i in active],
                                   gen_cfg, eos_ids, gen_chunk)
            for k, i in enumerate(active):
                st = R[i]
                gen_ids, logps = gens[k]
                if not gen_ids:
                    st["done"] = True
                    continue
                st["running_ids"] = st["running_ids"] + gen_ids
                # Record where this turn's MODEL tokens sit in the completion, so a
                # per-step advantage can be written onto exactly those positions
                # later. GRPO broadcasts one scalar per sequence; a per-step signal
                # summed into that scalar is projected onto span{1}, which is why
                # the scalar-sum form of the lookahead advantage could not move the
                # gradient. Per-token delivery is the only channel for per-step
                # credit.
                span_start = len(st["comp_ids"])
                st["comp_ids"] += gen_ids
                st["comp_logps"] += logps
                st["comp_mask"] += [1] * len(gen_ids)
                st["tok_adv"] += [0.0] * len(gen_ids)
                st["turn_spans"].append((span_start, span_start + len(gen_ids)))
                text = tok.decode(gen_ids, skip_special_tokens=True)
                if not _dbg["logged"]:
                    log.info(f"[mt-rollout SAMPLE pid={st['spec'].get('problem_id')} turn={turn} "
                             f"ntok={len(gen_ids)} ends_eos={gen_ids[-1] in eos_ids}] "
                             f">>>{text[:300]} ... {text[-150:]}")
                    if turn >= 1:
                        _dbg["logged"] = True
                st["raw_outputs"].append(text)
                st["token_counts"].append(len(gen_ids))
                thinking, _ = formatter.extract_thinking(text)
                parsed = parse_grammar_action(text)
                if parsed is None:
                    st["parse_success"].append(False)
                    st["action_sequence"].append(text.strip()[:80])
                    st["action_history"].append({"action": text.strip()[:120], "fea_result": st["state"], "thinking": thinking})
                else:
                    do_probe = (
                        probe_enabled
                        and probe_budget[0] > 0
                        and st.get("program") is not None
                    )
                    pre_truss = copy.deepcopy(st["truss"]) if do_probe else None
                    try:
                        st["truss"] = _apply_action(st["truss"], parsed)
                        st["state"] = _analyze_truss(st["truss"], st["goals"])
                        st["n_fea"] += 1
                        st["parse_success"].append(True)
                    except Exception as e:  # noqa: BLE001
                        log.debug(f"[mt-rollout] action {parsed!r} failed: {e}")
                        st["parse_success"].append(False)
                    if pre_truss is not None:
                        probe_budget[0] -= 1
                        result = _probe_here(
                            pre_truss, st, _analyze_truss, _apply_action,
                            probe_state, truss_candidate_actions,
                            probe_max_candidates, probe_rng,
                            probe_alpha, probe_tau, probe_gamma,
                        )
                        if result is not None:
                            st["probes"].append(result)
                            # Write this turn's counterfactual advantage onto the
                            # tokens the policy actually generated for it.
                            if st["turn_spans"]:
                                lo, hi = st["turn_spans"][-1]
                                adv = float(result.get("advantage", 0.0))
                                for _i in range(lo, min(hi, len(st["tok_adv"]))):
                                    st["tok_adv"][_i] = adv
                    st["action_sequence"].append(parsed)
                    st["action_history"].append({"action": parsed, "fea_result": st["state"], "thinking": thinking})
                st["state_history"].append(st["state"])

                if st["state"].get("is_feasible", False) or turn == max_turns - 1:
                    st["done"] = True
                    continue
                # Inter-turn glue: the assistant terminator, the FEA result as a
                # user turn, and the next generation prompt. Environment tokens,
                # so env_mask=0 (TRL pops it as tool_mask and gives them no
                # gradient) and logprob 0.0.
                fea_text = ("[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n"
                            + designbench_prompt.format_eval_result(st["state"]))
                if glue_fn is None:
                    delta = []
                    if turn == 0:
                        log.warning("[mt-rollout] no chat glue available; turns >= 1 "
                                    "are being trained without FEA feedback in context")
                else:
                    delta = glue_fn(fea_text, gen_ids)
                st["comp_ids"] += delta
                st["comp_logps"] += [0.0] * len(delta)
                st["comp_mask"] += [0] * len(delta)
                st["tok_adv"] += [0.0] * len(delta)      # environment tokens carry no credit
                st["running_ids"] = st["running_ids"] + delta

        # ── assemble outputs (1:1 with prompts) ──────────────────────────────
        out: dict[str, list] = {"prompt_ids": [], "completion_ids": [], "logprobs": [],
                                "env_mask": [], "rollout_result": [],
                                "step_token_advantages": []}
        eos0 = tok.eos_token_id or tok.pad_token_id or 0
        for prompt, st in zip(prompts, R):
            if not st.get("ok") or not st.get("comp_ids"):
                out["prompt_ids"].append(st.get("prompt_ids") or tok(prompt, add_special_tokens=False)["input_ids"])
                out["completion_ids"].append([eos0]); out["logprobs"].append([0.0])
                out["env_mask"].append([1]); out["rollout_result"].append(None)
                out["step_token_advantages"].append([0.0])
                continue
            rr = RolloutResult(
                problem_id=st["spec"].get("problem_id", ""), action_sequence=st["action_sequence"],
                raw_outputs=st["raw_outputs"], state_history=st["state_history"],
                final_state=st["state"], initial_state=st["initial_state"],
                token_counts=st["token_counts"], parse_success=st["parse_success"],
                reaches_solution=bool(st["state"].get("is_feasible", False)),
                n_fea_calls=st["n_fea"], n_steps=len(st["action_sequence"]),
            )
            if st.get("probes"):
                rr.tree_metrics["lookahead_probes"] = list(st["probes"])
            out["prompt_ids"].append(st["prompt_ids"])
            out["completion_ids"].append(st["comp_ids"])
            out["logprobs"].append(st["comp_logps"])
            out["env_mask"].append(st["comp_mask"])
            out["rollout_result"].append(rr)
            tok_adv = st.get("tok_adv") or [0.0] * len(st["comp_ids"])
            if len(tok_adv) != len(st["comp_ids"]):     # never ship a misaligned vector
                log.warning("[mt-rollout] tok_adv length %d != completion length %d; padding",
                            len(tok_adv), len(st["comp_ids"]))
                tok_adv = (tok_adv + [0.0] * len(st["comp_ids"]))[: len(st["comp_ids"])]
            # Credit must land only on tokens the MODEL generated. A non-zero value
            # on an environment token would train the policy on text it never chose.
            stray = sum(1 for a, m in zip(tok_adv, st["comp_mask"]) if a != 0.0 and m == 0)
            if stray and not _dbg.get("stray_warned"):
                log.warning("[mt-rollout] %d per-token advantages fell on environment "
                            "tokens; zeroing them", stray)
                _dbg["stray_warned"] = True
            if stray:
                tok_adv = [a if m == 1 else 0.0 for a, m in zip(tok_adv, st["comp_mask"])]
            out["step_token_advantages"].append(tok_adv)
        return out

    return rollout_func
