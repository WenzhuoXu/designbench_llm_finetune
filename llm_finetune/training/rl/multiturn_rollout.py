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

import logging
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
    from llm_finetune.training.rl.rewards import RolloutResult

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
                              comp_ids=[], comp_logps=[], comp_mask=[])
                    st["running_ids"] = apply_tpl(
                        designbench_prompt.build_messages(spec, init, [], formatter))
                    st["prompt_ids"] = list(st["running_ids"])
                except Exception as e:  # noqa: BLE001
                    log.warning(f"[mt-rollout] reset failed: {e}")
                    st["done"] = True
                    st["ok"] = False
            R.append(st)

        _dbg = {"logged": False}

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
                st["comp_ids"] += gen_ids
                st["comp_logps"] += logps
                st["comp_mask"] += [1] * len(gen_ids)
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
                    try:
                        st["truss"] = _apply_action(st["truss"], parsed)
                        st["state"] = _analyze_truss(st["truss"], st["goals"])
                        st["n_fea"] += 1
                        st["parse_success"].append(True)
                    except Exception as e:  # noqa: BLE001
                        log.debug(f"[mt-rollout] action {parsed!r} failed: {e}")
                        st["parse_success"].append(False)
                    st["action_sequence"].append(parsed)
                    st["action_history"].append({"action": parsed, "fea_result": st["state"], "thinking": thinking})
                st["state_history"].append(st["state"])

                if st["state"].get("is_feasible", False) or turn == max_turns - 1:
                    st["done"] = True
                    continue
                # inter-turn glue (DesignBench format), env tokens (no grad)
                next_ids = apply_tpl(designbench_prompt.build_messages(
                    st["spec"], st["initial_state"], st["action_history"], formatter))
                run = st["running_ids"]
                if len(next_ids) > len(run) and next_ids[:len(run)] == run:
                    delta = next_ids[len(run):]
                else:
                    log.warning(f"[mt-rollout] prefix mismatch pid={st['spec'].get('problem_id')} turn={turn}")
                    delta = next_ids[len(run):] if len(next_ids) > len(run) else []
                st["comp_ids"] += delta
                st["comp_logps"] += [0.0] * len(delta)
                st["comp_mask"] += [0] * len(delta)
                st["running_ids"] = next_ids

        # ── assemble outputs (1:1 with prompts) ──────────────────────────────
        out: dict[str, list] = {"prompt_ids": [], "completion_ids": [], "logprobs": [],
                                "env_mask": [], "rollout_result": []}
        eos0 = tok.eos_token_id or tok.pad_token_id or 0
        for prompt, st in zip(prompts, R):
            if not st.get("ok") or not st.get("comp_ids"):
                out["prompt_ids"].append(st.get("prompt_ids") or tok(prompt, add_special_tokens=False)["input_ids"])
                out["completion_ids"].append([eos0]); out["logprobs"].append([0.0])
                out["env_mask"].append([1]); out["rollout_result"].append(None)
                continue
            rr = RolloutResult(
                problem_id=st["spec"].get("problem_id", ""), action_sequence=st["action_sequence"],
                raw_outputs=st["raw_outputs"], state_history=st["state_history"],
                final_state=st["state"], initial_state=st["initial_state"],
                token_counts=st["token_counts"], parse_success=st["parse_success"],
                reaches_solution=bool(st["state"].get("is_feasible", False)),
                n_fea_calls=st["n_fea"], n_steps=len(st["action_sequence"]),
            )
            out["prompt_ids"].append(st["prompt_ids"])
            out["completion_ids"].append(st["comp_ids"])
            out["logprobs"].append(st["comp_logps"])
            out["env_mask"].append(st["comp_mask"])
            out["rollout_result"].append(rr)
        return out

    return rollout_func
