"""DesignBench-native prompt formatting for RL rollouts and eval.

Turn-0 of the ablation study found that the GRPO rollout and the eval harness
built prompts with ``chat_formatter.build_messages`` (terse ``_spec_to_problem_text``
+ a reformatted ``  Mass: X kg`` FEA block + a "Continue optimizing…" instruction),
which is a DIFFERENT wording/structure from what the model was SFT'd on. The
warmstart was trained on DesignBench's native format produced by
``DesignBench/truss_tot/trace_extractor.py`` (``_format_evaluation_result``) and
``scripts/generate_modification_trees.py`` (``_generate_comprehensive_problem_text``),
then reshaped by ``WarmstartTransform``. The mismatch put the model
out-of-distribution → it rambled ~512 tok/turn (vs ~50 in-distribution) and
likely produced worse actions. See ``docs/ablation_ledger.md``.

This module reproduces the *exact* SFT-time format from the env state dict and
the problem spec, so RL/eval prompts match training:

  system:  TRUSS_SYSTEM_PROMPT
  user:    <problem text>\\n\\nINITIAL STATE ANALYSIS:\\n<eval>
  asst:    <think>…</think>\\n<action>…</action>          (model turns)
  user:    [Simulation Result]\\nSTRUCTURAL ANALYSIS RESULT:\\n<eval>
  …

``<eval>`` mirrors ``_format_evaluation_result`` exactly; the problem text mirrors
``_generate_comprehensive_problem_text`` (joints/members/loading/goals).
"""

from __future__ import annotations

from llm_finetune.data.processors.chat_formatter import TRUSS_SYSTEM_PROMPT

_SUPPORT_MAP = {"pinned": "pinned", "roller_y": "roller", "roller_x": "roller",
                "free": "free", None: "free"}

# Few-shot exemplar to format-enable a base reasoning model (e.g. qwen3_30b teacher):
# it reasons well but won't reliably commit to the <action> block from instructions
# alone. This shows ONE complete, decisive turn (reasoning → single valid action).
# The action mirrors the SFT data format (SCALE_PARAM(all_members, thickness, 1.224)).
FEWSHOT_EXAMPLE = """

Example of a correct response (for a DIFFERENT problem — do not reuse these values):
<think>
Buckling FOS is below the 1.5 minimum, so the compression members are too slender. I will increase all member wall thicknesses by ~22% to raise the moment of inertia and the buckling capacity.
</think>
<action>SCALE_PARAM(all_members, thickness, 1.224)</action>

Always finish your turn with exactly one such <action>...</action> block."""


def format_eval_result(state: dict) -> str:
    """Reproduce DesignBench ``_format_evaluation_result`` from an env state dict.

    The env state has mass / fos_buckling / fos_yielding / deflection /
    is_feasible — the same fields the SFT-time formatter used.
    """
    lines = []
    if state.get("mass") is not None:
        lines.append(f"Mass: {float(state['mass']):.2f} kg")
    if state.get("fos_buckling") is not None:
        lines.append(f"Factor of Safety (buckling): {float(state['fos_buckling']):.2f}")
    if state.get("fos_yielding") is not None:
        lines.append(f"Factor of Safety (yielding): {float(state['fos_yielding']):.2f}")
    if state.get("deflection") is not None:
        lines.append(f"Maximum deflection: {float(state['deflection']):.4e} m")
    feasible = bool(state.get("is_feasible", False))
    lines.append(f"Status: {'FEASIBLE ✓' if feasible else 'INFEASIBLE ✗'}")

    if not feasible:
        viol = []
        fb, fy = state.get("fos_buckling"), state.get("fos_yielding")
        dfl = state.get("deflection")
        # Include WHICH member is critical (DesignBench's analyze_truss exposes this) so the
        # model targets the right member instead of guessing — the key information bottleneck.
        def _mtag(key):
            mid = state.get(key)
            if mid is None or mid == [] or mid == "":
                return ""
            ids = mid if isinstance(mid, list) else [mid]
            return f" — worst member(s): {', '.join('M'+str(i) for i in ids)}"
        if fb is not None and float(fb) < 1.5:
            viol.append(f"  - Buckling FOS {float(fb):.2f} < 1.5 (minimum required){_mtag('min_fos_buckling_member_id')}")
        if fy is not None and float(fy) < 1.5:
            viol.append(f"  - Yielding FOS {float(fy):.2f} < 1.5 (minimum required){_mtag('min_fos_yielding_member_id')}")
        # NOTE: an explicit mass-violation imperative ("EXCEEDS — reduce material") was tested
        # (mt_t4) and REGRESSED feasibility 68%→52% (over-conservative). Reverted. The mass
        # LIMIT is still shown on the Mass line for awareness without the strong imperative.
        if dfl is not None and float(dfl) > 0.01:
            viol.append(f"  - Deflection {float(dfl):.4e} m > 0.01 m (maximum allowed)")
        if viol:
            lines.append("\nConstraint violations:")
            lines.extend(viol)
    return "\n".join(lines)


def _structure_description(topology: dict) -> str:
    lines = []
    joints = topology.get("joints", [])
    lines.append(f"Joints ({len(joints)} total):")
    for idx, j in enumerate(joints):
        pos = j.get("position", [0.0, 0.0, 0.0])
        sup = _SUPPORT_MAP.get(j.get("support"), j.get("support") or "free")
        lines.append(f"  J{j.get('id', idx)}: pos=({pos[0]:.2f}, {pos[1]:.2f}) [{sup}]")

    members = topology.get("members", [])
    lines.append(f"\nMembers ({len(members)} total):")
    for idx, m in enumerate(members):
        j = m.get("joints", [None, None])
        mat = m.get("material", "Unknown")
        shape = m.get("shape", {})
        stype = shape.get("type", "Unknown")
        params = ", ".join(f"{k}={float(v):.4f}" for k, v in shape.items() if k != "type")
        lines.append(f"  M{idx}: J{j[0]}-J{j[1]} [{mat}, {stype}({params})]")
    return "\n".join(lines)


def build_problem_text(spec: dict) -> str:
    """Reproduce DesignBench ``_generate_comprehensive_problem_text`` from a spec."""
    lines = [f"PROBLEM: {spec.get('description', spec.get('problem_id', 'Truss optimization'))}", ""]
    lines.append("INITIAL STRUCTURE:")
    lines.append(_structure_description(spec.get("topology", {})))
    lines.append("")

    loading = spec.get("loading", [])
    if loading:
        lines.append("LOADING:")
        for load in loading:
            f = load.get("force", [0, 0, 0])
            lines.append(f"  Joint J{load.get('joint')}: Force=({f[0]:.0f}, {f[1]:.0f}, {f[2]:.0f}) N")
        lines.append("")

    goals = spec.get("goals", {})
    lines.append("DESIGN GOALS:")
    if "minimum_fos_buckling" in goals:
        lines.append(f"  - Minimum Factor of Safety (buckling): {goals['minimum_fos_buckling']}")
    if "minimum_fos_yielding" in goals:
        lines.append(f"  - Minimum Factor of Safety (yielding): {goals['minimum_fos_yielding']}")
    if "maximum_mass" in goals:
        lines.append(f"  - Maximum allowable mass: {float(goals['maximum_mass']):.2f} kg")
    lines.append("  - Objective: Minimize mass while satisfying FOS constraints")
    return "\n".join(lines)


def build_messages(spec: dict, initial_state: dict, action_history: list, formatter,
                   few_shot: bool = False) -> list[dict]:
    """Build the SFT-format conversation for a partial multi-turn rollout.

    Args:
        spec: problem spec dict.
        initial_state: env state after reset (initial FEA).
        action_history: list of {"action": str, "fea_result": state, "thinking": str}.
        formatter: ChatFormatter (used for assistant-turn formatting so prior
            turns reproduce the model's own <think>/<action> output format).
        few_shot: if True, append FEWSHOT_EXAMPLE to the system prompt (for
            format-enabling a base teacher model that won't emit <action> alone).
    """
    sys_content = TRUSS_SYSTEM_PROMPT + (FEWSHOT_EXAMPLE if few_shot else "")
    msgs = [{"role": "system", "content": sys_content}]
    msgs.append({
        "role": "user",
        "content": build_problem_text(spec) + "\n\nINITIAL STATE ANALYSIS:\n"
        + format_eval_result(initial_state),
    })
    for turn in action_history:
        msgs.append({
            "role": "assistant",
            "content": formatter._format_assistant_turn(turn["action"], turn.get("thinking", "")),
        })
        msgs.append({
            "role": "user",
            "content": "[Simulation Result]\nSTRUCTURAL ANALYSIS RESULT:\n"
            + format_eval_result(turn["fea_result"]),
        })
    return msgs
