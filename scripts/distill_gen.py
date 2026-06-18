"""Two-model distillation trajectory generation (teacher reasons, student acts).

Per the 2026-06-16 strategy ([[project_cot_distillation_strategy]]): the advanced
teacher (qwen3_30b-Thinking) produces excellent CoT but expresses actions only in
prose; the warmstart student (qwen3_14b) emits valid grammar 100% of the time but
reasons weakly. So we combine them per turn:

  1. teacher reasons about the current state  → full CoT (what we distill)
  2. student is PREFILLED with that CoT and continues from "<action>" → grammar action
     (the assistant turn becomes <think>{teacher CoT}</think><action>{student action}</action>,
      which is simultaneously the rollout step AND the SFT distillation target)
  3. apply the action via FEA → next state; repeat to feasibility / max_steps.

Outputs: per-problem feasibility (does teacher-reasoning + student-grounding SOLVE the
task?) and a JSONL of distillation trajectories for SFT into qwen3_14b.

Usage (GPU job; 2xH100 for 30B teacher + 14B student):
  python scripts/distill_gen.py \
    --teacher-config configs/model/qwen3_30b_a3b_thinking_2507.yaml \
    --student-checkpoint checkpoints/sft/gold_warmstart_qwen3_14b_fixed_20260521_001258/final \
    --student-config configs/model/qwen3_14b.yaml \
    --problems-dir /ocean/.../DesignBench/data/problems \
    --out results/distill/traj.jsonl --max-problems 6 --max-steps 12
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/ocean/projects/mch250030p/wxu7/DesignBench")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _load_cfg(path):
    """Compose a model config YAML (resolves Hydra defaults: [base])."""
    from omegaconf import OmegaConf
    path = Path(path)
    cfg = OmegaConf.load(path)
    defaults = cfg.pop("defaults", []) or []
    merged = OmegaConf.create({})
    for d in defaults:
        if isinstance(d, str) and d != "_self_" and (path.parent / f"{d}.yaml").exists():
            merged = OmegaConf.merge(merged, _load_cfg(path.parent / f"{d}.yaml"))
    return OmegaConf.merge(merged, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-config", required=True)
    ap.add_argument("--student-config", default="configs/model/qwen3_14b.yaml")
    ap.add_argument("--student-checkpoint", required=True)
    ap.add_argument("--problems-dir", default="/ocean/projects/mch250030p/wxu7/DesignBench/data/problems")
    ap.add_argument("--out", default="results/distill/traj.jsonl")
    ap.add_argument("--max-problems", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--teacher-max-tokens", type=int, default=4096)
    ap.add_argument("--student-max-tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from llm_finetune.models.loader import load_model_and_tokenizer
    from llm_finetune.data.processors import designbench_prompt
    from llm_finetune.data.processors.chat_formatter import ChatFormatter, ThinkingMode
    from llm_finetune.envs.truss_env import (
        _analyze_truss, _apply_action, _load_truss_and_goals, parse_grammar_action,
    )
    from peft import PeftModel

    # ── load teacher (30B thinking) ──────────────────────────────────────────
    tcfg = _load_cfg(args.teacher_config)
    tcfg.device_map = "auto"; tcfg.use_gradient_checkpointing = False
    tcfg.use_compile = False; tcfg.use_lora = False
    log.info(f"Loading TEACHER {tcfg.model_name_or_path}")
    teacher, ttok = load_model_and_tokenizer(tcfg)
    teacher.eval()
    tfmt = ChatFormatter.from_model_id(tcfg.model_name_or_path, ttok, thinking_mode=ThinkingMode.QWEN3)

    # ── load student (14B warmstart adapter) ────────────────────────────────
    scfg = _load_cfg(args.student_config)
    scfg.device_map = "auto"; scfg.use_gradient_checkpointing = False
    scfg.use_compile = False; scfg.use_lora = False
    log.info(f"Loading STUDENT base {scfg.model_name_or_path} + adapter {args.student_checkpoint}")
    student, stok = load_model_and_tokenizer(scfg)
    student = PeftModel.from_pretrained(student, args.student_checkpoint).merge_and_unload()
    student.eval()
    sfmt = ChatFormatter.from_model_id(scfg.model_name_or_path, stok, thinking_mode=ThinkingMode.QWEN3)
    tdev = next(teacher.parameters()).device
    sdev = next(student.parameters()).device

    def gen(model, tok, ids, max_new):
        t = torch.tensor([ids], dtype=torch.long).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(t, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][t.shape[1]:], skip_special_tokens=True)

    problem_files = sorted(Path(args.problems_dir).glob("auto_problem_*.json"))[:args.max_problems]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = []
    fout = open(args.out, "w")

    for pf in problem_files:
        spec = json.load(open(pf))
        pid = spec.get("problem_id", pf.stem)
        truss, goals = _load_truss_and_goals(spec)
        state = _analyze_truss(truss, goals)
        initial_state = state
        action_history, traj = [], []
        for step in range(args.max_steps):
            if state.get("is_feasible"):
                break
            messages = designbench_prompt.build_messages(spec, initial_state, action_history, sfmt)
            # 1) teacher reasons
            t_ids = tfmt.apply_template(messages, add_generation_prompt=True, tokenize=True)
            cot = gen(teacher, ttok, t_ids, args.teacher_max_tokens).strip()
            # 2) student acts, PREFILLED with the teacher's CoT CONCLUSION, continuing from
            #    <action>. Prefill only the last ~250 words (the decision) — the full 4k-token
            #    CoT is OOD for the warmstart's ~50-tok think distribution. Full CoT kept for
            #    the distillation record.
            cot_tail = " ".join(cot.split()[-250:])
            s_prefix = list(sfmt.apply_template(messages, add_generation_prompt=True, tokenize=True))
            prefill = stok(f"<think>\n{cot_tail}\n</think>\n<action>", add_special_tokens=False)["input_ids"]
            act_gen = gen(student, stok, s_prefix + prefill, args.student_max_tokens)
            # Normalize teacher-contaminated member labels: the problem text renders members
            # as "M0,M1,..."; the teacher's reasoning uses "M6", and the student copies it, but
            # the grammar validator requires a numeric id. Strip the M prefix: M6 -> 6.
            import re as _re
            action = parse_grammar_action(_re.sub(r"\bM(\d+)\b", r"\1", "<action>" + act_gen))
            if step < 3:
                log.info(f"[{pid} step {step}] student_raw[:200]={('<action>'+act_gen)[:200]!r} parsed={action!r}")
            ok = False
            if action is not None:
                try:
                    truss = _apply_action(truss, action); state = _analyze_truss(truss, goals); ok = True
                except Exception as e:  # noqa: BLE001
                    log.debug(f"apply failed: {e}")
            traj.append({"step": step, "cot": cot, "action": action, "ok": ok,
                         "fos_b": state.get("fos_buckling"), "feasible": state.get("is_feasible")})
            action_history.append({"action": action or act_gen[:80], "fea_result": state, "thinking": cot})
            if step < 2:
                log.info(f"[{pid} step {step}] action={action!r} ok={ok} fos_b={state.get('fos_buckling'):.3f} "
                         f"cot[:160]={cot[:160]}")
        feasible = bool(state.get("is_feasible"))
        rec = {"problem_id": pid, "feasible": feasible, "n_steps": len(traj),
               "init_fos_b": initial_state.get("fos_buckling"), "final_fos_b": state.get("fos_buckling"),
               "trajectory": traj}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        results.append(rec)
        log.info(f"{pid}: feasible={feasible} fos_b {initial_state.get('fos_buckling'):.3f}"
                 f"->{state.get('fos_buckling'):.3f} steps={len(traj)} "
                 f"gram={sum(t['ok'] for t in traj)/max(len(traj),1):.2f}")

    fout.close()
    n = len(results)
    agg = {"n_problems": n, "feasibility_rate": sum(r["feasible"] for r in results) / max(n, 1),
           "mean_final_fos_b": sum((r["final_fos_b"] or 0) for r in results) / max(n, 1)}
    log.info(f"\n=== DISTILL-GEN AGG ===\n{json.dumps(agg, indent=2)}")
    json.dump({"aggregate": agg, "per_problem": [{k: v for k, v in r.items() if k != "trajectory"}
                                                  for r in results]},
              open(Path(args.out).with_suffix(".summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
