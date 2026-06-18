"""Generate figures for the status report (training dynamics + eval comparison)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/report"; os.makedirs(OUT, exist_ok=True)

# (label, metrics.jsonl path, color)
RUNS = [
    ("single-turn (broken protocol)", "logs/qwen3_14b_v2_01b_alpha5_20260522_2152/metrics.jsonl", "#999999"),
    ("multi-turn, no feedback",        "logs/mt_s1_01b_alpha5_20260613_1234/metrics.jsonl",        "#1f77b4"),
    ("multi-turn + grammar reward",    "logs/mt_t2_grammar_20260614_1012/metrics.jsonl",           "#2ca02c"),
    ("multi-turn + critical-member fb","logs/mt_t3_01b_critfeedback_20260616_0900/metrics.jsonl",  "#d62728"),
    ("+ mass feedback (regressed)",    "logs/mt_t4_01b_massfb_20260617_0439/metrics.jsonl",        "#9467bd"),
]

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("step") is not None and r.get("kl", 0) > 0]  # drop final zero summary
    return rows

# ---------- Figure 1: training dynamics (kl, reward, grad_norm) ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for label, path, c in RUNS:
    if not os.path.exists(path):
        continue
    r = load(path)
    s = [x["step"] for x in r]
    ax[0].plot(s, [x.get("kl", 0) for x in r], label=label, color=c, lw=2)
    ax[1].plot(s, [x.get("reward", 0) for x in r], label=label, color=c, lw=2, alpha=0.85)
    ax[2].plot(s, [float(x.get("grad_norm", 0)) for x in r], label=label, color=c, lw=2, alpha=0.85)
ax[0].axhline(0.01, ls="--", c="k", lw=0.8, alpha=0.5)
ax[0].set_ylim(0, 0.045)  # cap so the meaningful range is visible (grammar run spikes to 0.29, clipped)
ax[0].set_title("KL from reference (policy movement)"); ax[0].set_xlabel("GRPO step"); ax[0].set_ylabel("kl")
ax[0].annotate("0.01 'is it learning?' bar", (5, 0.0112), fontsize=8, color="k")
ax[0].annotate("single-turn: FLAT ~0.003\n(policy frozen)", (95, 0.004), fontsize=8, color="#555")
ax[1].set_title("Mean reward (r_env, noisy across problem batches)"); ax[1].set_xlabel("GRPO step"); ax[1].set_ylabel("reward")
ax[2].set_title("Grad norm"); ax[2].set_xlabel("GRPO step"); ax[2].set_ylabel("grad_norm")
ax[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("RL training dynamics across the tuning progression", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_training_dynamics.png", dpi=130); plt.close(fig)

# ---------- Figure 2: eval feasibility / grammar / FOS comparison ----------
EVALS = [
    ("warmstart\n(no RL)",            "results/eval/sft_warmstart_mt25/eval_results.json"),
    ("GRPO\nno feedback",             "results/eval/mt_s1_01b_alpha5_eval/eval_results.json"),
    ("GRPO\n+grammar",                "results/eval/mt_t2_grammar_eval/eval_results.json"),
    ("GRPO\n+crit-member\n(CHAMPION)","results/eval/mt_t3_01b_cf_final_eval/eval_results.json"),
    ("GRPO\n+crit+mass\n(regressed)", "results/eval/mt_t4_massfb_ckpt50_eval/eval_results.json"),
]
labels, feas, gram, fos = [], [], [], []
for name, p in EVALS:
    if not os.path.exists(p): continue
    a = json.load(open(p))["aggregate"]
    labels.append(name); feas.append(a["feasibility_rate"]); gram.append(a["mean_grammar_success_rate"]); fos.append(a["mean_final_fos_buckling"])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
bars = ax[0].bar(labels, feas, color=["#999","#1f77b4","#2ca02c","#d62728","#9467bd"][:len(labels)])
ax[0].set_ylabel("feasibility rate"); ax[0].set_title("Eval feasibility (held-out truss problems, multi-turn)")
ax[0].set_ylim(0, 1.0)
for b, v in zip(bars, feas): ax[0].text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.0%}", ha="center", fontweight="bold")
x = range(len(labels))
ax[1].bar([i-0.2 for i in x], gram, width=0.4, label="grammar success", color="#ff7f0e")
ax[1].bar([i+0.2 for i in x], [f/3 for f in fos], width=0.4, label="mean FOS_buckling /3", color="#17becf")
ax[1].axhline(1.5/3, ls="--", c="k", lw=0.8, alpha=0.6); ax[1].annotate("FOS=1.5 feasibility line", (0, 1.5/3+0.01), fontsize=8)
ax[1].set_xticks(list(x)); ax[1].set_xticklabels(labels, fontsize=8); ax[1].set_title("Grammar compliance & final FOS")
ax[1].legend(fontsize=8)
fig.suptitle("Evaluation results across the tuning progression", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_eval_comparison.png", dpi=130); plt.close(fig)

# ---------- Figure 3: loss + completion length ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for label, path, c in RUNS:
    if not os.path.exists(path): continue
    r = load(path); s = [x["step"] for x in r]
    ax[0].plot(s, [x.get("loss", 0) for x in r], label=label, color=c, lw=2, alpha=0.85)
    ax[1].plot(s, [x.get("completions/mean_length", 0) for x in r], label=label, color=c, lw=2, alpha=0.85)
ax[0].set_title("GRPO loss"); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss")
ax[1].set_title("Mean completion length (reasoning tokens/rollout)"); ax[1].set_xlabel("step"); ax[1].set_ylabel("tokens")
ax[0].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_loss_complen.png", dpi=130); plt.close(fig)

print("wrote:", os.listdir(OUT))
# also dump a compact metrics summary table
print("\n=== final-step training metrics per run ===")
for label, path, c in RUNS:
    if not os.path.exists(path): continue
    r = load(path)
    if r:
        last = r[-1]
        print(f"{label:34} steps={len(r):3} kl_final={last.get('kl',0):.4f} reward_final={last.get('reward',0):.1f} "
              f"kl_max={max(x.get('kl',0) for x in r):.4f}")
