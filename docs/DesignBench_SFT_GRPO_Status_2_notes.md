# DesignBench SFT + GRPO — Status Deck (v2)
## Per-slide talking points

_Companion to `DesignBench_SFT_GRPO_Status_2.pptx`. Use these bullets to check the narrative flows slide to slide._

### Slide 1 — Title — opening frame
- Open with the one-line thesis: this is a study of how to teach a mid-size (14B) open model to solve a hard engineering task — iterative truss design — and the headline finding is that the BOTTLENECK was the training environment, not the model's reasoning or the RL math.
- Anchor the audience on the single number that matters: 0% feasible after supervised warmstart → 68% feasible after the full framework. Everything else explains how and why.
- Set scope: Qwen3-14B with LoRA, a multi-turn grammar-action policy, trained with GRPO under a potential-based reward. Mention every claim is backed by a logged audit trail.
- Promise the structure: task → framework architecture → the two training stages → grounded results.

### Slide 2 — Executive summary
- This slide is the whole talk compressed — use it to give the audience a map before any detail.
- What we built: a clean two-stage pipeline (SFT then GRPO) wrapped around a multi-turn environment where the model edits the truss one grammar action at a time and a physics solver scores each edit.
- What it achieved: 0%→68% held-out feasibility; and critically, a strong zero-shot reasoner inside the SAME loop solves 4/6 with no training — foreshadowing the punchline.
- The one idea: don't frame the gains as bug fixes. The reasoning was always there; the wins came from designing the environment well — train/test protocol fidelity and a complete observation.
- Right rail gives four numbers you'll return to: 0%, 68%, 4/6, and α=5 (the single reward hyperparameter).

### Slide 3 — Agenda
- Four parts. Stress that the order is deliberate: the framework is presented as a designed system BEFORE any results, so the results read as validation rather than a debugging diary.
- Part 1 frames the problem (an MDP) and shows the architecture on one diagram.
- Parts 2–3 are the two training stages, each with its unique design decision (loss masking for SFT; rollout structure + reward for GRPO).
- Part 4 is the evidence: the progression, training dynamics, ablations, and the experiment that reframes the whole project.

### Slide 4 — Divider — Part 1
- Transition: 'Before any results, here is the problem and the machine that solves it.'
- Goal of this section: by the end the audience should be able to draw the architecture from memory.

### Slide 5 — The optimization task
- Define the task concretely: take an infeasible pin-jointed truss and edit it until it satisfies all constraints. This is DesignBench's iterative truss-optimization benchmark (20 problems).
- State = the full structure (joints, members, materials, pipe r/t cross-sections) plus the FEA verdict. Show the truss; point out member M4 (J6–J7) is the buckling-critical member under load P — we'll return to it.
- Feasible means THREE constraints hold simultaneously: FOS_buckling ≥ 1.5 (usually the binding one), FOS_yielding ≥ 1.5, and mass ≤ limit. A design that fixes buckling but blows the mass budget is still a failure — this tension drives later results.
- Action space: exactly one grammar edit per turn, e.g. SCALE_PARAM(4, r, 1.36). Six action types total. After each action the FEA solver re-scores — that is the environment.
- Why one-edit-per-turn matters: it turns design into a sequential decision problem (an MDP), which is what makes RL the natural tool — set up Part 1's architecture.

### Slide 6 — Framework architecture
- This is the slide the audience should remember. Walk it left to right: 5,000 DesignBench expert traces → SFT warmstart (format) → GRPO (policy) → a deployed 14B grammar-action agent.
- Emphasize the division of labour: SFT teaches the model HOW to speak (format/grammar), GRPO teaches it WHAT to do (the policy). They are not redundant — covered in Parts 2 and 3.
- The red chevron between SFT and GRPO marks where capability is actually acquired.
- The navy band is the key design principle: ONE multi-turn FEA environment, used identically during GRPO rollouts and during evaluation. think → act → simulate → observe.
- Foreshadow two environment-design choices that recur in results: (1) the loop is multi-turn at both train and test (protocol fidelity), and (2) the observation in 'observe' names the critical member. We do NOT call these 'fixes' — they are properties of a correct environment.
- Code anchors for Q&A: rollout loop multiturn_rollout.py:182–242; feedback string designbench_prompt.py:73–82.

### Slide 7 — The multi-turn environment
- Walk the left column as one turn of the loop: state s_t (infeasible) → the model writes <think> then exactly one <action> → the FEA solver executes and re-scores → a [Simulation Result] turn returns the verdict, which becomes s_{t+1}. Loop up to 5 turns or until feasible.
- Frame the two boxes on the right as DESIGN PRINCIPLES, not patches. Principle 1 — protocol fidelity: the model is trained with the same one-action-per-turn loop it runs at test time, so the learning signal targets the real behaviour. (Contrast briefly: an earlier single-turn protocol asked for the whole trajectory at once and the policy could not learn — but present this as 'why fidelity matters', not as a bug story.)
- Principle 2 — complete observation: the solver knows which member is about to buckle; the observation must include it ('worst member(s): M4'). This is the single most consequential design choice for accuracy.
- These two principles are exactly what the results section will quantify, so plant them firmly here.
- Code anchors: max_turns=5 / multi_turn=true in grpo_mt_01b.yaml; feedback construction designbench_prompt.py:73–82.

### Slide 8 — A real trace (replaces the old 'what is a single turn' slide)
- This is the slide the feedback explicitly asked for: an ACTUAL trajectory as a flowchart, not an abstract explanation of a conversation.
- Read the top row left→right: start infeasible (FOS_b 0.44, worst member M4) → think (locate M4) → action SCALE_PARAM(4,r,1.36) → FEA returns 0.48, still M4 → next action grows member 7 → turns 3–6 keep growing the critical members → FEASIBLE at FOS_b 1.69 in 7 turns.
- The 'what to notice' band is the takeaway: each turn the policy targets the member NAMED in the feedback. The competence is grounding, not raw cleverness.
- The code block shows the real turn-0 generation so the audience sees genuine CoT + a single grammar action — concrete, not schematic.
- Transition to Part 2: 'For any of this to work, the model first has to speak this format perfectly — that is what SFT does.'

### Slide 9 — Divider — Part 2
- Transition: the policy can only be learned if the model reliably emits the format. SFT is a lightweight warmstart for exactly that.
- Key message to set up: SFT teaches FORMAT, not strategy. The unique design decision is selective loss masking.

### Slide 10 — SFT data transform + purpose
- Explain the data first: 5,000 DesignBench expert traces (multi-turn, with <think> and step-level FEA feedback) at DesignBench/data/sft/train.jsonl.
- The WarmstartReasoningTarget reformats each raw trace into the exact inference-time conversation: (1) merge the problem spec + initial-state analysis into one user turn; (2) remap every FEA system message to a user turn prefixed '[Simulation Result]'; (3) extract the grammar action from inside <think> into its own <action> tag. (warmstart_transform.py)
- Why: training data must be token-for-token the shape the model will see at rollout. Same shape ⇒ the format learned transfers to deployment.
- State the division of labour explicitly: SFT teaches format/grammar/turn-taking; it does NOT teach which member to fix or how much — that is the policy, learned by GRPO. Keep SFT lightweight (low lr) so RL retains plasticity.
- This motivates the next slide: if SFT only supervises format, exactly which tokens get loss?

### Slide 11 — Selective loss masking — the SFT centerpiece
- This answers the feedback's explicit ask: 'what token is masked, what is not.'
- Show the token row: everything the model READS — the system prompt, the problem, the initial state, every [Simulation Result] — is masked to labels = −100 (grey, no gradient). Everything the model WRITES — <think>…</think> and <action>…</action> (and <answer> at the end) — is supervised with cross-entropy (green).
- The principle in one line: 'supervise what the model must generate; mask what it only reads.' This is implemented in get_loss_mask via regex spans over <think>/<action>/<answer> (targets.py).
- Why it matters: if you supervise the prompt/feedback (the naive full_sequence target, ~82% of tokens), the model overfits to reproducing the simulator's text and loses RL plasticity. Selective masking (~64%) keeps loss on the model's own tokens.
- Mention the research hook: the supervision target is pluggable (data.target_fn); default gold_curriculum_warmstart tightens supervision to the final turn for a curriculum. No training-infra change to swap.
- Note the ratios (64% vs 82%) are illustrative of the two targets from the loss-mask analysis figure.

### Slide 12 — SFT result — sets up GRPO
- Three numbers from the warmstart eval: grammar success 1.00, feasibility 0%, mean FOS_buckling 0.41 (results/eval/sft_warmstart_mt25).
- Reframe 0% as success: SFT's job was format, and grammar is perfect. The model can produce any valid action; it just lacks the policy for WHICH action. Don't let the audience read 0% as the framework failing.
- This is the clean hand-off to GRPO: a reliable grammar gives RL a well-formed action space and a stable reference policy to regularize against (the KL anchor). Capacity is there; RL supplies direction.
- Transition to Part 3: 'Now the interesting part — how GRPO turns perfect form into a working policy.'

### Slide 13 — Divider — Part 3
- Transition: GRPO is where the policy is actually learned. Three design pieces: how rollouts are sampled (group + tree), and how they are scored (potential reward).

### Slide 14 — GRPO group rollout
- Explain GRPO simply: for each problem, sample a GROUP of K=8 rollouts; the advantage of each is its reward minus the group mean, divided by the group std. No separate value network — the group is its own baseline.
- Walk the diagram: one problem → 8 rollouts, each gets a scalar reward → advantage A=(r−μ)/σ → policy-gradient update, regularized by a KL term (β=0.04) back to the SFT warmstart.
- Three reasons GRPO fits this setting: (1) no critic to train — cheaper and more stable for a 14B LLM; (2) ÷σ self-normalizes across problems whose FEA reward magnitudes differ a lot; (3) the KL anchor protects the grammar earned in Stage 1.
- Config anchors (grpo_base.yaml / grpo_mt_01b.yaml): group_size=8, kl_coef=0.04, lr 5e-6, 100 steps. grpo_trainer.py:159 sets num_generations=group_size.
- Next: a richer way to sample the group — tree expansion.

### Slide 15 — Tree-expanded / multi-branch rollout
- This answers the feedback's 'multi-branch rollout' ask. Beyond the flat group of 8, GRPO can expand each state into a shallow action TREE.
- Depth D=1, branching b=5: sample one candidate per action class (SCALE, ADD, MODIFY, REMOVE, MOVE) — stratified so the branches are genuinely different moves rather than near-duplicates.
- Each branch is scored by the same potential Φ; the advantage of branch k is γΦ(s_H^k) minus the MEAN potential over its sibling branches — a tighter, lookahead baseline (grpo_trainer.py:278–342; posterior/tree_expansion.py).
- Benefits: stratified coverage of the action space, lower-variance advantages, and bounded cost (~20 FEA calls/state, validated in verify_tree_expansion.py).
- Be honest: the 68% champion has tree expansion OFF (use_tree_expansion=false). Present it as an available variance-reduction mechanism in the framework, not as the source of the headline number.

### Slide 16 — Potential-based reward
- The framework deliberately avoids a 'reward zoo'. The champion uses ONE term: a potential-based per-step shaping reward r_env = γΦ(s′) − Φ(s).
- The potential Φ(s) = log(m₀/m) − α·V(s): a mass-utility term minus α times a smooth violation score V (softplus deficits on buckling FOS, yielding FOS, and deflection). (potential.py:47–68; rewards.py:388–420.)
- Two properties make it well-behaved: DENSE — every turn gets a gradient, no sparse terminal-only signal; and POLICY-INVARIANT — by Ng/Harada/Russell 1999, potential shaping telescopes to γᴴΦ(s_H) − Φ(s₀), so it cannot change which policy is optimal, only speed learning.
- It is physics-grounded (computed from the FEA state) and has effectively ONE hyperparameter: α=5 (constraint price), γ=0.99. We'll show α is non-monotone in the ablations.
- Config: reward_fn=composite with reward_weights.lagrangian_potential=1.0, cost_fn=null (grpo_mt_01b.yaml).

### Slide 17 — Engineering / systems
- Set expectations: multi-turn RL is expensive because the cost is in the rollout loop (generation + FEA), not the loss. ~511 s/step on 8×H100, 4–6k tokens/rollout, ~14 h for 100 steps (under the 24 h wall).
- The enabling trick: a turn-major BATCHED rollout. At each turn, all still-active rollouts are batched into one generate() call (left-padded, KV-cached, chunked by memory). Finished rollouts drop out so the batch shrinks. (multiturn_rollout.py:13–15, 58–117.)
- Why vLLM is disabled here: vLLM emits a whole completion in one shot, but the multi-turn contract is stateful — each turn's prompt depends on the prior FEA reply — so per-turn HF generate() is required. vLLM still serves the single-turn path. (grpo_trainer.py:211–215.)
- Frame the trade-off as principled: we give up generation throughput to keep train/test protocol fidelity — and fidelity is exactly what made the policy learnable.
- This closes the framework half of the talk; next is evidence.

### Slide 18 — Divider — Part 4
- Transition: with the framework defined, the results read as validation of the two environment-design principles from Part 1.

### Slide 19 — The 0%→68% progression
- This is the headline chart (real eval data, results/report/deck/eval_feas.png). Five configurations of ONE framework, GRPO and reward held fixed throughout.
- Read the steps on the right as an ablation of environment design, NOT as a list of bug fixes: (1) making the rollout multi-turn unfreezes learning, 0%→12%; (2) adding the critical-member to the observation is the decisive lever, 12%→68%; (3) adding a HARD mass rule over-corrects, 68%→52%.
- Hammer the structural point: the model, the optimizer, and the reward did not change between these bars. Only the environment did. That is the evidence for 'the bottleneck was the environment.'
- The +grammar bar (4%) is a preview of the ablations slide — adding a grammar reward actually hurt.
- Numbers are 25-problem evals (±~8%) except the champion, which has a 50-problem confirmation at 68%.

### Slide 20 — Training dynamics — KL
- Explain why KL, not reward: r_env magnitude is problem-dependent and each batch samples different problems, so the reward curve (middle of the original fig) is too noisy to read step-to-step. KL from the reference policy is the clean 'is it moving?' signal.
- Real data (kl_curves.png from logs/*/metrics.jsonl). Single-turn run is frozen at ~0.003 for all 200 steps — a flat policy. Multi-turn no-feedback is learnable but slow. The champion climbs to ~0.035 and crosses the 0.01 'is it learning?' bar by ~step 35.
- ≈10× more movement once the observation is complete — the point is that better information yields a SHARPER gradient, not merely a higher endpoint.
- The grammar-reward run spikes (clipped at top) — unstable; ties to the next slide's ablation.

### Slide 21 — Reward ablations
- Methodology: change exactly one thing per run (clean ablations logged in docs/ablation_ledger.md).
- α sweep (no feedback): feasibility is NON-monotone — 8% at α=2, 12% at α=5, 4% at α=10. This refutes the design-doc prediction that higher α is always better; α=10 over-weights the constraint and destabilizes the FOS landscape. Peak at α=5.
- Adding a grammar-compliance reward did NOT recover format and slightly hurt feasibility — grammar drift is fragility-induced (tiny policy moves destroy the thin r=32 warmstart format), not something a reward can fix.
- Adding an explicit 'mass exceeds limit, reduce material' rule made the policy over-conservative: it hugged FOS=1.5 and traded mass-overshoot fails for FOS-undershoot fails, 68%→52%.
- Verdict: the clean single-term reward at α=5 is best. Combined with the progression slide, the message is consistent — the leverage was the environment, not reward engineering.

### Slide 22 — Held-out evaluation detail
- Define the metrics precisely (eval_checkpoint.py:340–350): feasibility_rate = fraction of problems where all three constraints pass (is_feasible); grammar_success = fraction of turns whose action parses & executes; mean FOS_buckling.
- The grouped bars (real data) show the trade: warmstart has grammar 1.00 but FOS far below the line; training pushes FOS up while grammar drifts to ~0.45–0.67.
- Table contrasts warmstart vs champion directly: 0%→68% feasibility, 1.00→0.67 grammar, 0.41→1.38–1.63 FOS.
- Key nuance: the grammar–strength trade is a KNOB, not a law. ~33% of turns are wasted on grammar drift — that's recoverable headroom (KL-to-warmstart tuning), not a fundamental ceiling.
- Caveat: 25-problem cells carry ±~8%; champion confirmed at 68% on a 50-problem eval.

### Slide 23 — The information bottleneck — the punchline
- This is the intellectual climax. Run as a controlled experiment: take a strong 30B reasoner zero-shot in the same multi-turn loop. With the critical member named in the observation it solves 4/6; remove ONLY that field and it solves 0/6.
- The only variable is one field in the observation. A frozen, untrained model swings from 0/6 to 4/6. That isolates the cause: the limit was an incomplete observation, not reasoning ability or the RL algorithm.
- This is the justification for the talk's thesis and for framing the gains as environment design. RL's role is to amplify and stabilize this same competence into the trained 14B policy (68%).
- Tree-search corollary: a short search over edits reaches FOS 2.12 — the action space is rich enough; the policy just needs the right signal.
- Land the line: 'We didn't make the model smarter. We let it see what it needed to see.'

### Slide 24 — A solved trajectory
- Concrete close to the results: the champion solving auto_problem_000 in 7 turns, FOS_buckling 0.44 → 1.69, crossing the 1.5 line monotonically (status_report §3).
- Each step targets a critical member named by the feedback: step 0 grows M4, step 1 grows M7, and so on. No wasted edits on irrelevant members.
- This is the behaviour the whole study was chasing: grounded reasoning that acts on the right member every turn.
- Use it to transition from 'what works' to 'what still breaks' — 68% means ~32% still fail.

### Slide 25 — The remaining 32%
- Be candid about the ceiling: the 'add missing information to the observation' lever produced 12%→68% but has saturated (the mass tweak regressed).
- The remaining failures split three ways: (1) mass over-shoot on 3 problems — FOS fixed but mass exceeded; needs SOFT reward-side balancing, not a hard rule; (2) degenerate FEA on 4 — structurally broken or very heavy intermediate states; needs robust state handling; (3) grammar drift wastes ~33% of turns — the policy forgets the warmstart format; needs a format-protecting KL constraint.
- The unifying lesson from the ablations: hard prompt imperatives over-correct (mass), so the next gains are softer/structural changes — sets up the roadmap.

### Slide 26 — Roadmap
- Three priorities on the main path, each mapped to a failure mode from the previous slide: (1) soft FOS↔mass reward balancing; (2) a format-protecting KL constraint to stop grammar drift; (3) robust handling of degenerate FEA states.
- One reserve: distillation. The 30B teacher already produces trajectories that solve ~67%, so distilling those into the 14B is the fallback if the reward-side and KL changes plateau.
- Message: the path to >90% is concrete and prioritized — a list of named fixes, not open-ended research.

### Slide 27 — Key takeaways
- Close on the four lessons, in the order the talk built them.
- 1 — The bottleneck was the environment: with GRPO and the reward held fixed, protocol fidelity then a complete observation drove 0%→68%.
- 2 — SFT's contribution is form, achieved by selective loss masking (supervise written tokens, mask read tokens) — perfect grammar without sacrificing RL plasticity.
- 3 — Simplicity won on the algorithm side: GRPO with group/tree-relative advantages and one policy-invariant potential (α=5) beat every reward add-on.
- 4 — Information completeness beats raw reasoning capacity: the 0/6→4/6 zero-shot swing is the proof; RL amplifies that competence into the trained policy.
- End with the one-liner: we didn't make the model smarter — we built an environment that let a capable model show what it already knew.

