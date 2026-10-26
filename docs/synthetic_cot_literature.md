# Synthetic chain-of-thought as SFT data: what the literature says, mapped onto the dataxp results

Compiled 2026-10-19. Six searchers (one per angle), then one adversarial checker per paper that opened the
primary source and confirmed title, first author, venue, year and the stated finding (66 papers found,
54 confirmed as stated, 12 confirmed with a corrected finding, 0 rejected). Venue tiers: **top** =
NeurIPS / ICML / ICLR / ACL / EMNLP / NAACL / AAAI / AISTATS / JMLR / TMLR / TACL / Nature / Science;
**second** = COLM, Findings, workshops, COLT; **preprint** = arXiv only. The synthesis below was written
from the verified list only; the empirical inputs are those of docs/sft_probe_rat2.md.

## Synthesis

**1. The answer**

Synthetic CoT beats action-only SFT when the trace is the computation that produced the action and the student cannot already do it: search traces (Yang et al., NeurIPS 2022; Hu & Clune, NeurIPS 2023; Su et al., ICLR 2025), a stronger reasoner's traces (Guo et al., Nature 2025; Muennighoff et al., EMNLP 2025), many rationales per example (Ho et al., ACL 2023; Li et al., ACL 2023; Guha et al., ICLR 2026), rationales naming a mediator the model must infer (Lampinen et al., ICML 2022; Prystawski et al., NeurIPS 2023), or an inherently serial label (Li et al., ICLR 2024; Wies et al., ICLR 2023). It hurts when the rationale is one post-hoc narration of a decision made elsewhere: a single rationale on a task answer-only already handles (Ho; Magister et al., ACL 2023; Li, −10 on OBQA), explain-then-predict (Camburu et al., NeurIPS 2018, −2.4), noisy rationales (Wiegreffe et al., EMNLP 2021, −4 to −6), a trace harder to predict than the move itself (Zhou et al., ICLR 2024), a fixed myopic teacher (Silver et al., Nature 2017; Anthony et al., NeurIPS 2017), and LoRA rather than pretraining-scale installation (Ye et al., ICLR 2025). rat2 is one post-hoc rationale per one-step argmax move, LoRA, 578 rows: every hurt condition, no help condition. The −0.10 is the predicted outcome, not a contradiction of the literature.

**2. Mechanism, paper by paper**

- Compounding error. Foster et al. (NeurIPS 2024): under log-loss BC the horizon is not the issue; per-step error on learner-visited states is. Swamy et al. (NeurIPS 2022): BC with history latches onto its own earlier outputs, and rat adds 270 tokens of its own text to every turn's history. Predicted the A-B-A-B flips (84 vs 53). Confirmed.
- Teacher-forced shortcut. Bachmann & Nagarajan (ICML 2024): the hard token is the choice among levers; rat2 narrates one lever and compares none, so its supervised tokens carry no comparison gradient. Bengio et al. (NeurIPS 2015) and Arora et al. (Findings ACL 2022, second tier): the action is then conditioned on a long self-generated prefix never present in training. Predicted accurate rationales (83% of predictions hold) with worse actions. Confirmed.
- Causal confusion. de Haan et al. (NeurIPS 2019): rat2 is written from the move and its outcome, so it sits downstream of the action like the brake light; the model learns "worst violation → direct lever" as the action's predictor. Predicted: the routine survives at test (56→58% openings), fires where Φ would not (heat sink 43 vs 12), and is worst when the shortcut is wrong (shuf −0.33/−0.46: the think block is conditioned on, not ignored). DX-28's undo routine is the same mechanism minus the outcome. Confirmed.
- Unfaithful post-hoc rationales. Turpin et al. (NeurIPS 2023), Jacovi & Goldberg (ACL 2020), Chen et al. (ICML 2024): 83% auditor approval is plausibility, not evidence the rationale is why argmax Φ chose the move. Pruthi et al. (TACL 2022) and Hase & Bansal (LNLS 2022 workshop, second tier): an explanation helps only if it is cheaper to learn from than from (state, move) pairs; "worst violation → lever" is computable from the pair. Predicted null-to-negative. Confirmed.
- Search traces vs optimal paths. Gandhi et al. (COLM 2024, second tier), Lehnert et al. (COLM 2024, second tier), Su et al.: the gain came from traces containing rejected alternatives. rat2 contains none. These predict no gain; the loss comes from the three mechanisms above.
- Expert iteration / RL. Chang et al. (ICML 2015): imitating a greedy reference is roll-in and roll-out with the reference and cannot exceed it. Chu et al. (ICML 2025): trajectory SFT memorises; that is thermal2 (untuned 101 > rat 76 ≈ act 73). Yue et al. (NeurIPS 2025): the base already samples the gold move 21-28% at the 79th-99th percentile, the sharpening regime where RL beats SFT. Confirmed on P2.

**3. Predictions**

(a) Search-shaped rationales. Matches the procedure-cloning condition (Yang), the sub-task decomposition of argmax Φ (Wies), one consistent step function (Joshi et al., COLT 2025, second tier), routing through real mediators (Prystawski); simulator-verified candidate predictions meet Kambhampati et al.'s (ICML 2024) sound-critic condition. Prediction: the P1 loss disappears; the gain over act is small, because a one-step argmax over 2-4 candidates is a shallow function (Feng et al., NeurIPS 2023; Saparov et al., ICLR 2025) and the search-trace advantage is mostly sample efficiency that shrinks with model size (Lehnert; Lee et al., ICLR 2024). It will not beat the untuned base on thermal2 by itself. Render several candidate orderings per move (Li; Guha), drop trace parts at random (Su: trace-drop beat full traces), and budget 4-10× tokens.

(b) Expert iteration + GRPO. This is what the theory prescribes: labels on learner-visited states (Ross et al., AISTATS 2011), learner roll-ins against a cost-sensitive reference (Chang), on-policy filtered rationales over external ones (Zelikman et al., NeurIPS 2022; Singh et al., TMLR 2024), minimal SFT then RL against the verifier (Chu), per-step over outcome reward (Wang et al., ICLR 2026; Lightman et al., ICLR 2024). Prediction: the largest held-out gain, and the only option that can close the P2 gap. Guards: train on the Φ-ranked distribution, not the argmax (Anthony: +50 Elo; Silver et al., Nature 2016: an argmax prior loses diversity); KL against collapse (Wang); expect most of the gain in round 1 (Singh); use Φ-rejected rollouts for a verifier (Hosseini et al., COLM 2024, second tier); STaR's hint-shortcut applies to Φ-selection, so audit rationales against counterfactual candidates rather than with auditors.

**4. What the literature does not settle**

- No paper tests post-hoc rationale SFT at 27B LoRA on a sequential task; Ye et al. is pretraining-scale math, Chu et al. an 11B vision model.
- Data regime for (a): Nye et al. (DL4C workshop 2022, second tier) flips at tens of thousands of traces, Lee at 1-4k, Su at 100k, s1 at 1k but with a stronger teacher. 578 rows is below all of them.
- Recoverability: Ross & Bagnell (AISTATS 2010) make the multiplier T² or kT depending on whether a wrong move is undone within k steps; k is unmeasured for design, and the flips show the learner can undo but not when to stop.
- Whether any SFT keeps the untuned base's held-out edge (Fu et al., ICML 2023 collapse; no mixed-format ablation exists).
- Faithfulness tests exist only for QA (Chen; Lanham et al., preprint); none for design moves.
- Answer-filtering nulls (Guha; Hosseini) are 7-13B math/code; whether Φ-selection in (b) transfers is untested.

## Verified papers

| # | tier | paper | venue | angle |
|---|---|---|---|---|
| 1 | top | Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models (Singh et al.) | Transactions on Machine Learning Research | distill |
| 2 | top | DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning (Guo et al. (DeepSeek-AI)) | Nature, vol. 645, pp. 633-638 | distill |
| 3 | top | Large Language Models Are Reasoning Teachers (Ho et al.) | ACL 2023 | distill |
| 4 | top | OpenThoughts: Data Recipes for Reasoning Models (Guha et al.) | ICLR 2026 | distill |
| 5 | top | STaR: Bootstrapping Reasoning With Reasoning (Zelikman et al.) | NeurIPS 2022 | distill |
| 6 | top | Specializing Smaller Language Models towards Multi-Step Reasoning (Fu et al.) | ICML 2023 | distill |
| 7 | top | Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step (Li et al.) | ACL 2023 — Proceedings of the 61st Annual Meeting of the Ass | distill |
| 8 | top | Teaching Small Language Models to Reason (Magister et al.) | ACL 2023 | distill |
| 9 | top | s1: Simple test-time scaling (Muennighoff et al.) | EMNLP 2025 | distill |
| 10 | top | Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations (Chen et al.) | ICML 2024 | faithfulness |
| 11 | top | Evaluating Explanations: How Much Do Explanations from the Teacher Aid Students? (Pruthi et al.) | Transactions of the Association for Computational Linguistic | faithfulness |
| 12 | top | Explanation-based Finetuning Makes Models More Robust to Spurious Cues (Ludan et al.) | ACL 2023 | faithfulness |
| 13 | top | Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting (Turpin et al.) | NeurIPS 2023 | faithfulness |
| 14 | top | Measuring Association Between Labels and Free-Text Rationales (Wiegreffe et al.) | EMNLP 2021 | faithfulness |
| 15 | top | Tell me why! Explanations support learning relational and causal structure (Lampinen et al.) | ICML 2022 | faithfulness |
| 16 | top | Towards Faithfully Interpretable NLP Systems: How should we define and evaluate faithfulness? (Jacovi & Goldberg) | ACL 2020 | faithfulness |
| 17 | top | e-SNLI: Natural Language Inference with Natural Language Explanations (Camburu et al.) | NeurIPS 2018 | faithfulness |
| 18 | top | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger) (Ross et al.) | AISTATS 2011 | imitation |
| 19 | top | A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (Silver et al.) | Science, vol. 362, issue 6419, pp. 1140-1144, 7 December 201 | imitation |
| 20 | top | Causal Confusion in Imitation Learning (de Haan et al.) | NeurIPS 2019 | imitation |
| 21 | top | Efficient Reductions for Imitation Learning (Ross & Bagnell) | AISTATS 2010 | imitation |
| 22 | top | Is Behavior Cloning All You Need? Understanding Horizon in Imitation Learning (Foster et al.) | NeurIPS 2024 | imitation |
| 23 | top | Learning to Search Better than Your Teacher (Chang et al.) | ICML 2015 | imitation |
| 24 | top | Mastering the game of Go with deep neural networks and tree search (Silver et al.) | Nature, vol. 529, pp. 484–489, 2016 | imitation |
| 25 | top | Mastering the game of Go without human knowledge (Silver et al.) | Nature, vol. 550, pp. 354–359, 19 October 2017 | imitation |
| 26 | top | Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks (Bengio et al.) | NeurIPS 2015 | imitation |
| 27 | top | Sequence Model Imitation Learning with Unobserved Contexts (Swamy et al.) | NeurIPS 2022 | imitation |
| 28 | top | Thinking Fast and Slow with Deep Learning and Tree Search (Anthony et al.) | NeurIPS 2017 | imitation |
| 29 | top | Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspective (Wang et al.) | ICLR 2026 | rl-process |
| 30 | top | Chain of Thoughtlessness? An Analysis of CoT in Planning (Stechly et al.) | NeurIPS 2024 | rl-process |
| 31 | top | Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (Yue et al.) | NeurIPS 2025 | rl-process |
| 32 | top | Let's Verify Step by Step (Lightman et al.) | ICLR 2024 | rl-process |
| 33 | top | Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks (Kambhampati et al.) | ICML 2024 | rl-process |
| 34 | top | Reasoning with Language Model is Planning with World Model (Hao et al.) | EMNLP 2023 | rl-process |
| 35 | top | SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training (Chu et al.) | ICML 2025 | rl-process |
| 36 | top | Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Parameters for Reasoning (Snell et al.) | ICLR 2025 | rl-process |
| 37 | top | Thinking vs. Doing: Improving Agent Reasoning by Scaling Test-Time Interaction (arXiv title: Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction) (Shen et al.) | NeurIPS 2025 | rl-process |
| 38 | top | Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al.) | NeurIPS 2023 | rl-process |
| 39 | top | Chain of Thought Imitation with Procedure Cloning (Yang et al.) | NeurIPS 2022 | search-traces |
| 40 | top | Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces (Su et al.) | ICLR 2025 | search-traces |
| 41 | top | Faith and Fate: Limits of Transformers on Compositionality (Dziri et al.) | NeurIPS 2023 | search-traces |
| 42 | top | Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems (Ye et al.) | ICLR 2025 | search-traces |
| 43 | top | Teaching Arithmetic to Small Transformers (Lee et al.) | ICLR 2024 | search-traces |
| 44 | top | The Pitfalls of Next-Token Prediction (Bachmann and Nagarajan) | ICML 2024 | search-traces |
| 45 | top | Thought Cloning: Learning to Think while Acting by Imitating Human Thinking (Hu and Clune) | NeurIPS 2023 | search-traces |
| 46 | top | Transformers Learn Shortcuts to Automata (Liu et al.) | ICLR 2023 | search-traces |
| 47 | top | Transformers Struggle to Learn to Search (Saparov et al.) | ICLR 2025 | search-traces |
| 48 | top | Chain of Thought Empowers Transformers to Solve Inherently Serial Problems (Zhiyuan Li et al.) | ICLR 2024 | theory |
| 49 | top | Sub-Task Decomposition Enables Learning in Sequence to Sequence Tasks (Noam Wies et al.) | ICLR 2023 | theory |
| 50 | top | The Expressive Power of Transformers with Chain of Thought (William Merrill et al.) | ICLR 2024 | theory |
| 51 | top | Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective (Guhao Feng et al.) | NeurIPS 2023 | theory |
| 52 | top | What Algorithms can Transformers Learn? A Study in Length Generalization (Hattie Zhou et al.) | ICLR 2024 | theory |
| 53 | top | Why think step by step? Reasoning emerges from the locality of experience (Ben Prystawski et al.) | NeurIPS 2023 | theory |
| 54 | second | Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes (Hsieh et al.) | Findings of the Association for Computational Linguistics: A | distill |
| 55 | second | LIMO: Less is More for Reasoning (Ye et al.) | COLM 2025 | distill |
| 56 | second | V-STaR: Training Verifiers for Self-Taught Reasoners (Hosseini et al.) | COLM 2024 | distill |
| 57 | second | Investigating the Benefits of Free-Form Rationales (Sun et al.) | Findings of the Association for Computational Linguistics: E | faithfulness |
| 58 | second | Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning (Paul et al.) | Findings of the Association for Computational Linguistics: E | faithfulness |
| 59 | second | When Can Models Learn From Explanations? A Formal Framework for Understanding the Roles of Explanation Data (Hase & Bansal) | First Workshop on Learning with Natural Language Supervision | faithfulness |
| 60 | second | Why Exposure Bias Matters: An Imitation Learning Perspective of Error Accumulation in Language Generation (Arora et al.) | Findings of the Association for Computational Linguistics: A | imitation |
| 61 | second | Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping (Lehnert et al.) | COLM 2024 | search-traces |
| 62 | second | Stream of Search (SoS): Learning to Search in Language (Gandhi et al.) | COLM 2024 | search-traces |
| 63 | second | A Theory of Learning with Autoregressive Chain of Thought (Nirmit Joshi et al.) | COLT 2025 — Proceedings of the 38th Annual Conference on Lea | theory |
| 64 | second | Show Your Work: Scratchpads for Intermediate Computation with Language Models (Maxwell Nye et al.) | ICLR 2022 Workshop on Deep Learning for Code | theory |
| 65 | preprint | Orca: Progressive Learning from Complex Explanation Traces of GPT-4 (Mukherjee et al.) | arXiv preprint 2306.02707, June 2023 | distill |
| 66 | preprint | Measuring Faithfulness in Chain-of-Thought Reasoning (Lanham et al.) | arXiv preprint, 2023 | faithfulness |

## Findings as verified

### 1. Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models

Singh et al.. Transactions on Machine Learning Research (TMLR), April 2024 (earlier version also at ICLR 2024 DPFM workshop). https://arxiv.org/abs/2312.06585

**Finding.** ReST-EM: sample 32 (MATH) or 64 (APPS) solutions per problem from PaLM 2, keep those passing a binary correctness check (ground-truth answer for MATH, test cases for APPS; at most 10 kept per problem), fine-tune the *base pretrained* model on them (re-starting from base every iteration), and repeat for a few iterations. Fine-tuning on these self-generated, filtered solutions "substantially outperforms" SFT on the datasets' human-written solutions, especially for PaLM 2-L (the SFT comparison is shown in Figures 2-3; the text gives no numeric margin over SFT). The point gains the paper states are ReST-EM's improvement over the starting (pretrained) model, reported to show that gains grow with model size: MATH test accuracy +5.94 for PaLM 2-S vs +6.34 for PaLM 2-L; APPS +5.6 for PaLM 2-S* vs +6.4 for PaLM 2-L. With majority voting over 64 samples on MATH, ReST-EM PaLM 2-L reaches 48.82 vs 44.02 for the base model. On APPS most of the gain comes from the first iteration and later iterations regress (attributed to overfitting on the small training set); on MATH improvements after the first iteration are small but the paper says MATH still benefits from multiple iterations. Iterating beats one big batch: a single iteration on a 3x-larger self-generated dataset gave 40.3% pass@1 vs 41.0% / 41.9% at iterations 2 / 3. Just 1,000 MATH questions already gave significant gains, and 4,000 questions did slightly worse than 2,000. The ReST-EM models also transferred to held-out GSM8K and HumanEval, and PaLM 2-L fine-tuned with ReST-EM scored on the Hungarian HS finals exam above all compared models except GPT-4.

**Relevance here.** Shows that correctness-filtered rationales sampled from the policy itself beat rationales written by a different process (humans), which is an argument for on-policy rationales over externally rendered ones for the gold moves. It also sets expectations: most of the gain arrives in one round and more questions or iterations can hurt.

### 2. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning

Guo et al. (DeepSeek-AI). Nature, vol. 645, pp. 633-638 (2025); published 17 September 2025 (received 14 Feb 2025, accepted 17 Jul 2025); DOI 10.1038/s41586-025-09422-z. https://doi.org/10.1038/s41586-025-09422-z

**Finding.** 800K SFT samples generated by DeepSeek-R1 (reasoning traces plus answers, with rejection sampling on correctness) used to fine-tune Qwen2.5 and Llama bases with no RL stage. DeepSeek-R1-Distill-Qwen-32B: AIME24 pass@1 72.6, cons@64 83.3, MATH-500 94.3, GPQA 62.1, LiveCodeBench 57.2. Controlled comparison on the same Qwen-32B base: 10K+ steps of large-scale RL (DeepSeek-R1-Zero-Qwen-32B) reached only 47.0 AIME pass@1 / 60.0 cons@64 / 91.6 MATH-500 / 55.0 GPQA / 40.2 LCB. Authors' conclusions: distilling a stronger model's traces into a ~30B model beats running RL on it, and going further still needs stronger bases and larger RL.

**Relevance here.** Strongest top-venue evidence that trace SFT at ~30B scale beats RL on the same base, but the traces came from a model that reasons better than the student on the target task. The analogy breaks when the 'teacher' is a one-step optimiser whose rationale is a narration of an argmax rather than a search the student could not do itself.

### 3. Large Language Models Are Reasoning Teachers

Ho et al.. ACL 2023 (Proceedings of the 61st Annual Meeting of the ACL, Volume 1: Long Papers, pp. 14852-14882, Toronto, July 2023; DOI 10.18653/v1/2023.acl-long.830). https://aclanthology.org/2023.acl-long.830/

**Finding.** Fine-tune-CoT: a 175B teacher writes zero-shot CoT for training questions, rationales are kept only if the final answer is correct, and GPT-3 students (0.3B/1.3B/6.7B) are fine-tuned on them. Against vanilla answer-only fine-tuning of the same 6.7B student, CoT fine-tuning won on 7 of 12 tasks (e.g. MultiArith 33.3 vs 15.0, Date Understanding 60.4 vs 14.4, Shuffled Objects 64.4 vs 33.8, Coin Flip 98.7 vs 72.0) but lost on 5 (CommonsenseQA 56.8 vs 76.2, StrategyQA 55.0 vs 65.2, SVAMP 12.7 vs 20.7, SingleEq 20.4 vs 24.3, AddSub 21.0 vs 25.2). 'Diverse reasoning' (64 teacher rationales per question instead of 1) lifted MultiArith 33.3->53.3 and SVAMP 12.7->30.3, pushing SVAMP and AddSub past answer-only; StrategyQA stayed below it. Answer-based filtering left 27.6% of 'correct' Date Understanding rationales with reasoning errors, and filtering by hand-checked rationale quality lowered student accuracy because it shrank the data.

**Relevance here.** Direct answer to 'when does it hurt': CoT SFT underperformed action-only SFT whenever the task was one where answer-only supervision already worked well (commonsense/strategy QA, simple arithmetic) and only one rationale per example was available; it helped on procedural/algorithmic tasks and when many diverse rationales per example were used. A single rendered rationale per gold move is the configuration that lost.

### 4. OpenThoughts: Data Recipes for Reasoning Models

Guha et al.. ICLR 2026 (oral). https://proceedings.iclr.cc/paper_files/paper/2026/hash/b010241b9f1cdfc7d4c392db899cef86-Abstract-Conference.html

**Finding.** 1,000+ controlled SFT ablations: each condition generates 31,600 examples with the full pipeline and fine-tunes Qwen2.5-7B-Instruct (Sec. 3). Answer filtering did not help (camera-ready Table 6, math-data sweep, overall Average column): 41.9±0.4 with no filtering (not compute-controlled, 63,200 pairs, i.e. 2x the data) vs 41.6±0.4 random filtering vs 40.0±0.5 GPT-4o-mini verification; the Math-Avg sub-column is 65.6 / 64.8 / 61.4. The paper concludes "We do not perform answer filtering because no filtering strategy outperformed the baseline, which uses all the answers", so the final OpenThoughts3 pipeline keeps unverified traces. Multiple answers per question (Table 5, science-data sweep, overall Average): exact-dedup with 16x sampling 36.2±0.5 vs exact-dedup 1x 35.0±0.4 vs no-dedup 1x 35.5±0.3 (Science-Avg sub-column 49.7 vs 47.5 vs 46.9); caption: fewer questions annotated more times performs similarly or better than more questions annotated fewer times; the final pipeline uses 16x for all domains (for math, 4x was best and 16x second). Teacher choice (Table 7): QwQ-32B as teacher beat DeepSeek-R1 by 1.9 (code sweep, 44.2 vs 42.3 overall Average) and 2.6 (math sweep, 44.2 vs 41.6) points, "despite the fact that QwQ-32B scores lower on average"; the paper states R1 outperforms QwQ-32B by 9%, 8% and 23% on CodeElo, GPQA Diamond and JEEBench (Table 28: 53.7 vs 44.3, 73.7 vs 65.0, 92.3 vs 69.9). Question filtering (Table 4): the best strategies are LLM-based, difficulty rating (GPT-4o-mini) for code and LLM response-length for math/science; the best strategy improved the overall average by 4% (math) and 6% (code) over the RANDOM-filtering baseline, and LLM-based filters beat classical fastText/embedding filters in all domains (no single number given for that gap). Scaling to 1.2M examples with QwQ-32B as teacher gives OpenThinker3-7B 53.3 AIME25 / 51.7 LCB 06/24-01/25 / 53.7 GPQA-Diamond, overall average 55.3 vs 42.9 for DeepSeek-R1-Distill-Qwen-7B (+12.4) (Table 1; the abstract rounds AIME25 to 53%).

**Relevance here.** Largest controlled study of what makes synthetic CoT work for SFT: more traces per question and harder questions matter, correctness filtering of the rationales does not, and teacher strength on the benchmark does not predict teacher usefulness. For a design policy this means the generator's trace style (how it explores and checks) matters more than whether each rationale ends at the gold move.

### 5. STaR: Bootstrapping Reasoning With Reasoning

Zelikman et al.. NeurIPS 2022 (Advances in Neural Information Processing Systems 35, Main Conference Track). https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html

**Finding.** GPT-J generates its own rationales, keeps only those whose final answer is correct, and fine-tunes on them; the 'rationalization' variant additionally feeds the correct answer as a hint and lets the model write a post-hoc rationale for problems it got wrong. CommonsenseQA: direct answer-only fine-tune 60.0%, STaR without rationalization 68.8%, with rationalization 72.5% (GPT-3 direct fine-tune 73.0%). GSM8K: answer-only 5.8%, STaR 10.1%, STaR+rationalization 10.7%. n-digit arithmetic: 89.5% with rationalization vs 76.3% for a 10K-example no-rationale baseline. Rationalization helps because it 'exposes the model to difficult problems which otherwise would not have appeared in its finetuning dataset.' Caveats stated by the authors: correctness is only assumed to imply rationale quality; in some runs the model learned that the answer always matched the hint and produced answers that ignored the reasoning; many kept rationales 'were not actually arguments for why the answer should be true.'

**Relevance here.** This is the closest published analogue to writing rationales for known-correct actions: rationales conditioned on the gold answer improved the policy over answer-only SFT, but only by a few points once self-generated rationales were already used, and the hint-shortcut failure (model copies the hinted answer, reasoning becomes decorative) is exactly the risk when every training rationale is written to justify a pre-chosen argmax move. Rationales here are on-policy (the model's own), which the other papers show matters.

### 6. Specializing Smaller Language Models towards Multi-Step Reasoning

Fu et al.. ICML 2023 (Proceedings of the 40th International Conference on Machine Learning, PMLR 202:10421-10430). https://proceedings.mlr.press/v202/fu23d.html

**Finding.** FlanT5 (250M/760M/3B/11B) fine-tuned on GSM8K chain-of-thought solutions distilled from code-davinci-002 (40 samples per training question, correct ones kept, ~130K examples; distribution matching on the teacher's top-5 token probabilities), with four data formats mixed during tuning (in-context/zero-shot x answer-only/CoT; Fig. 1B). GSM8K test accuracy before -> after (Table 1): 250M 3.0 -> 13.4; 760M 6.9 -> 20.2; 3B 13.5 -> 22.4; 11B 16.1 -> 27.1. OOD per dataset (Table 1; order 11B/3B/760M/250M): MultiArith +11.3/+18.3/+25.5/+22.7; ASDiv +1.1/+7.7/+13.7/+16.7; SVAMP -4.1/+6.1/+13.6/+10.4. The paper does not report an OOD average; computing the mean of the three OOD deltas per size gives +2.8 (11B), +10.7 (3B), +17.6 (760M), +16.6 (250M), and 11B SVAMP actually fell (39.7 -> 35.6). Cost: BigBench-Hard CoT accuracy collapsed at every size (11B 41.8 -> 0.0; 3B 35.8 -> 0.0; 760M 30.9 -> 0.3; 250M 25.9 -> 0.1); BBH answer-only 11B 47.4 -> 19.6, 3B 39.9 -> 3.2, 760M 30.3 -> 6.5, 250M 24.2 -> 3.1. Data format (Fig. 4, Sec. 4.3): tuning only on in-context-format data preserves both in-context and zero-shot ability; tuning only on zero-shot-format data raises zero-shot accuracy but gradually loses in-context learning ability. The authors mix formats for this reason and recommend it, but report no mixed-vs-unmixed ablation. Base checkpoint (Table 2, GSM8K validation accuracy, not test; raw T5 reported only at 250M/760M/3B, no 11B): FlanT5 ends higher than raw T5 (3B 23.8 vs 20.6; 760M 21.8 vs 16.2; 250M 15.2 vs 14.2) while raw T5's absolute gain is larger (3B +19.9 vs +10.3; 760M +15.4 vs +14.9; 250M +12.4 vs +12.2).

**Relevance here.** Documents the main failure mode of narrow CoT SFT: the policy's general reasoning collapses even while the target task improves. A LoRA-tuned 27B policy trained only on one design domain's rationales should be checked for the same collapse, since the design task relies on the base model's numeric and physical reasoning that the rationale data may overwrite.

### 7. Symbolic Chain-of-Thought Distillation: Small Models Can Also "Think" Step-by-Step

Li et al.. ACL 2023 — Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Toronto, July 2023, pp. 2665–2679, DOI 10.18653/v1/2023.acl-long.150. https://aclanthology.org/2023.acl-long.150/

**Finding.** OPT-1.3B students distilled from code-davinci-002 (GPT-3 175B) CoTs, N=30 rationales sampled per training instance at T=1.0 by default. Full-data (supervised) accuracy, label-only vs one greedy CoT vs SCoTD (Table 2b): CommonsenseQA 63.0 / 68.2 / 67.0; QuaRel 59.0 / 71.2 / 83.8; OpenBookQA 60.2 / 50.0 / 67.0. On OpenBookQA one greedy rationale per example was 10.2 points below label-only and SCoTD was 6.8 points above it; on CommonsenseQA, however, one greedy CoT (68.2) beat 30 sampled CoTs (67.0). In the supervised setting the authors DID apply correctness filtering (dropping teacher CoTs whose predicted label was wrong), which discarded 40.4% (CSQA), 45.0% (OBQA) and 34.2% (QuaRel) of the 30 samples, so the Full-data SCoTD rows were trained on roughly 16-20 correct CoTs per instance, not 30; the few-shot rows use unfiltered CoTs and self-consistency decoding. N=50 gave no further gain over N=30 (67.2/84.9/67.0). In the Section 4 downsampling ablation on the correct-only corpus with a budget of ~5 CoTs per instance (Figure 7): diversity (S-BERT clustering) and teacher-likelihood filters modestly beat random subsampling on all three tasks (QuaRel 79.1 and 80.6 vs 74.8; OBQA 61.9 and 62.6 vs 61.4; CSQA 65.1 and 64.8 vs 64.5), while an open-endedness filter was below random on all three (71.9 / 58.6 / 63.3); none of the 5-per-instance subsets approached the full set (83.8 / 67.0 / 67.0). The authors characterize random subsampling as "a strong baseline" with "some evidence that diversity among the rationales is important", conclude that "the sheer volume of the CoTs is a key driving force", and state in the abstract that "sampling many reasoning chains per instance from the teacher is paramount."

**Relevance here.** Clean label-only control for a 1.3B student: one rationale per gold label can hurt, many rationales per gold label help, and which rationales are kept matters less than how many. For a 12-20 turn design task this argues for several distinct rationales per gold move rather than one canonical rendering, or expecting no gain.

### 8. Teaching Small Language Models to Reason

Magister et al.. ACL 2023 (Proceedings of the 61st Annual Meeting of the ACL, Volume 2: Short Papers), pages 1773–1781, Toronto, July 2023; DOI 10.18653/v1/2023.acl-short.151. https://aclanthology.org/2023.acl-short.151/

**Finding.** T5 students fine-tuned on PaLM-540B or GPT-3 CoTs, with teacher CoTs that did not match the gold answer discarded. GSM8K, T5-XXL: baseline (no CoT) 8.11% -> 21.99% (PaLM CoTs), 18.42% (GPT-3 CoTs), 19.94% (dataset's human CoTs); MAWPS 54.2 -> 70.4. StrategyQA went the other way: baseline 68.12%, PaLM CoTs 67.15%, GPT-3 CoTs 63.77%, while the dataset's human-written CoTs gave 71.98%. Giving the teacher the gold answer when generating CoTs raised teacher CoT accuracy from 59.98% to 79.37%; the authors say most of that gain came from fixing missing or wrong steps rather than copying the answer. A 220M student trained on CoTs matched the untrained 11B baseline.

**Relevance here.** Shows both regimes in one paper: synthetic CoT helped a lot on multi-step arithmetic but synthetic (teacher-written) CoT for StrategyQA scored below no-CoT fine-tuning while human CoT scored above it, i.e. the sign of the effect depends on whether the rationale carries real information the student lacks. It also reports the effect of answer-conditioned rationale generation, the same trick as rendering rationales for a known gold move.

### 9. s1: Simple test-time scaling

Muennighoff et al.. EMNLP 2025 (main conference), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20275-20321, Suzhou, Nov 4-9 2025, DOI 10.18653/v1/2025.emnlp-main.1025. https://aclanthology.org/2025.emnlp-main.1025/

**Finding.** Qwen2.5-32B-Instruct SFT'd on 1,000 question/trace pairs (traces distilled from Gemini 2.0 Flash Thinking) selected for quality (no API/format errors), difficulty (drop questions Qwen2.5-7B or 32B already solves; prefer long traces) and diversity (50 domains). 26 minutes on 16 H100s, 5 epochs. AIME24 / MATH500 / GPQA: 1K-random 36.7 / 90.6 / 52.0; 1K-diverse 26.7 / 91.2 / 54.6; 1K-longest 33.3 / 90.4 / 59.6; s1K 50.0 / 93.0 / 57.6; full 59K 53.3 / 92.8 / 58.1 (AIME difference vs s1K has bootstrap CI [-13.3, +20.0], so not significant) at 394 vs 7 H100-hours. Budget forcing ('Wait') extended AIME24 from 50% to 57%. No answer-only SFT baseline is reported; the comparison is against the un-tuned base and o1-preview.

**Relevance here.** At the same model scale as the question (32B), a few hundred to a thousand hard, long, diverse traces from a genuinely stronger reasoner were enough and 59x more data added nothing; 'random' or 'longest' selection alone lost 13-17 AIME points. The traces here come from a reasoner that out-thinks the student, which a greedy optimiser's narrated argmax does not.

### 10. Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations

Chen et al.. ICML 2024 (Proceedings of the 41st International Conference on Machine Learning, PMLR 235:7880-7904); ICML virtual site lists it as "Spotlight Poster". https://proceedings.mlr.press/v235/chen24bl.html

**Finding.** Simulation precision (how often an explanation lets a simulator predict the model's output on LLM-generated counterfactual inputs) is 77.3% (GPT-3.5) / 81.1% (GPT-4) on StrategyQA and 86.3% / 93.0% on SHP reward modelling. CoT vs post-hoc explanations differ by at most ~3 points and inconsistently in sign. Precision is uncorrelated with human plausibility (Pearson 0.012, Spearman 0.021), so optimising explanations for human approval (RLHF) does not make them predictive of the model's behaviour.

**Relevance here.** Supplies the counterfactual test you would actually want for a design policy: does the rationale predict what the model does on neighbouring states? The paper's result that CoT-before-answer and explanation-after-answer are equally (un)predictive undercuts the intuition that putting the rationale before the action makes it causal.

### 11. Evaluating Explanations: How Much Do Explanations from the Teacher Aid Students?

Pruthi et al.. Transactions of the Association for Computational Linguistics (TACL), vol. 10, pp. 359–375, 2022 (DOI 10.1162/tacl_a_00465; published April 2022). https://aclanthology.org/2022.tacl-1.21/

**Finding.** Frames explanation value as the simulation-accuracy gain of a student trained to mimic a teacher when explanations are available only at training time, never at test time. With a BERT-base IMDb teacher, top-10% attention explanations raise a BERT-base student from 91.5 to 93.9 at 1k examples (attention regularization, Table 5) and a BiLSTM student from 71.1 to 78.1 at 4k (multitask learning, Table 6; layer conductance is marginally higher at 79.5 there). For QA (Natural Questions / QED), human-expert explanations raise a BERT-base student from 28.9 to 37.3 F1 at 500 examples via attention alignment loss, and the gain shrinks to +5.0 at 2,500 examples, the largest size available for expert explanations (Table 4). The '+0.6 at 16k' figure is from a separate experiment with a SpanBERT QA model as teacher (Table 7): +7.9 F1 at 250 examples (25.0 to 32.9) falls to +0.6/+0.7 at 16k (61.6 to 62.2/62.3), which the authors describe as the benefit vanishing. Trivial label-leaking explanations do not outperform the no-explanation control by the authors' statement (significantly worse for BERT students; numerically higher but not statistically significant for the plain BiLSTM multitask student, 77.4 vs 71.1 at 4k).

**Relevance here.** Explanations-as-communication theory with the two conditions that matter for the design setting: explanations help most in the low-data regime and vanish as (x, y) data grows, and an explanation that only re-encodes the label is worthless. With 5k multi-turn traces and a deterministic argmax-Φ teacher, the framework predicts a small, data-size-dependent gain at best.

### 12. Explanation-based Finetuning Makes Models More Robust to Spurious Cues

Ludan et al.. ACL 2023 (Proceedings of the 61st Annual Meeting of the ACL, Volume 1: Long Papers, Toronto, July 2023, pp. 4420-4441, DOI 10.18653/v1/2023.acl-long.242). https://aclanthology.org/2023.acl-long.242/

**Finding.** Finetuning GPT-3 davinci to emit a free-text explanation with its answer, on training sets with an artificially inserted spurious cue, reduces the accuracy drop when the cue is removed by +1.2 (ComVE), +9.1 (CREAK), +15.4 (e-SNLI), +6.5 (SBIC) relative to answer-only finetuning. The benefit holds across model families and grows with model size, and model-generated explanations work in place of human ones.

**Relevance here.** The strongest positive case for explanation-as-target: the gain is in robustness to shortcut features, not in-distribution accuracy, and it is larger for larger models. If the greedy-optimiser traces contain shortcut regularities (e.g. 'always scale member 3 first'), rationale supervision may stop a 27B LoRA policy from locking onto them; if the task has no such cues, this mechanism predicts little change.

### 13. Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting

Turpin et al.. NeurIPS 2023 (Advances in Neural Information Processing Systems 36), Main Conference Track, poster. https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html

**Finding.** Adding biasing features (e.g. reordering few-shot options so the correct answer is always '(A)') shifts GPT-3.5 and Claude 1.0 answers and drops accuracy by up to 36% across 13 BIG-Bench Hard tasks; the CoT never mentions the bias and instead rationalises the biased answer. On a social-bias task the CoT justifies stereotype-consistent answers without mentioning the stereotype.

**Relevance here.** Direct evidence that CoT text is routinely post-hoc rationalisation of a decision reached by other means. For a policy whose gold move is argmax Φ, a model can learn to emit a fluent rationale that is decoupled from the mechanism that picks the move; rationale plausibility on held-out states is therefore not a proxy for better actions.

### 14. Measuring Association Between Labels and Free-Text Rationales

Wiegreffe et al.. EMNLP 2021 (main conference; Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10266–10284, Anthology ID 2021.emnlp-main.804). https://aclanthology.org/2021.emnlp-main.804/

**Finding.** T5-base trained to jointly produce label and free-text rationale (I→OR) vs label only (I→O): e-SNLI 90.95 → 90.81 (−0.14), CoS-E v1.0 69.16 → 64.84 (−4.32), CoS-E v1.11 61.75 → 55.61 (−6.14); authors state 'adding rationalization results in some loss in accuracy', largest where the human rationales are noisiest (CoS-E). The joint model's label and rationale do degrade together under input noise (robustness equivalence), so they are associated even though accuracy fell.

**Relevance here.** Shows that jointly training on rationales can cost task accuracy, with the cost scaling with rationale noise. For synthetic rationales generated by another model, the CoS-E case (low-quality rationales, −4 to −6 points) is the risk scenario; the e-SNLI case (clean rationales, ~0 change) is the best case.

### 15. Tell me why! Explanations support learning relational and causal structure

Lampinen et al.. ICML 2022 (Proceedings of the 39th International Conference on Machine Learning, PMLR 162:11868-11890), spotlight. https://proceedings.mlr.press/v162/lampinen22a.html

**Finding.** RL agents in odd-one-out and causal-intervention environments trained with an auxiliary loss to predict language explanations of task structure and of why their choice was right/wrong: 2D relational task 91.3% vs 61.9% without explanations; 3D 92.7% vs 29.5% (chance); causal experiments 96.9% vs 24.0% (easy), 90.5% vs 24.6% (hard); deconfounded OOD generalisation shifts to the explained feature (95.5% colour / 87.5% shape / 86.2% texture). Explanations were prediction targets, not inputs; providing them as inputs 'was not beneficial' and interfered with the target benefit. Behaviour-irrelevant explanations helped less, context-irrelevant ones not at all.

**Relevance here.** The closest published analogue to the design-agent setting (sequential decisions, simulator feedback, sparse reward) and the clearest positive result: explanations help enormously when they name the latent relational/causal structure the agent must infer and are tied to the agent's own choices. It also specifies the failure modes: explanations that are irrelevant to the agent's behaviour or to the task structure give nothing, and feeding them as context rather than as a target can hurt.

### 16. Towards Faithfully Interpretable NLP Systems: How should we define and evaluate faithfulness?

Jacovi & Goldberg. ACL 2020 (Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, main conference, pp. 4198–4205). https://aclanthology.org/2020.acl-main.386/

**Finding.** Opinion piece separating plausibility ("how convincing the interpretation is to humans") from faithfulness ("how accurately it reflects the true reasoning process of the model"). Their one illustrative example of a method that satisfies one property without the other is post-hoc text generation, where a separate "generator" component is "learned with supervision of textual explanations" (citing Zaidan & Eisner 2008, Rajani et al. 2019, Strout et al. 2019): "plausibility is the dominating property, while there is no faithfulness guarantee." Via a fictional construction (a heat-map explainer that emits random content words when the model is right and random punctuation when it is wrong), they argue that in HCI/IUI evaluations measured by END-TASK USER PERFORMANCE, "any degree of correlation between plausibility and model performance will result in increased user performance, regardless of any notion of faithfulness" — so user performance is unrelated to faithfulness. Separately (Section 5 guideline), they argue human judgement should not be used for faithfulness evaluation because humans cannot judge faithfulness ("if they understood the model, interpretation would be unnecessary") and no supervision can be obtained, so human judgement "measures plausibility." They also call for replacing the binary faithfulness notion with a graded one.

**Relevance here.** The theoretical frame for the whole question: a rationale written after the fact for a greedy-optimiser move is a plausibility artefact, not a description of why the move is correct, so any benefit from training on it must come from somewhere other than 'teaching the model the reason'. It also warns that human-rated rationale quality is not evidence the rationale carries the decision-relevant information.

### 17. e-SNLI: Natural Language Inference with Natural Language Explanations

Camburu et al.. NeurIPS 2018 (Advances in Neural Information Processing Systems 31, pp. 9539-9549). https://proceedings.neurips.cc/paper/2018/hash/4c7a167bb329bd92580a99ce422d6fa6-Abstract.html

**Finding.** 570k human-written explanations added to SNLI. Training to predict the label and generate the explanation (PredictAndExplain) gives 83.96% vs 84.01% for the explanation-free InferSent baseline (no gain); generating the explanation first and predicting from it (ExplainThenPredict) drops to 81.59% / 81.71% (−2.3 to −2.4 points), which the authors describe as 'sacrificing a bit of performance'.

**Relevance here.** The classic null-to-negative result for explanation-as-training-target: supervising the rationale alongside the label did nothing for label accuracy, and conditioning the label on a generated rationale hurt. Explain-then-predict is the exact structure of a <think>…</think> then action policy.

### 18. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)

Ross et al.. AISTATS 2011 (Proceedings of the 14th International Conference on Artificial Intelligence and Statistics, PMLR / JMLR W&CP vol. 15, pp. 627-635, Fort Lauderdale, FL). https://proceedings.mlr.press/v15/ross11a.html

**Finding.** Theorem 2.1 (restated supervised bound): J(pi) <= J(pi*) + T^2 eps. Theorem 3.2: for DAgger with N = O~(uT) iterations there is a pi_hat with J(pi_hat) <= J(pi*) + u T eps_N + O(1), where u bounds the increase in expert cost-to-go from a single non-expert action and u <= 1 for 0-1 loss; finite-sample version needs N ~ u^2 T^2 log(1/delta). Super Tux Kart: supervised performance did not improve with more data, SMILe still fell about twice per lap after 20 iterations, DAgger never fell after 15 iterations. Super Mario Bros: DAgger 3030 distance (beta_i = 0.5^{i-1}), supervised described as 'such a low score'. OCR: supervised with previous-character features 83.6%, DAgger 85.5%.

**Relevance here.** The remedy for compounding error is labels on states the learner itself visits, not richer labels on the expert's states. Synthetic CoT written on gold trajectories is purely off-policy and inherits the T^2 regime; because the greedy argmax-Phi teacher can be queried on any state the policy reaches, DAgger-style relabelling is cheap in this setting and is the intervention the theory actually recommends.

### 19. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play

Silver et al.. Science, vol. 362, issue 6419, pp. 1140-1144, 7 December 2018 (DOI 10.1126/science.aar6404). https://doi.org/10.1126/science.aar6404

**Finding.** Training target is the MCTS visit distribution pi (policy) and the game outcome z (value), loss (z - v)^2 - pi^T log p + c||theta||^2; the search trace itself is never a target. 700,000 steps from random initialisation (44M chess games, 24M shogi); surpassed Stockfish after 4 h (300k steps), Elmo after 2 h (110k), AlphaGo Lee after 30 h (74k). Chess: 155 wins, 6 losses in 1,000 games vs Stockfish; shogi: 91.2% vs Elmo; Go: 61% vs the 3-day AlphaGo Zero. AlphaZero searched ~60k positions/s vs ~60M for Stockfish.

**Relevance here.** The 'reasoning process' that generated the supervision was a search whose output is a policy-improved distribution over actions plus an outcome; nothing about the search's intermediate reasoning was learned as text. The transferable recipe for the design task is to train on Phi-ranked distributions over candidate parameter changes and on episode outcomes, which is a different object from a rationale written to justify a single gold action.

### 20. Causal Confusion in Imitation Learning

de Haan et al.. NeurIPS 2019 (Advances in Neural Information Processing Systems 32), oral. https://papers.nips.cc/paper/9343-causal-confusion-in-imitation-learning

**Finding.** Formalises 'causal misidentification': expert actions depend on causal state variables, but because the imitator's past actions influence both state and nuisance variables, nuisance correlates (effects of the action, e.g. a brake light) stay correlated with the action in demonstrations; a cloner fits them, gets low training error, and fails under distribution shift, so 'access to more information can yield worse performance'. CONFOUNDED inputs (state plus previous action) need many more demonstrations than ORIGINAL on MountainCar, Hopper and Pong or never match; on Pong CONFOUNDED has lower validation loss but lower reward. MountainCar: disentangled vs entangled policy -137 vs -145 (policy-execution intervention), -140 vs -165 (expert queries). DAgger needs hundreds of expert queries on MountainCar and tens of thousands on Hopper to reach comparable reward. Cited GTA-V driving table: with history 144.92 distance / 6.49 collisions vs without history 268.95 / 3.38.

**Relevance here.** A rationale written for a known-correct action is causally downstream of that action, the same structure as the brake light. Once rationales sit in the multi-turn context (previous turns' CoT plus the current turn's self-generated CoT), the model can learn to read the action off textual correlates rather than off the simulator feedback. The diagnostic the paper gives is directly usable: lower training or validation loss with worse rollout reward.

### 21. Efficient Reductions for Imitation Learning

Ross & Bagnell. AISTATS 2010 (Proceedings of the 13th International Conference on Artificial Intelligence and Statistics, Chia Laguna, Sardinia; JMLR W&CP vol. 9, pp. 661-668). https://proceedings.mlr.press/v9/ross10a.html

**Finding.** Theorem 2.1: if the cloned policy has 0-1 error eps under the expert's state distribution d_pi*, then J(pi_hat) <= J(pi*) + T^2 eps, and the bound is tight (a construction incurs (1-eps T) J(pi*) + T^2 eps). Forward training gives J(pi_T) <= J(pi*) + T eps u, where u is how much the expert's cost-to-go rises after one deviation; when the expert can recover within k steps the bound is k T eps_bar (linear in T). SMILe needs O(T^2 (ln T)^{3/2}) iterations to reach O(T eps_A + 1). Super Tux Kart (steering from pixels): the supervised approach needed 45 laps of data before reasonable performance; SMILe (alpha=0.1) matched that after 15 laps and was much better after 45. Mario Bros: supervised was 'only good on the easiest difficulty levels'; SMILe performed better at harder difficulties.

**Relevance here.** This is the formal source of 'compounding error' for per-step behaviour cloning: with 12-20 turns the worst-case multiplier on per-step error is T^2 = 144-400, but the paper itself says the multiplier collapses to kT when the expert can undo a mistake in k steps. Whether a rationale helps or hurts enters only through eps (per-step error under the learner's class) and the recoverability of the design task (one parameter change per turn is usually reversible), not through any separate mechanism.

### 22. Is Behavior Cloning All You Need? Understanding Horizon in Imitation Learning

Foster et al.. NeurIPS 2024 (Advances in Neural Information Processing Systems 37, Main Conference Track; listed as Spotlight Poster). https://arxiv.org/abs/2407.15007

**Finding.** For a deterministic expert, log-loss behaviour cloning satisfies J(pi*) - J(pi_hat) <= 8 R log(2|Pi|/delta)/n with probability 1-delta, where R is the range of cumulative reward: horizon-independent for sparse reward, linear in H for dense reward with parameter-sharing (stationary) policy classes, O(H^2 log max_h |Pi_h| / n) only without parameter sharing. Indicator (0-1) loss BC has gap R H log(|Pi|/delta)/n, quadratic in H under dense reward, and both steps are tight. Lower bound: a 2-state, 2-action MDP with |Pi| = 2 on which any online or offline IL algorithm suffers >= c H / n, so DAgger-style online IL cannot beat offline log-loss BC without extra assumptions on the policy class. Experiments: Walker2d imitator regret independent of horizon, BeamRider slightly improves with horizon, transformer language generation largely independent of H.

**Relevance here.** The modern statement for cross-entropy SFT (what LoRA on a 27B model optimises): the quadratic horizon penalty is an artefact of 0-1 analysis, and the imitation gap is governed by supervised complexity log|Pi| and realizability. Adding a rationale changes exactly those two things: it can lower the effective complexity of the state-to-action map or it can break realizability if the rationale is not computable from what the policy observes. So the theory predicts CoT helps iff it reduces per-step supervised error for this policy class, and that the 12-20 turn horizon is not itself the issue.

### 23. Learning to Search Better than Your Teacher

Chang et al.. ICML 2015 (Proceedings of the 32nd International Conference on Machine Learning, Lille; PMLR / JMLR W&CP vol. 37, pp. 2058-2066). https://proceedings.mlr.press/v37/changb15.html

**Finding.** Theorem 1: rolling in with the reference policy admits a distribution where cost-sensitive regret goes to zero but the structured loss gap to the reference stays constant, because the learner never visits states caused by its own mistakes. Theorem 2: rolling out with the reference admits arbitrarily large regret to one-step deviations. LOLS (roll-in with learner, mixed roll-outs) bounds beta * (regret to reference) + (1-beta) * (regret to one-step deviations) with no assumption on reference quality. KDD Cup 99 cost-sensitive classification with a deliberately bad reference: roll-in/roll-out with reference 1.670 average cost vs roll-in with learned policy 0.266. Dependency parsing UAS with a bad reference: reference/reference 68.7, LOLS 89.4, SEARN 63.4; with an optimal reference (POS tagging) all roll-in choices are within 95.58-95.61.

**Relevance here.** A greedy one-step argmax of Phi is a suboptimal reference on a 12-20 step problem. Pure imitation of its traces, with or without rationales, is roll-in and roll-out with the reference, which can at best match it and by Theorem 1 need not even do that. Improving beyond the greedy teacher needs the learner's own roll-ins and cost-sensitive targets; since Phi is a computable cost here, that is available and is what the theory says to spend compute on rather than on prose about the argmax.

### 24. Mastering the game of Go with deep neural networks and tree search

Silver et al.. Nature, vol. 529, pp. 484–489, 2016 (issue dated 28 January 2016; published online 27 January 2016; doi:10.1038/nature16961). https://www.nature.com/articles/nature16961

**Finding.** SL policy network trained on 29.4M KGS positions (160,000 games) predicted held-out expert moves at 57.0%; small accuracy gains gave large strength gains. The RL policy network (policy-gradient self-play initialised from SL) won more than 80% of games against the SL network and 85% against Pachi with no search, yet inside MCTS 'the SL policy network performed better in AlphaGo than the stronger RL policy network, presumably because humans select a diverse beam of promising moves, whereas RL optimizes for the single best move'. A value network regressed on complete KGS games memorised outcomes (MSE 0.37 test vs 0.19 train, because successive positions differ by one stone but share the target); trained on 30M positions each from a distinct self-play game it reached 0.226 / 0.234.

**Relevance here.** Three lessons for the design setting: imitation accuracy and policy strength are different quantities; a prior trained on a single deterministic target (the Phi argmax) loses the move diversity that makes a policy useful inside tree-of-thought search; and 12-20-turn traces with a near-constant per-trajectory target are exactly the correlated-sample structure that produced memorisation in the KGS value network.

### 25. Mastering the game of Go without human knowledge

Silver et al.. Nature, vol. 550, pp. 354–359, 19 October 2017 (doi:10.1038/nature24270). https://www.nature.com/articles/nature24270

**Finding.** A network of identical architecture trained by supervised learning on KGS amateur games reached 60.4% move-prediction accuracy on the KGS test set and 54.3% on the GoKifu professional validation set, versus 49.0% (20-block, 3 days) and 51.3% (40-block, 40 days) for the self-play RL network on GoKifu (RL networks were evaluated on GoKifu only; Extended Data Table 1). The SL player had better initial Elo and, early in training, lower MSE on human game outcomes, but by the end the RL network's outcome MSE was lower (GoKifu: 0.177 for RL 20-block and 0.180 for RL 40-block vs 0.207 for SL; Extended Data Table 2, Fig. 3c). The published text states: "Supervised learning achieved a better initial performance, and was better at predicting human professional moves (Fig. 3). Notably, although supervised learning achieved higher move prediction accuracy, the self-learned player performed much better overall, defeating the human-trained player within the first 24 h of training." AlphaGo Zero (20 blocks, 4.9 million self-play games, ~3 days, 1,600 MCTS simulations/~0.4 s per move) beat AlphaGo Lee 100–0; the 40-block, ~40-day instance (29 million games) reached 5,185 Elo vs 4,858 for AlphaGo Master, 3,739 for AlphaGo Lee and 3,144 for AlphaGo Fan, and beat AlphaGo Master 89–11. The introduction states that even reliable expert data sets "may impose a ceiling on the performance of systems trained in this manner".

**Relevance here.** Higher fidelity to the teacher's decisions (60.4% vs 49-51%) went with a weaker policy. Synthetic CoT optimises the imitation metric for the teacher's choice; when the teacher is a myopic one-step optimiser, closer imitation moves the policy toward the teacher's ceiling rather than toward the optimum, and improvements show up only when the training target is produced by a process that improves on the current policy.

### 26. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks

Bengio et al.. NeurIPS 2015 (Advances in Neural Information Processing Systems 28, then called NIPS 2015). https://papers.nips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html

**Finding.** Training conditions each token on the true previous token while inference conditions on the model's own token, so early errors feed back and the model enters states never seen in training. MSCOCO captioning: baseline BLEU-4 28.8 / CIDEr 89.5, scheduled sampling 30.6 / 92.1, always-sampling collapses to 11.2 / 49.7. Constituency parsing F1 86.54 -> 88.08. Speech: teacher-forced next-step frame error rate got worse (15.0 -> 19.8) while free-running decoding FER improved (46.0 -> 42.0).

**Relevance here.** The teacher-forced token metric and the free-running metric move in opposite directions, so per-token loss on gold rationales says nothing about rollouts. With CoT, the action token is conditioned on a self-generated rationale prefix of hundreds of tokens that was never in the training prefix distribution; the longer the synthetic rationale, the more exposure the decision has to the model's own drift. The always-sampling collapse also warns that naive on-policy prefixes without a schedule can fail.

### 27. Sequence Model Imitation Learning with Unobserved Contexts

Swamy et al.. NeurIPS 2022 (36th Conference on Neural Information Processing Systems), poster; Advances in Neural Information Processing Systems 35. https://proceedings.nips.cc/paper_files/paper/2022/file/708e58b0b99e3e62d42022b4564bad7a-Paper-Conference.pdf

**Finding.** Contextual MDP where the expert sees a hidden context c and the learner sees only the history. On-policy imitation (DAgger-style or reward matching) has limiting imitation gap <= eps_on(infinity) * H, i.e. it matches the expert's time-averaged value once its moment-matching error vanishes; off-policy behaviour cloning is bounded only by the sum of errors over the whole episode, and Theorem 5.5 shows this is tight even when the asymptotic error is zero (early mistakes persist: 'latching'). In a causal-bandit toy, off-policy consistency shows sharp phase transitions while on-policy methods are consistent wherever the problem is identifiable. PyBullet HalfCheetah (expert return 560) and Ant with target velocity revealed at t=200 (expert 300), history length 5: adding history reduces BC return, DAgger-SEQ shows no decline; abstract: 'off-policy approaches actually perform worse when given access to history'.

**Relevance here.** The argmax-Phi teacher is an expert with privileged context (it evaluates Phi on candidate moves; the policy only sees the transcript and simulator messages). The policy must infer that context from history, and BC with history latches onto its own earlier actions. Synthetic CoT that narrates Phi is an attempt to write the hidden context into the observable history; it can only help if the narrated quantities are inferable from what the policy actually sees at test time, otherwise it is a label the model cannot reproduce on-policy and on-policy correction is what fixes it.

### 28. Thinking Fast and Slow with Deep Learning and Tree Search

Anthony et al.. NeurIPS 2017 (Advances in Neural Information Processing Systems 30, "NIPS 2017"), pp. 5360-5370. https://proceedings.neurips.cc/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html

**Finding.** Expert Iteration: an apprentice network is trained by imitation on moves chosen by an MCTS expert (10,000 simulations per move) that is itself guided by the apprentice, with DAgger-style aggregation on apprentice-visited states. On 9x9 Hex: tree-policy targets (cross-entropy to the normalised visit distribution) beat chosen-action targets (log-loss on the argmax move) by 50 +/- 13 Elo at equal top-1 error (47.7% vs 47.0%); three further DAgger batches of 100k moves added about 120 Elo; the apprentice won 87 of 162 games against the MCTS it imitated, i.e. it plateaued at its expert. ExIt learned stronger policies faster and more stably than REINFORCE over 5 runs; adding a value head over ~7.4M additional move choices significantly outperformed policy-only ExIt. Final agent beat MoHex 1.0 75.3% at 10k iterations, 59.3% at 100k.

**Relevance here.** Imitation of a fixed expert ceilings at that expert (87/162 vs its own MCTS); the gains came from (a) an expert that is improved by searching on top of the apprentice and (b) training on the search's full distribution rather than a single argmax action. The analogue of 'synthetic CoT' that helped here is a distributional or value target produced by a stronger search, not a textual explanation of why the argmax is right; a one-step argmax-Phi teacher is the fixed-expert case.

### 29. Benefits and Pitfalls of Reinforcement Learning for Language Model Planning: A Theoretical Perspective

Wang et al.. ICLR 2026 (poster). https://arxiv.org/abs/2509.22613

**Finding.** Graph-planning abstraction (Erdős-Rényi 100 nodes, p=0.15, ~20% of reachable pairs with K=10 demonstration paths each). Theorem 3.1: SFT learns the empirical co-occurrence frequencies of the demonstrations ('SFT memorizes co-occurrence relationships in the training dataset') and so cannot use transitivity; continual SFT test accuracy 'constantly decreases'. Policy gradient reaches 100% training accuracy but suffers diversity collapse (eventually 'only one path per pair') unless KL-regularised, which trades training accuracy for diversity; PG beats SFT 'primarily because its iterative data generation process encourages exploration and effectively expands the training dataset'. Q-learning with process (per-step) rewards gives 'significantly better test accuracy than the PG model', while Q-learning with outcome rewards 'collapses ... to near-zero accuracy'. Blocksworld check (4 blocks, 73-node graph, 50k SFT paths): SFT 'may not learn the complete adjacency', Q-learning 'nearly recovers' it.

**Relevance here.** The theoretical account the user asked for: SFT on gold paths learns demonstration co-occurrence statistics, not the transition structure, and adding text to each step does not change that objective; generalizable planning comes from exploration plus per-step rewards, which is precisely what a simulator returning Φ at every turn provides.

### 30. Chain of Thoughtlessness? An Analysis of CoT in Planning

Stechly et al.. NeurIPS 2024 (poster). https://neurips.cc/virtual/2024/poster/93898

**Finding.** Blocksworld table-to-stack (261 instances), GPT-4: zero-shot 3.83%, zero-shot CoT 4.98%, problem-class-specific n-shot 8.81%, 'universal algorithm' CoT 51.3%, stacking-specific CoT 59.3%; GPT-4-Turbo 19.1 → 21.0 → 40.6; Claude-3-Opus 9.96 → 10.34 → 24.5. Gains appear only with prompts 'exceedingly specific to their problem class' and vanish once the query stack size exceeds the demonstrated sizes; self-consistency does not change this (24.5% / 26.4%). On scaled synthetic domains with GPT-4-Turbo, manual CoT gives CoinFlip 98.89%, LastLetterConcat 51.06%, multi-step arithmetic 50.43%, degrading with step count; an incorrect CoT prompt scores 96.76% on CoinFlip, about the same as the correct one (98.89%).

**Relevance here.** Direct evidence that demonstrated reasoning traces do not teach a general procedure: the model learns problem-specific surface patterns that break beyond the demonstrated size, and incorrect traces work as well as correct ones. Rationales written after the fact for optimiser-chosen moves are this kind of trace; expect any gain to be in-distribution pattern matching, not transfer to longer or different design episodes.

### 31. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

Yue et al.. NeurIPS 2025 (Oral; Best Paper Runner-up). https://arxiv.org/abs/2504.13837

**Finding.** Across Qwen2.5 and LLaMA families, six RLVR algorithms and math/code/visual benchmarks, RL-trained models beat the base at small k but the base model overtakes them at pass@k for k in the tens to hundreds. Omni-MATH-Test, Qwen2.5-7B: base pass@1 10.2 / pass@256 69.1 vs GRPO 25.1 / 68.3; during GRPO training pass@1 rises 26.1 → 42.5 while test pass@256 falls 68.3 → 63.9. AIME24 at k=1024: 13.3% of problems solved only by the base model, 0% only by the RL model; all 19 problems the RL model solves are among the 23 the base solves. Minerva-32B base is ~9% above RL at k=128. In contrast, DeepSeek-R1-Distill-Qwen-7B's pass@k is 'consistently and significantly above' its base at all k, i.e. distillation adds reasoning patterns not in the base distribution.

**Relevance here.** Separates the two mechanisms the user is choosing between: RL re-weights behaviours already in the base model (sharpening), while SFT on a stronger teacher's traces can add new behaviours. Synthetic CoT is worth it when the trace source does something the 27B model cannot already sample; if the model already produces the Φ-argmax moves at some rate, RL against the simulator is the cheaper, non-narrowing route.

### 32. Let's Verify Step by Step

Lightman et al.. ICLR 2024 (poster), Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=v8L0pN6EOi

**Finding.** With a GPT-4-base generator and 1,860 samples per problem on a MATH test subset, a process-supervised reward model (PRM, trained on 800k human step labels) selects a correct answer 78.2% of the time vs 72.4% for an outcome-supervised RM (ORM) and 69.6% for majority voting, and the gap widens with N. Out-of-distribution (224 recent AP Calculus/Chemistry/Physics and AMC questions): PRM 72.9% vs ORM 63.8% vs majority 61.3%, PRM best in every category. Active learning on step labels is ~2.6x more data-efficient. Note: Math-Shepherd (Wang et al., ACL 2024, aclanthology.org/2024.acl-long.510) reproduces PRM > ORM with step labels generated automatically by Monte-Carlo completions (no human labels): Mistral-7B step-level PPO reaches GSM8K 84.1 / MATH 33.0 vs ORM-PPO 81.8 / 31.3 vs base 77.9 / 28.6.

**Relevance here.** Establishes that credit assigned per step beats credit assigned per outcome, and that step credit can be obtained from rollouts rather than written rationales. In the user's setting the simulator already returns an exact per-step signal (ΔΦ), so the marginal value of rationale text is the question; this literature says the gain comes from the step-level signal, not from the prose.

### 33. Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks

Kambhampati et al.. ICML 2024 (Proceedings of the 41st International Conference on Machine Learning, PMLR 235, pp. 22895–22907; Spotlight Poster). https://proceedings.mlr.press/v235/kambhampati24a.html

**Finding.** Summarises their planning benchmarks: GPT-4 produces valid Blocksworld plans on 34.3% (one-shot) / 34.6% (zero-shot) of 600 instances, falling to 4.3% / 0.16% when action and object names are obfuscated (Mystery Blocksworld); 'fine-tuning does not seem to have a major effect on this dismal performance'; CoT and ReAct are 'largely ineffective'; iterative self-critique makes results worse because the model cannot recognise a correct solution. With an external sound verifier (VAL) back-prompting the LLM, Blocksworld rises to 82% within 15 rounds and Logistics to 70% (Mystery ~10%). The paper argues LLMs 'can't self-improve by generating synthetic data' from their own critiques, but that plans vetted by sound critics would form 'a better corpus of synthetic data' for fine-tuning.

**Relevance here.** States the critique position with numbers: fine-tuned LLMs do approximate retrieval of plans; correctness comes from the sound verifier in the loop. It draws the line the user needs: synthetic training data is useful when each item is vetted by a sound critic (an argmax-Φ move is), whereas a rationale written to justify that move is not vetted by anything.

### 34. Reasoning with Language Model is Planning with World Model

Hao et al.. EMNLP 2023 (main conference), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore, pp. 8154-8173. https://aclanthology.org/2023.emnlp-main.507/

**Finding.** Blocksworld plan-generation success on 2/4/6-step problems: CoT with LLaMA-33B 0.17/0.02/0.00, CoT with GPT-4 0.50/0.63/0.40, RAP(10 iterations) 1.00/0.86/0.26, RAP(20) 1.00/0.88/0.42 (64% average; LLaMA-33B+RAP beats GPT-4+CoT by 33% relative). RAP repurposes the LLM as a world model (predict next state after an action) and runs MCTS with task rewards (goal conditions met) and the LLM's action likelihood. GSM8K with LLaMA-33B: CoT 29.4%, CoT+SC(10) 46.8%, RAP(10) 48.6%, +aggregation 51.6%.

**Relevance here.** In a sequential state-changing task, what moved success from near zero to near one was state prediction plus reward-guided search, not better verbal rationales. The user's task has an exact world model (the simulator) and an exact potential Φ, so an LLM rationale would be an imitation of what the environment already computes.

### 35. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training

Chu et al.. ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:10818-10838). https://proceedings.mlr.press/v267/chu25c.html

**Finding.** Llama-3.2-Vision-11B in multi-turn 'sequential revision' episodes with verifier feedback (GeneralPoints arithmetic cards, V-IRL navigation). Under rule changes (OOD): RL with outcome reward improves GP-L +3.5 (11.5 → 15.0) and V-IRL-L +11.0 (80.8 → 91.8), while SFT on trajectories drops GP-L −8.1 (→ 3.4) and V-IRL-L −79.5 (→ 1.3); visual OOD: V-IRL-VL RL +61.1 (16.7 → 77.8) vs SFT −5.6; V-IRL mini-benchmark SOTA 44.0 → 77.8. RL from the raw base model: 'all trials fail' (no parseable actions), so SFT is required to fix the output format; RL could not recover OOD performance from an over-fitted SFT checkpoint. More verification turns per episode raise OOD gains (+0.48 at 1 step, +2.99 at 5, +5.99 at 10).

**Relevance here.** The closest published analogue of the user's loop (one action per turn, verifier feedback, 11B model). SFT on full trajectories memorised the training rules; outcome-reward RL generalised; SFT's only necessary role was establishing the action format, and too much of it blocked later RL. Argues for a minimal format-SFT followed by RL against the simulator, not a large synthetic-CoT SFT.

### 36. Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Parameters for Reasoning

Snell et al.. ICLR 2025 (Oral). https://arxiv.org/abs/2408.03314 ; https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html ; https://iclr.cc/virtual/2025/oral/31924

**Finding.** PaLM 2-S* on MATH with a PRM trained from Monte-Carlo rollouts without human labels (PRM800K labels were 'largely ineffective' for their model). PRM-guided beam search beats best-of-N on medium-difficulty problems (levels 3-4) and at low budgets, but on easy problems (levels 1-2) beam search gets worse as budget grows, attributed to over-exploitation of the PRM; on level-5 problems no method makes meaningful progress. Sequential self-revision narrowly beats parallel sampling, easy questions benefit most from revision, and a naive revision policy turned ~38% of correct answers into incorrect ones. Compute-optimal allocation by difficulty gives >4x (2-4x in the discussion) efficiency over best-of-N. FLOPs-matched against pretraining, test-time compute beats a 14x larger model only on easy/intermediate questions and low inference load; pretraining wins on the hardest bins.

**Relevance here.** Whatever 'reasoning at inference' buys is delivered by a verifier plus search, and even a learned verifier over-optimizes on easy cases; on problems beyond the base model nothing at test time helps. For a 12-20-turn design loop with a real simulator, the analogue is search against the simulator, which argues against expecting rationale text alone to add capability.

### 37. Thinking vs. Doing: Improving Agent Reasoning by Scaling Test-Time Interaction (arXiv title: Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction)

Shen et al.. NeurIPS 2025 (Main Conference Track, poster). https://papers.nips.cc/paper_files/paper/2025/hash/f7c4783621a20f5f11a316cd249e0252-Abstract-Conference.html

**Finding.** WebArena subset (h=30): action-only prompting 14.76% vs per-step CoT 23.81%; adding 'check again' interaction steps raises success to ≥28%. Scaling per-step reasoning (budget forcing) plateaus at ~0.26 and changes the chosen action only ~15% of the time; per-step best-of-n gives <3% at matched compute; at a fixed token budget, spending it on more environment steps has the steepest gain, and over-rechecking/over-thinking trends downward (agents revise correct answers). TTI, a curriculum online-RL method that lengthens the interaction horizon for Gemma 3 12B: WebVoyager 64.8% vs zero-shot 55.8%, fixed-horizon RL h=10 59.1%, h=30 45.2%; WebArena 26.1% vs 18.3%. Per-step token counts fell during training as trajectories grew longer.

**Relevance here.** In multi-turn environments with feedback, the useful 'reasoning' is more interaction with the environment, not longer per-step thought; extra per-step CoT rarely changes the action and can hurt. For a 12-20-turn design episode with simulator feedback, this favours spending capacity on acting and reading feedback over learning to narrate each move.

### 38. Tree of Thoughts: Deliberate Problem Solving with Large Language Models

Yao et al.. NeurIPS 2023 (Advances in Neural Information Processing Systems 36, Main Conference Track, Oral). https://proceedings.neurips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html

**Finding.** On Game of 24, GPT-4 with chain-of-thought prompting solves 4% of instances; the same model with explicit branching over intermediate 'thoughts', self-evaluation of partial states, lookahead and backtracking (ToT) solves 74%. Similar gains reported on Creative Writing and Mini Crosswords.

**Relevance here.** A linear chain of reasoning text is nearly useless on a search problem; the 18x gain comes from an explicit search procedure with state evaluation. The reasoning that helps is a procedure, not a narrative, and in the user's setting the evaluation step is already performed by the FEA simulator.

### 39. Chain of Thought Imitation with Procedure Cloning

Yang et al.. NeurIPS 2022 (Advances in Neural Information Processing Systems 35, Main Conference Track, poster). https://proceedings.neurips.cc/paper_files/paper/2022/hash/ebdb990471f653dffb425eff03c7c980-Abstract-Conference.html

**Finding.** Procedure cloning (PC) imitates the expert's intermediate computations (BFS expansion sequence, MCTS rollouts) in addition to the action; behavioral cloning (BC) imitates actions only. Discrete maze with BFS traces: PC 100% on unseen test mazes vs BC 0% (BC ~100% on training mazes); in 32x32 mazes vanilla BC 0% success and auxiliary-loss / augmentation BC variants help only in the small maze. Image-based bimanual sweep, 1000 demos: PC 83.9% vs 78.2% for implicit BC. AntMaze and MinAtar (MCTS traces): PC 'consistently provides significant benefits over vanilla BC' and 'generalizes the best across all games and all settings' (values in figures).

**Relevance here.** The sequential-decision version of the question: when the demonstrator is a search procedure, cloning its computation generalizes to unseen configurations while action-only cloning memorizes the training layouts. The trace that helps is the expert's actual computation; for a greedy argmax-Φ expert the analogue is the enumerated candidate evaluations, not a verbal explanation.

### 40. Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces

Su et al.. ICLR 2025 (Poster). https://arxiv.org/abs/2410.09918

**Finding.** Same maze/Sokoban setup as Searchformer, 100k training examples, 15M (maze) / 46M (Sokoban) params, identical data budget across conditions. Maze 30x30 1-Optimal: Solution-Only 30.0%, Complete-Trace (Searchformer) 93.3%, Dualformer trained on randomly dropped traces: slow mode 97.6% with 854 vs 1538 average trace tokens, fast mode (no trace emitted at inference) 80.0%. Maze 15x15: Solution-Only 72.0, Complete-Trace 94.6, Dualformer fast 91.8. Sokoban 1-Optimal: Solution-Only 86.8, Complete-Trace 92.9, Dualformer fast 97.3 / slow 94.5. Trace-drop mixture for maze: p(complete)=0.45, three partial-drop levels 1/6 each, full drop 0.05.

**Relevance here.** Strongest controlled 'actions-only vs trace' numbers in this line (30% vs 93% at equal data). Also shows the trace helps as a training signal even when not produced at test time (fast mode 80% vs solution-only 30%), and that randomly dropping parts of the trace beats always-full traces. Suggests a rationale-dropout curriculum rather than all-or-nothing CoT.

### 41. Faith and Fate: Limits of Transformers on Compositionality

Dziri et al.. NeurIPS 2023 (Advances in Neural Information Processing Systems 36, Main Conference Track, Spotlight Poster). https://proceedings.neurips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract.html

**Finding.** GPT-3 fine-tuned on multi-digit multiplication (1.8M pairs, exhaustive up to 4x4 with k1*k2<=9), dynamic programming (142K) and logic puzzles (41K), in question-answer and question-scratchpad-answer formats: near-perfect in-distribution in both formats, but 'fails entirely in generalizing to OOD cases—in particular, wider or deeper computation graphs'; scratchpad fine-tuning does not change the OOD outcome. Few-shot GPT-4: 59% on 3x3-digit, 92% with scratchpad, 4% on 4x4; the scratchpad gain 'diminishes to zero as complexity increases'. 82.3% of correct 4x2-digit final answers contained at least one wrong intermediate step. Relative-information-gain analysis shows surface shortcuts (e.g., last-digit features, RIG 1.0) that explain the partial successes.

**Relevance here.** Scratchpad/CoT supervision buys in-distribution accuracy but not compositional generalization beyond the training graph size. The 82.3% figure is a warning for 'rationales written for known-correct actions': a correct action does not certify its rationale, so synthetic rationales checked only against action correctness can be systematically wrong while looking fine.

### 42. Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems

Ye et al.. ICLR 2025 (poster). https://proceedings.iclr.cc/paper_files/paper/2025/hash/c239bac713017b0b2257b7622bf8aab3-Abstract-Conference.html

**Finding.** Synthetic iGSM math. Pretraining on 'retry' data (erroneous step, [BACK], correction) vs the same token budget of error-free data: iGSM-med op=23 accuracy 78% -> 94-95% at retry_rate 0.5; no label masking of error tokens needed; cheaply faked mistakes (retry_weak) still reach ~91%. Inference-time fixes on error-free-trained models do little: beam search 16/32 no gain, retry-upon-regret +2% (78 -> 80). Deferring retry data to fine-tuning: LoRA 'falls short of pretraining directly with the retry data' and at small rank 'even underperforms compared to pretraining with error-free data'; full fine-tuning learns it only with very many samples (2T tokens, no better than pretraining with them).

**Relevance here.** The most direct evidence on the LoRA question: error-correcting reasoning traces help when the skill is installed at pretraining scale, but LoRA SFT on such traces does not install it and can hurt. Expect LoRA on synthetic CoT to teach format and surface conventions, not a new reasoning procedure.

### 43. Teaching Arithmetic to Small Transformers

Lee et al.. ICLR 2024 (poster). https://proceedings.iclr.cc/paper_files/paper/2024/hash/6bf82fdcbd92b6a7793b3894422d2437-Abstract-Conference.html

**Finding.** NanoGPT trained from scratch on 3-digit addition in four data formats. Plain answer-only format plateaus 'slightly over 85%' even with 10,000 samples (87.18%); reversed-digit output reaches 99.97% after a phase transition between 1,000 and 4,000 samples; simplified scratchpad reaches 100% with 2,000 samples; detailed scratchpad (full chain of intermediate steps) reaches 100% with 1,000 samples, i.e. >=10x fewer than plain. All formats fail to generalize beyond the trained digit lengths.

**Relevance here.** Intermediate-step supervision mainly buys sample efficiency and convergence speed in-distribution, and a cheap formatting change (output order) can matter as much as CoT; no format fixes length generalization. In LoRA fine-tuning with a limited trace budget, sample efficiency is the plausible benefit of CoT, while generalization gains are not guaranteed.

### 44. The Pitfalls of Next-Token Prediction

Bachmann and Nagarajan. ICML 2024 (Proceedings of the 41st International Conference on Machine Learning, PMLR 235, pp. 2296-2318). https://proceedings.mlr.press/v235/bachmann24a.html

**Finding.** Path-star graph: d paths of length l from a start node; output the path to a target. Teacher-forced next-token training fits 200k examples to 100% train accuracy but test accuracy equals random guessing of the first edge, 1/d: G2,5 49.8/48.9/48.5 (GPT-Mini / GPT2-large / Mamba), G5,5 19.1/19.4/20.2, G10,5 8.1/10.3/9.3, G20,5 0.0/3.5/0.0. Mechanism ('Clever Hans cheat'): every token after the first is trivially predictable from the teacher-forced prefix (Acc_cheat ~100%), so the one hard, lookahead-requiring token gets no useful supervision (Acc_1st ~1/d). Teacherless training (predict all tokens from dummy inputs): GPT2-large 99.9 (G2,5), 98.8 (G2,10), 99.0 (G5,5), 97.8 (G10,5), but 0.0 on G2,20 and G20,5; GPT-Mini and Mamba mostly cannot fit it. Reversed-order targets: ~99-100% almost everywhere.

**Relevance here.** Theory for when a rationale is useless: if the rationale is derivable from the action (or written after it), it is a Clever Hans token that is predictable from the prefix and supplies no planning gradient. A rationale that precedes and determines the action moves the hard computation into learnable tokens (analogous to the reversed-order gain). One-step-lookahead argmax-Φ moves are less exposed than long plans, so the benefit of CoT scales with how much lookahead the move actually requires.

### 45. Thought Cloning: Learning to Think while Acting by Imitating Human Thinking

Hu and Clune. NeurIPS 2023 (Advances in Neural Information Processing Systems 36, Main Conference Track), Spotlight Poster. https://proceedings.neurips.cc/paper_files/paper/2023/hash/8b4ba64e549c410185c4d3eac3a81726-Abstract-Conference.html

**Finding.** BabyAI BossLevel. Thought Cloning (TC) imitates synthetic natural-language thoughts (the oracle planner's subgoals) plus actions; BC imitates actions only. Final success 96.2±0.8% (TC) vs 91.2±0.9% (BC); TC learns much faster ('BC requires orders of magnitude more time to achieve a performance similar to TC's early-stage results'), significant at 25/50/75/100% of training (p=0.012, 0.008, 0.021, 0.008). Zero-shot out-of-distribution (behavioral and cognitive difficulty axes): TC 'substantially outperforms' BC, significant at every difficulty level (Mann-Whitney p<0.05).

**Relevance here.** In a multi-step agentic task, thoughts that are the planner's real subgoals give a modest in-distribution gain (+5 points), a large sample-efficiency gain, and a growing OOD gain. The thoughts are derived from the same oracle that produced the actions, i.e. process-faithful synthetic CoT, not model-written rationales; that faithfulness is the condition under which the gain appeared.

### 46. Transformers Learn Shortcuts to Automata

Liu et al.. ICLR 2023 (notable-top-5% / oral). https://openreview.net/forum?id=De4FYqjFueZ

**Finding.** Theory: any semiautomaton run for T steps is simulable by an O(log T)-depth Transformer (Thm 1), and by a depth independent of T (O(|Q|^2 log|Q|)) when the associated transformation group is solvable (Thm 2); constant depth is impossible for non-solvable automata unless TC0=NC1 (Thm 4). Empirics: GPT-2-style Transformers with 1-16 layers trained with standard SGD on freshly sampled length-100 sequences reach >99% in-distribution accuracy on all 19 semiautomata tested (including A5, S5), reported as the best of 20 replicates; training is unstable. With indirect (latent-state) supervision, in-distribution accuracy on Dyck4,8 / Grid9 / S5 / C4 / D8 is 100 / 100 / 99.6 / 99.9 / 100 (arXiv v2 Fig. 5a; camera-ready main-text Fig. 4a shows S5 at 98.0, i.e. 98-100%), LSTM 100% on all. Brittleness: on parity (C2), accuracy degrades as test Pr[sigma=1] moves away from the training value 0.5 (sequences of length 40; LSTM stays perfect); in length generalization, models trained on length 40 and tested on lengths 8-128 show Transformer accuracy that "drops sharply" at unseen lengths while the LSTM is perfect; with sparse state labels (p_reveal, length-100 sequences) the Transformer degrades significantly on S5 (but not C2) while the LSTM is robust. Mitigation: scratchpad (chain-of-thought) training combined with a recency bias guides the Transformer to a recurrent depth-T solution that is robust to the Pr[sigma=1] shift and generalizes to longer lengths; scratchpad alone gives no length-generalization improvement, and the fix forgoes the depth advantage of the shortcut.

**Relevance here.** Mechanistic analogy for the question: supervising only outputs lets the model learn a statistical shortcut that matches in-distribution and breaks under shift; supervising the intermediate state sequence forces the sequential algorithm. In the design loop the simulator feedback already externalizes state each turn, so the shortcut risk sits inside the per-turn choice (comparing candidate parameter changes), which is where intermediate supervision would have to land.

### 47. Transformers Struggle to Learn to Search

Saparov et al.. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/28d43a64147bf94921041599efdf791d-Abstract-Conference.html

**Finding.** Graph-connectivity search trained from scratch with effectively unlimited data. With a balanced training distribution (883M examples, <=41 vertices, lookahead <=20) the model 'performs near perfectly'; a model trained with lookahead <=12 generalizes only to 13-14. As max graph size grows (8 layers, hidden 16, 236M examples, 14 seeds) the probability of fitting the training set (accuracy >=0.995) 'becomes vanishingly small', and model size shows 'no discernible pattern' with the training needed. Chain-of-thought variants trained on DFS traces (3 layers) and selection-inference traces (4 layers) also 'struggle as the graph size increases', and increasing model scale does not help (max graph size 35 and 45 respectively).

**Relevance here.** Counterweight to SoS/Searchformer: training on in-context exploration traces does not remove the difficulty of learning search as problem size grows, and scale does not rescue it; CoT traces help only within the lookahead range the data demonstrates. For a one-step-lookahead design task, the benefit of search-style CoT is bounded by how much lookahead the gold data actually exhibits.

### 48. Chain of Thought Empowers Transformers to Solve Inherently Serial Problems

Zhiyuan Li et al.. ICLR 2024 (poster). https://arxiv.org/abs/2402.12875

**Finding.** Theorem 3.1: constant-depth, constant-precision Transformers without CoT, and even with O(log n) CoT steps, lie in AC0, so they cannot compute parity. Theorem 3.3: with T(n) CoT steps, constant-depth constant-precision Transformers with O(log n) embedding size solve any problem computable by a size-T(n) Boolean circuit (CoT[poly(n), log n] = P/poly). Experiments on composition of S5 permutations, iterated squaring, and circuit value: without CoT, S5 accuracy sits at about 20% (chance over 5 outputs) for most lengths and depths, while CoT solves it; the CoT gain is largest at small depth and long sequence length, and the 'cot' condition beats both 'base' and 'hint' on all three serial tasks.

**Relevance here.** Supplies the mechanism (each CoT token resets effective depth) and the empirical signature: CoT matters where the label is an inherently serial function and the model is shallow. For a 27B policy choosing one parameter change per turn, the question is whether ranking candidate moves by Phi is serial in this sense; if the state text already carries the quantities needed to pick the move in one pass, this theory predicts no gain from CoT supervision.

### 49. Sub-Task Decomposition Enables Learning in Sequence to Sequence Tasks

Noam Wies et al.. ICLR 2023 (poster). https://arxiv.org/abs/2204.02892

**Finding.** Theorem 1 (informal, = Corollary 2, Sec. 6): there exists a binary classification problem of size d such that (a) with sub-task decomposition supervision a simple seq2seq model reaches arbitrarily low zero-one loss eps with a number of gradient updates polynomial in d and 1/eps, and (b) without sub-task supervision, for ANY polynomial-time learner and constant eps, the zero-one loss exceeds 1/2 - eps. Part (b) rests on standard cryptographic / complexity hardness assumptions (Valiant 1984; Goldreich et al. 1986; Daniely & Shalev-Shwartz 2016) and concerns a hard function in P, NOT bit-subset parity. The bit-subset parity result is the separate Corollary 1 (Sec. 5.1, Appendix F Theorem 4): for parity of an unknown d/2-subset of d bits, any iterative gradient-based algorithm with polynomially bounded gradients needs exponentially many full-batch GD iterations to beat 1/2 - O(e^{-d}) (w.h.p. over the target parity; builds on Shalev-Shwartz et al. 2017 / Shamir 2018), and footnote 3 states this negative result holds only for full-batch GD, not SGD. Positive side for parity: an Elman RNN with ReLU, trained by SGD on only the hidden matrix W (input/output/initial-state weights frozen; Algorithm 1), on the input concatenated with the sub-task labels (pairwise parities arranged as a binary tree, serialized by inverse-BFS, T = 3d/2 - 2), reaches loss eps with poly(d, 1/eps) updates and polynomial width (Theorem 2 + Lemma 2; extended to finite-precision SGD and GD in Appendices C-D). General positive result (Sec. 6, Lemma 3): any function in P has a poly-size circuit with O(1)-fan-in gates; treating each gate as a sub-task supervised with teacher forcing makes the task learnable with polynomial size, samples and updates. Lemma 1: at test time, without ground-truth intermediate results, a union bound lets the error grow by at most a factor T - d. Experiments (Sec. 5.2, Appendix G, Table 1): BERT-base-sized (12-layer, 768-hidden) GPT-2-style Transformer decoder, 100k steps, inputs 8-256 bits, 3 seeds. Without intermediate supervision: 8 bits learned in 185 steps, 16 bits in 212K (run to 300K), 18 bits in 1.53M (run to 2M), and 32 bits never learned ('no learning after over 2M steps', ~50% test accuracy); the paper says the model 'can not learn beyond random guessing tasks with more than 18 bits'. With intermediate supervision: 16 bits in 333 steps, 32 bits in 1.48K, 64 in 11K, 128 in 58K (99.96% acc), 256 in 720K. The step gap is already ~600x at 16 bits and becomes total between 18 and 32 bits. 'Learned' = validation accuracy > 60%; a grokking-style jump to >95% follows immediately.

**Relevance here.** The cleanest theory of why training on the computation that produces the label beats training on the label: the intermediate tokens must be the actual sub-task values of a decomposition of the target function, and then each step is a simple, independently supervised problem. A rationale written after the fact for a known-correct move does not satisfy this condition unless it literally evaluates and compares the candidates; the theorem gives it no guarantee.

### 50. The Expressive Power of Transformers with Chain of Thought

William Merrill et al.. ICLR 2024 (poster). https://proceedings.iclr.cc/paper_files/paper/2024/hash/1f59721c106ea80f613299039112f651-Abstract-Conference.html

**Finding.** Purely theoretical. O(log n) intermediate decoding steps add essentially nothing beyond the TC0-type upper bound of a standard log-precision Transformer; linearly many steps (with projected pre-norm) let the decoder recognize all regular languages, with an upper bound of context-sensitive languages; polynomially many steps with generalized pre-norm recognize exactly the polynomial-time problems (P), the first exact characterization of a Transformer class in terms of a standard complexity class. No experiments.

**Relevance here.** The power bought by intermediate tokens scales with how many are generated and only matters for problems outside the one-pass class. A per-turn decision whose target is a shallow argmax is inside that class, and the environment loop (simulator feedback over 12-20 turns) already supplies the serial steps externally; so any benefit of synthetic CoT here must be statistical or optimization-related, not expressivity.

### 51. Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective

Guhao Feng et al.. NeurIPS 2023 (Advances in Neural Information Processing Systems 36, Main Conference Track; oral). https://arxiv.org/abs/2305.15408

**Finding.** Theorems 3.1-3.2: log-precision bounded-depth Transformers cannot directly output answers to arithmetic and linear-equation tasks unless model size grows super-polynomially in input length (assuming TC0 != NC1). Theorems 3.3-3.4: constant-size autoregressive Transformers solve both by generating step-by-step derivations. Theorem 4.7: with CoT, Transformers solve a general class of dynamic-programming problems (LIS, edit distance); Theorem 4.8: CFG membership cannot be solved directly. Experiments: direct-answer Transformers of depth 3-5 are mostly below 60% accuracy and degrade as input length grows (4-6 operators, 3-5 variables, LIS length 50-100, ED length 12-20), while a 3-layer CoT-trained Transformer is near-perfect on all four tasks at every difficulty; a 3-layer CoT model trained on 1-15 operators extrapolates to 16-18; it keeps >95% accuracy when 30% of training traces have a missing step and a corrupted token.

**Relevance here.** CoT training pays off when the label-producing computation has serial depth growing with the input (arithmetic chains, DP recurrences); the traces that work are the computation itself, and they tolerate noise. A greedy one-step argmax of a potential over a small candidate set is a shallow function of the state, so the expressivity argument does not by itself predict a gain from rationales in the DesignBench setting unless evaluating Phi per candidate is itself serial work the model must externalize.

### 52. What Algorithms can Transformers Learn? A Study in Length Generalization

Hattie Zhou et al.. ICLR 2024 (poster). https://proceedings.iclr.cc/paper_files/paper/2024/hash/45ed1a72597594c097152ef9cc187762-Abstract-Conference.html

**Finding.** RASP-Generalization Conjecture: a decoder-only Transformer length-generalizes when the next-token function is realizable by one causal Transformer for all lengths, is expressible as a short RASP-L program, and the training data is diverse enough to rule out shorter programs. Scratchpads cut both ways: an index-hint plus running-parity scratchpad gives perfect length-50 generalization for parity when trained to length 40, and reversed-order addition with index hints generalizes to 50-digit sums; but for the Mode task a scratchpad that lists token frequencies in ascending order makes the model significantly worse than no scratchpad at length 50 on both exact-match and final-answer accuracy, and converges more slowly (Figures 4c, 7c).

**Relevance here.** Direct evidence that intermediate-step supervision hurts when the chosen trace format makes the per-token function more complex than the direct state-to-answer map. A rationale schema for a truss move is a design choice: if it adds tokens whose prediction is harder than predicting the move itself, it can reduce generalization rather than improve it.

### 53. Why think step by step? Reasoning emerges from the locality of experience

Ben Prystawski et al.. NeurIPS 2023 (oral). https://arxiv.org/abs/2304.03843 (camera-ready v3, footer "37th Conference on Neural Information Processing Systems (NeurIPS 2023)"); oral listing: https://neurips.cc/virtual/2023/oral/73821

**Finding.** Theorem 3.1: an autoregressive estimator trained only on locally observed adjacent pairs of a chain Bayes net has strictly lower bias when it reasons through intermediate variables (scaffolded) than when predicting directly, with bias scaled by (1 - lambda)^2. Experiments: a 32.6M GPT-2-style model on 100-variable Bayes nets, 1M samples per net, 25 held-out high-mutual-information pairs. With locally structured training data, free-generation reasoning gives significantly lower MSE on held-out pairs than direct prediction; with fully observed data the two are nearly identical; with wrong-locality data neither recovers the true conditionals. Irrelevant intermediate variables (negative scaffolding) do not help. Direct prediction on fully observed data needs more than 3x the training to match reasoning on local data (about 650M vs 200M tokens). With only 1-2 Monte-Carlo samples, reasoning has higher error than direct prediction.

**Relevance here.** Gives the conditions under which intermediate steps help even when expressivity does not require them: the training data must cover the mediating variables locally but not the direct input-output link, and the steps must pass through the true mediators. For design traces, a rationale helps only if it routes through quantities (member stresses, FOS, mass) that mediate the move choice and that the model has seen locally; rationale content that is not a mediator contributes nothing.

### 54. Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

Hsieh et al.. Findings of the Association for Computational Linguistics: ACL 2023 (Toronto), pages 8003-8017, DOI 10.18653/v1/2023.findings-acl.507. https://aclanthology.org/2023.findings-acl.507/

**Finding.** T5 students (220M-11B) trained with PaLM-540B rationales. Rationales are used as an auxiliary output head (multi-task loss L_label + lambda*L_rationale), not generated before the answer at test time. 220M T5: standard fine-tuning vs 'single-task' (rationale and label concatenated in one output) vs multi-task: e-SNLI 88.38 / 88.88 / 89.51; ANLI 43.58 / 43.50 / 49.58; CQA 62.19 / 61.37 / 63.29; SVAMP 62.63 / 63.00 / 65.50. Multi-task beat standard fine-tuning on ANLI by 8 points on average and standard distillation by 13; it reached the finetuning baseline with 12.5% of e-SNLI data and with 20-75% less data on the other tasks. Rationales from a 20B teacher helped less than from 540B (ANLI 48.15 vs 49.58) but still beat no-rationale training.

**Relevance here.** The configuration that actually resembles 'think then act' SFT (rationale concatenated with the answer in one output) gave zero gain over answer-only training at 220M and was slightly worse on two of four tasks; the gain came from using the rationale as auxiliary supervision that is never generated at inference. For a small-capacity policy, rationales as a side loss rather than as a prefix is the version with evidence.

### 55. LIMO: Less is More for Reasoning

Ye et al.. COLM 2025. https://arxiv.org/abs/2502.03387

**Finding.** Qwen2.5-32B-Instruct full-parameter SFT'd on 800 curated problems with long reasoning chains (v1 said 817; the COLM version says 800). Table 1 pass@1 — AIME24: LIMO 63.3, OpenThoughts-114k 50.2, NuminaMath-100k 6.5, base 16.5; MATH500: 95.6 / 80.6 / 59.2 / 79.4. The 78.1 / 58.3 / 32.3 (base 49.9) figures are the AVERAGE OVER ALL 10 BENCHMARKS (3 in-domain + 7 OOD), not an OOD-only average; OOD-only averages computed from the table's seven OOD rows are 75.1 / 53.0 / 30.9 (base 48.4). Reasoning-chain quality ablation (Fig. 3; 500 questions, each with multiple correct solutions, graded L1-L5 by a rule-based score: length 30%, validation words 20%, tentative words 25%, connectives 25%): accuracy rises monotonically — read from the plot, AIME24 roughly 4 -> 21 -> 44 -> 49 -> 53 and MATH500 roughly 47 -> 83 -> 88 -> 92 -> 93 from L1 to L5; the text says "the substantial gap between L5 and L1 solutions" shows chain quality "significantly influences" performance. Question difficulty ablation (Fig. 4): AIME24 Simple-500 34.6, Complex-500 40.0, Advanced-500 51.5 (MATH500 88.0 / 89.4 / 91.2); the paper's "16% improvement ... reaching 51.5%" is relative to Simple-500. Base model dominates (Fig. 5): same LIMO data gives Qwen1.5-32B-Chat 9.2 AIME24 / 65.2 MATH500 vs Qwen2.5-32B-Instruct 63.3 / 95.6. Within-pipeline data-size ablation (Fig. 7): AIME24 gains over base of +41.0 at 400 samples, +5.8 at 800, then +0.9/+4.1/+1.3 at 1.2k/1.6k/2k, i.e. diminishing returns beyond ~800. The "quality over quantity" headline comparison against OpenThoughts/NuminaMath is across datasets that differ in question source and chain source, not only size — this is a reader's caveat, not a statement the paper makes.

**Relevance here.** Supports rationale SFT only as a template that elicits competence the base model already has from pre-training (the 54-point swing between Qwen1.5 and Qwen2.5 on identical data). For a truss-design action space absent from pre-training, this mechanism is weaker, and the chain-quality ablation says shallow, formulaic rationales (L1-L2 style) are the ones that give the least.

### 56. V-STaR: Training Verifiers for Self-Taught Reasoners

Hosseini et al.. COLM 2024 (1st Conference on Language Modeling, Philadelphia, 7-9 Oct 2024). https://arxiv.org/abs/2402.06457

**Finding.** LLaMA-2 7B/13B and CodeLLaMA, LoRA fine-tuned for 3 iterations with k=16 samples per problem. The generator is trained STaR-style on correct self-generated solutions (final-answer match for math, test cases for code); unlike STaR, the incorrect solutions are not discarded but used with the correct ones to train a DPO verifier that ranks 64 candidates at test time. V-STaR improves test accuracy by 6-17 points over STaR and over a verifier trained without iteration on math, and by 4-12 points on code; 7B V-STaR exceeds LLaMA-2-70B 8-shot on GSM8K and nearly matches CodeLLaMA-34B (48% pass@1) on HumanEval. A fourth iteration added 0.3% on MBPP. The verifier gains exceeded the generator gains.

**Relevance here.** In a setting with simulator feedback, the rollouts that do not match the gold move are a free signal for a verifier or value model, and the paper finds that is where most of the improvement is, not in further policy SFT on correct rationales. It argues for pairing any rationale SFT with a learned ranker over candidate moves rather than relying on the policy alone.

### 57. Investigating the Benefits of Free-Form Rationales

Sun et al.. Findings of the Association for Computational Linguistics: EMNLP 2022 (Abu Dhabi, Dec 7-11 2022), pp. 5867-5882, ACL Anthology 2022.findings-emnlp.432, DOI 10.18653/v1/2022.findings-emnlp.432. https://aclanthology.org/2022.findings-emnlp.432/

**Finding.** With rationales as model inputs (T5-base IR->O), QuaRTz knowledge statements (designed not to leak the answer) raise accuracy 70.88% -> 81.51% when 100% of training instances carry rationales (Table 8; at 5% it is 67.86%, below the 70.88% baseline). On the CoS-E no-leak/background-knowledge test subset (about 4% of the test set), training with only 5% of rationales lifts accuracy 40.68% -> 59.89% (+47.2% relative; Table 6 c5). Leakage is the dominant confound, not the weakest effect: label-leaking CoS-E rationales give the LARGEST gains (leak-no-bg 46.80 -> 89.03, leak-bg 46.34 -> 87.40 at 5%; 96.34 / 97.56 at 100%), while rationales that neither leak nor add background knowledge give the SMALLEST (45.97 -> 54.72 at 5%; Sec. 4, Table 6 c7-c8). ECQA's fixed ordering (correct option explained first) yields a spurious 93.94% at 5% unshuffled training; testing that same model on sentence-shuffled rationales drops it to 76.66% (baseline 54.95%), and training on shuffled rationales gives 86.35% on shuffled test (the abstract's 57.14% relative gain) - the ordering shortcut is removed but a large genuine gain remains (Table 7; Table 6 r7). T5-generated rationales are judged to provide background knowledge less often than crowdsourced ones (ECQA 22.50% vs 43.44%; CoS-E 20.00% vs 28.33%; Table 9), give a weaker training signal (models trained on them never match crowdsourced-trained models; Table 10 vs Table 6, Sec. 5, App. D), and have negative simulatability (CoS-E -13.1 = 43.9-57.0; ECQA -12.36 = 44.64-57.00; App. A.2).

**Relevance here.** Separates two effects that get conflated in 'synthetic CoT helps': rationales help when they add information the label does not already contain, and model-generated rationales add less of it than human ones. Also a concrete warning that an ordering artefact in rationales can masquerade as reasoning gains, which is easy to reproduce with templated rationales for argmax moves.

### 58. Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning

Paul et al.. Findings of the Association for Computational Linguistics: EMNLP 2024 (Miami, November 2024), pp. 15012-15032, DOI 10.18653/v1/2024.findings-emnlp.882. https://aclanthology.org/2024.findings-emnlp.882/

**Finding.** Causal-mediation analysis over 12 LLMs: GPT-4 changes its answer only ~30% of the time when its reasoning chain is counterfactually perturbed. A direct train-on-CoT vs train-on-answer comparison with T5-large: StrategyQA SFT 57.6 → SFT+CoT 63.6; QuaRel 74.6 → 77.7; OBQA 65.0 → 65.5; QASC 58.6 → 59.4. Their FRODO method (train the rationale generator with implicit-causality preference pairs and the answer module to be counterfactually robust) reaches 68.4 / 83.4 / 70.2 / 64.2, with +4.5 robustness (CIE) and +2.6 OOD over SFT.

**Relevance here.** One of the few controlled SFT-vs-SFT+CoT numbers: CoT supervision helps a lot on the one dataset that needs multi-step reasoning (+6 StrategyQA) and ~0.5 points elsewhere. Gains beyond that required training the model to depend on its reasoning, not just to emit it, which is the regime where post-hoc rationales for argmax-Φ moves would need to encode the Φ comparison to be useful.

### 59. When Can Models Learn From Explanations? A Formal Framework for Understanding the Roles of Explanation Data

Hase & Bansal. First Workshop on Learning with Natural Language Supervision (LNLS 2022), co-located with ACL 2022, Dublin, May 2022, pp. 29-39, DOI 10.18653/v1/2022.lnls-1.4. https://aclanthology.org/2022.lnls-1.4/

**Finding.** Two versions exist and the claimed finding merges them. (A) The peer-reviewed LNLS 2022 workshop paper (11 pp.) contains only: the graphical-model framework (explanations as inputs / targets / priors), the synthetic integer-counting task, and RQ1-RQ2 on synthetic data. There, explanation retrieval (RoBERTa-base + Sentence-RoBERTa) reaches accuracy above 98%, about 37 points over a no-explanation baseline (~62%); a baseline with 10x training data (50,000 points) reaches 87.11% and still does not close the gap; the no-explanation model falls to 62.2% when there are 500 unique explanations. The workshop version does NOT list six preconditions and does NOT report any e-SNLI / TACRED / SemEval experiments; its Appendix A explicitly defers "does retrieval of explanations improve model performance on existing natural language datasets?" to the arXiv technical report. It does state (Sec. 2.2) that using free-text explanations as targets or structured variables "do not appear to be promising approaches," and that the only known successes are explanations used as inputs (ExpBERT global set, retrieval, and 280B-parameter in-context learning per Lampinen et al. 2022). (B) The arXiv technical report (2102.02201v2, Feb 2021, 25 pp.) is the source of the rest of the claim: its abstract states "accuracies upwards of 95%" vs "below 65%" for baselines; Sec. 6.8 / Table 3 reports RoBERTa vs TEXTCAT retrieval on e-SNLI, SemEval and TACRED at several training sizes with "no statistically significant improvements" (effect sizes between -0.35 and +0.63 points, all 95% CIs covering zero; e.g. e-SNLI full 91.06 vs 91.41, SemEval full 76.94 vs 76.91, TACRED full 88.29 vs 88.59); and Sec. 7 lists six preconditions: (1) the model can better infer relevant latent information from x plus explanation than from x alone; (2) explanations are linked to query points by an easily computable index feature; (3) explanations are sufficiently relevant across data points; (4) the classifier has a known or identifiable interpretation of the explanations; (5) the classifier reaches sufficient quality before the retriever is trained; (6) the initial retriever has sufficient quality before the classifier is trained. The authors say only that "the reason for this lies somewhere in the six preconditions" and call for a future diagnostic procedure; they do NOT attribute the null result to explanations being less computable or compressive than the label mapping. The nearest statement is a separate observation that, with a retrieval model, whatever information explanations provide "can be inferred from the input alone," so a model may learn the input-to-latent map directly, and that as LMs store more knowledge parametrically the need for retrieved explanations shrinks.

**Relevance here.** The formal account of when explanations are useful as supervision: they must encode something that is cheaper to learn from the explanation than from (x, y) pairs alone. A rationale that restates 'parameter k was changed because it raised Φ most' is a function of the action, not extra information about the state→action mapping, so the framework predicts a null effect unless the rationale carries the Φ structure in a form the model can reuse.

### 60. Why Exposure Bias Matters: An Imitation Learning Perspective of Error Accumulation in Language Generation

Arora et al.. Findings of the Association for Computational Linguistics: ACL 2022 (Dublin, May 2022), pages 700-710. https://aclanthology.org/2022.findings-acl.58

**Finding.** Shows teacher-forced MLE is behaviour cloning under a particular per-step loss and imports the Ross bounds T eps <= R <= T^2 eps. Defines AccErr(l) = R_<=l / eps_<=l and %ExAccErr (excess over the linear l*eps floor). With GPT-2 (117M) fine-tuned on WikiText-103 as the oracle, 50-token prompts, up to 512 tokens, ~20k prompts: AccErr grows near-quadratically with length; %ExAccErr greedy 60.96%, beam-5 69.72%, sampling T=1 39.37%, T=1.2 24.75%, top-k 100 35.37%, top-p 0.94 48.71%. Log-perplexity correlates with per-step error at rho = 0.9997 but with length-normalised regret at only rho = 0.4003.

**Relevance here.** Direct measurement that an MLE-trained autoregressive LM compounds errors across its own generated prefix, worst under greedy/beam decoding, and that held-out perplexity cannot see it. Within one design turn, a synthetic rationale is a long free-running prefix before the action, so 'the rationale lowered validation loss' is not evidence that the policy improved; only rollout metrics are.

### 61. Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping

Lehnert et al.. COLM 2024 (Conference on Language Modeling); earlier version at ICLR 2024 Workshop on LLM Agents. https://arxiv.org/abs/2402.14083

**Finding.** Encoder-decoder Transformers trained on A* execution traces (search-augmented) vs on the optimal plan only (solution-only). Sokoban, 200 unseen puzzles: solution-only 45M 90.3% solved / 86.8% optimal; search-augmented 45M 92.5 / 90.8; Searchformer (bootstrap fine-tuned, step 3) 95.5 / 93.7; solution-only 175M 95.7 / 90.0; solution-only 757M 96.5 / 92.2. Maze: search-augmented models match solution-only with 5-10x fewer parameters and 10x less data, and solution-only performance 'degrades significantly' at <=100k training sequences. Trace sequences are thousands of tokens, ~10x longer than solution-only outputs; bootstrapping yields traces with 26.8% fewer steps than A*.

**Relevance here.** The trace advantage is largest in the low-data / small-model regime and shrinks when the solution-only model is scaled (757M solution-only matches 45M search-augmented on solve rate, still behind on optimality). For a 27B LoRA policy with a modest trace budget the low-data regime is the relevant one, but the gain is mostly sample efficiency, at a ~10x token cost per example.

### 62. Stream of Search (SoS): Learning to Search in Language

Gandhi et al.. COLM 2024 (listed as Spotlight on the official COLM 2024 accepted-papers page; the authors' own sites call it an Oral). https://arxiv.org/abs/2404.03683

**Finding.** 250M GPT-Neo trained from scratch on Countdown. Two 500k-trajectory datasets: optimal paths only (OP) vs full search streams with dead ends and backtracking (only 57.1% of SoS training traces end correct). Held-out accuracy: OP model 25.73% vs SoS model 51.27% (49.00% on unseen targets). Policy improvement on top of SoS: +STaR 55.59%, +APA 56.52%; the improved models solve ~36% of problems the data-generating heuristic solvers could not, and ~4% of problems no symbolic strategy solves.

**Relevance here.** Cleanest evidence that training on the process (including evaluated-and-rejected alternatives) doubles accuracy over training on the optimal action sequence alone. Caveat for the design setting: the traces are genuine search dynamics, not post-hoc rationales, and the model is pretrained from scratch; the analogue is exposing the candidates the argmax-Φ optimiser compared, not a narrative written after the fact.

### 63. A Theory of Learning with Autoregressive Chain of Thought

Nirmit Joshi et al.. COLT 2025 — Proceedings of the 38th Annual Conference on Learning Theory, PMLR 291:3161-3212 (eds. Haghtalab & Moitra), 30 Jun–4 Jul 2025. https://proceedings.mlr.press/v291/joshi25a.html

**Finding.** Models the prompt-to-answer map as T iterations of one time-invariant step function from a base class F. With the chain observed, sample complexity is about VCdim(F) * log T (Theorem 3.4), essentially independent of chain length; with only prompt-answer pairs (end-to-end), it is about T * VCdim(F) (Theorem 3.2) and the linear dependence on T is unavoidable in general (Theorem 3.3). For linear-threshold step functions, CoT learning reduces to LP feasibility and runs in polynomial time (Corollary 4.3) while end-to-end learning is not polynomial-time learnable under a standard threshold-circuit hardness assumption (Theorem 4.4); more generally CoT learning is tractable whenever F has a polynomial-time consistency oracle (Corollary 3.7). All guarantees assume realizability: the supervision chain is generated by a step function in F.

**Relevance here.** Formalizes exactly when intermediate-step supervision beats answer-only supervision: a T-fold statistical saving and a computational one, but only if the supervised chain is produced by a single consistent step function that actually generates the answer. Rationales written post hoc in varying styles, or traces from a process other than the one that picked the move, break realizability, and the cited time-dependent analysis then gives back the linear-in-T cost.

### 64. Show Your Work: Scratchpads for Intermediate Computation with Language Models

Maxwell Nye et al.. ICLR 2022 Workshop on Deep Learning for Code (DL4C), poster; arXiv:2112.00114 v1, 30 Nov 2021. https://arxiv.org/abs/2112.00114

**Finding.** Fine-tuned 8B model on polynomial evaluation: 31.8% direct vs 50.7% with scratchpad. 137B fine-tuned on synthetic Python execution: 20% direct vs 41.5% scratchpad tracing. On real MBPP programs with only the 374 training tasks (1,122 examples), direct execution beats scratchpad, 10.3% vs 5.1% per-task (Section 5.2.1, 'Performance is poor in the very-low-data regime'); after augmenting with 17k model-sampled programs, scratchpad rises to 17.3% while direct execution falls to 5.1%; adding CodeNet and single-line traces reaches 26.6%. Addition (2M-1B params, trained on 1-8 digits): scratchpad models solve 9-10 digit out-of-distribution sums above a critical size while direct models fail at every size.

**Relevance here.** The scratchpad is the execution trace that produces the label, and its benefit is large but conditional on data volume: with about 1k traces the plain label-only model wins, and the ordering flips only once there are tens of thousands of traces. The user's 5k multi-turn traces sit near this boundary, so the data regime, not just the trace format, decides whether synthetic CoT helps.

### 65. Orca: Progressive Learning from Complex Explanation Traces of GPT-4

Mukherjee et al.. arXiv preprint 2306.02707, June 2023 (cs.CL); title page marked "Work in progress"; Microsoft Research lists venue as "arXiv: Computation and Language"; no OpenReview, proceedings, or journal record found. https://arxiv.org/abs/2306.02707

**Finding.** LLaMA-13B fine-tuned on ~5M ChatGPT and ~1M GPT-4 responses to FLAN-v2 queries, where system prompts asked the teacher for step-by-step explanations ('explanation tuning') instead of terse answers. Zero-shot BigBench-Hard: Vicuna-13B 23.3, Orca-13B 49.7, ChatGPT 48.9, GPT-4 67.4. AGIEval: Vicuna 29.3, Orca 41.7, ChatGPT 47.2. The paper reports that Vicuna keeps ~85% of ChatGPT quality on open-ended generation but only ~62% on reasoning, and attributes Orca's gain to imitating the reasoning process rather than the style. No controlled ablation with the same queries and plain (non-explanation) teacher responses is reported; training data size also differs from Vicuna by two orders of magnitude.

**Relevance here.** The most-cited evidence that explanation traces beat plain answers, but it is confounded (different data volume and query set), so it supports the direction without isolating the rationale effect. For the engineering-design question it should be read as suggestive, not as a controlled rationale-vs-action comparison.

### 66. Measuring Faithfulness in Chain-of-Thought Reasoning

Lanham et al.. arXiv preprint, 2023 (arXiv:2307.13702, submitted 17 Jul 2023); no peer-reviewed venue found. https://arxiv.org/abs/2307.13702

**Finding.** Truncating or corrupting the CoT changes the final answer <10% of the time on ARC-Easy/ARC-Challenge/OpenBookQA but >60% on AQuA; early-answering AOC ranges 0.02 (ARC-Easy) to 0.44 (AQuA). Accuracy gain from CoT (e.g. AQuA 28→43, ARC-Easy 96→96) is only weakly correlated with reliance on it. Replacing CoT with filler tokens gives no gain, so the benefit is not extra test-time compute. The 13B model is the most faithful on 6 of 8 tasks (inverse scaling with capability). Caveat from Bentham, Stringham & Marasović (TMLR 2024, arXiv:2402.14897): after normalising for answer-choice bias the metric correlates with accuracy at R²=0.74, so 'faithfulness' here partly measures task difficulty.

**Relevance here.** The reasoning text matters to the answer only on tasks the model cannot solve without it. If the truss step is easy for the 27B model once it has seen simulator feedback, the rationale will be ignored at inference and training on it buys little; if the step is genuinely hard, the rationale is load-bearing and its content has to be correct. The task-dependence, not a blanket yes/no, is the finding.
