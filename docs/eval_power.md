# Evaluation power: why the T5 arms could not have shown anything

## The number

Arm-vs-arm discordance on the 34-problem eval split, averaged over all 253 T5
arm pairs, is **0.152** — arms disagree on ~15% of problems. Feeding that into a
paired McNemar design at 80% power, alpha = 0.05:

| protocol | effective n | MDE |
|---|---|---|
| **34 problems x 1 sample  (what every T5 arm used)** | 34 | **18.7 pp** |
| 34 x 4 | 136 | 9.3 pp |
| 66 x 1 | 66 | 13.4 pp |
| 66 x 4 | 264 | 6.7 pp |
| 100 x 4 | 400 | 5.5 pp |
| **120 x 4  (proposed)** | 480 | **5.0 pp** |
| 120 x 8 | 960 | 3.5 pp |

The protocol used for all eight arms could only resolve an effect of **>= 18.7
percentage points**. Swapping a potential function does not move end-task
feasibility by 18.7pp. The experiment was incapable of detecting any plausible
effect before a single GPU-hour was spent.

## What the arms actually reported

Phi_v1 (t5a) vs Phi_v2 (t5b), matched checkpoints:

| ckpt | delta (v2 - v1) | discordant | p |
|---|---|---|---|
| 25 | +0.000 | 2 / 2 | 1.000 |
| 50 | +0.000 | 2 / 2 | 1.000 |
| 75 | -0.059 | 3 / 5 | 0.727 |

Phi-only arms (t5ao vs t5bo): +0.029, +0.059, -0.059, +0.029; every p >= 0.5.

Every observed delta is far inside the 18.7pp resolution floor. These are not
evidence of no effect. They are non-measurements.

## Cost of fixing it

The 34x1 eval cost 0.58 SU. 120x4 is 14x the work, so **~8 SU per arm** — about
a fifth of one 36 SU training arm. Buying a 5.0pp MDE costs less than a quarter
of the price of the arm being measured.

## Correction: 120 problems is not available

The table above prices protocols up to 120x4, but this benchmark does not have
120 held-out problems. `data/splits/truss_v1_auto.json` is 66 train / 34 eval of
129 total, and 4 of the 34 are degenerate mechanisms where Phi is constant, so
30 are usable. The reachable protocol is 34xK:

| protocol | effective n | MDE | cost, both arms of an A/B |
|---|---|---|---|
| 34 x 1  (what all T5 arms used) | 34 | 18.7 pp | ~1 SU |
| 34 x 4 | 136 | 9.4 pp | ~5 SU |
| 34 x 8 | 272 | 6.6 pp | ~9 SU |
| **34 x 16** | **544** | **4.7 pp** | **~19 SU** |

**Standing decision for the per-token A/B: 34 x 16, paired by problem and seed.**
19 SU to make an already-committed 72 SU of training interpretable. Anything that
needs more resolution than 4.7pp needs more held-out problems, not more samples.

## Protocol from here

Any arm intended to support or refute a claim is evaluated at **the largest
34xK the question justifies (K=16 for a decisive A/B), paired by problem and
seed against its control**, reported with McNemar
discordant counts and an exact p-value — never as a bare rate difference. Arms
evaluated at 34x1 are exploratory and may not be quoted as evidence.

Related: `tests/test_shaping_invisibility.py` (why the shaping arms varied a
quantity the estimator discards) and `docs/theory_potential_and_feedback.md`.
