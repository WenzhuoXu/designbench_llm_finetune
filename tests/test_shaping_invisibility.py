"""Why potential-based reward shaping cannot move group-normalized GRPO.

Ng, Harada & Russell shaping adds F(s,a,s') = gamma*Phi(s') - Phi(s) to each
transition. Summed over an episode with the same discount, it telescopes:

    sum_t gamma^t [gamma*Phi(s_{t+1}) - Phi(s_t)] = gamma^T Phi(s_T) - Phi(s_0)

Every rollout in a GRPO group starts from the SAME state, so Phi(s_0) is a group
constant and mean-centering deletes it. What survives is gamma^T Phi(s_T) -- a
reweighting of TERMINAL quality. Every intermediate Phi value cancels exactly.

GRPO assigns one scalar advantage per sequence (span{1}), so shaping cannot add
the per-step credit it exists to provide. This is the reason arms T5a/T5b/T5c
returned nulls: they varied a quantity the estimator is structurally blind to.
"""
import numpy as np
import pytest


def shaped_transition_sum(phis, gamma):
    """sum_t gamma^t (gamma*Phi(s_{t+1}) - Phi(s_t)) computed term by term."""
    return sum(gamma**t * (gamma * phis[t + 1] - phis[t]) for t in range(len(phis) - 1))


def group_advantages(returns, scale="group"):
    """GRPO advantage: mean-centre within the group, optionally scale by std."""
    r = np.asarray(returns, dtype=float)
    a = r - r.mean()
    if scale == "group":
        s = r.std()
        if s > 1e-12:
            a = a / s
    return a


class TestTelescoping:
    def test_shaping_sum_equals_terminal_minus_initial(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            T = int(rng.integers(2, 25))
            gamma = float(rng.uniform(0.5, 1.0))
            phis = rng.normal(0, 10, size=T + 1)
            got = shaped_transition_sum(phis, gamma)
            want = gamma**T * phis[-1] - phis[0]
            assert got == pytest.approx(want, rel=1e-9, abs=1e-9)


class TestIntermediatePhiCancels:
    def test_perturbing_every_interior_phi_changes_no_advantage(self):
        """The load-bearing claim: interior Phi is free, advantages don't move."""
        rng = np.random.default_rng(1)
        for _ in range(200):
            K = int(rng.integers(4, 17))          # group size
            T = int(rng.integers(3, 20))          # episode length
            gamma = float(rng.uniform(0.9, 1.0))
            base = rng.normal(0, 1, size=K)       # task returns
            phi0 = float(rng.normal())            # shared start state
            phiT = rng.normal(0, 5, size=K)       # terminal potentials

            def adv(interior_scale):
                out = []
                for k in range(K):
                    phis = np.empty(T + 1)
                    phis[0] = phi0
                    phis[1:T] = rng_interior[k][: T - 1] * interior_scale
                    phis[T] = phiT[k]
                    out.append(base[k] + shaped_transition_sum(phis, gamma))
                return group_advantages(out)

            rng_interior = [rng.normal(0, 50, size=max(T - 1, 1)) for _ in range(K)]
            a_small = adv(0.0)     # interior Phi identically zero
            a_huge = adv(1000.0)   # interior Phi wildly different
            assert np.allclose(a_small, a_huge, atol=1e-8), (
                "interior potential changed the advantage; telescoping broken"
            )

    def test_shaped_equals_reward_plus_terminal_potential(self):
        """Shaping is exactly equivalent to adding gamma^T Phi(s_T) to the reward."""
        rng = np.random.default_rng(2)
        for _ in range(200):
            K = int(rng.integers(4, 17))
            T = int(rng.integers(3, 20))
            gamma = float(rng.uniform(0.9, 1.0))
            base = rng.normal(0, 1, size=K)
            phi0 = float(rng.normal())
            shaped, equiv = [], []
            for k in range(K):
                phis = np.concatenate(([phi0], rng.normal(0, 20, size=T)))
                shaped.append(base[k] + shaped_transition_sum(phis, gamma))
                equiv.append(base[k] + gamma**T * phis[-1])
            assert np.allclose(group_advantages(shaped), group_advantages(equiv), atol=1e-8)


class TestWhatDoesSurvive:
    def test_terminal_potential_does_change_advantages(self):
        """Sanity: the theorem says interior is invisible, not that shaping is inert."""
        base = np.array([0.0, 0.0, 0.0, 0.0])
        a_flat = group_advantages(base + np.array([0.0, 0.0, 0.0, 0.0]))
        a_term = group_advantages(base + np.array([1.0, 2.0, 3.0, 4.0]))
        assert not np.allclose(a_flat, a_term)

    def test_per_token_advantages_can_see_interior_phi(self):
        """The escape hatch: per-STEP advantages are not in span{1}, so interior
        Phi reaches the gradient. This is why T8 is the only online route left."""
        gamma = 1.0
        phis_a = np.array([0.0, 5.0, 1.0, 2.0])   # same endpoints...
        phis_b = np.array([0.0, -5.0, 9.0, 2.0])  # ...different interior
        step_a = np.array([gamma * phis_a[t + 1] - phis_a[t] for t in range(3)])
        step_b = np.array([gamma * phis_b[t + 1] - phis_b[t] for t in range(3)])
        assert step_a.sum() == pytest.approx(step_b.sum())     # sequence-level: identical
        assert not np.allclose(step_a, step_b)                  # per-step: distinguishable
