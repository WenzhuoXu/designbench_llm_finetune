"""Preference rounds (llm_finetune/rl_dataxp/dpo_rows.py): the two-pass gradient of a step against autograd of the
joint loss, on a small stand-in policy (torch only), the stable sigmoid/softplus, the replay mix and the sharding."""
import copy
import math

import pytest

torch = pytest.importorskip("torch")

from llm_finetune.rl_dataxp import dpo_rows as DR  # noqa: E402


class _Policy:
    """logprobs(prompt, completion, T): a causal scorer (running mean of embeddings -> linear head), the same
    interface as trainer.Policy; ref_logprobs: a frozen copy taken at construction, before a perturbation."""

    def __init__(self, V=40, D=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Embedding(V, D).double()      # float64: the test compares the math, not rounding
        self.head = torch.nn.Linear(D, V).double()
        with torch.no_grad():
            for p in list(self.emb.parameters()) + list(self.head.parameters()):
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.5)
        self.ref = (copy.deepcopy(self.emb), copy.deepcopy(self.head))
        with torch.no_grad():                    # the policy moves away from the reference: margins are not 0
            for p in list(self.emb.parameters()) + list(self.head.parameters()):
                p.add_(torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.2)
        self.params = list(self.emb.parameters()) + list(self.head.parameters())
        self.device = torch.device("cpu")
        self.model = torch.nn.ModuleList([self.emb, self.head])
        self.ref_mode = "init"

    @staticmethod
    def _lp(emb, head, prompt, comp, T):
        ids = torch.tensor(list(prompt) + list(comp))
        e = emb(ids[:-1])
        h = e.cumsum(0) / torch.arange(1, len(ids)).unsqueeze(1)
        logits = head(h[len(prompt) - 1:]) / T
        return torch.log_softmax(logits, -1).gather(1, torch.tensor(list(comp)).unsqueeze(1)).squeeze(1)

    def logprobs(self, prompt, comp, T, raw=False):
        return self._lp(self.emb, self.head, prompt, comp, T), None

    def ref_logprobs(self, prompt, comp, T):
        with torch.no_grad():
            return self._lp(self.ref[0], self.ref[1], prompt, comp, T).detach()


class _Dist:
    world, rank = 1, 0

    def all_reduce_grads(self, params):
        pass

    def all_reduce_(self, t, op=None):
        pass


class _Opt:
    def __init__(self, params):
        self.params = params
        self.grads = None

    def step(self):
        self.grads = [p.grad.detach().clone() for p in self.params]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None


def _items(seed=1, n_pairs=5, n_rows=2):
    g = torch.Generator().manual_seed(seed)
    r = lambda lo, hi, n: torch.randint(lo, hi, (n,), generator=g).tolist()  # noqa: E731
    out = []
    for i in range(n_pairs):
        out.append({"kind": "pair", "prompt_ids": r(1, 40, 6 + i), "chosen_ids": r(1, 40, 4 + i % 3),
                    "rejected_ids": r(1, 40, 3 + i % 4), "domain": "d", "rejected_kind": "refused" if i % 2 else "no_gain"})
    for i in range(n_rows):
        out.append({"kind": "row", "prompt_ids": r(1, 40, 5 + i), "completion_ids": r(1, 40, 5), "domain": "e"})
    return out


def _joint_loss(pol, items, p):
    beta, alpha, T = p["beta"], p["alpha"], p["temperature"]
    total = 0.0
    for s in items:
        if s["kind"] == "row":
            lp, _ = pol.logprobs(s["prompt_ids"], s["completion_ids"], T)
            total = total + p["replay_weight"] * (-lp.mean())
            continue
        lc, _ = pol.logprobs(s["prompt_ids"], s["chosen_ids"], T)
        lr, _ = pol.logprobs(s["prompt_ids"], s["rejected_ids"], T)
        rc = pol.ref_logprobs(s["prompt_ids"], s["chosen_ids"], T).sum()
        rr = pol.ref_logprobs(s["prompt_ids"], s["rejected_ids"], T).sum()
        m = beta * ((lc.sum() - rc) - (lr.sum() - rr))
        total = total + torch.nn.functional.softplus(-m) + alpha * (-lc.mean())
    return total / len(items)


@pytest.mark.parametrize("beta,alpha,T", [(0.1, 1.0, 1.0), (0.5, 0.0, 0.8), (2.0, 0.3, 1.0)])
def test_two_pass_gradient_is_the_gradient_of_the_joint_loss(beta, alpha, T):
    pol = _Policy()
    items = _items()
    p = {"beta": beta, "alpha": alpha, "temperature": T, "replay_weight": 0.7, "max_grad_norm": 1e9}
    opt = _Opt(pol.params)
    m = DR.train_step(pol, items, _Dist(), opt, p)
    got = opt.grads
    for q in pol.params:
        q.grad = None
    loss = _joint_loss(pol, items, p)
    loss.backward()
    want = [q.grad.detach().clone() for q in pol.params]
    for a, b in zip(got, want):
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-12), (a - b).abs().max()
    # the reported loss of the pairs is the mean pair loss
    pair_losses = []
    for s in items:
        if s["kind"] == "pair":
            pair_losses.append(float(_joint_loss(pol, [s], p).detach()))
    assert m["loss"] == pytest.approx(sum(pair_losses) / len(pair_losses), rel=1e-5)
    assert m["pairs"] == 5 and m["replay_rows"] == 2 and m["refused_pairs"] == 2
    assert m["rejected_recompute_maxdiff"] < 1e-5


def test_margin_is_zero_at_the_reference():
    pol = _Policy()
    with torch.no_grad():
        for q, r in zip(pol.params, list(pol.ref[0].parameters()) + list(pol.ref[1].parameters())):
            q.copy_(r)
    items = [s for s in _items() if s["kind"] == "pair"]
    p = {"beta": 0.1, "alpha": 1.0, "temperature": 1.0, "replay_weight": 1.0, "max_grad_norm": 1e9}
    m = DR.train_step(pol, items, _Dist(), _Opt(pol.params), p)
    assert m["margin"] == pytest.approx(0.0, abs=1e-9) and m["loss"] == pytest.approx(math.log(2) + m["nll_c_per_token"], rel=0.5)
    assert m["chosen_logratio"] == pytest.approx(0.0, abs=1e-9) and m["rejected_logratio"] == pytest.approx(0.0, abs=1e-9)


def test_score_pairs_reports_the_margin():
    pol = _Policy()
    items = [s for s in _items() if s["kind"] == "pair"]
    d = DR.score_pairs(pol, items, _Dist(), 0.1, 1.0)
    want = []
    for s in items:
        lc = float(pol.logprobs(s["prompt_ids"], s["chosen_ids"], 1.0)[0].sum())
        lr = float(pol.logprobs(s["prompt_ids"], s["rejected_ids"], 1.0)[0].sum())
        rc = float(pol.ref_logprobs(s["prompt_ids"], s["chosen_ids"], 1.0).sum())
        rr = float(pol.ref_logprobs(s["prompt_ids"], s["rejected_ids"], 1.0).sum())
        want.append((lc - rc) - (lr - rr))
    assert d["dev_pairs"] == 5
    assert d["dev_margin"] == pytest.approx(sum(want) / 5, rel=1e-5)
    assert d["dev_acc"] == pytest.approx(sum(x > 0 for x in want) / 5)


def test_sigmoid_and_softplus_are_stable():
    for x in (-1e4, -50.0, -1.0, 0.0, 1.0, 50.0, 1e4):
        s = DR.sigmoid(x)
        assert 0.0 <= s <= 1.0 and math.isfinite(s)
        assert math.isfinite(DR.softplus(x)) and DR.softplus(x) >= 0.0
    assert DR.sigmoid(0.0) == 0.5 and DR.softplus(0.0) == pytest.approx(math.log(2))
    assert DR.softplus(1e4) == pytest.approx(1e4) and DR.softplus(-1e4) == pytest.approx(0.0, abs=1e-300)
    assert DR.softplus(3.0) == pytest.approx(math.log1p(math.exp(3.0)))


def test_mixed_items_draws_replay_rows_without_replacement():
    pairs = [{"chosen": {}, "rejected": {}, "i": i} for i in range(20)]
    replay = [{"messages": [], "j": j} for j in range(30)]
    a = DR.mixed_items(pairs, replay, 0.25, seed=3)
    assert len(a) == 25 and a[:20] == pairs
    js = [r["j"] for r in a[20:]]
    assert len(set(js)) == 5 and js == sorted(js)
    assert a == DR.mixed_items(pairs, replay, 0.25, seed=3) and a != DR.mixed_items(pairs, replay, 0.25, seed=4)
    assert len(DR.mixed_items(pairs, replay[:2], 0.25, 0)) == 22 and len(DR.mixed_items(pairs, [], 0.25, 0)) == 20


def test_shard_items_splits_every_item_once():
    items = _items(n_pairs=7, n_rows=5)
    shares = [DR.shard_items(items, 3, r) for r in range(3)]
    ids = sorted(id(s) for sh in shares for s in sh)
    assert ids == sorted(id(s) for s in items)
    loads = [sum(DR.item_cost(s) for s in sh) for sh in shares]
    assert max(loads) - min(loads) <= max(DR.item_cost(s) for s in items)
