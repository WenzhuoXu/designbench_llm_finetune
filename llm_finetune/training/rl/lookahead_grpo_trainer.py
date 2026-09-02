"""GRPO with per-token lookahead advantages.

Why this exists. GRPO computes ONE advantage per sequence and broadcasts it to
every token: the signal it can deliver lives in ``span{1_[k]}``. A per-step
quantity folded into that scalar -- which is what ``lookahead_advantage`` as a
reward term does, summing the per-step counterfactual advantages into one number
-- is therefore projected onto that span before it ever reaches the gradient.
Decompose a per-token vector as

    A^k  =  mean_t(A^k_t) * 1  +  A~^k ,      A~^k orthogonal to 1

and the scalar form keeps only the first component. No group-level affine
transform of any per-sequence reward can produce the second. Measured
consequence: the scalar form (arms T5c, T5co) moved the policy's action ranking
not at all -- paired per-step rank_frac against its Phi_v2-only control,
t = +0.67 over 14 matched steps, point estimate in the wrong direction.

TRL 1.0 already accommodates this. ``grpo_trainer.py:2291`` documents that a
subclass may supply advantages of shape (B, T) and the loss will use them
per-token. And ``grpo_trainer.py:1949`` merges a rollout_func's extra fields into
the ``inputs`` list IN PLACE, so the per-token vector the rollout computed is
readable here after ``super()`` returns, already sliced to this process.

The delivered advantage keeps the trajectory term and adds the per-step one:

    A_token = A_sequence  +  kappa * zscore(A_step)

The trajectory term is what GRPO always had; the second is the channel that was
missing. z-scoring over the batch's model tokens makes kappa dimensionless, so it
does not have to be retuned when Phi's scale changes between problems or domains.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def make_lookahead_trainer_cls():
    """Build the subclass at call time.

    ``from trl import GRPOTrainer`` at module scope fails on this cluster's login
    nodes (trl 1.0 wants ``torch.distributed.fsdp.FSDPModule``, absent in the
    installed torch 2.4) while succeeding on the compute nodes where training
    actually runs. llm_finetune/training/rl/grpo_trainer.py:97 imports inside a
    function for the same reason; do likewise so this module stays importable
    everywhere, including for tests that never construct a trainer.
    """
    from trl import GRPOTrainer

    class LookaheadGRPOTrainer(GRPOTrainer):
        """GRPOTrainer that injects per-step lookahead credit at the token level.

        ``kappa`` scales the per-token term against the trajectory term. 0.0
        reproduces stock GRPO exactly, which makes it the control for this arm.
        """

        def __init__(self, *args, lookahead_kappa: float = 1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.lookahead_kappa = float(lookahead_kappa)
            self._lookahead_logged = False

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)

            def _once(msg):
                if not self._lookahead_logged:
                    log.info(msg)
                    self._lookahead_logged = True

            return inject_per_token_advantages(
                output, inputs, self.lookahead_kappa, on_first=_once)

    return LookaheadGRPOTrainer


def inject_per_token_advantages(output, inputs, kappa, *, on_first=None):
    """Add the per-step term to a stock GRPO advantage tensor.

        A_token = A_sequence + kappa * zscore(A_step)

    ``output`` is what GRPOTrainer._generate_and_score_completions returned;
    ``inputs`` is the same list it was given, which TRL has already mutated in
    place with the rollout_func's extra fields (grpo_trainer.py:1949). Returns
    ``output`` unchanged when there is nothing to inject, so the caller is always
    safe.
    """
    import torch

    if kappa == 0.0:
        return output
    advantages = output.get("advantages")
    completion_ids = output.get("completion_ids")
    if advantages is None or completion_ids is None or advantages.dim() != 1:
        return output       # nothing to do, or already per-token

    width = completion_ids.shape[1]
    rows = []
    for inp in inputs:
        vec = inp.get("step_token_advantages")
        if not isinstance(vec, (list, tuple)) or not vec:
            rows.append([0.0] * width)
            continue
        v = list(vec)[:width]
        rows.append(v + [0.0] * (width - len(v)))

    if len(rows) != advantages.shape[0]:
        # Alignment is the one thing that must never be guessed: a silent mismatch
        # would attribute one rollout's credit to another rollout's tokens.
        if on_first:
            on_first(f"[lookahead-grpo] {len(rows)} per-token vectors for "
                     f"{advantages.shape[0]} sequences; using sequence-level advantages")
        return output

    step = torch.tensor(rows, dtype=advantages.dtype, device=advantages.device)

    # A per-token arm whose step vector is all zeros IS the control, silently.
    # Four sweeps in this project's history measured a disconnected mechanism;
    # say so on the first batch instead of discovering it after 36 SU.
    if on_first:
        nz = int((step != 0).sum().item())
        tot = int(step.numel())
        seqs_with_credit = int(((step != 0).any(dim=1)).sum().item())
        msg = (f"[lookahead-grpo] per-token credit: {nz}/{tot} tokens nonzero, "
               f"{seqs_with_credit}/{step.shape[0]} sequences carry credit")
        on_first(msg if nz else msg + "  <<< ZERO CREDIT: this arm is the control")

    # Standardise over CREDIT-BEARING model tokens only. Environment tokens carry
    # 0 by construction and are masked out of the loss; including them would drag
    # the mean toward zero and shrink the signal.
    mask = output["completion_mask"].to(step.dtype) if output.get("completion_mask") is not None \
        else torch.ones_like(step)
    active = mask * (step != 0).to(step.dtype)
    n_active = active.sum()
    if n_active > 1:
        mean = (step * active).sum() / n_active
        var = ((step - mean) ** 2 * active).sum() / n_active
        step = ((step - mean) / (var.sqrt() + 1e-6)) * active
    else:
        step = torch.zeros_like(step)

    output["advantages"] = advantages.unsqueeze(1) + kappa * step
    if on_first:
        on_first(f"[lookahead-grpo] per-token advantages active: kappa={kappa:.3f}, "
                 f"shape={tuple(output['advantages'].shape)}, "
                 f"credit-bearing tokens={int(n_active.item())}/{int(mask.sum().item())}")
    return output
