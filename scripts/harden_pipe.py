"""
Make the pipe domain produce trajectories again.

The harvest showed pipe solving 60/60 with ZERO recorded turns: the do-nothing continuation --
plain size_pass iterated to the horizon -- reaches feasibility before the search ever chooses
anything. A domain that its own sizing rule solves outright contributes no decisions, and a
turn-balanced corpus cannot include it.

The cause is traceable to the A1 margin fix. Redefining a node's margin from press/preq (floored
at 1e-3) to P0/(preq + drops) left the feasibility test identical -- both say P0 >= preq + drops
-- but size_pass reads margins to compute its factors, so the new form gives it a usable
gradient everywhere instead of a flat floor. That was the point for the observation, and the
side effect was a much stronger heuristic: pipe's heuristic previously solved 0.8240.

The budget was priced at 1.08x the material of a UNIFORM reference diameter, which is generous
against any sensibly graded profile, so mass never bound and only pressure did. It is now priced
off the lightest strength-feasible design -- size_pass iterated at margin 1.00 to convergence --
so a fixed-margin pass at 1.05 oversizes and busts the cap, while choosing the margin correctly
fits. Strength and mass then bind together, which is the condition that makes the truss
interesting.
"""
import io

p = "/ocean/projects/mch250030p/wxu7/llm_finetune/design_agent/da_pipe.py"
s = io.open(p, encoding="utf-8").read()

old = '''        st["dmax"] = 4.0 * ref
        st["B"] = 1.08 * sum(L[i] * ref ** 2 for i in range(n))'''

new = '''        st["dmax"] = 4.0 * ref

        # Price the budget off the LIGHTEST strength-feasible profile rather than a uniform
        # reference. Iterate the sizing rule at margin 1.00 to convergence; that design is
        # about as light as strength permits, so a pass at margin 1.05 oversizes and exceeds
        # the cap. Both constraints then bind, which a uniform reference at 1.08x never did.
        st["d"] = [ref] * n
        st["cache"] = None
        for _ in range(60):
            em = self.element_margins(st)
            if min(em) >= 0.999 and max(em) <= 1.06:
                break
            for i in range(n):
                m = em[i] if em[i] > 1e-9 else 1e-9
                st["d"][i] = min(st["dmax"], max(0.05,
                                                 st["d"][i] * min(1.5, max(0.7, (1.0 / m) ** 0.2))))
            st["cache"] = None
        st["B"] = 1.03 * sum(L[i] * st["d"][i] ** 2 for i in range(n))'''

assert old in s, "pipe budget block not found"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("pipe budget repriced off the minimum-strength profile")
