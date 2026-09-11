"""
Make the synthetic domain contribute usable trajectories.

The diagnostic split 60 synth instances into 2 kept, 45 solved-with-zero-turns, 13 unsolved.
Forty-five are solved by the do-nothing continuation before the search chooses anything, so they
record no decisions; the 125 turns that made synth look like the highest-yielding domain came
almost entirely from the 13 FAILURES, whose trajectories are discarded. Its real usable yield
was about two trajectories per sixty instances.

The cause is the same one pipe had. The budget is drawn as

    B = sum(w) * U(1.5, 3.2)

which has nothing to do with what the instance actually needs, so mass almost never binds and
only the per-element requirement does -- and that requirement is exactly what the sizing rule
is built to satisfy. Pricing the budget off the lightest requirement-feasible design instead
makes both constraints bind, which is the condition that leaves the search something to do.
"""
import io

p = "/ocean/projects/mch250030p/wxu7/llm_finetune/design_agent/da_synth.py"
s = io.open(p, encoding="utf-8").read()

old = '''        x = [1.0] * n
        B = sum(w) * r.uniform(1.5, 3.2)
        return {"a": a, "k": k, "w": w, "x": x, "B": B, "n": n}'''

new = '''        x = [1.0] * n
        st = {"a": a, "k": k, "w": w, "x": x, "B": float("inf"), "n": n}

        # Price the budget off the LIGHTEST requirement-feasible design rather than a multiple
        # of total weight. Iterate the sizing rule at margin 1.00 to convergence: that design is
        # about as light as the requirement permits, so a pass at a safety margin oversizes and
        # exceeds the cap. Both constraints then bind and the search has a trade-off to make.
        # Drawn independently, the cap almost never bound and the sizing rule solved outright.
        probe = list(x)
        for _ in range(60):
            g = [a[i] * probe[i] ** k[i] for i in range(n)]
            if min(g) >= 0.999 and max(g) <= 1.06:
                break
            for i in range(n):
                m = g[i] if g[i] > 1e-9 else 1e-9
                probe[i] = min(4.0, max(0.05,
                                        probe[i] * min(1.5, max(0.7, (1.0 / m) ** (1.0 / max(k[i], 0.3))))))
        st["B"] = 1.04 * sum(w[i] * probe[i] ** 2 for i in range(n))
        return st'''

assert old in s, "synth load tail not found"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("synth budget repriced off the minimum-requirement design")
