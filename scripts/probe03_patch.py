import pathlib
p = pathlib.Path("/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/probe03_magnitude_swap.py")
s = p.read_text()

old_crit = '''    crit = sorted(((min(float(getattr(m, "fos_buckling", 9e99) or 9e99),
                        float(getattr(m, "fos_yielding", 9e99) or 9e99)), i)
                   for i, m in enumerate(truss.members)))
    order = [i for _, i in crit]'''
new_crit = '''    try:
        _analyze_truss(truss, goals)      # per-member FOS only exists after analysis
    except Exception:
        return None

    def _fb(m):
        try:
            v = float(getattr(m, "fos_buckling", float("inf")) or float("inf"))
        except Exception:
            v = float("inf")
        return v if v == v else float("inf")

    def _fy(m):
        try:
            v = float(getattr(m, "fos_yielding", float("inf")) or float("inf"))
        except Exception:
            v = float("inf")
        return v if v == v else float("inf")

    crit = sorted(((min(_fb(m), _fy(m)), i) for i, m in enumerate(truss.members)))
    order = [i for _, i in crit]'''
assert old_crit in s, "crit block not found"
s = s.replace(old_crit, new_crit)

old_feas = '''    def feas(nt):
        return bool(_analyze_truss(nt, goals).get("is_feasible"))'''
new_feas = '''    def feas(nt):
        try:
            return bool(_analyze_truss(nt, goals).get("is_feasible"))
        except Exception:
            return False'''
assert old_feas in s, "feas block not found"
s = s.replace(old_feas, new_feas)

old_p = '''            fb = float(getattr(m, "fos_buckling", float("inf")) or float("inf"))
            fy = float(getattr(m, "fos_yielding", float("inf")) or float("inf"))'''
new_p = '''            fb = _fb(m); fy = _fy(m)'''
assert old_p in s, "arm P block not found"
s = s.replace(old_p, new_p)

old_call = '''            r = run(json.load(open(p)), FAC, a.K, rng)'''
new_call = '''            try:
                r = run(json.load(open(p)), FAC, a.K, rng)
            except Exception:
                r = None'''
assert old_call in s, "call site not found"
s = s.replace(old_call, new_call)

p.write_text(s)
print("patched OK")
