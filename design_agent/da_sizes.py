import json, glob, collections, os
os.chdir("/ocean/projects/mch250030p/wxu7/DesignBench/data")


def size(s):
    if not isinstance(s, dict):
        return 0
    t = s.get("topology")
    if isinstance(t, dict):
        return len(t.get("members") or [])
    if isinstance(t, list):
        return len(t)
    return 0


for d in ("problems", "problems_hard", "problems_gen", "hf_benchmark"):
    fs = sorted(glob.glob(d + "/**/*.json", recursive=True))
    c = collections.Counter()
    for f in fs[:600]:
        try:
            s = json.load(open(f))
        except Exception:
            continue
        m = size(s)
        if m:
            c[m] += 1
    if c:
        ks = sorted(c)
        print("%-14s files=%-5d scanned=%-4d members %d..%d  dist=%s"
              % (d, len(fs), sum(c.values()), ks[0], ks[-1],
                 dict(sorted(c.items())[:14])))
    else:
        print("%-14s files=%-5d  no topology found" % (d, len(fs)))
