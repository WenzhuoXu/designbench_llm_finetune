"""scripts/rl_probe/ab_adapters.py: two adapters on the same problems, paired per problem."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ab_adapters", os.path.join(ROOT, "scripts", "rl_probe", "ab_adapters.py"))
AB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AB)


def episode(variant, pid, member, f20, turns=(), **extra):
    ep = {"variant": variant, "problem_id": pid, "feasible@12": False, "feasible@20": f20, "turns": list(turns)}
    ep.update(extra)
    return {"key": "x/r%d/%s" % (member, pid), "member": member, "variant": variant, "problem_id": pid, "episode": ep}


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_table_pairs_problems_and_skips_feasible_starts(tmp_path, capsys):
    refused = {"fail": "illegal:MOVE(1) is not allowed", "completion_tokens": 10}
    applied = {"fail": None, "completion_tokens": 30}
    a = [episode("fam", "p1", 1, True, [applied]), episode("fam", "p1", 2, False, [refused, applied]),
         episode("fam", "p2", 1, False, [refused]), episode("fam", "p2", 2, False, [refused]),
         episode("fam", "p3", 1, True, feasible_start=True), episode("fam", "p4", 1, True, crashed=True)]
    b = [episode("fam", "p1", 1, True, [applied]), episode("fam", "p1", 2, True, [applied]),
         episode("fam", "p2", 1, True, [applied]), episode("fam", "p2", 2, False, [refused]),
         episode("fam", "p3", 1, True, feasible_start=True), episode("fam", "p4", 1, True)]
    pa, pb = str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")
    write(pa, a)
    write(pb, b)
    assert AB.main([pa, pb, "--names", "old", "new", "--boot", "200", "--groups", "g=fam"]) == 0
    out = capsys.readouterr().out
    # p3 starts feasible and p4 crashed in one arm: 4 episodes of 2 problems are compared
    assert "in both 4 " in out
    row = [line for line in out.splitlines() if line.startswith("fam ")][0]
    assert row.split("|")[0].split() == ["fam", "2", "4"]
    assert row.split("|")[2].split()[:3] == ["0.250", "0.750", "+0.500"]
    # both problems have more feasible runs in the second arm; the runs of p1 disagree in the first arm
    # and those of p2 in the second
    assert row.split("|")[3].split()[:3] == ["0", "/", "2"]
    assert row.split("|")[4].split() == ["1", "/", "1"]
    turns = [line for line in out.splitlines() if line.startswith("fam ")][1]
    assert turns.split("|")[2].split() == ["60.0%", "25.0%"]


def test_sign_test_is_two_sided():
    assert AB.sign_p(0, 0) == 1.0
    assert abs(AB.sign_p(0, 5) - 2 / 32) < 1e-12
    assert AB.sign_p(3, 3) == 1.0
