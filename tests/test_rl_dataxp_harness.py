"""dataxp GRPO rollouts against the real evaluation harness and a fake vLLM server.

Needs the harness directory (scripts/dataxp_policy_eval_v3.py + assemble_dataxp_sft.py,
dataxp_code_v3/, test/problems_old3.jsonl), e.g. a copy of Orchard's
/project/community/wxu2/corpus/harness:

    DX_HARNESS_DIR=/path/to/harness pytest tests/test_rl_dataxp_harness.py

The fake server answers every chat completion with a legal verb-spelled move chosen by a hash of
the prompt, so it is deterministic and stateless, and returns token ids / logprobs only when the
request asks for them, as vLLM does.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

HDIR = os.environ.get("DX_HARNESS_DIR")
pytestmark = pytest.mark.skipif(not HDIR or not os.path.isdir(HDIR or ""), reason="DX_HARNESS_DIR not set")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANT_KEYS = {"circuit2": "SET_BIAS_CURRENT", "thermal2": "MOVE_SINK", "frame3d2": "INSTALL_MEMBER"}


def paths():
    return (os.path.join(HDIR, "scripts", "dataxp_policy_eval_v3.py"), os.path.join(HDIR, "dataxp_code_v3"),
            os.path.join(HDIR, "test", "problems_old3.jsonl"))


def problems():
    return [json.loads(l) for l in open(paths()[2])]


@pytest.fixture(scope="module")
def candidates():
    """Legal moves at each problem's start, verb-spelled."""
    code = paths()[1]
    out = subprocess.run([sys.executable, "-c", """
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
from dataxp.domains import get_domain
res = {}
for line in open(sys.argv[2]):
    r = json.loads(line); d = get_domain(r['variant']); st = d.load(r['spec'])
    res[r['variant']] = [d.format_move(m) for m in d.legal_moves(st, np.random.default_rng(0))]
print(json.dumps(res))
""", code, paths()[2]], capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED="0"), check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


class FakeVLLM:
    def __init__(self, candidates, honour_token_ids=True):
        self.cands = candidates
        self.honour = honour_token_ids
        self.bodies = []
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                raw = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send({"data": [{"id": "base"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.bodies.append(body)
                self._send(fake.reply(body))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def reply(self, body):
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        variant = next((v for v, k in VARIANT_KEYS.items() if k in user), None)
        h = hashlib.sha1((system + user).encode()).digest()
        if variant is None:
            text = "no idea"
        else:
            c = self.cands[variant]
            text = "Thinking about it.\n<action>%s</action>" % c[int.from_bytes(h[:4], "big") % len(c)]
        ch = {"index": 0, "message": {"role": "assistant", "content": text, "reasoning_content": None},
              "finish_reason": "stop"}
        out = {"id": "x", "model": body["model"], "choices": [ch],
               "usage": {"prompt_tokens": 16, "completion_tokens": len(text)}}
        if body.get("return_token_ids") and self.honour:
            ch["token_ids"] = [ord(x) % 251 for x in text]
            out["prompt_token_ids"] = [b % 251 for b in h] + [len(user) % 251]
        if body.get("logprobs"):
            ch["logprobs"] = {"content": [{"token": x, "logprob": -0.5} for x in text]}
        return out

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server(candidates):
    s = FakeVLLM(candidates)
    yield s
    s.close()


def env():
    return dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=REPO)


def test_harness_args_are_the_harness_defaults():
    sys.path.insert(0, REPO)
    from llm_finetune.rl_dataxp.harness import harness_args, load_harness
    H = load_harness(paths()[0])
    a = harness_args(H, ["--code", paths()[1], "--out", os.devnull])
    assert (a.mode, a.policy, a.turns, a.max_tokens, a.temperature, a.native_prompt, a.loop_guard) == (
        "episodes", "llm", 20, 2048, 0.0, "served", False)
    assert H.cmd_episodes.__name__ == "cmd_episodes"            # restored after the capture


def test_run_one_records_tokens_and_injects(server):
    sys.path.insert(0, REPO)
    from llm_finetune.rl_dataxp.harness import run_one
    p = problems()[0]
    res = run_one({"script": paths()[0], "variant": p["variant"], "spec": p["spec"], "seed": 123,
                   "inject": {"return_token_ids": True, "logprobs": True, "top_p": 1.0, "top_k": 0},
                   "argv": ["--code", paths()[1], "--no-think", "--model", "policy_s00000", "--base-url", server.url,
                            "--temperature", "0.8", "--turns", "4", "--out", os.devnull]})
    ep, calls = res["episode"], res["calls"]
    assert res["rl_error"] is None
    assert len(calls) == len(ep["turns"]) == 4
    assert all(c["prompt_token_ids"] and c["token_ids"] and c["logprobs"] for c in calls)
    bodies = server.bodies
    assert all(b["return_token_ids"] and b["top_p"] == 1.0 and b["top_k"] == 0 for b in bodies)
    assert all(b["model"] == "policy_s00000" and b["temperature"] == 0.8 for b in bodies)
    assert len({b["seed"] for b in bodies}) == 4                   # one seed per call
    assert [c["seed"] for c in calls] == [b["seed"] for b in bodies]


def run_rollout(tmp_path, url, eps, turns=6, temperature=0.0, name="out"):
    req = {"script": paths()[0], "harness_argv": ["--code", paths()[1], "--no-think"], "servers": [url], "model": "base",
           "temperature": temperature, "turns": turns,
           "inject": {"return_token_ids": True, "logprobs": True}, "workers": 3, "env": {}, "episodes": eps}
    rp, op = tmp_path / ("%s.request.json" % name), tmp_path / ("%s.jsonl" % name)
    rp.write_text(json.dumps(req))
    pr = subprocess.run([sys.executable, "-m", "llm_finetune.rl_dataxp.rollout", "run", "--request", str(rp),
                         "--out", str(op)], cwd=REPO, env=env(), capture_output=True, text=True)
    return pr, op


def episodes_for(probs, group="g"):
    return [{"key": "%s/%s/m0" % (group, p["spec"]["problem_id"]), "group": "%s/%s" % (group, p["spec"]["problem_id"]),
             "member": 0, "variant": p["variant"], "spec": p["spec"], "seed": 7, "server": 0} for p in probs]


def test_parity_with_the_evaluation_harness(server, tmp_path):
    """The same problems through the harness's own episodes mode and through RL rollouts, at T=0
    against the same deterministic server: identical episodes, turn for turn."""
    script, code, probs_file = paths()
    out = tmp_path / "eval"
    pr = subprocess.run([sys.executable, script, "episodes", "--code", code, "--no-think", "--policy", "llm", "--model", "base",
                         "--base-url", server.url, "--temperature", "0", "--turns", "6",
                         "--variants", "circuit2", "thermal2", "frame3d2", "--n", "1",
                         "--problems-file", probs_file, "--workers", "3", "--out", str(out)],
                        env=env(), capture_output=True, text=True)
    assert pr.returncode == 0, pr.stdout + pr.stderr
    ev = {json.loads(l)["problem_id"]: json.loads(l) for l in open(str(out) + ".episodes.jsonl")}
    pr, op = run_rollout(tmp_path, server.url, episodes_for(problems()))
    assert pr.returncode == 0, pr.stdout + pr.stderr
    rl = {json.loads(l)["problem_id"]: json.loads(l) for l in open(op)}
    assert set(ev) == set(rl) and len(ev) == 3
    n_moves = 0
    for pid in ev:
        a, b = dict(ev[pid]), dict(rl[pid]["episode"])
        a.pop("policy"), b.pop("policy")
        assert a == b, pid
        n_moves += sum(1 for t in a["turns"] if t.get("move"))
        assert len(rl[pid]["calls"]) == len(a["turns"])
    assert n_moves > 0                                             # the moves were applied, not all refused


def test_rollout_resumes_and_cuts_a_partial_line(server, tmp_path):
    eps = episodes_for(problems()[:2])
    pr, op = run_rollout(tmp_path, server.url, eps[:1], turns=3)
    assert pr.returncode == 0, pr.stderr
    first = op.read_text()
    with open(op, "a") as f:
        f.write('{"key": "half-writ')                           # a job killed mid-write
    n_before = len(server.bodies)
    pr, op = run_rollout(tmp_path, server.url, eps, turns=3)
    assert pr.returncode == 0, pr.stderr
    lines = op.read_text().splitlines()
    assert lines[0] == first.strip() and len(lines) == 2
    assert {json.loads(l)["key"] for l in lines} == {e["key"] for e in eps}
    assert len(server.bodies) - n_before == 3                      # only the missing episode ran


def test_rollout_fails_loudly_without_token_ids(candidates, tmp_path):
    s = FakeVLLM(candidates, honour_token_ids=False)
    try:
        pr, op = run_rollout(tmp_path, s.url, episodes_for(problems()[:1]), turns=2)
    finally:
        s.close()
    assert pr.returncode == 4
    assert json.loads(op.read_text().splitlines()[0])["rl_error"] == "no_token_ids"


def test_problems_builder_drops_eval_and_heldout(tmp_path):
    sys.path.insert(0, REPO)
    from llm_finetune.rl_dataxp import problems as P
    script, code, probs_file = paths()
    probs = problems()
    corpus = tmp_path / "circuit2.jsonl"
    c2 = [p["spec"] for p in probs if p["variant"] == "circuit2"][0]
    other = dict(c2, problem_id="circuit2_9999", goals=dict(c2["goals"], minimum_gain_db=c2["goals"]["minimum_gain_db"] + 1))
    corpus.write_text("\n".join(json.dumps(s) for s in (c2, dict(c2, problem_id="renamed"), other, other)) + "\n")
    out = tmp_path / "train.jsonl"
    s = P.build(P_args(script, code, ["circuit2=%s" % corpus], [probs_file], ["thermal2"], str(out)))
    c = s["per_variant"]["circuit2"]
    assert c["read"] == 4 and c["eval_id"] == 1 and c["eval_spec"] == 1 and c["duplicate"] == 1
    assert c.get("kept", 0) + c.get("feasible_start", 0) == 1
    with pytest.raises(SystemExit, match="held-out"):
        P.build(P_args(script, code, ["thermal2=%s" % corpus], [probs_file], ["thermal2"], str(out)))


def P_args(script, code, corpus, evals, heldout, out):
    import argparse
    return argparse.Namespace(script=script, code=code, corpus=corpus, eval_problems=evals, heldout=heldout, out=out)
