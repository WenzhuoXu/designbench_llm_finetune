"""cache_test.py <server url .../v1> [tokens=6000]: does the server reuse a prompt between requests? Unique prompts
(random token ids, so nothing else on the server shares them), one generated token each, latency per request.
  A: a prompt alone, then the same again                       (reuse after a request has finished)
  B: a prompt alone, then 7 of the same at once                (the sampler's pattern: first reply, then its siblings)
  C: a prompt alone, then the same plus 40 more tokens         (the sampler's continuation of a reply)
  D: 7 of a new prompt at once, nothing before                 (reuse inside one batch)
"""
import json
import random
import sys
import threading
import time
import urllib.request

url = sys.argv[1].rstrip("/")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
rng = random.Random(int(time.time()))


def prompt(k=n):
    return [rng.randrange(2000, 150000) for _ in range(k)]


def metrics():
    root = url[:-3] if url.endswith("/v1") else url
    out = {}
    with urllib.request.urlopen(root + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:prefix_cache_queries_total") or line.startswith("vllm:prefix_cache_hits_total"):
                out[line.split("{")[0].split(":")[1]] = float(line.rsplit(" ", 1)[1])
    return out


def ask(ids, box=None, i=0):
    body = {"model": "dx38mtR", "prompt": ids, "max_tokens": 1, "temperature": 0.8, "add_special_tokens": False, "seed": i}
    t0 = time.time()
    req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    dt = time.time() - t0
    if box is not None:
        box[i] = dt
    return dt, out.get("usage")


def many(ids, k=7):
    box = [None] * k
    th = [threading.Thread(target=ask, args=(ids, box, i)) for i in range(k)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return box


def fmt(v):
    return " ".join("%.2f" % x for x in v)


m0 = metrics()
p = prompt()
a1, usage = ask(p)
a2, _ = ask(p)
print("usage of one request:", usage)
print("A  alone %.2f s, the same again %.2f s" % (a1, a2))
p = prompt()
b1, _ = ask(p)
print("B  alone %.2f s, then 7 at once: %s s" % (b1, fmt(many(p))))
p = prompt()
c1, _ = ask(p)
c2, _ = ask(p + prompt(40))
print("C  alone %.2f s, the same plus 40 tokens %.2f s" % (c1, c2))
p = prompt()
print("D  7 of a new prompt at once: %s s" % fmt(many(p)))
m1 = metrics()
print("server counters during the test (other traffic included): queries +%.0f, hits +%.0f; this test asked %d prompt tokens" % (
    m1.get("prefix_cache_queries_total", 0) - m0.get("prefix_cache_queries_total", 0),
    m1.get("prefix_cache_hits_total", 0) - m0.get("prefix_cache_hits_total", 0), n * 19 + 40))
