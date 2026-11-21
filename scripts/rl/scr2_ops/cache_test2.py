"""cache_test2.py <server url .../v1> [tokens=6000]: which request patterns reuse a prompt already computed. Unique random
prompts, one generated token per request; for each pattern the latency of every request and the server's prefix-cache
counters over just that pattern (other traffic on the server adds to both counters)."""
import json
import random
import sys
import threading
import time
import urllib.request

url = sys.argv[1].rstrip("/")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
rng = random.Random(int(time.time()))
root = url[:-3] if url.endswith("/v1") else url


def prompt(k=n):
    return [rng.randrange(2000, 150000) for _ in range(k)]


def metrics():
    out = {}
    with urllib.request.urlopen(root + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:prefix_cache_queries_total") or line.startswith("vllm:prefix_cache_hits_total"):
                out[line.split("{")[0].split(":")[1]] = float(line.rsplit(" ", 1)[1])
    return out.get("prefix_cache_queries_total", 0.0), out.get("prefix_cache_hits_total", 0.0)


def ask(ids, box=None, i=0, wait=0.0, max_tokens=1):
    if wait:
        time.sleep(wait)
    body = {"model": "dx38mtR", "prompt": ids, "max_tokens": max_tokens, "temperature": 0.8, "add_special_tokens": False, "seed": 7}
    t0 = time.time()
    req = urllib.request.Request(url + "/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        json.load(r)
    dt = time.time() - t0
    if box is not None:
        box[i] = dt
    return dt


def many(ids, k, stagger=0.0):
    box = [None] * k
    th = [threading.Thread(target=ask, args=(ids, box, i, i * stagger)) for i in range(k)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return box


def run(name, fn, asked):
    q0, h0 = metrics()
    t0 = time.time()
    lat = fn()
    dt = time.time() - t0
    q1, h1 = metrics()
    print("%-46s %s s | counters: queries +%6.0f hits +%6.0f in %.1f s (this pattern asked %d)" % (
        name, " ".join("%.2f" % x for x in lat), q1 - q0, h1 - h0, dt, asked), flush=True)


q0, h0 = metrics()
time.sleep(3.0)
q1, h1 = metrics()
print("other traffic alone, 3 s: queries +%.0f hits +%.0f" % (q1 - q0, h1 - h0))
p = prompt()
run("1 new prompt alone", lambda: [ask(p)], n)
run("the same once more", lambda: [ask(p)], n)
run("the same, 2 at once", lambda: many(p, 2), 2 * n)
run("the same, 7 at once", lambda: many(p, 7), 7 * n)
run("the same, 7 started 0.3 s apart", lambda: many(p, 7, 0.3), 7 * n)
run("the same, 7 one after another", lambda: [ask(p) for _ in range(7)], 7 * n)
p2 = prompt()
run("a new prompt, 7 at once", lambda: many(p2, 7), 7 * n)
run("that prompt again, 7 at once", lambda: many(p2, 7), 7 * n)
