"""metrics_window.py <seconds> <server url .../v1> [...]: the servers' counters over a window: prefix-cache queries and
hits, finished requests, prompt and generation tokens, preemptions; per finished request, the prompt tokens that were
not found in the cache."""
import sys
import time
import urllib.request

secs = float(sys.argv[1])
urls = [u.rstrip("/") for u in sys.argv[2:]]
WANT = ["prefix_cache_queries_total", "prefix_cache_hits_total", "request_success_total", "prompt_tokens_total",
        "generation_tokens_total", "num_preemptions_total", "prompt_tokens_cached_total", "request_prefill_time_seconds_sum",
        "request_queue_time_seconds_sum", "request_decode_time_seconds_sum", "request_inference_time_seconds_sum"]


def read(u, names=False):
    root = u[:-3] if u.endswith("/v1") else u
    out = {}
    with urllib.request.urlopen(root + "/metrics", timeout=30) as r:
        text = r.read().decode()
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:"):
            continue
        name = line.split("{")[0].split(" ")[0][5:]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + val
    if names:
        print("metric names:", " ".join(sorted(k for k in out if not k.endswith("_bucket") and not k.endswith("_created"))))
    return out


a = [read(u, names=(i == 0)) for i, u in enumerate(urls)]
time.sleep(secs)
b = [read(u) for u in urls]
for u, x, y in zip(urls, a, b):
    d = {k: y.get(k, 0.0) - x.get(k, 0.0) for k in WANT}
    n = max(d["request_success_total"], 1.0)
    miss = d["prefix_cache_queries_total"] - d["prefix_cache_hits_total"]
    print("%s  %.0f s: requests finished %4.0f | queried %7.0f hit %7.0f (%.0f%%) not in cache %7.0f = %5.0f per request | "
          "prompt tokens %7.0f, generated %6.0f (%.0f/s) | preemptions %.0f | per request: queue %.2f s, prefill %.2f s, decode %.2f s" % (
              u.split(":")[-1].split("/")[0], secs, d["request_success_total"], d["prefix_cache_queries_total"], d["prefix_cache_hits_total"],
              100.0 * d["prefix_cache_hits_total"] / max(d["prefix_cache_queries_total"], 1.0), miss, miss / n,
              d["prompt_tokens_total"], d["generation_tokens_total"], d["generation_tokens_total"] / secs, d["num_preemptions_total"],
              d["request_queue_time_seconds_sum"] / n, d["request_prefill_time_seconds_sum"] / n, d["request_decode_time_seconds_sum"] / n))
