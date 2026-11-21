"""rate.py [lines] [minutes]: the servers' own load lines (one per 10 s) averaged over the last <lines>, and the replies
written to samples.jsonl in the last <minutes> (by the records' order in the file and the file's size now)."""
import glob
import os
import re
import sys
import time

L = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = "/project/community/wxu2/rl/screen/dx38mtR_scr2_fcq"
PAT = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d).*prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
                 r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")


def tail(path, n_bytes=400000):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n_bytes))
        return f.read().decode("utf-8", "replace").splitlines()


print(time.strftime("%F %T"), "last %d load lines per server" % L)
tot_g = tot_p = 0.0
for path in sorted(glob.glob(OUT + "/logs/vllm_server_*_job173954.log")):
    rows = [m.groups() for m in (PAT.search(x) for x in tail(path)) if m][-L:]
    if not rows:
        print(os.path.basename(path), "no load lines")
        continue
    mean = lambda i: sum(float(r[i]) for r in rows) / len(rows)
    tot_g += mean(2)
    tot_p += mean(1)
    print("%s %s..%s gen %6.0f tok/s, prompt %6.0f tok/s, running %4.1f, waiting %5.1f, KV %5.1f%%, hit rate (since start) %s%%" % (
        "server " + os.path.basename(path)[12:13], rows[0][0][6:], rows[-1][0][6:], mean(2), mean(1), mean(3), mean(4), mean(5), rows[-1][6]))
print("all servers: generation %.0f tok/s, prompt %.0f tok/s" % (tot_g, tot_p))
s = OUT + "/samples.jsonl"
n = sum(1 for _ in open(s, "rb"))
print("samples.jsonl: %d lines, %.1f MB, last written %s" % (n, os.path.getsize(s) / 1e6, time.strftime("%T", time.localtime(os.path.getmtime(s)))))
