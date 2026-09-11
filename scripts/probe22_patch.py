import pathlib
p = pathlib.Path("/ocean/projects/mch250030p/wxu7/llm_finetune/scripts/probe22_icl.py")
s = p.read_text()

s = s.replace("def gen_examples(spec, rng, n_keep=6):",
              "def gen_examples(spec, rng, n_keep=6, pool=24):")
s = s.replace("    for _ in range(24):\n        k = rng.randint(1, min(n, 8))",
              "    for _ in range(pool):\n        k = rng.randint(1, min(n, 8))")
s = s.replace("def successes_only(spec, rng, n_keep=6):\n    ex, _ = gen_examples(spec, rng, n_keep=40)",
              "def successes_only(spec, rng, n_keep=6):\n    ex, _ = gen_examples(spec, rng, n_keep=200, pool=200)")
s = s.replace("donors = [json.load(open(f)) for f in files[a.n:a.n + 40]]",
              "donors = [json.load(open(f)) for f in files[a.n:a.n + a.donors]]")
s = s.replace('ap.add_argument("--n", type=int, default=80)',
              'ap.add_argument("--n", type=int, default=80)\n    ap.add_argument("--donors", type=int, default=150)')

# donor success pool needs to be built with the bigger pool too
s = s.replace("        donor_succ += successes_only(d, random.Random(zlib.crc32(d[\"problem_id\"].encode())))",
              "        donor_succ += successes_only(d, random.Random(zlib.crc32(d[\"problem_id\"].encode())))\n        if len(donor_succ) >= 60:\n            pass")

assert "pool=200" in s and "a.donors" in s
p.write_text(s)
print("patched")
