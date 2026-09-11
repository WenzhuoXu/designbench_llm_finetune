"""Does forbidding shrink help when the signal is CLEAN? Reads the saved mechanism rows."""
import json
rows = [json.loads(l) for l in
        open("/ocean/projects/mch250030p/wxu7/llm_finetune/results/da_why_rows.jsonl")]
cl = [r for r in rows if r["mode"] == "perm" and r["level"] == 0.0]
print("uncorrupted cell only (rho = 1: the signal a real agent actually gets)")
print("  mean rho check %.4f, n=%d\n" % (sum(r["rho"] for r in cl) / len(cl), len(cl)))
print("  domain  slack   shrink allowed    grow only     diff")
for dm in ("truss", "synth"):
    for sl in (1.0, 4.0):
        a = [r["y"] for r in cl if r["domain"] == dm and r["slack"] == sl and r["clip"] == 0.7]
        b = [r["y"] for r in cl if r["domain"] == dm and r["slack"] == sl and r["clip"] == 1.0]
        if not a or not b:
            continue
        print("  %-6s  %4.1f    %.3f (n=%3d)     %.3f (n=%3d)   %+.3f"
              % (dm, sl, sum(a) / len(a), len(a), sum(b) / len(b), len(b),
                 sum(b) / len(b) - sum(a) / len(a)))
print("\nall cells pooled (mostly corrupted signals, for contrast)")
for dm in ("truss", "synth"):
    for sl in (1.0, 4.0):
        a = [r["y"] for r in rows if r["domain"] == dm and r["slack"] == sl and r["clip"] == 0.7]
        b = [r["y"] for r in rows if r["domain"] == dm and r["slack"] == sl and r["clip"] == 1.0]
        print("  %-6s  %4.1f    %.3f            %.3f         %+.3f"
              % (dm, sl, sum(a) / len(a), sum(b) / len(b), sum(b) / len(b) - sum(a) / len(a)))
