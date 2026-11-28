"""Before a new-answer-form arm (11-28; arm D-unclear): the new arm's prompts against arm D's, per question (CPU,
tokenizer only).

    python form_render_check.py --queries queries_v3.jsonl --tokenizer MODEL --subset subset_300.tsv --form dirunc \\
        --ref-targets LIB_V6_DIR --ref-form dir --ref-preds preds_base_v3_dir_8k.jsonl --insert TEXT

The new prompt is rendered as think_client.py --no-defined-terms --target FORM --named-metrics renders it (this
directory's targets.question, the chat template's defaults); arm D's with the reference directory's targets.question
in REF_FORM. Checks, per question:
  ref tokens     arm D's prompt rendered here has exactly the prompt_tokens arm D recorded
  one insertion  the decoded new prompt equals arm D's decoded prompt with TEXT inserted once, and nothing else differs
Prints the token-count change. Exit 1 on any failure.
"""
import argparse
import collections
import difflib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("CON1_DIR") or HERE)
import skill_client as SC  # noqa: E402
import targets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--form", required=True)
    ap.add_argument("--ref-targets", required=True)
    ap.add_argument("--ref-form", required=True)
    ap.add_argument("--ref-preds", required=True)
    ap.add_argument("--insert", required=True)
    a = ap.parse_args()
    targets.DEFINED_DEFAULT = False
    spec = importlib.util.spec_from_file_location("targets_ref", os.path.join(a.ref_targets, "targets.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    if hasattr(ref, "DEFINED_DEFAULT"):
        ref.DEFINED_DEFAULT = False
    ref_pt = {}
    for l in open(a.ref_preds, encoding="utf-8"):
        r = json.loads(l)
        ref_pt[(r["sid"], r["qi"])] = r.get("prompt_tokens")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sub = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(a.subset) if l.strip()}
    c = collections.Counter()
    dt = collections.Counter()
    first = None
    for l in open(a.queries, encoding="utf-8"):
        q = json.loads(l)
        for j, m in enumerate(q["moves"]):
            if (q["sid"], j) not in sub:
                continue
            mets = sorted(q["metrics0"])
            new_msgs = [{"role": "system", "content": q["system"]},
                        {"role": "user", "content": q["user"] + targets.question(a.form, m["move"], True, mets)}]
            old_msgs = [{"role": "system", "content": q["system"]},
                        {"role": "user", "content": q["user"] + ref.question(a.ref_form, m["move"], True, mets)}]
            ni, oi = SC.prompt_ids(tok, new_msgs), SC.prompt_ids(tok, old_msgs)
            c["prompts"] += 1
            ok_n = ref_pt.get((q["sid"], j)) == len(oi)
            c["ref tokens == arm D's prompt_tokens"] += ok_n
            nt, ot = tok.decode(ni), tok.decode(oi)
            ops = [(t, ot[i1:i2], nt[j1:j2]) for t, i1, i2, j1, j2 in
                   difflib.SequenceMatcher(None, ot, nt, autojunk=False).get_opcodes() if t != "equal"]
            ok_i = nt.count(a.insert) == 1 and nt.replace(a.insert, "", 1) == ot
            c["new == arm D + one insertion"] += ok_i
            dt[len(ni) - len(oi)] += 1
            if ok_n and ok_i:
                c["all checks"] += 1
            elif first is None:
                first = (q["sid"], j, ops[:3])
    print(dict(c))
    print("token change (new - arm D):", dict(dt))
    print("first failure:", first)
    bad = c["prompts"] - c["all checks"]
    print("FORM_RENDER_CHECK %s (%d of %d prompts failed)" % ("OK" if not bad else "FAILED", bad, c["prompts"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
