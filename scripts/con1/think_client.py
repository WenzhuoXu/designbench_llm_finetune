"""Thinking diagnostic, client (con1-2m-rows, 11-27; the parent session's request): held-out-family thermal2 questions
on PLACE_BLOCK and MOVE_SINK, zero-shot, the facetm question as in the existing runs, with room to think first.

    python think_client.py --queries queries_v2.jsonl --tokenizer MODEL --base-url URL --model base|LAB --label L \\
        --out preds_L.jsonl [--budget 1000] [--min-think 0] [--answer-tokens 1024] [--workers 48]

Prompt: con1's skill_client.messages_for(q, j, False, 'facetm') -> conseq_format.prompt_ids (imported unchanged; ends
'<|im_start|>assistant\\n<think>\\n', the layout con1's answers were trained and asked in). Temperature 0, in steps:
  1. think: up to --budget tokens, stopping at '</think>' or <|im_end|>. With --min-think K the first K tokens are
     generated with '</think>', <|im_end|> and <|endoftext|> banned (logit bias -100), then the rest of the budget as
     above. think_end: 'closed' (the model wrote </think>), 'budget' (cut at the budget), 'im_end' (ended its turn
     inside the think block).
  2. answer: the prompt + think.strip() + '\\n</think>\\n\\n' (the trained answer layout; a think block cut by the
     budget, or ended by <|im_end|>, is closed the same way), up to --answer-tokens, stop at <|im_end|>.
One line per question in the usual preds schema (sid, variant, group, predictor, pass='zero', qi, target, move, text =
the answer after </think>, finish) plus think (raw text), think_tokens, think_end, answer_tokens, prompt_tokens.
Resumable by (sid, qi): each line is written and flushed as soon as its question is done.
Which questions (11-27 generalisation for the all-family base run): --groups / --variants / --verbs (default: all),
--subset FILE (lines 'sid<TAB>qi': only those). --base-url takes several servers; a state's questions stay on one
(hash of the sid), as con1's skill_client does. CON1_DIR: where con1's skill_client.py / dirlayer.py are (default this
file's directory if it holds them, else drc's con1 repo).
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
CON1 = os.environ.get("CON1_DIR") or (HERE if os.path.exists(os.path.join(HERE, "skill_client.py"))
                                      else "/storage/wxu/con1/repo/scripts/con1")
sys.path.insert(0, CON1)
import skill_client as SC  # noqa: E402  (con1's client: messages_for, prompt_ids; read only)
import dirlayer as DL  # noqa: E402

LOCK = threading.Lock()


def call(base, model, ids, max_tokens, stop_ids, bias=None):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "stop_token_ids": stop_ids,
            "skip_special_tokens": False}
    if bias:
        body["logit_bias"] = bias
    req = urllib.request.Request(base.rstrip("/") + "/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                j = json.load(r)
            ch = j["choices"][0]
            return (ch.get("text") or "", ch.get("finish_reason"), ch.get("stop_reason"),
                    (j.get("usage") or {}).get("completion_tokens") or 0)
        except Exception as e:                             # noqa: BLE001
            if attempt == 4:
                raise RuntimeError(str(e)[:300])
            time.sleep(5 * (attempt + 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--base-url", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--min-think", type=int, default=0)
    ap.add_argument("--answer-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--verbs", nargs="+", default=None)
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--target", default="facetm", help="answer form (targets.FORMATS); facetm = every run before 11-28")
    ap.add_argument("--named-metrics", action="store_true",
                    help="the question names the metrics to answer: all of the state's metrics0, sorted (targets.question)")
    ap.add_argument("--user-suffix", default=None,
                    help="a sentence added as the last line of the user message ('\\n' + the sentence; 11-28 arm D-brief)")
    ap.add_argument("--defined-terms", action="store_true",
                    help="(the default since lib_v8; kept so older command lines still run)")
    ap.add_argument("--no-defined-terms", action="store_true",
                    help="opt out: the question without the definitions of feasible and 'same', as every arm before "
                         "11-28 (arm D) was asked; for arms that must stay comparable with arm D")
    ap.add_argument("--reasoning-effort", default=None, choices=("xhigh", "medium", "low"),
                    help="the chat template's reasoning_effort variable (default: unset = the template's default, "
                         "xhigh); the prompt is then rendered by apply_chat_template(..., reasoning_effort=X)")
    a = ap.parse_args(argv)
    if a.defined_terms and a.no_defined_terms:
        ap.error("--defined-terms and --no-defined-terms together")
    import targets
    targets.DEFINED_DEFAULT = not a.no_defined_terms         # read by targets.question here and in skill_client
    a.defined_terms = targets.DEFINED_DEFAULT
    from targets import question
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    IM_END, THINK_END = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("</think>")
    EOT = tok.convert_tokens_to_ids("<|endoftext|>")
    assert tok.encode("</think>", add_special_tokens=False) == [THINK_END], "'</think>' is not one token"
    ban = {str(t): -100 for t in (IM_END, THINK_END, EOT)}
    qs = [json.loads(l) for l in open(a.queries, encoding="utf-8")]
    qs = [q for q in qs if (a.groups is None or q["group"] in a.groups) and (a.variants is None or q["variant"] in a.variants)]
    sub = None
    if a.subset:
        sub = {(l.split("\t")[0], int(l.split("\t")[1])) for l in open(a.subset, encoding="utf-8") if l.strip()}
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(l)
                if not str(r.get("finish", "")).startswith("error"):
                    done.add((r["sid"], r["qi"]))
            except ValueError:
                pass
    jobs = [(q, j) for q in qs for j, m in enumerate(q["moves"])
            if (a.verbs is None or DL.verb_of(m) in a.verbs) and (sub is None or (q["sid"], j) in sub)
            and (q["sid"], j) not in done]
    print("%s: %d questions (%d done), budget %d, min think %d" % (a.label, len(jobs), len(done), a.budget, a.min_think),
          flush=True)

    def render(msgs):
        if a.reasoning_effort is None:                       # the template's default: every run before 11-28
            return SC.prompt_ids(tok, msgs)
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      reasoning_effort=a.reasoning_effort)
        ids = list(ids if isinstance(ids, list) else ids["input_ids"])
        if not tok.decode(ids[-4:]).endswith("<think>\n"):
            raise ValueError("the prompt does not end with '<think>\\n'")
        return ids

    def one(job):
        q, j = job
        if a.target == "facetm" and not (a.named_metrics or a.user_suffix):   # skill_client's own messages (earlier
            msgs = SC.messages_for(q, j, False, "facetm")                    # runs' prompt with --no-defined-terms)
        else:                                                # skill_client's zero-shot layout with this form's question
            msgs = [{"role": "system", "content": q["system"]},
                    {"role": "user", "content": q["user"] + question(
                        a.target, q["moves"][j]["move"], True, sorted(q["metrics0"]) if a.named_metrics else None)
                        + ("\n" + a.user_suffix if a.user_suffix else "")}]
        ids = render(msgs)
        url = a.base_url[int(hashlib.sha1(q["sid"].encode()).hexdigest()[:8], 16) % len(a.base_url)]
        rec = {"sid": q["sid"], "variant": q["variant"], "group": q["group"], "predictor": a.label, "pass": "zero", "qi": j,
               "target": a.target, "named_metrics": a.named_metrics, "user_suffix": a.user_suffix,
               "defined_terms": a.defined_terms, "reasoning_effort": a.reasoning_effort,
               "move": q["moves"][j]["move"],
               "prompt_tokens": len(ids)}
        try:
            think, n, end = "", 0, None
            if a.min_think > 0:
                t, fin, sr, k = call(url, a.model, ids, a.min_think, [], ban)
                think, n = t, k
            if n < a.budget:
                cur = ids + tok.encode(think, add_special_tokens=False) if think else ids
                t, fin, sr, k = call(url, a.model, cur, a.budget - n, [IM_END, THINK_END])
                think, n = think + t, n + k
                end = "budget" if fin == "length" else "closed" if sr == THINK_END else "im_end" if sr == IM_END else str(sr)
            else:
                end = "budget"
            think = think.replace("</think>", "").replace("<|im_end|>", "")
            aids = ids + tok.encode(think.strip() + "\n</think>\n\n", add_special_tokens=False)
            ans, fin, sr, k = call(url, a.model, aids, a.answer_tokens, [IM_END])
            rec.update(text=ans.replace("<|im_end|>", "").strip(), finish=fin, think=think, think_tokens=n,
                       think_end=end, answer_tokens=k)
        except RuntimeError as e:
            rec.update(text=None, finish="error: %s" % e)
        return rec

    t0, n = time.time(), 0
    if os.path.exists(a.out) and os.path.getsize(a.out) and open(a.out, "rb").read()[-1:] != b"\n":
        with open(a.out, "a", encoding="utf-8") as fo:      # a line torn by a kill: start the next on its own line
            fo.write("\n")
    with open(a.out, "a", encoding="utf-8") as fo, ThreadPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(one, jb) for jb in jobs]):
            with LOCK:
                fo.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
                fo.flush()
            n += 1
            if n % 50 == 0:
                print("%d/%d in %.0f s" % (n, len(jobs), time.time() - t0), flush=True)
    print("THINK_CLIENT_DONE %s %d in %.0f s" % (a.label, n, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
