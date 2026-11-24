"""A scripted stand-in for the vLLM chat server, for plumbing tests of the harness's multi-message turns (v3rc
--checks). No model: the reply is chosen by the variant named in the system prompt and by how many user messages
the conversation has so far; every request is appended to --log as one JSON line (the messages the model would see).

Script (per variant, user message n = 1, 2, 3, ...): reply n is SCRIPT[variant][min(n, len) - 1]. The defaults check
one unreadable move, one legal move, then commit; a turn with --checks 1 therefore exercises the budget rule.

usage: python fake_chat_server.py --port 8799 --log requests.jsonl
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = {
    "grid2op": ["<check>NOT_A_VERB(1)</check>", "<check>DO_NOTHING()</check>", "<action>DO_NOTHING()</action>"],
    "engibench": ["<check>NOT_A_VERB(1)</check>", "<check>ERODE(0, 1)</check>", "<action>DILATE(0, 1)</action>"],
    "photonics": ["<check>NOT_A_VERB(1)</check>", "<check>FILL_RECT(3, 18, 8, 20)</check>",
                  "<action>FILL_RECT(3, 18, 8, 20)</action>"],
}
LOCK = threading.Lock()


def variant_of(system):
    s = system.lower()
    if "grid2op" in s:
        return "grid2op"
    if "photonic" in s or "waveguide" in s or "wavelength" in s:
        return "photonics"
    return "engibench"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"data": [{"id": "fake"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        v = variant_of(msgs[0]["content"])
        n = sum(1 for m in msgs if m["role"] == "user")
        steps = SCRIPT[v]
        content = "Scripted reply %d for %s.\n%s" % (n, v, steps[min(n, len(steps)) - 1])
        if ARGS.actions_only:              # every tag an <action> (tests of --check-actions)
            content = content.replace("<check>", "<action>").replace("</check>", "</action>")
        with LOCK, open(ARGS.log, "a", encoding="utf-8") as fo:
            fo.write(json.dumps({"variant": v, "n_user": n, "messages": msgs, "reply": content}) + "\n")
        self._send({"choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content,
                                             "reasoning_content": "scripted reasoning"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8799)
ap.add_argument("--log", required=True)
ap.add_argument("--actions-only", action="store_true", help="write every scripted tag as <action>")
ARGS = ap.parse_args()
ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()
