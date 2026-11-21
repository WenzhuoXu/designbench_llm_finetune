"""swap_run.py <dir> <workers> [states|""] [extra sampler arguments ...]: exec the saved sampler command (orig.cmdline, orig.environ, orig.cwd in <dir>)
with another --workers and, if given, another --states and the extra arguments at the end. Nothing else of the command
or its environment changes."""
import os
import sys

d, n = sys.argv[1], int(sys.argv[2])
states = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
argv = open(os.path.join(d, "orig.cmdline"), "rb").read().split(b"\0")
if argv and argv[-1] == b"":
    argv.pop()
argv = [x.decode() for x in argv]
env = {}
for kv in open(os.path.join(d, "orig.environ"), "rb").read().split(b"\0"):
    if b"=" in kv:
        k, v = kv.split(b"=", 1)
        env[k.decode()] = v.decode()


def put(flag, value):
    assert argv.count(flag) == 1, (flag, argv.count(flag))
    argv[argv.index(flag) + 1] = value


assert n >= 1
put("--workers", str(n))
if states:
    assert os.path.getsize(states) > 0
    put("--states", states)
argv.extend(sys.argv[4:])
os.chdir(open(os.path.join(d, "orig.cwd")).read().strip())
print("=== tuned sampler: " + " ".join(argv), flush=True)
os.execve(argv[0], argv, env)
