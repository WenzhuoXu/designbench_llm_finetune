"""run_env.py <dir> [NAME=VALUE ...] -- command ...: exec the command in the launcher's environment (orig.environ in <dir>,
saved from its sampler) and working directory, with the given variables added."""
import os
import sys

d = sys.argv[1]
cut = sys.argv.index("--")
env = {}
for kv in open(os.path.join(d, "orig.environ"), "rb").read().split(b"\0"):
    if b"=" in kv:
        k, v = kv.split(b"=", 1)
        env[k.decode()] = v.decode()
for kv in sys.argv[2:cut]:
    k, v = kv.split("=", 1)
    env[k] = v
cmd = sys.argv[cut + 1:]
os.chdir(open(os.path.join(d, "orig.cwd")).read().strip())
os.execve(cmd[0], cmd, env)
