"""dx_eval_v3.sh = dx_eval.sh with CODE_OVERRIDE, the evaluation script, the instruction file, the
problems file and the client python as parameters, defaulting to the v3 harness when EP_VARIANTS
names a native-verb variant and to the original paths otherwise. Every replacement is asserted to
match exactly once, so a drifted dx_eval.sh fails loudly instead of producing a half-patched copy."""
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()


def rep(old, new, count=1):
    global text
    n = text.count(old)
    assert n == count, "expected %d match(es) of %r, found %d" % (count, old[:60], n)
    text = text.replace(old, new)


rep("#!/bin/bash\n# DX-20 evaluation driver:",
    "#!/bin/bash\n"
    "# dx_eval_v3.sh (2026-10-23): dx_eval.sh with the evaluation script, CODE_OVERRIDE, INSTR_FILE,\n"
    "# EP_PROBLEMS and the client python as parameters. When EP_VARIANTS names a native-verb variant\n"
    "# (grid2op photonics molopt engibench) they default to the v3 harness under /storage/wxu/corpus/harness\n"
    "# (dataxp_policy_eval_v3.py, dataxp_code_v3, problems_v3_native.jsonl, the corpus env with rdkit /\n"
    "# grid2op / ceviche); otherwise to the original paths, so an old-variant run is the run dx_eval.sh makes.\n"
    "#   EVAL_SCRIPT=<path>  CODE_OVERRIDE=<dataxp package dir>  INSTR_FILE=<txt>  EP_PROBLEMS=<jsonl>\n"
    "#   CLIENT_PY_OVERRIDE=<python>  EP_NATIVE_PROMPT=served|probe (v3 script only)  REPO=<collaborator repo>\n"
    "# The native task instruction is built by the v3 script itself (dataxp.ownrows.common.instruction_for);\n"
    "# INSTR_FILE and EP_PARAMLINE are passed through and ignored for native variants.\n"
    "#\n# DX-20 evaluation driver:")
rep('REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n',
    '# v3 lives outside the collaborator\'s repo: name it explicitly (the states/probe parts and the F3\n'
    '# instruction default are resolved against it)\n'
    'REPO="${REPO:-/storage/wxu/sftprobe/repo}"\n')
rep('CODE="${CODE_OVERRIDE:-$CODE}"; RUN1_FULL="${DEDUP_DIR:-$RUN1_FULL}"\n',
    'RUN1_FULL="${DEDUP_DIR:-$RUN1_FULL}"\n'
    '# v3 parameters (see the header). NATIVE=1 when EP_VARIANTS names a native-verb variant.\n'
    'V3="${V3:-/storage/wxu/corpus/harness}"\n'
    'NATIVE=0; for v in ${EP_VARIANTS:-}; do case " grid2op photonics molopt engibench " in *" $v "*) NATIVE=1;; esac; done\n'
    'if [ "$NATIVE" = 1 ]; then\n'
    '  EVAL_SCRIPT="${EVAL_SCRIPT:-$V3/scripts/dataxp_policy_eval_v3.py}"\n'
    '  CODE="${CODE_OVERRIDE:-$V3/dataxp_code_v3}"\n'
    '  EP_PROBLEMS="${EP_PROBLEMS:-$V3/problems_v3_native.jsonl}"\n'
    '  CLIENT_PY="${CLIENT_PY_OVERRIDE:-/storage/wxu/corpus/env/bin/python}"\n'
    '  INSTR_DEFAULT="$REPO/configs/dataxp/task_instruction_V2.txt"   # passed through; ignored for native variants\n'
    'else\n'
    '  EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO/scripts/dataxp_policy_eval.py}"\n'
    '  CODE="${CODE_OVERRIDE:-$CODE}"\n'
    '  CLIENT_PY="${CLIENT_PY_OVERRIDE:-$CLIENT_PY}"\n'
    '  INSTR_DEFAULT="$REPO/configs/dataxp/task_instruction_F3.txt"\n'
    'fi\n'
    '[ -f "$EVAL_SCRIPT" ] || { echo "ABORT: no evaluation script at $EVAL_SCRIPT"; exit 2; }\n'
    '[ -d "$CODE/dataxp" ] || { echo "ABORT: no dataxp package under $CODE"; exit 2; }\n')
rep('INSTR="${INSTR_FILE:-$REPO/configs/dataxp/task_instruction_F3.txt}"\n',
    'INSTR="${INSTR_FILE:-$INSTR_DEFAULT}"\n')
rep('echo "=== code=$CODE dedup=$RUN1_FULL instr=$INSTR',
    'echo "=== script=$EVAL_SCRIPT client=$CLIENT_PY native=$NATIVE"\n'
    'echo "=== code=$CODE dedup=$RUN1_FULL instr=$INSTR')
rep("run scripts/dataxp_policy_eval.py ", 'run "$EVAL_SCRIPT" ', count=4)
rep("${EP_PARAMLINE:+--param-line} ${EP_GUARD:+--loop-guard} \\\n",
    "${EP_PARAMLINE:+--param-line} ${EP_GUARD:+--loop-guard} ${EP_NATIVE_PROMPT:+--native-prompt $EP_NATIVE_PROMPT} \\\n")
open(dst, "w", encoding="utf-8", newline="\n").write(text)
print("wrote", dst, len(text), "chars")
