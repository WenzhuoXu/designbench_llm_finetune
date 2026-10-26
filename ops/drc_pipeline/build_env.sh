#!/bin/bash
# Build the dataxp corpus venv on drc (CPU only).
# Interpreter: /home/wxu/.local/bin/python3.12 (3.12.14, uv-managed) - the local pins
# were validated on Python 3.12.7; the system python3 is 3.10.12.
# The venv's real directory is /home/wxu/corpus_env on the root filesystem (28 GB free);
# /storage is 100% full (4 GB free), so /storage/wxu/corpus/env is a symlink to it and
# every job uses /storage/wxu/corpus/env/bin/python as agreed.
# Install order follows the requirement files: numpy/scipy pins first, grid2op, photonics,
# molopt, then engibench (whose resolver may lift numpy/pandas), then re-pin numpy/scipy/
# pandas/contourpy LAST and let `pip check` name whatever still conflicts.
set -u
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/wxu/.local/bin/python3.12
REAL=/home/wxu/corpus_env
LINK=/storage/wxu/corpus/env
REQ=/storage/wxu/corpus/code/dataxp
log() { echo; echo "##### $(date '+%F %T') $*"; }

log "venv"
if [ ! -x "$REAL/bin/python" ]; then "$PY" -m venv "$REAL" || { echo "FAIL venv"; exit 2; }; fi
if [ ! -e "$LINK" ]; then ln -s "$REAL" "$LINK"; fi
ls -la "$LINK" "$LINK/bin/python"
P="$LINK/bin/python"
"$P" --version
"$P" -c "import sys; print('prefix', sys.prefix)"
"$P" -m pip install -U pip wheel setuptools

log "step 1: numpy/scipy pins"
"$P" -m pip install numpy==1.26.4 scipy==1.13.1 || echo "FAIL step1"
log "step 2: requirements_grid2op.txt"
"$P" -m pip install -r "$REQ/requirements_grid2op.txt" || echo "FAIL step2"
log "step 3: requirements_photonics.txt"
"$P" -m pip install -r "$REQ/requirements_photonics.txt" || echo "FAIL step3"
log "step 4: requirements_molopt.txt"
"$P" -m pip install -r "$REQ/requirements_molopt.txt" || echo "FAIL step4"
log "step 5: engibench packages (the file says: install, then re-pin)"
"$P" -m pip install engibench==0.2.0 cvxopt==1.3.3 mmapy==0.3.1 ceviche==0.1.3 "autograd>=1.6" "seaborn>=0.13" "threadpoolctl>=3.1" || echo "FAIL step5"
log "step 6: re-pin numpy/scipy/pandas/contourpy last"
"$P" -m pip install numpy==1.26.4 scipy==1.13.1 pandas==2.3.3 "contourpy<1.4" || echo "FAIL step6"

log "pip check"
"$P" -m pip check
log "freeze (relevant)"
"$P" -m pip freeze | grep -i -E '^(numpy|scipy|pandas|grid2op|lightsim2grid|pandapower|ceviche|ceviche-challenges|rdkit|pytdc|scikit-learn|engibench|cvxopt|mmapy|autograd|threadpoolctl|contourpy|matplotlib|unyt|seaborn|gymnasium|datasets|pyarrow)=='
log "imports"
for m in numpy scipy pandas grid2op lightsim2grid pandapower ceviche ceviche_challenges rdkit tdc sklearn engibench cvxopt mmapy autograd threadpoolctl; do
  "$P" -c "import $m; print('%-20s' % '$m', getattr($m, '__version__', '?'))" 2>&1 | tail -1
done
log "disk"
du -sh "$REAL"; df -h / /storage | tail -2
log "done"
