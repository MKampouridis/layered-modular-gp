#!/bin/bash
# Sets up the Python environment on CERES, once (docs/CERES.md). CERES asks for conda
# environments to be built on a compute node, so start an interactive session first:
#   qrsh -q interactive.q
#   cd ~/layered-modular-gp && bash scripts/ceres/setup.sh
#
# CERES runs CentOS 7 (glibc 2.17); recent pip wheels may need a newer glibc, so the
# environment comes from conda-forge (Miniforge in ~/miniforge3). Safe to run again.
set -eo pipefail
case "$(hostname)" in
    compute-*) ;;
    *) echo "Please run this on a compute node: start one with 'qrsh -q interactive.q'." >&2; exit 1 ;;
esac
unset PYTHONPATH
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PREFIX="$HOME/miniforge3"
ENV=lmgp
if [ ! -x "$PREFIX/bin/conda" ]; then
    curl -fsSL -o "$HOME/miniforge-installer.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$HOME/miniforge-installer.sh" -b -p "$PREFIX"
    rm -f "$HOME/miniforge-installer.sh"
fi
source "$PREFIX/etc/profile.d/conda.sh"
if ! conda run -n "$ENV" python -c "import numpy, scipy, pandas, pyarrow, yaml, pmlb, pytest, hypothesis" 2>/dev/null; then
    conda create -y -n "$ENV" -c conda-forge python=3.12 "numpy>=2.0,<2.6" scipy "pandas>=2.2" \
        "pyarrow>=17" pyyaml requests pytest pytest-xdist hypothesis git
    conda run -n "$ENV" pip install --no-deps pmlb
fi
conda activate "$ENV"
pip install --no-deps -e "$REPO"
cd "$REPO"
# Fetch the SRBench datasets once, here, rather than from 60 array elements at the same time.
python -c "from lmgp.bench.srbench import G1_SUBSET, load_xy; [load_xy(n) for n in G1_SUBSET]; print('pmlb data cached')"
python -c "import numpy, scipy, pandas, pyarrow; print('environment ready: numpy', numpy.__version__, 'pandas', pandas.__version__)"
source scripts/ceres/env.sh
python -m pytest -q -p no:warnings -n 4
