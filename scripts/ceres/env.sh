# Sourced by every CERES job script: activates the environment and fixes the
# settings that reproducibility depends on (docs/CERES.md).
unset PYTHONPATH   # CERES sets one for its own Pythons; it would load their packages
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lmgp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
# CERES mixes CPU types, and numpy picks CPU-specific code (e.g. AVX-512) whose
# rounding differs between them. Switch off every such path this CPU has, so a
# run gives the same result on every node (found by llm-primitives-stgp, D-054).
NPY_DISABLE_CPU_FEATURES="$(python -c "from numpy._core._multiarray_umath import __cpu_dispatch__ as d, __cpu_features__ as f; print(' '.join(x for x in d if f.get(x)))")"
if [ -n "$NPY_DISABLE_CPU_FEATURES" ]; then export NPY_DISABLE_CPU_FEATURES; else unset NPY_DISABLE_CPU_FEATURES; fi
