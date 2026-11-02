# Running on CERES

CERES is the University of Essex cluster: CentOS 7 (glibc 2.17), with jobs managed by Sun Grid Engine (`qsub`). Array elements are numbered by `SGE_TASK_ID`, starting from 1. Log in with `ssh <user>@ceres.essex.ac.uk`, from campus or over the VPN, with MFA. Every node shares the home folder.

This setup follows the two sibling projects that already run on CERES. llm-primitives-stgp's `docs/CERES.md` records the cluster's rules, which regime-fusion-gp checked on 29 October 2026:

- **Queue:** `all.q`, one slot per core. There is no run-time or memory limit. Jobs may be suspended while a higher-priority queue is busy, and they resume by themselves.
- **Array limits:**
  - each array must set `-tc`, at most 60 elements at once;
  - about 250 elements may run at once across all of a user's arrays, and the other two projects already use some of that;
  - an array counts as one job, and a user may have at most 260 jobs.
- **Python:** a conda environment from Miniforge in `~/miniforge3`, already installed. It is built on a compute node (`qrsh -q interactive.q`), not on the login node.
- **Reproducibility across nodes:** CERES mixes CPU types, and numpy's CPU-specific code paths round differently on each. `scripts/ceres/env.sh` switches them off, so a given config and seed gives the same result on every node. llm-primitives-stgp tested this on every node type (its D-054).

**Michael submits all jobs.** The commands below are for him.

## 1. Code and environment (once)

On the login node:

```bash
git clone https://github.com/MKampouridis/layered-modular-gp.git ~/layered-modular-gp
```

Then on a compute node. It takes a few minutes, and ends by caching the SRBench data and running the tests:

```bash
qrsh -q interactive.q
cd ~/layered-modular-gp && bash scripts/ceres/setup.sh
exit
```

Later, `cd ~/layered-modular-gp && git pull` brings in changes. Re-run `setup.sh` only if the dependencies change.

## 2. Pilot timing (first job)

There are 40 array elements, one per (arm, task, budget) up to 1e11, each on one core.

On the login node, outside `qrsh`, because `qsub` does not work inside it:

```bash
cd ~/layered-modular-gp && git pull && mkdir -p logs
source scripts/ceres/env.sh
python scripts/ceres/pilot_timing.py --list --max-measured 1e11     # prints 40
qsub -t 1-2 -tc 2 scripts/ceres/pilot.sge                            # two test elements first
qstat -u $USER                                                       # qw = queued, r = running
cat logs/lmgp-pilot.o*                                               # check there is no Traceback
qsub -t 3-40 -tc 40 scripts/ceres/pilot.sge                          # the rest
```

When `qstat -u $USER` shows nothing left:

```bash
python scripts/ceres/pilot_timing.py --estimate-only --max-measured 1e11 --out results/pilot_ceres
```

Back on the Mac:

```bash
rsync -av <user>@ceres.essex.ac.uk:layered-modular-gp/results/pilot_ceres results/
```

## 3. A batch of runs

Any batch config under `configs/` works. Task *i* is run *i* of the config, so the numbering depends only on the config. Finished runs are skipped, so an element that failed or was stopped can simply be resubmitted.

```bash
cd ~/layered-modular-gp && git pull && mkdir -p logs && source scripts/ceres/env.sh
python scripts/ceres/batch_tasks.py status configs/<batch>.yaml --count 1    # prints the qsub line
qsub -t 1-N -tc 60 -v CONFIG=configs/<batch>.yaml,COUNT=1,OUT=results/<batch> scripts/ceres/runs.sge
```

- `COUNT=K` makes each element do K runs, for batches of short runs.
- The `status` command shows progress.
- When it is complete, collect the results on CERES and copy them back:

```bash
python scripts/run_local.py configs/<batch>.yaml --collect --out results/<batch>
rsync -av <user>@ceres.essex.ac.uk:layered-modular-gp/results/<batch> results/
```

## When something goes wrong

- **State `Eqw`:** the element failed to start. `qstat -j <job>` says why, and `qmod -c <job>` clears it once fixed.
- **Stopping:** `qdel <job>` stops a whole array; `qdel <job> -t 5-9` stops some of its elements.
- **Portability check:** before the first full campaign, re-run two or three tasks of the CERES batch on the Mac and compare their `final.parquet` files. Units, evaluations and generations must match exactly.
