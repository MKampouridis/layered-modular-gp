"""Runs of a batch config as numbered tasks, for SGE job arrays on CERES.

  python scripts/ceres/batch_tasks.py status CONFIG [--count K] [--out DIR]
  python scripts/ceres/batch_tasks.py run CONFIG --index I [--count K] [--out DIR]

Task i is run i of `expand_batch(CONFIG)`, so the numbering is fixed by the
config alone. `run` executes tasks I .. I+K-1, skipping any whose
final.parquet already exists (written last and atomically), and any that
another live job is running (its RUNNING lock was touched in the last 30
minutes), so an element can be resubmitted safely.
`status` prints how many tasks are done and the matching qsub line.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lmgp.run.config import expand_batch, load_yaml, run_id  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["status", "run"])
    ap.add_argument("config")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--out", default=None, help="default: results/<config name>")
    a = ap.parse_args(argv)
    runs = expand_batch(load_yaml(a.config))
    out = Path(a.out or f"results/{Path(a.config).stem}")
    done = [(out / run_id(c) / "final.parquet").exists() for c in runs]
    if a.mode == "status":
        n_el = math.ceil(len(runs) / a.count)
        print(f"{len(runs)} tasks, {sum(done)} done, {len(runs) - sum(done)} to do; "
              f"{n_el} array elements at COUNT={a.count}")
        print(f"qsub -t 1-{n_el} -tc 60 -v CONFIG={a.config},COUNT={a.count},OUT={out} scripts/ceres/runs.sge")
        return 0
    from lmgp.run.runner import RunLocked, run_config

    for i in range(a.index, min(a.index + a.count, len(runs))):
        if done[i]:
            print(f"task {i}: {run_id(runs[i])} already done", flush=True)
            continue
        try:
            r = run_config(runs[i], out)
        except RunLocked as e:
            print(f"task {i}: skipped, {e}", flush=True)
            continue
        print(json.dumps({"task": i, **{k: r.get(k) for k in ("run_id", "units", "evals", "wall", "solved",
                                                             "test_r2")}}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
