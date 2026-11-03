"""Run one configuration and print progress as it goes (diagnosing slow runs).

  .venv/bin/python scripts/trace_run.py --arm a2 --task '{"kind": "parity", "n": 6}' --budget 1e10

Prints, every 10 generations or 20 seconds: elapsed time, units, evaluations,
best fitness and the population's expanded size.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmgp.run import runner  # noqa: E402
from lmgp.run.config import population_for_budget  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="a2")
    ap.add_argument("--task", default='{"kind": "parity", "n": 6}')
    ap.add_argument("--budget", type=float, default=1e10)
    ap.add_argument("--seed-index", type=int, default=0)
    a = ap.parse_args()
    orig, t0, last = runner.Run._log, time.perf_counter(), [time.perf_counter()]

    def _log(self, log, gen, pop, fits, scal, counter, S, partial=False):
        orig(self, log, gen, pop, fits, scal, counter, S, partial)
        now = time.perf_counter()
        if gen % 10 == 0 or now - last[0] > 20:
            es = [p.expanded_size() for p in pop]
            print(f"gen {gen} t={now - t0:.0f}s units={counter.units:.3g} evals={counter.evals} "
                  f"best={min(fits):.4g} expanded mean={np.mean(es):.0f} max={max(es)}", flush=True)
            last[0] = now

    runner.Run._log = _log
    cfg = {"arm": a.arm, "task": json.loads(a.task), "budget": a.budget,
           "pop_size": population_for_budget(a.budget), "max_generations": 10 ** 9,
           "stop_on_solution": False, "seed_index": a.seed_index}
    r = runner.run_config(cfg)
    print(f"done in {time.perf_counter() - t0:.0f}s: units {r['units']:.3g}, evals {r['evals']}, "
          f"generations {r['generations']}")


if __name__ == "__main__":
    main()
