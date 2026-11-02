"""Pilot timing: wall-clock per run at each budget level, and a CPU-hour
estimate for the full study.

For each (arm, task, budget) it runs one complete run to the budget (Boolean
runs do not stop at a solution, so the timing is the worst case) and records
wall time, units, evaluations and throughput (units per second). Budget
levels above --max-measured are not run; their time is extrapolated from the
throughput measured at the highest level actually run for that arm and task,
and marked as extrapolated.

  python scripts/ceres/pilot_timing.py --out results/pilot --jobs 4
  python scripts/ceres/pilot_timing.py --estimate-only --out results/pilot

On CERES the same script runs as a single job (see pilot_job.sh); timings
from a laptop are only indicative.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lmgp.run.config import population_for_budget  # noqa: E402
from lmgp.run.runner import run_config  # noqa: E402

BUDGETS = [1e8, 1e9, 1e10, 1e11, 1e12]
ARMS = ["a1", "a2"]
TASKS = {
    "comp_L2_R2": {"kind": "comp", "levels": 2, "reuse": 2, "noise": 0.01},
    "comp_L4_R4": {"kind": "comp", "levels": 4, "reuse": 4, "noise": 0.01},
    "parity6": {"kind": "parity", "n": 6},
    "mux11": {"kind": "mux", "k": 3},
    "srb_227_cpu_small": {"kind": "srbench", "name": "227_cpu_small"},
}

# Full-study run counts (research plan): 30 seeds, 9 arms, 5 budgets.
SEEDS, N_ARMS = 30, 9
STAGE1 = {  # task class -> number of tasks, and which pilot task stands in for it
    "synthetic (48 cells)": (48, ["comp_L2_R2", "comp_L4_R4"]),
    "parity 5-8": (4, ["parity6"]),
    "multiplexer 11, 20": (2, ["mux11"]),
}
STAGE2 = {"SRBench subset (assume 30 datasets)": (30, ["srb_227_cpu_small"])}


def one(arm, tname, budget):
    cfg = {"arm": arm, "task": TASKS[tname], "budget": budget, "pop_size": population_for_budget(budget),
           "max_generations": 10 ** 9, "stop_on_solution": False, "seed_index": 0}
    t = time.perf_counter()
    r = run_config(cfg)
    wall = time.perf_counter() - t
    return {"arm": arm, "task": tname, "budget": budget, "pop_size": cfg["pop_size"], "wall_s": wall,
            "units": r["units"], "evals": r["evals"], "generations": r["generations"],
            "units_per_s": r["units"] / wall, "measured": True}


def estimate(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    for (arm, task), g in df.groupby(["arm", "task"]):
        top = g.sort_values("budget").iloc[-1]
        for b in BUDGETS:
            m = g[g.budget == b]
            if len(m):
                rows.append(m.iloc[0].to_dict())
            else:
                rows.append({"arm": arm, "task": task, "budget": b, "pop_size": population_for_budget(b),
                             "wall_s": b / top.units_per_s, "units": b, "evals": None, "generations": None,
                             "units_per_s": top.units_per_s, "measured": False})
    full = pd.DataFrame(rows)
    lines = ["Hours per run (mean over pilot arms; * = extrapolated):"]
    piv = full.pivot_table(index="task", columns="budget", values="wall_s", aggfunc="mean") / 3600
    lines.append(piv.to_string(float_format=lambda v: f"{v:.3g}"))
    total = 0.0
    for stage, spec in (("Stage 1", STAGE1), ("Stage 2", STAGE2)):
        for cls, (n_tasks, stand_ins) in spec.items():
            per_budget = full[full.task.isin(stand_ins)].groupby("budget").wall_s.mean() / 3600
            h = float(per_budget.sum()) * n_tasks * N_ARMS * SEEDS
            total += h
            top = float(per_budget.iloc[-1]) * n_tasks * N_ARMS * SEEDS
            lines.append(f"{stage} {cls}: {h:,.0f} CPU-h (top budget {top:,.0f})")
    lines.append(f"TOTAL Stage 1+2 at {len(BUDGETS)} budgets {BUDGETS[0]:.0e}-{BUDGETS[-1]:.0e}: {total:,.0f} CPU-h")
    lines.append("Dropping the top budget: "
                 f"{total - sum(float(full[full.task.isin(s)].groupby('budget').wall_s.mean().iloc[-1]) / 3600 * n * N_ARMS * SEEDS for spec in (STAGE1, STAGE2) for _, (n, s) in spec.items()):,.0f} CPU-h")
    return full, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pilot")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-measured", type=float, default=1e10)
    ap.add_argument("--estimate-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "pilot_measured.parquet"
    if not a.estimate_only:
        jobs = [(arm, t, b) for b in BUDGETS if b <= a.max_measured for arm in ARMS for t in TASKS]
        done = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        if len(done):
            have = set(zip(done.arm, done.task, done.budget))
            jobs = [j for j in jobs if j not in have]
        # Largest first so the long runs start early.
        jobs.sort(key=lambda j: -j[2])
        rows = [] if not len(done) else done.to_dict("records")
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(one, *zip(*jobs)) if jobs else []:
                rows.append(r)
                print(r, flush=True)
                pd.DataFrame(rows).to_parquet(f, index=False)
    df = pd.read_parquet(f)
    full, text = estimate(df)
    full.to_csv(out / "pilot_estimate.csv", index=False)
    (out / "pilot_estimate.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
