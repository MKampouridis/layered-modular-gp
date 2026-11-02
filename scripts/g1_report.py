"""Summarise the G1 known-result checks.

  python scripts/g1_report.py --boolean results/g1/boolean_final.parquet \
      --srbench results/g1/srbench_final.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmgp.analysis.effort import bootstrap_effort  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def boolean_report(path):
    df = pd.read_parquet(path)
    rows = []
    for (task, arm), g in df.groupby(["task", "arm"]):
        uts = [u if s else None for u, s in zip(g.units_to_solution, g.solved)]
        ind = [(gs + 1) * p if s else None for gs, p, s in zip(g.gen_to_solution, g.pop_size, g.solved)]
        e_u, lo_u, hi_u, _ = bootstrap_effort(uts)
        e_i, lo_i, hi_i, _ = bootstrap_effort(ind)
        rows.append({
            "task": task, "arm": arm, "runs": len(g), "solved": int(g.solved.sum()),
            "success_rate": g.solved.mean(),
            "median_units_to_solution": np.nanmedian([u for u in uts if u is not None] or [np.nan]),
            "effort_units": e_u, "effort_units_ci": f"[{lo_u:.3g}, {hi_u:.3g}]",
            "effort_individuals": e_i, "effort_individuals_ci": f"[{lo_i:.3g}, {hi_i:.3g}]",
            "median_solution_size": g[g.solved].model_size.median(),
            "median_wall_s": g.wall.median(),
        })
    return pd.DataFrame(rows)


def srbench_report(path):
    df = pd.read_parquet(path)
    df["dataset"] = df.task.str.replace("srbench_", "", regex=False)
    ours = df.groupby("dataset").agg(runs=("test_r2", "size"), lmgp_A1_median_r2=("test_r2", "median"),
                                     lmgp_A1_iqr=("test_r2", lambda s: s.quantile(.75) - s.quantile(.25)),
                                     median_size=("model_size", "median"), median_wall_s=("wall", "median"))
    pub = pd.read_feather(ROOT / "data" / "srbench" / "black-box_results.feather")
    pub = pub[pub.dataset.isin(ours.index)]
    piv = pub.pivot_table(index="dataset", columns="algorithm", values="r2_test", aggfunc="median")
    cols = [c for c in ["gplearn", "EPLEX", "AFP", "Operon", "Linear"] if c in piv]
    return ours.join(piv[cols].add_prefix("SRBench_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boolean")
    ap.add_argument("--srbench")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.boolean:
        print(boolean_report(a.boolean).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if a.srbench:
        print(srbench_report(a.srbench).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
