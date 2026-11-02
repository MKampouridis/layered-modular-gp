"""Run every run of a batch config locally in parallel (development and G1
validation only; large runs go to CERES as array jobs).

  python scripts/run_local.py configs/g1/boolean.yaml --out results/g1 --jobs 8
  python scripts/run_local.py configs/g1/boolean.yaml --collect --out results/g1

Runs whose final.parquet already exists are skipped, so the script can be
re-run after an interruption. --collect concatenates all final.parquet files
under --out into <out>/<batch-name>_final.parquet.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmgp.run.config import expand_batch, load_yaml, run_id  # noqa: E402
from lmgp.run.runner import run_config  # noqa: E402


def _one(cfg, out):
    r = run_config(cfg, out)
    return r["run_id"], r.get("solved"), r.get("test_r2"), r["wall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    name = Path(a.config).stem
    runs = expand_batch(load_yaml(a.config))
    out = Path(a.out)
    if not a.collect:
        todo = [c for c in runs if not (out / run_id(c) / "final.parquet").exists()]
        print(f"{len(runs)} runs, {len(todo)} to do", flush=True)
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = [ex.submit(_one, c, out) for c in todo]
            for i, f in enumerate(as_completed(futs), 1):
                print(i, *f.result(), flush=True)
    frames = [pd.read_parquet(out / run_id(c) / "final.parquet") for c in runs
              if (out / run_id(c) / "final.parquet").exists()]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out / f"{name}_final.parquet", index=False)
    print(f"collected {len(df)} runs -> {out / (name + '_final.parquet')}")


if __name__ == "__main__":
    main()
