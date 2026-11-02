"""Seeding. Every run draws all randomness from one Generator built from its
seed; no module uses global random state."""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEEDS_FILE = Path(__file__).resolve().parents[2] / "configs" / "seeds.txt"


def load_seeds(path: Path | str = SEEDS_FILE) -> list[int]:
    return [int(s) for s in Path(path).read_text().split()]


def seed_for(index: int, path: Path | str = SEEDS_FILE) -> int:
    return load_seeds(path)[index]


def make_rng(seed: int, *stream: int) -> np.random.Generator:
    """Generator for `seed`, optionally a named sub-stream (e.g. data
    generation versus evolution) so that changing one does not shift the
    other."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(stream)))
