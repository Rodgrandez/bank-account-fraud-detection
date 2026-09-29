import numpy as np

from fraud import config


def interval(stat, n: int, n_boot: int | None = None, seed: int | None = None) -> dict:
    """Percentile bootstrap (95%). Replicates where the statistic is undefined (NaN) are left out."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    reps = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot or config.N_BOOT)], dtype=float)
    reps = reps[np.isfinite(reps)]
    lo, hi = (np.percentile(reps, [2.5, 97.5]) if len(reps) else (np.nan, np.nan))
    return {"value": float(stat(np.arange(n))), "lo": float(lo), "hi": float(hi)}
