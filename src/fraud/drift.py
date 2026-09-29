import numpy as np
import pandas as pd

EPS = 1e-6


def _psi_from_shares(e: np.ndarray, a: np.ndarray) -> float:
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected, actual, bins: int = 10) -> float:
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    if len(np.unique(e[~np.isnan(e)])) <= bins:
        # low-cardinality numerics (e.g. 0/1 flags): decile edges can collapse to a single bin, so bin by value
        return psi_categorical(e, a)
    edges = np.unique(np.nanquantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(x):
        nan = np.isnan(x)
        counts = np.histogram(x[~nan], bins=edges)[0]
        return np.r_[counts, nan.sum()] / len(x)

    return _psi_from_shares(shares(e), shares(a))


def psi_categorical(expected, actual) -> float:
    def shares(x):
        s = pd.Series(x, dtype="object")
        return s.where(s.notna(), "<NA>").value_counts(normalize=True)

    e, a = shares(expected), shares(actual)
    cats = list(dict.fromkeys([*e.index, *a.index]))
    return _psi_from_shares(e.reindex(cats, fill_value=0).to_numpy(float), a.reindex(cats, fill_value=0).to_numpy(float))
