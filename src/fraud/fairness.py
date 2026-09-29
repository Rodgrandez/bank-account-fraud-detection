import numpy as np
import pandas as pd

from fraud import config
from fraud.decision import confusion_at, fpr_threshold

LABELS = {True: f"age>={config.AGE_CUTOFF}", False: f"age<{config.AGE_CUTOFF}"}


def older(age) -> np.ndarray:
    return np.asarray(age) >= config.AGE_CUTOFF


def group_rates(y, alerts, group) -> pd.DataFrame:
    y, alerts, group = np.asarray(y), np.asarray(alerts, bool), np.asarray(group, bool)
    rows = {}
    for g in (True, False):
        m = group == g
        c = confusion_at(y[m], alerts[m])
        rows[LABELS[g]] = {"n": int(m.sum()), "n_fraud": int(y[m].sum()), "fpr": c["fpr"], "recall": c["recall"]}
    return pd.DataFrame.from_dict(rows, orient="index")


def fpr_ratio(y, alerts, group) -> float:
    r = group_rates(y, alerts, group)
    num, den = r.loc[LABELS[True], "fpr"], r.loc[LABELS[False], "fpr"]
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0 else float("nan")


def group_thresholds(scores, y, group, fpr: float) -> dict:
    scores, y, group = np.asarray(scores), np.asarray(y), np.asarray(group, bool)
    return {g: fpr_threshold(scores[group == g], y[group == g], fpr) for g in (True, False)}


def apply_group_thresholds(scores, group, th: dict) -> np.ndarray:
    scores, group = np.asarray(scores), np.asarray(group, bool)
    return np.where(group, scores >= th[True], scores >= th[False])
