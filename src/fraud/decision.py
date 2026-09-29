import numpy as np
import pandas as pd


def cost_threshold(ratio: float) -> float:
    """Alert when p * L >= c, i.e. p >= c / L = 1 / R (Bayes rule with calibrated probabilities)."""
    return 1.0 / ratio


def fpr_threshold(scores, y, fpr: float) -> float:
    """Smallest threshold t such that the share of negatives with score >= t does not exceed `fpr`."""
    neg = np.sort(np.asarray(scores)[np.asarray(y) == 0])
    k = int(np.floor(fpr * len(neg)))
    if k == 0:
        return float(np.nextafter(neg[-1], np.inf))
    cand = neg[-k]
    return float(cand if (neg >= cand).sum() <= k else np.nextafter(cand, np.inf))


def confusion_at(y, alerts) -> dict:
    y, alerts = np.asarray(y).astype(bool), np.asarray(alerts).astype(bool)
    tp, fp = int((alerts & y).sum()), int((alerts & ~y).sum())
    fn, tn = int((~alerts & y).sum()), int((~alerts & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "alert_rate": (tp + fp) / len(y)}


def cost_per_1000(y, alerts, ratio: float) -> float:
    """Expected cost per 1,000 applications, in units of one review: R x missed frauds + reviews."""
    c = confusion_at(y, alerts)
    return 1000 * (ratio * c["fn"] + c["tp"] + c["fp"]) / len(np.asarray(y))


def reference_costs(y, ratio: float) -> dict:
    y = np.asarray(y)
    return {"review_none": 1000 * ratio * y.mean(), "review_all": 1000.0}


def lift_table(scores, y, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(-np.asarray(scores), kind="stable")
    y_sorted = np.asarray(y)[order]
    bins = np.array_split(y_sorted, n_bins)
    total, base = y_sorted.sum(), y_sorted.mean()
    rows, cum = [], 0
    for i, b in enumerate(bins):
        cum += b.sum()
        rows.append({"decile": i + 1, "n": len(b), "fraud_rate": b.mean(), "capture": cum / total,
                     "lift": b.mean() / base})
    return pd.DataFrame(rows)


def capture_at(scores, y, top: float) -> float:
    order = np.argsort(-np.asarray(scores), kind="stable")
    k = int(np.rint(top * len(order)))
    y = np.asarray(y)
    return float(y[order[:k]].sum() / y.sum())
