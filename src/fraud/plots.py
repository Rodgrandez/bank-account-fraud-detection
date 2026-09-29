from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from fraud import config


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def performance(r, arrays, folder) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    y = arrays["y_f"]
    for name, label in (("xgb", "XGBoost"), ("lr", "Logistic regression")):
        p, rc, _ = precision_recall_curve(y, arrays[name]["s_f"])
        ax.plot(rc, p, label=f"{label} (PR-AUC {r['future'][name]['pr_auc']['value']:.3f})")
        rec = r["future"][name]["recall_budget"]["value"]
        ax.axvline(rec, ls=":", lw=1, color=ax.lines[-1].get_color())
    ax.set(xlabel="Recall", ylabel="Precision", title="Months 6-7: precision-recall (dotted: recall at 5% budget)")
    ax.legend(frameon=False)
    return _save(fig, Path(folder) / "performance.png")


def calibration(r, arrays, folder) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 5))
    y = arrays["y_f"]
    for key, label in (("s_f", "XGBoost raw"), ("p_f", "XGBoost isotonic")):
        frac, mean = calibration_curve(y, arrays["xgb"][key], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=label)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color="grey", ls="--", lw=1)
    ax.set(xlabel="Predicted probability", ylabel="Observed fraud rate", title="Calibration, months 6-7")
    ax.legend(frameon=False)
    return _save(fig, Path(folder) / "calibration.png")


def lift_gain(r, arrays, folder) -> Path:
    import pandas as pd
    lt = pd.DataFrame(r["lift"]["deciles"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(lt["decile"], lt["lift"])
    a1.set(xlabel="Score decile (1 = highest)", ylabel="Lift", title="Lift by decile")
    a2.plot(np.r_[0, lt["decile"] * 10], np.r_[0, lt["capture"] * 100], marker="o", label="Model")
    a2.plot([0, 100], [0, 100], color="grey", ls="--", label="Random")
    a2.set(xlabel="% of applications reviewed", ylabel="% of fraud captured", title="Cumulative gain")
    a2.legend(frameon=False)
    return _save(fig, Path(folder) / "lift_gain.png")


def cost_sensitivity(r, arrays, folder) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ratios = [int(k) for k in r["cost"]]
    model = [r["cost"][str(k)]["cost_per_1000"]["value"] for k in ratios]
    none = [r["cost"][str(k)]["review_none"] for k in ratios]
    ax.plot(ratios, model, marker="o", label="Model (threshold 1/R)")
    ax.plot(ratios, none, marker="s", label="Review nothing")
    ax.axhline(1000, color="grey", ls="--", label="Review everything")
    ax.set(xlabel="R = loss per missed fraud / cost of one review", ylabel="Cost per 1,000 applications (reviews)",
           title="Expected cost, months 6-7")
    ax.legend(frameon=False)
    return _save(fig, Path(folder) / "cost_sensitivity.png")


def fairness_plot(r, arrays, folder) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    groups = list(r["fairness"]["before"]["rates"])
    x = np.arange(len(groups))
    for j, (key, label) in enumerate((("before", "Single threshold"), ("after", "Group thresholds"))):
        vals = [r["fairness"][key]["rates"][g]["fpr"] for g in groups]
        ax.bar(x + (j - 0.5) * 0.38, vals, width=0.38, label=label)
    ax.axhline(config.FPR_BUDGET, color="grey", ls="--", lw=1)
    ax.set_xticks(x, groups)
    ax.set(ylabel="False-positive rate", title="False-positive rate by age group, months 6-7")
    ax.legend(frameon=False)
    return _save(fig, Path(folder) / "fairness.png")


def drift_plot(r, arrays, folder) -> Path:
    import pandas as pd
    d = pd.DataFrame(r["drift"]["by_month"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, title in zip(axes, ("fraud_rate", "psi_score", "recall"),
                              ("Fraud rate", "Score PSI vs months 0-4", "Recall at 5% budget (months 0-4 in-sample)"),
                              strict=True):
        ax.plot(d["month"], d[col], marker="o")
        ax.axvspan(5.5, 7.5, color="grey", alpha=0.15)
        ax.set(xlabel="Month", title=title)
    axes[1].axhline(config.PSI_ALERT, color="red", ls="--", lw=1)
    return _save(fig, Path(folder) / "drift.png")


def all_figures(r, arrays, folder) -> list[Path]:
    return [f(r, arrays, folder) for f in (performance, calibration, lift_gain, cost_sensitivity, fairness_plot,
                                           drift_plot)]
