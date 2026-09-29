import json
import re
from pathlib import Path

import numpy as np

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def write_results(results: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=_default), encoding="utf-8")
    return path


def _ci(d: dict, digits: int = 3) -> str:
    return f"{d['value']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def results_markdown(r: dict) -> str:
    x, lr = r["future"]["xgb"], r["future"]["lr"]
    c50 = r["cost"]["50"]
    fb, fa = r["fairness"]["before"]["fpr_ratio"], r["fairness"]["after"]["fpr_ratio"]
    cc = r["fairness"]["cost_of_correction"]
    lines = [
        ("Evaluated once on the unseen months 6-7 with everything fixed on months 0-5. "
        "95% bootstrap intervals in brackets."),
        "",
        "| Metric (months 6-7) | XGBoost | Logistic regression |",
        "|---|---|---|",
        (f"| Recall at the 5% false-positive budget (threshold fixed on month 5) | {_ci(x['recall_budget'])} "
        f"| {_ci(lr['recall_budget'])} |"),
        f"| Realised false-positive rate | {_ci(x['fpr_realized'])} | {_ci(lr['fpr_realized'])} |",
        f"| PR-AUC | {_ci(x['pr_auc'])} | {_ci(lr['pr_auc'])} |",
        f"| Brier score (isotonic) | {x['brier']['isotonic']:.4f} | {lr['brier']['isotonic']:.4f} |",
        (f"| Recall at 5% FPR, threshold re-set on months 6-7 (paper-comparable) | {x['recall_oracle_5fpr']:.3f} "
        f"| {lr['recall_oracle_5fpr']:.3f} |"),
        "",
        (f"- **Model comparison:** XGBoost minus logistic regression recall "
        f"{_ci(r['comparison']['xgb_minus_lr_recall'])} "
        f"({'significant' if r['comparison']['significant'] else 'not significant'})."),
        (f"- **Cost (loss per missed fraud = 50 reviews):** {c50['cost_per_1000']['value']:.1f} review-units per "
        f"1,000 applications with the model, vs {c50['review_none']:.1f} reviewing nothing and "
        f"{c50['review_all']:.0f} reviewing everything."),
        (f"- **Fairness:** false-positive-rate ratio (age>=50 / <50) {_ci(fb, 2)} before and {_ci(fa, 2)} after "
        f"group thresholds; recall {cc['recall_before']:.3f} -> {cc['recall_after']:.3f}."),
        f"- **Monitoring alarms in months 6-7:** {len(r['drift']['monitoring'])}.",
        f"- **Recommendation to the risk committee: {r['recommendation']['decision']}.**",
    ]
    return "\n".join(lines)


def update_readme(readme, results: dict) -> None:
    text = Path(readme).read_text(encoding="utf-8")
    block = f"{START}\n{results_markdown(results)}\n{END}"
    Path(readme).write_text(re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text,
                                   flags=re.DOTALL), encoding="utf-8")


def write_model_card(r: dict, path) -> Path:
    d, x = r["data"], r["future"]["xgb"]
    evidence = "\n".join(f"- {e}" for e in r["recommendation"]["evidence"])
    alarms = "\n".join(f"- Month {a['month']}: {a['rule']} ({a['value']:.3f})" for a in r["drift"]["monitoring"]) \
        or "- None in months 6-7."
    text = f"""# Model card: bank account opening fraud (XGBoost)

## Intended use
Rank new bank account applications for manual fraud review under a fixed review budget (5% false-positive rate).
Decision support for a fraud team, not an automatic rejection system.

## Data
Feedzai Bank Account Fraud (BAF) suite, Base variant (Jesus et al., NeurIPS 2022), CC BY-NC-SA 4.0.
Synthetic data generated from a real fraud dataset (CTGAN with differential-privacy noise).
{d['n_rows']:,} applications over 8 months, fraud prevalence {d['prevalence']:.4f}.
Training months 0-4, calibration/decision month 5, evaluation months 6-7 (used once).

## Performance (months 6-7)
- Recall at the 5% budget: {x['recall_budget']['value']:.3f} (95% CI {x['recall_budget']['lo']:.3f}-{x['recall_budget']['hi']:.3f})
- Realised false-positive rate: {x['fpr_realized']['value']:.3f}
- PR-AUC: {x['pr_auc']['value']:.3f}; ROC-AUC: {x['roc_auc']:.3f}; Brier (isotonic): {x['brier']['isotonic']:.4f}

## Fairness
Protected attribute: age (>=50 vs <50); age is not a model input, but other variables can act as proxies.
False-positive-rate ratio: {r['fairness']['before']['fpr_ratio']['value']:.2f} before, {r['fairness']['after']['fpr_ratio']['value']:.2f} after group-specific thresholds.
Group-specific thresholds treat applicants differently by age and may be legally problematic; they are shown as a
trade-off analysis, not as a deployment recommendation.

## Limitations
- Synthetic data: results show the method, not the performance on any real bank's portfolio.
- No monetary amounts: costs use an assumed ratio of loss per missed fraud to review cost (20, 50, 100).
- Two months of out-of-time evaluation.

## Monitoring
Rules: score PSI > 0.25; monthly fraud rate outside the months 0-5 range; realised false-positive rate outside
[3%, 7%]. Triggered in the evaluation months:
{alarms}

## Recommendation: {r['recommendation']['decision']}
{evidence}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
