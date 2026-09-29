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
    cc, wa = r["fairness"]["cost_of_correction"], r["fairness"]["with_age"]
    rates_b, rates_a = r["fairness"]["before"]["rates"], r["fairness"]["after"]["rates"]
    groups = r["data"]["age_groups"]
    old = next(g for g in groups if ">=" in g)
    young = next(g for g in groups if ">=" not in g)
    months = {m["month"]: m for m in r["drift"]["by_month"]}
    rec = r["recommendation"]
    lines = [
        ("Evaluated once on the unseen months 6-7 with everything fixed on months 0-5. 95% bootstrap intervals in "
         "brackets; they resample applications and are conditional on the fitted model and the month-5 threshold."),
        "",
        "| Metric (months 6-7) | XGBoost | Logistic regression |",
        "|---|---|---|",
        (f"| Recall at the 5% false-positive budget (threshold fixed on month 5) | {_ci(x['recall_budget'])} "
         f"| {_ci(lr['recall_budget'])} |"),
        f"| Realised false-positive rate | {_ci(x['fpr_realized'])} | {_ci(lr['fpr_realized'])} |",
        (f"| Recall at exactly 5% FPR, threshold re-set on months 6-7 (approximately paper-comparable) "
         f"| {x['recall_oracle_5fpr']:.3f} | {lr['recall_oracle_5fpr']:.3f} |"),
        f"| PR-AUC | {_ci(x['pr_auc'])} | {_ci(lr['pr_auc'])} |",
        f"| Brier score (isotonic) | {x['brier']['isotonic']:.4f} | {lr['brier']['isotonic']:.4f} |",
        "",
        (f"- **Budget overshoot:** the threshold fixed on month 5 produced a realised false-positive rate of "
         f"{x['fpr_realized']['value']:.3f} for XGBoost in months 6-7 (budget 0.05), so part of its recall comes "
         f"from extra alerts; that is why it exceeds the recall with the threshold re-set to exactly 5%."),
        (f"- **Model comparison:** XGBoost minus logistic regression recall "
         f"{_ci(r['comparison']['xgb_minus_lr_recall'])} at their month-5 thresholds "
         f"({'significant' if r['comparison']['significant'] else 'not significant'}); at an equal false-positive "
         f"rate of 5% the gap is {_ci(r['comparison']['xgb_minus_lr_recall_equal_fpr'])}."),
        (f"- **Cost (loss per missed fraud = 50 reviews, an assumption):** {c50['cost_per_1000']['value']:.1f} "
         f"review-units per 1,000 applications with the model, vs {c50['review_none']:.1f} reviewing nothing and "
         f"{c50['review_all']:.0f} reviewing everything."),
        "",
        "**Fairness (age, which is not a model input)**",
        "",
        (f"- With the single threshold, legitimate applicants aged 50+ are flagged at "
         f"{rates_b[old]['fpr']:.3f} vs {rates_b[young]['fpr']:.3f} for under 50: false-positive-rate ratio "
         f"{_ci(fb, 2)}."),
        (f"- The groups have very different fraud base rates ({groups[old]['fraud_rate']:.4f} vs "
         f"{groups[young]['fraud_rate']:.4f}), so calibration and equal error rates cannot both hold; false "
         f"positives are prioritised because they are the harm to legitimate applicants."),
        (f"- Group thresholds (a trade-off analysis, not a recommendation, since they treat applicants differently "
         f"by age) bring the ratio to {_ci(fa, 2)}, at a recall cost of {cc['recall_before']:.3f} -> "
         f"{cc['recall_after']:.3f}; both groups then exceed the budget ({rates_a[old]['fpr']:.3f} and "
         f"{rates_a[young]['fpr']:.3f})."),
        (f"- A model trained with age as an input reaches recall {wa['recall_budget']['value']:.3f} but a larger "
         f"disparity (ratio {wa['fpr_ratio']['value']:.2f}): excluding age costs almost no detection."),
        "",
        "**Drift**",
        "",
        (f"- Fraud rate rose from {months[5]['fraud_rate']:.4f} (month 5) to {months[6]['fraud_rate']:.4f} and "
         f"{months[7]['fraud_rate']:.4f} (months 6-7), outside the months 0-5 range; the score distribution stayed "
         f"stable (score PSI <= {max(months[m]['psi_score'] for m in (5, 6, 7)):.3f})."),
        (f"- Recall at the fixed threshold: {x['valid_b_recall_budget']:.3f} on month 5 (decision half), "
         f"{months[6]['recall']:.3f} and {months[7]['recall']:.3f} in months 6 and 7, with realised false-positive "
         f"rates {months[6]['fpr']:.3f} and {months[7]['fpr']:.3f}."),
        "",
        f"**Recommendation to the risk committee: {rec['decision']}.**",
        *(f"- Condition: {c}" for c in rec.get("conditions", [])),
        "",
        ("_Deviation from the pre-declared plan: after the first evaluation, a bug that scored the feature PSI of "
         "binary variables as zero was fixed and the evaluation re-run; no model, threshold or headline metric "
         "changed. The recommendation rule was also revised to judge the single-threshold policy, since group "
         "thresholds are not recommended for deployment._"),
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
    conditions = "\n".join(f"- {c}" for c in r["recommendation"].get("conditions", [])) or "- None."
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

The two groups have very different fraud base rates, so calibration and equal error rates cannot both hold;
false positives are prioritised because they are the harm to legitimate applicants. A model trained with age as
an input has recall {r['fairness']['with_age']['recall_budget']['value']:.3f} and a false-positive-rate ratio of
{r['fairness']['with_age']['fpr_ratio']['value']:.2f}.

## Limitations
- Synthetic data: results show the method, not the performance on any real bank's portfolio.
- Bootstrap intervals are conditional on the fitted model and the month-5 threshold; they do not include
  month-to-month variation or threshold uncertainty.
- Early stopping and hyperparameter scores share the rolling-origin validation month (optimism contained in tuning).
- Isotonic calibration barely changes the Brier score: the raw scores were already close to calibrated.
- No monetary amounts: costs use an assumed ratio of loss per missed fraud to review cost (20, 50, 100).
- Two months of out-of-time evaluation.

## Monitoring
Rules: score PSI > 0.25; monthly fraud rate outside the months 0-5 range; realised false-positive rate outside
[3%, 7%]. Triggered in the evaluation months:
{alarms}

## Recommendation: {r['recommendation']['decision']}
{evidence}

Conditions:
{conditions}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
