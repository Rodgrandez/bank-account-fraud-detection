# Model card: bank account opening fraud (XGBoost)

## Intended use
Rank new bank account applications for manual fraud review under a fixed review budget (5% false-positive rate).
Decision support for a fraud team, not an automatic rejection system.

## Data
Feedzai Bank Account Fraud (BAF) suite, Base variant (Jesus et al., NeurIPS 2022), CC BY-NC-SA 4.0.
Synthetic data generated from a real fraud dataset (CTGAN with differential-privacy noise).
1,000,000 applications over 8 months, fraud prevalence 0.0110.
Training months 0-4, calibration/decision month 5, evaluation months 6-7 (used once).

## Performance (months 6-7)
- Recall at the 5% budget: 0.582 (95% CI 0.564-0.601)
- Realised false-positive rate: 0.062
- PR-AUC: 0.192; ROC-AUC: 0.893; Brier (isotonic): 0.0125

## Fairness
Protected attribute: age (>=50 vs <50); age is not a model input, but other variables can act as proxies.
False-positive-rate ratio: 2.31 before, 1.10 after group-specific thresholds.
Group-specific thresholds treat applicants differently by age and may be legally problematic; they are shown as a
trade-off analysis, not as a deployment recommendation.

## Limitations
- Synthetic data: results show the method, not the performance on any real bank's portfolio.
- No monetary amounts: costs use an assumed ratio of loss per missed fraud to review cost (20, 50, 100).
- Two months of out-of-time evaluation.

## Monitoring
Rules: score PSI > 0.25; monthly fraud rate outside the months 0-5 range; realised false-positive rate outside
[3%, 7%]. Triggered in the evaluation months:
- Month 6: fraud rate outside months 0-5 range (0.013)
- Month 7: fraud rate outside months 0-5 range (0.015)

## Recommendation: approve with conditions
- Recall at the 5% false-positive budget in the unseen months: 0.582 (95% CI 0.564-0.601); realised false-positive rate 0.062.
- XGBoost minus logistic regression recall: +0.090 (95% CI +0.075 to +0.104).
- False-positive-rate ratio (age>=50 / <50) after group thresholds: 1.10 (95% CI 1.05-1.15); before: 2.31.
- Monitoring rules triggered in the unseen months: fraud rate outside months 0-5 range (month 6), fraud rate outside months 0-5 range (month 7).
