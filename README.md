# Bank account opening fraud: cost-based decisions, fairness audit and drift

[![CI](https://github.com/Rodgrandez/bank-account-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Rodgrandez/bank-account-fraud-detection/actions/workflows/ci.yml)

A fraud model is only useful if a fraud team can operate it: a fixed review budget, calibrated probabilities for
cost-based decisions, equal treatment across customer groups and a plan for when the data drifts. This project
builds and validates such a model on the public Feedzai Bank Account Fraud dataset with a strictly time-based
design: tuned on months 0-4, calibrated and thresholded on month 5, and evaluated once on the unseen months 6-7.

![Precision-recall](reports/figures/performance.png)

<!-- RESULTS:START -->
Evaluated once on the unseen months 6-7 with everything fixed on months 0-5. 95% bootstrap intervals in brackets; they resample applications and are conditional on the fitted model and the month-5 threshold.

| Metric (months 6-7) | XGBoost | Logistic regression |
|---|---|---|
| Recall at the 5% false-positive budget (threshold fixed on month 5) | 0.582 [0.564, 0.601] | 0.492 [0.474, 0.511] |
| Realised false-positive rate | 0.062 [0.061, 0.063] | 0.052 [0.051, 0.053] |
| Recall at exactly 5% FPR, threshold re-set on months 6-7 (approximately paper-comparable) | 0.541 | 0.481 |
| PR-AUC | 0.192 [0.178, 0.206] | 0.151 [0.139, 0.165] |
| Brier score (isotonic) | 0.0125 | 0.0128 |

- **Budget overshoot:** the threshold fixed on month 5 produced a realised false-positive rate of 0.062 for XGBoost in months 6-7 (budget 0.05), so part of its recall comes from extra alerts; that is why it exceeds the recall with the threshold re-set to exactly 5%.
- **Model comparison:** XGBoost minus logistic regression recall 0.090 [0.075, 0.104] at their month-5 thresholds (significant); at an equal false-positive rate of 5% the gap is 0.060 [0.045, 0.075].
- **Cost (loss per missed fraud = 50 reviews, an assumption):** 321.3 review-units per 1,000 applications with the model, vs 701.9 reviewing nothing and 1000 reviewing everything.

**Fairness (age, which is not a model input)**

- With the single threshold, legitimate applicants aged 50+ are flagged at 0.118 vs 0.051 for under 50: false-positive-rate ratio 2.31 [2.23, 2.39].
- The groups have very different fraud base rates (0.0234 vs 0.0083), so calibration and equal error rates cannot both hold; false positives are prioritised because they are the harm to legitimate applicants.
- Group thresholds (a trade-off analysis, not a recommendation, since they treat applicants differently by age) bring the ratio to 1.10 [1.05, 1.15], at a recall cost of 0.582 -> 0.560; both groups then exceed the budget (0.067 and 0.061).
- A model trained with age as an input reaches recall 0.585 but a larger disparity (ratio 3.06): excluding age costs almost no detection.

**Drift**

- Fraud rate rose from 0.0118 (month 5) to 0.0134 and 0.0147 (months 6-7), outside the months 0-5 range; the score distribution stayed stable (score PSI <= 0.010).
- Recall at the fixed threshold: 0.552 on month 5 (decision half), 0.582 and 0.583 in months 6 and 7, with realised false-positive rates 0.069 and 0.054.

**Recommendation to the risk committee: approve with conditions.**
- Condition: Monthly monitoring with the declared rules; triggered in months 6-7: fraud rate outside months 0-5 range.
- Condition: The realised false-positive rate (0.062) exceeds the 5% budget: re-set the budget threshold every month on the latest labelled month.
- Condition: Applicants aged 50+ are flagged 2.31x as often when legitimate: a documented fairness and legal review is required before go-live.

_Deviation from the pre-declared plan: after the first evaluation, a bug that scored the feature PSI of binary variables as zero was fixed and the evaluation re-run; no model, threshold or headline metric changed. The recommendation rule was also revised to judge the single-threshold policy, since group thresholds are not recommended for deployment._
<!-- RESULTS:END -->

## Data
Feedzai Bank Account Fraud (BAF) suite, Base variant: about one million synthetic bank account applications over
eight months, generated from a real fraud dataset with CTGAN and differential-privacy noise (Jesus et al., 2022,
"Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation", NeurIPS). License
CC BY-NC-SA 4.0; the data are downloaded by `make data` and not redistributed here.

## Design (declared before looking at months 6-7)
- **Validation:** hyperparameters and early stopping by rolling origin inside months 0-4 (both use the same
  validation month, so tuning scores are slightly optimistic; months 5-7 are untouched); month 5 split into a
  calibration half and a decision half; months 6-7 used once. No resampling (it distorts probabilities).
- **Decisions:** (i) review budget: threshold giving 5% false positives on month 5, applied unchanged to the future;
  (ii) cost rule: alert when calibrated probability >= 1/R, where R is the ratio of the loss from a missed fraud to
  the cost of one review (BAF has no amounts, so R in {20, 50, 100} is an explicit assumption).
- **Fairness:** age is not a model input; the audit compares false-positive rates for applicants aged 50+ and
  under 50, and measures the cost of equalising them with group thresholds. Group thresholds treat applicants
  differently by age and may be legally problematic, so they are presented as a trade-off, not a recommendation.
- **Uncertainty:** 1,000 bootstrap replicates on months 6-7; XGBoost is called better than the logistic benchmark
  only if the interval for the recall difference excludes zero.

See [the model card](reports/model_card.md) for limitations and monitoring rules.

## Reproduce
```bash
conda env create -f environment.yml && conda activate fraud-detection
make all      # needs a Kaggle API token; downloads the data, tunes, evaluates and writes reports/
```
The hyperparameter search (8 XGBoost configurations x 2 rolling folds, twice, on ~600k rows) took about 8 hours on a
laptop; evaluation and reports take a few minutes.

License: MIT (code). Data: CC BY-NC-SA 4.0 (Feedzai).
