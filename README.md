# Bank account opening fraud: cost-based decisions, fairness audit and drift

[![CI](https://github.com/Rodgrandez/bank-account-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Rodgrandez/bank-account-fraud-detection/actions/workflows/ci.yml)

A fraud model is only useful if a fraud team can operate it: a fixed review budget, calibrated probabilities for
cost-based decisions, equal treatment across customer groups and a plan for when the data drifts. This project
builds and validates such a model on the public Feedzai Bank Account Fraud dataset with a strictly time-based
design: tuned on months 0-4, calibrated and thresholded on month 5, and evaluated once on the unseen months 6-7.

![Precision-recall](reports/figures/performance.png)

<!-- RESULTS:START -->
Evaluated once on the unseen months 6-7 with everything fixed on months 0-5. 95% bootstrap intervals in brackets.

| Metric (months 6-7) | XGBoost | Logistic regression |
|---|---|---|
| Recall at the 5% false-positive budget (threshold fixed on month 5) | 0.582 [0.564, 0.601] | 0.492 [0.474, 0.511] |
| Realised false-positive rate | 0.062 [0.061, 0.063] | 0.052 [0.051, 0.053] |
| PR-AUC | 0.192 [0.178, 0.206] | 0.151 [0.139, 0.165] |
| Brier score (isotonic) | 0.0125 | 0.0128 |
| Recall at 5% FPR, threshold re-set on months 6-7 (paper-comparable) | 0.541 | 0.481 |

- **Model comparison:** XGBoost minus logistic regression recall 0.090 [0.075, 0.104] (significant).
- **Cost (loss per missed fraud = 50 reviews):** 321.3 review-units per 1,000 applications with the model, vs 701.9 reviewing nothing and 1000 reviewing everything.
- **Fairness:** false-positive-rate ratio (age>=50 / <50) 2.31 [2.23, 2.39] before and 1.10 [1.05, 1.15] after group thresholds; recall 0.582 -> 0.560.
- **Monitoring alarms in months 6-7:** 2.
- **Recommendation to the risk committee: approve with conditions.**
<!-- RESULTS:END -->

## Data
Feedzai Bank Account Fraud (BAF) suite, Base variant: about one million synthetic bank account applications over
eight months, generated from a real fraud dataset with CTGAN and differential-privacy noise (Jesus et al., 2022,
"Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation", NeurIPS). License
CC BY-NC-SA 4.0; the data are downloaded by `make data` and not redistributed here.

## Design (declared before looking at months 6-7)
- **Validation:** hyperparameters and early stopping by rolling origin inside months 0-4; month 5 split into a
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
