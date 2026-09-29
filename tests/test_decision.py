import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from fraud import calibration, decision


def test_cost_threshold_is_bayes_rule():
    assert decision.cost_threshold(50) == pytest.approx(0.02)


def test_fpr_threshold_hits_budget_on_continuous_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.02).astype(int)
    s = rng.random(20000) + y
    t = decision.fpr_threshold(s, y, 0.05)
    assert decision.confusion_at(y, s >= t)["fpr"] == pytest.approx(0.05, abs=0.001)


def test_fpr_threshold_with_ties_never_exceeds_budget():
    y = np.r_[np.zeros(1000, int), np.ones(20, int)]
    s = np.r_[np.repeat([0.1, 0.5, 0.9], [900, 60, 40]), np.full(20, 0.9)]   # 100 negatives tied at the top
    t = decision.fpr_threshold(s, y, 0.05)
    assert decision.confusion_at(y, s >= t)["fpr"] <= 0.05


def test_confusion_and_costs():
    y = np.array([1, 1, 0, 0, 0])
    alerts = np.array([True, False, True, False, False])
    c = decision.confusion_at(y, alerts)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 2) and c["recall"] == 0.5
    # cost = R * missed frauds + number of reviews, per 1000 applications
    assert decision.cost_per_1000(y, alerts, 50) == pytest.approx(1000 * (50 * 1 + 2) / 5)
    ref = decision.reference_costs(y, 50)
    assert ref["review_all"] == 1000 and ref["review_none"] == pytest.approx(1000 * 50 * 2 / 5)


def test_recall_is_nan_without_positives():
    assert np.isnan(decision.confusion_at(np.zeros(5, int), np.ones(5, bool))["recall"])


def test_lift_and_capture():
    y = np.r_[np.ones(10, int), np.zeros(90, int)]
    s = np.r_[np.linspace(0.9, 1, 10), np.linspace(0, 0.5, 90)]
    lt = decision.lift_table(s, y, 10)
    assert lt.loc[0, "capture"] == 1.0 and lt.loc[0, "lift"] == pytest.approx(10.0)
    assert decision.capture_at(s, y, 0.10) == 1.0


def test_isotonic_calibration_improves_brier_on_holdout():
    rng = np.random.default_rng(1)
    p = rng.beta(0.5, 20, 40000)
    y = (rng.random(40000) < p).astype(int)
    raw = np.sqrt(p)                                   # monotone but badly calibrated score
    cal = calibration.Calibrator("isotonic").fit(raw[:20000], y[:20000])
    out = cal.transform(raw[20000:])
    assert np.all((out >= 0) & (out <= 1))
    assert brier_score_loss(y[20000:], out) < brier_score_loss(y[20000:], raw[20000:])
    platt = calibration.Calibrator("platt").fit(raw[:20000], y[:20000]).transform(raw[20000:])
    assert brier_score_loss(y[20000:], platt) < brier_score_loss(y[20000:], raw[20000:])
