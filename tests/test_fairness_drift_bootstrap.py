import numpy as np
import pandas as pd
import pytest

from fraud import bootstrap, decision, drift, fairness


def test_group_rates_and_fpr_ratio():
    y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    alerts = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0], bool)
    group = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], bool)
    r = fairness.group_rates(y, alerts, group)
    assert r.loc["age>=50", "fpr"] == pytest.approx(0.5) and r.loc["age<50", "fpr"] == pytest.approx(0.25)
    assert r.loc["age<50", "recall"] == 0.0
    assert fairness.fpr_ratio(y, alerts, group) == pytest.approx(2.0)


def test_fpr_ratio_handles_group_without_negatives():
    y = np.array([1, 1, 0, 0])
    alerts = np.array([1, 0, 1, 0], bool)
    group = np.array([1, 1, 0, 0], bool)                     # the older group has no legitimate applicants
    assert np.isnan(fairness.fpr_ratio(y, alerts, group))


def test_group_thresholds_equalise_fpr_and_keep_budget():
    rng = np.random.default_rng(0)
    n = 40000
    group = rng.random(n) < 0.3
    y = (rng.random(n) < 0.02).astype(int)
    s = rng.random(n) + 0.1 * group + y                     # older group scores higher: more false positives
    t = decision.fpr_threshold(s, y, 0.05)
    before = fairness.fpr_ratio(y, s >= t, group)
    th = fairness.group_thresholds(s, y, group, 0.05)
    after_alerts = fairness.apply_group_thresholds(s, group, th)
    assert before > 1.5
    assert fairness.fpr_ratio(y, after_alerts, group) == pytest.approx(1.0, abs=0.05)
    assert decision.confusion_at(y, after_alerts)["fpr"] <= 0.05


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50000)
    assert drift.psi(a, rng.normal(0, 1, 50000)) < 0.01
    assert drift.psi(a, rng.normal(1, 1, 50000)) > 0.25
    x = np.r_[a[:1000], np.full(100, np.nan)]
    assert np.isfinite(drift.psi(x, x))                     # missing values form their own bin


def test_psi_categorical():
    a = pd.Series(["a"] * 80 + ["b"] * 20)
    assert drift.psi_categorical(a, a) == pytest.approx(0.0)
    assert drift.psi_categorical(a, pd.Series(["a"] * 20 + ["b"] * 70 + ["c"] * 10)) > 0.25


def test_bootstrap_interval_covers_known_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 1, 2000)
    out = bootstrap.interval(lambda i: x[i].mean(), len(x), n_boot=300, seed=1)
    assert out["lo"] < 5 < out["hi"] and out["value"] == pytest.approx(x.mean())


def test_bootstrap_ignores_nan_replicates():
    y = np.r_[1, np.zeros(99, int)]                           # most resamples miss the single fraud
    out = bootstrap.interval(lambda i: y[i].sum() / y[i].sum() if y[i].sum() else np.nan, 100, n_boot=200, seed=0)
    assert out["lo"] == out["hi"] == 1.0


def test_psi_detects_shift_in_binary_variable():
    # quantile edges collapse on a 0/1 variable; low-cardinality numerics must be binned by value
    # with 38.7% ones every decile edge is 0 or 1, so a 38.7% -> 46.5% shift (phone_home_valid in BAF) was scored 0
    a = np.r_[np.zeros(6130), np.ones(3870)]
    b = np.r_[np.zeros(5350), np.ones(4650)]
    assert drift.psi(a, a) == pytest.approx(0.0)
    assert drift.psi(a, b) == pytest.approx(drift.psi_categorical(a, b)) and drift.psi(a, b) > 0.02
