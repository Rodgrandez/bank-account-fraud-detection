import numpy as np
import pandas as pd
from conftest import make_baf

from fraud import config, data, features, split


def _df():
    return data.clean(make_baf())[0]


def test_split_blocks_by_month_without_overlap():
    s = split.split(_df())
    assert set(s["train"]["month"]) == set(config.TRAIN_MONTHS)
    assert set(s["valid_a"]["month"]) == set(s["valid_b"]["month"]) == {5}
    assert set(s["future"]["month"]) == {6, 7}
    idx = [set(v.index) for v in s.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(idx) for b in idx[i + 1:])
    assert sum(len(v) for v in s.values()) == len(_df())


def test_valid_halves_are_stratified_and_deterministic():
    a = split.split(_df())
    b = split.split(_df())
    assert a["valid_a"].index.equals(b["valid_a"].index)
    ra, rb = a["valid_a"]["fraud_bool"].mean(), a["valid_b"]["fraud_bool"].mean()
    assert abs(ra - rb) < 0.01


def test_rolling_folds_follow_declared_months():
    s = split.split(_df())
    folds = split.rolling_folds(s["train"])
    tr0, va0 = folds[0]
    assert set(s["train"].loc[tr0, "month"]) == {0, 1, 2} and set(s["train"].loc[va0, "month"]) == {3}
    tr1, va1 = folds[1]
    assert set(s["train"].loc[tr1, "month"]) == {0, 1, 2, 3} and set(s["train"].loc[va1, "month"]) == {4}


def test_feature_columns_exclude_target_month_and_age_by_default():
    nums, cats = features.feature_columns(_df())
    assert "fraud_bool" not in nums + cats and "month" not in nums + cats and "customer_age" not in nums
    assert cats == config.CATEGORICAL
    nums_age, _ = features.feature_columns(_df(), with_age=True)
    assert "customer_age" in nums_age


def test_preprocessor_fits_only_training_rows():
    s = split.split(_df())
    nums, cats = features.feature_columns(s["train"])
    prep = features.make_preprocessor(nums, cats, scale=True).fit(s["train"][nums + cats])
    medians = prep.named_transformers_["num"].named_steps["impute"].statistics_
    assert np.allclose(medians, s["train"][nums].median().to_numpy())
    wild = s["future"].copy()
    wild[nums] = wild[nums] * 1000                              # extreme future values must not move train stats
    prep2 = features.make_preprocessor(nums, cats, scale=True).fit(s["train"][nums + cats])
    prep2.transform(wild[nums + cats])
    assert np.allclose(prep2.named_transformers_["num"].named_steps["impute"].statistics_, medians)


def test_unseen_category_in_future_is_ignored():
    s = split.split(_df())
    nums, cats = features.feature_columns(s["train"])
    prep = features.make_preprocessor(nums, cats, scale=False).fit(s["train"][nums + cats])
    fut = s["future"][nums + cats].copy()
    fut.loc[fut.index[:5], "device_os"] = "x11"
    out = prep.transform(fut)
    assert out.shape[0] == len(fut) and not pd.isna(out[:, len(nums):]).any()
