import numpy as np
from conftest import make_baf

from fraud import config, data, features, models, split


def _setup():
    s = split.split(data.clean(make_baf(n=12000))[0])
    nums, cats = features.feature_columns(s["train"])
    return s, nums, cats


def test_tune_lr_picks_from_declared_grid():
    s, nums, cats = _setup()
    out = models.tune_lr(s["train"], split.rolling_folds(s["train"]), nums, cats)
    assert out["best"]["C"] in config.LR_GRID and len(out["grid"]) == len(config.LR_GRID)
    assert 0 < out["best"]["cv_pr_auc"] <= 1


def test_tune_xgb_reports_trees_from_early_stopping(monkeypatch):
    monkeypatch.setattr(config, "XGB_GRID", {"max_depth": (2,), "learning_rate": (0.1,), "min_child_weight": (1,)})
    monkeypatch.setattr(config, "XGB_MAX_TREES", 60)
    monkeypatch.setattr(config, "EARLY_STOP", 10)
    s, nums, cats = _setup()
    out = models.tune_xgb(s["train"], split.rolling_folds(s["train"]), nums, cats)
    assert 1 <= out["best"]["n_estimators"] <= 60 and out["best"]["max_depth"] == 2


def test_tuning_never_uses_month_five_or_later():
    s, _, _ = _setup()
    seen = set()
    for fit_idx, val_idx in split.rolling_folds(s["train"]):
        seen |= set(s["train"].loc[fit_idx, "month"]) | set(s["train"].loc[val_idx, "month"])
    assert max(seen) == 4


def test_final_models_predict_probabilities_and_importance():
    s, nums, cats = _setup()
    lr = models.make_lr(0.1, nums, cats).fit(s["train"][nums + cats], s["train"]["fraud_bool"])
    xgb = models.make_xgb({"max_depth": 2, "learning_rate": 0.1, "min_child_weight": 1}, 30, nums, cats)
    xgb.fit(s["train"][nums + cats], s["train"]["fraud_bool"])
    for m in (lr, xgb):
        p = m.predict_proba(s["future"][nums + cats])[:, 1]
        assert p.shape == (len(s["future"]),) and np.all((p >= 0) & (p <= 1))
    imp = models.original_importance(xgb, nums, cats)
    assert set(imp.index) <= set(nums + cats) and imp.index[0] in {"device_os", "velocity_6h"}
