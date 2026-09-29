import numpy as np
from conftest import make_baf

from fraud import config, data


def test_config_split_is_declared():
    assert config.TRAIN_MONTHS == (0, 1, 2, 3, 4) and config.VALID_MONTH == 5 and config.FUTURE_MONTHS == (6, 7)
    assert config.ROLLING_FOLDS == [((0, 1, 2), 3), ((0, 1, 2, 3), 4)]
    assert config.FPR_BUDGET == 0.05 and config.COST_RATIOS == (20, 50, 100) and config.CENTRAL_RATIO == 50


def test_clean_turns_negative_sentinels_into_nan():
    raw = make_baf()
    df, _ = data.clean(raw)
    for c in config.SENTINEL_NEGATIVE:
        assert (df[c].dropna() >= 0).all()
        assert df[c].isna().sum() == (raw[c] < 0).sum()


def test_clean_drops_constant_columns_but_never_target_or_month():
    raw = make_baf()
    raw.loc[:, "fraud_bool"] = 0                                  # even a constant target must be kept
    df, dropped = data.clean(raw)
    assert dropped == ["device_fraud_count"]
    assert {"fraud_bool", "month"} <= set(df.columns) and "device_fraud_count" not in df.columns


def test_interim_roundtrip(tmp_path):
    raw_path = tmp_path / "Base.csv"
    make_baf().to_csv(raw_path, index=False)
    out = data.build_interim(raw_path, tmp_path / "interim" / "base.parquet")
    df, dropped = data.load_interim(out)
    assert len(df) == 8000 and dropped == ["device_fraud_count"] and np.isnan(df["session_length_in_minutes"]).any()
