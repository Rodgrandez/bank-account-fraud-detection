import pandas as pd
from sklearn.model_selection import train_test_split

from fraud import config


def split(df: pd.DataFrame, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    month = df[config.MONTH]
    valid = df[month == config.VALID_MONTH]
    a_idx, b_idx = train_test_split(valid.index, test_size=0.5, stratify=valid[config.TARGET], random_state=seed)
    return {"train": df[month.isin(config.TRAIN_MONTHS)],
            "valid_a": valid.loc[a_idx], "valid_b": valid.loc[b_idx],
            "future": df[month.isin(config.FUTURE_MONTHS)]}


def rolling_folds(train: pd.DataFrame) -> list[tuple[pd.Index, pd.Index]]:
    m = train[config.MONTH]
    return [(train.index[m.isin(fit)], train.index[m == val]) for fit, val in config.ROLLING_FOLDS]
