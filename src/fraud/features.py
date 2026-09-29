import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud import config


def feature_columns(df: pd.DataFrame, with_age: bool = False) -> tuple[list[str], list[str]]:
    excluded = {config.TARGET, config.MONTH} | (set() if with_age else {config.AGE})
    cats = [c for c in config.CATEGORICAL if c in df.columns]
    nums = [c for c in df.columns if c not in excluded and c not in cats]
    return nums, cats


def make_preprocessor(nums: list[str], cats: list[str], scale: bool) -> ColumnTransformer:
    """scale=True (logistic regression): median imputation + standardisation.
    scale=False (XGBoost): numeric columns pass through with NaN, which trees handle natively."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]) if scale \
        else "passthrough"
    return ColumnTransformer([
        ("num", numeric, nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ])
