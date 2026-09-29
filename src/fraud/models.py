import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud import config
from fraud.features import make_preprocessor


def make_lr(C: float, nums, cats) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(nums, cats, scale=True)),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


def _xgb(params: dict, n_estimators: int, **extra) -> XGBClassifier:
    return XGBClassifier(n_estimators=n_estimators, tree_method="hist", eval_metric="aucpr",
                         random_state=config.SEED, n_jobs=-1, **params, **extra)


def make_xgb(params: dict, n_estimators: int, nums, cats) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(nums, cats, scale=False)), ("clf", _xgb(params, n_estimators))])


def tune_lr(train, folds, nums, cats) -> dict:
    X, y = train[nums + cats], train[config.TARGET]
    grid = []
    for C in config.LR_GRID:
        scores = [average_precision_score(y.loc[va], make_lr(C, nums, cats).fit(X.loc[tr], y.loc[tr])
                                          .predict_proba(X.loc[va])[:, 1]) for tr, va in folds]
        grid.append({"C": C, "cv_pr_auc": float(np.mean(scores))})
    return {"best": max(grid, key=lambda r: r["cv_pr_auc"]), "grid": grid}


def tune_xgb(train, folds, nums, cats) -> dict:
    X, y = train[nums + cats], train[config.TARGET]
    grid = []
    for combo in itertools.product(*config.XGB_GRID.values()):
        params = dict(zip(config.XGB_GRID, combo, strict=True))
        scores, trees = [], []
        for tr, va in folds:
            prep = make_preprocessor(nums, cats, scale=False).fit(X.loc[tr])
            Xtr, Xva = prep.transform(X.loc[tr]), prep.transform(X.loc[va])
            clf = _xgb(params, config.XGB_MAX_TREES, early_stopping_rounds=config.EARLY_STOP)
            clf.fit(Xtr, y.loc[tr], eval_set=[(Xva, y.loc[va])], verbose=False)
            scores.append(average_precision_score(y.loc[va], clf.predict_proba(Xva)[:, 1]))
            trees.append(clf.best_iteration + 1)
        grid.append({**params, "cv_pr_auc": float(np.mean(scores)), "n_estimators": int(np.rint(np.mean(trees)))})
    return {"best": max(grid, key=lambda r: r["cv_pr_auc"]), "grid": grid}


def original_importance(pipeline: Pipeline, nums, cats) -> pd.Series:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    gain = pipeline.named_steps["clf"].get_booster().get_score(importance_type="gain")
    out = {}
    for i, name in enumerate(names):
        col = name.split("__", 1)[1]
        orig = col if col in nums else next(c for c in cats if col.startswith(f"{c}_"))
        out[orig] = out.get(orig, 0.0) + gain.get(f"f{i}", 0.0)
    return pd.Series(out).sort_values(ascending=False)
