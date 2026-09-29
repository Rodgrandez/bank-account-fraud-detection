import json
import sys

import joblib
import pandas as pd

from fraud import config, data, evaluate, features, models, plots, report, split

MODELS_DIR = config.ROOT / "data" / "models"


def _load():
    df, dropped = data.load_interim(config.INTERIM)
    s = split.split(df)
    s["_all"] = df
    cols = {"xgb": features.feature_columns(s["train"]), "lr": features.feature_columns(s["train"]),
            "xgb_with_age": features.feature_columns(s["train"], with_age=True)}
    return s, cols, dropped


def stage_data():
    data.build_interim(data.download())


def stage_tune():
    s, cols, _ = _load()
    folds = split.rolling_folds(s["train"])
    tuning = {"lr": models.tune_lr(s["train"], folds, *cols["lr"]),
              "xgb": models.tune_xgb(s["train"], folds, *cols["xgb"]),
              "xgb_with_age": models.tune_xgb(s["train"], folds, *cols["xgb_with_age"])}
    fitted = {"lr": models.make_lr(tuning["lr"]["best"]["C"], *cols["lr"])}
    for name in ("xgb", "xgb_with_age"):
        b = tuning[name]["best"]
        params = {k: b[k] for k in config.XGB_GRID}
        fitted[name] = models.make_xgb(params, b["n_estimators"], *cols[name])
    for name, m in fitted.items():
        nums, cats = cols[name]
        m.fit(s["train"][nums + cats], s["train"][config.TARGET])
        print(f"   fitted {name}", flush=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"fitted": fitted, "tuning": tuning}, MODELS_DIR / "models.joblib")
    config.TABLES.mkdir(parents=True, exist_ok=True)
    for name, t in tuning.items():
        pd.DataFrame(t["grid"]).to_csv(config.TABLES / f"tuning_{name}.csv", index=False)


def stage_evaluate():
    s, cols, dropped = _load()
    bundle = joblib.load(MODELS_DIR / "models.joblib")
    results, arrays = evaluate.run(s, bundle["fitted"], cols, bundle["tuning"], dropped)
    report.write_results(results, config.REPORTS / "results.json")
    joblib.dump(arrays, MODELS_DIR / "arrays.joblib")


def stage_report():
    results = json.loads((config.REPORTS / "results.json").read_text(encoding="utf-8"))
    arrays = joblib.load(MODELS_DIR / "arrays.joblib")
    plots.all_figures(results, arrays, config.FIGURES)
    pd.DataFrame(results["drift"]["by_month"]).to_csv(config.TABLES / "drift_by_month.csv", index=False)
    pd.DataFrame(results["lift"]["deciles"]).to_csv(config.TABLES / "lift.csv", index=False)
    report.update_readme(config.ROOT / "README.md", results)
    report.write_model_card(results, config.REPORTS / "model_card.md")


STAGES = {"data": [stage_data], "tune": [stage_tune], "evaluate": [stage_evaluate], "report": [stage_report],
          "all": [stage_data, stage_tune, stage_evaluate, stage_report]}

if __name__ == "__main__":
    for fn in STAGES[sys.argv[1] if len(sys.argv) > 1 else "all"]:
        print(f"== {fn.__name__}", flush=True)
        fn()
