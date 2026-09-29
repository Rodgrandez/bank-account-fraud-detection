import json

from conftest import make_baf

from fraud import data, evaluate, features, models, plots, report, split


def _run(n_boot=50):
    df, dropped = data.clean(make_baf(n=16000, seed=3))
    s = split.split(df)
    s["_all"] = df
    cols = {"xgb": features.feature_columns(s["train"]), "lr": features.feature_columns(s["train"]),
            "xgb_with_age": features.feature_columns(s["train"], with_age=True)}
    p = {"max_depth": 2, "learning_rate": 0.1, "min_child_weight": 1}
    fitted = {"xgb": models.make_xgb(p, 40, *cols["xgb"]), "lr": models.make_lr(0.1, *cols["lr"]),
              "xgb_with_age": models.make_xgb(p, 40, *cols["xgb_with_age"])}
    for name, m in fitted.items():
        nums, cats = cols[name]
        m.fit(s["train"][nums + cats], s["train"]["fraud_bool"])
    tuning = {k: {"best": {"cv_pr_auc": 0.1}} for k in fitted}
    return evaluate.run(s, fitted, cols, tuning, dropped, n_boot=n_boot)


def test_results_have_declared_keys_and_valid_intervals():
    res, _ = _run()
    f = res["future"]["xgb"]["recall_budget"]
    assert f["lo"] <= f["value"] <= f["hi"]
    assert set(res["cost"]) == {"20", "50", "100"}
    assert res["cost"]["50"]["threshold"] == 0.02
    assert {"before", "after", "with_age", "cost_of_correction", "by_month"} <= set(res["fairness"])
    assert [m["month"] for m in res["drift"]["by_month"]] == list(range(8))
    assert res["recommendation"]["decision"] in {"approve", "approve with conditions", "postpone"}
    assert len(res["recommendation"]["evidence"]) >= 3


def test_budget_threshold_comes_from_valid_b_only():
    _, arrays = _run()
    # the realised FPR on month-5 half B, where the threshold was set, respects the budget
    assert arrays["xgb"]["valid_b_fpr"] <= 0.05


def test_recommendation_rule():
    base = {"comparison": {"xgb_minus_lr_recall": {"value": 0.1, "lo": 0.05, "hi": 0.15}},
            "drift": {"monitoring": []},
            "fairness": {"after": {"fpr_ratio": {"value": 1.0, "lo": 0.9, "hi": 1.1}},
                         "before": {"fpr_ratio": {"value": 1.5, "lo": 1.3, "hi": 1.7}}},
            "future": {"xgb": {"recall_budget": {"value": 0.5, "lo": 0.45, "hi": 0.55},
                               "fpr_realized": {"value": 0.05, "lo": 0.049, "hi": 0.051}}}}
    assert evaluate.recommend(base)["decision"] == "approve"
    alarm = {**base, "drift": {"monitoring": [{"month": 7, "rule": "score PSI > 0.25", "value": 0.3}]}}
    assert evaluate.recommend(alarm)["decision"] == "approve with conditions"
    worse = {**base, "comparison": {"xgb_minus_lr_recall": {"value": -0.1, "lo": -0.15, "hi": -0.05}}}
    assert evaluate.recommend(worse)["decision"] == "postpone"


def test_readme_block_uses_results(tmp_path):
    res, arrays = _run()
    readme = tmp_path / "README.md"
    readme.write_text("# T\n<!-- RESULTS:START -->\nSTALE-BLOCK-XYZ\n<!-- RESULTS:END -->\nend\n", encoding="utf-8")
    report.update_readme(readme, res)
    text = readme.read_text(encoding="utf-8")
    r = res["future"]["xgb"]["recall_budget"]
    assert "STALE-BLOCK-XYZ" not in text and f"{r['value']:.3f}" in text and text.endswith("end\n")
    report.write_results(res, tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text())["recommendation"] == res["recommendation"]
    report.write_model_card(res, tmp_path / "card.md")
    assert res["recommendation"]["decision"] in (tmp_path / "card.md").read_text(encoding="utf-8")
    paths = plots.all_figures(res, arrays, tmp_path / "fig")
    assert len(paths) == 6 and all(p.stat().st_size > 0 for p in paths)
