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
                         "before": {"fpr_ratio": {"value": 1.0, "lo": 0.95, "hi": 1.05}}},
            "future": {"xgb": {"recall_budget": {"value": 0.5, "lo": 0.45, "hi": 0.55},
                               "fpr_realized": {"value": 0.05, "lo": 0.049, "hi": 0.051}}}}
    assert evaluate.recommend(base)["decision"] == "approve"
    alarm = {**base, "drift": {"monitoring": [{"month": 7, "rule": "score PSI > 0.25", "value": 0.3}]}}
    assert evaluate.recommend(alarm)["decision"] == "approve with conditions"
    worse = {**base, "comparison": {"xgb_minus_lr_recall": {"value": -0.1, "lo": -0.15, "hi": -0.05}}}
    assert evaluate.recommend(worse)["decision"] == "postpone"


def test_recommendation_judges_the_deployed_single_threshold_policy():
    # group thresholds are not recommended for deployment, so the fairness condition must use the single threshold
    base = {"comparison": {"xgb_minus_lr_recall": {"value": 0.1, "lo": 0.05, "hi": 0.15}},
            "drift": {"monitoring": []},
            "fairness": {"after": {"fpr_ratio": {"value": 1.0, "lo": 0.95, "hi": 1.05}},
                         "before": {"fpr_ratio": {"value": 2.3, "lo": 2.2, "hi": 2.4}}},
            "future": {"xgb": {"recall_budget": {"value": 0.5, "lo": 0.45, "hi": 0.55},
                               "fpr_realized": {"value": 0.062, "lo": 0.061, "hi": 0.063}}}}
    out = evaluate.recommend(base)
    assert out["decision"] == "approve with conditions"
    assert any("age" in c for c in out["conditions"]) and any("false-positive" in c for c in out["conditions"])


def test_results_include_equal_fpr_comparison():
    res, _ = _run()
    d = res["comparison"]["xgb_minus_lr_recall_equal_fpr"]
    assert d["lo"] <= d["value"] <= d["hi"]


def test_readme_explains_budget_overshoot_and_fairness_caveats():
    res, _ = _run()
    text = report.results_markdown(res)
    low = text.lower()
    assert "budget overshoot" in low and "equal false-positive rate" in low
    assert "base rates" in text and "with age as an input" in text and "Deviation from the pre-declared plan" in text


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
