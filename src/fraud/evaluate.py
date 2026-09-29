import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fraud import bootstrap, config, decision, drift, fairness
from fraud.calibration import Calibrator
from fraud.models import original_importance


def _scores(model, df, cols) -> np.ndarray:
    nums, cats = cols
    return model.predict_proba(df[nums + cats])[:, 1]


def _budget_metrics(y, s, t, n_boot):
    alerts = s >= t
    n = len(y)
    return {
        "recall_budget": bootstrap.interval(lambda i: decision.confusion_at(y[i], alerts[i])["recall"], n, n_boot),
        "fpr_realized": bootstrap.interval(lambda i: decision.confusion_at(y[i], alerts[i])["fpr"], n, n_boot),
        "pr_auc": bootstrap.interval(lambda i: average_precision_score(y[i], s[i]) if y[i].any() else np.nan,
                                     n, n_boot),
        "roc_auc": float(roc_auc_score(y, s)),
        "recall_oracle_5fpr": decision.confusion_at(y, s >= decision.fpr_threshold(s, y, config.FPR_BUDGET))[
            "recall"],
    }


def run(splits, fitted, cols, tuning, dropped, n_boot=None):
    tr, va, vb, fu = (splits[k] for k in ("train", "valid_a", "valid_b", "future"))
    y_a, y_b, y_f = (d[config.TARGET].to_numpy() for d in (va, vb, fu))
    res = {"future": {}, "tuning": {k: v["best"] for k, v in tuning.items()}}
    arrays = {}
    for name, model in fitted.items():
        s_a, s_b, s_f = (_scores(model, d, cols[name]) for d in (va, vb, fu))
        iso = Calibrator("isotonic").fit(s_a, y_a)
        platt = Calibrator("platt").fit(s_a, y_a)
        t = decision.fpr_threshold(s_b, y_b, config.FPR_BUDGET)
        m = _budget_metrics(y_f, s_f, t, n_boot)
        m["valid_b_recall_budget"] = decision.confusion_at(y_b, s_b >= t)["recall"]
        p_f = iso.transform(s_f)
        m["brier"] = {"raw": float(brier_score_loss(y_f, s_f)), "isotonic": float(brier_score_loss(y_f, p_f)),
                      "platt": float(brier_score_loss(y_f, platt.transform(s_f)))}
        res["future"][name] = m
        arrays[name] = {"s_b": s_b, "s_f": s_f, "p_f": p_f, "p_b": iso.transform(s_b), "t": t,
                        "valid_b_fpr": decision.confusion_at(y_b, s_b >= t)["fpr"]}
    arrays["y_f"], arrays["y_b"] = y_f, y_b

    a, b = arrays["xgb"], arrays["lr"]
    diff = bootstrap.interval(
        lambda i: decision.confusion_at(y_f[i], a["s_f"][i] >= a["t"])["recall"]
        - decision.confusion_at(y_f[i], b["s_f"][i] >= b["t"])["recall"], len(y_f), n_boot)
    res["comparison"] = {"xgb_minus_lr_recall": diff, "significant": bool(diff["lo"] > 0 or diff["hi"] < 0)}

    def recall_at_budget_fpr(y, s):
        return decision.confusion_at(y, s >= decision.fpr_threshold(s, y, config.FPR_BUDGET))["recall"]

    # sensitivity: both models at the same 5% false-positive rate (threshold re-set within each replicate)
    res["comparison"]["xgb_minus_lr_recall_equal_fpr"] = bootstrap.interval(
        lambda i: recall_at_budget_fpr(y_f[i], a["s_f"][i]) - recall_at_budget_fpr(y_f[i], b["s_f"][i]),
        len(y_f), n_boot)

    res["cost"] = {}
    for ratio in config.COST_RATIOS:
        th = decision.cost_threshold(ratio)
        alerts = a["p_f"] >= th
        grid = np.linspace(0.001, 0.5, 500)
        costs_b = [decision.cost_per_1000(y_b, a["p_b"] >= g, ratio) for g in grid]
        at_rule = decision.cost_per_1000(y_b, a["p_b"] >= th, ratio)
        res["cost"][str(ratio)] = {
            "threshold": th,
            "cost_per_1000": bootstrap.interval(lambda i, al=alerts, r=ratio: decision.cost_per_1000(y_f[i], al[i], r),
                                                len(y_f), n_boot),
            **decision.reference_costs(y_f, ratio),
            "recall": decision.confusion_at(y_f, alerts)["recall"],
            "alert_rate": decision.confusion_at(y_f, alerts)["alert_rate"],
            "valid_b_regret_pct": float(100 * (at_rule - min(costs_b)) / min(costs_b)) if min(costs_b) else 0.0,
        }

    lt = decision.lift_table(a["s_f"], y_f)
    res["lift"] = {"top5_capture": decision.capture_at(a["s_f"], y_f, 0.05),
                   "top10_capture": decision.capture_at(a["s_f"], y_f, 0.10), "deciles": lt.to_dict("records")}

    g_f, g_b = fairness.older(fu[config.AGE]), fairness.older(vb[config.AGE])
    alerts_before = a["s_f"] >= a["t"]
    th = fairness.group_thresholds(a["s_b"], y_b, g_b, config.FPR_BUDGET)
    alerts_after = fairness.apply_group_thresholds(a["s_f"], g_f, th)
    wa = arrays["xgb_with_age"]

    def ratio_ci(al):
        return bootstrap.interval(lambda i: fairness.fpr_ratio(y_f[i], al[i], g_f[i]), len(y_f), n_boot)

    all_df = {}
    by_month = []
    for month in range(8):
        d = splits["_all"][splits["_all"][config.MONTH] == month]
        s_m = _scores(fitted["xgb"], d, cols["xgb"])
        by_month.append({"month": month, "fpr_ratio": fairness.fpr_ratio(d[config.TARGET].to_numpy(), s_m >= a["t"],
                                                                        fairness.older(d[config.AGE])),
                         "in_sample": month in config.TRAIN_MONTHS})
        all_df[month] = (d, s_m)
    res["fairness"] = {
        "before": {"fpr_ratio": ratio_ci(alerts_before), "rates": fairness.group_rates(y_f, alerts_before, g_f)
                   .to_dict("index")},
        "after": {"fpr_ratio": ratio_ci(alerts_after), "rates": fairness.group_rates(y_f, alerts_after, g_f)
                  .to_dict("index")},
        "with_age": {"fpr_ratio": ratio_ci(wa["s_f"] >= wa["t"]),
                     "recall_budget": res["future"]["xgb_with_age"]["recall_budget"]},
        "cost_of_correction": {
            "recall_before": decision.confusion_at(y_f, alerts_before)["recall"],
            "recall_after": decision.confusion_at(y_f, alerts_after)["recall"],
            "cost_central_before": decision.cost_per_1000(y_f, alerts_before, config.CENTRAL_RATIO),
            "cost_central_after": decision.cost_per_1000(y_f, alerts_after, config.CENTRAL_RATIO)},
        "by_month": by_month,
    }

    ref_scores = np.concatenate([all_df[m][1] for m in config.TRAIN_MONTHS])
    hist_rates = [float(all_df[m][0][config.TARGET].mean()) for m in range(6)]
    drift_rows, alarms = [], []
    for month in range(8):
        d, s_m = all_df[month]
        y_m = d[config.TARGET].to_numpy()
        c = decision.confusion_at(y_m, s_m >= a["t"])
        row = {"month": month, "fraud_rate": float(y_m.mean()), "psi_score": drift.psi(ref_scores, s_m),
               "recall": c["recall"], "fpr": c["fpr"], "pr_auc": float(average_precision_score(y_m, s_m)),
               "in_sample": month in config.TRAIN_MONTHS}
        drift_rows.append(row)
        if month in config.FUTURE_MONTHS:
            if row["psi_score"] > config.PSI_ALERT:
                alarms.append({"month": month, "rule": f"score PSI > {config.PSI_ALERT}", "value": row["psi_score"]})
            if not min(hist_rates) <= row["fraud_rate"] <= max(hist_rates):
                alarms.append({"month": month, "rule": "fraud rate outside months 0-5 range", "value": row["fraud_rate"]})
            if not config.FPR_BAND[0] <= row["fpr"] <= config.FPR_BAND[1]:
                alarms.append({"month": month, "rule": f"realised FPR outside {config.FPR_BAND}", "value": row["fpr"]})
    nums, cats = cols["xgb"]
    top = original_importance(fitted["xgb"], nums, cats).head(10).index
    res["drift"] = {"by_month": drift_rows, "monitoring": alarms, "psi_top_features": {
        c: (drift.psi_categorical(tr[c], fu[c]) if c in cats else drift.psi(tr[c], fu[c])) for c in top}}

    full = splits["_all"]
    res["data"] = {"n_rows": len(full), "prevalence": float(full[config.TARGET].mean()),
                   "by_month": [{"month": int(m), "n": len(g), "fraud_rate": float(g[config.TARGET].mean())}
                                for m, g in full.groupby(config.MONTH)],
                   "dropped_constant": dropped,
                   "age_groups": {fairness.LABELS[k]: {"n": len(g), "fraud_rate": float(g[config.TARGET].mean())}
                                  for k, g in full.groupby(fairness.older(full[config.AGE]))}}
    res["recommendation"] = recommend(res)
    return res, arrays


def recommend(res: dict) -> dict:
    """Rule declared in the plan, judged on the policy that would be deployed (a single threshold)."""
    diff = res["comparison"]["xgb_minus_lr_recall"]
    before, after = res["fairness"]["before"]["fpr_ratio"], res["fairness"]["after"]["fpr_ratio"]
    alarms = res["drift"]["monitoring"]
    rec = res["future"]["xgb"]["recall_budget"]
    fpr = res["future"]["xgb"]["fpr_realized"]
    evidence = [
        (f"Recall at the 5% false-positive budget in the unseen months: {rec['value']:.3f} "
         f"(95% CI {rec['lo']:.3f}-{rec['hi']:.3f}); realised false-positive rate {fpr['value']:.3f}."),
        (f"XGBoost minus logistic regression recall: {diff['value']:+.3f} (95% CI {diff['lo']:+.3f} to "
         f"{diff['hi']:+.3f})."),
        (f"False-positive-rate ratio (age>=50 / <50) with the single threshold: {before['value']:.2f} "
         f"(95% CI {before['lo']:.2f}-{before['hi']:.2f}); group thresholds would bring it to {after['value']:.2f} "
         f"but treat applicants differently by age."),
        "Monitoring rules triggered in the unseen months: "
        + (", ".join(f"{a['rule']} (month {a['month']})" for a in alarms) if alarms else "none") + ".",
    ]
    conditions = []
    if alarms:
        conditions.append("Monthly monitoring with the declared rules; triggered in months 6-7: "
                          + ", ".join(sorted({a["rule"] for a in alarms})) + ".")
    if fpr["lo"] > config.FPR_BUDGET:
        conditions.append(f"The realised false-positive rate ({fpr['value']:.3f}) exceeds the 5% budget: re-set the "
                          "budget threshold every month on the latest labelled month.")
    if not before["lo"] <= 1 <= before["hi"]:
        conditions.append(f"Applicants aged 50+ are flagged {before['value']:.2f}x as often when legitimate: a "
                          "documented fairness and legal review is required before go-live.")
    if diff["hi"] < 0:
        decision_ = "postpone"
    elif conditions:
        decision_ = "approve with conditions"
    else:
        decision_ = "approve"
    return {"decision": decision_, "evidence": evidence, "conditions": conditions}
