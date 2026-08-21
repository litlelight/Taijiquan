from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .agreement import metrics


def _error_agreement_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, feature), one in predictions.groupby(["condition", "feature"], sort=True):
        result = metrics(one.qualisys.to_numpy(), one.prediction.to_numpy())
        rows.append({
            "condition": condition,
            "feature": feature,
            "n_units": int(len(one)),
            "mae": result["mae"],
            "rmse": result["rmse"],
            "icc_a1": result["icc_a1"],
            "icc_c1": result["icc_c1"],
            "bias": result["bias"],
        })
    return pd.DataFrame(rows)


def run_calibration_diagnostics(oof_path: Path, coefficients_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(oof_path)
    uncalibrated = oof.loc[oof.calibrator.eq("uncalibrated")].copy()
    ridge = oof.loc[oof.calibrator.eq("ridge_per_feature")].copy()

    rows = []
    for outer_held in sorted(uncalibrated.participant_id.unique()):
        for feature in sorted(uncalibrated.feature.unique()):
            feature_rows = uncalibrated.loc[uncalibrated.feature.eq(feature)]
            train_mean = float(feature_rows.loc[feature_rows.participant_id.ne(outer_held), "qualisys"].mean())
            held = feature_rows.loc[feature_rows.participant_id.eq(outer_held)]
            for item in held.itertuples(index=False):
                rows.append({
                    "participant_id": item.participant_id,
                    "gesture_id": item.gesture_id,
                    "feature": feature,
                    "qualisys": item.qualisys,
                    "prediction": train_mean,
                    "training_qualisys_mean": train_mean,
                    "training_participant_count": 11,
                    "outer_held_excluded": True,
                })
    intercept = pd.DataFrame(rows)
    intercept.to_csv(out_dir / "r20b_intercept_only_oof_predictions.csv.gz", index=False, compression="gzip")

    comparisons = [
        uncalibrated.rename(columns={"kinect": "prediction"}).assign(condition="uncalibrated_kinect"),
        intercept.assign(condition="intercept_only_loso"),
        ridge.rename(columns={"kinect": "prediction"}).assign(condition="ridge_per_feature"),
    ]
    combined = pd.concat(
        [one[["participant_id", "gesture_id", "feature", "qualisys", "prediction", "condition"]] for one in comparisons],
        ignore_index=True,
    )
    agreement = _error_agreement_rows(combined)
    agreement.to_csv(out_dir / "r20b_three_way_agreement.csv", index=False)
    wide = agreement.pivot(index="feature", columns="condition", values=["mae", "rmse", "icc_a1", "icc_c1"])
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    wide["ridge_rmse_gain_vs_intercept"] = wide.rmse_intercept_only_loso - wide.rmse_ridge_per_feature
    wide["ridge_mae_gain_vs_intercept"] = wide.mae_intercept_only_loso - wide.mae_ridge_per_feature
    wide["ridge_beats_intercept_rmse"] = wide.ridge_rmse_gain_vs_intercept > 0
    wide.to_csv(out_dir / "r20b_feature_comparison.csv", index=False)

    coefficients = pd.read_csv(coefficients_path)
    stability = coefficients.groupby("feature", as_index=False).agg(
        n_outer_folds=("outer_held_participant", "nunique"),
        slope_median=("slope", "median"),
        slope_q1=("slope", lambda x: x.quantile(0.25)),
        slope_q3=("slope", lambda x: x.quantile(0.75)),
        slope_min=("slope", "min"),
        slope_max=("slope", "max"),
        intercept_median=("intercept", "median"),
        intercept_q1=("intercept", lambda x: x.quantile(0.25)),
        intercept_q3=("intercept", lambda x: x.quantile(0.75)),
        intercept_min=("intercept", "min"),
        intercept_max=("intercept", "max"),
        alpha_median=("selected_alpha", "median"),
        alpha_min=("selected_alpha", "min"),
        alpha_max=("selected_alpha", "max"),
        positive_slope_folds=("slope", lambda x: int((x > 0).sum())),
    )
    stability["slope_iqr"] = stability.slope_q3 - stability.slope_q1
    stability["intercept_iqr"] = stability.intercept_q3 - stability.intercept_q1
    stability["positive_slope_percent"] = 100.0 * stability.positive_slope_folds / stability.n_outer_folds
    alpha_counts = (
        coefficients.groupby(["feature", "selected_alpha"]).size().rename("fold_count").reset_index()
    )
    stability.to_csv(out_dir / "r20b_calibration_coefficient_stability.csv", index=False)
    alpha_counts.to_csv(out_dir / "r20b_selected_alpha_counts.csv", index=False)

    checks = {
        "oof_units_per_condition": int(len(uncalibrated)),
        "intercept_oof_rows": int(len(intercept)),
        "features": int(agreement.feature.nunique()),
        "conditions": sorted(agreement.condition.unique().tolist()),
        "outer_held_excluded_all": bool(intercept.outer_held_excluded.all()),
        "coefficient_features": int(stability.feature.nunique()),
        "coefficient_outer_folds_min": int(stability.n_outer_folds.min()),
        "finite_core_metrics": bool(np.isfinite(agreement[["mae", "rmse", "icc_a1", "icc_c1"]]).all().all()),
    }
    checks["r20b_pass"] = bool(
        checks["oof_units_per_condition"] == 2496
        and checks["intercept_oof_rows"] == 2496
        and checks["features"] == 16
        and checks["conditions"] == ["intercept_only_loso", "ridge_per_feature", "uncalibrated_kinect"]
        and checks["outer_held_excluded_all"]
        and checks["coefficient_features"] == 16
        and checks["coefficient_outer_folds_min"] == 12
        and checks["finite_core_metrics"]
    )
    (out_dir / "r20b_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks
