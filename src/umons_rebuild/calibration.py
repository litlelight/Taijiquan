from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .agreement import agreement_rows, metrics
from .features import FEATURE_COLUMNS_V2
from .io import parse_metadata
from .pipeline import PRIMARY


CALIBRATION_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
RANKING_ALPHAS = [0.1, 1.0, 10.0, 100.0]
RANKING_FEATURES = [feature for feature in FEATURE_COLUMNS_V2 if feature != "duration_s"]
FEATURE_UNITS = {
    "knee_angle_mean_deg": "deg", "hip_angle_mean_deg": "deg",
    "ankle_angle_mean_deg": "deg", "shoulder_angle_mean_deg": "deg",
    "elbow_angle_mean_deg": "deg", "trunk_sagittal_mean_deg": "deg",
    "trunk_frontal_mean_deg": "deg", "pelvis_dispersion": "trunk-normalized",
    "pelvis_path_length": "trunk-normalized", "bilateral_symmetry_percent": "%",
    "sparc_smoothness": "dimensionless", "peak_angular_velocity_deg_s": "deg/s",
    "p95_angular_velocity_deg_s": "deg/s", "rms_angular_velocity_deg_s": "deg/s",
    "duration_s": "s", "speed_proxy": "trunk-normalized/s",
}


def _paired_participant_gesture(features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    selected = features.loc[features.condition.eq(PRIMARY)]
    grouped = selected.groupby(
        ["participant_id", "gesture_id", "sensor"], as_index=False
    )[FEATURE_COLUMNS_V2].mean()
    q = grouped.loc[grouped.sensor.eq("qualisys")].set_index(["participant_id", "gesture_id"])[FEATURE_COLUMNS_V2]
    k = grouped.loc[grouped.sensor.eq("kinect")].set_index(["participant_id", "gesture_id"])[FEATURE_COLUMNS_V2]
    q, k = q.sort_index(), k.sort_index()
    if not q.index.equals(k.index):
        raise AssertionError("Kinect and Qualisys participant-gesture grids differ")
    index = q.index.to_frame(index=False)
    return index, k.to_numpy(float), q.to_numpy(float), index.participant_id.to_numpy()


def _fit_univariate_ridge(
    x_train: np.ndarray, y_train: np.ndarray, x_apply: np.ndarray, alpha: float
) -> tuple[np.ndarray, float, float]:
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, float)
    x_apply = np.asarray(x_apply, float)
    mean_x = float(np.mean(x_train))
    scale_x = float(np.std(x_train, ddof=0))
    if scale_x < 1e-12:
        scale_x = 1.0
    z = (x_train - mean_x) / scale_x
    mean_y = float(np.mean(y_train))
    coefficient_z = float(np.sum(z * (y_train - mean_y)) / (np.sum(z ** 2) + alpha))
    slope = coefficient_z / scale_x
    intercept = mean_y - slope * mean_x
    return intercept + slope * x_apply, slope, intercept


def _select_calibration_alpha(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, list[dict]]:
    records = []
    unique = np.asarray(sorted(np.unique(groups)))
    for alpha in CALIBRATION_ALPHAS:
        errors = []
        for held in unique:
            train, test = groups != held, groups == held
            prediction, _, _ = _fit_univariate_ridge(x[train], y[train], x[test], alpha)
            scale = float(np.std(y[train], ddof=1))
            if scale < 1e-12:
                scale = 1.0
            errors.append(float(np.sqrt(np.mean(((prediction - y[test]) / scale) ** 2))))
        records.append({"alpha": alpha, "inner_standardized_rmse": float(np.mean(errors))})
    selected = min(records, key=lambda row: row["inner_standardized_rmse"])["alpha"]
    return float(selected), records


def _calibrate_matrix(
    k_train: np.ndarray,
    q_train: np.ndarray,
    groups_train: np.ndarray,
    k_apply: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[dict]]:
    transformed = np.empty_like(k_apply, dtype=float)
    audit = []
    for feature_index, feature in enumerate(feature_names):
        alpha, records = _select_calibration_alpha(
            k_train[:, feature_index], q_train[:, feature_index], groups_train
        )
        prediction, slope, intercept = _fit_univariate_ridge(
            k_train[:, feature_index], q_train[:, feature_index], k_apply[:, feature_index], alpha
        )
        transformed[:, feature_index] = prediction
        audit.append({
            "feature": feature,
            "selected_alpha": alpha,
            "slope": slope,
            "intercept": intercept,
            "inner_best_standardized_rmse": min(row["inner_standardized_rmse"] for row in records),
        })
    return transformed, audit


def run_per_feature_calibration(
    feature_path: Path,
    out_dir: Path,
    repetitions: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(feature_path)
    index, k, q, groups = _paired_participant_gesture(features)
    participants = np.asarray(sorted(np.unique(groups)))
    calibrated = np.full_like(q, np.nan)
    coefficient_rows, selection_rows = [], []

    for outer_index, held in enumerate(participants):
        train, test = groups != held, groups == held
        for feature_index, feature in enumerate(FEATURE_COLUMNS_V2):
            alpha, records = _select_calibration_alpha(
                k[train, feature_index], q[train, feature_index], groups[train]
            )
            prediction, slope, intercept = _fit_univariate_ridge(
                k[train, feature_index], q[train, feature_index], k[test, feature_index], alpha
            )
            calibrated[test, feature_index] = prediction
            coefficient_rows.append({
                "outer_held_participant": held,
                "feature": feature,
                "selected_alpha": alpha,
                "slope": slope,
                "intercept": intercept,
                "calibration_training_participant_count": int(len(participants) - 1),
                "held_participant_excluded": True,
            })
            for record in records:
                selection_rows.append({
                    "outer_held_participant": held,
                    "feature": feature,
                    "candidate_alpha": record["alpha"],
                    "inner_standardized_rmse": record["inner_standardized_rmse"],
                    "selected": record["alpha"] == alpha,
                })
        print(f"R19_R20_PROGRESS {outer_index + 1}/12 held={held}", flush=True)

    if not np.isfinite(calibrated).all():
        raise AssertionError("Missing per-feature outer-LOSO calibration predictions")

    prediction_rows = []
    for row_index, row in index.iterrows():
        for feature_index, feature in enumerate(FEATURE_COLUMNS_V2):
            prediction_rows.extend([
                {
                    "participant_id": row.participant_id,
                    "gesture_id": row.gesture_id,
                    "feature": feature,
                    "calibrator": "uncalibrated",
                    "qualisys": q[row_index, feature_index],
                    "kinect": k[row_index, feature_index],
                },
                {
                    "participant_id": row.participant_id,
                    "gesture_id": row.gesture_id,
                    "feature": feature,
                    "calibrator": "ridge_per_feature",
                    "qualisys": q[row_index, feature_index],
                    "kinect": calibrated[row_index, feature_index],
                },
            ])
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(out_dir / "r19_r20_per_feature_oof_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(coefficient_rows).to_csv(out_dir / "r19_r20_calibration_coefficients.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out_dir / "r19_r20_calibration_inner_selection.csv", index=False)

    agreement = agreement_rows(
        predictions, ["calibrator", "feature"], repetitions, seed
    )
    agreement.insert(2, "unit", agreement.feature.map(FEATURE_UNITS))
    agreement.to_csv(out_dir / "r19_r20_calibrated_agreement.csv", index=False)

    error_rows = []
    for (calibrator, participant, feature), one in predictions.groupby(
        ["calibrator", "participant_id", "feature"]
    ):
        result = metrics(one.qualisys, one.kinect)
        error_rows.append({
            "calibrator": calibrator,
            "participant_id": participant,
            "feature": feature,
            "unit": FEATURE_UNITS[feature],
            "mae": result["mae"],
            "rmse": result["rmse"],
            "bias": result["bias"],
        })
    errors = pd.DataFrame(error_rows)
    errors.to_csv(out_dir / "r20_outer_loso_feature_errors.csv", index=False)
    before = errors.loc[errors.calibrator.eq("uncalibrated")]
    after = errors.loc[errors.calibrator.eq("ridge_per_feature")]
    changes = before.merge(after, on=["participant_id", "feature", "unit"], suffixes=("_before", "_after"))
    for metric_name in ["mae", "rmse", "bias"]:
        changes[f"{metric_name}_change_after_minus_before"] = (
            changes[f"{metric_name}_after"] - changes[f"{metric_name}_before"]
        )
    q_scale = {
        feature: float(np.std(q[:, feature_index], ddof=1))
        for feature_index, feature in enumerate(FEATURE_COLUMNS_V2)
    }
    changes["qualisys_feature_sd"] = changes.feature.map(q_scale)
    changes["standardized_rmse_before"] = changes.rmse_before / changes.qualisys_feature_sd
    changes["standardized_rmse_after"] = changes.rmse_after / changes.qualisys_feature_sd
    changes["standardized_rmse_improvement"] = (
        changes.standardized_rmse_before - changes.standardized_rmse_after
    )
    changes.to_csv(out_dir / "r20_outer_loso_feature_error_changes.csv", index=False)
    participant_summary = changes.groupby("participant_id", as_index=False).agg(
        mean_standardized_rmse_before=("standardized_rmse_before", "mean"),
        mean_standardized_rmse_after=("standardized_rmse_after", "mean"),
        mean_standardized_rmse_improvement=("standardized_rmse_improvement", "mean"),
        features_with_lower_rmse=("rmse_change_after_minus_before", lambda x: int((x < 0).sum())),
    )
    participant_summary.to_csv(out_dir / "r20_outer_loso_participant_summary.csv", index=False)

    checks = {
        "participant_gesture_units": int(len(index)),
        "participants": int(len(participants)),
        "features": int(len(FEATURE_COLUMNS_V2)),
        "oof_predictions_complete": bool(np.isfinite(calibrated).all()),
        "per_feature_single_input_mapping": True,
        "outer_held_participant_excluded": bool(pd.DataFrame(coefficient_rows).held_participant_excluded.all()),
        "agreement_conditions": sorted(agreement.calibrator.unique().tolist()),
    }
    checks["r19_r20_pass"] = bool(
        checks["participant_gesture_units"] == 156
        and checks["participants"] == 12
        and checks["features"] == len(FEATURE_COLUMNS_V2)
        and checks["oof_predictions_complete"]
        and checks["per_feature_single_input_mapping"]
        and checks["outer_held_participant_excluded"]
        and checks["agreement_conditions"] == ["ridge_per_feature", "uncalibrated"]
    )
    (out_dir / "r19_r20_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not checks["r19_r20_pass"]:
        raise AssertionError(f"R19-R20 failed: {checks}")
    return {"checks": checks, "index": index, "k": k, "q": q, "groups": groups}


def _wide(
    index: pd.DataFrame, values: np.ndarray, participants: list[str], feature_names: list[str]
) -> np.ndarray:
    frame = index.copy()
    for feature_index, feature in enumerate(feature_names):
        frame[feature] = values[:, feature_index]
    wide = frame.pivot(index="participant_id", columns="gesture_id", values=feature_names)
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return wide.reindex(participants).to_numpy(float)


def _ranking_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> np.ndarray:
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


def _band(values: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, thresholds, right=True).astype(int)


def _ranking_metrics(y: np.ndarray, prediction: np.ndarray, thresholds: tuple[float, float]) -> dict:
    return {
        "spearman_rho": float(stats.spearmanr(y, prediction).statistic),
        "kendall_tau_b": float(stats.kendalltau(y, prediction, variant="b").statistic),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "qwk": float(cohen_kappa_score(_band(y, thresholds), _band(prediction, thresholds), weights="quadratic")),
    }


def _simple_nested_ridge_predictions(
    matrix: np.ndarray, participants: np.ndarray, y: np.ndarray, condition: str
) -> tuple[list[dict], list[dict]]:
    predictions, selections = [], []
    for outer_index, held in enumerate(participants):
        train, test = participants != held, participants == held
        inner_losses = []
        for alpha in RANKING_ALPHAS:
            errors = []
            for inner_held in participants[train]:
                inner_train = train & (participants != inner_held)
                inner_test = participants == inner_held
                pred = _ranking_fit_predict(matrix[inner_train], y[inner_train], matrix[inner_test], alpha)
                errors.append(float(abs(pred[0] - y[inner_test][0])))
            inner_losses.append({"alpha": alpha, "inner_mae": float(np.mean(errors))})
        selected = min(inner_losses, key=lambda row: row["inner_mae"])["alpha"]
        pred = _ranking_fit_predict(matrix[train], y[train], matrix[test], selected)
        predictions.append({
            "condition": condition,
            "participant_id": held,
            "true_skill": float(y[test][0]),
            "prediction": float(pred[0]),
            "selected_ranking_alpha": selected,
        })
        for record in inner_losses:
            selections.append({
                "condition": condition,
                "outer_held_participant": held,
                "candidate_ranking_alpha": record["alpha"],
                "inner_mae": record["inner_mae"],
                "selected": record["alpha"] == selected,
            })
    return predictions, selections


def run_r21_nested_reference(
    feature_path: Path,
    metadata_path: Path,
    out_dir: Path,
    thresholds: tuple[float, float],
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(feature_path)
    index_all, k_all, q_all, groups_all = _paired_participant_gesture(features)
    selected_indices = [FEATURE_COLUMNS_V2.index(feature) for feature in RANKING_FEATURES]
    k, q = k_all[:, selected_indices], q_all[:, selected_indices]
    index = index_all.copy()
    metadata = parse_metadata(metadata_path).sort_values("participant_id")
    participants = metadata.participant_id.to_numpy()
    y = metadata.skill_mean.to_numpy(float)
    label_lookup = dict(zip(participants, y))

    raw_wide = _wide(index, k, list(participants), RANKING_FEATURES)
    qualisys_wide = _wide(index, q, list(participants), RANKING_FEATURES)
    prediction_rows, selection_rows = [], []
    for condition, matrix in [("qualisys_ridge_reference", qualisys_wide), ("kinect_ridge_uncalibrated", raw_wide)]:
        pred, selections = _simple_nested_ridge_predictions(matrix, participants, y, condition)
        prediction_rows.extend(pred)
        selection_rows.extend(selections)

    leakage_rows, calibration_audit_rows = [], []
    for outer_index, outer_held in enumerate(participants):
        outer_train_participants = participants[participants != outer_held]
        outer_train_rows = groups_all != outer_held
        outer_test_rows = groups_all == outer_held
        inner_predictions = {alpha: [] for alpha in RANKING_ALPHAS}
        inner_truth = []

        for inner_held in outer_train_participants:
            calibration_participants = outer_train_participants[outer_train_participants != inner_held]
            calibration_train_rows = np.isin(groups_all, calibration_participants)
            inner_validation_rows = groups_all == inner_held
            apply_values = np.vstack([k[calibration_train_rows], k[inner_validation_rows]])
            calibrated_apply, audit = _calibrate_matrix(
                k[calibration_train_rows], q[calibration_train_rows], groups_all[calibration_train_rows],
                apply_values, RANKING_FEATURES,
            )
            apply_index = pd.concat(
                [index.loc[calibration_train_rows], index.loc[inner_validation_rows]], ignore_index=True
            )
            calibrated_train = calibrated_apply[: int(calibration_train_rows.sum())]
            calibrated_validation = calibrated_apply[int(calibration_train_rows.sum()):]
            train_index = apply_index.iloc[: len(calibrated_train)].reset_index(drop=True)
            validation_index = apply_index.iloc[len(calibrated_train):].reset_index(drop=True)
            x_train = _wide(train_index, calibrated_train, list(calibration_participants), RANKING_FEATURES)
            x_validation = _wide(validation_index, calibrated_validation, [inner_held], RANKING_FEATURES)
            y_train = np.asarray([label_lookup[p] for p in calibration_participants], float)
            inner_truth.append(label_lookup[inner_held])
            for alpha in RANKING_ALPHAS:
                inner_predictions[alpha].append(float(_ranking_fit_predict(x_train, y_train, x_validation, alpha)[0]))
            leakage_rows.append({
                "outer_held_participant": outer_held,
                "inner_validation_participant": inner_held,
                "calibration_training_participant_count": int(len(calibration_participants)),
                "calibration_training_participants": "|".join(calibration_participants),
                "inner_validation_excluded_from_calibration": inner_held not in set(calibration_participants),
                "outer_test_excluded_from_calibration": outer_held not in set(calibration_participants),
            })
            for one in audit:
                calibration_audit_rows.append({
                    "scope": "inner_ranking_selection",
                    "outer_held_participant": outer_held,
                    "inner_validation_participant": inner_held,
                    **one,
                })

        inner_truth_array = np.asarray(inner_truth, float)
        ranking_losses = [
            {"alpha": alpha, "inner_mae": float(np.mean(np.abs(np.asarray(values) - inner_truth_array)))}
            for alpha, values in inner_predictions.items()
        ]
        selected_ranking_alpha = min(ranking_losses, key=lambda row: row["inner_mae"])["alpha"]
        for record in ranking_losses:
            selection_rows.append({
                "condition": "calibrated_kinect_nested_ridge",
                "outer_held_participant": outer_held,
                "candidate_ranking_alpha": record["alpha"],
                "inner_mae": record["inner_mae"],
                "selected": record["alpha"] == selected_ranking_alpha,
            })

        apply_values = np.vstack([k[outer_train_rows], k[outer_test_rows]])
        calibrated_all, outer_audit = _calibrate_matrix(
            k[outer_train_rows], q[outer_train_rows], groups_all[outer_train_rows],
            apply_values, RANKING_FEATURES,
        )
        apply_index = pd.concat(
            [index.loc[outer_train_rows], index.loc[outer_test_rows]], ignore_index=True
        )
        calibrated_train = calibrated_all[: int(outer_train_rows.sum())]
        calibrated_test = calibrated_all[int(outer_train_rows.sum()):]
        train_index = apply_index.iloc[: len(calibrated_train)].reset_index(drop=True)
        test_index = apply_index.iloc[len(calibrated_train):].reset_index(drop=True)
        x_train = _wide(train_index, calibrated_train, list(outer_train_participants), RANKING_FEATURES)
        x_test = _wide(test_index, calibrated_test, [outer_held], RANKING_FEATURES)
        y_train = np.asarray([label_lookup[p] for p in outer_train_participants], float)
        prediction = _ranking_fit_predict(x_train, y_train, x_test, selected_ranking_alpha)
        prediction_rows.append({
            "condition": "calibrated_kinect_nested_ridge",
            "participant_id": outer_held,
            "true_skill": label_lookup[outer_held],
            "prediction": float(prediction[0]),
            "selected_ranking_alpha": selected_ranking_alpha,
        })
        for one in outer_audit:
            calibration_audit_rows.append({
                "scope": "outer_final_fit",
                "outer_held_participant": outer_held,
                "inner_validation_participant": "",
                **one,
            })
        print(f"R21_PROGRESS {outer_index + 1}/12 held={outer_held}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(out_dir / "r21_nested_ridge_predictions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out_dir / "r21_ranking_inner_selection.csv", index=False)
    leakage = pd.DataFrame(leakage_rows)
    leakage.to_csv(out_dir / "r21_leakage_audit.csv", index=False)
    pd.DataFrame(calibration_audit_rows).to_csv(out_dir / "r21_calibration_alpha_audit.csv", index=False)

    metric_rows = []
    for condition, one in predictions.groupby("condition"):
        ordered = one.set_index("participant_id").loc[participants]
        metric_rows.append({
            "condition": condition,
            "duration_excluded": True,
            "feature_count_per_gesture": len(RANKING_FEATURES),
            **_ranking_metrics(ordered.true_skill.to_numpy(), ordered.prediction.to_numpy(), thresholds),
        })
    ranking_metrics = pd.DataFrame(metric_rows)
    ranking_metrics.to_csv(out_dir / "r21_nested_ridge_metrics.csv", index=False)
    prediction_wide = predictions.pivot(index="participant_id", columns="condition", values="prediction")
    affine_difference = float(np.max(np.abs(
        prediction_wide["calibrated_kinect_nested_ridge"]
        - prediction_wide["kinect_ridge_uncalibrated"]
    )))

    checks = {
        "conditions": sorted(predictions.condition.unique().tolist()),
        "predictions_per_condition": predictions.groupby("condition").size().to_dict(),
        "duration_excluded": True,
        "feature_count_per_gesture": len(RANKING_FEATURES),
        "inner_audit_rows": int(len(leakage)),
        "all_inner_validation_excluded": bool(leakage.inner_validation_excluded_from_calibration.all()),
        "all_outer_test_excluded": bool(leakage.outer_test_excluded_from_calibration.all()),
        "all_predictions_finite": bool(np.isfinite(predictions.prediction).all()),
        "maximum_calibrated_vs_uncalibrated_prediction_difference": affine_difference,
        "affine_invariance_expected_with_standardized_ridge": True,
    }
    checks["r21_pass"] = bool(
        checks["conditions"] == [
            "calibrated_kinect_nested_ridge", "kinect_ridge_uncalibrated", "qualisys_ridge_reference"
        ]
        and all(value == 12 for value in checks["predictions_per_condition"].values())
        and checks["duration_excluded"]
        and checks["feature_count_per_gesture"] == len(FEATURE_COLUMNS_V2) - 1
        and checks["inner_audit_rows"] == 12 * 11
        and checks["all_inner_validation_excluded"]
        and checks["all_outer_test_excluded"]
        and checks["all_predictions_finite"]
        and checks["maximum_calibrated_vs_uncalibrated_prediction_difference"] < 1e-10
    )
    (out_dir / "r21_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not checks["r21_pass"]:
        raise AssertionError(f"R21 failed: {checks}")
    return checks
