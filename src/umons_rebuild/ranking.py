from __future__ import annotations

import json
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .geometry import kinect_common_joints, qualisys_common_joints
from .features import FEATURE_COLUMNS_V2, extract_features_v2, qualisys_bandwidth_conditions
from .gate_b_extensions import _load_scales, _row
from .io import Paths, parse_metadata, read_qualisys_member, zip_member_map
from .pipeline import PRIMARY


PRIMARY_FEATURES = [
    "knee_angle_mean_deg", "hip_angle_mean_deg", "ankle_angle_mean_deg",
    "shoulder_angle_mean_deg", "elbow_angle_mean_deg",
    "trunk_sagittal_mean_deg", "trunk_frontal_mean_deg",
    "pelvis_dispersion", "pelvis_path_length", "bilateral_symmetry_percent",
    "sparc_smoothness", "p95_angular_velocity_deg_s", "speed_proxy",
]
RMS_SENSITIVITY_FEATURES = [
    feature if feature != "p95_angular_velocity_deg_s" else "rms_angular_velocity_deg_s"
    for feature in PRIMARY_FEATURES
]
DURATION_SENSITIVITY_FEATURES = PRIMARY_FEATURES + ["duration_s"]
MODELS = ["ridge", "elasticnet", "random_forest", "ordinal", "xgboost"]
MODEL_CANDIDATES = {
    "ridge": [{"alpha": value} for value in [0.1, 1.0, 10.0, 100.0]],
    "elasticnet": [
        {"alpha": alpha, "l1_ratio": ratio}
        for alpha, ratio in [(0.01, 0.2), (0.1, 0.5), (1.0, 0.8)]
    ],
    "random_forest": [
        {"n_estimators": 300, "max_depth": depth, "max_features": fraction}
        for depth, fraction in [(3, 0.7), (None, 1.0)]
    ],
    "ordinal": [{"alpha": value} for value in [0.1, 1.0, 10.0]],
    "xgboost": [
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.8},
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8},
    ],
}


def _band(values: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    return np.digitize(values, thresholds, right=True).astype(int)


def _metrics(y: np.ndarray, prediction: np.ndarray, thresholds: tuple[float, float]) -> dict[str, float]:
    return {
        "spearman_rho": float(stats.spearmanr(y, prediction).statistic),
        "kendall_tau_b": float(stats.kendalltau(y, prediction, variant="b").statistic),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "qwk": float(cohen_kappa_score(_band(y, thresholds), _band(prediction, thresholds), weights="quadratic")),
    }


def _bootstrap_rho(y: np.ndarray, prediction: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(y), len(y))
        value = stats.spearmanr(y[selected], prediction[selected]).statistic
        if np.isfinite(value):
            values.append(value)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _fit_predict(name: str, params: dict, x_train, y_train, x_test, thresholds, seed):
    scaler = StandardScaler()
    train = scaler.fit_transform(x_train)
    test = scaler.transform(x_test)
    if name == "ridge":
        prediction = Ridge(**params).fit(train, y_train).predict(test)
    elif name == "elasticnet":
        prediction = ElasticNet(**params, max_iter=20000, random_state=seed).fit(train, y_train).predict(test)
    elif name == "random_forest":
        prediction = RandomForestRegressor(**params, random_state=seed, n_jobs=-1).fit(train, y_train).predict(test)
    elif name == "xgboost":
        from xgboost import XGBRegressor
        prediction = XGBRegressor(
            **params, objective="reg:squarederror", reg_lambda=1.0,
            random_state=seed, n_jobs=1, verbosity=0,
        ).fit(train, y_train).predict(test)
    elif name == "ordinal":
        import mord
        classes = _band(np.asarray(y_train), thresholds)
        model = mord.LogisticAT(**params).fit(train, classes)
        probabilities = model.predict_proba(test)
        overall = float(np.mean(y_train))
        centers = np.array([
            np.mean(np.asarray(y_train)[classes == cls]) if np.any(classes == cls) else overall
            for cls in range(probabilities.shape[1])
        ])
        prediction = probabilities @ centers
    else:
        raise KeyError(name)
    return np.asarray(prediction, dtype=float)


def _assert_rf_prediction_in_training_range(prediction: float, y_train: np.ndarray) -> None:
    lower, upper = float(np.min(y_train)), float(np.max(y_train))
    if not (lower - 1e-12 <= prediction <= upper + 1e-12):
        raise AssertionError(
            f"Random-forest prediction {prediction:.12g} is outside training-label range [{lower:.12g}, {upper:.12g}]"
        )


def _select_model(name, x, y, thresholds, seed):
    records = []
    for candidate_index, params in enumerate(MODEL_CANDIDATES[name]):
        errors = []
        for held in range(len(y)):
            train = np.arange(len(y)) != held
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = _fit_predict(
                    name, params, x[train], y[train], x[[held]], thresholds,
                    seed + candidate_index * 100 + held,
                )[0]
            errors.append(abs(float(prediction) - float(y[held])))
        records.append({"params": params, "inner_mae": float(np.mean(errors))})
    selected = min(records, key=lambda row: row["inner_mae"])
    return selected["params"], records


def _nested_predictions(name, condition, x, y, participants, thresholds, seed):
    rows = []
    for outer_index, held in enumerate(participants):
        test = participants == held
        train = ~test
        params, records = _select_model(name, x[train], y[train], thresholds, seed + outer_index * 1000)
        prediction = _fit_predict(name, params, x[train], y[train], x[test], thresholds, seed + outer_index)[0]
        if name == "random_forest":
            _assert_rf_prediction_in_training_range(float(prediction), y[train])
        rows.append({
            "condition": condition,
            "sensor": condition.split("__", 1)[0],
            "model": name,
            "participant_id": held,
            "true_skill": float(y[test][0]),
            "prediction": float(prediction),
            "selected_params_json": json.dumps(params, sort_keys=True),
            "inner_best_mae": float(min(row["inner_mae"] for row in records)),
            "training_label_min": float(np.min(y[train])),
            "training_label_max": float(np.max(y[train])),
            "prediction_within_training_label_range": bool(
                float(np.min(y[train])) - 1e-12 <= float(prediction) <= float(np.max(y[train])) + 1e-12
            ),
            "outer_held_excluded": True,
        })
    return pd.DataFrame(rows)


def _aggregate_paired(features: pd.DataFrame, sensor: str, columns: list[str]) -> pd.DataFrame:
    selected = features.loc[features.condition.eq(PRIMARY) & features.sensor.eq(sensor)]
    return selected.groupby(["participant_id", "gesture_id"], as_index=False)[columns].mean()


def _wide(long: pd.DataFrame, participants: list[str], columns: list[str]) -> tuple[np.ndarray, list[str]]:
    table = long.pivot(index="participant_id", columns="gesture_id", values=columns)
    table = table.swaplevel(0, 1, axis=1).sort_index(axis=1).reindex(participants)
    names = [f"{gesture}__{feature}" for gesture, feature in table.columns]
    matrix = table.to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Ranking matrix contains missing/non-finite values")
    return matrix, names


def _low_dimensional(long: pd.DataFrame, participants: list[str], columns: list[str]) -> np.ndarray:
    matrix = long.groupby("participant_id")[columns].median().reindex(participants).to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Low-dimensional ranking matrix contains missing/non-finite values")
    return matrix


def build_full_qualisys_cache(paths: Paths, scale_path: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scales = _load_scales(scale_path)
    members = zip_member_map(paths.segmented_qualisys_zip, ".tsv")
    rows = []
    started = time.time()
    with zipfile.ZipFile(paths.segmented_qualisys_zip) as archive:
        for index, segment_id in enumerate(sorted(members), 1):
            times, points_mm, marker_names, _ = read_qualisys_member(archive, members[segment_id])
            common_mm, names = qualisys_common_joints(points_mm, marker_names)
            q_t, q_points = qualisys_bandwidth_conditions(times, common_mm / 1000.0, 6.0)["q1_bandwidth_matched"]
            values, _ = extract_features_v2(q_t, q_points, names, scales[(segment_id[:3], "qualisys")])
            rows.append(_row(segment_id, "qualisys", "full_qualisys_6hz", values))
            if index % 200 == 0 or index == len(members):
                print(f"FULL_Q_PROGRESS {index}/{len(members)} elapsed={time.time()-started:.1f}s", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False, compression="gzip")
    return result


def _prediction_metrics(predictions: pd.DataFrame, thresholds, repetitions, seed):
    rows = []
    for index, ((condition, model), one) in enumerate(predictions.groupby(["condition", "model"], sort=True)):
        ordered = one.sort_values("participant_id")
        result = _metrics(ordered.true_skill.to_numpy(), ordered.prediction.to_numpy(), thresholds)
        lower, upper = _bootstrap_rho(
            ordered.true_skill.to_numpy(), ordered.prediction.to_numpy(), repetitions, seed + index
        )
        rows.append({"condition": condition, "sensor": ordered.sensor.iloc[0], "model": model,
                     **result, "spearman_bootstrap_ci_lower": lower, "spearman_bootstrap_ci_upper": upper})
    return pd.DataFrame(rows)


def _select_group_ridge_alpha(x, y_rows, groups, participant_labels, thresholds, seed):
    records = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        errors = []
        for held in np.unique(groups):
            train, test = groups != held, groups == held
            prediction = _fit_predict("ridge", {"alpha": alpha}, x[train], y_rows[train], x[test], thresholds, seed)
            errors.append(abs(float(np.mean(prediction)) - float(participant_labels[held])))
        records.append((alpha, float(np.mean(errors))))
    return min(records, key=lambda item: item[1])[0]


def _long_loso_ridge(long, columns, participant_labels, thresholds, seed):
    rows = []
    groups = long.participant_id.to_numpy()
    x = long[columns].to_numpy(float)
    y_rows = np.array([participant_labels[p] for p in groups], dtype=float)
    for outer_index, held in enumerate(sorted(participant_labels)):
        train, test = groups != held, groups == held
        alpha = _select_group_ridge_alpha(
            x[train], y_rows[train], groups[train], participant_labels, thresholds, seed + outer_index
        )
        prediction = _fit_predict("ridge", {"alpha": alpha}, x[train], y_rows[train], x[test], thresholds, seed)
        rows.append({"participant_id": held, "true_skill": participant_labels[held],
                     "prediction": float(np.mean(prediction)), "selected_alpha": alpha})
    return pd.DataFrame(rows)


def _random_split_ridge(long, columns, participant_labels, thresholds, repetitions, seed):
    rng = np.random.default_rng(seed)
    metric_rows, prediction_rows = [], []
    ordered = long.sort_values(["participant_id", "gesture_id"]).reset_index(drop=True)
    groups = ordered.participant_id.to_numpy()
    x = ordered[columns].to_numpy(float)
    y_rows = np.array([participant_labels[p] for p in groups], dtype=float)
    for repetition in range(repetitions):
        test_indices = []
        for participant in sorted(participant_labels):
            available = np.flatnonzero(groups == participant)
            test_indices.extend(rng.choice(available, size=4, replace=False).tolist())
        test = np.zeros(len(ordered), dtype=bool)
        test[np.asarray(test_indices, dtype=int)] = True
        train = ~test
        alpha = _select_group_ridge_alpha(
            x[train], y_rows[train], groups[train], participant_labels, thresholds, seed + repetition
        )
        raw_prediction = _fit_predict("ridge", {"alpha": alpha}, x[train], y_rows[train], x[test], thresholds, seed)
        test_rows = pd.DataFrame({"participant_id": groups[test], "prediction": raw_prediction})
        participant_prediction = test_rows.groupby("participant_id").prediction.mean().reindex(sorted(participant_labels))
        y = np.array([participant_labels[p] for p in participant_prediction.index], dtype=float)
        metric_rows.append({"repetition": repetition, "selected_alpha": alpha, **_metrics(y, participant_prediction.to_numpy(), thresholds)})
        for participant, prediction in participant_prediction.items():
            prediction_rows.append({"repetition": repetition, "participant_id": participant,
                                    "true_skill": participant_labels[participant], "prediction": prediction})
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def _ridge_response_weights(x_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    kernel = train @ train.T + alpha * np.eye(len(train))
    h = test @ train.T @ np.linalg.solve(kernel, np.eye(len(train)))
    h = h.reshape(-1)
    return h + (1.0 - h.sum()) / len(train)


def _precompute_ridge_nested(x: np.ndarray):
    alphas = [0.1, 1.0, 10.0, 100.0]
    cache = []
    for outer in range(len(x)):
        outer_train = np.flatnonzero(np.arange(len(x)) != outer)
        one = {"outer": outer, "outer_train": outer_train, "alphas": {}}
        for alpha in alphas:
            inner = []
            for validation in outer_train:
                inner_train = outer_train[outer_train != validation]
                weight = _ridge_response_weights(x[inner_train], x[[validation]], alpha)
                inner.append((validation, inner_train, weight))
            outer_weight = _ridge_response_weights(x[outer_train], x[[outer]], alpha)
            one["alphas"][alpha] = {"inner": inner, "outer_weight": outer_weight}
        cache.append(one)
    return cache


def _fast_nested_ridge_prediction(y: np.ndarray, cache) -> tuple[np.ndarray, list[float]]:
    prediction = np.empty(len(y), dtype=float)
    selected_alphas = []
    for one in cache:
        losses = []
        for alpha, details in one["alphas"].items():
            errors = [
                abs(float(weight @ y[inner_train]) - float(y[validation]))
                for validation, inner_train, weight in details["inner"]
            ]
            losses.append((alpha, float(np.mean(errors))))
        selected = min(losses, key=lambda item: item[1])[0]
        details = one["alphas"][selected]
        prediction[one["outer"]] = float(details["outer_weight"] @ y[one["outer_train"]])
        selected_alphas.append(selected)
    return prediction, selected_alphas


def _full_pipeline_permutation(condition, x, y, thresholds, repetitions, seed):
    cache = _precompute_ridge_nested(x)
    observed_prediction, observed_alphas = _fast_nested_ridge_prediction(y, cache)
    observed = _metrics(y, observed_prediction, thresholds)
    rng = np.random.default_rng(seed)
    null_rows = []
    for repetition in range(repetitions):
        permuted = rng.permutation(y)
        prediction, selected = _fast_nested_ridge_prediction(permuted, cache)
        null_rows.append({
            "condition": condition,
            "repetition": repetition,
            "spearman_rho": stats.spearmanr(permuted, prediction).statistic,
            "selected_alpha_0_1_count": selected.count(0.1),
            "selected_alpha_1_count": selected.count(1.0),
            "selected_alpha_10_count": selected.count(10.0),
            "selected_alpha_100_count": selected.count(100.0),
        })
    null = pd.DataFrame(null_rows)
    p_value = float((1 + (null.spearman_rho.abs() >= abs(observed["spearman_rho"])).sum()) / (repetitions + 1))
    summary = {"condition": condition, **observed, "permutation_repetitions": repetitions,
               "two_sided_full_pipeline_p": p_value,
               "observed_selected_alphas_json": json.dumps(observed_alphas)}
    observed_rows = pd.DataFrame({"participant_index": np.arange(len(y)), "true_skill": y,
                                  "prediction": observed_prediction, "selected_alpha": observed_alphas})
    return summary, null, observed_rows


def _true_jackknife(condition, x, y, participants, thresholds, seed):
    rows = []
    for omitted_index, omitted in enumerate(participants):
        keep = participants != omitted
        sub_x, sub_y, sub_participants = x[keep], y[keep], participants[keep]
        predictions = _nested_predictions(
            "ridge", condition, sub_x, sub_y, sub_participants, thresholds, seed + omitted_index * 10000
        )
        rows.append({"condition": condition, "omitted_participant": omitted,
                     "rerun_participant_count": len(sub_participants), **_metrics(
                         predictions.true_skill.to_numpy(), predictions.prediction.to_numpy(), thresholds
                     )})
    return pd.DataFrame(rows)


def run_ranking_v3(
    paths: Paths,
    paired_feature_path: Path,
    scale_path: Path,
    out_dir: Path,
    thresholds: tuple[float, float],
    bootstrap_repetitions: int = 2000,
    permutation_repetitions: int = 1000,
    random_split_repetitions: int = 200,
    seed: int = 20260815,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = parse_metadata(paths.official_repo_root / "Metadata.txt").sort_values("participant_id")
    participants = metadata.participant_id.to_numpy()
    participant_list = participants.tolist()
    y = metadata.skill_mean.to_numpy(float)
    labels = dict(zip(participant_list, y))
    paired = pd.read_csv(paired_feature_path)
    q_long = _aggregate_paired(paired, "qualisys", FEATURE_COLUMNS_V2)
    k_long = _aggregate_paired(paired, "kinect", FEATURE_COLUMNS_V2)

    q_primary, primary_names = _wide(q_long, participant_list, PRIMARY_FEATURES)
    k_primary, _ = _wide(k_long, participant_list, PRIMARY_FEATURES)

    # R25: fixed five-model by two-sensor exploratory grid.
    r25_prediction_path = out_dir / "r25_nested_loso_predictions_10_conditions.csv"
    r25_metric_path = out_dir / "r25_model_comparison_10_conditions.csv"
    if r25_prediction_path.exists() and r25_metric_path.exists():
        print("R25_REUSE_VALIDATED_EXISTING", flush=True)
        r25_predictions = pd.read_csv(r25_prediction_path)
        r25_metrics = pd.read_csv(r25_metric_path)
    else:
        r25_predictions = []
        for sensor, matrix in [("qualisys", q_primary), ("kinect", k_primary)]:
            for model_index, model in enumerate(MODELS):
                print(f"R25_START {sensor} {model}", flush=True)
                r25_predictions.append(_nested_predictions(
                    model, f"{sensor}__primary_wide", matrix, y, participants,
                    thresholds, seed + model_index * 100000 + (0 if sensor == "qualisys" else 500000),
                ))
        r25_predictions = pd.concat(r25_predictions, ignore_index=True)
        r25_predictions.to_csv(r25_prediction_path, index=False)
        r25_metrics = _prediction_metrics(r25_predictions, thresholds, bootstrap_repetitions, seed + 1000000)
        r25_metrics.to_csv(r25_metric_path, index=False)

    # R22: supplementary feature-set and dimensionality sensitivities.
    sensitivity_predictions = []
    for sensor, long in [("qualisys", q_long), ("kinect", k_long)]:
        primary = q_primary if sensor == "qualisys" else k_primary
        primary_existing = r25_predictions.loc[
            r25_predictions.condition.eq(f"{sensor}__primary_wide") & r25_predictions.model.eq("ridge")
        ].copy()
        sensitivity_predictions.append(primary_existing)
        duration_matrix, _ = _wide(long, participant_list, DURATION_SENSITIVITY_FEATURES)
        rms_matrix, _ = _wide(long, participant_list, RMS_SENSITIVITY_FEATURES)
        low_matrix = _low_dimensional(long, participant_list, PRIMARY_FEATURES)
        for label, matrix in [("duration_sensitivity_wide", duration_matrix),
                              ("rms_velocity_sensitivity_wide", rms_matrix),
                              ("primary_lowdim_median", low_matrix)]:
            sensitivity_predictions.append(_nested_predictions(
                "ridge", f"{sensor}__{label}", matrix, y, participants, thresholds, seed + len(sensitivity_predictions) * 10000
            ))
    r22_predictions = pd.concat(sensitivity_predictions, ignore_index=True)
    r22_predictions.to_csv(out_dir / "r22_feature_set_and_lowdim_predictions.csv", index=False)
    r22_metrics = _prediction_metrics(r22_predictions, thresholds, bootstrap_repetitions, seed + 2000000)
    r22_metrics.to_csv(out_dir / "r22_feature_set_and_lowdim_metrics.csv", index=False)

    # R23: corrected full-Qualisys availability and repetition equalization.
    full_cache = build_full_qualisys_cache(paths, scale_path, out_dir / "full_qualisys_feature_cache_v3.csv.gz")
    full_long = full_cache.groupby(["participant_id", "gesture_id"], as_index=False)[PRIMARY_FEATURES].mean()
    paired_q_segments = paired.loc[paired.condition.eq(PRIMARY) & paired.sensor.eq("qualisys")]
    paired_counts = paired_q_segments.groupby(["participant_id", "gesture_id"]).size().rename("paired_n")
    rng = np.random.default_rng(seed + 3000000)
    equalized_parts = []
    for key, one in full_cache.groupby(["participant_id", "gesture_id"], sort=True):
        n = int(paired_counts.loc[key])
        selected = rng.choice(one.index.to_numpy(), size=n, replace=False)
        equalized_parts.append(full_cache.loc[selected])
    equalized_cache = pd.concat(equalized_parts).sort_values("segment_id")
    equalized_cache.to_csv(out_dir / "r23_qualisys_equalized_segment_selection.csv.gz", index=False, compression="gzip")
    equalized_long = equalized_cache.groupby(["participant_id", "gesture_id"], as_index=False)[PRIMARY_FEATURES].mean()
    r23_predictions = []
    for label, long in [("qualisys_full_2149", full_long), ("qualisys_paired_1815", q_long),
                        ("qualisys_equalized_1815", equalized_long)]:
        matrix, _ = _wide(long, participant_list, PRIMARY_FEATURES)
        r23_predictions.append(_nested_predictions("ridge", label, matrix, y, participants, thresholds, seed + len(r23_predictions) * 10000))
    r23_predictions = pd.concat(r23_predictions, ignore_index=True)
    r23_predictions.to_csv(out_dir / "r23_missingness_predictions.csv", index=False)
    r23_metrics = _prediction_metrics(r23_predictions, thresholds, bootstrap_repetitions, seed + 3100000)
    r23_metrics["segment_count"] = r23_metrics.condition.map({
        "qualisys_full_2149": len(full_cache), "qualisys_paired_1815": len(paired_q_segments),
        "qualisys_equalized_1815": len(equalized_cache),
    })
    r23_metrics.to_csv(out_dir / "r23_missingness_metrics.csv", index=False)

    # R24: identical participant×gesture representation under LOSO vs identity-leaky random splits.
    r24_summary_rows, random_metric_parts, random_prediction_parts = [], [], []
    for sensor, long in [("qualisys", q_long), ("kinect", k_long)]:
        long_primary = long[["participant_id", "gesture_id"] + PRIMARY_FEATURES]
        loso = _long_loso_ridge(long_primary, PRIMARY_FEATURES, labels, thresholds, seed + 4000000)
        loso.to_csv(out_dir / f"r24_{sensor}_long_loso_predictions.csv", index=False)
        loso_metrics = _metrics(loso.true_skill.to_numpy(), loso.prediction.to_numpy(), thresholds)
        random_metrics, random_predictions = _random_split_ridge(
            long_primary, PRIMARY_FEATURES, labels, thresholds, random_split_repetitions,
            seed + 4100000 + (0 if sensor == "qualisys" else 100000),
        )
        random_metrics["sensor"] = sensor
        random_predictions["sensor"] = sensor
        random_metric_parts.append(random_metrics)
        random_prediction_parts.append(random_predictions)
        r24_summary_rows.append({
            "sensor": sensor,
            **{f"participant_loso_{key}": value for key, value in loso_metrics.items()},
            **{f"random_split_mean_{key}": float(random_metrics[key].mean()) for key in ["spearman_rho", "kendall_tau_b", "mae", "rmse", "qwk"]},
            **{f"random_split_p2_5_{key}": float(random_metrics[key].quantile(0.025)) for key in ["spearman_rho", "mae", "rmse", "qwk"]},
            **{f"random_split_p97_5_{key}": float(random_metrics[key].quantile(0.975)) for key in ["spearman_rho", "mae", "rmse", "qwk"]},
            **{f"random_minus_loso_{key}": float(random_metrics[key].mean() - loso_metrics[key]) for key in ["spearman_rho", "mae", "rmse", "qwk"]},
            "random_split_repetitions": random_split_repetitions,
        })
    pd.concat(random_metric_parts, ignore_index=True).to_csv(out_dir / "r24_random_split_metrics_distribution.csv", index=False)
    pd.concat(random_prediction_parts, ignore_index=True).to_csv(out_dir / "r24_random_split_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(r24_summary_rows).to_csv(out_dir / "r24_random_split_vs_participant_loso.csv", index=False)

    # R26: full-pipeline label permutation only for the two Ridge reference conditions.
    permutation_summaries, permutation_nulls, permutation_observed = [], [], []
    for condition, matrix in [("qualisys_ridge", q_primary), ("kinect_ridge", k_primary)]:
        summary, null, observed = _full_pipeline_permutation(
            condition, matrix, y, thresholds, permutation_repetitions,
            seed + 5000000 + (0 if condition.startswith("qualisys") else 100000),
        )
        permutation_summaries.append(summary)
        permutation_nulls.append(null)
        observed["condition"] = condition
        observed["participant_id"] = participants
        permutation_observed.append(observed)
    permutation_summary = pd.DataFrame(permutation_summaries)
    ordered_indices = permutation_summary.two_sided_full_pipeline_p.sort_values().index.tolist()
    holm = np.empty(len(permutation_summary), dtype=float)
    running = 0.0
    for rank, index in enumerate(ordered_indices):
        adjusted = min(1.0, (len(permutation_summary) - rank) * permutation_summary.loc[index, "two_sided_full_pipeline_p"])
        running = max(running, adjusted)
        holm[index] = running
    permutation_summary["holm_p_two_reference_conditions"] = holm
    permutation_summary.to_csv(out_dir / "r26_full_pipeline_permutation_summary.csv", index=False)
    pd.concat(permutation_nulls, ignore_index=True).to_csv(out_dir / "r26_full_pipeline_permutation_null.csv.gz", index=False, compression="gzip")
    pd.concat(permutation_observed, ignore_index=True).to_csv(out_dir / "r26_fast_ridge_observed_predictions.csv", index=False)

    # R27: genuinely rerun the nested pipeline after omitting each participant.
    jackknife = pd.concat([
        _true_jackknife("qualisys_ridge", q_primary, y, participants, thresholds, seed + 6000000),
        _true_jackknife("kinect_ridge", k_primary, y, participants, thresholds, seed + 6100000),
    ], ignore_index=True)
    jackknife.to_csv(out_dir / "r27_true_participant_jackknife.csv", index=False)

    checks = {
        "primary_base_feature_count": len(PRIMARY_FEATURES),
        "primary_wide_predictor_count": len(primary_names),
        "r25_conditions": int(r25_metrics.shape[0]),
        "r25_predictions": int(r25_predictions.shape[0]),
        "r25_sensors": sorted(r25_metrics.sensor.unique().tolist()),
        "r25_models": sorted(r25_metrics.model.unique().tolist()),
        "calibrated_condition_present": bool(r25_metrics.condition.str.contains("calibrated").any()),
        "r22_conditions": int(r22_metrics.shape[0]),
        "full_qualisys_segments": int(len(full_cache)),
        "paired_qualisys_segments": int(len(paired_q_segments)),
        "equalized_qualisys_segments": int(len(equalized_cache)),
        "r24_random_split_rows": int(sum(len(one) for one in random_metric_parts)),
        "r26_null_rows": int(sum(len(one) for one in permutation_nulls)),
        "r27_rows": int(len(jackknife)),
        "r27_rerun_participant_count_all_11": bool(jackknife.rerun_participant_count.eq(11).all()),
        "finite_r25_metrics": bool(np.isfinite(r25_metrics[["spearman_rho", "kendall_tau_b", "mae", "rmse", "qwk"]]).all().all()),
    }
    checks["r22_r27_pass"] = bool(
        checks["primary_base_feature_count"] == 13
        and checks["primary_wide_predictor_count"] == 169
        and checks["r25_conditions"] == 10
        and checks["r25_predictions"] == 120
        and checks["r25_sensors"] == ["kinect", "qualisys"]
        and checks["r25_models"] == sorted(MODELS)
        and not checks["calibrated_condition_present"]
        and checks["r22_conditions"] == 8
        and checks["full_qualisys_segments"] == 2149
        and checks["paired_qualisys_segments"] == 1815
        and checks["equalized_qualisys_segments"] == 1815
        and checks["r24_random_split_rows"] == 2 * random_split_repetitions
        and checks["r26_null_rows"] == 2 * permutation_repetitions
        and checks["r27_rows"] == 24
        and checks["r27_rerun_participant_count_all_11"]
        and checks["finite_r25_metrics"]
    )
    (out_dir / "r22_r27_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks
