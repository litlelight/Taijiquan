from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import Paths, parse_metadata
from .pipeline import PRIMARY
from .ranking import (
    PRIMARY_FEATURES,
    _aggregate_paired,
    _fast_nested_ridge_prediction,
    _fit_predict,
    _full_pipeline_permutation,
    _long_loso_ridge,
    _metrics,
    _nested_predictions,
    _precompute_ridge_nested,
    _select_group_ridge_alpha,
    _wide,
)


def _random_segment_split_ridge(
    segments: pd.DataFrame,
    participant_labels: dict[str, float],
    thresholds: tuple[float, float],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ordered = segments.sort_values(["participant_id", "segment_id"]).reset_index(drop=True)
    groups = ordered.participant_id.to_numpy()
    x = ordered[PRIMARY_FEATURES].to_numpy(float)
    y_rows = np.array([participant_labels[participant] for participant in groups], dtype=float)
    metric_rows, prediction_rows = [], []
    for repetition in range(repetitions):
        test_indices = []
        for participant in sorted(participant_labels):
            available = np.flatnonzero(groups == participant)
            n_test = max(1, int(round(0.30 * len(available))))
            test_indices.extend(rng.choice(available, size=n_test, replace=False).tolist())
        test = np.zeros(len(ordered), dtype=bool)
        test[np.asarray(test_indices, dtype=int)] = True
        train = ~test
        alpha = _select_group_ridge_alpha(
            x[train], y_rows[train], groups[train], participant_labels, thresholds, seed + repetition
        )
        raw_prediction = _fit_predict(
            "ridge", {"alpha": alpha}, x[train], y_rows[train], x[test], thresholds, seed + repetition
        )
        held = pd.DataFrame({"participant_id": groups[test], "prediction": raw_prediction})
        participant_prediction = held.groupby("participant_id").prediction.mean().reindex(sorted(participant_labels))
        y = np.array([participant_labels[p] for p in participant_prediction.index], dtype=float)
        metrics = _metrics(y, participant_prediction.to_numpy(), thresholds)
        metric_rows.append({
            "repetition": repetition,
            "selected_alpha": alpha,
            "train_segments": int(train.sum()),
            "test_segments": int(test.sum()),
            **metrics,
        })
        for participant, prediction in participant_prediction.items():
            prediction_rows.append({
                "repetition": repetition,
                "participant_id": participant,
                "true_skill": participant_labels[participant],
                "prediction": float(prediction),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def _distribution_summary(table: pd.DataFrame, metrics: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in metrics:
        result.update({
            f"{metric}_mean": float(table[metric].mean()),
            f"{metric}_median": float(table[metric].median()),
            f"{metric}_p2_5": float(table[metric].quantile(0.025)),
            f"{metric}_p97_5": float(table[metric].quantile(0.975)),
            f"{metric}_min": float(table[metric].min()),
            f"{metric}_max": float(table[metric].max()),
        })
    return result


def _repeated_equalization(
    full_cache: pd.DataFrame,
    paired: pd.DataFrame,
    participant_list: list[str],
    y: np.ndarray,
    thresholds: tuple[float, float],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_q = paired.loc[paired.condition.eq(PRIMARY) & paired.sensor.eq("qualisys")]
    counts = paired_q.groupby(["participant_id", "gesture_id"]).size().to_dict()
    grouped_indices = {
        key: one.index.to_numpy()
        for key, one in full_cache.groupby(["participant_id", "gesture_id"], sort=True)
    }
    rng = np.random.default_rng(seed)
    metric_rows, selection_rows = [], []
    for repetition in range(repetitions):
        selected_indices = []
        for key in sorted(grouped_indices):
            chosen = rng.choice(grouped_indices[key], size=int(counts[key]), replace=False)
            selected_indices.extend(chosen.tolist())
        selected = full_cache.loc[selected_indices]
        long = selected.groupby(["participant_id", "gesture_id"], as_index=False)[PRIMARY_FEATURES].mean()
        matrix, _ = _wide(long, participant_list, PRIMARY_FEATURES)
        prediction, selected_alphas = _fast_nested_ridge_prediction(y, _precompute_ridge_nested(matrix))
        metric_rows.append({
            "repetition": repetition,
            "segment_count": len(selected),
            **_metrics(y, prediction, thresholds),
            "alpha_0_1_count": selected_alphas.count(0.1),
            "alpha_1_count": selected_alphas.count(1.0),
            "alpha_10_count": selected_alphas.count(10.0),
            "alpha_100_count": selected_alphas.count(100.0),
        })
        if repetition < 10:
            selection_rows.extend({"repetition": repetition, "segment_id": value} for value in selected.segment_id)
    return pd.DataFrame(metric_rows), pd.DataFrame(selection_rows)


def run_final_closure(
    paths: Paths,
    paired_feature_path: Path,
    full_qualisys_cache_path: Path,
    out_dir: Path,
    thresholds: tuple[float, float],
    equalization_repetitions: int = 500,
    segment_split_repetitions: int = 200,
    permutation_repetitions: int = 10000,
    seed: int = 20260815,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = parse_metadata(paths.official_repo_root / "Metadata.txt").sort_values("participant_id")
    participants = metadata.participant_id.to_numpy()
    participant_list = participants.tolist()
    y = metadata.skill_mean.to_numpy(float)
    labels = dict(zip(participant_list, y))
    paired = pd.read_csv(paired_feature_path)
    full_cache = pd.read_csv(full_qualisys_cache_path)
    q_long = _aggregate_paired(paired, "qualisys", PRIMARY_FEATURES)
    k_long = _aggregate_paired(paired, "kinect", PRIMARY_FEATURES)
    q_wide, _ = _wide(q_long, participant_list, PRIMARY_FEATURES)
    k_wide, _ = _wide(k_long, participant_list, PRIMARY_FEATURES)

    # Exact segment-level leakage sensitivity.
    segment_summary_rows, segment_metric_parts, segment_prediction_parts = [], [], []
    for sensor in ("qualisys", "kinect"):
        segments = paired.loc[
            paired.condition.eq(PRIMARY) & paired.sensor.eq(sensor),
            ["segment_id", "participant_id", "gesture_id"] + PRIMARY_FEATURES,
        ].copy()
        loso = _long_loso_ridge(segments, PRIMARY_FEATURES, labels, thresholds, seed + (0 if sensor == "qualisys" else 100000))
        loso.to_csv(out_dir / f"segment_level_{sensor}_participant_loso_predictions.csv", index=False)
        loso_metrics = _metrics(loso.true_skill.to_numpy(), loso.prediction.to_numpy(), thresholds)
        random_metrics, random_predictions = _random_segment_split_ridge(
            segments, labels, thresholds, segment_split_repetitions,
            seed + 200000 + (0 if sensor == "qualisys" else 100000),
        )
        random_metrics["sensor"] = sensor
        random_predictions["sensor"] = sensor
        segment_metric_parts.append(random_metrics)
        segment_prediction_parts.append(random_predictions)
        segment_summary_rows.append({
            "sensor": sensor,
            "segments": len(segments),
            **{f"participant_loso_{key}": value for key, value in loso_metrics.items()},
            **_distribution_summary(random_metrics, ["spearman_rho", "mae", "rmse", "qwk"]),
            "random_minus_loso_spearman_mean": float(random_metrics.spearman_rho.mean() - loso_metrics["spearman_rho"]),
            "random_split_repetitions": segment_split_repetitions,
        })
    segment_metrics = pd.concat(segment_metric_parts, ignore_index=True)
    segment_metrics.to_csv(out_dir / "exact_segment_random_split_metrics_distribution.csv", index=False)
    pd.concat(segment_prediction_parts, ignore_index=True).to_csv(
        out_dir / "exact_segment_random_split_predictions.csv.gz", index=False, compression="gzip"
    )
    segment_summary = pd.DataFrame(segment_summary_rows)
    segment_summary.to_csv(out_dir / "exact_segment_random_split_vs_participant_loso.csv", index=False)

    # Repeated Qualisys repetition equalization.
    equalized, selections = _repeated_equalization(
        full_cache, paired, participant_list, y, thresholds, equalization_repetitions, seed + 400000
    )
    equalized.to_csv(out_dir / "equalized_repetition_metrics_distribution.csv", index=False)
    selections.to_csv(out_dir / "equalized_repetition_first10_selections.csv.gz", index=False, compression="gzip")
    paired_prediction, _ = _fast_nested_ridge_prediction(y, _precompute_ridge_nested(q_wide))
    equalized_summary = {
        "repetitions": equalization_repetitions,
        "segments_per_repetition": 1815,
        "paired_only_spearman_rho": _metrics(y, paired_prediction, thresholds)["spearman_rho"],
        **_distribution_summary(equalized, ["spearman_rho", "mae", "rmse", "qwk"]),
    }
    (out_dir / "equalized_repetition_summary.json").write_text(json.dumps(equalized_summary, indent=2), encoding="utf-8")

    # Formal 10,000-repeat full-pipeline permutation.
    permutation_summaries, permutation_nulls, permutation_observed = [], [], []
    for condition, matrix in [("qualisys_ridge", q_wide), ("kinect_ridge", k_wide)]:
        summary, null, observed = _full_pipeline_permutation(
            condition, matrix, y, thresholds, permutation_repetitions,
            seed + 500000 + (0 if condition.startswith("qualisys") else 100000),
        )
        permutation_summaries.append(summary)
        permutation_nulls.append(null)
        observed["condition"] = condition
        observed["participant_id"] = participants
        permutation_observed.append(observed)
    permutation_summary = pd.DataFrame(permutation_summaries)
    order = permutation_summary.two_sided_full_pipeline_p.sort_values().index.tolist()
    holm = np.empty(len(permutation_summary))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * permutation_summary.loc[index, "two_sided_full_pipeline_p"]))
        holm[index] = running
    permutation_summary["holm_p_two_reference_conditions"] = holm
    permutation_summary.to_csv(out_dir / "permutation_10000_summary.csv", index=False)
    pd.concat(permutation_nulls, ignore_index=True).to_csv(
        out_dir / "permutation_10000_null.csv.gz", index=False, compression="gzip"
    )
    pd.concat(permutation_observed, ignore_index=True).to_csv(
        out_dir / "permutation_10000_observed_predictions.csv", index=False
    )

    # Recompute both random-forest conditions with the fold-wise assertion active.
    rf_predictions = pd.concat([
        _nested_predictions("random_forest", "qualisys__primary_wide", q_wide, y, participants, thresholds, seed + 700000),
        _nested_predictions("random_forest", "kinect__primary_wide", k_wide, y, participants, thresholds, seed + 800000),
    ], ignore_index=True)
    rf_predictions.to_csv(out_dir / "rf_predictions_with_training_range_assertion.csv", index=False)
    rf_checks = rf_predictions.groupby("condition", as_index=False).agg(
        folds=("participant_id", "count"),
        all_within_training_range=("prediction_within_training_label_range", "all"),
        minimum_margin_above_lower=("prediction", lambda x: float(np.min(
            x.to_numpy() - rf_predictions.loc[x.index, "training_label_min"].to_numpy()
        ))),
        minimum_margin_below_upper=("prediction", lambda x: float(np.min(
            rf_predictions.loc[x.index, "training_label_max"].to_numpy() - x.to_numpy()
        ))),
    )
    rf_checks.to_csv(out_dir / "rf_training_range_assertion_summary.csv", index=False)

    checks = {
        "exact_segment_sensors": sorted(segment_summary.sensor.tolist()),
        "exact_segment_count_each": sorted(segment_summary.segments.tolist()),
        "segment_random_split_rows": len(segment_metrics),
        "equalization_repetitions": len(equalized),
        "equalization_segments_all_1815": bool(equalized.segment_count.eq(1815).all()),
        "permutation_rows": int(sum(len(one) for one in permutation_nulls)),
        "permutation_repetitions_each": permutation_repetitions,
        "rf_folds": len(rf_predictions),
        "rf_all_within_training_range": bool(rf_predictions.prediction_within_training_label_range.all()),
        "config_permutation_repetitions": permutation_repetitions,
    }
    checks["final_closure_pass"] = bool(
        checks["exact_segment_sensors"] == ["kinect", "qualisys"]
        and checks["exact_segment_count_each"] == [1815, 1815]
        and checks["segment_random_split_rows"] == 2 * segment_split_repetitions
        and checks["equalization_repetitions"] == equalization_repetitions
        and checks["equalization_segments_all_1815"]
        and checks["permutation_rows"] == 2 * permutation_repetitions
        and checks["rf_folds"] == 24
        and checks["rf_all_within_training_range"]
    )
    (out_dir / "final_closure_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks
