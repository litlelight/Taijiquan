from __future__ import annotations

import json
import math
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .agreement import agreement_rows
from .geometry import kinect_common_joints, qualisys_common_joints
from .features import (
    FEATURE_COLUMNS_V2,
    extract_features_v2,
    framewise_trunk_lengths,
    qualisys_bandwidth_conditions,
    timestamp_aware_kinect,
)
from .io import Paths, read_kinect_member, read_qualisys_member, segment_id_parts, zip_member_map
from .pipeline import PRIMARY


MAPPING_FEATURES = [
    "knee_angle_mean_deg", "hip_angle_mean_deg", "ankle_angle_mean_deg",
    "shoulder_angle_mean_deg", "elbow_angle_mean_deg",
    "trunk_sagittal_mean_deg", "trunk_frontal_mean_deg",
]
LAG_CORE_FEATURES = [
    "knee_angle_mean_deg", "pelvis_dispersion", "pelvis_path_length", "speed_proxy",
    "sparc_smoothness", "peak_angular_velocity_deg_s",
    "p95_angular_velocity_deg_s", "rms_angular_velocity_deg_s",
]
JITTER_FEATURES = [
    "sparc_smoothness", "peak_angular_velocity_deg_s",
    "p95_angular_velocity_deg_s", "rms_angular_velocity_deg_s",
]


def _row(segment_id: str, sensor: str, condition: str, values: dict) -> dict:
    parts = segment_id_parts(segment_id)
    return {
        "segment_id": segment_id,
        "participant_id": parts["participant"],
        "gesture_id": parts["gesture"],
        "sensor": sensor,
        "condition": condition,
        **{feature: values[feature] for feature in FEATURE_COLUMNS_V2},
    }


def _paired_long(cache: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    selected_features = features or FEATURE_COLUMNS_V2
    grouped = cache.groupby(
        ["participant_id", "gesture_id", "condition", "sensor"], as_index=False
    )[selected_features].mean()
    long = grouped.melt(
        id_vars=["participant_id", "gesture_id", "condition", "sensor"],
        value_vars=selected_features, var_name="feature", value_name="value",
    )
    return long.pivot(
        index=["participant_id", "gesture_id", "condition", "feature"],
        columns="sensor", values="value",
    ).reset_index()


def _load_scales(scale_path: Path) -> dict[tuple[str, str], float]:
    table = pd.read_csv(scale_path)
    return {
        (row.participant, row.sensor): float(row.median_trunk_m)
        for row in table.itertuples(index=False)
    }


def _alternative_qualisys_scales(paths: Paths, paired: list[str], out_dir: Path) -> dict[str, float]:
    members = zip_member_map(paths.segmented_qualisys_zip, ".tsv")
    blocks: dict[str, list[np.ndarray]] = defaultdict(list)
    with zipfile.ZipFile(paths.segmented_qualisys_zip) as archive:
        for index, segment_id in enumerate(paired, 1):
            _, points_mm, marker_names, _ = read_qualisys_member(archive, members[segment_id])
            alternative_mm, names = qualisys_common_joints(points_mm, marker_names, alternative=True)
            blocks[segment_id[:3]].append(framewise_trunk_lengths(alternative_mm / 1000.0, names))
            if index % 200 == 0 or index == len(paired):
                print(f"ALT_SCALE_PROGRESS {index}/{len(paired)}", flush=True)
    rows, scales = [], {}
    for participant, participant_blocks in sorted(blocks.items()):
        values = np.concatenate(participant_blocks)
        median = float(np.median(values))
        scales[participant] = median
        rows.append({
            "participant_id": participant,
            "mapping": "alternative",
            "median_trunk_m": median,
            "sd_trunk_m": float(np.std(values, ddof=1)),
            "cv": float(np.std(values, ddof=1) / median),
            "n_valid_frames": int(len(values)),
        })
    pd.DataFrame(rows).to_csv(out_dir / "alternative_mapping_participant_scale.csv", index=False)
    return scales


def _interpolate_points(times: np.ndarray, points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    output = np.empty((len(grid), points.shape[1], 3), dtype=float)
    for joint in range(points.shape[1]):
        for axis in range(3):
            output[:, joint, axis] = np.interp(grid, times, points[:, joint, axis])
    return output


def _duration_summaries(qc_path: Path, out_dir: Path) -> pd.DataFrame:
    qc = pd.read_csv(qc_path)

    def summarize(one: pd.DataFrame, group_type: str, group_value: str) -> dict:
        difference_ms = one.duration_difference_s.to_numpy(float) * 1000.0
        absolute = np.abs(difference_ms)
        return {
            "group_type": group_type,
            "group_value": group_value,
            "n_segments": int(len(one)),
            "mean_difference_ms": float(np.mean(difference_ms)),
            "median_difference_ms": float(np.median(difference_ms)),
            "sd_difference_ms": float(np.std(difference_ms, ddof=1)),
            "iqr_lower_ms": float(np.percentile(difference_ms, 25)),
            "iqr_upper_ms": float(np.percentile(difference_ms, 75)),
            "pct2_5_ms": float(np.percentile(difference_ms, 2.5)),
            "pct97_5_ms": float(np.percentile(difference_ms, 97.5)),
            "median_absolute_difference_ms": float(np.median(absolute)),
            "absolute_gt_16_7ms_percent": float(100 * np.mean(absolute > 16.7)),
            "absolute_gt_33_3ms_percent": float(100 * np.mean(absolute > 33.3)),
            "absolute_gt_33_34ms_percent": float(100 * np.mean(absolute > 33.34)),
            "within_one_nominal_30hz_frame_percent": float(100 * np.mean(absolute <= 33.34)),
            "absolute_gt_50ms_percent": float(100 * np.mean(absolute > 50.0)),
        }

    overall = pd.DataFrame([summarize(qc, "overall", "all")])
    participant = pd.DataFrame([
        summarize(one, "participant", str(key)) for key, one in qc.groupby("participant_id")
    ])
    gesture = pd.DataFrame([
        summarize(one, "gesture", str(key)) for key, one in qc.groupby("gesture_id")
    ])
    overall.to_csv(out_dir / "segment_duration_mismatch_summary.csv", index=False)
    participant.to_csv(out_dir / "segment_duration_mismatch_by_participant.csv", index=False)
    gesture.to_csv(out_dir / "segment_duration_mismatch_by_gesture.csv", index=False)
    return overall


def _jitter_sensitivity(
    primary: pd.DataFrame, out_dir: Path, repetitions: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kinect = primary.loc[primary.sensor.eq("kinect")].sort_values(
        ["jump_outlier_rate", "segment_id"], ascending=[False, True]
    )
    n = len(kinect)
    exclusions = {
        "all_segments": set(),
        "exclude_top_1pct_jump": set(kinect.head(math.ceil(n * 0.01)).segment_id),
        "exclude_top_5pct_jump": set(kinect.head(math.ceil(n * 0.05)).segment_id),
    }
    caches = []
    exclusion_rows = []
    for condition, excluded in exclusions.items():
        one = primary.loc[~primary.segment_id.isin(excluded)].copy()
        one["condition"] = condition
        caches.append(one)
        retained_k = one.loc[one.sensor.eq("kinect")]
        exclusion_rows.append({
            "condition": condition,
            "excluded_segments": int(len(excluded)),
            "retained_segments": int(retained_k.segment_id.nunique()),
            "retained_participant_gesture_units": int(
                retained_k[["participant_id", "gesture_id"]].drop_duplicates().shape[0]
            ),
            "maximum_retained_jump_outlier_rate": float(retained_k.jump_outlier_rate.max()),
        })
    cache = pd.concat(caches, ignore_index=True)
    paired = _paired_long(cache, JITTER_FEATURES)
    agreement = agreement_rows(paired, ["condition", "feature"], repetitions, seed)
    agreement.to_csv(out_dir / "kinect_jitter_exclusion_agreement.csv", index=False)
    pd.DataFrame(exclusion_rows).to_csv(out_dir / "kinect_jitter_exclusion_definition.csv", index=False)

    q = primary.loc[primary.sensor.eq("qualisys"), ["segment_id"] + JITTER_FEATURES]
    k = primary.loc[primary.sensor.eq("kinect"), ["segment_id", "participant_id", "jump_outlier_rate"] + JITTER_FEATURES]
    joined = k.merge(q, on="segment_id", suffixes=("_k", "_q"))
    rng = np.random.default_rng(seed + 10000)
    participants = np.asarray(sorted(joined.participant_id.unique()))
    correlation_rows = []
    for feature in JITTER_FEATURES:
        error = np.abs(joined[f"{feature}_k"] - joined[f"{feature}_q"])
        observed = float(stats.spearmanr(joined.jump_outlier_rate, error).statistic)
        bootstrap = []
        blocks = {p: joined.loc[joined.participant_id.eq(p)] for p in participants}
        for _ in range(repetitions):
            selected = rng.choice(participants, len(participants), replace=True)
            one = pd.concat([blocks[p] for p in selected], ignore_index=True)
            one_error = np.abs(one[f"{feature}_k"] - one[f"{feature}_q"])
            bootstrap.append(float(stats.spearmanr(one.jump_outlier_rate, one_error).statistic))
        correlation_rows.append({
            "feature": feature,
            "n_segments": int(len(joined)),
            "spearman_jump_vs_absolute_error": observed,
            "cluster_bootstrap_ci_lower": float(np.nanpercentile(bootstrap, 2.5)),
            "cluster_bootstrap_ci_upper": float(np.nanpercentile(bootstrap, 97.5)),
        })
    correlation = pd.DataFrame(correlation_rows)
    correlation.to_csv(out_dir / "kinect_jump_velocity_error_association.csv", index=False)
    return agreement, correlation


def run_gate_b_extensions(
    paths: Paths,
    feature_path: Path,
    qc_path: Path,
    lag_path: Path,
    scale_path: Path,
    out_dir: Path,
    repetitions: int = 2000,
    seed: int = 20260815,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(feature_path)
    primary = features.loc[features.condition.eq(PRIMARY)].copy()
    scales = _load_scales(scale_path)
    k_members = zip_member_map(paths.segmented_kinect_zip, ".txt")
    q_members = zip_member_map(paths.segmented_qualisys_zip, ".tsv")
    paired_segments = sorted(set(k_members) & set(q_members))
    alt_scales = _alternative_qualisys_scales(paths, paired_segments, out_dir)

    lags = pd.read_csv(lag_path)
    high_conf = lags.loc[
        lags.search_window_s.eq(1.0)
        & lags.max_cross_correlation.ge(0.5)
        & ~lags.hit_search_boundary.astype(bool)
    ].set_index("segment_id")

    alternative_q_rows: list[dict] = []
    cutoff_rows: list[dict] = []
    lag_after_rows: list[dict] = []
    lag_qc_rows: list[dict] = []
    started = time.time()
    with zipfile.ZipFile(paths.segmented_kinect_zip) as kz, zipfile.ZipFile(paths.segmented_qualisys_zip) as qz:
        for index, segment_id in enumerate(paired_segments, 1):
            kt_abs, kp_mm = read_kinect_member(kz, k_members[segment_id])
            qt, qp_mm, marker_names, _ = read_qualisys_member(qz, q_members[segment_id])
            kj_mm, names = kinect_common_joints(kp_mm)
            qj_mm, qnames = qualisys_common_joints(qp_mm, marker_names, alternative=False)
            q_alt_mm, alt_names = qualisys_common_joints(qp_mm, marker_names, alternative=True)
            if names != qnames or names != alt_names:
                raise AssertionError("Common joint order differs")
            kj_m, qj_m, q_alt_m = kj_mm / 1000.0, qj_mm / 1000.0, q_alt_mm / 1000.0
            participant = segment_id[:3]

            q_alt_t, q_alt = qualisys_bandwidth_conditions(qt, q_alt_m, 6.0)["q1_bandwidth_matched"]
            alt_values, _ = extract_features_v2(q_alt_t, q_alt, names, alt_scales[participant])
            alternative_q_rows.append(_row(segment_id, "qualisys", "alternative_mapping", alt_values))

            cutoff_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            for cutoff in (4.0, 8.0):
                k_t, k_points = timestamp_aware_kinect(kt_abs, kj_m, cutoff_hz=cutoff)
                q_t, q_points = qualisys_bandwidth_conditions(qt, qj_m, cutoff)["q1_bandwidth_matched"]
                q_values, _ = extract_features_v2(q_t, q_points, names, scales[(participant, "qualisys")])
                k_values, _ = extract_features_v2(k_t, k_points, names, scales[(participant, "kinect")])
                condition = f"cutoff_{int(cutoff)}hz"
                cutoff_rows.append(_row(segment_id, "qualisys", condition, q_values))
                cutoff_rows.append(_row(segment_id, "kinect", condition, k_values))
                cutoff_cache[cutoff] = (q_t, q_points, k_t, k_points)

            if segment_id in high_conf.index:
                q_t, q_points = qualisys_bandwidth_conditions(qt, qj_m, 6.0)["q1_bandwidth_matched"]
                k_t, k_points = timestamp_aware_kinect(kt_abs, kj_m, cutoff_hz=6.0)
                lag_s = float(high_conf.loc[segment_id, "best_lag_ms"]) / 1000.0
                shifted_k_t = k_t + lag_s
                start = max(float(q_t[0]), float(shifted_k_t[0]))
                stop = min(float(q_t[-1]), float(shifted_k_t[-1]))
                grid = np.arange(start, stop + 1e-9, 1.0 / 30.0)
                if len(grid) >= 8:
                    aligned_q = _interpolate_points(q_t, q_points, grid)
                    aligned_k = _interpolate_points(shifted_k_t, k_points, grid)
                    aligned_time = grid - grid[0]
                    q_values, _ = extract_features_v2(
                        aligned_time, aligned_q, names, scales[(participant, "qualisys")]
                    )
                    k_values, _ = extract_features_v2(
                        aligned_time, aligned_k, names, scales[(participant, "kinect")]
                    )
                    lag_after_rows.append(_row(segment_id, "qualisys", "lag_corrected", q_values))
                    lag_after_rows.append(_row(segment_id, "kinect", "lag_corrected", k_values))
                    lag_qc_rows.append({
                        "segment_id": segment_id,
                        "participant_id": participant,
                        "gesture_id": segment_id[9:12],
                        "estimated_lag_ms": float(high_conf.loc[segment_id, "best_lag_ms"]),
                        "max_cross_correlation": float(high_conf.loc[segment_id, "max_cross_correlation"]),
                        "aligned_duration_s": float(aligned_time[-1]),
                        "aligned_frames": int(len(grid)),
                    })

            if index % 100 == 0 or index == len(paired_segments):
                print(f"GATE_B_EXTENSION_FEATURE_PROGRESS {index}/{len(paired_segments)} elapsed={time.time()-started:.1f}s", flush=True)

    id_columns = ["segment_id", "participant_id", "gesture_id", "sensor"]
    primary_compact = primary[id_columns + FEATURE_COLUMNS_V2].copy()

    primary_mapping = primary_compact.copy()
    primary_mapping["condition"] = "primary_mapping"
    alternative_k = primary_compact.loc[primary_compact.sensor.eq("kinect")].copy()
    alternative_k["condition"] = "alternative_mapping"
    mapping_cache = pd.concat(
        [primary_mapping, alternative_k, pd.DataFrame(alternative_q_rows)], ignore_index=True
    )
    mapping_cache.to_csv(out_dir / "alternative_mapping_feature_cache.csv.gz", index=False, compression="gzip")
    mapping_paired = _paired_long(mapping_cache, MAPPING_FEATURES)
    mapping_agreement = agreement_rows(
        mapping_paired, ["condition", "feature"], repetitions, seed + 10000
    )
    mapping_agreement.to_csv(out_dir / "alternative_mapping_agreement.csv", index=False)

    cutoff_6 = primary_compact.copy()
    cutoff_6["condition"] = "cutoff_6hz"
    cutoff_cache = pd.concat([cutoff_6, pd.DataFrame(cutoff_rows)], ignore_index=True)
    cutoff_cache.to_csv(out_dir / "cutoff_4_6_8_feature_cache.csv.gz", index=False, compression="gzip")
    cutoff_paired = _paired_long(cutoff_cache)
    cutoff_agreement = agreement_rows(
        cutoff_paired, ["condition", "feature"], repetitions, seed + 20000
    )
    cutoff_agreement.to_csv(out_dir / "cutoff_4_6_8_agreement.csv", index=False)

    valid_lag_segments = set(row["segment_id"] for row in lag_qc_rows)
    lag_before = primary_compact.loc[primary_compact.segment_id.isin(valid_lag_segments)].copy()
    lag_before["condition"] = "high_confidence_before"
    lag_cache = pd.concat([lag_before, pd.DataFrame(lag_after_rows)], ignore_index=True)
    lag_cache.to_csv(out_dir / "high_confidence_lag_correction_feature_cache.csv.gz", index=False, compression="gzip")
    pd.DataFrame(lag_qc_rows).to_csv(out_dir / "high_confidence_lag_correction_qc.csv", index=False)
    lag_paired = _paired_long(lag_cache, LAG_CORE_FEATURES)
    lag_agreement = agreement_rows(
        lag_paired, ["condition", "feature"], repetitions, seed + 30000
    )
    lag_agreement.to_csv(out_dir / "high_confidence_lag_correction_agreement.csv", index=False)
    before = lag_agreement.loc[lag_agreement.condition.eq("high_confidence_before")]
    after = lag_agreement.loc[lag_agreement.condition.eq("lag_corrected")]
    lag_delta = before.merge(after, on="feature", suffixes=("_before", "_after"))
    for metric in ["icc_a1", "icc_c1", "pearson_r", "bias", "rmse"]:
        lag_delta[f"{metric}_change_after_minus_before"] = (
            lag_delta[f"{metric}_after"] - lag_delta[f"{metric}_before"]
        )
    lag_delta.to_csv(out_dir / "high_confidence_lag_correction_change.csv", index=False)

    duration = _duration_summaries(qc_path, out_dir)
    jitter_agreement, jitter_correlation = _jitter_sensitivity(
        primary, out_dir, repetitions, seed + 40000
    )

    checks = {
        "paired_segments": int(len(paired_segments)),
        "mapping_conditions": sorted(mapping_agreement.condition.unique().tolist()),
        "mapping_features": sorted(mapping_agreement.feature.unique().tolist()),
        "cutoff_conditions": sorted(cutoff_agreement.condition.unique().tolist()),
        "cutoff_feature_count": int(cutoff_agreement.feature.nunique()),
        "duration_segments": int(duration.n_segments.iloc[0]),
        "high_confidence_lag_segments": int(len(valid_lag_segments)),
        "lag_conditions": sorted(lag_agreement.condition.unique().tolist()),
        "jitter_conditions": sorted(jitter_agreement.condition.unique().tolist()),
        "jitter_feature_count": int(jitter_agreement.feature.nunique()),
        "finite_mapping_results": bool(np.isfinite(mapping_agreement[["icc_a1", "icc_c1", "bias", "rmse"]]).all().all()),
        "finite_cutoff_results": bool(np.isfinite(cutoff_agreement[["icc_a1", "icc_c1", "bias", "rmse"]]).all().all()),
        "finite_lag_results": bool(np.isfinite(lag_agreement[["icc_a1", "icc_c1", "bias", "rmse"]]).all().all()),
        "finite_jitter_results": bool(np.isfinite(jitter_agreement[["icc_a1", "icc_c1", "bias", "rmse"]]).all().all()),
        "finite_jitter_association": bool(np.isfinite(jitter_correlation.select_dtypes(include=[np.number])).all().all()),
    }
    checks["gate_b_extended_pass"] = bool(
        checks["paired_segments"] == 1815
        and checks["mapping_conditions"] == ["alternative_mapping", "primary_mapping"]
        and checks["mapping_features"] == sorted(MAPPING_FEATURES)
        and checks["cutoff_conditions"] == ["cutoff_4hz", "cutoff_6hz", "cutoff_8hz"]
        and checks["cutoff_feature_count"] == len(FEATURE_COLUMNS_V2)
        and checks["duration_segments"] == 1815
        and checks["high_confidence_lag_segments"] > 0
        and checks["lag_conditions"] == ["high_confidence_before", "lag_corrected"]
        and checks["jitter_conditions"] == ["all_segments", "exclude_top_1pct_jump", "exclude_top_5pct_jump"]
        and checks["jitter_feature_count"] == len(JITTER_FEATURES)
        and checks["finite_mapping_results"]
        and checks["finite_cutoff_results"]
        and checks["finite_lag_results"]
        and checks["finite_jitter_results"]
        and checks["finite_jitter_association"]
    )
    (out_dir / "gate_b_extended_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    if not checks["gate_b_extended_pass"]:
        raise AssertionError(f"Extended Gate B failed: {checks}")
    return checks
