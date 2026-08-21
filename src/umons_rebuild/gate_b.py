from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .agreement import agreement_rows, cluster_bootstrap, metrics
from .geometry import kinect_common_joints
from .features import FEATURE_COLUMNS_V2, extract_features_v2, timestamp_aware_kinect
from .io import Paths, label_segments, read_kinect_member, zip_member_map
from .pipeline import PRIMARY


def _paired_long(features: pd.DataFrame, condition: str) -> pd.DataFrame:
    selected = features.loc[features.condition == condition]
    grouped = selected.groupby(["participant_id", "gesture_id", "sensor"], as_index=False)[FEATURE_COLUMNS_V2].mean()
    melted = grouped.melt(
        id_vars=["participant_id", "gesture_id", "sensor"],
        value_vars=FEATURE_COLUMNS_V2,
        var_name="feature", value_name="value",
    )
    return melted.pivot(
        index=["participant_id", "gesture_id", "feature"], columns="sensor", values="value"
    ).reset_index()


def _within_kinect_timestamp_comparison(features: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    identity = ["segment_id", "participant_id", "gesture_id"]
    actual = features.loc[(features.condition == PRIMARY) & (features.sensor == "kinect")]
    fixed = features.loc[(features.condition == "fixed30_sensor_scale") & (features.sensor == "kinect")]
    actual = actual.groupby(["participant_id", "gesture_id"], as_index=False)[FEATURE_COLUMNS_V2].mean()
    fixed = fixed.groupby(["participant_id", "gesture_id"], as_index=False)[FEATURE_COLUMNS_V2].mean()
    rows = []
    for index, feature in enumerate(FEATURE_COLUMNS_V2):
        joined = actual[["participant_id", "gesture_id", feature]].merge(
            fixed[["participant_id", "gesture_id", feature]],
            on=["participant_id", "gesture_id"], suffixes=("_timestamp", "_fixed30"),
        )
        joined = joined.rename(columns={f"{feature}_timestamp": "qualisys", f"{feature}_fixed30": "kinect"})
        result = {"feature": feature, **metrics(joined.qualisys, joined.kinect)}
        result.update(cluster_bootstrap(joined, "qualisys", "kinect", "participant_id", repetitions, seed + index))
        rows.append(result)
    return pd.DataFrame(rows)


def _native_unit_errors(features: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    primary = _paired_long(features, PRIMARY)
    unscaled = _paired_long(features, "scale_none")
    units = {
        "knee_angle_mean_deg": "deg", "hip_angle_mean_deg": "deg",
        "ankle_angle_mean_deg": "deg", "shoulder_angle_mean_deg": "deg",
        "elbow_angle_mean_deg": "deg", "trunk_sagittal_mean_deg": "deg",
        "trunk_frontal_mean_deg": "deg", "pelvis_dispersion": "mm",
        "pelvis_path_length": "mm", "bilateral_symmetry_percent": "%",
        "sparc_smoothness": "dimensionless",
        "peak_angular_velocity_deg_s": "deg/s",
        "p95_angular_velocity_deg_s": "deg/s",
        "rms_angular_velocity_deg_s": "deg/s",
        "duration_s": "s", "speed_proxy": "mm/s",
    }
    rows = []
    for index, feature in enumerate(FEATURE_COLUMNS_V2):
        source = unscaled if feature in {"pelvis_dispersion", "pelvis_path_length", "speed_proxy"} else primary
        one = source.loc[source.feature == feature].copy()
        if units[feature] in {"mm", "mm/s"}:
            one[["qualisys", "kinect"]] *= 1000.0
        result = {"feature": feature, "unit": units[feature], **metrics(one.qualisys, one.kinect)}
        result.update(cluster_bootstrap(one, "qualisys", "kinect", "participant_id", repetitions, seed + index))
        rows.append(result)
    return pd.DataFrame(rows)


def _lag_and_drift(lags: pd.DataFrame, out_dir: Path, repetitions: int, seed: int) -> dict:
    summary_rows = []
    for window, one in lags.groupby("search_window_s"):
        absolute = one.best_lag_ms.abs()
        summary_rows.append({
            "search_window_s": window,
            "n_segments": int(len(one)),
            "median_lag_ms": float(one.best_lag_ms.median()),
            "lag_iqr_lower_ms": float(one.best_lag_ms.quantile(0.25)),
            "lag_iqr_upper_ms": float(one.best_lag_ms.quantile(0.75)),
            "lag_95pct_lower_ms": float(one.best_lag_ms.quantile(0.025)),
            "lag_95pct_upper_ms": float(one.best_lag_ms.quantile(0.975)),
            "absolute_lag_median_ms": float(absolute.median()),
            "hit_boundary_percent": float(100 * one.hit_search_boundary.mean()),
            "corr_lt_0_2_percent": float(100 * (one.max_cross_correlation < 0.2).mean()),
            "corr_lt_0_3_percent": float(100 * (one.max_cross_correlation < 0.3).mean()),
            "corr_ge_0_5_percent": float(100 * (one.max_cross_correlation >= 0.5).mean()),
        })
    lag_summary = pd.DataFrame(summary_rows)
    lag_summary.to_csv(out_dir / "pelvis_residual_lag_quality_summary.csv", index=False)

    one_second = lags.loc[lags.search_window_s == 1.0].dropna(subset=["segment_start_s", "best_lag_ms"])
    drift_rows = []
    for recording, one in one_second.groupby("recording_id"):
        if len(one) < 3 or one.segment_start_s.nunique() < 2:
            continue
        fit = stats.linregress(one.segment_start_s, one.best_lag_ms)
        drift_rows.append({
            "participant_id": one.participant_id.iloc[0],
            "recording_id": recording,
            "n_segments": int(len(one)),
            "slope_ms_per_s": float(fit.slope),
            "intercept_ms": float(fit.intercept),
            "r": float(fit.rvalue),
            "p_value": float(fit.pvalue),
            "slope_se": float(fit.stderr),
        })
    drift = pd.DataFrame(drift_rows)
    drift.to_csv(out_dir / "recording_clock_drift.csv", index=False)
    participant_slopes = drift.groupby("participant_id").slope_ms_per_s.mean()
    rng = np.random.default_rng(seed)
    participants = participant_slopes.index.to_numpy()
    bootstrap = []
    for _ in range(repetitions):
        selected = rng.choice(participants, len(participants), replace=True)
        bootstrap.append(float(np.mean([participant_slopes.loc[p] for p in selected])))
    drift_summary = {
        "recordings_with_estimable_drift": int(len(drift)),
        "participant_clusters": int(len(participant_slopes)),
        "mean_recording_slope_ms_per_s": float(drift.slope_ms_per_s.mean()),
        "median_recording_slope_ms_per_s": float(drift.slope_ms_per_s.median()),
        "participant_equal_weight_mean_slope_ms_per_s": float(participant_slopes.mean()),
        "participant_cluster_bootstrap_ci_lower": float(np.percentile(bootstrap, 2.5)),
        "participant_cluster_bootstrap_ci_upper": float(np.percentile(bootstrap, 97.5)),
    }
    (out_dir / "recording_clock_drift_summary.json").write_text(json.dumps(drift_summary, indent=2), encoding="utf-8")
    return {"lag": lag_summary.to_dict(orient="records"), "drift": drift_summary}


def run_boundary_shift(
    paths: Paths,
    features: pd.DataFrame,
    scales: dict[tuple[str, str], float],
    out_dir: Path,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    _, label_by_recording = label_segments(paths.labels_zip)
    raw_members = zip_member_map(paths.raw_kinect_zip, ".txt")
    q_primary = features.loc[(features.condition == PRIMARY) & (features.sensor == "qualisys")]
    available = set(q_primary.segment_id)
    shifted_rows = []
    with zipfile.ZipFile(paths.raw_kinect_zip) as zf:
        for recording_index, recording_id in enumerate(sorted(raw_members), 1):
            timestamps, points = read_kinect_member(zf, raw_members[recording_id])
            common_mm, names = kinect_common_joints(points)
            common = common_mm / 1000.0
            for segment in label_by_recording[recording_id]:
                segment_id = segment["segment_id"]
                if segment_id not in available or segment["end_s"] is None:
                    continue
                start_target = timestamps[0] + segment["start_s"]
                end_target = timestamps[0] + segment["end_s"]
                beginning = int(np.argmin(np.abs(timestamps - start_target)))
                ending = int(np.argmin(np.abs(timestamps - end_target)))
                for shift in (-2, -1, 0, 1, 2):
                    start = max(0, beginning + shift)
                    stop = min(len(timestamps) - 1, ending + shift)
                    if stop - start + 1 < 8:
                        continue
                    kt, kp = timestamp_aware_kinect(timestamps[start:stop + 1], common[start:stop + 1])
                    values, _ = extract_features_v2(kt, kp, names, scales[(segment_id[:3], "kinect")])
                    shifted_rows.append({
                        "segment_id": segment_id,
                        "participant_id": segment_id[:3],
                        "gesture_id": segment_id[9:12],
                        "shift_frames": shift,
                        **{feature: values[feature] for feature in FEATURE_COLUMNS_V2},
                    })
            if recording_index % 20 == 0:
                print(f"SHIFT_PROGRESS {recording_index}/{len(raw_members)}", flush=True)
    shifted = pd.DataFrame(shifted_rows)
    shifted.to_csv(out_dir / "boundary_shift_feature_cache.csv.gz", index=False, compression="gzip")

    q_agg = q_primary.groupby(["participant_id", "gesture_id"], as_index=False)[FEATURE_COLUMNS_V2].mean()
    comparison_rows = []
    selected_features = [
        "knee_angle_mean_deg", "hip_angle_mean_deg", "elbow_angle_mean_deg",
        "sparc_smoothness", "peak_angular_velocity_deg_s",
        "p95_angular_velocity_deg_s", "rms_angular_velocity_deg_s", "speed_proxy",
    ]
    for shift, group in shifted.groupby("shift_frames"):
        k_agg = group.groupby(["participant_id", "gesture_id"], as_index=False)[selected_features].mean()
        joined = q_agg.merge(k_agg, on=["participant_id", "gesture_id"], suffixes=("_q", "_k"))
        for feature_index, feature in enumerate(selected_features):
            one = joined[["participant_id", f"{feature}_q", f"{feature}_k"]].rename(
                columns={f"{feature}_q": "qualisys", f"{feature}_k": "kinect"}
            )
            result = {"shift_frames": int(shift), "shift_nominal_ms": shift / 30 * 1000, "feature": feature}
            result.update(metrics(one.qualisys, one.kinect))
            result.update(cluster_bootstrap(one, "qualisys", "kinect", "participant_id", repetitions, seed + int(shift) * 100 + feature_index))
            comparison_rows.append(result)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "boundary_shift_agreement_v2.csv", index=False)
    return comparison


def run_gate_b(
    paths: Paths,
    feature_path: Path,
    lag_path: Path,
    joint_path: Path,
    scales: dict[tuple[str, str], float],
    out_dir: Path,
    repetitions: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(feature_path)
    lags = pd.read_csv(lag_path)
    joints = pd.read_csv(joint_path)

    all_agreement = []
    for condition_index, condition in enumerate(sorted(features.condition.unique())):
        paired = _paired_long(features, condition)
        result = agreement_rows(paired, ["feature"], repetitions, seed + condition_index * 1000)
        result.insert(0, "condition", condition)
        all_agreement.append(result)
        print(f"V2_AGREEMENT_PROGRESS {condition_index + 1}/{features.condition.nunique()} {condition}", flush=True)
    agreement = pd.concat(all_agreement, ignore_index=True)
    agreement.to_csv(out_dir / "agreement_all_conditions_v2.csv", index=False)
    agreement.loc[agreement.condition.isin(["scale_none", PRIMARY, "scale_shared_qualisys"])].to_csv(
        out_dir / "scale_normalization_agreement.csv", index=False
    )
    agreement.loc[agreement.condition.isin([PRIMARY, "qualisys_rate_only_6hz", "qualisys_native_reference"])].to_csv(
        out_dir / "qualisys_bandwidth_agreement.csv", index=False
    )
    agreement.loc[agreement.condition == PRIMARY].to_csv(out_dir / "primary_agreement_icc_bias.csv", index=False)

    timestamp_comparison = _within_kinect_timestamp_comparison(features, repetitions, seed + 20000)
    timestamp_comparison.to_csv(out_dir / "timestamp_pipeline_comparison.csv", index=False)
    native_errors = _native_unit_errors(features, repetitions, seed + 30000)
    native_errors.to_csv(out_dir / "native_unit_measurement_errors.csv", index=False)

    primary = _paired_long(features, PRIMARY)
    static = agreement_rows(
        primary.loc[primary.gesture_id.isin(["G01", "G02"])],
        ["gesture_id", "feature"], repetitions, seed + 40000,
    )
    static.to_csv(out_dir / "static_posture_agreement_v2.csv", index=False)
    gesture = agreement_rows(primary, ["gesture_id", "feature"], repetitions, seed + 50000)
    gesture.to_csv(out_dir / "gesture_wise_agreement_v2.csv", index=False)
    joint_features = [name for name in FEATURE_COLUMNS_V2 if "angle_mean_deg" in name or name.startswith("trunk_")]
    joint_agreement = agreement_rows(primary.loc[primary.feature.isin(joint_features)], ["feature"], repetitions, seed + 60000)
    joint_agreement.to_csv(out_dir / "joint_wise_feature_agreement_v2.csv", index=False)
    joint_summary = joints.groupby("joint", as_index=False).agg(
        n_segments=("position_rmse_mm", "size"),
        median_position_rmse_mm=("position_rmse_mm", "median"),
        mean_position_rmse_mm=("position_rmse_mm", "mean"),
        p95_position_rmse_mm=("position_rmse_mm", lambda x: np.percentile(x, 95)),
    )
    joint_summary.to_csv(out_dir / "joint_position_error_summary_v2.csv", index=False)

    lag_drift = _lag_and_drift(lags, out_dir, repetitions, seed + 70000)
    shift = run_boundary_shift(paths, features, scales, out_dir, repetitions, seed + 80000)

    scale_features = {"pelvis_dispersion", "pelvis_path_length", "speed_proxy"}
    required = agreement.loc[(agreement.condition == PRIMARY) & agreement.feature.isin(scale_features)]
    checks = {
        "gate_a_file_pass": bool(json.loads((feature_path.parent / "gate_a_sanity_checks.json").read_text(encoding="utf-8"))["gate_a_pass"]),
        "primary_features_complete": int(agreement.loc[agreement.condition == PRIMARY].feature.nunique()) == len(FEATURE_COLUMNS_V2),
        "icc_a_and_c_present": bool(required[["icc_a1", "icc_c1"]].notna().all().all()),
        "all_primary_bootstrap_ci_present": bool(required[["icc_a1_ci_lower", "icc_a1_ci_upper", "icc_c1_ci_lower", "icc_c1_ci_upper"]].notna().all().all()),
        "bland_altman_midpoint_pass": bool((agreement.loa_midpoint_error < 1e-10).all()),
        "static_gestures": sorted(static.gesture_id.unique().tolist()),
        "gesture_count": int(gesture.gesture_id.nunique()),
        "shift_conditions": sorted(map(int, shift.shift_frames.unique())),
        "lag_search_windows": sorted(lags.search_window_s.unique().tolist()),
        "recording_drift_rows": lag_drift["drift"]["recordings_with_estimable_drift"],
    }
    checks["gate_b_pass"] = bool(
        checks["gate_a_file_pass"]
        and checks["primary_features_complete"]
        and checks["icc_a_and_c_present"]
        and checks["all_primary_bootstrap_ci_present"]
        and checks["bland_altman_midpoint_pass"]
        and checks["static_gestures"] == ["G01", "G02"]
        and checks["gesture_count"] == 13
        and checks["shift_conditions"] == [-2, -1, 0, 1, 2]
        and checks["lag_search_windows"] == [0.5, 1.0]
    )
    (out_dir / "gate_b_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not checks["gate_b_pass"]:
        raise AssertionError(f"Gate B failed: {checks}")
    return checks
