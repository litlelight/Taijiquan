from __future__ import annotations

import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import fixed_body_frame, kinect_common_joints, qualisys_common_joints
from .features import (
    FEATURE_COLUMNS_V2,
    extract_features_v2,
    fixed_30hz_kinect,
    framewise_trunk_lengths,
    qualisys_bandwidth_conditions,
    residual_lag,
    timestamp_aware_kinect,
)
from .io import (
    Paths,
    label_segments,
    read_kinect_member,
    read_qualisys_member,
    segment_id_parts,
    zip_member_map,
)


PRIMARY = "primary_timestamp_sensor_scale"


def _feature_row(segment_id: str, sensor: str, condition: str, values: dict) -> dict:
    parts = segment_id_parts(segment_id)
    return {
        "segment_id": segment_id,
        "participant_id": parts["participant"],
        "recording_id": segment_id[:9],
        "gesture_id": parts["gesture"],
        "sensor": sensor,
        "condition": condition,
        **values,
    }


def _collect_participant_scales(paths: Paths, out_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    k_members = zip_member_map(paths.segmented_kinect_zip, ".txt")
    q_members = zip_member_map(paths.segmented_qualisys_zip, ".tsv")
    paired = sorted(set(k_members) & set(q_members))
    values: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    started = time.time()
    with zipfile.ZipFile(paths.segmented_kinect_zip) as kz, zipfile.ZipFile(paths.segmented_qualisys_zip) as qz:
        for index, segment_id in enumerate(paired, 1):
            _, kp = read_kinect_member(kz, k_members[segment_id])
            _, qp, marker_names, _ = read_qualisys_member(qz, q_members[segment_id])
            kj, names = kinect_common_joints(kp)  # Kinect source is millimetres.
            qj, qnames = qualisys_common_joints(qp, marker_names)  # Qualisys source is millimetres.
            if names != qnames:
                raise AssertionError("Common joint order differs")
            participant = segment_id[:3]
            values[(participant, "kinect")].append(framewise_trunk_lengths(kj / 1000.0, names))
            values[(participant, "qualisys")].append(framewise_trunk_lengths(qj / 1000.0, names))
            if index % 100 == 0 or index == len(paired):
                print(f"SCALE_PROGRESS {index}/{len(paired)} elapsed={time.time()-started:.1f}s", flush=True)

    rows, scales = [], {}
    for (participant, sensor), blocks in sorted(values.items()):
        one = np.concatenate(blocks)
        median = float(np.median(one))
        sd = float(np.std(one, ddof=1))
        scales[(participant, sensor)] = median
        rows.append({
            "participant": participant,
            "sensor": sensor,
            "median_trunk_m": median,
            "sd_trunk_m": sd,
            "cv": sd / median,
            "n_valid_frames": int(len(one)),
            "analysis_subset": "1815 strictly paired segments",
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "participant_trunk_scale.csv", index=False)
    wide = table.pivot(index="participant", columns="sensor", values="median_trunk_m").reset_index()
    wide = wide.rename(columns={"kinect": "kinect_trunk_m", "qualisys": "qualisys_trunk_m"})
    wide["ratio_kinect_to_qualisys"] = wide.kinect_trunk_m / wide.qualisys_trunk_m
    wide["kinect_source_unit"] = "mm converted to m"
    wide["qualisys_source_unit"] = "mm converted to m"
    wide.to_csv(out_dir / "coordinate_unit_check.csv", index=False)
    return table, scales


def _timestamp_distribution(paths: Paths, out_dir: Path) -> pd.DataFrame:
    members = zip_member_map(paths.raw_kinect_zip, ".txt")
    rows = []
    with zipfile.ZipFile(paths.raw_kinect_zip) as zf:
        for recording_id in sorted(members):
            timestamps, _ = read_kinect_member(zf, members[recording_id])
            intervals = np.diff(timestamps)
            positive = intervals[intervals > 0]
            rows.append({
                "recording_id": recording_id,
                "participant": recording_id[:3],
                "n_frames": int(len(timestamps)),
                "mean_interval_ms": float(np.mean(positive) * 1000),
                "median_interval_ms": float(np.median(positive) * 1000),
                "interval_sd_ms": float(np.std(positive, ddof=1) * 1000),
                "interval_iqr_ms": float(np.subtract(*np.percentile(positive * 1000, [75, 25]))),
                "effective_fps": float(1.0 / np.mean(positive)),
                "minimum_instantaneous_fps": float(1.0 / np.max(positive)),
                "maximum_instantaneous_fps": float(1.0 / np.min(positive)),
                "dropped_interval_count_gt_50ms": int(np.sum(positive > 0.050)),
                "dropped_interval_rate_gt_50ms": float(np.mean(positive > 0.050)),
                "has_dropped_interval_gt_50ms": bool(np.any(positive > 0.050)),
                "duplicate_or_nonpositive_interval_count": int(np.sum(intervals <= 0)),
            })
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "kinect_timestamp_distribution.csv", index=False)
    return result


def _joint_errors(
    segment_id: str,
    qt: np.ndarray,
    qp_m: np.ndarray,
    kt: np.ndarray,
    kp_m: np.ndarray,
    names: list[str],
) -> list[dict]:
    qb, _, _ = fixed_body_frame(qp_m, names)
    kb, _, _ = fixed_body_frame(kp_m, names)
    duration = min(float(qt[-1] - qt[0]), float(kt[-1] - kt[0]))
    if duration <= 0:
        return []
    grid = np.linspace(0, duration, 100)
    participant = segment_id[:3]
    gesture = segment_id[9:12]
    rows = []
    for joint_index, joint in enumerate(names):
        q = np.column_stack([np.interp(grid, qt, qb[:, joint_index, axis]) for axis in range(3)])
        k = np.column_stack([np.interp(grid, kt, kb[:, joint_index, axis]) for axis in range(3)])
        difference_mm = (k - q) * 1000.0
        rows.append({
            "segment_id": segment_id,
            "participant_id": participant,
            "gesture_id": gesture,
            "joint": joint,
            "position_rmse_mm": float(np.sqrt(np.mean(np.sum(difference_mm ** 2, axis=1)))),
            "bias_x_mm": float(np.mean(difference_mm[:, 0])),
            "bias_y_mm": float(np.mean(difference_mm[:, 1])),
            "bias_z_mm": float(np.mean(difference_mm[:, 2])),
        })
    return rows


def build_v2_cache(paths: Paths, out_dir: Path, seed: int = 20260815) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scale_table, scales = _collect_participant_scales(paths, out_dir)
    timestamp_table = _timestamp_distribution(paths, out_dir)
    k_members = zip_member_map(paths.segmented_kinect_zip, ".txt")
    q_members = zip_member_map(paths.segmented_qualisys_zip, ".tsv")
    paired = sorted(set(k_members) & set(q_members))
    _, label_by_recording = label_segments(paths.labels_zip)
    label_lookup = {
        row["segment_id"]: row
        for rows in label_by_recording.values()
        for row in rows
    }
    feature_rows, qc_rows, lag_rows, joint_rows = [], [], [], []
    started = time.time()

    with zipfile.ZipFile(paths.segmented_kinect_zip) as kz, zipfile.ZipFile(paths.segmented_qualisys_zip) as qz:
        for index, segment_id in enumerate(paired, 1):
            kt_abs, kp = read_kinect_member(kz, k_members[segment_id])
            qt, qp, marker_names, _ = read_qualisys_member(qz, q_members[segment_id])
            kj_mm, names = kinect_common_joints(kp)
            kj_m = kj_mm / 1000.0
            qj_mm, qnames = qualisys_common_joints(qp, marker_names)
            qj_m = qj_mm / 1000.0
            if names != qnames:
                raise AssertionError("Common joint order differs")
            participant = segment_id[:3]
            q_scale = scales[(participant, "qualisys")]
            k_scale = scales[(participant, "kinect")]

            kt, k_timestamp = timestamp_aware_kinect(kt_abs, kj_m)
            kft, k_fixed = fixed_30hz_kinect(kj_m)
            q_conditions = qualisys_bandwidth_conditions(qt, qj_m)
            q1t, q1 = q_conditions["q1_bandwidth_matched"]
            q2t, q2 = q_conditions["q2_rate_only_6hz"]
            q3t, q3 = q_conditions["q3_native_reference"]

            fq_primary, tq = extract_features_v2(q1t, q1, names, q_scale)
            fk_primary, tk = extract_features_v2(kt, k_timestamp, names, k_scale)
            fq_none, _ = extract_features_v2(q1t, q1, names, None)
            fk_none, _ = extract_features_v2(kt, k_timestamp, names, None)
            fq_shared, _ = extract_features_v2(q1t, q1, names, q_scale)
            fk_shared, _ = extract_features_v2(kt, k_timestamp, names, q_scale)
            fk_fixed, _ = extract_features_v2(kft, k_fixed, names, k_scale)
            fq_rate, _ = extract_features_v2(q2t, q2, names, q_scale)
            fq_native, _ = extract_features_v2(q3t, q3, names, q_scale)

            pairs = [
                (PRIMARY, fq_primary, fk_primary),
                ("fixed30_sensor_scale", fq_primary, fk_fixed),
                ("scale_none", fq_none, fk_none),
                ("scale_shared_qualisys", fq_shared, fk_shared),
                ("qualisys_rate_only_6hz", fq_rate, fk_primary),
                ("qualisys_native_reference", fq_native, fk_primary),
            ]
            for condition, fq, fk in pairs:
                feature_rows.append(_feature_row(segment_id, "qualisys", condition, fq))
                feature_rows.append(_feature_row(segment_id, "kinect", condition, fk))

            intervals = np.diff(kt_abs)
            qc_rows.append({
                "segment_id": segment_id,
                "participant_id": participant,
                "recording_id": segment_id[:9],
                "gesture_id": segment_id[9:12],
                "qualisys_frames": int(len(qt)),
                "kinect_frames": int(len(kt_abs)),
                "qualisys_duration_s": fq_primary["duration_s"],
                "kinect_duration_s": fk_primary["duration_s"],
                "duration_difference_s": fk_primary["duration_s"] - fq_primary["duration_s"],
                "mean_interval_ms": float(np.mean(intervals) * 1000),
                "median_interval_ms": float(np.median(intervals) * 1000),
                "interval_sd_ms": float(np.std(intervals, ddof=1) * 1000),
                "interval_iqr_ms": float(np.subtract(*np.percentile(intervals * 1000, [75, 25]))),
                "effective_fps": float(1.0 / np.mean(intervals)),
                "dropped_interval_rate_gt_50ms": float(np.mean(intervals > 0.050)),
            })

            label = label_lookup.get(segment_id, {})
            for window in (0.5, 1.0):
                lag = residual_lag(q1t, tq["pelvis_speed_profile"], kt, tk["pelvis_speed_profile"], window)
                lag_rows.append({
                    "segment_id": segment_id,
                    "participant_id": participant,
                    "recording_id": segment_id[:9],
                    "gesture_id": segment_id[9:12],
                    "segment_start_s": label.get("start_s", np.nan),
                    "search_window_s": window,
                    "best_lag_ms": lag["best_lag_s"] * 1000,
                    "max_cross_correlation": lag["max_correlation"],
                    "hit_search_boundary": lag["hit_boundary"],
                    "correlation_quality": (
                        "poor_lt_0.2" if lag["max_correlation"] < 0.2 else
                        "weak_0.2_to_0.3" if lag["max_correlation"] < 0.3 else
                        "moderate_0.3_to_0.5" if lag["max_correlation"] < 0.5 else "good_ge_0.5"
                    ),
                })
            joint_rows.extend(_joint_errors(segment_id, q1t, q1, kt, k_timestamp, names))

            if index % 50 == 0 or index == len(paired):
                print(f"V2_FEATURE_PROGRESS {index}/{len(paired)} elapsed={time.time()-started:.1f}s", flush=True)

    features = pd.DataFrame(feature_rows)
    qc = pd.DataFrame(qc_rows)
    lags = pd.DataFrame(lag_rows)
    joints = pd.DataFrame(joint_rows)
    features.to_csv(out_dir / "feature_cache_v2.csv.gz", index=False, compression="gzip")
    qc.to_csv(out_dir / "segment_time_qc_v2.csv.gz", index=False, compression="gzip")
    lags.to_csv(out_dir / "pelvis_residual_lag.csv", index=False)
    joints.to_csv(out_dir / "joint_position_error_v2.csv.gz", index=False, compression="gzip")

    primary = features.loc[features.condition == PRIMARY]
    angle_columns = [name for name in FEATURE_COLUMNS_V2 if "angle_mean_deg" in name or "trunk_" in name]
    sanity = {
        "coordinate_units_unified_to_m": True,
        "participant_scales_in_human_range": bool(scale_table.median_trunk_m.between(0.15, 1.0).all()),
        "participant_scale_ratio_no_1000x": bool(
            pd.read_csv(out_dir / "coordinate_unit_check.csv").ratio_kinect_to_qualisys.between(0.25, 4.0).all()
        ),
        "paired_segments": int(primary.segment_id.nunique()),
        "feature_rows": int(len(features)),
        "conditions": sorted(features.condition.unique().tolist()),
        "participant_gesture_units": int(primary[["participant_id", "gesture_id"]].drop_duplicates().shape[0]),
        "joint_angles_in_0_180": bool(((primary[angle_columns] >= 0) & (primary[angle_columns] <= 180)).all().all()),
        "duration_positive_and_below_120s": bool(primary.duration_s.between(0, 120, inclusive="neither").all()),
        "p99_p95_angular_velocity_deg_s": float(primary.p95_angular_velocity_deg_s.quantile(0.99)),
        "peak_angular_velocity_max_deg_s": float(primary.peak_angular_velocity_deg_s.max()),
        "numeric_values_finite": bool(np.isfinite(primary[FEATURE_COLUMNS_V2].to_numpy()).all()),
        "timestamp_sd_unit_is_ms": bool(timestamp_table.interval_sd_ms.median() > 0.1),
        "cache_generation": "fresh_v2_no_legacy_cache_read",
    }
    sanity["gate_a_pass"] = bool(
        sanity["participant_scales_in_human_range"]
        and sanity["participant_scale_ratio_no_1000x"]
        and sanity["paired_segments"] == 1815
        and sanity["participant_gesture_units"] == 156
        and sanity["joint_angles_in_0_180"]
        and sanity["duration_positive_and_below_120s"]
        and sanity["numeric_values_finite"]
        and sanity["timestamp_sd_unit_is_ms"]
    )
    (out_dir / "gate_a_sanity_checks.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    if not sanity["gate_a_pass"]:
        raise AssertionError(f"Gate A failed: {sanity}")
    return {
        "feature_path": str(out_dir / "feature_cache_v2.csv.gz"),
        "qc_path": str(out_dir / "segment_time_qc_v2.csv.gz"),
        "lag_path": str(out_dir / "pelvis_residual_lag.csv"),
        "joint_path": str(out_dir / "joint_position_error_v2.csv.gz"),
        "scale_path": str(out_dir / "participant_trunk_scale.csv"),
        "sanity": sanity,
        "scales": scales,
    }
