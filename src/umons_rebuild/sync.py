from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .agreement import agreement_rows
from .geometry import kinect_common_joints, qualisys_common_joints
from .features import extract_features_v2, qualisys_bandwidth_conditions, residual_lag_windows, timestamp_aware_kinect
from .gate_b_extensions import LAG_CORE_FEATURES, _interpolate_points, _load_scales, _paired_long, _row
from .gate_b import _lag_and_drift
from .io import Paths, read_kinect_member, read_qualisys_member, segment_id_parts, zip_member_map
from .pipeline import PRIMARY


def run_sync_v3(
    paths: Paths,
    feature_path: Path,
    scale_path: Path,
    old_lag_path: Path,
    out_dir: Path,
    repetitions: int = 2000,
    seed: int = 20260815,
) -> dict:
    """Recompute only residual synchronization with lag-wise overlap Pearson correlation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = _load_scales(scale_path)
    primary = pd.read_csv(feature_path)
    primary = primary.loc[primary.condition.eq(PRIMARY)].copy()
    id_columns = ["segment_id", "participant_id", "gesture_id", "sensor"]
    primary_compact = primary[id_columns + list(primary.columns[primary.columns.isin(LAG_CORE_FEATURES)])].copy()
    old_lags_full = pd.read_csv(old_lag_path)
    segment_starts = (
        old_lags_full.loc[old_lags_full.search_window_s.eq(1.0)]
        .set_index("segment_id").segment_start_s.to_dict()
    )

    k_members = zip_member_map(paths.segmented_kinect_zip, ".txt")
    q_members = zip_member_map(paths.segmented_qualisys_zip, ".tsv")
    paired_segments = sorted(set(k_members) & set(q_members))
    lag_rows: list[dict] = []
    corrected_rows: list[dict] = []
    correction_qc: list[dict] = []
    started = time.time()

    with zipfile.ZipFile(paths.segmented_kinect_zip) as kz, zipfile.ZipFile(paths.segmented_qualisys_zip) as qz:
        for index, segment_id in enumerate(paired_segments, 1):
            kt_abs, kp_mm = read_kinect_member(kz, k_members[segment_id])
            qt, qp_mm, marker_names, _ = read_qualisys_member(qz, q_members[segment_id])
            kj_mm, names = kinect_common_joints(kp_mm)
            qj_mm, qnames = qualisys_common_joints(qp_mm, marker_names)
            if names != qnames:
                raise AssertionError("Common joint order differs")
            participant = segment_id[:3]
            parts = segment_id_parts(segment_id)
            q_t, q_points = qualisys_bandwidth_conditions(qt, qj_mm / 1000.0, 6.0)["q1_bandwidth_matched"]
            k_t, k_points = timestamp_aware_kinect(kt_abs, kj_mm / 1000.0, cutoff_hz=6.0)
            _, q_trace = extract_features_v2(q_t, q_points, names, scales[(participant, "qualisys")])
            _, k_trace = extract_features_v2(k_t, k_points, names, scales[(participant, "kinect")])

            lag_results = residual_lag_windows(
                q_t, q_trace["pelvis_speed_profile"],
                k_t, k_trace["pelvis_speed_profile"],
                (0.5, 1.0),
            )
            one_second = None
            for window in (0.5, 1.0):
                lag = lag_results[window]
                row = {
                    "segment_id": segment_id,
                    "participant_id": parts["participant"],
                    "recording_id": segment_id[:9],
                    "gesture_id": parts["gesture"],
                    "segment_start_s": float(segment_starts.get(segment_id, np.nan)),
                    "search_window_s": window,
                    "best_lag_ms": float(lag["best_lag_s"] * 1000.0),
                    "max_cross_correlation": float(lag["max_correlation"]),
                    "hit_search_boundary": bool(lag["hit_boundary"]),
                    "correlation_method": "lag-wise overlap Pearson",
                }
                lag_rows.append(row)
                if window == 1.0:
                    one_second = row

            if one_second is not None and one_second["max_cross_correlation"] >= 0.5 and not one_second["hit_search_boundary"]:
                lag_s = one_second["best_lag_ms"] / 1000.0
                shifted_k_t = k_t + lag_s
                start = max(float(q_t[0]), float(shifted_k_t[0]))
                stop = min(float(q_t[-1]), float(shifted_k_t[-1]))
                grid = np.arange(start, stop + 1e-9, 1.0 / 30.0)
                if len(grid) >= 8:
                    aligned_q = _interpolate_points(q_t, q_points, grid)
                    aligned_k = _interpolate_points(shifted_k_t, k_points, grid)
                    aligned_time = grid - grid[0]
                    q_values, _ = extract_features_v2(aligned_time, aligned_q, names, scales[(participant, "qualisys")])
                    k_values, _ = extract_features_v2(aligned_time, aligned_k, names, scales[(participant, "kinect")])
                    corrected_rows.append(_row(segment_id, "qualisys", "lag_corrected_v3", q_values))
                    corrected_rows.append(_row(segment_id, "kinect", "lag_corrected_v3", k_values))
                    correction_qc.append({
                        "segment_id": segment_id,
                        "participant_id": participant,
                        "gesture_id": parts["gesture"],
                        "estimated_lag_ms": one_second["best_lag_ms"],
                        "max_cross_correlation": one_second["max_cross_correlation"],
                        "aligned_duration_s": float(aligned_time[-1]),
                        "aligned_frames": int(len(grid)),
                    })

            if index % 100 == 0 or index == len(paired_segments):
                print(f"SYNC_V3_PROGRESS {index}/{len(paired_segments)} elapsed={time.time()-started:.1f}s", flush=True)

    lags = pd.DataFrame(lag_rows)
    lags.to_csv(out_dir / "pelvis_residual_lag_v3.csv", index=False)
    _lag_and_drift(lags, out_dir, repetitions, seed + 1000)

    old = old_lags_full.loc[old_lags_full.search_window_s.eq(1.0), ["segment_id", "best_lag_ms", "max_cross_correlation", "hit_search_boundary"]]
    new = lags.loc[lags.search_window_s.eq(1.0), ["segment_id", "best_lag_ms", "max_cross_correlation", "hit_search_boundary"]]
    comparison = old.merge(new, on="segment_id", suffixes=("_v2_global_n", "_v3_overlap"))
    comparison["lag_change_ms"] = comparison.best_lag_ms_v3_overlap - comparison.best_lag_ms_v2_global_n
    comparison["correlation_change"] = comparison.max_cross_correlation_v3_overlap - comparison.max_cross_correlation_v2_global_n
    comparison.to_csv(out_dir / "lag_v2_v3_segment_comparison.csv", index=False)

    valid = {row["segment_id"] for row in correction_qc}
    before = primary_compact.loc[primary_compact.segment_id.isin(valid)].copy()
    before["condition"] = "high_confidence_before_v3"
    cache = pd.concat([before, pd.DataFrame(corrected_rows)], ignore_index=True)
    cache.to_csv(out_dir / "high_confidence_lag_correction_feature_cache_v3.csv.gz", index=False, compression="gzip")
    pd.DataFrame(correction_qc).to_csv(out_dir / "high_confidence_lag_correction_qc_v3.csv", index=False)
    paired = _paired_long(cache, LAG_CORE_FEATURES)
    agreement = agreement_rows(paired, ["condition", "feature"], repetitions, seed + 2000)
    agreement.to_csv(out_dir / "high_confidence_lag_correction_agreement_v3.csv", index=False)
    before_metrics = agreement.loc[agreement.condition.eq("high_confidence_before_v3")]
    after_metrics = agreement.loc[agreement.condition.eq("lag_corrected_v3")]
    delta = before_metrics.merge(after_metrics, on="feature", suffixes=("_before", "_after"))
    for metric in ["icc_a1", "icc_c1", "pearson_r", "bias", "rmse"]:
        delta[f"{metric}_change_after_minus_before"] = delta[f"{metric}_after"] - delta[f"{metric}_before"]
    delta.to_csv(out_dir / "high_confidence_lag_correction_change_v3.csv", index=False)

    one_second = lags.loc[lags.search_window_s.eq(1.0)]
    checks = {
        "paired_segments": len(paired_segments),
        "lag_rows": len(lags),
        "high_confidence_lag_segments": len(valid),
        "correlation_method": "lag-wise overlap Pearson",
        "median_lag_ms": float(one_second.best_lag_ms.median()),
        "median_correlation": float(one_second.max_cross_correlation.median()),
        "median_absolute_lag_change_v3_minus_v2_ms": float(comparison.lag_change_ms.abs().median()),
        "same_lag_percent": float(100.0 * comparison.lag_change_ms.eq(0).mean()),
        "finite_agreement": bool(np.isfinite(agreement[["icc_a1", "icc_c1", "bias", "rmse"]]).all().all()),
    }
    checks["sync_v3_pass"] = bool(
        checks["paired_segments"] == 1815
        and checks["lag_rows"] == 3630
        and checks["high_confidence_lag_segments"] > 0
        and checks["finite_agreement"]
    )
    (out_dir / "sync_v3_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks
