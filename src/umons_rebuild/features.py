from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from .geometry import (
    _idx,
    angle_three,
    clean_positions,
    derivative,
    fixed_body_frame,
    lowpass,
    sparc,
)


FEATURE_COLUMNS_V2 = [
    "knee_angle_mean_deg",
    "hip_angle_mean_deg",
    "ankle_angle_mean_deg",
    "shoulder_angle_mean_deg",
    "elbow_angle_mean_deg",
    "trunk_sagittal_mean_deg",
    "trunk_frontal_mean_deg",
    "pelvis_dispersion",
    "pelvis_path_length",
    "bilateral_symmetry_percent",
    "sparc_smoothness",
    "peak_angular_velocity_deg_s",
    "p95_angular_velocity_deg_s",
    "rms_angular_velocity_deg_s",
    "duration_s",
    "speed_proxy",
]

SCALE_SENSITIVE_FEATURES = ["pelvis_dispersion", "pelvis_path_length", "speed_proxy"]
VELOCITY_FEATURES = [
    "peak_angular_velocity_deg_s",
    "p95_angular_velocity_deg_s",
    "rms_angular_velocity_deg_s",
]


def uniform_resample_strict(
    times_s: np.ndarray,
    points_m: np.ndarray,
    target_rate_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an irregular trajectory to a strictly uniform physical-time grid."""
    t = np.asarray(times_s, dtype=float)
    t = t - t[0]
    keep = np.r_[True, np.diff(t) > 1e-9]
    t, p = t[keep], np.asarray(points_m, dtype=float)[keep]
    if len(t) < 2 or t[-1] <= 0:
        return np.array([0.0]), p[:1]
    n = int(np.floor(t[-1] * target_rate_hz)) + 1
    grid = np.arange(n, dtype=float) / target_rate_hz
    if t[-1] - grid[-1] > 0.5 / target_rate_hz:
        grid = np.r_[grid, grid[-1] + 1.0 / target_rate_hz]
    interpolator = interp1d(
        t, p, axis=0, kind="linear", bounds_error=False,
        fill_value=(p[0], p[-1]), assume_sorted=True,
    )
    return grid, interpolator(grid)


def timestamp_aware_kinect(
    timestamps_s: np.ndarray,
    points_m: np.ndarray,
    cutoff_hz: float = 6.0,
    target_rate_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    clean, _ = clean_positions(points_m)
    grid, uniform = uniform_resample_strict(timestamps_s, clean, target_rate_hz)
    return grid, lowpass(uniform, target_rate_hz, cutoff_hz)


def fixed_30hz_kinect(
    points_m: np.ndarray,
    cutoff_hz: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    clean, _ = clean_positions(points_m)
    times = np.arange(len(clean), dtype=float) / 30.0
    return times, lowpass(clean, 30.0, cutoff_hz)


def qualisys_bandwidth_conditions(
    times_s: np.ndarray,
    points_m: np.ndarray,
    common_cutoff_hz: float = 6.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    clean, _ = clean_positions(points_m)
    rate = 1.0 / np.median(np.diff(times_s))
    filtered = lowpass(clean, rate, common_cutoff_hz)
    q1_t, q1_p = uniform_resample_strict(times_s, filtered, 30.0)
    return {
        "q1_bandwidth_matched": (q1_t, q1_p),
        "q2_rate_only_6hz": (times_s - times_s[0], filtered),
        "q3_native_reference": (times_s - times_s[0], clean),
    }


def framewise_trunk_lengths(points_m: np.ndarray, names: list[str]) -> np.ndarray:
    pelvis = points_m[:, _idx(names, "pelvis"), :]
    shoulder = points_m[:, _idx(names, "spine_shoulder"), :]
    values = np.linalg.norm(shoulder - pelvis, axis=1)
    return values[np.isfinite(values) & (values > 0)]


def extract_features_v2(
    times_s: np.ndarray,
    points_m: np.ndarray,
    names: list[str],
    scale_m: float | None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Extract features after unit/time harmonisation using a fixed participant scale."""
    times = np.asarray(times_s, dtype=float)
    times = times - times[0]
    clean, qc = clean_positions(points_m)
    body_m, _, segment_trunk_m = fixed_body_frame(clean, names)
    scale = 1.0 if scale_m is None else float(scale_m)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid fixed scale: {scale}")
    body = body_m / scale

    def point(name: str) -> np.ndarray:
        return body[:, _idx(names, name), :]

    angles: dict[str, list[np.ndarray]] = {key: [] for key in ("knee", "hip", "ankle", "shoulder", "elbow")}
    for side in ("l", "r"):
        angles["knee"].append(angle_three(point(f"hip_{side}"), point(f"knee_{side}"), point(f"ankle_{side}")))
        angles["hip"].append(angle_three(point("pelvis"), point(f"hip_{side}"), point(f"knee_{side}")))
        angles["ankle"].append(angle_three(point(f"knee_{side}"), point(f"ankle_{side}"), point(f"foot_{side}")))
        angles["shoulder"].append(angle_three(point("spine_shoulder"), point(f"shoulder_{side}"), point(f"elbow_{side}")))
        angles["elbow"].append(angle_three(point(f"shoulder_{side}"), point(f"elbow_{side}"), point(f"wrist_{side}")))

    trunk = point("spine_shoulder") - point("pelvis")
    sagittal = np.arctan2(trunk[:, 0], np.maximum(np.abs(trunk[:, 2]), 1e-12))
    frontal = np.arctan2(trunk[:, 1], np.maximum(np.abs(trunk[:, 2]), 1e-12))
    pelvis = point("pelvis")
    pelvis_centered = pelvis - np.nanmean(pelvis, axis=0)
    # Positional SD in a fixed body frame: RMS Euclidean deviation from trajectory mean.
    pelvis_dispersion = float(np.sqrt(np.nanmean(np.sum(pelvis_centered ** 2, axis=1))))
    pelvis_path = float(np.nansum(np.linalg.norm(np.diff(pelvis, axis=0), axis=1)))

    left = np.nanmean(np.stack([angles["knee"][0], angles["hip"][0]], axis=1), axis=1)
    right = np.nanmean(np.stack([angles["knee"][1], angles["hip"][1]], axis=1), axis=1)
    symmetry = 100.0 * np.nanmean(np.abs(left - right) / (0.5 * (np.abs(left) + np.abs(right)) + 1e-9))

    endpoints = ["wrist_l", "wrist_r", "ankle_l", "ankle_r", "pelvis"]
    linear_velocities = [np.linalg.norm(derivative(point(name), times), axis=1) for name in endpoints]
    speed_profile = np.nanmean(np.stack(linear_velocities, axis=1), axis=1)
    pelvis_speed = np.linalg.norm(derivative(point("pelvis"), times), axis=1)
    sample_rate = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 1.0

    angular_velocity_series = []
    for family in ("knee", "hip", "ankle", "shoulder", "elbow"):
        for series in angles[family]:
            angular_velocity_series.append(np.abs(derivative(series, times)))
    angular_velocity = np.degrees(np.stack(angular_velocity_series, axis=1))

    result = {
        "knee_angle_mean_deg": float(np.degrees(np.nanmean(angles["knee"]))),
        "hip_angle_mean_deg": float(np.degrees(np.nanmean(angles["hip"]))),
        "ankle_angle_mean_deg": float(np.degrees(np.nanmean(angles["ankle"]))),
        "shoulder_angle_mean_deg": float(np.degrees(np.nanmean(angles["shoulder"]))),
        "elbow_angle_mean_deg": float(np.degrees(np.nanmean(angles["elbow"]))),
        "trunk_sagittal_mean_deg": float(np.degrees(np.nanmean(np.abs(sagittal)))),
        "trunk_frontal_mean_deg": float(np.degrees(np.nanmean(np.abs(frontal)))),
        "pelvis_dispersion": pelvis_dispersion,
        "pelvis_path_length": pelvis_path,
        "bilateral_symmetry_percent": float(symmetry),
        "sparc_smoothness": sparc(speed_profile, sample_rate),
        "peak_angular_velocity_deg_s": float(np.nanmax(angular_velocity)),
        "p95_angular_velocity_deg_s": float(np.nanpercentile(angular_velocity, 95)),
        "rms_angular_velocity_deg_s": float(np.sqrt(np.nanmean(angular_velocity ** 2))),
        "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "speed_proxy": float(np.nanmedian(speed_profile)),
        "segment_trunk_length_m": segment_trunk_m,
        "scale_used_m": scale,
        **qc,
    }
    traces = {
        "pelvis_speed_profile": pelvis_speed,
        "speed_profile": speed_profile,
        "pelvis": pelvis,
        "body": body,
    }
    return result, traces


def _overlap_pearson_at_lags(q: np.ndarray, k: np.ndarray, lags: np.ndarray) -> np.ndarray:
    correlations = np.full(len(lags), np.nan, dtype=float)
    for index, lag in enumerate(lags):
        if lag < 0:
            x, y = q[:lag], k[-lag:]
        elif lag > 0:
            x, y = q[lag:], k[:-lag]
        else:
            x, y = q, k
        n = len(x)
        if n < 8:
            continue
        sum_x, sum_y = float(np.sum(x)), float(np.sum(y))
        centered_x_ss = float(np.dot(x, x) - sum_x * sum_x / n)
        centered_y_ss = float(np.dot(y, y) - sum_y * sum_y / n)
        denominator = np.sqrt(max(centered_x_ss, 0.0) * max(centered_y_ss, 0.0))
        if denominator > 1e-12:
            correlations[index] = float((np.dot(x, y) - sum_x * sum_y / n) / denominator)
    return correlations


def residual_lag_windows(
    time_q: np.ndarray,
    pelvis_speed_q: np.ndarray,
    time_k: np.ndarray,
    pelvis_speed_k: np.ndarray,
    max_lag_windows_s: tuple[float, ...],
    sample_rate_hz: float = 60.0,
) -> dict[float, dict[str, float | bool]]:
    duration = min(float(time_q[-1] - time_q[0]), float(time_k[-1] - time_k[0]))
    if duration <= 0.25:
        return {
            window: {"best_lag_s": np.nan, "max_correlation": np.nan, "hit_boundary": False}
            for window in max_lag_windows_s
        }
    grid = np.arange(0.0, duration, 1.0 / sample_rate_hz)
    q = np.interp(grid, time_q - time_q[0], pelvis_speed_q)
    k = np.interp(grid, time_k - time_k[0], pelvis_speed_k)
    largest_lag = int(round(max(max_lag_windows_s) * sample_rate_hz))
    all_lags = np.arange(-largest_lag, largest_lag + 1, dtype=int)
    all_correlations = _overlap_pearson_at_lags(q, k, all_lags)
    results: dict[float, dict[str, float | bool]] = {}
    for window in max_lag_windows_s:
        max_lag = int(round(window * sample_rate_hz))
        keep = np.abs(all_lags) <= max_lag
        selected_lags, selected_correlation = all_lags[keep], all_correlations[keep]
        if not np.isfinite(selected_correlation).any():
            results[window] = {"best_lag_s": np.nan, "max_correlation": np.nan, "hit_boundary": False}
            continue
        best_index = int(np.nanargmax(selected_correlation))
        best_lag = int(selected_lags[best_index])
        results[window] = {
            "best_lag_s": float(best_lag / sample_rate_hz),
            "max_correlation": float(selected_correlation[best_index]),
            "hit_boundary": bool(abs(best_lag) == max_lag),
        }
    return results


def residual_lag(
    time_q: np.ndarray,
    pelvis_speed_q: np.ndarray,
    time_k: np.ndarray,
    pelvis_speed_k: np.ndarray,
    max_lag_s: float,
    sample_rate_hz: float = 60.0,
) -> dict[str, float | bool]:
    return residual_lag_windows(
        time_q, pelvis_speed_q, time_k, pelvis_speed_k, (max_lag_s,), sample_rate_hz
    )[max_lag_s]
