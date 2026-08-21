from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt

from .io import COMMON_JOINTS, KINECT_JOINTS


FEATURE_COLUMNS = [
    "knee_angle_mean_deg", "hip_angle_mean_deg", "ankle_angle_mean_deg",
    "shoulder_angle_mean_deg", "elbow_angle_mean_deg",
    "trunk_sagittal_mean_deg", "trunk_frontal_mean_deg",
    "pelvis_dispersion", "pelvis_path_length",
    "bilateral_symmetry_percent", "sparc_smoothness",
    "peak_angular_velocity_deg_s", "duration_s", "speed_proxy",
]


def _idx(names: list[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise KeyError(f"Missing point {name}") from exc


def _mean_points(points: np.ndarray, names: list[str], selected: list[str]) -> np.ndarray:
    return np.nanmean(points[:, [_idx(names, name) for name in selected], :], axis=1)


def kinect_common_joints(points: np.ndarray) -> tuple[np.ndarray, list[str]]:
    indices = [_idx(KINECT_JOINTS, name) for name in COMMON_JOINTS]
    return points[:, indices, :].copy(), list(COMMON_JOINTS)


def qualisys_common_joints(
    markers: np.ndarray, marker_names: list[str], alternative: bool = False
) -> tuple[np.ndarray, list[str]]:
    def one(name: str) -> np.ndarray:
        return markers[:, _idx(marker_names, name), :]

    side = {"l": "L", "r": "R"}
    joints: dict[str, np.ndarray] = {}
    joints["pelvis"] = _mean_points(markers, marker_names, ["L_IAS", "R_IAS", "L_IPS", "R_IPS"])
    if alternative:
        joints["spine_shoulder"] = _mean_points(markers, marker_names, ["CLAV", "CV7", "STRN", "TV10"])
    else:
        joints["spine_shoulder"] = _mean_points(markers, marker_names, ["LAC", "RAC"])
    joints["head"] = _mean_points(markers, marker_names, ["LBHD", "RBHD", "RFHD", "LFHD"])
    for short, prefix in side.items():
        joints[f"shoulder_{short}"] = one(f"{prefix}AC")
        joints[f"elbow_{short}"] = _mean_points(markers, marker_names, [f"{prefix}_HLE", f"{prefix}_HME"])
        joints[f"wrist_{short}"] = (
            one(f"{prefix}_RSP") if alternative else
            _mean_points(markers, marker_names, [f"{prefix}_RSP", f"{prefix}_USP"])
        )
        joints[f"hip_{short}"] = (
            _mean_points(markers, marker_names, [f"{prefix}_IAS", f"{prefix}_IPS"])
            if alternative else one(f"{prefix}_FTC")
        )
        joints[f"knee_{short}"] = _mean_points(markers, marker_names, [f"{prefix}_FLE", f"{prefix}_FME"])
        joints[f"ankle_{short}"] = (
            one(f"{prefix}_FAL") if alternative else
            _mean_points(markers, marker_names, [f"{prefix}_FAL", f"{prefix}_TAM"])
        )
        joints[f"foot_{short}"] = (
            one(f"{prefix}_FM2") if alternative else
            _mean_points(markers, marker_names, [f"{prefix}_FM1", f"{prefix}_FM2", f"{prefix}_FM5"])
        )
    stacked = np.stack([joints[name] for name in COMMON_JOINTS], axis=1)
    return stacked, list(COMMON_JOINTS)


def clean_positions(points: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    result = points.astype(np.float64, copy=True)
    nonfinite = ~np.isfinite(result)
    missing_rate = float(nonfinite.mean())
    for joint in range(result.shape[1]):
        for axis in range(3):
            x = result[:, joint, axis]
            good = np.isfinite(x)
            if good.all():
                continue
            if not good.any():
                x[:] = 0.0
            else:
                idx = np.arange(len(x))
                x[~good] = np.interp(idx[~good], idx[good], x[good])
    jumps = np.linalg.norm(np.diff(result, axis=0), axis=2)
    med = np.nanmedian(jumps, axis=0)
    mad = np.nanmedian(np.abs(jumps - med), axis=0) + 1e-9
    outlier_rate = float((jumps > med + 10.0 * mad).mean()) if jumps.size else 0.0
    return result, {"missing_rate": missing_rate, "jump_outlier_rate": outlier_rate}


def lowpass(points: np.ndarray, sample_rate: float, cutoff_hz: float) -> np.ndarray:
    if len(points) < 16 or cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return points.copy()
    sos = butter(4, cutoff_hz / (sample_rate / 2.0), btype="low", output="sos")
    flat = points.reshape(len(points), -1)
    try:
        filtered = sosfiltfilt(sos, flat, axis=0)
    except ValueError:
        return points.copy()
    return filtered.reshape(points.shape)


def resample_uniform(times: np.ndarray, points: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    t = t - t[0]
    keep = np.r_[True, np.diff(t) > 0]
    t = t[keep]
    p = points[keep]
    if len(t) < 2 or t[-1] <= 0:
        return np.array([0.0]), p[:1]
    new_t = np.arange(0.0, t[-1] + 0.5 / sample_rate, 1.0 / sample_rate)
    f = interp1d(t, p, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
    return new_t, f(new_t)


def bandwidth_match_qualisys(
    times: np.ndarray, points: np.ndarray, cutoff_hz: float = 6.0, target_rate: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    sample_rate = 1.0 / np.median(np.diff(times))
    filtered = lowpass(points, sample_rate, cutoff_hz)
    return resample_uniform(times, filtered, target_rate)


def fixed_body_frame(points: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    pelvis = points[:, _idx(names, "pelvis"), :]
    hip_l = points[:, _idx(names, "hip_l"), :]
    hip_r = points[:, _idx(names, "hip_r"), :]
    shoulder = points[:, _idx(names, "spine_shoulder"), :]
    n_ref = max(1, min(len(points), int(round(len(points) * 0.1))))
    origin = np.nanmedian(pelvis[:n_ref], axis=0)
    lateral = np.nanmedian(hip_l[:n_ref] - hip_r[:n_ref], axis=0)
    lateral[2] = 0.0
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.array([0.0, 1.0, 0.0])
    lateral = lateral / np.linalg.norm(lateral)
    vertical = np.array([0.0, 0.0, 1.0])
    if np.nanmedian(shoulder[:n_ref, 2] - pelvis[:n_ref, 2]) < 0:
        vertical *= -1.0
    forward = np.cross(lateral, vertical)
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    lateral = np.cross(vertical, forward)
    rotation = np.stack([forward, lateral, vertical], axis=1)
    transformed = (points - origin) @ rotation
    trunk_length = float(np.nanmedian(np.linalg.norm(shoulder - pelvis, axis=1)))
    return transformed, rotation, trunk_length


def angle_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u = a - b
    v = c - b
    denom = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    cosine = np.sum(u * v, axis=1) / np.maximum(denom, 1e-12)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, times, axis=0, edge_order=1)


def sparc(speed: np.ndarray, sample_rate: float, amp_threshold: float = 0.05, max_freq: float = 20.0) -> float:
    v = np.asarray(speed, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 4 or np.max(np.abs(v)) <= 1e-12:
        return np.nan
    magnitude = np.abs(np.fft.rfft(v))
    if magnitude[0] <= 1e-12:
        magnitude = magnitude / np.max(magnitude)
    else:
        magnitude = magnitude / magnitude[0]
    freqs = np.fft.rfftfreq(len(v), d=1.0 / sample_rate)
    eligible = np.where((freqs <= max_freq) & (magnitude >= amp_threshold))[0]
    # Very smooth, strongly DC-dominated profiles can have no non-zero bin above
    # the amplitude threshold.  Use the first resolvable non-zero frequency in
    # that case so a valid moving trial remains finite and reproducible.
    cutoff_idx = max(1, int(eligible[-1]) if len(eligible) else 1)
    cutoff = float(freqs[cutoff_idx])
    if cutoff <= 0:
        return np.nan
    f_norm = freqs[: cutoff_idx + 1] / cutoff
    m = magnitude[: cutoff_idx + 1]
    return -float(np.sum(np.sqrt(np.diff(f_norm) ** 2 + np.diff(m) ** 2)))


def extract_features(
    times: np.ndarray,
    points: np.ndarray,
    names: list[str],
    scale_mode: str = "sensor",
    shared_scale: float | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    times = np.asarray(times, dtype=float)
    times = times - times[0]
    clean, qc = clean_positions(points)
    body, _, trunk_length = fixed_body_frame(clean, names)
    if scale_mode == "none":
        scale = 1.0
    elif scale_mode == "shared":
        if shared_scale is None or shared_scale <= 0:
            raise ValueError("shared_scale must be supplied for shared scale mode")
        scale = float(shared_scale)
    elif scale_mode == "sensor":
        scale = max(trunk_length, 1e-9)
    else:
        raise ValueError(f"Unknown scale mode: {scale_mode}")
    scaled = body / scale

    def p(name: str) -> np.ndarray:
        return scaled[:, _idx(names, name), :]

    angles: dict[str, list[np.ndarray]] = {k: [] for k in ("knee", "hip", "ankle", "shoulder", "elbow")}
    for s in ("l", "r"):
        angles["knee"].append(angle_three(p(f"hip_{s}"), p(f"knee_{s}"), p(f"ankle_{s}")))
        angles["hip"].append(angle_three(p("pelvis"), p(f"hip_{s}"), p(f"knee_{s}")))
        angles["ankle"].append(angle_three(p(f"knee_{s}"), p(f"ankle_{s}"), p(f"foot_{s}")))
        angles["shoulder"].append(angle_three(p("spine_shoulder"), p(f"shoulder_{s}"), p(f"elbow_{s}")))
        angles["elbow"].append(angle_three(p(f"shoulder_{s}"), p(f"elbow_{s}"), p(f"wrist_{s}")))

    trunk = p("spine_shoulder") - p("pelvis")
    sagittal = np.arctan2(trunk[:, 0], np.maximum(np.abs(trunk[:, 2]), 1e-12))
    frontal = np.arctan2(trunk[:, 1], np.maximum(np.abs(trunk[:, 2]), 1e-12))
    pelvis = p("pelvis")
    pelvis_ref = np.nanmedian(pelvis[: max(1, min(len(pelvis), int(round(len(pelvis) * 0.1))))], axis=0)
    pelvis_delta = pelvis - pelvis_ref
    pelvis_dispersion = float(np.sqrt(np.nanmean(np.sum(pelvis_delta ** 2, axis=1))))
    pelvis_path = float(np.nansum(np.linalg.norm(np.diff(pelvis, axis=0), axis=1)))

    left = np.nanmean(np.stack([angles["knee"][0], angles["hip"][0]], axis=1), axis=1)
    right = np.nanmean(np.stack([angles["knee"][1], angles["hip"][1]], axis=1), axis=1)
    symmetry = 100.0 * np.nanmean(np.abs(left - right) / (0.5 * (np.abs(left) + np.abs(right)) + 1e-9))

    endpoint_names = ["wrist_l", "wrist_r", "ankle_l", "ankle_r", "pelvis"]
    velocities = [np.linalg.norm(derivative(p(name), times), axis=1) for name in endpoint_names]
    speed_profile = np.nanmean(np.stack(velocities, axis=1), axis=1)
    sample_rate = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 1.0
    smoothness = sparc(speed_profile, sample_rate)
    angular_velocities = []
    for family in ("knee", "hip", "ankle", "shoulder", "elbow"):
        for series in angles[family]:
            angular_velocities.append(np.abs(derivative(series, times)))
    peak_angular_velocity = float(np.nanmax(np.stack(angular_velocities, axis=1)))

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
        "sparc_smoothness": smoothness,
        "peak_angular_velocity_deg_s": float(np.degrees(peak_angular_velocity)),
        "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "speed_proxy": float(np.nanmedian(speed_profile)),
        "trunk_length_native": trunk_length,
        **qc,
    }
    traces = {
        "speed_profile": speed_profile,
        "pelvis": pelvis,
        "trunk_sagittal": sagittal,
        "trunk_frontal": frontal,
    }
    return result, traces


def residual_lag_seconds(
    time_a: np.ndarray, signal_a: np.ndarray, time_b: np.ndarray, signal_b: np.ndarray,
    sample_rate: float = 60.0, max_lag_s: float = 0.5,
) -> tuple[float, float]:
    duration = min(float(time_a[-1] - time_a[0]), float(time_b[-1] - time_b[0]))
    if duration <= 0.25:
        return np.nan, np.nan
    grid = np.arange(0.0, duration, 1.0 / sample_rate)
    if len(grid) < 10:
        return np.nan, np.nan
    a = np.interp(grid, time_a - time_a[0], signal_a)
    b = np.interp(grid, time_b - time_b[0], signal_b)
    a = (a - np.mean(a)) / (np.std(a) + 1e-12)
    b = (b - np.mean(b)) / (np.std(b) + 1e-12)
    max_lag = int(round(max_lag_s * sample_rate))
    corr = np.correlate(a, b, mode="full") / len(grid)
    lags = np.arange(-len(grid) + 1, len(grid))
    keep = np.abs(lags) <= max_lag
    best = np.argmax(corr[keep])
    selected_lags = lags[keep]
    selected_corr = corr[keep]
    return float(selected_lags[best] / sample_rate), float(selected_corr[best])
