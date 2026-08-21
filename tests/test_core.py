from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from umons_rebuild.agreement import icc_a1, icc_c1, metrics
from umons_rebuild.features import extract_features_v2, residual_lag, uniform_resample_strict
from umons_rebuild.geometry import sparc
from umons_rebuild.gate_b_extensions import _interpolate_points
from umons_rebuild.calibration import _band, _fit_univariate_ridge, _wide
from umons_rebuild.io import load_config, parse_metadata, resolve_paths
from umons_rebuild.ranking import PRIMARY_FEATURES, _assert_rf_prediction_in_training_range, _ridge_response_weights


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ("config.local.yml" if (ROOT / "config.local.yml").exists() else "config.yml")


def test_official_skill_means_and_ids():
    paths = resolve_paths(load_config(CONFIG), ROOT)
    metadata_path = paths.official_repo_root / "Metadata.txt"
    if not metadata_path.exists():
        pytest.skip("Official UMONS-TAICHI data are not bundled with the reproducibility code.")
    metadata = parse_metadata(metadata_path)
    assert metadata.participant_id.tolist() == [f"P{i:02d}" for i in range(1, 13)]
    assert np.allclose(
        metadata.skill_mean,
        [9.43, 9.57, 8.67, 8.07, 7.23, 8.50, 6.77, 7.43, 6.85, 6.10, 4.97, 5.85],
        atol=0.005,
    )


def test_bland_altman_midpoint_identity():
    result = metrics(np.array([1.0, 2.0, 4.0]), np.array([1.5, 1.8, 5.0]))
    midpoint = (result["loa_lower"] + result["loa_upper"]) / 2.0
    assert abs(midpoint - result["bias"]) < 1e-12


def test_frozen_tertile_bands_are_four_four_four():
    labels = np.array([4.97, 5.85, 6.10, 6.77, 6.85, 7.23, 7.43, 8.07, 8.50, 8.67, 9.43, 9.57])
    assert np.bincount(_band(labels, (6.81, 8.285))).tolist() == [4, 4, 4]


def test_uniform_resample_uses_exact_thirty_hz_grid():
    times = np.array([10.0, 10.031, 10.069, 10.101, 10.137])
    points = np.zeros((len(times), 1, 3))
    grid, _ = uniform_resample_strict(times, points, target_rate_hz=30.0)
    assert np.allclose(np.diff(grid), 1.0 / 30.0, atol=1e-12)


def test_icc_consistency_exceeds_absolute_agreement_under_additive_bias():
    reference = np.arange(1.0, 13.0)
    biased = reference + 5.0
    assert icc_c1(reference, biased) > 0.999
    assert icc_c1(reference, biased) > icc_a1(reference, biased)


def test_pelvis_dispersion_is_deviation_about_trajectory_mean():
    names = [
        "pelvis", "spine_shoulder", "head", "shoulder_l", "shoulder_r",
        "elbow_l", "elbow_r", "wrist_l", "wrist_r", "hip_l", "hip_r",
        "knee_l", "knee_r", "ankle_l", "ankle_r", "foot_l", "foot_r",
    ]
    base = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.8],
        [-0.2, 0.0, 0.5], [0.2, 0.0, 0.5], [-0.4, 0.0, 0.35],
        [0.4, 0.0, 0.35], [-0.5, 0.0, 0.2], [0.5, 0.0, 0.2],
        [-0.12, 0.0, 0.0], [0.12, 0.0, 0.0], [-0.12, 0.0, -0.4],
        [0.12, 0.0, -0.4], [-0.12, 0.0, -0.8], [0.12, 0.0, -0.8],
        [-0.12, 0.15, -0.85], [0.12, 0.15, -0.85],
    ])
    times = np.arange(30) / 30.0
    stationary = np.repeat(base[None, :, :], len(times), axis=0)
    values, _ = extract_features_v2(times, stationary, names, scale_m=0.5)
    assert values["pelvis_dispersion"] < 1e-10

    translated = stationary.copy()
    translated[:, :, 0] += np.linspace(0.0, 0.2, len(times))[:, None]
    values, _ = extract_features_v2(times, translated, names, scale_m=0.5)
    expected = np.std(np.linspace(0.0, 0.2, len(times)) / 0.5, ddof=0)
    assert np.isclose(values["pelvis_dispersion"], expected, rtol=1e-6)


def test_sparc_is_finite_for_valid_dc_dominated_motion():
    speed = 1.0 + 0.001 * np.sin(np.linspace(0, 2 * np.pi, 120))
    assert np.isfinite(sparc(speed, sample_rate=30.0))


def test_lag_time_shift_sign_aligns_delayed_kinect_signal():
    times = np.arange(0.0, 4.0, 1.0 / 60.0)
    qualisys = np.exp(-((times - 1.5) / 0.15) ** 2)
    kinect = np.exp(-((times - 1.7) / 0.15) ** 2)
    lag_s = -0.2
    grid = np.arange(0.0, 3.8, 1.0 / 60.0)
    aligned_kinect = np.interp(grid, times + lag_s, kinect)
    aligned_qualisys = np.interp(grid, times, qualisys)
    assert np.corrcoef(aligned_qualisys, aligned_kinect)[0, 1] > 0.99


def test_overlap_normalized_lag_recovers_delayed_kinect_signal():
    times = np.arange(0.0, 4.0, 1.0 / 60.0)
    qualisys = np.exp(-((times - 1.5) / 0.15) ** 2) + 0.35 * np.exp(-((times - 2.7) / 0.22) ** 2)
    kinect = np.exp(-((times - 1.7) / 0.15) ** 2) + 0.35 * np.exp(-((times - 2.9) / 0.22) ** 2)
    result = residual_lag(times, qualisys, times, kinect, max_lag_s=0.5, sample_rate_hz=60.0)
    assert np.isclose(result["best_lag_s"], -0.2, atol=1.0 / 60.0)
    assert result["max_correlation"] > 0.99
    assert not result["hit_boundary"]


def test_interpolate_points_preserves_joint_axis_shape():
    times = np.array([0.0, 1.0])
    points = np.zeros((2, 2, 3))
    points[1, :, :] = 1.0
    grid = np.array([0.0, 0.5, 1.0])
    interpolated = _interpolate_points(times, points, grid)
    assert interpolated.shape == (3, 2, 3)
    assert np.allclose(interpolated[1], 0.5)


def test_public_config_is_portable_and_not_claimed_preregistered():
    config_text = (ROOT / "config.yml").read_text(encoding="utf-8")
    assert "C:/Users/" not in config_text
    assert "Pre-registered" not in config_text
    assert "Pre-specified for the corrected reanalysis" in config_text


def test_v3_primary_ranking_feature_set_is_frozen_without_duration_peak_or_rms():
    assert len(PRIMARY_FEATURES) == 13
    assert "p95_angular_velocity_deg_s" in PRIMARY_FEATURES
    assert "duration_s" not in PRIMARY_FEATURES
    assert "peak_angular_velocity_deg_s" not in PRIMARY_FEATURES
    assert "rms_angular_velocity_deg_s" not in PRIMARY_FEATURES
    specification = (ROOT / "ANALYSIS_SPEC.md").read_text(encoding="utf-8")
    assert "Frozen on 2026-08-20 before inspecting any R22–R27 v3 ranking result." in specification


def test_fast_ridge_response_weights_match_standardized_sklearn_ridge():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(20260820)
    x_train = rng.normal(size=(9, 15))
    y_train = rng.normal(size=9)
    x_test = rng.normal(size=(1, 15))
    for alpha in (0.1, 1.0, 10.0, 100.0):
        scaler = StandardScaler().fit(x_train)
        expected = Ridge(alpha=alpha).fit(scaler.transform(x_train), y_train).predict(scaler.transform(x_test))[0]
        observed = _ridge_response_weights(x_train, x_test, alpha) @ y_train
        assert np.isclose(observed, expected, atol=1e-10)


def test_rf_training_range_assertion_accepts_bounds_and_rejects_extrapolation():
    labels = np.array([4.97, 6.10, 9.57])
    _assert_rf_prediction_in_training_range(4.97, labels)
    _assert_rf_prediction_in_training_range(9.57, labels)
    with pytest.raises(AssertionError):
        _assert_rf_prediction_in_training_range(9.58, labels)


def test_v4_final_closure_specification_is_frozen():
    specification = (ROOT / "ANALYSIS_SPEC.md").read_text(encoding="utf-8")
    assert "Random splitting uses 200 seeded repetitions." in specification
    assert "repeated 500 times" in specification
    assert "10,000 times" in specification
    assert "training-label range" in specification


def test_per_feature_ridge_uses_only_the_corresponding_feature():
    x = np.linspace(-2.0, 2.0, 100)
    y = 3.0 + 2.0 * x
    prediction, slope, intercept = _fit_univariate_ridge(x, y, x, alpha=0.01)
    assert np.sqrt(np.mean((prediction - y) ** 2)) < 0.001
    assert np.isclose(slope, 2.0, atol=0.001)
    assert np.isclose(intercept, 3.0, atol=0.001)


def test_participant_gesture_wide_matrix_has_deterministic_shape():
    participants = ["P01", "P02"]
    gestures = ["G01", "G02"]
    index = pd.DataFrame(
        [(participant, gesture) for participant in participants for gesture in gestures],
        columns=["participant_id", "gesture_id"],
    )
    values = np.arange(8, dtype=float).reshape(4, 2)
    wide = _wide(index, values, participants, ["f1", "f2"])
    assert wide.shape == (2, 4)
