from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPECTED = json.loads((ROOT / "reference/key_results.json").read_text(encoding="utf-8"))
ATOL = float(EXPECTED["tolerance"])


def close(name: str, observed: float, expected: float) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=ATOL):
        raise AssertionError(f"{name}: {observed} != {expected} (atol={ATOL})")


def main() -> None:
    required = [
        ROOT / "artifacts_r01_r18/02_gate_b/primary_agreement_icc_bias.csv",
        ROOT / "artifacts_gate_b_extensions/alternative_mapping_agreement.csv",
        ROOT / "artifacts_stage_d/02_calibration_diagnostics/r20b_feature_comparison.csv",
        ROOT / "artifacts_stage_d/01_sync_v3/pelvis_residual_lag_quality_summary.csv",
        ROOT / "artifacts_stage_d/01_sync_v3/high_confidence_lag_correction_change_v3.csv",
        ROOT / "artifacts_final_freeze/01_final_closure/permutation_10000_summary.csv",
        ROOT / "artifacts_final_freeze/01_final_closure/exact_segment_random_split_vs_participant_loso.csv",
        ROOT / "artifacts_final_freeze/01_final_closure/equalized_repetition_summary.json",
        ROOT / "artifacts_final_freeze/01_final_closure/rf_training_range_assertion_summary.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Generated outputs are missing. Run `python run_analysis.py --clean` first.\nMissing:\n  " + "\n  ".join(missing))

    primary = pd.read_csv(required[0]).set_index("feature")
    for feat, values in EXPECTED["primary_agreement"].items():
        for col, expected in values.items(): close(f"primary:{feat}:{col}", primary.loc[feat,col], expected)

    mapping = pd.read_csv(ROOT / "artifacts_gate_b_extensions/alternative_mapping_agreement.csv")
    hip = mapping[(mapping.condition == "alternative_mapping") & (mapping.feature == "hip_angle_mean_deg")].iloc[0]
    for col, expected in EXPECTED["alternative_mapping_hip"].items(): close(f"mapping:hip:{col}", hip[col], expected)

    diag = pd.read_csv(ROOT / "artifacts_stage_d/02_calibration_diagnostics/r20b_feature_comparison.csv").set_index("feature")
    colmap = {"icc_a1_ridge":"icc_a1_ridge_per_feature", "rmse_ridge":"rmse_ridge_per_feature", "rmse_intercept":"rmse_intercept_only_loso"}
    for feat, values in EXPECTED["calibration"].items():
        for key, expected in values.items(): close(f"calibration:{feat}:{key}", diag.loc[feat,colmap[key]], expected)

    lag_summary = pd.read_csv(ROOT / "artifacts_stage_d/01_sync_v3/pelvis_residual_lag_quality_summary.csv")
    lag_change = pd.read_csv(ROOT / "artifacts_stage_d/01_sync_v3/high_confidence_lag_correction_change_v3.csv").set_index("feature")
    close("sync:median_lag_ms_1s", lag_summary.loc[lag_summary.search_window_s.eq(1.0),"median_lag_ms"].iloc[0], EXPECTED["sync"]["median_lag_ms_1s"])
    if int(lag_change.n_units_after.max()) != int(EXPECTED["sync"]["high_confidence_units"]): raise AssertionError("sync:high_confidence_units")
    close("sync:p95_icc_change", lag_change.loc["p95_angular_velocity_deg_s","icc_a1_change_after_minus_before"], EXPECTED["sync"]["p95_icc_change"])
    close("sync:sparc_icc_change", lag_change.loc["sparc_smoothness","icc_a1_change_after_minus_before"], EXPECTED["sync"]["sparc_icc_change"])

    perm = pd.read_csv(ROOT / "artifacts_final_freeze/01_final_closure/permutation_10000_summary.csv").set_index("condition")
    for cond, values in EXPECTED["ranking"].items():
        for col, expected in values.items(): close(f"ranking:{cond}:{col}", perm.loc[cond,col], expected)

    exact = pd.read_csv(ROOT / "artifacts_final_freeze/01_final_closure/exact_segment_random_split_vs_participant_loso.csv").set_index("sensor")
    for sensor, values in EXPECTED["leakage"].items():
        for col, expected in values.items(): close(f"leakage:{sensor}:{col}", exact.loc[sensor,col], expected)

    equal = json.loads((ROOT / "artifacts_final_freeze/01_final_closure/equalized_repetition_summary.json").read_text(encoding="utf-8"))
    for key, expected in EXPECTED["equalization"].items():
        if isinstance(expected, int):
            if int(equal[key]) != expected: raise AssertionError(f"equalization:{key}")
        else: close(f"equalization:{key}", equal[key], expected)

    rf = pd.read_csv(ROOT / "artifacts_final_freeze/01_final_closure/rf_training_range_assertion_summary.csv").set_index("condition")
    for cond, expected in EXPECTED["rf_training_range"].items():
        if bool(rf.loc[cond,"all_within_training_range"]) != bool(expected): raise AssertionError(f"rf:{cond}")

    print("KEY_RESULT_VERIFICATION_PASS")

if __name__ == "__main__":
    main()
