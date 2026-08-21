from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umons_rebuild.io import load_config, parse_metadata, resolve_paths
from umons_rebuild.pipeline import PRIMARY
from umons_rebuild.features import FEATURE_COLUMNS_V2
from umons_rebuild.ranking import (
    PRIMARY_FEATURES, DURATION_SENSITIVITY_FEATURES, RMS_SENSITIVITY_FEATURES, MODELS,
    _aggregate_paired, _wide, _low_dimensional, _nested_predictions, _prediction_metrics,
    build_full_qualisys_cache, _true_jackknife,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ranking analyses retained in the final revised manuscript.")
    parser.add_argument("--config", default=str(ROOT / "config.yml"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = (ROOT / "artifacts_stage_d/03_r22_r27").resolve()
    expected_parent = (ROOT / "artifacts_stage_d").resolve()
    if args.clean and output.exists():
        if output.parent != expected_parent or output.name != "03_r22_r27":
            raise RuntimeError(f"Refusing to clean unexpected path: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg, ROOT)
    thresholds = tuple(map(float, cfg["qwk_band_thresholds"]))
    bootstrap = int(cfg.get("bootstrap_repetitions", 2000))
    seed = int(cfg.get("random_seed", 20260815))
    started = time.time()

    metadata = parse_metadata(paths.official_repo_root / "Metadata.txt").sort_values("participant_id")
    participants = metadata.participant_id.to_numpy()
    participant_list = participants.tolist()
    y = metadata.skill_mean.to_numpy(float)
    paired = pd.read_csv(ROOT / "artifacts_r01_r18/01_feature_pipeline/feature_cache_v2.csv.gz")
    q_long = _aggregate_paired(paired, "qualisys", FEATURE_COLUMNS_V2)
    k_long = _aggregate_paired(paired, "kinect", FEATURE_COLUMNS_V2)
    q_primary, primary_names = _wide(q_long, participant_list, PRIMARY_FEATURES)
    k_primary, _ = _wide(k_long, participant_list, PRIMARY_FEATURES)

    # Exploratory five-model × two-sensor comparison retained in Supplementary Results.
    predictions = []
    for sensor, matrix in [("qualisys", q_primary), ("kinect", k_primary)]:
        for model_index, model in enumerate(MODELS):
            print(f"MODEL_START {sensor} {model}", flush=True)
            predictions.append(_nested_predictions(
                model, f"{sensor}__primary_wide", matrix, y, participants, thresholds,
                seed + model_index * 100000 + (0 if sensor == "qualisys" else 500000),
            ))
    r25_predictions = pd.concat(predictions, ignore_index=True)
    r25_predictions.to_csv(output / "r25_nested_loso_predictions_10_conditions.csv", index=False)
    r25_metrics = _prediction_metrics(r25_predictions, thresholds, bootstrap, seed + 1000000)
    r25_metrics.to_csv(output / "r25_model_comparison_10_conditions.csv", index=False)

    # Frozen feature-set sensitivities.
    sensitivity_predictions = []
    for sensor, long in [("qualisys", q_long), ("kinect", k_long)]:
        primary_existing = r25_predictions.loc[
            r25_predictions.condition.eq(f"{sensor}__primary_wide") & r25_predictions.model.eq("ridge")
        ].copy()
        sensitivity_predictions.append(primary_existing)
        duration_matrix, _ = _wide(long, participant_list, DURATION_SENSITIVITY_FEATURES)
        rms_matrix, _ = _wide(long, participant_list, RMS_SENSITIVITY_FEATURES)
        low_matrix = _low_dimensional(long, participant_list, PRIMARY_FEATURES)
        for label, matrix in [
            ("duration_sensitivity_wide", duration_matrix),
            ("rms_velocity_sensitivity_wide", rms_matrix),
            ("primary_lowdim_median", low_matrix),
        ]:
            sensitivity_predictions.append(_nested_predictions(
                "ridge", f"{sensor}__{label}", matrix, y, participants, thresholds,
                seed + len(sensitivity_predictions) * 10000,
            ))
    r22_predictions = pd.concat(sensitivity_predictions, ignore_index=True)
    r22_predictions.to_csv(output / "r22_feature_set_and_lowdim_predictions.csv", index=False)
    r22_metrics = _prediction_metrics(r22_predictions, thresholds, bootstrap, seed + 2000000)
    r22_metrics.to_csv(output / "r22_feature_set_and_lowdim_metrics.csv", index=False)

    # Full Qualisys availability and deterministic one-draw equalization used as the Stage-D precursor.
    scale_path = ROOT / "artifacts_r01_r18/01_feature_pipeline/participant_trunk_scale.csv"
    full_cache = build_full_qualisys_cache(paths, scale_path, output / "full_qualisys_feature_cache_v3.csv.gz")
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
    equalized_cache.to_csv(output / "r23_qualisys_equalized_segment_selection.csv.gz", index=False, compression="gzip")
    equalized_long = equalized_cache.groupby(["participant_id", "gesture_id"], as_index=False)[PRIMARY_FEATURES].mean()
    r23_predictions = []
    for label, long in [
        ("qualisys_full_2149", full_long),
        ("qualisys_paired_1815", q_long),
        ("qualisys_equalized_1815", equalized_long),
    ]:
        matrix, _ = _wide(long, participant_list, PRIMARY_FEATURES)
        r23_predictions.append(_nested_predictions(
            "ridge", label, matrix, y, participants, thresholds,
            seed + len(r23_predictions) * 10000,
        ))
    r23_predictions = pd.concat(r23_predictions, ignore_index=True)
    r23_predictions.to_csv(output / "r23_missingness_predictions.csv", index=False)
    r23_metrics = _prediction_metrics(r23_predictions, thresholds, bootstrap, seed + 3100000)
    r23_metrics["segment_count"] = r23_metrics.condition.map({
        "qualisys_full_2149": len(full_cache),
        "qualisys_paired_1815": len(paired_q_segments),
        "qualisys_equalized_1815": len(equalized_cache),
    })
    r23_metrics.to_csv(output / "r23_missingness_metrics.csv", index=False)

    # True participant jackknife; final exact leakage and 10,000 permutations are run only once in final_closure.
    jackknife = pd.concat([
        _true_jackknife("qualisys_ridge", q_primary, y, participants, thresholds, seed + 6000000),
        _true_jackknife("kinect_ridge", k_primary, y, participants, thresholds, seed + 6100000),
    ], ignore_index=True)
    jackknife.to_csv(output / "r27_true_participant_jackknife.csv", index=False)

    checks = {
        "primary_feature_count": len(PRIMARY_FEATURES),
        "primary_predictor_count": len(primary_names),
        "model_conditions": int(len(r25_metrics)),
        "full_qualisys_segments": int(len(full_cache)),
        "paired_qualisys_segments": int(len(paired_q_segments)),
        "equalized_qualisys_segments": int(len(equalized_cache)),
        "jackknife_rows": int(len(jackknife)),
        "calibrated_ranking_condition_absent": bool(~r25_predictions.condition.str.contains("calibrated").any()),
    }
    checks["pass"] = bool(
        checks["primary_feature_count"] == 13
        and checks["primary_predictor_count"] == 169
        and checks["model_conditions"] == 10
        and checks["full_qualisys_segments"] == 2149
        and checks["paired_qualisys_segments"] == 1815
        and checks["equalized_qualisys_segments"] == 1815
        and checks["jackknife_rows"] == 24
        and checks["calibrated_ranking_condition_absent"]
    )
    (output / "ranking_core_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not checks["pass"]:
        raise RuntimeError(f"Ranking checks failed: {checks}")
    print(f"RANKING_CORE_PASS elapsed={time.time()-started:.1f}s", flush=True)

if __name__ == "__main__":
    main()
