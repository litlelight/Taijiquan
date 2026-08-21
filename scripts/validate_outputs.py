from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umons_rebuild.ranking import _metrics


def recompute(pred_path: Path, metric_path: Path, thresholds=(6.81, 8.285)) -> float:
    predictions = pd.read_csv(pred_path)
    saved = pd.read_csv(metric_path)
    diffs = []
    for (condition, model), one in predictions.groupby(["condition", "model"]):
        expected = saved.loc[saved.condition.eq(condition) & saved.model.eq(model)].iloc[0]
        calc = _metrics(one.true_skill.to_numpy(), one.prediction.to_numpy(), thresholds)
        diffs.extend(abs(float(expected[k]) - v) for k, v in calc.items())
    return float(max(diffs))


def main() -> None:
    ranking = ROOT / "artifacts_stage_d/03_r22_r27"
    errors = {
        "r25": recompute(ranking / "r25_nested_loso_predictions_10_conditions.csv", ranking / "r25_model_comparison_10_conditions.csv"),
        "r22": recompute(ranking / "r22_feature_set_and_lowdim_predictions.csv", ranking / "r22_feature_set_and_lowdim_metrics.csv"),
        "r23": recompute(ranking / "r23_missingness_predictions.csv", ranking / "r23_missingness_metrics.csv"),
    }
    sync = json.loads((ROOT / "artifacts_stage_d/01_sync_v3/sync_v3_checks.json").read_text(encoding="utf-8"))
    cal = json.loads((ROOT / "artifacts_stage_d/02_calibration_diagnostics/r20b_checks.json").read_text(encoding="utf-8"))
    core = json.loads((ranking / "ranking_core_checks.json").read_text(encoding="utf-8"))
    checks = {
        "metric_recompute_max_abs_error": max(errors.values()),
        "sync_pass": bool(sync["sync_v3_pass"]),
        "calibration_diagnostics_pass": bool(cal["r20b_pass"]),
        "ranking_core_pass": bool(core["pass"]),
    }
    checks["pass"] = bool(checks["metric_recompute_max_abs_error"] < 1e-12 and checks["sync_pass"] and checks["calibration_diagnostics_pass"] and checks["ranking_core_pass"])
    print(json.dumps(checks, indent=2))
    if not checks["pass"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
