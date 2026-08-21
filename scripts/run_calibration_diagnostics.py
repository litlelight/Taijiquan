from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umons_rebuild.calibration_diagnostics import run_calibration_diagnostics


def main() -> None:
    output = (ROOT / "artifacts_stage_d/02_calibration_diagnostics").resolve()
    expected_parent = (ROOT / "artifacts_stage_d").resolve()
    if output.exists():
        if output.parent != expected_parent or output.name != "02_calibration_diagnostics":
            raise RuntimeError(f"Refusing to clean unexpected path: {output}")
        shutil.rmtree(output)
    base = ROOT / "artifacts_r19_r21/01_per_feature_calibration"
    checks = run_calibration_diagnostics(
        base / "r19_r20_per_feature_oof_predictions.csv.gz",
        base / "r19_r20_calibration_coefficients.csv",
        output,
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
