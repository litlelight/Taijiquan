from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umons_rebuild.calibration import run_per_feature_calibration, run_r21_nested_reference
from umons_rebuild.io import load_config, resolve_paths


OUTPUT_NAME = "artifacts_r19_r21"
DEFAULT_CONFIG = ROOT / ("config.local.yml" if (ROOT / "config.local.yml").exists() else "config.yml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe per-feature calibration experiments R19-R21.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    args = parser.parse_args()

    extended_gate = ROOT / "artifacts_gate_b_extensions/gate_b_extended_checks.json"
    if not extended_gate.exists() or not json.loads(extended_gate.read_text(encoding="utf-8"))["gate_b_extended_pass"]:
        raise RuntimeError("R19-R21 blocked until the extended Gate B passes")

    output = (ROOT / OUTPUT_NAME).resolve()
    if args.clean and output.exists():
        if output.parent != ROOT.resolve() or output.name != OUTPUT_NAME:
            raise RuntimeError(f"Refusing to clean unexpected path: {output}")
        shutil.rmtree(output)
    calibration_dir = output / "01_per_feature_calibration"
    nested_dir = output / "02_nested_ranking_reference"
    output.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg, ROOT)
    repetitions = args.bootstrap_repetitions or int(cfg.get("bootstrap_repetitions", 2000))
    seed = int(cfg.get("random_seed", 20260815))
    thresholds = tuple(map(float, cfg["qwk_band_thresholds"]))
    feature_path = ROOT / "artifacts_r01_r18/01_feature_pipeline/feature_cache_v2.csv.gz"
    started = time.time()

    calibration = run_per_feature_calibration(
        feature_path=feature_path,
        out_dir=calibration_dir,
        repetitions=repetitions,
        seed=seed,
    )
    print("R19_R20_PASS", flush=True)
    r21 = run_r21_nested_reference(
        feature_path=feature_path,
        metadata_path=paths.official_repo_root / "Metadata.txt",
        out_dir=nested_dir,
        thresholds=thresholds,
        seed=seed + 100000,
    )
    print("R21_PASS", flush=True)
    summary = {
        "scope": "R19-R21 only; R22-R30 not run",
        "bootstrap_repetitions": repetitions,
        "random_seed": seed,
        "elapsed_seconds": time.time() - started,
        "r19_r20": calibration["checks"],
        "r21": r21,
    }
    (output / "r19_r21_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"R19_R21_COMPLETE elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
