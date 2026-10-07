from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umons_rebuild.final_closure import run_final_closure
from umons_rebuild.io import load_config, resolve_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four final reviewer-requested closure analyses.")
    parser.add_argument("--config", default=str(ROOT / ("config.local.yml" if (ROOT / "config.local.yml").exists() else "config.yml")))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--equalizations", type=int, default=500)
    parser.add_argument("--segment-splits", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=None)
    args = parser.parse_args()
    output = (ROOT / "artifacts_final_freeze/01_final_closure").resolve()
    expected_parent = (ROOT / "artifacts_final_freeze").resolve()
    if args.clean and output.exists():
        if output.parent != expected_parent or output.name != "01_final_closure":
            raise RuntimeError(f"Refusing to clean unexpected path: {output}")
        shutil.rmtree(output)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, ROOT)
    started = time.time()
    checks = run_final_closure(
        paths,
        ROOT / "artifacts_r01_r18/01_feature_pipeline/feature_cache_v2.csv.gz",
        ROOT / "artifacts_stage_d/03_r22_r27/full_qualisys_feature_cache_v3.csv.gz",
        ROOT / "artifacts_stage_d/03_r22_r27/r25_nested_loso_predictions_10_conditions.csv",
        output,
        tuple(cfg["qwk_band_thresholds"]),
        args.equalizations,
        args.segment_splits,
        args.permutations or int(cfg["permutation_repetitions"]),
        int(cfg["random_seed"]),
    )
    summary = {"elapsed_seconds": time.time() - started, "checks": checks}
    (output / "final_closure_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"FINAL_CLOSURE_PASS elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
