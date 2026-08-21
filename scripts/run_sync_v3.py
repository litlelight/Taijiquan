from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umons_rebuild.io import load_config, resolve_paths
from umons_rebuild.sync import run_sync_v3


OUTPUT_NAME = "artifacts_stage_d/01_sync_v3"
DEFAULT_CONFIG = ROOT / ("config.local.yml" if (ROOT / "config.local.yml").exists() else "config.yml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run synchronization with overlap-normalized correlation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    args = parser.parse_args()
    output = (ROOT / OUTPUT_NAME).resolve()
    expected_parent = (ROOT / "artifacts_stage_d").resolve()
    if args.clean and output.exists():
        if output.parent != expected_parent or output.name != "01_sync_v3":
            raise RuntimeError(f"Refusing to clean unexpected path: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, ROOT)
    repetitions = args.bootstrap_repetitions or int(cfg.get("bootstrap_repetitions", 2000))
    seed = int(cfg.get("random_seed", 20260815))
    base = ROOT / "artifacts_r01_r18/01_feature_pipeline"
    started = time.time()
    checks = run_sync_v3(
        paths,
        base / "feature_cache_v2.csv.gz",
        base / "participant_trunk_scale.csv",
        base / "pelvis_residual_lag.csv",
        output,
        repetitions,
        seed,
    )
    summary = {"elapsed_seconds": time.time() - started, "checks": checks}
    (output / "sync_v3_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SYNC_V3_PASS elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
