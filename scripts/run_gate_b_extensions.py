from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umons_rebuild.gate_b_extensions import run_gate_b_extensions
from umons_rebuild.io import load_config, resolve_paths


OUTPUT_NAME = "artifacts_gate_b_extensions"
DEFAULT_CONFIG = ROOT / ("config.local.yml" if (ROOT / "config.local.yml").exists() else "config.yml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the conditional Gate B sensitivity experiments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    args = parser.parse_args()

    output = (ROOT / OUTPUT_NAME).resolve()
    if args.clean and output.exists():
        if output.parent != ROOT.resolve() or output.name != OUTPUT_NAME:
            raise RuntimeError(f"Refusing to clean unexpected path: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    paths = resolve_paths(cfg, ROOT)
    repetitions = args.bootstrap_repetitions or int(cfg.get("bootstrap_repetitions", 2000))
    seed = int(cfg.get("random_seed", 20260815))
    base = ROOT / "artifacts_r01_r18/01_feature_pipeline"
    started = time.time()
    checks = run_gate_b_extensions(
        paths=paths,
        feature_path=base / "feature_cache_v2.csv.gz",
        qc_path=base / "segment_time_qc_v2.csv.gz",
        lag_path=base / "pelvis_residual_lag.csv",
        scale_path=base / "participant_trunk_scale.csv",
        out_dir=output,
        repetitions=repetitions,
        seed=seed,
    )
    summary = {
        "scope": "Conditional Gate B closure before R19",
        "bootstrap_repetitions": repetitions,
        "random_seed": seed,
        "elapsed_seconds": time.time() - started,
        "checks": checks,
    }
    (output / "gate_b_extension_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"EXTENDED_GATE_B_PASS elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
