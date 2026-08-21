from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umons_rebuild.gate_b import run_gate_b
from umons_rebuild.io import load_config, resolve_paths
from umons_rebuild.pipeline import build_v2_cache


OUTPUT_NAME = "artifacts_r01_r18"
DEFAULT_CONFIG = ROOT / ("config.local.yml" if (ROOT / "config.local.yml").exists() else "config.yml")


def _clean_output(target: Path) -> None:
    resolved = target.resolve()
    if resolved.parent != ROOT.resolve() or resolved.name != OUTPUT_NAME:
        raise RuntimeError(f"Refusing to clean unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only UMONS-TAICHI experiments R01-R18.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg, ROOT)
    output = ROOT / OUTPUT_NAME
    if args.clean:
        _clean_output(output)
    feature_dir = output / "01_feature_pipeline"
    gate_b_dir = output / "02_gate_b"
    output.mkdir(parents=True, exist_ok=True)

    repetitions = args.bootstrap_repetitions or int(cfg.get("bootstrap_repetitions", 2000))
    seed = int(cfg.get("random_seed", 20260815))
    started = time.time()
    print(f"R01_R18_START bootstrap_repetitions={repetitions} seed={seed}", flush=True)

    stage_a = build_v2_cache(paths, feature_dir, seed=seed)
    print("GATE_A_PASS", flush=True)
    gate_b = run_gate_b(
        paths=paths,
        feature_path=Path(stage_a["feature_path"]),
        lag_path=Path(stage_a["lag_path"]),
        joint_path=Path(stage_a["joint_path"]),
        scales=stage_a["scales"],
        out_dir=gate_b_dir,
        repetitions=repetitions,
        seed=seed,
    )
    print("GATE_B_PASS", flush=True)

    summary = {
        "scope": "R01-R18 only; stopped before calibration, ranking, and machine learning",
        "bootstrap_repetitions": repetitions,
        "random_seed": seed,
        "elapsed_seconds": time.time() - started,
        "gate_a": stage_a["sanity"],
        "gate_b": gate_b,
    }
    (output / "r01_r18_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "sha256_manifest.csv"):
        manifest.append(f'{path.relative_to(output).as_posix()},{path.stat().st_size},{_sha256(path)}')
    (output / "sha256_manifest.csv").write_text(
        "relative_path,size_bytes,sha256\n" + "\n".join(manifest) + "\n", encoding="utf-8"
    )
    print(f"R01_R18_COMPLETE elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
