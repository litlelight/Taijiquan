from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yml"


def run_step(name: str, args: list[str]) -> dict:
    started = time.time()
    print(f"STEP_START {name}", flush=True)
    completed = subprocess.run([sys.executable, *args], cwd=ROOT)
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(f"Step {name} failed with exit code {completed.returncode}")
    print(f"STEP_PASS {name} elapsed={elapsed:.1f}s", flush=True)
    return {"step": name, "elapsed_seconds": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the frozen UMONS-TAICHI analyses used in the revised manuscript.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--clean", action="store_true", help="Regenerate all derived outputs.")
    parser.add_argument("--skip-md5", action="store_true", help="Skip full archive MD5 checks after the first validated run.")
    args = parser.parse_args()
    cfg = str(Path(args.config).resolve())
    clean = ["--clean"] if args.clean else []
    started = time.time()
    steps = []

    data_args = ["check_data.py", "--config", cfg] + (["--skip-md5"] if args.skip_md5 else [])
    steps.append(run_step("data_check", data_args))
    steps.append(run_step("unit_tests", ["-m", "pytest", "tests/test_core.py", "-q"]))
    steps.append(run_step("primary_pipeline", ["scripts/run_r01_r18.py", "--config", cfg, *clean, "--bootstrap-repetitions", "2000"]))
    steps.append(run_step("mechanism_sensitivities", ["scripts/run_gate_b_extensions.py", "--config", cfg, *clean, "--bootstrap-repetitions", "2000"]))
    steps.append(run_step("synchronization", ["scripts/run_sync_v3.py", "--config", cfg, *clean, "--bootstrap-repetitions", "2000"]))
    steps.append(run_step("calibration", ["scripts/run_r19_r21.py", "--config", cfg, *clean, "--bootstrap-repetitions", "2000"]))
    steps.append(run_step("calibration_diagnostics", ["scripts/run_calibration_diagnostics.py"]))
    steps.append(run_step("ranking", ["scripts/run_ranking.py", "--config", cfg, *clean]))
    steps.append(run_step("final_closure", ["scripts/run_final_closure.py", "--config", cfg, *clean, "--equalizations", "500", "--segment-splits", "200", "--permutations", "10000"]))
    steps.append(run_step("internal_consistency", ["scripts/validate_outputs.py"]))
    steps.append(run_step("key_result_verification", ["verify_key_results.py"]))

    summary = {
        "version": "1.0.1",
        "status": "PASS",
        "parameters": {"bootstrap": 2000, "random_segment_splits": 200, "equalizations": 500, "permutations": 10000},
        "elapsed_seconds": time.time() - started,
        "steps": steps,
    }
    Path("reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("REPRODUCTION_PASS", flush=True)

if __name__ == "__main__":
    main()
