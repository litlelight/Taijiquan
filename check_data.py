from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import yaml

EXPECTED_MD5 = {
    "Labels.zip": "35de96a035c9d3b41619ac1f9932f946",
    "Kinect.zip": "d2802095beec3736ecfb35ab4eac70e8",
    "Segmented_Kinect.zip": "f783bbe1c8c19dbebb9ca3dbc2265f83",
    "TSV.zip": "904272e5599f51cb473ce8396c3bc4df",
    "Segmented_TSV.zip": "03969cc686f0a750437069c116ff4e21",
}
EXPECTED_COUNTS = {
    "labels": 131,
    "raw_kinect": 111,
    "raw_qualisys": 131,
    "segmented_kinect": 1815,
    "segmented_qualisys": 2149,
    "participants": 12,
}

ROOT = Path(__file__).resolve().parent


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def count_members(path: Path, suffix: str) -> int:
    with zipfile.ZipFile(path) as zf:
        return sum(
            name.lower().endswith(suffix.lower()) and not name.lower().endswith("license.txt")
            for name in zf.namelist()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the official UMONS-TAICHI inputs before running the analysis.")
    parser.add_argument("--config", default=str(ROOT / "config.yml"))
    parser.add_argument("--skip-md5", action="store_true", help="Skip the slower full-file MD5 pass.")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    def p(value: str) -> Path:
        x = Path(value)
        return x if x.is_absolute() else (base / x).resolve()

    paths = {
        "Labels.zip": p(cfg["paths"]["labels_zip"]),
        "Kinect.zip": p(cfg["paths"]["raw_kinect_zip"]),
        "Segmented_Kinect.zip": p(cfg["paths"]["segmented_kinect_zip"]),
        "TSV.zip": p(cfg["paths"]["raw_qualisys_zip"]),
        "Segmented_TSV.zip": p(cfg["paths"]["segmented_qualisys_zip"]),
    }
    metadata = p(cfg["paths"]["official_repo_root"]) / "Metadata.txt"
    missing = [str(x) for x in [metadata, *paths.values()] if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  " + "\n  ".join(missing) + "\nSee DATA.md.")

    checks = {}
    if not args.skip_md5:
        for name, path in paths.items():
            observed = md5(path)
            checks[f"md5:{name}"] = observed == EXPECTED_MD5[name]
            if observed != EXPECTED_MD5[name]:
                raise RuntimeError(f"MD5 mismatch for {name}: {observed} != {EXPECTED_MD5[name]}")

    counts = {
        "labels": count_members(paths["Labels.zip"], ".lab"),
        "raw_kinect": count_members(paths["Kinect.zip"], ".txt"),
        "raw_qualisys": count_members(paths["TSV.zip"], ".tsv"),
        "segmented_kinect": count_members(paths["Segmented_Kinect.zip"], ".txt"),
        "segmented_qualisys": count_members(paths["Segmented_TSV.zip"], ".tsv"),
        "participants": sum(line.startswith("P") for line in metadata.read_text(encoding="utf-8").splitlines()),
    }
    for key, expected in EXPECTED_COUNTS.items():
        if counts[key] != expected:
            raise RuntimeError(f"Unexpected {key} count: {counts[key]} != {expected}")
    result = {"status": "PASS", "counts": counts, "md5_checked": not args.skip_md5}
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
