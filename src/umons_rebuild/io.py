from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


KINECT_JOINTS = [
    "pelvis", "spine_mid", "neck", "head",
    "shoulder_l", "elbow_l", "wrist_l", "hand_l",
    "shoulder_r", "elbow_r", "wrist_r", "hand_r",
    "hip_l", "knee_l", "ankle_l", "foot_l",
    "hip_r", "knee_r", "ankle_r", "foot_r",
    "spine_shoulder", "handtip_l", "thumb_l", "handtip_r", "thumb_r",
]

COMMON_JOINTS = [
    "pelvis", "spine_shoulder", "head",
    "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
    "wrist_l", "wrist_r", "hip_l", "hip_r", "knee_l", "knee_r",
    "ankle_l", "ankle_r", "foot_l", "foot_r",
]

OFFICIAL_GESTURES = {
    "G01": "Beginning position (Wuji)",
    "G02": "Tree posture (Taiji)",
    "G03": "Open and close lotus flower",
    "G04": "Bring sky and earth together",
    "G05": "Canalize energy",
    "G06": "Drive the monkey away",
    "G07": "Move hands like clouds",
    "G08": "Part the wild horse's mane",
    "G09": "Golden rooster stands on one leg",
    "G10": "Fair lady works shuttles",
    "G11": "Kick with heel",
    "G12": "Brush knee and twist step",
    "G13": "Grasp the bird's tail",
}

LABEL_MAP = {
    "ex1": "G01D01", "ex2": "G02D01", "ex3": "G03D01", "ex4": "G04D01",
    "ex5l": "G05D01", "ex5r": "G05D02",
    "tech1l": "G06D01", "tech1r": "G06D02",
    "tech2l": "G07D01", "tech2r": "G07D02",
    "tech3l": "G08D01", "tech3r": "G08D02",
    "tech4l": "G09D01", "tech4r": "G09D02",
    "tech5l": "G10D01", "tech5r": "G10D02",
    "tech6l": "G11D01", "tech6r": "G11D02",
    "tech7l": "G12D01", "tech7r": "G12D02",
    "tech8l": "G13D01", "tech8r": "G13D02",
}

SEGMENT_RE = re.compile(
    r"^(?P<participant>P\d{2})(?P<trial>T\d{2})(?P<clip>C\d{2})"
    r"(?P<gesture>G\d{2})(?P<direction>D\d{2})(?P<sample>S\d{2})$"
)


@dataclass(frozen=True)
class Paths:
    official_repo_root: Path
    labels_zip: Path
    raw_kinect_zip: Path
    segmented_kinect_zip: Path
    raw_qualisys_zip: Path
    segmented_qualisys_zip: Path
    output_dir: Path


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_paths(cfg: dict, project_root: Path) -> Paths:
    p = cfg["paths"]
    config_root = Path(cfg.get("_config_path", project_root / "config.yml")).resolve().parent

    def portable(value: str) -> Path:
        expanded = Path(os.path.expandvars(value))
        return expanded.resolve() if expanded.is_absolute() else (config_root / expanded).resolve()

    return Paths(
        official_repo_root=portable(p["official_repo_root"]),
        labels_zip=portable(p["labels_zip"]),
        raw_kinect_zip=portable(p["raw_kinect_zip"]),
        segmented_kinect_zip=portable(p["segmented_kinect_zip"]),
        raw_qualisys_zip=portable(p["raw_qualisys_zip"]),
        segmented_qualisys_zip=portable(p["segmented_qualisys_zip"]),
        output_dir=(project_root / cfg["output_dir"]).resolve(),
    )


def md5(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def parse_metadata(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if not parts or not re.fullmatch(r"P\d{2}", parts[0]):
            continue
        skill_values = [float(parts[7]), float(parts[8]), float(parts[9])]
        rows.append({
            "participant_id": parts[0],
            "gender": parts[1],
            "age": float(parts[2]),
            "weight_kg": float(parts[3]),
            "height_cm": float(parts[4]),
            "practice_years": float(parts[5]),
            "category": parts[6],
            "skill1": skill_values[0],
            "skill2": skill_values[1],
            "skill3": skill_values[2],
            "skill_mean": float(parts[10]),
            "skill_mean_recomputed": float(np.mean(skill_values)),
        })
    result = pd.DataFrame(rows)
    if len(result) != 12:
        raise ValueError(f"Expected 12 participants, found {len(result)}")
    return result


def segment_id_parts(segment_id: str) -> dict[str, str]:
    m = SEGMENT_RE.fullmatch(segment_id)
    if not m:
        raise ValueError(f"Invalid segment id: {segment_id}")
    return m.groupdict()


def archive_segment_ids(zip_path: Path, suffix: str) -> set[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return {
            Path(name).stem
            for name in zf.namelist()
            if name.lower().endswith(suffix.lower()) and not name.lower().endswith("license.txt")
        }


def archive_recording_ids(zip_path: Path, suffix: str) -> set[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return {
            Path(name).stem
            for name in zf.namelist()
            if name.lower().endswith(suffix.lower()) and not name.lower().endswith("license.txt")
        }


def label_segments(labels_zip: Path) -> tuple[set[str], dict[str, list[dict]]]:
    expected: set[str] = set()
    by_recording: dict[str, list[dict]] = {}
    with zipfile.ZipFile(labels_zip) as zf:
        for name in sorted(n for n in zf.namelist() if n.lower().endswith(".lab")):
            recording_id = Path(name).stem
            text = zf.read(name).decode("utf-8")
            annotations = []
            counters = Counter()
            for line in text.splitlines():
                fields = line.split()
                if len(fields) != 2:
                    continue
                annotations.append({"time_s": float(fields[0]), "label": fields[1]})
            segments = []
            for i, ann in enumerate(annotations):
                label = ann["label"]
                if label == "_":
                    continue
                if label not in LABEL_MAP:
                    raise ValueError(f"Unknown label {label} in {recording_id}")
                mapped = LABEL_MAP[label]
                counters[mapped] += 1
                segment_id = f"{recording_id}{mapped}S{counters[mapped]:02d}"
                end_s = annotations[i + 1]["time_s"] if i + 1 < len(annotations) else None
                row = {
                    "segment_id": segment_id,
                    "recording_id": recording_id,
                    "start_s": ann["time_s"],
                    "end_s": end_s,
                    "label": label,
                }
                segments.append(row)
                expected.add(segment_id)
            by_recording[recording_id] = segments
    return expected, by_recording


def zip_member_map(zip_path: Path, suffix: str) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as zf:
        return {
            Path(name).stem: name
            for name in zf.namelist()
            if name.lower().endswith(suffix.lower()) and not name.lower().endswith("license.txt")
        }


def read_kinect_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    values = np.fromstring(data.decode("ascii"), sep=" ", dtype=np.float64)
    if values.size % 76:
        raise ValueError(f"Kinect value count {values.size} is not divisible by 76")
    matrix = values.reshape(-1, 76)
    timestamps_s = matrix[:, 0] / 1000.0
    positions = matrix[:, 1:].reshape(-1, 25, 3)
    return timestamps_s, positions


def read_kinect_member(zf: zipfile.ZipFile, member: str) -> tuple[np.ndarray, np.ndarray]:
    return read_kinect_bytes(zf.read(member))


def read_qualisys_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    lines = data.splitlines()
    if len(lines) < 11:
        raise ValueError("Qualisys TSV has fewer than 11 lines")
    header = {}
    for line in lines[:10]:
        fields = line.decode("utf-8", errors="replace").split("\t")
        header[fields[0]] = fields[1:]
    n_frames = int(header["NO_OF_FRAMES"][0])
    n_markers = int(header["NO_OF_MARKERS"][0])
    frequency = float(header["FREQUENCY"][0])
    marker_names = [x for x in header["MARKER_NAMES"] if x]
    body = b"\n".join(lines[10:]).decode("ascii")
    values = np.fromstring(body, sep="\t", dtype=np.float64)
    expected = n_frames * n_markers * 3
    if values.size != expected:
        raise ValueError(f"Qualisys values={values.size}, expected={expected}")
    positions = values.reshape(n_frames, n_markers, 3)
    times_s = np.arange(n_frames, dtype=np.float64) / frequency
    return times_s, positions, marker_names, frequency


def read_qualisys_member(zf: zipfile.ZipFile, member: str):
    return read_qualisys_bytes(zf.read(member))


def qualisys_header(zf: zipfile.ZipFile, member: str) -> dict[str, object]:
    with zf.open(member) as stream:
        lines = [stream.readline().decode("utf-8", errors="replace").rstrip("\r\n") for _ in range(10)]
    values = {parts[0]: parts[1:] for parts in (line.split("\t") for line in lines)}
    return {
        "n_frames": int(values["NO_OF_FRAMES"][0]),
        "n_markers": int(values["NO_OF_MARKERS"][0]),
        "frequency": float(values["FREQUENCY"][0]),
        "marker_names": [x for x in values["MARKER_NAMES"] if x],
    }
