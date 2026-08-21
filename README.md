# UMONS-TAICHI feature-validation reproducibility package

Companion code for the revised manuscript:

**Feature-dependent agreement between markerless and marker-based motion capture during continuous Taijiquan movement**

This repository contains the **single frozen analysis pipeline** used for the revised results. It intentionally does **not** include the UMONS-TAICHI dataset, generated multi-gigabyte artifacts, development notebooks, obsolete analysis variants, or internal revision history.

## What this repository reproduces

- source-level reconstruction of the 1,815 strict Kinect–Qualisys pairs;
- timestamp-aware preprocessing and kinematic feature extraction;
- ICC(A,1), ICC(C,1), CCC, bias/error and mechanism-focused sensitivity analyses;
- participant-isolated per-feature calibration with an intercept-only diagnostic;
- participant-level nested-LOSO ranking;
- exact random-segment leakage analysis (200 repeats);
- Qualisys repetition equalization (500 repeats);
- 10,000 full-pipeline participant-label permutations with Holm correction;
- participant jackknife and random-forest training-range safeguards.

The scientific choices are summarized in [`ANALYSIS_SPEC.md`](ANALYSIS_SPEC.md).

## 1. Get the official data

See [`DATA.md`](DATA.md) for direct official download links and Zenodo MD5 checksums. The required archives are **not committed to GitHub**. After placing them under `data/`, run:

```bash
python check_data.py
```

The checker verifies archive identity and the released-file counts used by the manuscript.

## 2. Create the environment

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Windows:

```powershell
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

## 3. Fast code sanity check

The data-independent tests run in seconds; the official-metadata test is automatically enabled once the dataset repository is present.

```bash
python -m pytest -q
```

The frozen release contains 17 unit tests covering the central agreement, feature, synchronization, calibration, split, and random-forest invariants.

## 4. Reproduce the paper analysis

One command is canonical:

```bash
python run_analysis.py --clean
```

The exact frozen settings are 2,000 participant-cluster bootstrap repetitions, 200 random segment splits, 500 repetition equalizations, and 10,000 full-pipeline permutations per pre-specified Ridge reference condition. The full run is I/O- and compute-intensive because it reads the original motion-capture archives and repeats the nested participant-level analyses.

After the first successful input validation, `--skip-md5` can be used to avoid re-hashing the multi-gigabyte source archives:

```bash
python run_analysis.py --clean --skip-md5
```

## 5. Verify the regenerated headline results

`run_analysis.py` ends by running this automatically, but it can also be called directly:

```bash
python verify_key_results.py
```

The verifier compares regenerated headline values with the compact frozen reference in `reference/key_results.json` at an absolute tolerance of `1e-8`. It checks the principal agreement results, hip mapping sensitivity, calibration, residual-lag sensitivity, final ranking/permutation results, exact leakage inflation, repetition equalization, and the random-forest training-range invariant.

A successful complete run ends with:

```text
KEY_RESULT_VERIFICATION_PASS
REPRODUCTION_PASS
```

## Repository layout

```text
.
├── README.md
├── DATA.md
├── ANALYSIS_SPEC.md
├── config.yml
├── requirements.txt
├── run_analysis.py              # only canonical full-analysis entry point
├── check_data.py                # validates official inputs before analysis
├── verify_key_results.py        # compact frozen-result verification
├── reference/key_results.json
├── src/umons_rebuild/           # analysis implementation
├── scripts/                     # internal stage runners used by run_analysis.py
└── tests/test_core.py
```

Generated `artifacts_*` directories and raw `data/` are ignored by Git.

## Reproducibility status

Before this streamlined GitHub release was prepared, the frozen pipeline passed a clean-room rerun in a newly created environment: 17 unit tests passed, 15 frozen output tables matched, and the maximum numerical difference was `5.329e-15` (tolerance `1e-10`). Those bulky audit outputs are not duplicated here; this public package instead exposes a compact key-result verifier plus the deterministic source code. Superseded diagnostic split/permutation runs are also omitted from the public execution path: the exact 1,815-segment leakage experiment and the final 10,000-permutation tests are each executed once, in the final-closure stage.

## Dataset citation

Tits, M., Laraba, S., Caulier, E., Tilmanne, J. & Dutoit, T. UMONS-TAICHI: a multimodal motion capture dataset of expertise in Taijiquan gestures. *Data in Brief* 19, 1214–1221 (2018). https://doi.org/10.1016/j.dib.2018.05.088

## Code citation

Please cite the accompanying manuscript and the archived repository DOI once the GitHub release is deposited on Zenodo.
