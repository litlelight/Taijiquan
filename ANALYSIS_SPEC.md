# Frozen analysis specification

This file records the analysis decisions used for the revised manuscript. It is intentionally concise; implementation details live in `src/umons_rebuild/`.

## Analysis population

- 12 participants and 13 gesture classes.
- 2,149 released segmented Qualisys trials.
- 1,815 strict Kinect–Qualisys segment pairs; 334 Qualisys-only segments.
- Primary agreement is summarized at participant × gesture level and uncertainty is participant-clustered.
- Participant-level ranking has **N = 12 independent labels**.

## Primary measurement pipeline

- Source coordinates are converted from mm to m.
- Participant-specific, sensor-specific trunk scale is fixed across all of that participant's segments.
- Kinect uses recorded timestamps, resampling to a uniform 30-Hz physical-time grid, then a 6-Hz zero-phase low-pass filter.
- Primary Qualisys reference is bandwidth-matched and resampled for paired feature extraction.
- Primary agreement uses ICC(A,1), ICC(C,1), CCC, Pearson r, MAE, RMSE, bias, Bland–Altman limits, and participant-cluster bootstrap uncertainty.

## Frozen ranking representation

The primary participant-wide representation contains 13 features × 13 gestures = 169 predictors. `p95_angular_velocity_deg_s` is included; `duration_s`, `peak_angular_velocity_deg_s`, and `rms_angular_velocity_deg_s` are not part of the primary feature set.

Frozen on 2026-08-20 before inspecting any R22–R27 v3 ranking result.

## Calibration

Per-feature affine Ridge calibration maps Kinect feature j to Qualisys feature j inside an outer participant LOSO. The held participant is excluded from fitting, scaling, and inner selection. Intercept-only LOSO prediction is the baseline used to distinguish signal recovery from shrinkage. Calibrated Kinect is a measurement experiment and is not a third ranking sensor condition.

## Final closure

- Random splitting uses 200 seeded repetitions.
- Qualisys repetition equalization is repeated 500 times.
- Full-pipeline participant-label permutation is repeated 10,000 times for each of the two pre-specified Ridge reference conditions.
- Holm correction is applied across those two reference conditions.
- Random-forest predictions are asserted to remain within the outer-fold training-label range.

The exact segment-level random-split experiment and the 10,000-permutation analysis supersede earlier diagnostic split/permutation variants for the final manuscript.
