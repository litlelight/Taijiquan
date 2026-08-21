from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _paired(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def icc_a1(x, y) -> float:
    values = np.column_stack(_paired(x, y))
    n, k = values.shape
    if n < 2:
        return np.nan
    grand = values.mean()
    rows = values.mean(axis=1)
    columns = values.mean(axis=0)
    msr = k * np.sum((rows - grand) ** 2) / (n - 1)
    msc = n * np.sum((columns - grand) ** 2) / (k - 1)
    residual = values - rows[:, None] - columns[None, :] + grand
    mse = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denominator) if denominator else np.nan


def icc_c1(x, y) -> float:
    values = np.column_stack(_paired(x, y))
    n, k = values.shape
    if n < 2:
        return np.nan
    grand = values.mean()
    rows = values.mean(axis=1)
    columns = values.mean(axis=0)
    msr = k * np.sum((rows - grand) ** 2) / (n - 1)
    residual = values - rows[:, None] - columns[None, :] + grand
    mse = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse
    return float((msr - mse) / denominator) if denominator else np.nan


def ccc(x, y) -> float:
    x, y = _paired(x, y)
    if len(x) < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    covariance = np.cov(x, y, ddof=1)[0, 1]
    denominator = vx + vy + (np.mean(x) - np.mean(y)) ** 2
    return float(2 * covariance / denominator) if denominator else np.nan


def metrics(x, y) -> dict[str, float]:
    x, y = _paired(x, y)
    difference = y - x
    mean_pair = (x + y) / 2.0
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    if len(x) > 2 and np.std(mean_pair) > 0:
        slope, intercept, pearson_diff_mean, slope_p, slope_se = stats.linregress(mean_pair, difference)
    else:
        slope = intercept = pearson_diff_mean = slope_p = slope_se = np.nan
    return {
        "n_units": int(len(x)),
        "icc_a1": icc_a1(x, y),
        "icc_c1": icc_c1(x, y),
        "ccc": ccc(x, y),
        "pearson_r": float(stats.pearsonr(x, y).statistic) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "bias": bias,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "proportional_bias_slope": float(slope),
        "proportional_bias_intercept": float(intercept),
        "difference_mean_r": float(pearson_diff_mean),
        "proportional_bias_p": float(slope_p),
        "proportional_bias_slope_se": float(slope_se),
    }


BOOTSTRAP_NAMES = [
    "icc_a1", "icc_c1", "ccc", "pearson_r", "mae", "rmse", "bias",
    "proportional_bias_slope",
]


def cluster_bootstrap(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    cluster_names = np.asarray(sorted(frame[cluster_col].unique()))
    blocks = {
        name: (group[x_col].to_numpy(float), group[y_col].to_numpy(float))
        for name, group in frame.groupby(cluster_col)
    }
    rng = np.random.default_rng(seed)
    values = {name: [] for name in BOOTSTRAP_NAMES}
    for _ in range(repetitions):
        selected = rng.choice(cluster_names, size=len(cluster_names), replace=True)
        x = np.concatenate([blocks[name][0] for name in selected])
        y = np.concatenate([blocks[name][1] for name in selected])
        result = metrics(x, y)
        for name in BOOTSTRAP_NAMES:
            values[name].append(result[name])
    output = {}
    for name, samples in values.items():
        output[f"{name}_ci_lower"] = float(np.nanpercentile(samples, 2.5))
        output[f"{name}_ci_upper"] = float(np.nanpercentile(samples, 97.5))
    return output


def agreement_rows(
    paired_long: pd.DataFrame,
    group_columns: list[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for index, (keys, group) in enumerate(paired_long.groupby(group_columns, dropna=False)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(metrics(group["qualisys"], group["kinect"]))
        row.update(cluster_bootstrap(group, "qualisys", "kinect", "participant_id", repetitions, seed + index))
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result["loa_midpoint_error"] = np.abs((result.loa_lower + result.loa_upper) / 2 - result.bias)
    return result
