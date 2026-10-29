"""The error measures of the USGS BFS procedure (pybfs 0.2.0), on plain arrays.

Each function reproduces the corresponding legacy computation exactly; the
docstrings name the legacy function and the behavior issues they inherit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def relative_error(q_obs: Array, q_sim: Array, flow_precision: float) -> Array:
    """Adjusted relative error ``(q_sim − q_obs)/(q_obs + Prec)`` ("AdjPctEr").

    Zero where both flows are zero. Legacy: the ``APE`` computation in ``bfs``.
    """
    error = (q_sim - q_obs) / (q_obs + flow_precision)
    error[(q_obs == 0) & (q_sim == 0)] = 0
    return error


def recession_weights(rel_error: Array, recession_age: NDArray[np.int64], storm_recession_rate: float) -> Array:
    """Error weights ``1 − exp(Rs·age)``, set to 1 where the model over-predicts ("Weight").

    The weight grows from 0 on rising limbs toward 1 in long recessions.
    Legacy: the ``Weight`` computation in ``bfs``. The manual (table 3) weights
    with Rb1 or Rb2 instead of Rs (suspected issue M13).
    """
    weight = 1 - np.exp(storm_recession_rate * recession_age)
    weight[rel_error > 0] = 1
    return weight


def usgs_weighted_error(q_obs: Array, q_sim: Array, *, rel_error: Array, weights: Array,
                        flow_precision: float) -> float:
    """Calibration objective of the USGS procedure, Σ W·(APE² − 1); lower is better.

    Days where the model over-predicts by more than ``Prec`` get weight 0, and
    so do days where both ``q_obs`` and ``q_sim`` are 0. In the legacy procedure
    ``q_sim`` is the baseflow and ``rel_error`` is computed from it. Legacy:
    ``calibrate.objective`` (see suspected issue C12 on its asymmetry).
    """
    w = weights.copy()
    w[rel_error > flow_precision / (q_obs + flow_precision)] = 0
    w[(q_obs == 0) & (q_sim == 0)] = 0
    return np.nansum(w * (-1 + rel_error**2))


def weighted_mean_abs_error(rel_error: Array, weights: Array) -> float:
    """Σ|APE·W| / ΣW, the model error reported by the procedure ("Error").

    Legacy: ``calibrate.calculate_error``. NaN if all weights are zero.
    """
    if np.nansum(weights) > 0:
        return np.nansum(np.abs(rel_error * weights)) / np.nansum(weights)
    return np.nan


def recession_rate_mismatch(table_storage: Array, table_discharge: Array, q_levels: Array,
                            recession_curve: tuple[float, float]) -> float:
    """Mismatch between the base table's recession rate and the empirical curve.

    For table rows at the discharges ``q_levels`` (log-spaced from Qmin/10 to
    Qmean), compares the linear-reservoir rate ``Q/S`` with the 10th-percentile
    recession rate ``RbI + RbS·log10 Q``, and returns the mean absolute
    difference [1/day]. Returns 100 if fewer than 10 usable rows. Legacy: the
    second half of ``calibrate.cal_basetable`` (the bounds checks and the
    ``Qthresh`` viability test live in `pybfs.calibration.usgs`).
    """
    intercept, slope = recession_curve
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = table_discharge / table_storage
    rate[np.isnan(rate)] = 0
    rate[np.isinf(rate)] = 0

    rows = np.maximum(np.searchsorted(table_discharge, q_levels, side="right") - 1, 0)
    q_rows = table_discharge[rows]
    positive = q_rows > 0
    if np.any(positive):
        receding = np.zeros(len(rows), dtype=bool)
        receding[positive] = (intercept + slope * np.log10(q_rows[positive])) < 0
        rows = np.unique(rows[positive & receding])
    else:
        rows = []
    if len(rows) < 10:
        return 100.0
    mismatch = np.nanmean(np.abs(rate[rows] + (intercept + slope * np.log10(table_discharge[rows]))))
    if np.isinf(mismatch) or np.isnan(mismatch):
        return 100.0
    return mismatch
