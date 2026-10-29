"""Flow metrics derived from a streamflow record (manual table 3, eqs. 17-18).

`flow_metrics` reproduces pybfs 0.2.0 exactly, including where it departs
from the manual and the R code (suspected issues U1-U4, M13): default
recession rates when none can be estimated, Rb1/Rb2 from a quantile
regression instead of the median/95th percentile, and the 10-day recession
test that keeps only the "minimum of window" condition.
"""

from __future__ import annotations

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg

from .params import FlowMetrics

#: Window lengths [steps] for the storm-flow and baseflow recession rates.
STORM_WINDOW = {"day": 2, "hour": 48}
BASE_WINDOW = {"day": 10, "hour": 240}

#: Defaults used when a rate cannot be estimated (Python-only; R returns NA, issue U1).
DEFAULT_RS_NO_VALID_RATES = -0.05
DEFAULT_RS = -0.1
DEFAULT_RB1 = -0.002
DEFAULT_RB2 = -0.001


def flow_metrics(qin, timestep='day', fr4rise=0.05):
    """Calculate streamflow metrics used for baseflow separation

    Parameters
    ----------
    qin : array-like
        Streamflow time series (m³/day or m³/hour)
    timestep : str, optional
        'day' (default) or 'hour'
    fr4rise : float, optional
        Fraction for rise detection, default 0.05

    Returns
    -------
    list
        [Qthresh, Rs, Rb1, Rb2, Prec, Fr4Rise, rb10], where rb10 is the
        (intercept, slope) array of the 10th-percentile regression of 10-day
        recession rate on log10 Q. If the metrics cannot be computed, the first
        five values are NaN. See `pybfs.params.FlowMetrics` for definitions.
    """
    q = np.array(qin, dtype=float)
    n = len(q)
    q[q < 0] = np.nan

    prec = _precision(q, timestep)
    if timestep not in STORM_WINDOW:
        raise ValueError("timestep must be 'day' or 'hour'")
    window = STORM_WINDOW[timestep]
    _fill_short_gaps(q, window)

    rs = _storm_recession_rate(q, window, prec, fr4rise)
    base_window = BASE_WINDOW[timestep]
    change = _ten_day_change(q, base_window)
    log_q, log_rate = _base_recession_rates(q, change, base_window)

    positive = q[q > 0]
    if len(positive) == 0:
        return _invalid(fr4rise)
    q_thresh = np.nanmin(positive)
    q_mean = np.nanmean(q)
    if np.isnan(q_thresh) or np.isnan(q_mean) or q_thresh <= 0 or q_mean <= 0:
        return _invalid(fr4rise)
    q_thresh = max(q_thresh, _flow_with_slowest_recession(q, q_mean, change))

    rb10, log_q_fit = _recession_regression(log_q, log_rate)
    rb1 = _rate_at(rb10, np.log10(q_mean), DEFAULT_RB1) if _usable(rb10) else DEFAULT_RB1
    if _usable(rb10) and len(log_q_fit) > 0:
        log_q_p10 = np.nanquantile(log_q_fit, 0.1)
        rb2 = DEFAULT_RB2 if np.isnan(log_q_p10) else _rate_at(rb10, log_q_p10, DEFAULT_RB2)
    else:
        rb2 = DEFAULT_RB2

    values = [q_thresh, rs, rb1, rb2, prec]
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return _invalid(fr4rise)
    return [q_thresh, rs, rb1, rb2, prec, fr4rise, rb10]


def flow_metrics_record(qin, timestep='day', fr4rise=0.05) -> FlowMetrics:
    """`flow_metrics` as a `FlowMetrics` (``recession_curve`` = rb10)."""
    return FlowMetrics.from_legacy(flow_metrics(qin, timestep, fr4rise))


def _invalid(fr4rise):
    return [np.nan, np.nan, np.nan, np.nan, np.nan, fr4rise, np.array([np.nan, np.nan])]


def _precision(q, timestep):
    """Prec: the larger of the smallest step above the 1% flow and 10% of the 1% flow."""
    q01 = np.nanquantile(q, 0.01) if np.any(~np.isnan(q)) else np.nan
    above = q - q01
    above = above[above > 0]
    prec = np.nanmin(above) if len(above) > 0 else np.nan
    prec = max(prec, 0.1 * q01)
    if timestep == 'hour':  # also at least the median daily range
        n = len(q)
        ranges = []
        for t in range(n):
            lo, hi = max(0, t - 23), min(n, t + 1)
            if hi > lo:
                ranges.append(np.nanmax(q[lo:hi]) - np.nanmin(q[lo:hi]))
        if ranges:
            prec = max(prec, np.nanmedian(ranges))
    return prec


def _fill_short_gaps(q, window):
    """In place: replace missing values in gaps shorter than ``window`` by the mean of the neighbors."""
    n = len(q)
    steps = np.arange(1, n + 1)
    since_valid = steps - np.maximum.accumulate(steps * (~np.isnan(q)))
    until_valid = (steps[::-1] - np.maximum.accumulate(steps[::-1] * (~np.isnan(q[::-1]))))[::-1]
    gap = np.maximum(since_valid, until_valid)

    forward = q.copy()
    backward = q.copy()
    for y in range(1, n - 1):
        if np.isnan(forward[y]):
            forward[y] = forward[y - 1]
        if np.isnan(backward[n - y - 1]):
            backward[n - y - 1] = backward[n - y]
    fill = np.isnan(q) & (gap < window)
    q[fill] = (forward[fill] + backward[fill]) / 2


def _storm_recession_rate(q, window, prec, fr4rise):
    """Rs: log of the 95th percentile of post-rise recessions as a fraction of the rise, per step."""
    n = len(q)
    is_recession = np.zeros(n, dtype=bool)
    rise_size = np.zeros(n)
    is_window_peak = np.zeros(n, dtype=bool)
    for y in range(window, n):
        qp = q[y - window:y + 1][::-1]  # q[y], q[y-1], ..., q[y-window]
        if not np.any(np.isnan(qp)):
            dq = qp[0:window] - qp[1:window + 1]
            rise_threshold = max(fr4rise * qp[0], prec)
            rise_size[y] = np.nansum(dq)
            if rise_size[y] < rise_threshold:
                rise_size[y] = np.nan
            is_recession[y] = np.all(dq < prec) & (np.nansum(dq) < 0)
            is_window_peak[y] = (qp[0] == np.nanmax(qp))

    # Change over the window as a fraction of the rise at its start
    change = np.full(n, np.nan)
    if n > window:  # shorter records: no windows (the legacy loop runs zero times)
        with np.errstate(divide='ignore', invalid='ignore'):
            change[window:] = (q[window:] - q[:n - window]) / rise_size[:n - window]
    change[~is_recession] = np.nan
    change[~np.concatenate([np.zeros(window, dtype=bool), is_window_peak[:n - window]])] = np.nan

    valid = change[~np.isnan(change)]
    p95 = np.nanquantile(change, 0.95) if len(valid) > 0 else np.nan
    if np.isnan(p95):
        if is_recession.sum() > 0 and len(valid) == 0:
            return DEFAULT_RS_NO_VALID_RATES
        return DEFAULT_RS
    if p95 <= -1:
        p95 = -0.9  # 100% recession -> 90%, so the log is finite
    rs = np.log(1 + p95) / window
    if np.isnan(rs) or np.isinf(rs):
        return DEFAULT_RS
    return rs


def _ten_day_change(q, window):
    """Fractional change over ``window`` steps, on steps that are the minimum of their window."""
    n = len(q)
    is_window_min = np.zeros(n, dtype=bool)
    for y in range(window, n):
        qp = q[y - window:y + 1][::-1]
        if np.any(~np.isnan(qp)):
            is_window_min[y] = (qp[0] == np.nanmin(qp))  # the only test kept from R (issue U3)
    change = np.full(n, np.nan)
    if n > window:
        previous = q[:n - window]
        ok = previous > 0
        with np.errstate(divide='ignore', invalid='ignore'):  # quotients outside `ok` are discarded
            change[window:][ok] = ((q[window:] - previous) / previous)[ok]
    change[~is_window_min] = np.nan
    return change


def _base_recession_rates(q, change, window):
    """log10 Q and the per-step log10 recession rate on steps with a valid ``change``."""
    keep = (q > 0) & (~np.isnan(q)) & (~np.isnan(change)) & (change != -1)
    if not np.any(keep):
        return np.array([]), np.array([])
    return np.log10(q[keep]), np.log10(1 + change[keep]) / window


def _flow_with_slowest_recession(q, q_mean, change):
    """Among flow percentiles up to the mean, the one below which recessions are slowest (or -inf).

    ``change`` is the 10-day fractional change from `_ten_day_change`.
    """
    low = q[(q > 0) & (q <= q_mean)]
    if len(low) == 0:
        return -np.inf
    levels = np.quantile(low, np.arange(0, 1.01, 0.01))
    p95_rates = []
    for level in levels:
        rates = change[(q < level) & (~np.isnan(change)) & (~np.isinf(change)) & (change < 0)]
        p95_rates.append(np.nanquantile(rates, 0.95) if len(rates) > 0 and np.any(~np.isnan(rates)) else np.nan)
    if len(p95_rates) == 0 or np.all(np.isnan(p95_rates)):
        return -np.inf
    return levels[np.nanargmax(p95_rates)]


def _recession_regression(log_q, log_rate):
    """10th-percentile regression of log_rate on log_q; returns (rb10, log_q used)."""
    nan_pair = np.array([np.nan, np.nan])
    if len(log_q) == 0 or len(log_rate) == 0:
        return nan_pair, np.array([])
    valid = ~(np.isnan(log_q) | np.isnan(log_rate) | np.isinf(log_q) | np.isinf(log_rate))
    log_q, log_rate = log_q[valid], log_rate[valid]
    if len(log_rate) < 9 or len(log_q) < 9 or np.nanmin(log_q) == np.nanmax(log_q):
        return nan_pair, log_q
    try:
        fit = QuantReg(log_rate, np.column_stack([np.ones(len(log_q)), log_q.reshape(-1, 1)])).fit(q=0.1)
    except Exception:
        return nan_pair, log_q
    if len(fit.params) >= 2:
        return np.array([fit.params[0], fit.params[1]]), log_q
    if len(fit.params) == 1:
        return np.array([fit.params[0], 0.0]), log_q
    return nan_pair, log_q


def _usable(rb10):
    return not np.any(np.isnan(rb10)) and not np.any(np.isinf(rb10))


def _rate_at(rb10, log_q, cap):
    """Regression rate at log_q, limited to at most ``cap`` (a negative number)."""
    rate = rb10[0] + rb10[1] * log_q
    if np.isnan(rate) or np.isinf(rate):
        return cap
    return min(rate, cap)
