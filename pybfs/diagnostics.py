# -*- coding: utf-8 -*-
"""Diagnostics on model output: flow fractions, credible intervals, and skill.

These work on pandas tables (the legacy ``bfs()``/``forecast()`` layouts) and
reproduce pybfs 0.2.0 exactly, including the behavior flagged in
``notes/suspected_issues.md`` (C14, U7, S1-S3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bfs import bfs
from .utilities import forecast

CI_COLUMNS = ['Qsim.L3.T', 'FrEr0.05', 'FrEr0.50', 'FrEr0.95']


# ---------------------------------------------------------------------------
# Flow fractions
# ---------------------------------------------------------------------------

def baseflow_fraction(q_obs, baseflow, cap_at_observed: bool = True) -> float:
    """Σ baseflow / Σ q_obs over days with observations.

    With ``cap_at_observed`` (as in the calibration's parameter-set rows),
    baseflow is limited to the observed flow on each day; without it, the
    plain ratio of sums (the procedure's initial row) is returned.
    """
    q_obs = np.asarray(q_obs)
    baseflow = np.asarray(baseflow)
    if not cap_at_observed:
        return np.nansum(baseflow) / np.nansum(q_obs)
    capped = baseflow.copy()
    over = baseflow > q_obs
    capped[over] = q_obs[over]
    observed = ~np.isnan(q_obs)
    return np.nansum(capped[observed]) / np.nansum(q_obs[observed])


def flow_fractions(bfs_out: pd.DataFrame) -> dict[str, float]:
    """Mean flow and baseflow, surface-flow, and direct-runoff fractions.

    Uses days with positive observed and finite simulated flow. Daily baseflow
    fractions are capped at 1, and the surface-flow fraction is defined as
    ``1 − baseflow fraction`` on every such day, so BFF + SFF + DRF can exceed 1
    (suspected issue C14). Fractions are rounded to 3 decimals, as before.
    """
    keep = bfs_out['Qob'] > 0
    keep[np.isnan(keep)] = False
    keep[np.isnan(bfs_out['Qsim'])] = False
    days = bfs_out[keep].copy()
    q_obs = days['Qob']

    bf_fraction = days['Baseflow'] / q_obs
    bf_fraction[bf_fraction > 1] = 1
    sf_fraction = days['SurfaceFlow'] / q_obs
    sf_fraction[(sf_fraction + bf_fraction) > 0] = 1 - bf_fraction[(sf_fraction + bf_fraction) > 0]

    total = np.nansum(q_obs)
    return {
        'Qmean': np.round(np.nanmean(q_obs), 6),
        'BFF': np.round(np.nansum(bf_fraction * q_obs) / total, 3),
        'SFF': np.round(np.nansum(sf_fraction * q_obs) / total, 3),
        'DRF': np.round(np.nansum(days['DirectRunoff']) / total, 3),
    }


# ---------------------------------------------------------------------------
# Credible intervals
# ---------------------------------------------------------------------------

def bf_ci(bfs_out):
    """Calculate credible intervals for baseflow separation

    Empirical 5-95% intervals of the fractional streamflow error
    (Qsim − Qob)/Qsim, by quantile of simulated flow, using days without direct
    runoff.

    Parameters
    ----------
    bfs_out : pd.DataFrame
        Output of ``bfs()`` (columns Qsim, Qob, DirectRunoff; the R names
        ``Qsim.L3`` etc. are also accepted).

    Returns
    -------
    tuple
        (ci_table, ci): the table of error quantiles by flow quantile
        (columns Qsim.L3.T, FrEr0.05, FrEr0.50, FrEr0.95) and the daily bounds
        CB0.05, CB0.95 [m³/day].
    """
    q_sim_col = 'Qsim' if 'Qsim' in bfs_out.columns else 'Qsim.L3'
    q_obs_col = 'Qob' if 'Qob' in bfs_out.columns else 'Qob.L3'
    runoff_col = 'DirectRunoff' if 'DirectRunoff' in bfs_out.columns else 'DirectRunoff.L3'
    q_sim = bfs_out[q_sim_col]
    has_runoff = bfs_out[runoff_col] > 0

    frac_error = (q_sim - bfs_out[q_obs_col]) / q_sim
    frac_error[np.abs(frac_error) == np.inf] = 1
    frac_error[has_runoff] = np.nan
    q_no_runoff = q_sim.copy()
    q_no_runoff[has_runoff] = np.nan

    positive = q_no_runoff[q_no_runoff > 0]
    q_quantiles = np.quantile(positive, np.arange(0.05, 1.0, 0.05)) if len(positive) > 0 else np.array([])

    # Error quantiles for each flow quantile, over the band between its neighbors.
    # Bins with no valid errors are skipped, which can shift later row positions (issue U7).
    rows = []
    for k in range(1, 18):
        if k < len(q_quantiles) - 1:
            in_band = (q_no_runoff > q_quantiles[k - 1]) & (q_no_runoff < q_quantiles[k + 1])
            band_errors = frac_error[in_band]
            if np.any(~np.isnan(band_errors)):
                rows.append([q_quantiles[k], np.nanquantile(band_errors, 0.05),
                             np.nanquantile(band_errors, 0.50), np.nanquantile(band_errors, 0.95)])
    ci_table = pd.DataFrame(rows, columns=CI_COLUMNS) if rows else pd.DataFrame(columns=CI_COLUMNS)

    # Daily bounds from the bin of each day's simulated flow
    bin_index = np.full(len(q_no_runoff), np.nan)
    if len(q_quantiles) >= 19 and len(ci_table) > 0:
        bin_edges = (q_quantiles[:18] + q_quantiles[1:19]) / 2
        for t in range(len(q_no_runoff)):
            if bfs_out[runoff_col].iloc[t] == 0:
                below = bfs_out[q_sim_col].iloc[t] < bin_edges
                if np.any(below):
                    bin_index[t] = np.where(below)[0][0]

    ci = pd.DataFrame(index=bfs_out.index)
    ci['CB0.05'] = np.nan
    ci['CB0.95'] = np.nan
    if len(ci_table) > 0:
        for t in range(len(q_no_runoff)):
            if not np.isnan(bin_index[t]) and int(bin_index[t]) < len(ci_table):
                k = int(bin_index[t])
                ci.loc[ci.index[t], 'CB0.05'] = bfs_out[q_sim_col].iloc[t] * (1 - ci_table['FrEr0.95'].iloc[k])
                ci.loc[ci.index[t], 'CB0.95'] = bfs_out[q_sim_col].iloc[t] * (1 - ci_table['FrEr0.05'].iloc[k])
    return (ci_table, ci)


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

def modified_strict_baseflow(hydrograph, quantile=0.8):
    """Identify baseflow-dominated days using the modified strict baseflow method.

    A statistical method that uses streamflow derivatives and a quantile threshold
    to flag days where observed streamflow is likely composed entirely of baseflow.
    Does not use any pybfs model parameters.

    Parameters
    ----------
    hydrograph : pd.DataFrame
        Observed streamflow with columns 'Date' (datetime) and 'Streamflow' (m³/day).
    quantile : float, optional
        Quantile value for the strict baseflow filter (default 0.8).

    Returns
    -------
    np.ndarray
        Boolean array of length len(hydrograph). True = baseflow day, False = non-baseflow day.

    Notes
    -----
    Missing days are treated as zero flow for the derivatives and threshold
    (suspected issue S2) and are never flagged as baseflow days.
    """
    q_obs = np.array(hydrograph['Streamflow'], dtype=float)
    missing = np.isnan(q_obs)
    q_filled = q_obs.copy()
    q_filled[missing] = 0.0

    is_baseflow_day = np.zeros(len(q_obs), dtype=bool)
    is_strict = _strict_baseflow(q_filled, quantile)
    if is_strict.sum() > 0:
        below_strict_mean = np.where(q_filled <= np.mean(q_filled[is_strict]))[0]
        is_baseflow_day[np.union1d(below_strict_mean, np.where(is_strict)[0])] = True
    is_baseflow_day[missing] = False
    return is_baseflow_day


def _strict_baseflow(q, quantile):
    """Days that pass all four strict-baseflow exclusion rules (True = baseflow)."""
    dq_central = (q[2:] - q[:-2]) / 2
    last = q.shape[0] - 1

    # 1. Rising or flat days
    rising = np.concatenate([[True], dq_central >= 0, [True]])

    # 2. Two days before each rising segment and three days after it
    rise_starts = np.where(rising[1:].astype(int) - rising[:-1].astype(int) == 1)[0] + 1
    rise_ends = np.where(rising[1:].astype(int) - rising[:-1].astype(int) == -1)[0]
    before = np.repeat([rise_starts], 2) - np.tile(range(1, 3), rise_starts.shape)
    after = np.repeat([rise_ends], 3) + np.tile(range(1, 4), rise_ends.shape)
    near_rise = np.full(q.shape, False)
    near_rise[np.concatenate([before, after]).clip(min=0, max=last)] = True

    # 3. Accelerating recessions in the upper quantile of flow, with a buffer
    accelerating = np.concatenate([[True], dq_central[1:] - dq_central[:-1] < 0, [True, True]])
    candidates = np.where(accelerating)[0]
    high_flow_days = candidates[q[candidates] >= np.quantile(q, quantile)]
    buffered = np.repeat([high_flow_days], 10) + np.tile(range(-4, 6), high_flow_days.shape)
    high_recession = np.full(q.shape, False)
    high_recession[buffered.clip(min=0, max=last)] = True

    # 4. Followed by a larger -dQ/dt (same expression as rule 3's candidates; issue S3)
    steepening = np.concatenate([[True], dq_central[1:] - dq_central[:-1] < 0, [True, True]])

    return ~(rising | near_rise | high_recession | steepening)


def separation_skill(hydrograph, SBT, basin_char, gw_hyd, flow, quantile=0.8):
    """Evaluate how well pybfs baseflow separation matches the modified strict baseflow reference.

    On days identified as baseflow-dominated by the modified strict method, observed
    streamflow Q is treated as the reference baseflow. Residuals Q - BF_bfs measure
    how well pybfs matches this reference.

    Parameters
    ----------
    hydrograph : pd.DataFrame
        Observed streamflow with columns 'Date' (datetime) and 'Streamflow' (m³/day).
    SBT : pd.DataFrame
        Baseflow table from base_table() with columns ['Xb','Z','S','Q'].
    basin_char, gw_hyd, flow : list
        Same as bfs().
    quantile : float, optional
        Quantile value used in modified strict baseflow (default 0.8).

    Returns
    -------
    skill_df : pd.DataFrame
        Daily record with columns: Date, Q, BF_strict, BF_bfs, RES.
    metrics : dict
        Overall metrics: RMSE, MAE (m³/day), n_days, frac_strict.
    """
    is_baseflow_day = modified_strict_baseflow(hydrograph, quantile=quantile)
    simulated = bfs(hydrograph, SBT, basin_char, gw_hyd, flow)

    skill_df = pd.DataFrame({
        'Date': pd.to_datetime(hydrograph['Date'].values),
        'Q': np.array(hydrograph['Streamflow'], dtype=float),
        'BF_strict': is_baseflow_day,
        'BF_bfs': simulated['Baseflow'].values,
        'RES': np.nan,
    })
    scored = skill_df['BF_strict'] & skill_df['Q'].notna() & skill_df['BF_bfs'].notna()
    skill_df.loc[scored, 'RES'] = skill_df.loc[scored, 'Q'] - skill_df.loc[scored, 'BF_bfs']

    residuals = skill_df.loc[scored, 'RES'].values
    n_days = int(scored.sum())
    metrics = {
        'RMSE': float(np.sqrt(np.mean(residuals ** 2))) if n_days else np.nan,
        'MAE': float(np.mean(np.abs(residuals))) if n_days else np.nan,
        'n_days': n_days,
        'frac_strict': n_days / len(skill_df) if len(skill_df) > 0 else np.nan,
    }
    return skill_df, metrics


def forecast_skill(hydrograph, SBT, basin_char, gw_hyd, flow,
                   min_days=30, max_days=90, min_sat=0.9, train_days=365,
                   quantile=0.8):
    """Evaluate pybfs forecast skill during naturally-occurring baseflow-dominated periods.

    Identifies contiguous sequences of baseflow-dominated days (via modified_strict_baseflow),
    runs a pybfs forecast initialized from observed conditions before each sequence, and
    computes error metrics comparing the forecasted baseflow to observed streamflow.

    Parameters
    ----------
    hydrograph : pd.DataFrame
        Observed streamflow with columns 'Date' (datetime) and 'Streamflow' (m³/day).
    SBT : pd.DataFrame
        Baseflow table from base_table() with columns ['Xb','Z','S','Q'].
    basin_char, gw_hyd, flow : list
        Same as bfs().
    min_days : int, optional
        Minimum sequence length in days to qualify (default 30).
    max_days : int, optional
        Maximum forecast period; longer sequences are truncated (default 90).
    min_sat : float, optional
        Minimum fraction of baseflow days within a sequence. Has no effect,
        because sequences are contiguous runs of baseflow days (issue S1).
    train_days : int, optional
        Number of days of observed data before the sequence used to initialize the
        forecast. Uses all available data if fewer days exist (default 365).
    quantile : float, optional
        Quantile value used in modified strict baseflow (default 0.8).

    Returns
    -------
    skill_df : pd.DataFrame
        Daily record with columns: Date, Q, BF, SEQ, FC, RES.
    summary_df : pd.DataFrame
        One row per qualifying sequence with columns: SEQ, LEN, SAT, RMSE, MAE.
    metrics : dict
        Overall metrics across all sequences: overall_RMSE, overall_MAE.
    """
    hydrograph = hydrograph.reset_index(drop=True)
    dates = pd.to_datetime(hydrograph['Date'].values)
    q_obs = np.array(hydrograph['Streamflow'], dtype=float)
    n = len(hydrograph)
    is_baseflow_day = modified_strict_baseflow(hydrograph, quantile=quantile)

    skill_df = pd.DataFrame({'Date': dates, 'Q': q_obs, 'BF': is_baseflow_day,
                             'SEQ': np.nan, 'FC': np.nan, 'RES': np.nan})

    sequences = []  # (first, last) day of each qualifying run, inclusive
    i = 0
    while i < n:
        if is_baseflow_day[i]:
            j = i
            while j < n and is_baseflow_day[j]:
                j += 1
            run_length = j - i
            run_share = run_length / run_length  # always 1.0 (issue S1)
            if run_length >= min_days and run_share >= min_sat:
                sequences.append((i, j - 1))
            i = j
        else:
            i += 1

    summary_rows = []
    sequence_id = 0
    for first, last in sequences:
        sequence_id += 1
        length = min(last - first + 1, max_days)
        end = first + length - 1
        skill_df.loc[first:end, 'SEQ'] = sequence_id

        spinup = hydrograph.iloc[max(0, first - train_days):first].reset_index(drop=True)
        if len(spinup) == 0:
            skill_df.loc[first:end, 'SEQ'] = np.nan
            sequence_id -= 1
            continue

        last_day = bfs(spinup, SBT, basin_char, gw_hyd, flow).iloc[-1]
        initial = tuple(last_day[['X', 'Zb.L', 'Zs.L', 'StBase', 'StSur', 'SurfaceFlow', 'Baseflow', 'Rech']])
        forecast_input = pd.DataFrame({'date': dates[first:first + length], 'streamflow': np.nan})
        predicted = forecast(forecast_input, SBT, basin_char, gw_hyd, flow, initial)['Baseflow'].values

        skill_df.loc[first:end, 'FC'] = predicted
        skill_df.loc[first:end, 'RES'] = skill_df.loc[first:end, 'Q'].values - predicted

        residuals = skill_df.loc[first:end, 'RES'].dropna().values
        summary_rows.append({
            'SEQ': sequence_id,
            'LEN': length,
            'SAT': float(is_baseflow_day[first:first + length].sum()) / length,
            'RMSE': float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) > 0 else np.nan,
            'MAE': float(np.mean(np.abs(residuals))) if len(residuals) > 0 else np.nan,
        })

    summary_df = (pd.DataFrame(summary_rows) if summary_rows
                  else pd.DataFrame(columns=['SEQ', 'LEN', 'SAT', 'RMSE', 'MAE']))
    all_residuals = skill_df['RES'].dropna().values
    metrics = {
        'overall_RMSE': float(np.sqrt(np.mean(all_residuals ** 2))) if len(all_residuals) > 0 else np.nan,
        'overall_MAE': float(np.mean(np.abs(all_residuals))) if len(all_residuals) > 0 else np.nan,
    }
    return skill_df, summary_df, metrics
