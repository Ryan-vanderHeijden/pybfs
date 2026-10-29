# -*- coding: utf-8 -*-
"""Calibration functions: the pybfs 0.2.0 ``calibrate`` interface.

Compatibility wrappers around `pybfs.calibration.usgs`. Names, signatures and
results are unchanged; new code should use ``pybfs.calibration`` directly.
"""

import numpy as np
import pandas as pd

from .calibration import usgs as _usgs
from .calibration.problem import Record
from .diagnostics import bf_ci, flow_fractions
from .frames import legacy_bfs_frame
from .bfs import _require_daily
from .objectives.usgs import usgs_weighted_error, weighted_mean_abs_error
from .params import FlowMetrics, ModelParameters


def calculate_error(bfs_out_df):
    """Mean absolute weighted error Σ|APE·W|/ΣW of a ``bfs()`` result (NaN if ΣW = 0)."""
    return weighted_mean_abs_error(bfs_out_df['AdjPctEr'].values, bfs_out_df['Weight'].values)


def objective(bfs_out_df, prec):
    """Objective function for calibration

    Σ W·(APE² − 1) over the ``bfs()`` result, ignoring days where the model
    over-predicts by more than ``prec`` and days with zero observed flow and
    baseflow. Lower is better. See `pybfs.objectives.usgs.usgs_weighted_error`.

    Parameters
    ----------
    bfs_out_df : pd.DataFrame
        Output from bfs() function
    prec : float
        Precision threshold (m³/day)

    Returns
    -------
    float
    """
    q_obs = bfs_out_df['Qob' if 'Qob' in bfs_out_df.columns else 'Qob.L3'].values
    baseflow = bfs_out_df['Baseflow' if 'Baseflow' in bfs_out_df.columns else 'Baseflow.L3'].values
    return usgs_weighted_error(q_obs, baseflow, bfs_out_df['AdjPctEr'].values, bfs_out_df['Weight'].values,
                               flow_precision=prec)


def ini_params(area, lb, x1, wb, por, beta, rb1, tmp_q):
    """Set initial values of Lb, Wb, and Kb

    See `pybfs.calibration.usgs.initial_geometry`.

    Returns
    -------
    pd.DataFrame
        One row with columns Lb, Wb, Kb
    """
    lb, wb, kb = _usgs.initial_geometry(area, lb, x1, wb, por, beta, rb1, tmp_q)
    return pd.DataFrame({'Lb': [lb], 'Wb': [wb], 'Kb': [kb]})


def _problem(build, streamflow_df, error_basis, basin_char, gw_hyd, flow):
    metrics = FlowMetrics.from_legacy(flow)
    record = Record.prepare(np.array(streamflow_df['Streamflow']), metrics)
    return build(record, ModelParameters.from_legacy(basin_char, gw_hyd), metrics, error_basis=error_basis)


def cal_initial(logx, streamflow_df, timestep, error_basis, basin_char, gw_hyd, flow, qmean):
    """Objective for log10 [Lb, Wb, ALPHA, Ks, Kb, Kz]; 100 for out-of-bounds values.

    See `pybfs.calibration.usgs.initial_problem`. ``qmean`` is unused, as before.
    """
    _require_daily(timestep)
    return _problem(_usgs.initial_problem, streamflow_df, error_basis, basin_char, gw_hyd, flow)(logx)


def cal_base(x, streamflow_df, timestep, error_basis, basin_char, gw_hyd, flow):
    """Objective for [Lb, Wb, Kb, Kz]; 100 for out-of-bounds values. See `pybfs.calibration.usgs.base_problem`."""
    _require_daily(timestep)
    return _problem(_usgs.base_problem, streamflow_df, error_basis, basin_char, gw_hyd, flow)(x)


def cal_surface(x, streamflow_df, timestep, error_basis, basin_char, gw_hyd, flow):
    """Objective for log10 [Wb, ALPHA, Ks, Kz]; 100 for out-of-bounds values.

    See `pybfs.calibration.usgs.surface_problem`. (pybfs 0.2.0 was called with
    [Lb, Wb, ALPHA, Ks] and always returned 100; suspected issue C1, fixed.)
    """
    _require_daily(timestep)
    return _problem(_usgs.surface_problem, streamflow_df, error_basis, basin_char, gw_hyd, flow)(x)


def cal_basetable(x, b, params, tmp_q):
    """Base-table recession objective for [Lb, X1, Wb, Kb] at exponent ``b``.

    ``params`` is [AREA, POR, Qmean, Qthresh, RbI, RbS]. See
    `pybfs.calibration.usgs.BaseTableProblem`.
    """
    area, por, qmean, qthresh, rbi, rbs = params
    return _usgs.BaseTableProblem.for_record(b, area, por, qmean, qthresh, (rbi, rbs), tmp_q)(x)


def bfs_calibrate(tmp_site, tmp_area, tmp_q, dys):
    """Calibrate a site

    Runs the three-step USGS calibration procedure (`pybfs.calibration.usgs.calibrate`).

    Parameters
    ----------
    tmp_site : str
        Site identification string
    tmp_area : float
        Drainage area (m²)
    tmp_q : array-like
        Streamflow time series (m³/day)
    dys : array-like
        Date vector (can be strings or datetime objects)

    Returns
    -------
    tuple
        (bf_params, bff, ci_table, bfs_out), or four Nones if the flow metrics
        cannot be computed:

        - bf_params: DataFrame with the calibrated parameters (rounded to 6 decimals)
        - bff: DataFrame with Qmean and baseflow, surface-flow, direct-runoff fractions
        - ci_table: DataFrame with credible intervals
        - bfs_out: DataFrame with the full BFS output of the final run
    """
    result = _usgs.calibrate(tmp_site, tmp_area, tmp_q, dys)
    if result is None:
        return (None, None, None, None)
    return legacy_outputs(result)


def legacy_outputs(result):
    """Convert a `CalibrationResult` to the legacy (bf_params, bff, ci_table, bfs_out) tuple."""
    bfs_out = legacy_bfs_frame(result.simulation, result.dates, result.flow, 'base')
    p = result.params
    values = [p.basin_length, p.base_profile_scale, p.base_width, p.drainable_porosity, p.surface_gradient,
              p.base_profile_exponent, p.surface_conductivity, p.base_conductivity, p.vertical_conductivity]
    values += result.flow.to_legacy() + [result.error, result.baseflow_fraction]
    bf_params = pd.DataFrame({'tmp.site': [result.site_id], 'tmp.area': [p.drainage_area],
                              **{name: [np.round(v, 6)] for name, v in zip(_usgs.HISTORY_COLUMNS[2:], values)}})
    ci_table = bf_ci(bfs_out)[0]
    fractions = flow_fractions(bfs_out)
    bff = pd.DataFrame({'tmp.site': [result.site_id], 'Qmean': [fractions['Qmean']], 'BFF': [fractions['BFF']],
                        'SFF': [fractions['SFF']], 'DRF': [fractions['DRF']], 'Error': [result.error]})
    return (bf_params, bff, ci_table, bfs_out)
