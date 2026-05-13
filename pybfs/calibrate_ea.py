# -*- coding: utf-8 -*-
"""Evolutionary algorithm calibration for PyBFS using NSGA-II.

Two-objective NSGA-II calibration:
  F[0]: Log-space RMSE of Qsim vs Qob on recession days (RecessCount.T > 0,
        Qob > 0). Directly measures the stated model intent: total simulated
        flow should match observed flow during recession and low-flow periods.
  F[1]: MAE between modeled baseflow log-recession rates and the empirical
        quantile regression from flow_metrics. Penalises wrong decay shape
        independently of level.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.decomposition.asf import ASF
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from .bfs import bfs
from .calibrate import calculate_error
from .utilities import base_table, flow_metrics

_POR = 0.15  # Default porosity — used by DE calibration; NSGA-II calibrates POR as a free parameter
N_WORKERS = 10


def recession_log_rmse(bfs_out_df):
    """Log-space RMSE of Qsim vs Qob on recession days.

    Only days where RecessCount.T > 0 and Qob > 0 are included, which
    corresponds to all non-rising timesteps with positive observed flow.
    Log-space is used so that errors at low flows count equally to errors
    at moderate flows across the orders-of-magnitude range of recession.

    Parameters
    ----------
    bfs_out_df : pd.DataFrame
        Output from bfs().

    Returns
    -------
    float
        RMSE of log10(Qsim) - log10(Qob) on recession days, or 999.0
        when fewer than 10 valid days are found.
    """
    qob = bfs_out_df['Qob'].values
    qsim = bfs_out_df['Qsim'].values
    recess = bfs_out_df['RecessCount.T'].values

    # Exclude days where Qob is zero or NaN, or Qsim is NaN (model failure).
    # Do NOT exclude days where Qsim <= 0 — near-zero Qsim is a degenerate
    # solution and must contribute a large error, not be silently dropped.
    mask = (recess > 0) & (qob > 0) & ~np.isnan(qob) & ~np.isnan(qsim)
    if mask.sum() < 10:
        return 999.0
    # Floor Qsim at 1e-6 * Qob so log10 is always defined. A zero or negative
    # Qsim on a recession day gets a log-error of roughly -6, keeping the
    # landscape smooth and allowing the optimizer to find the gradient.
    qsim_safe = np.maximum(qsim[mask], 1e-6 * qob[mask])
    log_errors = np.log10(qsim_safe) - np.log10(qob[mask])
    return float(np.sqrt(np.mean(log_errors ** 2)))


def recession_error(bfs_out_df, rb1, rb2, qthresh):
    """MAE between modeled and empirical daily log-recession rates.

    During recession periods the modeled baseflow should decline at a rate
    consistent with the empirical quantile-regression of log10(Q[t+1]/Q[t])
    on log10(Q[t]) (rb10 from flow_metrics).

    Parameters
    ----------
    bfs_out_df : pd.DataFrame
        Output from bfs().
    rb1 : float
        Intercept of empirical recession regression (rb10[0] from flow_metrics).
    rb2 : float
        Slope of empirical recession regression (rb10[1] from flow_metrics).
    qthresh : float
        Lower flow threshold below which recession comparison is skipped.

    Returns
    -------
    float
        Mean absolute error between modeled and expected log-recession rates,
        or 999.0 when fewer than 10 valid recession days are found.
    """
    qob = bfs_out_df['Qob'].values
    baseflow = bfs_out_df['Baseflow'].values
    recess = bfs_out_df['RecessCount.T'].values

    errors = []
    for t in range(1, len(qob)):
        if (
            recess[t] > 0
            and qob[t] > qthresh
            and baseflow[t] > 0
            and baseflow[t - 1] > 0
            and not np.isnan(qob[t])
            and baseflow[t] <= qob[t]
            and baseflow[t - 1] <= qob[t - 1]
        ):
            modeled_rate = np.log10(baseflow[t] / baseflow[t - 1])
            expected_rate = rb1 + rb2 * np.log10(qob[t])
            errors.append(abs(modeled_rate - expected_rate))

    if len(errors) < 10:
        return 999.0
    return float(np.mean(errors))


def _decode_x(x):
    """Decode optimization vector to physical parameters.

    Encoding: [log10(Lb), log10(Wb), log10(X1), log10(ALPHA), BETA,
               log10(Ks), log10(Kb), log10(Kz), log10(POR)]

    POR (index 8) is optional; defaults to _POR when absent so that
    8-element vectors from DE calibration remain compatible.
    """
    por = 10 ** x[8] if len(x) > 8 else _POR
    return (
        10 ** x[0],  # Lb
        10 ** x[1],  # Wb
        10 ** x[2],  # X1
        10 ** x[3],  # ALPHA
        x[4],        # BETA (linear)
        10 ** x[5],  # Ks
        10 ** x[6],  # Kb
        10 ** x[7],  # Kz
        por,         # POR
    )


def _compute_bff(bfs_out_df):
    """Baseflow fraction: sum(min(Baseflow, Qob)) / sum(Qob)."""
    tmp_bf = bfs_out_df['Baseflow'].copy()
    over = bfs_out_df['Baseflow'] > bfs_out_df['Qob']
    over.fillna(False, inplace=True)
    tmp_bf[over] = bfs_out_df['Qob'][over]
    valid = ~np.isnan(bfs_out_df['Qob'])
    denom = np.nansum(bfs_out_df['Qob'][valid])
    if denom == 0:
        return 0.0
    return float(np.nansum(tmp_bf[valid]) / denom)


def _run_bfs(x, streamflow_df, flow, basin_area):
    """Run base_table + bfs for one parameter vector.

    Returns
    -------
    tuple or None
        (bfs_out_df, basin_char, gw_hyd) on success, None on any failure.
    """
    lb, wb, x1, alpha, beta, ks, kb, kz, por = _decode_x(x)
    basin_char = [basin_area, lb, x1, wb, por]
    gw_hyd = [alpha, beta, ks, kb, kz]
    try:
        sbt = base_table(lb, x1, wb, beta, kb, streamflow_df, por)
        bfs_out = bfs(
            streamflow_df, sbt, basin_char, gw_hyd, flow,
            timestep='day', error_basis='base',
        )
        return bfs_out, basin_char, gw_hyd
    except Exception:
        return None


class BFSProblem(ElementwiseProblem):
    """pymoo ElementwiseProblem for NSGA-II calibration.

    Objectives
    ----------
    F[0] : Log-space RMSE of Qsim vs Qob on recession days (RecessCount.T > 0,
           Qob > 0). Measures how well total simulated flow matches observed
           flow during the periods the model is designed to capture.
    F[1] : Recession MAE between modeled baseflow log-recession rates and the
           empirical recession regression derived from flow_metrics. Measures
           the shape of the decay independently of level.

    Constraints
    -----------
    G[0] : Lb * Wb <= basin_area  (physical feasibility)
    G[1] : baseflow > Qob on <= 5% of valid days  (baseflow physically bounded)
    """

    def __init__(self, streamflow_df, flow, basin_area, rb1, rb2, **kwargs):
        self.streamflow_df = streamflow_df
        self.flow = flow
        self.basin_area = basin_area
        self.rb1 = rb1
        self.rb2 = rb2

        log_area = np.log10(basin_area)
        # Geometric params in log10-space; BETA is linear.
        xl = np.array([
            log_area / 2 - 2,  # log10(Lb)
            log_area / 2 - 4,  # log10(Wb)
            0.0,               # log10(X1) — X1 >= 1 m
            -4.0,              # log10(ALPHA) — min 0.01% gradient
            0.5,               # BETA
            -8.0,              # log10(Ks)
            -8.0,              # log10(Kb)
            -8.0,              # log10(Kz)
            -2.0,              # log10(POR) — min 0.01
        ])
        xu = np.array([
            log_area / 2 + 1,  # log10(Lb) — max 10*sqrt(area); prevents degenerate elongated geometry
            log_area / 2,      # log10(Wb) — max sqrt(area)
            log_area / 2 + 2,  # log10(X1)
            -1.0,              # log10(ALPHA) — max 10% gradient
            20.0,              # BETA
            5.0,               # log10(Ks)
            3.0,               # log10(Kb) — max 1000; prevents POR/Kb compensation that keeps k too high
            3.0,               # log10(Kz) — max 1000; was 5.0, large Kz caused artificially fast drainage
            -0.3,              # log10(POR) — max ~0.5
        ])

        super().__init__(n_var=9, n_obj=2, n_ieq_constr=2, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        lb, wb = 10 ** x[0], 10 ** x[1]

        result = _run_bfs(x, self.streamflow_df, self.flow, self.basin_area)
        if result is None:
            out['F'] = [999.0, 999.0]
            out['G'] = [lb * wb - self.basin_area, 999.0]
            return

        bfs_out, _, _ = result
        f0 = recession_log_rmse(bfs_out)
        f1 = recession_error(bfs_out, self.rb1, self.rb2, self.flow[0])
        valid = ~np.isnan(bfs_out['Qob']) & ~np.isnan(bfs_out['Baseflow'])
        bf_exceed_frac = float(np.mean(bfs_out.loc[valid, 'Baseflow'] > bfs_out.loc[valid, 'Qob']))
        out['F'] = [f0, f1]
        out['G'] = [lb * wb - self.basin_area, bf_exceed_frac - 0.05]


def _build_pareto_df(res_X, res_F, tmp_site, tmp_area, flow_vals):
    """Build Pareto front DataFrame from NSGA-II result arrays."""
    Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise = flow_vals[:6]
    rows = []
    for x, f in zip(res_X, res_F):
        lb, wb, x1, alpha, beta, ks, kb, kz, por = _decode_x(x)
        rows.append({
            'tmp.site': tmp_site,
            'tmp.area': tmp_area,
            'Lb': lb, 'Wb': wb, 'X1': x1,
            'ALPHA': alpha, 'BETA': beta,
            'Ks': ks, 'Kb': kb, 'Kz': kz,
            'POR': por,
            'Qthresh': Qthresh, 'Rs': Rs, 'Rb1': Rb1, 'Rb2': Rb2,
            'Prec': Prec, 'Frac4Rise': Frac4Rise,
            'RecessionRMSE': f[0],
            'RecessionError': f[1],
        })
    return pd.DataFrame(rows)


def bfs_calibrate_nsga2(
    tmp_site,
    tmp_area,
    tmp_q,
    dys,
    pop_size=200,
    n_gen=300,
    seed=None,
    n_jobs=N_WORKERS,
):
    """Calibrate baseflow separation parameters using NSGA-II.

    Performs multi-objective optimization over all 9 physical parameters
    (including POR as a free variable) simultaneously, returning the full
    Pareto front and the knee-point solution.

    Parameters
    ----------
    tmp_site : str
        Site identifier.
    tmp_area : float
        Drainage area (m²).
    tmp_q : array-like
        Streamflow time series (m³/day).
    dys : array-like
        Corresponding date vector.
    pop_size : int, optional
        NSGA-II population size. Default 200.
    n_gen : int, optional
        Number of generations. Default 300.
    seed : int or None, optional
        Random seed for reproducibility.
    n_jobs : int or None, optional
        Worker processes. None uses cpu_count - 1.

    Returns
    -------
    tuple
        (pareto_front_df, knee_params_df, bfs_out_df) where:

        - pareto_front_df: DataFrame with all Pareto-optimal solutions and
          their objectives (RecessionRMSE, RecessionError).
        - knee_params_df: DataFrame with the knee-point parameters (balanced
          trade-off between the two objectives, selected via ASF).
        - bfs_out_df: Full BFS output for the knee-point parameter set.

        Returns (None, None, None) if calibration fails.
    """
    tmp_q = np.asarray(tmp_q, dtype=float)

    flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    if flow_result is None or len(flow_result) < 6:
        print(f"  {tmp_site}: flow_metrics failed")
        return None, None, None

    invalid = [i for i, v in enumerate(flow_result[:6]) if np.isnan(v) or np.isinf(v)]
    if invalid:
        names = ['Qthresh', 'Rs', 'Rb1', 'Rb2', 'Prec', 'Fr4Rise']
        print(f"  {tmp_site}: invalid flow_metrics values at {[names[i] for i in invalid]}")
        return None, None, None

    Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise = flow_result[:6]
    rb10 = (
        flow_result[6]
        if len(flow_result) > 6 and not np.any(np.isnan(flow_result[6]))
        else np.array([-0.01, -0.001])
    )
    RbI, RbS = rb10[0], rb10[1]
    flow = [Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise]

    streamflow_df = pd.DataFrame({
        'Date': pd.to_datetime(dys),
        'Streamflow': tmp_q,
    })

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    # Use spawn context on all platforms to avoid fork+numba conflicts on Linux
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(n_jobs) if n_jobs > 1 else None
    try:
        if pool is not None:
            runner = StarmapParallelization(pool.starmap)
            problem = BFSProblem(
                streamflow_df, flow, tmp_area, RbI, RbS,
                elementwise_runner=runner,
            )
        else:
            problem = BFSProblem(streamflow_df, flow, tmp_area, RbI, RbS)

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        res = pymoo_minimize(
            problem,
            algorithm,
            get_termination('n_gen', n_gen),
            seed=seed,
            verbose=True,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if res.X is None or len(res.X) == 0:
        print(f"  {tmp_site}: NSGA-II returned no solutions")
        return None, None, None

    # Drop solutions with penalty objective values (bfs evaluation failures)
    feasible = np.all(res.F < 900.0, axis=1)
    if not np.any(feasible):
        print(f"  {tmp_site}: no feasible solutions in Pareto front")
        return None, None, None

    X_front = res.X[feasible]
    F_front = res.F[feasible]

    pareto_df = _build_pareto_df(X_front, F_front, tmp_site, tmp_area, flow_result)

    # Knee point: Achievement Scalarization Function on normalized objectives
    F_range = F_front.max(axis=0) - F_front.min(axis=0)
    F_norm = (F_front - F_front.min(axis=0)) / (F_range + 1e-10)
    knee_idx = ASF().do(F_norm, np.ones(2)).argmin()
    knee_x = X_front[knee_idx]

    result = _run_bfs(knee_x, streamflow_df, flow, tmp_area)
    if result is None:
        print(f"  {tmp_site}: final BFS run failed for knee-point solution")
        return pareto_df, None, None

    bfs_out, _, _ = result
    error = calculate_error(bfs_out)
    bff = _compute_bff(bfs_out)

    lb, wb, x1, alpha, beta, ks, kb, kz, por = _decode_x(knee_x)
    knee_params = pd.DataFrame({
        'tmp.site': [tmp_site],
        'tmp.area': [tmp_area],
        'Lb': [lb], 'X1': [x1], 'Wb': [wb], 'POR': [por],
        'ALPHA': [alpha], 'BETA': [beta],
        'Ks': [ks], 'Kb': [kb], 'Kz': [kz],
        'Qthresh': [Qthresh], 'Rs': [Rs], 'Rb1': [Rb1], 'Rb2': [Rb2],
        'Prec': [Prec], 'Frac4Rise': [Frac4Rise],
        'RecessionRMSE': [F_front[knee_idx, 0]],
        'RecessionError': [F_front[knee_idx, 1]],
        'BFF': [bff],
        'Error': [error],
    })

    return pareto_df, knee_params, bfs_out
