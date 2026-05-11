# -*- coding: utf-8 -*-
"""Evolutionary algorithm calibration for PyBFS using NSGA-II.

Two-objective NSGA-II calibration:
  F[0]: Log-space RMSE of Baseflow vs Qob on recession days (RecessCount.T > 0,
        Qob > 0). On recession days DirectRunoff has decayed to near zero so
        Qob ≈ true baseflow; penalising the baseflow component directly prevents
        the optimizer from substituting DirectRunoff to satisfy the flow objective.
  F[1]: Storage-discharge relationship error — |log10(k_modeled) - log10(k_target)|
        where k_modeled is the median k = Q/S from the storage-baseflow table and
        k_target is the empirical drainage rate derived from the observed recession
        regression at Qthresh. Directly penalises solutions where the base reservoir
        drains too fast or too slow.
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
    """Log-space RMSE of Baseflow vs Qob on recession days.

    Only days where RecessCount.T > 0 and Qob > 0 are included. On recession
    days DirectRunoff has decayed to near zero so Qob ≈ true baseflow, making
    this a direct measure of baseflow separation quality. Using Baseflow rather
    than Qsim prevents the optimizer from routing flow through DirectRunoff to
    satisfy the objective while leaving baseflow nearly flat.

    Parameters
    ----------
    bfs_out_df : pd.DataFrame
        Output from bfs().

    Returns
    -------
    float
        RMSE of log10(Baseflow) - log10(Qob) on recession days, or 999.0
        when fewer than 10 valid days are found.
    """
    qob = bfs_out_df['Qob'].values
    baseflow = bfs_out_df['Baseflow'].values
    recess = bfs_out_df['RecessCount.T'].values

    # Exclude days where Qob is zero or NaN, or Baseflow is NaN (model failure).
    # Do NOT exclude days where Baseflow <= 0 — near-zero baseflow is a degenerate
    # solution and must contribute a large error, not be silently dropped.
    mask = (recess > 0) & (qob > 0) & ~np.isnan(qob) & ~np.isnan(baseflow)
    if mask.sum() < 10:
        return 999.0
    # Floor Baseflow at 1e-6 * Qob so log10 is always defined.
    bf_safe = np.maximum(baseflow[mask], 1e-6 * qob[mask])
    log_errors = np.log10(bf_safe) - np.log10(qob[mask])
    return float(np.sqrt(np.mean(log_errors ** 2)))


def _k_target_from_flow_metrics(rb1, rb2, qthresh):
    """Empirical drainage rate k (per day) derived from the recession regression.

    Uses rb10: log10(Q[t+1]/Q[t]) = rb1 + rb2 * log10(Q[t]).
    For a linear reservoir log10(Q[t+1]/Q[t]) = -k / ln(10), giving
    k = -(rb1 + rb2 * log10(qthresh)) * ln(10).

    Falls back to 0.02 /day if the regression implies a non-physical value.
    """
    log_ratio = rb1 + rb2 * np.log10(qthresh)
    k = -log_ratio * np.log(10)
    if k <= 0 or k > 10:
        return 0.02
    return k


def sdr_error(sbt, k_target):
    """Storage-discharge relationship error.

    Penalises the log-ratio between the modeled median k = Q/S (per day) from
    the storage-baseflow table and the empirical drainage rate k_target derived
    from the observed recession regression. A value of 0 means the model drains
    at exactly the empirically observed rate; each unit is one order of magnitude.

    Parameters
    ----------
    sbt : pd.DataFrame
        Storage-baseflow table from base_table(), with columns 'Q' and 'S'.
    k_target : float
        Empirical drainage rate (per day) from _k_target_from_flow_metrics().

    Returns
    -------
    float
        |log10(k_modeled) - log10(k_target)|, or 999.0 on degenerate input.
    """
    valid = (sbt['Q'] > 0) & (sbt['S'] > 0)
    if valid.sum() < 3:
        return 999.0
    k_values = (sbt.loc[valid, 'Q'] / sbt.loc[valid, 'S']).values
    k_median = np.median(k_values)
    if k_median <= 0:
        return 999.0
    return abs(np.log10(k_median) - np.log10(k_target))


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
        (bfs_out_df, basin_char, gw_hyd, sbt) on success, None on any failure.
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
        return bfs_out, basin_char, gw_hyd, sbt
    except Exception:
        return None


class BFSProblem(ElementwiseProblem):
    """pymoo ElementwiseProblem for NSGA-II calibration.

    Objectives
    ----------
    F[0] : Log-space RMSE of Baseflow vs Qob on recession days (RecessCount.T > 0,
           Qob > 0). On recession days Qob ≈ true baseflow; targeting Baseflow
           directly prevents DirectRunoff from substituting to satisfy this objective.
    F[1] : Storage-discharge relationship error — |log10(k_modeled) - log10(k_target)|
           where k_modeled is the median Q/S from the storage-baseflow table and
           k_target is the empirical drainage rate from the recession regression.

    Constraints
    -----------
    G[0] : Lb * Wb <= basin_area  (physical feasibility)
    G[1] : baseflow > Qob on <= 5% of valid days  (baseflow physically bounded)
    """

    def __init__(self, streamflow_df, flow, basin_area, k_target, **kwargs):
        self.streamflow_df = streamflow_df
        self.flow = flow
        self.basin_area = basin_area
        self.k_target = k_target

        log_area = np.log10(basin_area)
        # Geometric params in log10-space; BETA is linear.
        xl = np.array([
            log_area / 2 - 1,  # log10(Lb) — min 0.1*sqrt(area); prevents degenerate short-basin geometry
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

        bfs_out, _, _, sbt = result
        f0 = recession_log_rmse(bfs_out)
        f1 = sdr_error(sbt, self.k_target)
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
            'SDRError': f[1],
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
          their objectives (RecessionRMSE, SDRError).
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
    k_target = _k_target_from_flow_metrics(RbI, RbS, Qthresh)

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
                streamflow_df, flow, tmp_area, k_target,
                elementwise_runner=runner,
            )
        else:
            problem = BFSProblem(streamflow_df, flow, tmp_area, k_target)

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

    bfs_out, _, _, _ = result
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
        'SDRError': [F_front[knee_idx, 1]],
        'BFF': [bff],
        'Error': [error],
    })

    return pareto_df, knee_params, bfs_out
