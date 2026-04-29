# -*- coding: utf-8 -*-
"""Single-objective Differential Evolution calibration for PyBFS.

Minimises bfs_objective: a recession-depth-weighted one-sided log-space RMSE
of Baseflow vs Qob.  Only days where Qob > 0 and RecessCount.T >= 1 are
scored.  Each qualifying day is weighted by sqrt(RecessCount.T) so that
longer recessions carry more influence than short ones without the tail of
extreme drought events dominating entirely.  Only under-prediction (Baseflow <
Qob) is penalised; over-prediction days contribute zero error so the optimizer
is free to keep Baseflow at or above observed flow during recession without a
double penalty from the constraint.

A separate physical constraint (Lb * Wb <= basin_area) and an excess-fraction
constraint (Baseflow > Qob on <= 5% of valid days) prevent degenerate
solutions that drive Baseflow far above Qob to silence the asymmetric
objective.

See calibrate_ea.py for the multi-objective NSGA-II version.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from .calibrate import calculate_error
from .utilities import flow_metrics
from .calibrate_ea import (
    _POR, N_WORKERS,
    _decode_x, _compute_bff, _run_bfs,
)



def bfs_objective(bfs_out_df):
    """Recession-depth-weighted one-sided log-space RMSE of Baseflow vs Qob.

    Scoring:
    - Only recession days (RecessCount.T >= 1, Qob > 0) are included.
    - Each day's weight = sqrt(RecessCount.T), so deeper recession days carry
      more influence than early-recession days without the tail of extreme
      drought events dominating entirely.
    - Only under-prediction (Baseflow < Qob) is penalised.  Days where
      Baseflow >= Qob contribute zero error.  The bf_exceed_frac constraint
      handles the upper bound so the optimizer is not doubly penalised.
    - A floor of 1e-6 * Qob is applied before taking logs so near-zero
      Baseflow produces a large but finite penalty.

    Parameters
    ----------
    bfs_out_df : pd.DataFrame
        Output from bfs().

    Returns
    -------
    float
        Weighted one-sided log-RMSE on recession days, or 999.0 when
        fewer than 10 qualifying days exist.
    """
    qob = bfs_out_df['Qob'].values
    baseflow = bfs_out_df['Baseflow'].values
    recess = bfs_out_df['RecessCount.T'].values

    mask = (recess >= 1) & (qob > 0) & ~np.isnan(qob) & ~np.isnan(baseflow)
    if mask.sum() < 10:
        return 999.0

    bf_safe = np.maximum(baseflow[mask], 1e-6 * qob[mask])
    log_errors = np.log10(bf_safe) - np.log10(qob[mask])
    under = np.minimum(log_errors, 0.0)

    weights = np.sqrt(recess[mask].astype(float))
    weights /= weights.mean()  # normalise so scale is comparable to unweighted

    return float(np.sqrt(np.average(under ** 2, weights=weights)))


class BFSProblemDE(ElementwiseProblem):
    """pymoo ElementwiseProblem for single-objective DE calibration.

    Objective
    ---------
    F[0] : Recession-depth-weighted one-sided log-RMSE of Baseflow vs Qob
           (see bfs_objective).

    Constraints
    -----------
    G[0] : Lb * Wb <= basin_area  (physical feasibility)
    G[1] : Baseflow > Qob on <= 5% of valid days  (prevents degeneracy where
           Baseflow >> Qob silences the asymmetric objective)
    """

    def __init__(self, streamflow_df, flow, basin_area, **kwargs):
        self.streamflow_df = streamflow_df
        self.flow = flow
        self.basin_area = basin_area

        log_area = np.log10(basin_area)
        xl = np.array([
            log_area / 2 - 2,  # log10(Lb)
            log_area / 2 - 4,  # log10(Wb)
            0.0,               # log10(X1)
            -4.0,              # log10(ALPHA)
            0.5,               # BETA
            -8.0,              # log10(Ks)
            -8.0,              # log10(Kb)
            -8.0,              # log10(Kz)
        ])
        xu = np.array([
            log_area / 2 + 2,  # log10(Lb)
            log_area / 2,      # log10(Wb)
            log_area / 2 + 2,  # log10(X1)
            -1.0,              # log10(ALPHA)
            20.0,              # BETA
            5.0,               # log10(Ks)
            5.0,               # log10(Kb)
            5.0,               # log10(Kz)
        ])

        super().__init__(n_var=8, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        lb, wb = 10 ** x[0], 10 ** x[1]

        result = _run_bfs(x, self.streamflow_df, self.flow, self.basin_area)
        if result is None:
            out['F'] = [999.0]
            out['G'] = [lb * wb - self.basin_area, 999.0]
            return

        bfs_out, _, _ = result
        f0 = bfs_objective(bfs_out)
        valid = ~np.isnan(bfs_out['Qob']) & ~np.isnan(bfs_out['Baseflow'])
        bf_exceed_frac = float(np.mean(bfs_out.loc[valid, 'Baseflow'] > bfs_out.loc[valid, 'Qob']))
        out['F'] = [f0]
        out['G'] = [lb * wb - self.basin_area, bf_exceed_frac - 0.05]


def bfs_calibrate_de(
    tmp_site,
    tmp_area,
    tmp_q,
    dys,
    pop_size=100,
    n_gen=500,
    seed=None,
    n_jobs=N_WORKERS,
):
    """Calibrate baseflow separation parameters using Differential Evolution.

    Single-objective optimisation minimising a recession-depth-weighted
    one-sided log-RMSE of Baseflow vs Qob (bfs_objective).  DE is used
    instead of NSGA-II because there is no Pareto front to explore — a
    single best solution is returned directly.

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
        DE population size. Default 100 (~12× n_var).
    n_gen : int, optional
        Number of generations. Default 500.
    seed : int or None, optional
        Random seed for reproducibility.
    n_jobs : int or None, optional
        Worker processes. None uses cpu_count - 1.

    Returns
    -------
    tuple
        (best_params_df, bfs_out_df) where:

        - best_params_df: DataFrame with the best-found parameters,
          objective value (BFSObjective), BFF, and Error.
        - bfs_out_df: Full BFS output for the best parameter set.

        Returns (None, None) if calibration fails.
    """
    tmp_q = np.asarray(tmp_q, dtype=float)

    flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    if flow_result is None or len(flow_result) < 6:
        print(f"  {tmp_site}: flow_metrics failed")
        return None, None

    invalid = [i for i, v in enumerate(flow_result[:6]) if np.isnan(v) or np.isinf(v)]
    if invalid:
        names = ['Qthresh', 'Rs', 'Rb1', 'Rb2', 'Prec', 'Fr4Rise']
        print(f"  {tmp_site}: invalid flow_metrics values at {[names[i] for i in invalid]}")
        return None, None

    Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise = flow_result[:6]
    flow = [Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise]

    streamflow_df = pd.DataFrame({
        'Date': pd.to_datetime(dys),
        'Streamflow': tmp_q,
    })

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    ctx = mp.get_context('spawn')
    pool = ctx.Pool(n_jobs) if n_jobs > 1 else None
    try:
        if pool is not None:
            runner = StarmapParallelization(pool.starmap)
            problem = BFSProblemDE(
                streamflow_df, flow, tmp_area,
                elementwise_runner=runner,
            )
        else:
            problem = BFSProblemDE(streamflow_df, flow, tmp_area)

        algorithm = DE(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            variant='DE/rand/1/bin',
            CR=0.7,
            F=0.5,
            dither='vector',
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

    if res.X is None:
        print(f"  {tmp_site}: DE returned no solution")
        return None, None

    best_x = res.X
    result = _run_bfs(best_x, streamflow_df, flow, tmp_area)
    if result is None:
        print(f"  {tmp_site}: final BFS run failed for best solution")
        return None, None

    bfs_out, _, _ = result
    obj = bfs_objective(bfs_out)
    bff = _compute_bff(bfs_out)
    error = calculate_error(bfs_out)

    lb, wb, x1, alpha, beta, ks, kb, kz = _decode_x(best_x)
    best_params = pd.DataFrame({
        'tmp.site': [tmp_site],
        'tmp.area': [tmp_area],
        'Lb': [lb], 'X1': [x1], 'Wb': [wb], 'POR': [_POR],
        'ALPHA': [alpha], 'BETA': [beta],
        'Ks': [ks], 'Kb': [kb], 'Kz': [kz],
        'Qthresh': [Qthresh], 'Rs': [Rs], 'Rb1': [Rb1], 'Rb2': [Rb2],
        'Prec': [Prec], 'Frac4Rise': [Frac4Rise],
        'BFSObjective': [obj],
        'BFF': [bff],
        'Error': [error],
    })

    return best_params, bfs_out
