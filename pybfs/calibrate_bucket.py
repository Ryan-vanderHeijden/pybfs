# -*- coding: utf-8 -*-
"""Calibration for the simplified two-bucket model (bfs_bucket).

Two entry points, both for the geometry-free model {Ss_max, ks, kr, k, n}:

bfs_calibrate_bucket_ga  (preferred)
------------------------------------
Single-objective GA over 4 free parameters. ks is not calibrated: it is fixed
per site at |Rs|, the storm-flow recession coefficient already reported by
flow_metrics (see surface_ks_from_recession). Free vars are
{Ss_max, kr, Qref, n}.

  F[0]: Log-space RMSE of Qsim vs Qob on recession days, weighted by the
        model's own Weight column so sustained recessions -- where the
        baseflow signal lives -- dominate (recession_weighted_log_rmse).
  G[0]: Timescale separation, base half-life >= 3x surface half-life.
  G[1]: Modeled baseflow <= Qob on at least 95% of days.

Fixing ks and enforcing G[0] are both load-bearing. The objective landscape is
bimodal: a slow "surface" bucket (half-life ~20 d) can masquerade as
groundwater, and an unconstrained search falls into that degenerate basin even
though it scores worse. Because the impulse W is inferred so that Qsim == Qob
on every rising limb, all calibration signal lives in recession days, and total
Qsim is indifferent to which bucket supplied the water.

bfs_calibrate_bucket_nsga2  (original, retained for comparison)
--------------------------------------------------------------
Two-objective NSGA-II over all 5 parameters, mirroring calibrate_ea.py so
results are directly comparable to the geometry model. Kept because it
reproduces the pre-fix behaviour; it exhibits the degeneracy described above.

  F[0]: recession_log_rmse.  F[1]: recession_error.  G[0]: baseflow bound.

Base-bucket reparameterization (both paths)
-------------------------------------------
Q = k * S**n is calibrated via a reference discharge Qref at a fixed anchor
storage Sref (= 50 * mean flow), with k = Qref / Sref**n derived internally.
This keeps the decision variable Qref physically bounded (a fraction of mean
flow) and well-scaled for every n, avoiding the ~10-order swing in k that a
free exponent would otherwise induce.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.decomposition.asf import ASF
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from .bfs_bucket import bfs_bucket
from .calibrate import calculate_error
from .calibrate_ea import recession_log_rmse, recession_error, _compute_bff
from .utilities import flow_metrics

N_WORKERS = 8

# Fixed anchor storage for the base-bucket reparameterization, as a multiple of
# mean flow (~50 mean-flow-days of storage). Only anchors the Qref scale; the
# fit is insensitive to its exact value because Qref absorbs it.
_SREF_MULT = 50.0


def _decode_x(x, sref):
    """Decode optimization vector to bucket parameters.

    Encoding: [log10(Ss_max), log10(ks), log10(kr), log10(Qref), n]

    Returns
    -------
    tuple
        (params_dict, qref, n) where params_dict = {Ss_max, ks, kr, k, n}
        with k = Qref / Sref**n.
    """
    ss_max = 10 ** x[0]
    ks = 10 ** x[1]
    kr = 10 ** x[2]
    qref = 10 ** x[3]
    n = x[4]
    k = qref / sref ** n
    params = {'Ss_max': ss_max, 'ks': ks, 'kr': kr, 'k': k, 'n': n}
    return params, qref, n


def _run_bucket(x, streamflow_df, flow, basin_area, sref):
    """Run bfs_bucket for one parameter vector.

    Returns
    -------
    tuple or None
        (bucket_out_df, params) on success, None on any failure.
    """
    params, _, _ = _decode_x(x, sref)
    try:
        out = bfs_bucket(
            streamflow_df, params, flow, basin_area=basin_area,
            timestep='day', error_basis='base',
        )
        return out, params
    except Exception:
        return None


class BucketProblem(ElementwiseProblem):
    """pymoo ElementwiseProblem for two-bucket NSGA-II calibration.

    Objectives mirror BFSProblem (recession_log_rmse, recession_error).

    Constraints
    -----------
    G[0] : baseflow > Qob on <= 5% of valid days  (baseflow physically bounded)
    """

    def __init__(self, streamflow_df, flow, basin_area, rb1, rb2, sref,
                 qmean, **kwargs):
        self.streamflow_df = streamflow_df
        self.flow = flow
        self.basin_area = basin_area
        self.rb1 = rb1
        self.rb2 = rb2
        self.sref = sref

        log_qmean = np.log10(qmean)
        xl = np.array([
            log_qmean - 1.0,   # log10(Ss_max) — min ~0.1 mean-flow-day capacity
            -2.3,              # log10(ks) — min 0.005 /day
            -4.0,              # log10(kr) — min 1e-4 /day
            log_qmean - 3.0,   # log10(Qref) — baseflow at Sref >= qmean/1000
            0.5,               # n
        ])
        xu = np.array([
            log_qmean + 2.7,   # log10(Ss_max) — max ~500 mean-flow-days
            0.7,               # log10(ks) — max 5 /day
            0.0,               # log10(kr) — max 1 /day
            log_qmean,         # log10(Qref) — baseflow at Sref <= mean flow
            3.0,               # n
        ])
        super().__init__(n_var=5, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        result = _run_bucket(x, self.streamflow_df, self.flow,
                             self.basin_area, self.sref)
        if result is None:
            out['F'] = [999.0, 999.0]
            out['G'] = [999.0]
            return

        bucket_out, _ = result
        f0 = recession_log_rmse(bucket_out)
        f1 = recession_error(bucket_out, self.rb1, self.rb2, self.flow[0])
        valid = ~np.isnan(bucket_out['Qob']) & ~np.isnan(bucket_out['Baseflow'])
        bf_exceed_frac = float(np.mean(
            bucket_out.loc[valid, 'Baseflow'] > bucket_out.loc[valid, 'Qob']
        ))
        out['F'] = [f0, f1]
        out['G'] = [bf_exceed_frac - 0.05]


def _build_pareto_df(res_X, res_F, tmp_site, tmp_area, flow_vals, sref):
    """Build Pareto front DataFrame from NSGA-II result arrays."""
    Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise = flow_vals[:6]
    rows = []
    for x, f in zip(res_X, res_F):
        params, qref, n = _decode_x(x, sref)
        rows.append({
            'tmp.site': tmp_site,
            'tmp.area': tmp_area,
            'Ss_max': params['Ss_max'], 'ks': params['ks'], 'kr': params['kr'],
            'k': params['k'], 'n': n, 'Qref': qref, 'Sref': sref,
            'Qthresh': Qthresh, 'Rs': Rs, 'Rb1': Rb1, 'Rb2': Rb2,
            'Prec': Prec, 'Frac4Rise': Frac4Rise,
            'RecessionRMSE': f[0],
            'RecessionError': f[1],
        })
    return pd.DataFrame(rows)


def bfs_calibrate_bucket_nsga2(
    tmp_site,
    tmp_area,
    tmp_q,
    dys,
    pop_size=200,
    n_gen=300,
    seed=None,
    n_jobs=N_WORKERS,
):
    """Calibrate the two-bucket model parameters using NSGA-II.

    Parameters mirror bfs_calibrate_nsga2 (calibrate_ea). Optimizes the
    5 bucket parameters {Ss_max, ks, kr, Qref, n} and returns the Pareto front
    plus the ASF knee-point solution.

    Returns
    -------
    tuple
        (pareto_front_df, knee_params_df, bucket_out_df), or (None, None, None)
        on failure.
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

    qmean = np.nanmean(tmp_q[tmp_q > 0])
    sref = _SREF_MULT * qmean

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
            problem = BucketProblem(
                streamflow_df, flow, tmp_area, RbI, RbS, sref, qmean,
                elementwise_runner=runner,
            )
        else:
            problem = BucketProblem(
                streamflow_df, flow, tmp_area, RbI, RbS, sref, qmean,
            )

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

    feasible = np.all(res.F < 900.0, axis=1)
    if not np.any(feasible):
        print(f"  {tmp_site}: no feasible solutions in Pareto front")
        return None, None, None

    X_front = res.X[feasible]
    F_front = res.F[feasible]

    pareto_df = _build_pareto_df(X_front, F_front, tmp_site, tmp_area, flow_result, sref)

    # Knee point: Achievement Scalarization Function on normalized objectives
    F_range = F_front.max(axis=0) - F_front.min(axis=0)
    F_norm = (F_front - F_front.min(axis=0)) / (F_range + 1e-10)
    knee_idx = ASF().do(F_norm, np.ones(2)).argmin()
    knee_x = X_front[knee_idx]

    result = _run_bucket(knee_x, streamflow_df, flow, tmp_area, sref)
    if result is None:
        print(f"  {tmp_site}: final run failed for knee-point solution")
        return pareto_df, None, None

    bucket_out, params = result
    error = calculate_error(bucket_out)
    bff = _compute_bff(bucket_out)

    _, qref, n = _decode_x(knee_x, sref)
    knee_params = pd.DataFrame({
        'tmp.site': [tmp_site],
        'tmp.area': [tmp_area],
        'Ss_max': [params['Ss_max']], 'ks': [params['ks']], 'kr': [params['kr']],
        'k': [params['k']], 'n': [n], 'Qref': [qref], 'Sref': [sref],
        'Qthresh': [Qthresh], 'Rs': [Rs], 'Rb1': [Rb1], 'Rb2': [Rb2],
        'Prec': [Prec], 'Frac4Rise': [Frac4Rise],
        'RecessionRMSE': [F_front[knee_idx, 0]],
        'RecessionError': [F_front[knee_idx, 1]],
        'BFF': [bff],
        'Error': [error],
    })

    return pareto_df, knee_params, bucket_out


# ---------------------------------------------------------------------------
# Single-objective calibration (recession-weighted log-RMSE + timescale
# separation). This is the reevaluated formulation: because the impulse is
# inferred so Qsim == Qob on every rising limb, the entire calibration signal
# lives in the recession days. A single recession-weighted objective therefore
# captures the model intent, and a timescale-separation constraint makes the
# baseflow attribution identifiable (surface bucket must be drained before the
# remainder is called baseflow).
# ---------------------------------------------------------------------------

_LN2 = np.log(0.5) * -1.0  # ln(2)
_SEP_FACTOR = 3.0          # base recession must be >= this x slower than surface


def surface_ks_from_recession(rs):
    """Surface discharge coefficient fixed objectively from the hydrograph.

    Rs (flow_metrics index 1) is the exponential coefficient for rapid /
    storm-flow recession [1/day, negative]. The surface bucket's Qs = ks*Ss
    recedes exponentially at rate ks, so ks is exactly |Rs| -- equivalently
    ks = ln2 / t_half_surf with t_half_surf = ln2/|Rs|.

    Fixing ks rather than calibrating it removes the degenerate solution branch
    in which a slow "surface" bucket masquerades as groundwater (a strong local
    optimum at t_half ~19.7 d that the objective only weakly disfavours).

    Raises
    ------
    ValueError
        If Rs is not a finite negative number.
    """
    if rs is None or not np.isfinite(rs) or rs >= 0:
        raise ValueError(f"Rs must be a finite negative recession rate, got {rs!r}")
    return float(abs(rs))


def recession_weighted_log_rmse(bfs_out_df):
    """Recession-weighted log-space RMSE of Qsim vs Qob over the full record.

    Uses the model's own Weight column (1 - exp(rs * recess_day), forced to 1
    on over-prediction), which up-weights sustained recessions -- exactly the
    days that carry baseflow information. Qsim is floored at 1e-6*Qob so a
    degenerate near-zero Qsim earns a large error rather than being dropped.

    Returns 999.0 when fewer than 20 weighted days are available.
    """
    qob = bfs_out_df['Qob'].values
    qsim = bfs_out_df['Qsim'].values
    w = bfs_out_df['Weight'].values

    mask = (qob > 0) & ~np.isnan(qob) & ~np.isnan(qsim) & ~np.isnan(w)
    if mask.sum() < 20:
        return 999.0
    ww = w[mask]
    sw = ww.sum()
    if sw <= 0:
        return 999.0
    qsim_safe = np.maximum(qsim[mask], 1e-6 * qob[mask])
    err2 = (np.log10(qsim_safe) - np.log10(qob[mask])) ** 2
    return float(np.sqrt(np.sum(ww * err2) / sw))


def _decode_x_fixed_ks(x, sref, ks):
    """Decode the 4-var vector for the fixed-ks model.

    Encoding: [log10(Ss_max), log10(kr), log10(Qref), n]; ks is supplied.

    Returns (params_dict, qref, n) with k = Qref / Sref**n.
    """
    ss_max = 10 ** x[0]
    kr = 10 ** x[1]
    qref = 10 ** x[2]
    n = x[3]
    k = qref / sref ** n
    params = {'Ss_max': ss_max, 'ks': ks, 'kr': kr, 'k': k, 'n': n}
    return params, qref, n


def _run_bucket_fixed_ks(x, streamflow_df, flow, basin_area, sref, ks):
    """Run bfs_bucket for one 4-var parameter vector with ks fixed."""
    params, _, _ = _decode_x_fixed_ks(x, sref, ks)
    try:
        out = bfs_bucket(
            streamflow_df, params, flow, basin_area=basin_area,
            timestep='day', error_basis='base',
        )
        return out, params
    except Exception:
        return None


def _base_thalf(qref, n, sref):
    """Base-bucket flow recession half-life at the reference storage (days).

    For Q = k*S**n the flow e-folding rate at S is k*n*S**(n-1); at S = Sref
    (with k = Qref/Sref**n) this simplifies to n*Qref/Sref, independent of the
    absolute k scale. Half-life = ln(2) * Sref / (n * Qref).
    """
    return _LN2 * sref / (n * qref)


class BucketProblemSO(ElementwiseProblem):
    """Single-objective bucket calibration with ks fixed from the hydrograph.

    Four free parameters {Ss_max, kr, Qref, n}; ks is set objectively to |Rs|
    (see surface_ks_from_recession), so the degenerate slow-surface branch is
    structurally unreachable rather than merely constrained.

    Objective
    ---------
    recession_weighted_log_rmse(Qsim, Qob).

    Constraints
    -----------
    G[0] : base half-life >= _SEP_FACTOR * surface half-life (timescale separation)
    G[1] : baseflow > Qob on <= 5% of valid days (baseflow physically bounded)
    """

    def __init__(self, streamflow_df, flow, basin_area, sref, qmean, ks, **kwargs):
        self.streamflow_df = streamflow_df
        self.flow = flow
        self.basin_area = basin_area
        self.sref = sref
        self.ks = ks

        log_qmean = np.log10(qmean)
        xl = np.array([
            log_qmean - 1.0,   # log10(Ss_max)
            -4.0,              # log10(kr)
            log_qmean - 3.0,   # log10(Qref)
            0.5,               # n
        ])
        xu = np.array([
            log_qmean + 2.7,   # log10(Ss_max)
            0.0,               # log10(kr) — max 1 /day
            log_qmean,         # log10(Qref)
            3.0,               # n
        ])
        super().__init__(n_var=4, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        result = _run_bucket_fixed_ks(x, self.streamflow_df, self.flow,
                                      self.basin_area, self.sref, self.ks)
        if result is None:
            out['F'] = [999.0]
            out['G'] = [999.0, 999.0]
            return

        bucket_out, params = result
        _, qref, n = _decode_x_fixed_ks(x, self.sref, self.ks)

        t_half_surf = _LN2 / params['ks']
        t_half_base = _base_thalf(qref, n, self.sref)
        g_sep = _SEP_FACTOR * t_half_surf - t_half_base

        valid = ~np.isnan(bucket_out['Qob']) & ~np.isnan(bucket_out['Baseflow'])
        bf_exceed_frac = float(np.mean(
            bucket_out.loc[valid, 'Baseflow'] > bucket_out.loc[valid, 'Qob']
        ))

        out['F'] = [recession_weighted_log_rmse(bucket_out)]
        out['G'] = [g_sep, bf_exceed_frac - 0.05]


def bfs_calibrate_bucket_ga(
    tmp_site,
    tmp_area,
    tmp_q,
    dys,
    pop_size=100,
    n_gen=150,
    seed=None,
    n_jobs=N_WORKERS,
):
    """Single-objective GA calibration of the two-bucket model, ks fixed.

    ks is set objectively from the storm-flow recession coefficient
    (ks = |Rs|, see surface_ks_from_recession) rather than calibrated, leaving
    four free parameters {Ss_max, kr, Qref, n}.

    Objective: recession-weighted log-RMSE (recession_weighted_log_rmse).
    Constraints: timescale separation + baseflow bound (see BucketProblemSO).

    Returns
    -------
    tuple
        (best_params_df, bucket_out_df), or (None, None) on failure.
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

    qmean = np.nanmean(tmp_q[tmp_q > 0])
    sref = _SREF_MULT * qmean

    try:
        ks = surface_ks_from_recession(Rs)
    except ValueError as exc:
        print(f"  {tmp_site}: cannot fix ks from recession — {exc}")
        return None, None
    print(f"  {tmp_site}: ks fixed at {ks:.4f} /day "
          f"(surface half-life {_LN2 / ks:.2f} d) from Rs")

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
            problem = BucketProblemSO(
                streamflow_df, flow, tmp_area, sref, qmean, ks,
                elementwise_runner=runner,
            )
        else:
            problem = BucketProblemSO(streamflow_df, flow, tmp_area, sref, qmean, ks)

        algorithm = GA(pop_size=pop_size, eliminate_duplicates=True)
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
        print(f"  {tmp_site}: GA returned no solution")
        return None, None

    best_x = res.X
    result = _run_bucket_fixed_ks(best_x, streamflow_df, flow, tmp_area, sref, ks)
    if result is None:
        print(f"  {tmp_site}: final run failed for best solution")
        return None, None

    bucket_out, params = result
    error = calculate_error(bucket_out)
    bff = _compute_bff(bucket_out)
    _, qref, n = _decode_x_fixed_ks(best_x, sref, ks)
    t_half_surf = _LN2 / params['ks']
    t_half_base = _base_thalf(qref, n, sref)

    best_params = pd.DataFrame({
        'tmp.site': [tmp_site],
        'tmp.area': [tmp_area],
        'Ss_max': [params['Ss_max']], 'ks': [params['ks']], 'kr': [params['kr']],
        'k': [params['k']], 'n': [n], 'Qref': [qref], 'Sref': [sref],
        'tHalfSurf': [t_half_surf], 'tHalfBase': [t_half_base],
        'Qthresh': [Qthresh], 'Rs': [Rs], 'Rb1': [Rb1], 'Rb2': [Rb2],
        'Prec': [Prec], 'Frac4Rise': [Frac4Rise],
        'RecessWeightedLogRMSE': [float(res.F[0])],
        'BFF': [bff],
        'Error': [error],
    })
    return best_params, bucket_out
