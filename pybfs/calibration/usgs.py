"""The USGS BFS calibration procedure (pybfs 0.2.0 ``bfs_calibrate``).

Three steps, each a sequence of Nelder-Mead searches (manual p. 15-16; the
code follows the revised R procedure in ``bfs/RecalibrationDescription.txt``):

1. ``step1``: calibrate Lb, Wb, α, Ks, Kb, Kz with β = 1 (``cal_initial``), then
   the surface parameters (``cal_surface``).
2. ``step2``: for β = 0.6, 0.7, ... fit the base table's recession rates to the
   empirical recession curve (``cal_basetable``); for the best β, calibrate
   Lb, Wb, Kb, Kz (``cal_base``) and the surface parameters again.
3. ``step3``: take the parameter set with the largest baseflow fraction and
   recalibrate the six parameters of step 1.

This module reproduces the legacy procedure exactly, including behavior
logged in ``notes/suspected_issues.md``; the relevant issue is cited where
it occurs (C1 cal_surface never runs the model, C2 no reltol, C3 the β sweep
stops after β > 10, C5-C8 ``ini_params``/X1 details, C10 α not re-read,
C11 max-BFF selection, C13 rounding to 6 decimals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..core.base_table import build_base_table
from ..core.results import SimulationResult
from ..diagnostics import baseflow_fraction
from ..frames import q_for_error
from ..objectives.legacy import (
    recession_rate_mismatch,
    recession_weights,
    relative_error,
    usgs_weighted_error,
    weighted_mean_abs_error,
)
from ..params import FlowMetrics, ModelParameters
from ..streamflow_metrics import flow_metrics
from .optimizers import nelder_mead_parscale
from .problem import CalibrationProblem, Record, run_simulation

log = logging.getLogger(__name__)

#: Search vectors of the sub-problems (legacy ``cal_*`` argument order).
INITIAL_NAMES = ("basin_length", "base_width", "surface_gradient", "surface_conductivity",
                 "base_conductivity", "vertical_conductivity")
SURFACE_NAMES = ("basin_length", "base_width", "surface_gradient", "surface_conductivity")
BASE_NAMES = ("basin_length", "base_width", "base_conductivity", "vertical_conductivity")

#: Columns of the parameter-set history (``bf_params``).
HISTORY_COLUMNS = ["tmp.site", "tmp.area", "Lb", "X1", "Wb", "POR", "ALPHA", "BETA", "Ks", "Kb", "Kz",
                   "Qthresh", "Rs", "Rb1", "Rb2", "Prec", "Frac4Rise", "Error", "BFF"]

INITIAL_POROSITY = 0.15
INITIAL_SURFACE_GRADIENT = 0.01
INITIAL_BETA = 1  # an int, as in the legacy code
BETA_STEP = 0.1


# ---------------------------------------------------------------------------
# Objectives of the sub-problems
# ---------------------------------------------------------------------------

def usgs_objective(result: SimulationResult, flow: FlowMetrics, error_basis: str = "base") -> float:
    """The procedure's calibration objective for one simulation (legacy ``objective``)."""
    rel_error = relative_error(result.q_obs, q_for_error(result, error_basis), flow.flow_precision)
    weights = recession_weights(rel_error, result.recession_age, flow.storm_recession_rate)
    return usgs_weighted_error(result.q_obs, result.baseflow, rel_error, weights,
                               flow_precision=flow.flow_precision)


def initial_problem(record: Record, params: ModelParameters, flow: FlowMetrics, *, error_basis: str = "base",
                    on_simulate=None) -> CalibrationProblem:
    """``cal_initial``: log10 of Lb, Wb, α, Ks, Kb, Kz."""
    area = params.drainage_area

    def is_feasible(logx):
        return not any([
            (10 ** logx[0] * 10 ** logx[1]) > area,  # Lb·Wb > AREA
            logx[2] > -1,                            # α > 0.1
            logx[3] < -8, logx[3] > 5,               # Ks outside [1e-8, 1e5]
            logx[4] < -8, logx[4] > 5,               # Kb
            logx[5] < -8, logx[5] > 5,               # Kz
        ])

    return CalibrationProblem(record, flow, params, INITIAL_NAMES, "log10",
                              lambda r, f: usgs_objective(r, f, error_basis), is_feasible, 100.0, on_simulate)


def surface_problem(record: Record, params: ModelParameters, flow: FlowMetrics, *, error_basis: str = "base",
                    on_simulate=None) -> CalibrationProblem:
    """``cal_surface``: log10 of Lb, Wb, α, Ks.

    The feasibility test is written for a vector [Wb, α, Ks, Kz] but receives
    [Lb, Wb, α, Ks], so ``log10(Wb) > -1`` rejects every realistic point and the
    model never runs (suspected issue C1, inherited from the R code).
    """
    area_per_length = params.drainage_area / params.basin_length

    def is_feasible(logx):
        return not any([10 ** logx[1] > area_per_length, logx[1] > -1, logx[2] < -8, logx[2] > 5,
                        logx[3] < -8, logx[3] > 5])

    return CalibrationProblem(record, flow, params, SURFACE_NAMES, "log10",
                              lambda r, f: usgs_objective(r, f, error_basis), is_feasible, 100.0, on_simulate)


def base_problem(record: Record, params: ModelParameters, flow: FlowMetrics, *, error_basis: str = "base",
                 on_simulate=None) -> CalibrationProblem:
    """``cal_base``: Lb, Wb, Kb, Kz (linear). Kb ≥ 1e-8 here; R had ``Kb < 10-8`` (issue C4)."""
    area = params.drainage_area

    def is_feasible(x):
        return not any([x[0] * x[1] > area, x[0] < 0, x[1] < 0, x[2] < 1e-8, x[2] > 1e5,
                        x[3] < 1e-8, x[3] > 1e5])

    return CalibrationProblem(record, flow, params, BASE_NAMES, "linear",
                              lambda r, f: usgs_objective(r, f, error_basis), is_feasible, 100.0, on_simulate)


@dataclass(frozen=True)
class BaseTableProblem:
    """``cal_basetable``: Lb, X1, Wb, Kb (linear) for a fixed β, without running the model.

    Scores how well the base table's linear recession rates Q/S match the
    empirical 10th-percentile recession curve between Qmin/10 and Qmean
    (`recession_rate_mismatch`). Returns 1000 for out-of-bounds parameters or
    a table whose largest discharge is below Qthresh.
    """

    beta: float
    drainage_area: float
    drainable_porosity: float
    baseflow_threshold: float
    recession_curve: tuple[float, float]
    flow_range: tuple[float, float]
    q_levels: np.ndarray

    @classmethod
    def for_record(cls, beta, drainage_area, drainable_porosity, mean_flow, baseflow_threshold, recession_curve,
                   q_obs) -> BaseTableProblem:
        from ..core.base_table import positive_flow_range
        q_obs = np.asarray(q_obs)
        q_min = np.nanmin(q_obs[q_obs > 0]) / 10
        q_levels = 10 ** np.linspace(np.log10(q_min), np.log10(mean_flow), 101)
        return cls(beta, drainage_area, drainable_porosity, baseflow_threshold, recession_curve,
                   positive_flow_range(q_obs), q_levels)

    def __call__(self, x) -> float:
        area = self.drainage_area
        if any([np.any(np.isinf(x)), x[0] <= 0, x[0] > area / x[2], x[1] <= 0, x[2] > area / x[0],
                x[3] < 1e-8, x[3] > 1e5]):
            return 1000.0
        table = build_base_table(x[0], x[1], x[2], self.beta, x[3], self.drainable_porosity, *self.flow_range)
        if np.nanmax(table.discharge) < self.baseflow_threshold:
            return 1000.0
        return recession_rate_mismatch(table.storage, table.discharge, self.q_levels, self.recession_curve)


# ---------------------------------------------------------------------------
# Initial parameter values
# ---------------------------------------------------------------------------

def initial_geometry(area, lb, x1, wb, por, beta, rb1, q_obs):
    """Initial Lb, Wb, Kb matching the recession rate Rb1 (legacy ``ini_params``).

    Lb is scaled until the Kb implied by Rb1 lies in [1e-7, 1e4] m/day at every
    grid point; then Wb is scaled until the intersect where baseflow first
    exceeds the mean flow lies between 0.1 and 0.9 Lb. Inherits issues C5-C7
    (``wb * area > lb`` compares a volume with a length; the returned Kb is
    circular).

    Returns
    -------
    tuple of float
        (Lb, Wb, Kb).
    """
    q_mean = np.nanmean(q_obs)

    iterate = True
    while iterate:
        iterate = False
        xb = np.linspace(0, lb, 1001)
        kb_for_rb1 = -rb1 * (por * (lb - xb / 2) * x1 ** beta) / (beta * xb ** (beta - 1))
        kb_for_rb1[xb == 0] = np.nan
        valid = kb_for_rb1[~np.isnan(kb_for_rb1)]
        if np.any(valid < 1e-7):
            lb = lb * 1.1
            iterate = True
        if np.any(valid > 1e4):
            lb = lb * 0.9
            iterate = True

    iterate = True
    while iterate:
        iterate = False
        xb = np.linspace(0, lb, 1001)
        z = (xb / x1) ** beta
        gradient = beta / (x1 ** beta) * xb ** (beta - 1)
        kb = -rb1 * (por * (lb - xb / 2) * x1 ** beta) / (beta * xb ** (beta - 1))
        kb[xb == 0] = np.nan
        q_grid = wb * z * kb * gradient
        q_grid[np.isnan(kb)] = np.nan

        above_mean = np.where(q_grid > q_mean)[0]
        i_mean = len(q_grid) - 1 if len(above_mean) == 0 else above_mean[0]
        if xb[i_mean] < (0.1 * lb):
            wb = 0.9 * wb  # move the intersect away from the outlet
            if wb > 10:
                iterate = True
        if xb[i_mean] > (0.9 * lb):
            wb = 1.1 * wb  # move it toward the outlet
            if wb * area > lb:
                lb = 0.9 * area / wb
                iterate = True

    kb = q_grid[i_mean] / (wb * z[i_mean] * gradient[i_mean])
    return lb, wb, kb


# ---------------------------------------------------------------------------
# Procedure
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Outcome of `calibrate`.

    Attributes
    ----------
    site_id : str
    params : ModelParameters
        Final parameters (unrounded).
    flow : FlowMetrics
    history : list of dict
        Parameter sets in the order they were recorded (initial, step 1,
        step 2); columns as in `HISTORY_COLUMNS`. Step 3 chooses from these.
    error : float
        Weighted mean absolute error of the final run ("Error").
    baseflow_fraction : float
        BFF of the final run, baseflow capped at observed flow.
    simulation : SimulationResult
        Final run.
    dates : pd.Series
    """

    site_id: str
    params: ModelParameters
    flow: FlowMetrics
    history: list[dict]
    error: float
    baseflow_fraction: float
    simulation: SimulationResult
    dates: pd.Series
    beta_sweep: list[list[float]] = field(default_factory=list)


@dataclass
class _State:
    """Mutable working values of the procedure (the legacy local variables)."""

    area: float
    lb: float
    x1: float
    wb: float
    por: float
    alpha: float
    beta: float
    ks: float
    kb: float
    kz: float

    def params(self) -> ModelParameters:
        return ModelParameters(self.area, self.lb, self.x1, self.wb, self.por, self.alpha, self.beta,
                               self.ks, self.kb, self.kz)


def calibrate(site_id, drainage_area, q_obs, dates, *,
              on_simulate: Callable[[ModelParameters], None] | None = None) -> CalibrationResult | None:
    """Calibrate the BFS model for one site with the USGS procedure.

    Parameters
    ----------
    site_id : str
        Site identifier (only recorded).
    drainage_area : float
        Drainage area [m²].
    q_obs : array-like
        Observed daily streamflow [m³/day].
    dates : array-like
        Dates of ``q_obs``.
    on_simulate : callable, optional
        Called with the parameters of every model run, in order.

    Returns
    -------
    CalibrationResult, or None if the flow metrics cannot be computed.
    """
    q_obs = np.asarray(q_obs)
    metrics = _flow_metrics_or_none(q_obs)
    if metrics is None:
        return None
    flow = FlowMetrics.from_legacy(metrics)
    q_mean = np.nanmean(q_obs[q_obs >= 0])
    rb10 = metrics[6] if len(metrics) > 6 else None
    recession_curve = (rb10[0], rb10[1]) if rb10 is not None and not np.any(np.isnan(rb10)) else (-0.01, -0.001)

    record = Record.prepare(q_obs, flow)
    dates = pd.to_datetime(dates)
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)

    # Initial values: Lb, Wb from the area, then adjusted to the recession rate
    lb = 2 * (drainage_area / 2) ** 0.5
    wb = drainage_area / lb / 10
    ws = wb / 2
    alpha = INITIAL_SURFACE_GRADIENT
    x1 = 1 / alpha
    lb, wb, kb = initial_geometry(drainage_area, lb, x1, wb, INITIAL_POROSITY, INITIAL_BETA,
                                  flow.base_recession_rate_at_mean, q_obs)
    s = _State(drainage_area, lb, x1, wb, INITIAL_POROSITY, alpha, INITIAL_BETA,
               ks=(1 - np.exp(flow.storm_recession_rate)) * INITIAL_POROSITY * (3 / 4 * ws) / alpha,
               kb=kb, kz=10 * q_mean / (lb * wb))

    sim = run_simulation(record, s.params(), flow, on_simulate)
    error = _error(sim, flow)
    bff = baseflow_fraction(sim.q_obs, sim.baseflow, cap_at_observed=False)
    log.info("Initial parameters: %s; Error=%.6f, BFF=%.6f", s.params(), error, bff)
    history = [_history_row(site_id, s, flow, error, bff)]

    error, bff = _step1(s, record, flow, history, site_id, error, bff, on_simulate)
    sweep = _step2(s, record, flow, history, site_id, error, bff, q_mean, recession_curve, q_obs, on_simulate)
    _step3(s, record, flow, history, on_simulate)

    sim = run_simulation(record, s.params(), flow, on_simulate)
    error = _error(sim, flow)
    bff = baseflow_fraction(sim.q_obs, sim.baseflow)
    log.info("Final parameters: %s; Error=%.6f, BFF=%.6f", s.params(), error, bff)
    return CalibrationResult(site_id, s.params(), flow, history, error, bff, sim, dates, sweep)


def _step1(s, record, flow, history, site_id, error, bff, on_simulate):
    """Calibrate with β = 1: six parameters, then the surface parameters (a no-op, C1)."""
    log.info("Step 1: initial calibration (beta=%s)", s.beta)
    result = nelder_mead_parscale(initial_problem(record, s.params(), flow, on_simulate=on_simulate),
                                  *_twice(np.log10([s.lb, s.wb, s.alpha, s.ks, s.kb, s.kz])))
    if np.all(np.isfinite(result.x)):
        s.lb, s.wb, s.alpha, s.ks, s.kb, s.kz = (10 ** v for v in result.x)  # X1 not updated (as before)
        result = nelder_mead_parscale(surface_problem(record, s.params(), flow, on_simulate=on_simulate),
                                      *_twice(np.log10([s.lb, s.wb, s.alpha, s.ks])))
        if np.all(np.isfinite(result.x)):
            s.lb, s.wb, s.alpha, s.ks = (10 ** v for v in result.x)
            sim = run_simulation(record, s.params(), flow, on_simulate)
            error = _error(sim, flow)
            bff = baseflow_fraction(sim.q_obs, sim.baseflow)
            log.info("Step 1 result: %s; Error=%.6f, BFF=%.6f", s.params(), error, bff)
        history.append(_history_row(site_id, s, flow, error, bff))
    return error, bff


def _step2(s, record, flow, history, site_id, error, bff, q_mean, recession_curve, q_obs, on_simulate):
    """β sweep on the base table, then calibrate the base and surface parameters for the best β."""
    log.info("Step 2: fitting the base-table recession for beta = 0.6, 0.7, ...")
    sweep = []  # rows [Lb, X1, Wb, beta, Kb, objective]
    n_valid = 0
    beta = 0.5
    last_objective = None
    while True:
        beta = beta + BETA_STEP
        # X1 such that Qb = Qmean at Xb = Lb/2 (derivation in Rfunction.bfs_calibrate.R)
        if (2 * beta - 1) != 0:
            s.x1 = (s.wb * s.kb * beta ** 2 / ((2 * beta - 1) * q_mean) * (s.lb / 2) ** (2 * beta - 1)) ** (1 / (2 * beta))
        else:
            s.x1 = np.nan
        if not np.isfinite(s.x1) or s.x1 < 1.0:
            s.x1 = max(1.0 / s.alpha, 1.0)  # Python-only fallback (issue C8)

        if np.isfinite(s.x1) and s.x1 > 0:
            problem = BaseTableProblem.for_record(beta, s.area, s.por, q_mean, flow.baseflow_threshold,
                                                  recession_curve, q_obs)
            result = nelder_mead_parscale(problem, *_twice(np.array([s.lb, s.x1, s.wb, s.kb])))
            if np.all(np.isfinite(result.x)):
                n_valid += 1
                sweep.append([result.x[0], result.x[1], result.x[2], beta, result.x[3], result.fun])
                last_objective = result.fun
                if n_valid % 5 == 0:
                    log.info("  beta=%.1f, objective=%.4f", beta, result.fun)

        # Stops at the first beta > 10: this compares the last value with itself (issue C3)
        if beta > 10 and sweep and last_objective is not None and n_valid > 0:
            if abs((last_objective - sweep[n_valid - 1][5]) / last_objective) < 0.001:
                break
        if beta == 20:  # never true in floating point (issue C3)
            break

    if not sweep:
        return sweep
    table = np.array(sweep)
    best = table[np.argmin(table[:, 5]), :5]
    s.lb, s.x1, s.wb, s.beta, s.kb = best
    log.info("  best beta=%.3f", s.beta)

    result = nelder_mead_parscale(base_problem(record, s.params(), flow, on_simulate=on_simulate),
                                  *_twice(np.array([s.lb, s.wb, s.kb, s.kz])))
    result.x[result.x < 0] = np.nan
    if np.all(np.isfinite(result.x)):
        # The surface problem is built from the parameters *before* this cal_base result (issue C9)
        stale = s.params()
        s.lb, s.wb, s.kb, s.kz = result.x
        result = nelder_mead_parscale(surface_problem(record, stale, flow, on_simulate=on_simulate),
                                      *_twice(np.log10([s.lb, s.wb, s.alpha, s.ks])))
        if np.all(np.isfinite(result.x)):
            s.lb, s.wb, s.alpha, s.ks = (10 ** v for v in result.x)
            run_params = s.params()
        else:
            run_params = stale
        sim = run_simulation(record, run_params, flow, on_simulate)
        error = _error(sim, flow)
        bff = baseflow_fraction(sim.q_obs, sim.baseflow)
        log.info("Step 2 result: %s; Error=%.6f, BFF=%.6f", s.params(), error, bff)
        history.append(_history_row(site_id, s, flow, error, bff))
    return sweep


def _step3(s, record, flow, history, on_simulate):
    """Recalibrate the six step-1 parameters from the recorded set with the largest BFF (C11)."""
    if len(history) <= 1:
        return
    log.info("Step 3: final calibration")
    g = _index_of_max([row["BFF"] for row in history[1:]]) + 1
    row = {k: np.float64(v) for k, v in history[g].items() if k != "tmp.site"}
    log.info("  selected row %d: BETA=%.3f, BFF=%.6f", g, row["BETA"], row["BFF"])
    # alpha is kept at its current value, not read from the row (issue C10)
    s.lb, s.x1, s.wb = row["Lb"], row["X1"], row["Wb"]
    s.beta, s.kb, s.ks, s.kz = row["BETA"], row["Kb"], row["Ks"], row["Kz"]
    result = nelder_mead_parscale(initial_problem(record, s.params(), flow, on_simulate=on_simulate),
                                  *_twice(np.log10([s.lb, s.wb, s.alpha, s.ks, s.kb, s.kz])))
    if np.all(np.isfinite(result.x)):
        s.lb, s.wb, s.alpha, s.ks, s.kb, s.kz = (10 ** v for v in result.x)  # X1 kept from the row


def _twice(x):
    """(x0, scale): the procedure starts each search at x0 with parscale = x0."""
    return x, x


def _error(sim: SimulationResult, flow: FlowMetrics) -> float:
    rel_error = relative_error(sim.q_obs, sim.baseflow, flow.flow_precision)
    return weighted_mean_abs_error(rel_error, recession_weights(rel_error, sim.recession_age,
                                                                flow.storm_recession_rate))


def _history_row(site_id, s: _State, flow: FlowMetrics, error, bff) -> dict:
    values = [site_id, s.area, s.lb, s.x1, s.wb, s.por, s.alpha, s.beta, s.ks, s.kb, s.kz] + flow.to_legacy()
    return dict(zip(HISTORY_COLUMNS, values + [error, bff]))


def _index_of_max(values) -> int:
    """Position of the first maximum, ignoring NaN (pandas ``idxmax`` semantics)."""
    values = np.array(values, dtype=float)
    if np.all(np.isnan(values)):
        raise ValueError("Encountered all NA values")
    return int(np.nanargmax(values))


def _flow_metrics_or_none(q_obs):
    try:
        metrics = flow_metrics(q_obs, timestep='day', fr4rise=0.05)
    except Exception:
        log.exception("flow_metrics failed")
        return None
    if metrics is None or len(metrics) < 6:
        log.warning("flow_metrics returned None or insufficient values")
        return None
    invalid = [i for i, v in enumerate(metrics[:6]) if np.isnan(v) or np.isinf(v)]
    if invalid:
        log.warning("flow_metrics returned invalid values at indices %s: %s", invalid,
                    [metrics[i] for i in invalid])
        return None
    return metrics
