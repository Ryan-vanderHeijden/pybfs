"""The BFS forward model: simulate streamflow components from an observed record.

BFS is a state-space model driven by the streamflow record itself (manual
p. 8): each step it carries the surface and base storages forward, and on
rising limbs it infers the precipitation impulse needed to reproduce the
observed flow. No precipitation input is used.

Order of calculations per step (manual p. 11-12):

1. Start-of-step fluxes from the previous step's state.
2. Impulse search (rising steps and the step after): multiply a trial
   impulse by each factor in `IMPULSE_SEARCH_FACTORS` while the residual is
   above ``max(Prec, Q/100)``.
3. End-of-step estimates from the start-of-step fluxes.
4. Fluxes are the average of start and end estimates (2nd-order Runge-Kutta).
5. Storages updated with the averaged fluxes; direct runoff from the impulse.

This module reproduces pybfs 0.2.0 (``pybfs.legacy.bfs``, numba path) bit for
bit, including behavior flagged in ``notes/suspected_issues.md`` (M1, M4,
M10, M12).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..params import FlowMetrics, ModelParameters
from .base_table import BaseTable, build_base_table, count_leading, max_storage, positive_flow_range
from .geometry import (
    infiltration_rate,
    recharge_rate,
    saturated_area_runoff,
    surface_discharge,
    surface_storage_from_thickness,
    surface_thickness_from_storage,
)
from .hydrograph import HydrographSignals, clean_flow, hydrograph_signals
from .results import SimulationResult

#: Step factors for the impulse search, applied in turn (coarse, then fine).
IMPULSE_SEARCH_FACTORS = np.array([2.0, 1.1])


def simulate(q_obs, params: ModelParameters, flow: FlowMetrics, *,
             base_table: BaseTable | None = None,
             signals: HydrographSignals | None = None) -> SimulationResult:
    """Run the BFS model over a daily record.

    Parameters
    ----------
    q_obs : array-like
        Observed daily streamflow [m³/day]; NaN marks missing days and negative
        values are treated as missing.
    params : ModelParameters
        Geometry and hydraulic parameters.
    flow : FlowMetrics
        Flow metrics; the model uses ``flow_precision`` and ``rise_fraction``.
    base_table : BaseTable, optional
        Base-reservoir table. Built from ``params`` and the positive range of
        ``q_obs`` if omitted. Pass it explicitly to reuse one table across runs
        or to use a table built from a different (e.g. longer) record.
    signals : HydrographSignals, optional
        Precomputed rise/recession flags for ``q_obs`` (they depend only on the
        record and ``flow``).

    Returns
    -------
    SimulationResult

    Warns
    -----
    UserWarning
        For negative parameters, ``Lb·Wb > AREA``, or a base table containing
        NaN. The run continues anyway, as in pybfs 0.2.0 (suspected issue M1).
    """
    q = clean_flow(q_obs)
    if np.all(np.isnan(q)):
        raise ValueError("q_obs has no valid (non-negative, non-missing) values")
    if base_table is None:
        base_table = build_base_table(params.basin_length, params.base_profile_scale, params.base_width,
                                      params.base_profile_exponent, params.base_conductivity,
                                      params.drainable_porosity, *positive_flow_range(q_obs))
    if signals is None:
        signals = hydrograph_signals(q, flow.flow_precision, flow.rise_fraction)
    _warn_on_bad_parameters(params, flow, base_table)
    return run_model(q, params, flow, base_table, signals)


def run_model(q: NDArray[np.float64], params: ModelParameters, flow: FlowMetrics,
              base_table: BaseTable, signals: HydrographSignals) -> SimulationResult:
    """Run the step loop on prepared inputs (no validation; for repeated calls).

    ``q`` must come from `clean_flow`, and ``signals`` from `hydrograph_signals`
    on the same ``q``.
    """
    n = len(q)
    flow_components = np.full((n, 3), np.nan)
    residual = np.full(n, np.nan)
    impulse = np.full(n, np.nan)
    thickness = np.full((n, 2), np.nan)
    storage = np.full((n, 2), np.nan)
    exchange = np.full((n, 2), np.nan)
    intersect = np.full(n, np.nan)

    xb_monotone, s_monotone, q_monotone = base_table.monotone
    _step_loop(
        q, np.nanmean(q), signals.is_rise,
        base_table.intersect_distance, base_table.saturated_thickness, base_table.storage,
        base_table.discharge, xb_monotone, s_monotone, q_monotone,
        params.basin_length, params.surface_gradient, params.surface_half_width, params.drainable_porosity,
        params.surface_conductivity, params.vertical_conductivity, flow.flow_precision,
        IMPULSE_SEARCH_FACTORS,
        intersect, flow_components, residual, impulse, thickness, storage, exchange,
    )
    return SimulationResult(
        q_obs=q, flow_components=flow_components, residual=residual, impulse=impulse,
        surface_storage=storage[:, 0], base_storage=storage[:, 1],
        surface_saturated_thickness=thickness[:, 0], base_saturated_thickness=thickness[:, 1],
        intersect_distance=intersect, infiltration=exchange[:, 0], recharge=exchange[:, 1],
        recession_age=signals.recession_age,
    )


def _warn_on_bad_parameters(params: ModelParameters, flow: FlowMetrics, base_table: BaseTable) -> None:
    # Same checks and messages as pybfs 0.2.0, which listed Ks twice and never
    # checked Kz (suspected issue M2); kept identical for now.
    p = params
    checked = [p.basin_length, p.base_profile_scale, p.base_width, p.surface_gradient, p.base_profile_exponent,
               p.surface_conductivity, p.base_conductivity, p.surface_conductivity, p.drainable_porosity,
               flow.baseflow_threshold, -flow.storm_recession_rate, -flow.base_recession_rate_at_mean,
               -flow.base_recession_rate_low, flow.flow_precision, flow.rise_fraction]
    if np.any(np.array(checked) < 0):
        warnings.warn("Negative parameter(s)", stacklevel=3)
    if p.basin_length * p.base_width > p.drainage_area:
        warnings.warn("lb x wb > area", stacklevel=3)
    if base_table.has_nan():
        warnings.warn("Cannot calculate discharge for base parameters", stacklevel=3)


@njit(cache=True)
def _row(column, index):
    return column[index] if 0 <= index < len(column) else np.nan


@njit(cache=True)
def _step_loop(q, q_mean, is_rise,
               tab_xb, tab_z, tab_s, tab_q, xb_monotone, s_monotone, q_monotone,
               lb, alpha, ws, por, ks, kz, prec, search_factors,
               intersect, flow_components, residual, impulse, thickness, storage, exchange):
    """Fill the output arrays in place. Arrays are indexed [step] or [step, surface|base]."""
    n = len(q)
    t = 0
    while t < n and math.isnan(q[t]):
        t += 1
    if t >= n:
        return

    s_base_max = max_storage(tab_s)
    s_surface_max = surface_storage_from_thickness(lb, alpha, ws, por, ws * alpha)

    # --- Initial state (manual p. 12; the code starts baseflow at min(Q, Qmean), issue M12)
    first_step = True
    q_base_start = min(q[t], q_mean)
    row = count_leading(tab_q, q_base_start, False, q_monotone)
    row = row - 1 if row > 0 else -1
    x_base_start = _row(tab_xb, row)
    s_base_start = _row(tab_s, row)
    base_capacity_left = s_base_max - s_base_start

    q_surface_start = q[t] - q_base_start
    q_surface_start = 0.0 if math.isnan(q_surface_start) else max(0.0, q_surface_start)
    z_surface_start = q_surface_start / (2 * lb * ks * alpha)
    z_surface_start = ws * alpha if math.isnan(z_surface_start) else min(z_surface_start, ws * alpha)
    s_surface_start = surface_storage_from_thickness(lb, alpha, ws, por, z_surface_start)
    surface_capacity_left = s_surface_max - s_surface_start
    infil_start = 0.0
    recharge_start = recharge_rate(lb, x_base_start, ws, kz, z_surface_start, por)

    while t < n:
        if not first_step:
            # --- 1. Start-of-step state and fluxes from the previous step
            x_base_start = intersect[t - 1]
            s_base_start = storage[t - 1, 1]
            row = count_leading(tab_xb, x_base_start, False, xb_monotone)
            row = row - 1 if row > 0 else -1
            q_base_start = _row(tab_q, row)

            z_surface_start = thickness[t - 1, 0]
            s_surface_start = storage[t - 1, 0]
            q_surface_start = surface_discharge(lb, alpha, ks, z_surface_start)

            surface_capacity_left = s_surface_max - s_surface_start
            base_capacity_left = s_base_max - s_base_start
            recharge_start = min(recharge_rate(lb, x_base_start, ws, kz, z_surface_start, por),
                                 base_capacity_left + q_base_start)
            infil_start = 0.0

            # --- 2. Impulse needed to reproduce the observed flow
            impulse[t] = 0.0
            residual_estimate = max(0.0, q[t] - q_base_start - q_surface_start)
            if t > 0 and residual_estimate > 0:
                if is_rise[t] or (t > 0 and is_rise[t - 1]):
                    impulse[t] = residual_estimate / (2 * lb * ws)
                    z_trial = z_surface_start
                    q_trial = q_surface_start
                    for factor in search_factors:
                        trial_impulse = impulse[t]
                        trial_residual = residual_estimate
                        while trial_residual > max(prec, q[t] / 100) and trial_impulse > 0:
                            impulse[t] = trial_impulse
                            residual_estimate = trial_residual
                            trial_impulse = factor * trial_impulse
                            infil_trial = min(infiltration_rate(lb, ws, ks, alpha, (z_surface_start + z_trial) / 2,
                                                                trial_impulse), surface_capacity_left)
                            s_trial = max(s_surface_start + infil_trial - recharge_start - q_trial, 0.0)
                            z_trial = surface_thickness_from_storage(lb, alpha, ws, por, s_trial)
                            q_trial = surface_discharge(lb, alpha, ks, z_trial)
                            runoff_trial = (saturated_area_runoff(lb, alpha, z_surface_start, trial_impulse)
                                            + saturated_area_runoff(lb, alpha, (z_trial - z_surface_start),
                                                                    trial_impulse / 2)
                                            + max(2 * lb * (ws - z_surface_start / alpha) * (impulse[t] - ks), 0.0))
                            trial_residual = q[t] - q_trial - runoff_trial - q_base_start

            infil_start = min(infiltration_rate(lb, ws, ks, alpha, z_surface_start, impulse[t]),
                              surface_capacity_left)

        # --- 3. End-of-step estimates
        s_surface_end = max(s_surface_start + infil_start - recharge_start - q_surface_start, 0.0)
        z_surface_end = surface_thickness_from_storage(lb, alpha, ws, por, s_surface_end)
        q_surface_end = surface_discharge(lb, alpha, ks, z_surface_end)
        infil_end = infiltration_rate(lb, ws, ks, alpha, z_surface_end, impulse[t])
        # NaN on the first step (impulse undefined): report the free capacity (issue M4)
        infil_end = surface_capacity_left if math.isnan(infil_end) else min(infil_end, surface_capacity_left)
        recharge_end = min(recharge_rate(lb, x_base_start, ws, kz, z_surface_end, por),
                           base_capacity_left + q_base_start)
        s_base_end = max(s_base_start + recharge_end - q_base_start, 0.0)
        row = max(count_leading(tab_s, s_base_end, True, s_monotone), 1) - 1
        z_base_end = _row(tab_z, row)
        q_base_end = _row(tab_q, row)

        # --- 4. Step-average fluxes
        flow_components[t, 0] = (q_surface_start + q_surface_end) / 2
        flow_components[t, 1] = (q_base_start + q_base_end) / 2
        exchange[t, 0] = (infil_start + infil_end) / 2
        exchange[t, 1] = (recharge_start + recharge_end) / 2

        # --- 5. Storage update and direct runoff
        if first_step:
            storage[t, 0] = s_surface_end
            storage[t, 1] = s_base_end
            thickness[t, 0] = z_surface_end
            thickness[t, 1] = z_base_end
        else:
            storage[t, 0] = max(storage[t - 1, 0] + exchange[t, 0] - flow_components[t, 0] - exchange[t, 1], 0.0)
            storage[t, 0] = min(storage[t, 0], s_surface_max)
            thickness[t, 0] = surface_thickness_from_storage(lb, alpha, ws, por, storage[t, 0])
            storage[t, 1] = max(storage[t - 1, 1] + exchange[t, 1] - flow_components[t, 1], 0.0)
            storage[t, 1] = min(storage[t, 1], s_base_max)
            row = max(count_leading(tab_s, storage[t, 1], False, s_monotone), 1) - 1
            thickness[t, 1] = _row(tab_z, row)
            # Saturated land at the start of the step, plus the newly saturated strip at half the
            # impulse, plus any impulse beyond the infiltration capacity
            flow_components[t, 2] = (saturated_area_runoff(lb, alpha, z_surface_start, impulse[t])
                                     + saturated_area_runoff(lb, alpha, (thickness[t, 0] - z_surface_start),
                                                             impulse[t] / 2)
                                     + max(2 * lb * (ws - z_surface_start / alpha) * (impulse[t] - ks), 0.0))

        residual[t] = q[t] - (flow_components[t, 0] + flow_components[t, 1] + flow_components[t, 2])
        row = max(count_leading(tab_s, storage[t, 1], False, s_monotone), 1) - 1
        intersect[t] = _row(tab_xb, row)

        t += 1
        first_step = False
