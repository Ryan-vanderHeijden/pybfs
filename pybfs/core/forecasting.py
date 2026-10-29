"""Baseflow forecast: project the reservoirs forward with no impulses.

Starting from a `ModelState` (usually the last step of a simulation), the
reservoirs drain and exchange water as in the model loop, but no impulses or
infiltration occur (no precipitation; manual p. 2). Reproduces pybfs 0.2.0
``forecast()``, including that the first forecast step repeats the initial
state rather than advancing it (suspected issue U8).
"""

from __future__ import annotations


import numpy as np
from numba import njit

from ..params import ModelParameters
from .base_table import BaseTable, count_total
from .geometry import (
    infiltration_rate,
    recharge_rate,
    surface_discharge,
    surface_storage_from_thickness,
    surface_thickness_from_storage,
)
from .results import ForecastResult, ModelState


def forecast(n_steps: int, params: ModelParameters, initial: ModelState, base_table: BaseTable) -> ForecastResult:
    """Forecast ``n_steps`` daily steps from ``initial``.

    Parameters
    ----------
    n_steps : int
        Number of steps to forecast.
    params : ModelParameters
        Parameters used for the simulation that produced ``initial``.
    initial : ModelState
        Starting state, e.g. ``simulate(...).final_state()``.
    base_table : BaseTable
        The base table used for that simulation.
    """
    baseflow = np.full(n_steps, np.nan)
    recharge = np.full(n_steps, np.nan)
    storage = np.full((n_steps, 2), np.nan)
    thickness = np.full((n_steps, 2), np.nan)
    if n_steps:
        xb_monotone, s_monotone, _ = base_table.monotone
        _forecast_loop(
            base_table.intersect_distance, base_table.saturated_thickness, base_table.storage,
            base_table.discharge, xb_monotone, s_monotone, np.max(base_table.storage),
            params.basin_length, params.surface_gradient, params.surface_half_width, params.drainable_porosity,
            params.surface_conductivity, params.vertical_conductivity,
            initial.intersect_distance, initial.base_saturated_thickness, initial.surface_saturated_thickness,
            initial.base_storage, initial.surface_storage, initial.baseflow, initial.recharge,
            baseflow, recharge, storage, thickness, np.full(n_steps, np.nan),
        )
    return ForecastResult(baseflow=baseflow, surface_storage=storage[:, 0], base_storage=storage[:, 1],
                          surface_saturated_thickness=thickness[:, 0], base_saturated_thickness=thickness[:, 1],
                          recharge=recharge)


@njit(cache=True)
def _row(column, index):
    return column[index] if 0 <= index < len(column) else np.nan


@njit(cache=True)
def _forecast_loop(tab_xb, tab_z, tab_s, tab_q, xb_monotone, s_monotone, s_base_max,
                   lb, alpha, ws, por, ks, kz,
                   x_initial, z_base_initial, z_surface_initial, s_base_initial, s_surface_initial,
                   baseflow_initial, recharge_initial,
                   baseflow, recharge, storage, thickness, intersect):
    s_surface_max = surface_storage_from_thickness(lb, alpha, ws, por, ws * alpha)
    impulse = 0.0
    for t in range(len(baseflow)):
        first_step = t == 0
        if first_step:
            x_base_start, s_base_start = x_initial, s_base_initial
            z_surface_start, s_surface_start = z_surface_initial, s_surface_initial
        else:
            x_base_start, s_base_start = intersect[t - 1], storage[t - 1, 1]
            z_surface_start, s_surface_start = thickness[t - 1, 0], storage[t - 1, 0]
        row = count_total(tab_xb, x_base_start, False, xb_monotone)
        q_base_start = tab_q[row - 1] if row > 0 else np.nan
        q_surface_start = surface_discharge(lb, alpha, ks, z_surface_start)
        surface_capacity_left = s_surface_max - s_surface_start
        base_capacity_left = s_base_max - s_base_start
        recharge_start = min(recharge_rate(lb, x_base_start, ws, kz, z_surface_start, por),
                             base_capacity_left + q_base_start)
        infil_start = 0.0

        # End-of-step estimates (no impulse, so infiltration is zero)
        s_surface_end = max(s_surface_start + infil_start - recharge_start - q_surface_start, 0.0)
        z_surface_end = surface_thickness_from_storage(lb, alpha, ws, por, s_surface_end)
        q_surface_end = surface_discharge(lb, alpha, ks, z_surface_end)
        infil_end = min(infiltration_rate(lb, ws, ks, alpha, z_surface_end, impulse), surface_capacity_left)
        recharge_end = min(recharge_rate(lb, x_base_start, ws, kz, z_surface_end, por),
                           base_capacity_left + q_base_start)
        s_base_end = max(s_base_start + recharge_end - q_base_start, 0.0)
        row = max(count_total(tab_s, s_base_end, True, s_monotone), 1) - 1
        q_base_end = _row(tab_q, row)

        if first_step:
            # The first step reports the initial state unchanged (issue U8)
            baseflow[t] = baseflow_initial
            recharge[t] = recharge_initial
            storage[t, 1] = s_base_initial
            storage[t, 0] = s_surface_initial
            thickness[t, 1] = z_base_initial
            thickness[t, 0] = z_surface_initial
        else:
            surface_flow = (q_surface_start + q_surface_end) / 2
            baseflow[t] = (q_base_start + q_base_end) / 2
            infiltration = (infil_start + infil_end) / 2
            recharge[t] = (recharge_start + recharge_end) / 2
            storage[t, 0] = max(storage[t - 1, 0] + infiltration - surface_flow - recharge[t], 0.0)
            storage[t, 0] = min(storage[t, 0], s_surface_max)
            thickness[t, 0] = surface_thickness_from_storage(lb, alpha, ws, por, storage[t, 0])
            storage[t, 1] = max(storage[t - 1, 1] + recharge[t] - baseflow[t], 0.0)
            storage[t, 1] = min(storage[t, 1], s_base_max)
            row = max(count_total(tab_s, storage[t, 1], False, s_monotone), 1) - 1
            thickness[t, 1] = _row(tab_z, row)
        row = max(count_total(tab_s, storage[t, 1], False, s_monotone), 1) - 1
        intersect[t] = _row(tab_xb, row)
