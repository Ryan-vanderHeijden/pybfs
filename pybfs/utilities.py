# -*- coding: utf-8 -*-
"""Utility functions: the pybfs 0.2.0 ``utilities`` interface.

Compatibility layer over `pybfs.core`. Names, signatures and results are
unchanged; new code should use `pybfs.core.geometry`,
`pybfs.core.base_table`, and `pybfs.data`.
"""

import numpy as np
import pandas as pd

from .core import geometry as _geometry
from .core.base_table import build_base_table, positive_flow_range
from .core.forecasting import forecast as _forecast
from .core.results import ModelState
from .frames import base_table_from_frame, legacy_base_table_frame, legacy_forecast_frame
from .data import get_values_for_site  # noqa: F401
from .streamflow_metrics import flow_metrics  # noqa: F401
from .params import ModelParameters

# The plain functions run the kernels' Python source (``py_func``), so they keep
# Python's float arithmetic exactly as in pybfs 0.2.0; the *_jit names are the
# compiled kernels. See `pybfs.core.geometry` for definitions and manual references.


def sur_z(lb, a, ws, por, ss):
    """Saturated thickness of the surface reservoir from storage [m]; see `surface_thickness_from_storage`."""
    return _geometry.surface_thickness_from_storage.py_func(lb, a, ws, por, ss)


def sur_store(lb, a, ws, por, zs):
    """Surface storage from saturated thickness [m³]; see `surface_storage_from_thickness`."""
    return _geometry.surface_storage_from_thickness.py_func(lb, a, ws, por, zs)


def sur_q(lb, a, ks, z):
    """Surface-reservoir discharge [m³/day]; see `surface_discharge`."""
    return _geometry.surface_discharge.py_func(lb, a, ks, z)


def dir_q(lb, a, z, i):
    """Direct runoff from saturated land [m³/day]; see `saturated_area_runoff`."""
    return _geometry.saturated_area_runoff.py_func(lb, a, z, i)


def infiltration(lb, ws, ks, a, zs, i):
    """Infiltration through unsaturated land [m³/day]; see `infiltration_rate`."""
    return _geometry.infiltration_rate.py_func(lb, ws, ks, a, zs, i)


def recharge(lb, xb, ws, kz, zs, por):
    """Recharge from surface to base reservoir [m³/day]; see `recharge_rate`."""
    return _geometry.recharge_rate.py_func(lb, xb, ws, kz, zs, por)


sur_z_jit = _geometry.surface_thickness_from_storage
sur_store_jit = _geometry.surface_storage_from_thickness
sur_q_jit = _geometry.surface_discharge
dir_q_jit = _geometry.saturated_area_runoff
infiltration_jit = _geometry.infiltration_rate
recharge_jit = _geometry.recharge_rate


def base_table(lb, x1, wb, b, kb, q, por):
    """Generates baseflow table relating discharge to storage and water levels

    Parameters
    ----------
    lb, x1, wb : float
        Basin length Lb, base profile scale X1, basin width Wb [m]
    b : float
        Base profile exponent β
    kb : float
        Base hydraulic conductivity [m/day]
    q : pd.DataFrame
        Streamflow with a 'Streamflow' column [m³/day]; sets the discharge range
    por : float
        Drainable porosity

    Returns
    -------
    pd.DataFrame
        Columns Xb [m], Z [m], S [m³], Q [m³/day]; rows with Xb ≥ Lb dropped
        (index keeps the original row positions). See `pybfs.core.base_table`.
    """
    table = build_base_table(lb, x1, wb, b, kb, por, *positive_flow_range(np.array(q['Streamflow'])))
    return legacy_base_table_frame(table)


def forecast(streamflow, SBT, basin_char, gw_hyd, flow, initial):
    """Forecast baseflow using PyBFS with provided initial conditions

    Projects baseflow and reservoir states forward with no precipitation
    (no impulses). See `pybfs.core.forecasting.forecast`.

    Parameters
    ----------
    streamflow : pd.DataFrame
        Columns 'date' (forecast dates) and 'streamflow' (ignored; use NaN)
    SBT : pd.DataFrame
        Baseflow table used for the simulation that produced ``initial``
    basin_char, gw_hyd, flow : list
        Same as ``bfs()``
    initial : sequence
        (X, Zb.L, Zs.L, StBase, StSur, SurfaceFlow, Baseflow, Rech) from the last
        row of a ``bfs()`` result

    Returns
    -------
    pd.DataFrame
        Columns Date, Baseflow [m³/day], StSur, StBase [m³], Zs, Zb [m], Rech [m³/day].
        The first row repeats ``initial`` (suspected issue U8).
    """
    dates = pd.to_datetime(streamflow["date"])
    result = _forecast(len(np.array(streamflow['streamflow'])), ModelParameters.from_legacy(basin_char, gw_hyd),
                       ModelState.from_legacy(initial), base_table_from_frame(SBT))
    return legacy_forecast_frame(result, dates)


def bf_ci(bfs_out):
    """Calculate credible intervals for baseflow separation; see `pybfs.diagnostics.bf_ci`."""
    from .diagnostics import bf_ci as _bf_ci
    return _bf_ci(bfs_out)
