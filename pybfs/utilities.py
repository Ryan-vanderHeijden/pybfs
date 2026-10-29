# -*- coding: utf-8 -*-
"""Utility functions: the pybfs 0.2.0 ``utilities`` interface.

Compatibility layer over `pybfs.core`. Names, signatures and results are
unchanged; new code should use `pybfs.core.geometry`,
`pybfs.core.base_table`, and `pybfs.data`.
"""

import numpy as np

from .core import geometry as _geometry
from .core.base_table import build_base_table, positive_flow_range
from .frames import legacy_base_table_frame
from .legacy.utilities import bf_ci, flow_metrics, forecast, get_values_for_site  # noqa: F401  (moved in later steps)

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
