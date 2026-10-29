# -*- coding: utf-8 -*-
"""PyBFS - Python Baseflow Separation

A Python implementation of Baseflow Separation algorithms for hydrological analysis.

Two interfaces are available:

- the array-based API: `simulate`, `ModelParameters`, `FlowMetrics`,
  `read_parameter_table`, and the `pybfs.core`, `pybfs.objectives` and
  `pybfs.calibration` packages;
- the pybfs 0.2.0 functions (`bfs`, `base_table`, `bfs_calibrate`, ...), kept
  with the same signatures and results as wrappers around the new code.

The original implementation is the git tag ``baseline-original``.
"""

__version__ = "0.2.0"
__author__ = "BYU Hydroinformatics"
__email__ = ""

# New API
from .params import ModelParameters, FlowMetrics, PARAMETERS
from .data import read_parameter_table, site_parameters
from .core.model import simulate
from .core.results import ModelState, SimulationResult

# Import from utilities module
from .utilities import (
    sur_z,
    sur_store,
    sur_q,
    dir_q,
    infiltration,
    recharge,
    get_values_for_site,
    base_table,
    forecast,
    flow_metrics,
    bf_ci,
)

# Import from bfs module
from .bfs import bfs

# Import from plot module
from .plot import (
    plot_baseflow_simulation,
    plot_forecast,
    plot_forecast_baseflow,
    plot_forecast_baseflow_streamflow,
)

# Import from skill module
from .skill import (
    modified_strict_baseflow,
    separation_skill,
    forecast_skill,
)

# Import from calibrate module
from .calibrate import (
    bfs_calibrate,
    objective,
    ini_params,
    cal_initial,
    cal_basetable,
    cal_base,
    cal_surface,
)

__all__ = [
    "ModelParameters",
    "FlowMetrics",
    "PARAMETERS",
    "read_parameter_table",
    "site_parameters",
    "simulate",
    "ModelState",
    "SimulationResult",
    "sur_z",
    "sur_store",
    "sur_q",
    "dir_q",
    "infiltration",
    "recharge",
    "get_values_for_site",
    "base_table",
    "flow_metrics",
    "bf_ci",
    "bfs",
    "forecast",
    "bfs_calibrate",
    "objective",
    "ini_params",
    "cal_initial",
    "cal_basetable",
    "cal_base",
    "cal_surface",
    "modified_strict_baseflow",
    "separation_skill",
    "forecast_skill",
    "plot_baseflow_simulation",
    "plot_forecast",
    "plot_forecast_baseflow",
    "plot_forecast_baseflow_streamflow",
]
