"""pandas views of model results, in new and in legacy (pybfs 0.2.0) layouts."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .core.base_table import BaseTable, is_monotone
from .core.results import ForecastResult, SimulationResult
from .objectives.legacy import recession_weights, relative_error
from .params import FlowMetrics

ErrorBasis = Literal["base", "total"]

LEGACY_BFS_COLUMNS = ["Date", "Qob", "Qsim", "SurfaceFlow", "Baseflow", "DirectRunoff", "X", "Eta", "StSur",
                      "StBase", "Impulse.L", "Zs.L", "Zb.L", "Infil", "Rech", "RecessCount.T", "AdjPctEr",
                      "Weight"]


def simulation_frame(result: SimulationResult, dates=None) -> pd.DataFrame:
    """One row per step, columns named after `SimulationResult` fields."""
    columns = {
        "q_obs": result.q_obs, "q_sim": result.q_sim, "surface_flow": result.surface_flow,
        "baseflow": result.baseflow, "direct_runoff": result.direct_runoff, "residual": result.residual,
        "impulse": result.impulse, "surface_storage": result.surface_storage, "base_storage": result.base_storage,
        "surface_saturated_thickness": result.surface_saturated_thickness,
        "base_saturated_thickness": result.base_saturated_thickness,
        "intersect_distance": result.intersect_distance, "infiltration": result.infiltration,
        "recharge": result.recharge, "recession_age": result.recession_age,
    }
    frame = pd.DataFrame(columns)
    if dates is not None:
        frame.insert(0, "date", pd.to_datetime(np.asarray(dates)))
    return frame


def q_for_error(result: SimulationResult, error_basis: ErrorBasis) -> np.ndarray:
    """The simulated series errors are computed on: baseflow, or the component sum."""
    if error_basis == "base":
        return result.flow_components[:, 1]
    if error_basis == "total":
        return np.sum(result.flow_components, axis=1)
    raise ValueError(f"error_basis must be 'base' or 'total', not {error_basis!r}")


def legacy_bfs_frame(result: SimulationResult, dates: pd.Series, flow: FlowMetrics,
                     error_basis: ErrorBasis = "total") -> pd.DataFrame:
    """The 18-column DataFrame returned by pybfs 0.2.0 ``bfs()``.

    ``dates`` is ``pd.to_datetime(streamflow["Date"])``; its index becomes the
    frame's index, as before.
    """
    rel_error = relative_error(result.q_obs, q_for_error(result, error_basis), flow.flow_precision)
    weights = recession_weights(rel_error, result.recession_age, flow.storm_recession_rate)
    frame = pd.DataFrame({
        "Date": dates, "Qob": result.q_obs, "Qsim": result.q_sim, "SurfaceFlow": result.surface_flow,
        "Baseflow": result.baseflow, "DirectRunoff": result.direct_runoff, "X": result.intersect_distance,
        "Eta": result.residual, "StSur": result.surface_storage, "StBase": result.base_storage,
        "Impulse.L": result.impulse, "Zs.L": result.surface_saturated_thickness,
        "Zb.L": result.base_saturated_thickness, "Infil": result.infiltration, "Rech": result.recharge,
        "RecessCount.T": result.recession_age, "AdjPctEr": rel_error, "Weight": weights,
    })
    return frame[LEGACY_BFS_COLUMNS]


def legacy_forecast_frame(result: ForecastResult, dates: pd.Series) -> pd.DataFrame:
    """The DataFrame returned by pybfs 0.2.0 ``forecast()``."""
    frame = pd.DataFrame({
        "Date": dates, "Baseflow": result.baseflow, "StSur": result.surface_storage,
        "StBase": result.base_storage, "Zs": result.surface_saturated_thickness,
        "Zb": result.base_saturated_thickness, "Rech": result.recharge,
    })
    return frame[["Date", "Baseflow", "StSur", "StBase", "Zs", "Zb", "Rech"]]


def legacy_base_table_frame(table: BaseTable) -> pd.DataFrame:
    """The ``Xb, Z, S, Q`` DataFrame returned by pybfs 0.2.0 ``base_table()``."""
    return pd.DataFrame({"Xb": table.intersect_distance, "Z": table.saturated_thickness, "S": table.storage,
                         "Q": table.discharge}, index=table.row_index)


def base_table_from_frame(frame: pd.DataFrame) -> BaseTable:
    """Inverse of `legacy_base_table_frame`, for callers passing a legacy SBT."""
    columns = [np.array(frame[c].values, dtype=float) for c in ("Xb", "Z", "S", "Q")]
    return BaseTable(*columns, row_index=frame.index.to_numpy(),
                     monotone=(is_monotone(columns[0]), is_monotone(columns[2]), is_monotone(columns[3])))
