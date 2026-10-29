"""Input helpers: streamflow tables and parameter tables.

This is the only module of the new API that deals with files and pandas
input formats. The model core works on plain arrays.
"""

from __future__ import annotations

from os import PathLike

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .params import BASIN_CHAR_FIELDS, GW_HYD_FIELDS, PARAMETERS, FlowMetrics, ModelParameters

#: Legacy parameter-table columns for `FlowMetrics`, in ``flow`` list order.
FLOW_COLUMNS = ("Qthresh", "Rs", "Rb1", "Rb2", "Prec", "Frac4Rise")


def streamflow_arrays(streamflow: pd.DataFrame) -> tuple[pd.Series, NDArray[np.float64]]:
    """Split a ``Date``/``Streamflow`` table into dates and a flow array.

    Parameters
    ----------
    streamflow : pd.DataFrame
        Columns ``Date`` (datetime-like) and ``Streamflow`` [m³/day].

    Returns
    -------
    dates : pd.Series
        ``pd.to_datetime(streamflow["Date"])``, keeping the input index.
    q_obs : ndarray
        Observed flow [m³/day] as a new array.
    """
    missing = {"Date", "Streamflow"} - set(streamflow.columns)
    if missing:
        raise KeyError(f"streamflow table is missing columns {sorted(missing)}")
    return pd.to_datetime(streamflow["Date"]), np.array(streamflow["Streamflow"])


def read_parameter_table(source: str | PathLike | pd.DataFrame,
                         ) -> dict[object, tuple[ModelParameters, FlowMetrics]]:
    """Read a parameter table in the USGS/pybfs column layout.

    Parameters
    ----------
    source : path or DataFrame
        Table with a ``site_no`` column and the columns ``AREA, Lb, X1, Wb, POR,
        ALPHA, BETA, Ks, Kb, Kz, Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise`` (the
        format of ``docs/files/bfs_params_50.csv``). Files written by
        ``bfs_calibrate`` use ``tmp.site`` and ``tmp.area`` instead of
        ``site_no`` and ``AREA``; those are accepted too.

    Returns
    -------
    dict
        ``{site_no: (ModelParameters, FlowMetrics)}``.
    """
    table = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    table = table.rename(columns={"tmp.site": "site_no", "tmp.area": "AREA"})
    symbols = [PARAMETERS[f].symbol for f in BASIN_CHAR_FIELDS + GW_HYD_FIELDS]
    out = {}
    for _, row in table.iterrows():
        params = ModelParameters(**dict(zip(BASIN_CHAR_FIELDS + GW_HYD_FIELDS, row[symbols].values)))
        out[row["site_no"]] = (params, FlowMetrics(*row[list(FLOW_COLUMNS)].values))
    return out


def site_parameters(table: dict, site_no) -> tuple[ModelParameters, FlowMetrics]:
    """Look up one site in the output of `read_parameter_table`; raise if absent."""
    if site_no in table:
        return table[site_no]
    example = next(iter(table), None)
    hint = f" (keys look like {example!r}, type {type(example).__name__})" if example is not None else ""
    raise KeyError(f"site {site_no!r} not in parameter table{hint}")


def get_values_for_site(df, site_no):
    """Extracts site-specific parameters from a DataFrame (pybfs 0.2.0 interface)

    Parameters
    ----------
    df : pd.DataFrame
        Columns site_no, AREA, Lb, X1, Wb, POR, ALPHA, BETA, Ks, Kb, Kz, Qthresh,
        Rs, Rb1, Rb2, Prec, Frac4Rise
    site_no : int
        Site identification number

    Returns
    -------
    tuple of (basin_char, gw_hyd, flow)
        Lists [AREA, Lb, X1, Wb, POR], [ALPHA, BETA, Ks, Kb, Kz] and [Qthresh, Rs,
        Rb1, Rb2, Prec, Frac4Rise]; each is ``None`` if ``site_no`` is not in the
        table (use `read_parameter_table` and `site_parameters` to get an error
        instead).
    """
    symbols = {name: PARAMETERS[name].symbol for name in BASIN_CHAR_FIELDS + GW_HYD_FIELDS}
    basin_char_columns = [symbols[f] for f in BASIN_CHAR_FIELDS]
    gw_hyd_columns = [symbols[f] for f in GW_HYD_FIELDS]
    rows = {}
    for _, row in df.iterrows():  # the last row wins for duplicated sites, as before
        rows[row['site_no']] = (list(row[basin_char_columns].values), list(row[gw_hyd_columns].values),
                                list(row[list(FLOW_COLUMNS)].values))
    return rows.get(site_no, (None, None, None))
