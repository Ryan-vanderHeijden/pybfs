"""Model parameters, flow metrics, and the parameter registry.

Definitions follow the USGS BFS manual (Konrad, 2022, SIR 2022-5114; "the
manual"), tables 3-5. All lengths are in the length unit of the streamflow
volumes (the bundled examples use m and m³/day), and all rates are per model
time step (one day).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

Transform = Literal["log10", "linear"]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Description of one model parameter.

    Attributes
    ----------
    name : str
        Field name in `ModelParameters`.
    symbol : str
        Symbol in the manual, which is also the column name in legacy
        parameter tables (e.g. ``"Lb"``).
    units : str
        Units, assuming streamflow in m³/day.
    description : str
        Definition, following the manual.
    transform : {"log10", "linear"}
        Default search space for calibration.
    bounds : tuple of float or None
        Physical limits (in natural units) for new calibration problems. The
        legacy USGS procedure uses its own, step-specific limits (see
        `pybfs.calibration.usgs`).
    """

    name: str
    symbol: str
    units: str
    description: str
    transform: Transform = "log10"
    bounds: tuple[float, float] | None = None


PARAMETERS: dict[str, ParameterSpec] = {
    spec.name: spec
    for spec in (
        ParameterSpec("drainage_area", "AREA", "m²",
                      "Measured drainage area upstream of the site (manual table 4). "
                      "Limits Lb × Wb. Not calibrated.", "linear"),
        ParameterSpec("basin_length", "Lb", "m",
                      "Effective length of the basin and channel (manual table 4); also the width "
                      "of the surface reservoir perpendicular to its flow (eq. 9).", "log10"),
        ParameterSpec("base_profile_scale", "X1", "m",
                      "Scaling parameter of the base-reservoir thickness, Zb = (x / X1)^β (manual eq. 5).",
                      "log10"),
        ParameterSpec("base_width", "Wb", "m",
                      "Effective basin width, perpendicular to the channel; the width of the base "
                      "reservoir (manual table 4, eq. 8). Each hillslope is Wb/2 wide.", "log10"),
        ParameterSpec("drainable_porosity", "POR", "–",
                      "Effective drainable porosity of both reservoirs (manual table 4).", "linear", (0.0, 1.0)),
        ParameterSpec("surface_gradient", "ALPHA", "–",
                      "Effective lateral hydraulic gradient of the surface reservoir: slope of its upper "
                      "boundary, so that Zs = α·Xs (manual table 5, eq. 3).", "log10", (1e-8, 0.1)),
        ParameterSpec("base_profile_exponent", "BETA", "–",
                      "Exponent of the base water-surface function Zb = (x / X1)^β (manual table 5, eq. 5).",
                      "linear", (0.5, 20.0)),
        ParameterSpec("surface_conductivity", "Ks", "m/day",
                      "Effective hydraulic conductivity of the surface reservoir (manual table 5, eq. 9). "
                      "The code also uses it as the infiltration-rate cap (manual eq. 14 says Kz).",
                      "log10", (1e-8, 1e5)),
        ParameterSpec("base_conductivity", "Kb", "m/day",
                      "Effective horizontal hydraulic conductivity of the base reservoir (manual table 5, eq. 8).",
                      "log10", (1e-8, 1e5)),
        ParameterSpec("vertical_conductivity", "Kz", "m/day",
                      "Effective vertical hydraulic conductivity for recharge (manual table 5). Used as the "
                      "maximum recharge depth per time step in min(Kz, POR·Zs) (eq. 16).",
                      "log10", (1e-8, 1e5)),
    )
}

#: Order of the legacy ``basin_char`` and ``gw_hyd`` lists.
BASIN_CHAR_FIELDS = ("drainage_area", "basin_length", "base_profile_scale", "base_width", "drainable_porosity")
GW_HYD_FIELDS = ("surface_gradient", "base_profile_exponent", "surface_conductivity",
                 "base_conductivity", "vertical_conductivity")


@dataclass(frozen=True, slots=True)
class ModelParameters:
    """Geometry and hydraulic parameters of the BFS model.

    Attributes
    ----------
    drainage_area : float
        AREA, measured drainage area [m²].
    basin_length : float
        Lb, effective basin and channel length [m].
    base_profile_scale : float
        X1, scale of the base water-surface profile [m].
    base_width : float
        Wb, effective basin width [m].
    drainable_porosity : float
        POR [–].
    surface_gradient : float
        α, lateral hydraulic gradient of the surface reservoir [–].
    base_profile_exponent : float
        β, exponent of the base water-surface profile [–].
    surface_conductivity : float
        Ks [m/day].
    base_conductivity : float
        Kb [m/day].
    vertical_conductivity : float
        Kz [m/day], used as the maximum recharge depth per step.

    See `PARAMETERS` for full definitions and manual references.
    """

    drainage_area: float
    basin_length: float
    base_profile_scale: float
    base_width: float
    drainable_porosity: float
    surface_gradient: float
    base_profile_exponent: float
    surface_conductivity: float
    base_conductivity: float
    vertical_conductivity: float

    @property
    def surface_half_width(self) -> float:
        """Width of one hillslope of the surface reservoir, Wb/2 [m]."""
        return self.base_width / 2

    def replace(self, **changes: float) -> ModelParameters:
        """Return a copy with some fields replaced."""
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_legacy(cls, basin_char: Sequence[float], gw_hyd: Sequence[float]) -> ModelParameters:
        """Build from the legacy ``basin_char`` and ``gw_hyd`` lists.

        Values are kept as given (including numpy scalar types), so results are
        unchanged by the conversion.
        """
        values = dict(zip(BASIN_CHAR_FIELDS, basin_char))
        values.update(zip(GW_HYD_FIELDS, gw_hyd))
        return cls(**values)

    def to_legacy(self) -> tuple[list[float], list[float]]:
        """Return the legacy ``(basin_char, gw_hyd)`` lists."""
        return ([getattr(self, f) for f in BASIN_CHAR_FIELDS],
                [getattr(self, f) for f in GW_HYD_FIELDS])

    def to_vector(self, names: Sequence[str], transform: Transform | Sequence[Transform] = "linear",
                  ) -> NDArray[np.float64]:
        """Return selected parameters as a search-space vector."""
        transforms = _expand(transform, len(names))
        return np.array([_forward(getattr(self, n), t) for n, t in zip(names, transforms)])

    def with_vector(self, names: Sequence[str], x: Sequence[float],
                    transform: Transform | Sequence[Transform] = "linear") -> ModelParameters:
        """Return a copy with ``names`` set from search-space vector ``x``."""
        transforms = _expand(transform, len(names))
        return self.replace(**{n: _inverse(v, t) for n, v, t in zip(names, x, transforms)})


@dataclass(frozen=True, slots=True)
class FlowMetrics:
    """Streamflow characteristics derived from the record (manual table 3).

    They are computed by `pybfs.streamflow_metrics.flow_metrics` and are not
    calibrated.

    Attributes
    ----------
    baseflow_threshold : float
        Qthresh [m³/day].
    storm_recession_rate : float
        Rs, first-order storm-flow recession coefficient, negative [1/day].
        Sets the error weights (`pybfs.objectives`).
    base_recession_rate_at_mean : float
        Rb1, baseflow recession coefficient at mean flow, negative [1/day].
    base_recession_rate_low : float
        Rb2, low-baseflow recession coefficient, negative [1/day]. The code's
        definition differs from the manual's (suspected issue M13).
    flow_precision : float
        Prec, low-flow measurement precision [m³/day].
    rise_fraction : float
        Frac4Rise, minimum fractional increase that marks a rise [–].
    recession_curve : tuple of float or None
        (RbI, RbS): intercept and slope of the 10th-percentile quantile
        regression of 10-day recession rate on log10 Q. ``None`` when not
        available (e.g. metrics read from a parameter table).
    """

    baseflow_threshold: float
    storm_recession_rate: float
    base_recession_rate_at_mean: float
    base_recession_rate_low: float
    flow_precision: float
    rise_fraction: float
    recession_curve: tuple[float, float] | None = None

    @classmethod
    def from_legacy(cls, flow: Sequence) -> FlowMetrics:
        """Build from the legacy ``flow`` list (6 values, optionally ``rb10`` 7th)."""
        curve = None
        if len(flow) > 6 and flow[6] is not None:
            curve = (flow[6][0], flow[6][1])
        return cls(*flow[:6], recession_curve=curve)

    def to_legacy(self) -> list[float]:
        """Return the legacy 6-element ``flow`` list."""
        return [self.baseflow_threshold, self.storm_recession_rate, self.base_recession_rate_at_mean,
                self.base_recession_rate_low, self.flow_precision, self.rise_fraction]


def _expand(transform, n):
    return [transform] * n if isinstance(transform, str) else list(transform)


def _forward(value: float, transform: Transform) -> float:
    return np.log10(value) if transform == "log10" else value


def _inverse(value: float, transform: Transform) -> float:
    return 10 ** value if transform == "log10" else value
