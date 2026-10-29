"""Model state and simulation results (plain arrays, no pandas)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ModelState:
    """Reservoir state at the end of a time step, used to start a forecast.

    Replaces the legacy 8-tuple ``(X, Zb.L, Zs.L, StBase, StSur, SurfaceFlow,
    Baseflow, Rech)``.

    Attributes
    ----------
    intersect_distance : float
        Xb, distance from the outlet to where the base water surface meets the channel [m].
    base_saturated_thickness : float
        Zb, saturated thickness of the base reservoir at Xb [m].
    surface_saturated_thickness : float
        Zs, saturated thickness of the surface reservoir at the section where
        the water surface intersects the land surface [m].
    base_storage : float
        Sb [m³].
    surface_storage : float
        Ss [m³].
    surface_flow : float
        Qs over the step [m³/day].
    baseflow : float
        Qb over the step [m³/day].
    recharge : float
        R over the step [m³/day].
    """

    intersect_distance: float
    base_saturated_thickness: float
    surface_saturated_thickness: float
    base_storage: float
    surface_storage: float
    surface_flow: float
    baseflow: float
    recharge: float

    def to_legacy(self) -> tuple[float, ...]:
        """Return the legacy ``initial`` tuple accepted by ``pybfs.forecast``."""
        return (self.intersect_distance, self.base_saturated_thickness, self.surface_saturated_thickness,
                self.base_storage, self.surface_storage, self.surface_flow, self.baseflow, self.recharge)

    @classmethod
    def from_legacy(cls, initial) -> ModelState:
        return cls(*initial)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Output of `pybfs.core.simulate.simulate`, one value per time step.

    Flows are step averages [m³/day]; storages and thicknesses are end-of-step
    values. NaN marks steps before the first observation, plus quantities the
    model leaves undefined on the first simulated step (direct runoff,
    impulse).

    Attributes
    ----------
    q_obs : ndarray
        Observed flow, negative values replaced by NaN.
    flow_components : ndarray, shape (n, 3)
        Surface flow, baseflow, and direct runoff, in that order. The three
        component properties below are views of this array.
    residual : ndarray
        q_obs minus the sum of components ("Eta").
    impulse : ndarray
        Inferred precipitation/snowmelt impulse I, a depth per step [m/day].
    surface_storage, base_storage : ndarray
        Ss and Sb [m³].
    surface_saturated_thickness : ndarray
        Zs [m] (see `ModelState`).
    base_saturated_thickness : ndarray
        Zb at the intersect [m].
    intersect_distance : ndarray
        Xb [m].
    infiltration, recharge : ndarray
        F and R, step averages [m³/day]. Infiltration on the first step is not
        meaningful (suspected issue M4).
    recession_age : ndarray of int
        Consecutive recession steps up to each step ("RecessCount.T").
    """

    q_obs: Array
    flow_components: Array
    residual: Array
    impulse: Array
    surface_storage: Array
    base_storage: Array
    surface_saturated_thickness: Array
    base_saturated_thickness: Array
    intersect_distance: Array
    infiltration: Array
    recharge: Array
    recession_age: NDArray[np.int64]

    @property
    def surface_flow(self) -> Array:
        return self.flow_components[:, 0]

    @property
    def baseflow(self) -> Array:
        return self.flow_components[:, 1]

    @property
    def direct_runoff(self) -> Array:
        return self.flow_components[:, 2]

    @property
    def q_sim(self) -> Array:
        """Total simulated flow; NaN on any step where a component is NaN."""
        total = np.sum(self.flow_components, axis=1)
        total[np.isnan(self.flow_components).any(axis=1)] = np.nan
        return total

    def final_state(self) -> ModelState:
        """State after the last step, for `pybfs.core.forecast.forecast`."""
        return ModelState(
            intersect_distance=self.intersect_distance[-1],
            base_saturated_thickness=self.base_saturated_thickness[-1],
            surface_saturated_thickness=self.surface_saturated_thickness[-1],
            base_storage=self.base_storage[-1],
            surface_storage=self.surface_storage[-1],
            surface_flow=self.surface_flow[-1],
            baseflow=self.baseflow[-1],
            recharge=self.recharge[-1],
        )


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """Output of `pybfs.core.forecast.forecast` (no observations, no impulses)."""

    baseflow: Array
    surface_storage: Array
    base_storage: Array
    surface_saturated_thickness: Array
    base_saturated_thickness: Array
    recharge: Array
