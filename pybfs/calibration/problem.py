"""Calibration problems: parameters in, a score out, no hidden state.

A `CalibrationProblem` maps a search vector to a score by updating some
parameters of a base parameter set, running the model on a prepared record,
and scoring the result. Everything that depends only on the record (cleaned
flows, rise/recession signals, the flow range of the base table) is computed
once in `Record`. Problems keep no state between evaluations, so they are
safe to evaluate in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.base_table import build_base_table, positive_flow_range
from ..core.hydrograph import HydrographSignals, clean_flow, hydrograph_signals
from ..core.model import simulate
from ..core.results import SimulationResult
from ..objectives import Objective
from ..params import FlowMetrics, ModelParameters, Transform

Array = NDArray[np.float64]

#: A scorer maps a simulation to the value to minimize.
Scorer = Callable[[SimulationResult, FlowMetrics], float]


@dataclass(frozen=True, slots=True)
class Record:
    """An observed daily record prepared for repeated model runs.

    Attributes
    ----------
    q_obs : ndarray
        Flow as given [m³/day] (negative values kept).
    q : ndarray
        Flow with negative values set to NaN (what the model uses).
    signals : HydrographSignals
        Rise/recession flags for ``q`` (depend on the flow metrics).
    flow_range : tuple of float
        Smallest and largest positive flow, which set the base-table range.
    """

    q_obs: Array
    q: Array
    signals: HydrographSignals
    flow_range: tuple[float, float]

    @classmethod
    def prepare(cls, q_obs, flow: FlowMetrics) -> Record:
        q_obs = np.asarray(q_obs)
        q = clean_flow(q_obs)
        return cls(q_obs, q, hydrograph_signals(q, flow.flow_precision, flow.rise_fraction),
                   positive_flow_range(q_obs))


@dataclass(frozen=True)
class ScoreSpec:
    """Score a simulation with a generic `Objective`.

    Attributes
    ----------
    objective : Objective
        ``objective(q_obs, q_sim, weights) -> float``.
    series : {"q_sim", "baseflow"}
        Which simulated series is compared with the observations.
    weights : callable, optional
        ``weights(result, simulated_series, flow) -> array`` of per-step weights.
    eval_mask : ndarray of bool, optional
        Steps to score, e.g. a calibration period or the steps after a spin-up.
        The model always runs over the full record.
    """

    objective: Objective
    series: Literal["q_sim", "baseflow"] = "q_sim"
    weights: Callable[[SimulationResult, Array, FlowMetrics], Array] | None = None
    eval_mask: NDArray[np.bool_] | None = None

    def __call__(self, result: SimulationResult, flow: FlowMetrics) -> float:
        simulated = result.baseflow if self.series == "baseflow" else result.q_sim
        observed = result.q_obs
        weights = self.weights(result, simulated, flow) if self.weights is not None else None
        if self.eval_mask is not None:
            observed, simulated = observed[self.eval_mask], simulated[self.eval_mask]
            weights = weights[self.eval_mask] if weights is not None else None
        return self.objective(observed, simulated, weights)


@dataclass(frozen=True)
class CalibrationProblem:
    """Objective function over a subset of model parameters.

    Attributes
    ----------
    record : Record
    flow : FlowMetrics
    base_params : ModelParameters
        Values of the parameters that are not searched.
    names : sequence of str
        `ModelParameters` fields in the search vector, in order.
    transform : "log10", "linear", or a sequence of them
        Search-space transform of each name.
    score : Scorer
        Value to minimize, from the simulation.
    is_feasible : callable, optional
        ``is_feasible(x) -> bool`` on the *search vector*. Infeasible points
        score ``penalty`` without running the model.
    penalty : float
    on_simulate : callable, optional
        Called with the `ModelParameters` of every model run (for tracing).
    """

    record: Record
    flow: FlowMetrics
    base_params: ModelParameters
    names: Sequence[str]
    transform: Transform | Sequence[Transform]
    score: Scorer
    is_feasible: Callable[[Array], bool] | None = None
    penalty: float = 100.0
    on_simulate: Callable[[ModelParameters], None] | None = None

    def parameters(self, x) -> ModelParameters:
        """The full parameter set for search vector ``x``."""
        return self.base_params.with_vector(self.names, x, self.transform)

    def simulate(self, params: ModelParameters) -> SimulationResult:
        return run_simulation(self.record, params, self.flow, self.on_simulate)

    def evaluate(self, x) -> float:
        if self.is_feasible is not None and not self.is_feasible(x):
            return self.penalty
        return self.score(self.simulate(self.parameters(x)), self.flow)

    __call__ = evaluate


def run_simulation(record: Record, params: ModelParameters, flow: FlowMetrics,
                   on_simulate: Callable[[ModelParameters], None] | None = None) -> SimulationResult:
    """Run the model on a prepared record (base table built from ``params``)."""
    if on_simulate is not None:
        on_simulate(params)
    table = build_base_table(params.basin_length, params.base_profile_scale, params.base_width,
                             params.base_profile_exponent, params.base_conductivity, params.drainable_porosity,
                             *record.flow_range)
    return simulate(record.q_obs, params, flow, base_table=table, signals=record.signals)
