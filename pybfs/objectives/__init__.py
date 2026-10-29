"""Objective functions: pure functions of observed and simulated series.

An objective takes observed and simulated arrays (and optional per-step
weights) and returns a scalar to **minimize**. It knows nothing about the
model. Which series is scored, with which weights and on which days, is set
by a `pybfs.calibration.problem.ScoreSpec`.

`legacy` holds the error measures of the USGS procedure, reproduced exactly;
its calibration objective needs more than one series, so the calibration
problem calls it through `pybfs.calibration.usgs.usgs_objective`.

Adding an objective (e.g. NSE or KGE) means adding one function with this
signature; model code is unaffected.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from . import legacy  # noqa: F401


class Objective(Protocol):
    """``objective(q_obs, q_sim, weights) -> float``, lower is better."""

    def __call__(self, q_obs: NDArray[np.float64], q_sim: NDArray[np.float64],
                 weights: NDArray[np.float64] | None) -> float: ...
