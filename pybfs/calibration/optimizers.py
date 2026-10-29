"""Optimizers for calibration problems.

An optimizer minimizes ``f(x) -> float`` from a starting point and returns a
`scipy.optimize.OptimizeResult` (at least ``x`` and ``fun``). New optimizers
for comparison (Phase 4) implement `Optimizer`; the USGS procedure uses
`nelder_mead_parscale`.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize


class Optimizer(Protocol):
    def __call__(self, f: Callable[[NDArray[np.float64]], float], x0: NDArray[np.float64]) -> OptimizeResult: ...


def nelder_mead_parscale(f: Callable[[NDArray[np.float64]], float], x0, scale, *,
                         maxiter: int = 1000, fatol: float = 1e-4) -> OptimizeResult:
    """Nelder-Mead in scaled coordinates, emulating R's ``optim(..., parscale=scale)``.

    The search runs on ``x / scale`` (zeros in ``scale`` are replaced by 1), so
    with ``scale = x0`` every search starts at the all-ones vector with scipy's
    default 5% initial simplex. The returned ``x`` is unscaled if finite.

    Reproduces pybfs 0.2.0 ``_create_scaled_minimize`` exactly. In particular,
    R's ``reltol=0.01`` stopping rule is **not** implemented; scipy stops on
    ``fatol`` and its default ``xatol=1e-4`` (suspected issue C2).
    """
    scale = np.array(scale, dtype=float)
    x0 = np.array(x0, dtype=float)
    scale = np.where(scale == 0, 1.0, scale)

    def scaled(z):
        return f(z * scale)

    result = minimize(scaled, x0 / scale, method='Nelder-Mead', options={'maxiter': maxiter, 'fatol': fatol})
    if np.all(np.isfinite(result.x)):
        result.x = result.x * scale
    return result
