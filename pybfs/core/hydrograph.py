"""Rise and recession classification of an observed hydrograph.

These signals depend only on the record and the flow metrics, so a
calibration computes them once instead of once per model run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class HydrographSignals:
    """Per-step rise/recession flags of a daily record.

    Attributes
    ----------
    is_rise : ndarray of bool
        Flow rose by more than ``rise_fraction`` of the flow and by more than
        ``flow_precision``. Impulses are allowed on these steps and the step
        after (manual p. 8).
    is_recession : ndarray of bool
        Flow fell, or rose by less than either threshold.
    recession_age : ndarray of int
        Consecutive recession steps up to and including each step.
    """

    is_rise: NDArray[np.bool_]
    is_recession: NDArray[np.bool_]
    recession_age: NDArray[np.int64]


def clean_flow(q_obs) -> NDArray[np.float64]:
    """Copy of the observed flow with negative values set to NaN (manual p. 9)."""
    q = np.array(q_obs)
    q[q < 0] = np.nan
    return q


def hydrograph_signals(q: NDArray[np.float64], flow_precision: float, rise_fraction: float) -> HydrographSignals:
    """Classify each daily step of ``q`` (already cleaned) as rise and/or recession.

    Parameters
    ----------
    q : ndarray
        Observed daily flow [m³/day] from `clean_flow`.
    flow_precision : float
        Prec [m³/day].
    rise_fraction : float
        Frac4Rise [–].
    """
    n = len(q)
    dq = np.zeros(n)
    dq[1:] = q[1:] - q[:-1]

    # Fractional change, with the zero-flow conventions of the USGS code
    dq_frac = np.zeros_like(dq, dtype=float)
    dq_frac[(dq == 0) & (q == 0)] = 0
    dq_frac[(dq < 0) & (q == 0)] = 1
    nonzero = q != 0
    dq_frac[nonzero] = dq[nonzero] / q[nonzero]

    is_rise = (dq_frac > rise_fraction) & (dq > flow_precision)
    is_recession = (dq_frac <= rise_fraction) | (dq < flow_precision)
    count = np.cumsum(is_recession)
    recession_age = count - np.maximum.accumulate((~is_recession).astype(int) * count)
    return HydrographSignals(is_rise, is_recession, recession_age)
