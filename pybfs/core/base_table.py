"""Storage-discharge lookup table for the base reservoir.

The base reservoir's water surface is ``Zb(x) = (x / X1)^β`` (manual eq. 5),
measured from the basin outlet. Discharge follows from Darcy's law at the
intersect Xb (eq. 8) and storage from integrating the profile up to Xb plus a
level reservoir from Xb to Lb (eq. 7, here evaluated in closed form). The
table tabulates Xb, Zb, Sb, and Qb for discharges spanning the observed flow
range, and the model interpolates nothing: it looks values up by row.

Construction reproduces pybfs 0.2.0 exactly, including the rounding of every
column to 5 significant digits (R's ``signif``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: Number of log-spaced discharges (a zero row is prepended).
N_DISCHARGES = 1000
SIGNIFICANT_DIGITS = 5


@dataclass(frozen=True, slots=True)
class BaseTable:
    """Base-reservoir lookup table, sorted by discharge.

    Attributes
    ----------
    intersect_distance : ndarray
        Xb, where the base water surface meets the channel [m].
    saturated_thickness : ndarray
        Zb at Xb [m].
    storage : ndarray
        Sb [m³].
    discharge : ndarray
        Qb [m³/day].
    row_index : ndarray of int
        Position of each row in the unfiltered table (rows with Xb ≥ Lb are
        dropped). Matches the index of the legacy ``base_table`` DataFrame.
    monotone : tuple of bool
        Whether ``(intersect_distance, storage, discharge)`` are non-decreasing
        and NaN-free, which allows binary-search lookups.
    """

    intersect_distance: Array
    saturated_thickness: Array
    storage: Array
    discharge: Array
    row_index: NDArray[np.int64]
    monotone: tuple[bool, bool, bool]

    @property
    def max_storage(self) -> float:
        """Largest storage in the table, floored at 0 (as the model loop computes it)."""
        return max_storage(self.storage)

    def has_nan(self) -> bool:
        return bool(np.isnan(self.intersect_distance).any() or np.isnan(self.saturated_thickness).any()
                    or np.isnan(self.storage).any() or np.isnan(self.discharge).any())

    def __len__(self) -> int:
        return len(self.discharge)


def positive_flow_range(q: Array) -> tuple[float, float]:
    """Smallest and largest positive, non-missing flow [m³/day]."""
    q = np.asarray(q)
    positive = q[~np.isnan(q)]
    positive = positive[positive > 0]
    return np.min(positive), np.max(positive)


def build_base_table(basin_length: float, base_profile_scale: float, base_width: float,
                     base_profile_exponent: float, base_conductivity: float, drainable_porosity: float,
                     q_min: float, q_max: float) -> BaseTable:
    """Build the base-reservoir table.

    Parameters
    ----------
    basin_length, base_profile_scale, base_width : float
        Lb, X1, Wb [m].
    base_profile_exponent : float
        β [–].
    base_conductivity : float
        Kb [m/day].
    drainable_porosity : float
        POR [–].
    q_min, q_max : float
        Positive flow range of the record [m³/day] (`positive_flow_range`).
        The table spans ``q_min/10`` to ``q_max``.
    """
    lb, x1, wb, b, kb, por = (basin_length, base_profile_scale, base_width,
                              base_profile_exponent, base_conductivity, drainable_porosity)
    log_range = np.log10([q_min / 10, q_max])
    q = np.concatenate([[0], np.logspace(log_range[0], log_range[1], num=N_DISCHARGES)])

    # Water-surface height at the intersect for each discharge (inverse of eq. 8 with eq. 5-6)
    if b != 0.5:
        z = ((q * x1 * (2 * b - 1)) / (wb * kb * b ** 2)) ** (b / (2 * b - 1))
    else:
        z = np.exp(q * 2 * x1 / (wb * kb))
    x = x1 * z ** (1 / b)
    s = wb * por * ((1 / x1 ** b) * (1 / (b + 1)) * x ** (b + 1) + (lb - x) * z)

    xb, z, s, q = (round_significant(v, SIGNIFICANT_DIGITS) for v in (x, z, s, q))
    keep = xb < lb
    columns = (xb[keep], z[keep], s[keep], q[keep])
    return BaseTable(*columns, row_index=np.flatnonzero(keep),
                     monotone=tuple(is_monotone(c) for c in (columns[0], columns[2], columns[3])))


def round_significant(x: Array, digits: int) -> Array:
    """Round to ``digits`` significant digits, like R's ``signif`` (zeros and NaN unchanged)."""
    keep = (x == 0) | np.isnan(x)
    result = np.zeros_like(x)
    rest = ~keep
    if np.any(rest):
        order = np.floor(np.log10(np.abs(x[rest])))
        result[rest] = np.round(x[rest] / (10 ** order), digits - 1) * (10 ** order)
    result[keep] = x[keep]
    return result


def is_monotone(values: Array) -> bool:
    """True if ``values`` is non-decreasing and has no NaN (binary-search lookups are then exact)."""
    return bool(not np.isnan(values).any() and np.all(np.diff(values) >= 0))


# ---------------------------------------------------------------------------
# Lookups
#
# The legacy code finds table rows by counting entries that satisfy a
# comparison. Two counting semantics occur:
#   leading: count until the first entry that fails (the numba model loop);
#   total:   count every entry that satisfies it (pandas ``.sum()``, forecast).
# For a non-decreasing, NaN-free column both equal a binary search, which is
# what these helpers use; otherwise they scan exactly like the legacy code.
# ---------------------------------------------------------------------------

@njit(cache=True)
def count_leading(values, threshold, strict, monotone):
    """Number of leading entries with ``v < threshold`` (strict) or ``v <= threshold``."""
    if monotone:
        return _bisect(values, threshold, strict)
    count = 0
    for v in values:
        if (v < threshold) if strict else (v <= threshold):
            count += 1
        else:
            break
    return count


@njit(cache=True)
def count_total(values, threshold, strict, monotone):
    """Number of entries with ``v < threshold`` (strict) or ``v <= threshold``."""
    if monotone:
        return _bisect(values, threshold, strict)
    count = 0
    for v in values:
        if (v < threshold) if strict else (v <= threshold):
            count += 1
    return count


@njit(cache=True)
def _bisect(values, threshold, strict):
    if np.isnan(threshold):
        return 0
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if (values[mid] < threshold) if strict else (values[mid] <= threshold):
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def max_storage(storage):
    """Maximum of ``storage``, floored at 0 and ignoring NaN (legacy loop semantics)."""
    largest = 0.0
    for s in storage:
        if s > largest:
            largest = s
    return largest
