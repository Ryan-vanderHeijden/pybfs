"""Bit-for-bit equivalence of the new core with the legacy implementation.

These complement the characterization tests: they draw many random inputs
and require *exact* equality with ``pybfs.legacy``.
"""

import numpy as np
import pytest

from pybfs.core import geometry as g
from pybfs.legacy import utilities as legacy

N_RANDOM = 100_000

KERNELS = [
    (g.surface_storage_from_thickness, legacy.sur_store_jit),
    (g.surface_thickness_from_storage, legacy.sur_z_jit),
    (g.surface_discharge, legacy.sur_q_jit),
    (g.saturated_area_runoff, legacy.dir_q_jit),
    (g.infiltration_rate, legacy.infiltration_jit),
    (g.recharge_rate, legacy.recharge_jit),
]


def _random_geometry(rng, n):
    lb = 10 ** rng.uniform(2, 5, n)
    alpha = 10 ** rng.uniform(-4, -1, n)
    ws = 10 ** rng.uniform(1, 4, n)
    por = rng.uniform(0.01, 0.5, n)
    zs = rng.uniform(0, 1, n) * ws * alpha
    return lb, alpha, ws, por, zs


def _args(new, rng, n):
    lb, alpha, ws, por, zs = _random_geometry(rng, n)
    k = 10 ** rng.uniform(-8, 5, n)
    impulse = np.where(rng.random(n) < 0.2, 0.0, 10 ** rng.uniform(-6, 1, n))
    xb = rng.uniform(0, 1, n) * lb
    storage = lb * (2 * ws * zs - zs * zs / alpha) * por * rng.uniform(0, 1.5, n)
    return {
        g.surface_storage_from_thickness: (lb, alpha, ws, por, zs),
        g.surface_thickness_from_storage: (lb, alpha, ws, por, storage),
        g.surface_discharge: (lb, alpha, k, zs),
        g.saturated_area_runoff: (lb, alpha, zs, impulse),
        g.infiltration_rate: (lb, ws, k, alpha, zs, impulse),
        g.recharge_rate: (lb, xb, ws, k, zs, por),
    }[new]


@pytest.mark.parametrize("new, old", KERNELS, ids=[k[0].__name__ for k in KERNELS])
def test_kernels_bitwise(new, old):
    rng = np.random.default_rng(20261029)
    args = _args(new, rng, N_RANDOM)
    a = np.array([new(*row) for row in zip(*args)])
    b = np.array([old(*row) for row in zip(*args)])
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Base table
# ---------------------------------------------------------------------------

import warnings  # noqa: E402

import pandas as pd  # noqa: E402

from pybfs.core import base_table as bt  # noqa: E402

import cases  # noqa: E402


def _random_tables(n, seed=7):
    """Random parameter sets around both example sites, beta across the sweep range."""
    rng = np.random.default_rng(seed)
    q = cases.load_12167000()["Streamflow"].to_numpy()
    for i in range(n):
        lb = 10 ** rng.uniform(3, 5)
        params = dict(
            lb=lb, x1=10 ** rng.uniform(0, 4), wb=10 ** rng.uniform(1.5, 4),
            beta=[0.5, 0.6, 1.0, 2.5, 10.1][i % 5] if i < 10 else rng.uniform(0.55, 10.5),
            kb=10 ** rng.uniform(-3, 5), por=rng.uniform(0.02, 0.4))
        yield params, q * (10 ** rng.uniform(-2, 2))


def test_base_table_matches_legacy():
    for params, q in _random_tables(300):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            old = legacy.base_table(params["lb"], params["x1"], params["wb"], params["beta"], params["kb"],
                                    pd.DataFrame({"Streamflow": q}), params["por"])
            new = bt.build_base_table(params["lb"], params["x1"], params["wb"], params["beta"], params["kb"],
                                      params["por"], *bt.positive_flow_range(q))
        np.testing.assert_array_equal(new.row_index, old.index.to_numpy())
        for col, attr in (("Xb", "intersect_distance"), ("Z", "saturated_thickness"),
                          ("S", "storage"), ("Q", "discharge")):
            np.testing.assert_array_equal(getattr(new, attr), old[col].to_numpy(), err_msg=f"{col} {params}")


def _legacy_leading(values, threshold, strict):
    """Transcription of the counting loops in legacy ``_bfs_core_loop``."""
    count = 0
    for v in values:
        if (v < threshold) if strict else (v <= threshold):
            count += 1
        else:
            break
    return count


def _queries(values, rng):
    finite = values[np.isfinite(values)]
    picks = rng.choice(finite, 50) if len(finite) else np.array([])
    span = (finite.min(), finite.max()) if len(finite) else (0.0, 1.0)
    return np.concatenate([picks, np.nextafter(picks, np.inf), np.nextafter(picks, -np.inf),
                           rng.uniform(span[0] - 1, span[1] * 1.1 + 1, 50),
                           [np.nan, -np.inf, np.inf, 0.0, -1.0]])


@pytest.mark.parametrize("strict", [False, True])
def test_lookups_match_legacy_semantics(strict):
    rng = np.random.default_rng(11)
    for params, q in _random_tables(60):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            table = bt.build_base_table(params["lb"], params["x1"], params["wb"], params["beta"],
                                        params["kb"], params["por"], *bt.positive_flow_range(q))
        columns = (table.intersect_distance, table.storage, table.discharge)
        for values, monotone in zip(columns, table.monotone):
            series = pd.Series(values)
            for t in _queries(values, rng):
                expected_total = int((series < t).sum() if strict else (series <= t).sum())
                assert bt.count_total(values, t, strict, monotone) == expected_total
                assert bt.count_leading(values, t, strict, monotone) == _legacy_leading(values, t, strict)


def test_lookups_fall_back_for_unsorted_columns():
    values = np.array([0.0, 2.0, 1.0, np.nan, 3.0])
    assert not bt._is_monotone(values)
    for t in (0.5, 1.5, 2.5, 5.0, np.nan):
        for strict in (False, True):
            assert bt.count_leading(values, t, strict, False) == _legacy_leading(values, t, strict)
            assert bt.count_total(values, t, strict, False) == int(
                (pd.Series(values) < t).sum() if strict else (pd.Series(values) <= t).sum())


def test_max_storage_matches_legacy_loop():
    for values in (np.array([0.0, 3.0, 2.0]), np.array([-1.0, -2.0]), np.array([np.nan, 1.0]), np.array([])):
        largest = 0.0
        for s in values:
            if s > largest:
                largest = s
        assert bt.max_storage(values) == largest
