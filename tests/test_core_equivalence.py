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
