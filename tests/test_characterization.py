"""Regression tests: current outputs must match tests/reference/ (baseline-original).

A failure means behavior changed. Treat it as a finding to explain, not a
reference to regenerate.
"""

import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import cases

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

# Forward computations are bit-identical across reruns; 1e-12 leaves room only
# for harmless reordering of floating-point operations. Calibration runs pass
# through ~900 Nelder-Mead decisions, so allow more, but still far below any
# change a person would call "the same answer, slightly different".
RTOL = {"calibration_12167000": 1e-9, "calibration_2312200": 1e-9}
DEFAULT_RTOL = 1e-12


def _check_versions():
    meta = json.loads((REFERENCE_DIR / "meta.json").read_text())
    from reference.generate import versions
    current = versions()
    changed = {k: (meta["versions"].get(k), v) for k, v in current.items()
               if k != "platform" and meta["versions"].get(k) != v}
    if changed:
        warnings.warn(f"Library versions differ from those used for the references: {changed}")


def _assert_matches(name, result, reference, rtol):
    assert sorted(result) == sorted(reference.files), f"{name}: set of outputs changed"
    for key in reference.files:
        expected = reference[key]
        actual = np.asarray(result[key])
        assert actual.shape == expected.shape, f"{name}/{key}: shape {actual.shape} != {expected.shape}"
        if expected.dtype.kind == "f":
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0, equal_nan=True,
                                       err_msg=f"{name}/{key}")
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=f"{name}/{key}")


def _params(case_map, marks=()):
    return [pytest.param(name, id=name, marks=marks) for name in case_map]


@pytest.mark.parametrize(
    "name", _params(cases.FAST_CASES) + _params(cases.SLOW_CASES, marks=pytest.mark.slow))
def test_matches_reference(name):
    _check_versions()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = cases.ALL_CASES[name]()
    with np.load(REFERENCE_DIR / f"{name}.npz") as reference:
        _assert_matches(name, result, reference, RTOL.get(name, DEFAULT_RTOL))


def test_bfs_does_not_mutate_input():
    sf, bc, gw, fl = cases.load_2312200()
    before = sf.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        cases.pybfs.bfs(sf, cases.sbt_for(sf, bc, gw), bc, gw, fl)
    assert sf.equals(before)
