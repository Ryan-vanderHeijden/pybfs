"""USGS scoring options (suspected issue M13)."""

import numpy as np
import pytest

import pybfs
from pybfs.calibration import usgs
from pybfs.frames import legacy_bfs_frame
from pybfs.objectives.usgs import recession_weights, relative_error
from pybfs.params import FlowMetrics, ModelParameters

import cases


@pytest.fixture(scope="module")
def run():
    sf = cases.load_12167000()
    bc, gw, fl = cases.params_12167000()
    flow = FlowMetrics.from_legacy(fl)
    result = pybfs.simulate(sf["Streamflow"].to_numpy(), ModelParameters.from_legacy(bc, gw), flow)
    return sf, result, flow, fl


def test_default_matches_legacy_objective_and_error(run):
    sf, result, flow, fl = run
    frame = legacy_bfs_frame(result, sf["Date"], flow, "base")
    assert usgs.usgs_objective(result, flow) == pybfs.objective(frame, fl[4])
    assert usgs.model_error(result, flow) == pybfs.calibrate.calculate_error(frame)


def test_spinup_excludes_leading_steps(run):
    _, result, flow, _ = run
    _, weights = usgs.UsgsScoring(spinup_steps=100).errors_and_weights(result, flow)
    _, default = usgs.DEFAULT_SCORING.errors_and_weights(result, flow)
    assert np.all(weights[:100] == 0)
    np.testing.assert_array_equal(weights[100:], default[100:])


def test_threshold_excludes_low_flows(run):
    _, result, flow, _ = run
    _, weights = usgs.UsgsScoring(exclude_below_threshold=True).errors_and_weights(result, flow)
    low = result.q_obs < flow.baseflow_threshold
    assert low.any()  # four zero-flow days at 12167000
    assert np.all(weights[low] == 0)


@pytest.mark.parametrize("basis, rate_field", [("base", "base_recession_rate_low"),
                                              ("total", "base_recession_rate_at_mean")])
def test_manual_weight_rate(run, basis, rate_field):
    _, result, flow, _ = run
    scoring = usgs.UsgsScoring(error_basis=basis, weight_rate="manual")
    rel_error, weights = scoring.errors_and_weights(result, flow)
    expected = recession_weights(rel_error, result.recession_age, getattr(flow, rate_field))
    np.testing.assert_array_equal(weights, expected)
    assert scoring.recession_rate(flow) == getattr(flow, rate_field)


def test_unknown_weight_rate_raises(run):
    _, result, flow, _ = run
    with pytest.raises(ValueError):
        usgs.UsgsScoring(weight_rate="rb3").errors_and_weights(result, flow)


def test_excluded_steps_do_not_affect_score(run):
    """Changing simulated flow inside the spin-up must not change the objective."""
    _, result, flow, _ = run
    scoring = usgs.UsgsScoring(spinup_steps=50)
    rel_error, weights = scoring.errors_and_weights(result, flow)
    perturbed = rel_error.copy()
    perturbed[:50] *= 3
    from pybfs.objectives.usgs import usgs_weighted_error
    a = usgs_weighted_error(result.q_obs, result.baseflow, rel_error, weights, flow_precision=flow.flow_precision)
    b = usgs_weighted_error(result.q_obs, result.baseflow, perturbed, weights, flow_precision=flow.flow_precision)
    assert a == b


@pytest.mark.slow
def test_calibrate_with_manual_scoring():
    sf = cases.load_12167000()
    result = usgs.calibrate("12167000", 6.71e8, sf["Streamflow"].to_numpy(), sf["Date"].to_numpy(),
                            scoring=usgs.MANUAL_SCORING)
    assert 0 < result.baseflow_fraction < 1
    assert np.isfinite(result.error)
