"""Unit tests for the model core (no reference data needed)."""

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import pybfs
from pybfs.core import base_table as bt
from pybfs.core.forecasting import forecast
from pybfs.core.model import simulate
from pybfs.frames import base_table_from_frame
from pybfs.params import FlowMetrics, ModelParameters

import cases


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
def test_lookups_match_scan_semantics(strict):
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
    assert not bt.is_monotone(values)
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


def test_base_table_index_is_a_range_for_contiguous_rows():
    """pandas returns a RangeIndex after boolean selection of a prefix; the legacy calibration relied on it."""
    sf = cases.load_12167000()
    for beta in (1.0, 2.5):
        frame = pybfs.base_table(9072.0, 100.0, 2202.0, beta, 2850.0, sf, 0.15)
        assert isinstance(frame.index, pd.RangeIndex)


def _site_2312200():
    sf, bc, gw, fl = cases.load_2312200()
    return sf, ModelParameters.from_legacy(bc, gw), FlowMetrics.from_legacy(fl), bc, gw, fl


def test_hourly_raises():
    sf, _, _, bc, gw, fl = _site_2312200()
    sbt = cases.sbt_for(sf, bc, gw)
    with pytest.raises(NotImplementedError):
        pybfs.bfs(sf, sbt, bc, gw, fl, timestep="hour")
    with pytest.raises(NotImplementedError):
        pybfs.cal_base([bc[1], bc[3], gw[3], gw[4]], sf, "hour", "base", bc, gw, fl)


def test_use_jit_false_warns_and_matches_default():
    sf, _, _, bc, gw, fl = _site_2312200()
    sbt = cases.sbt_for(sf, bc, gw)
    with pytest.warns(DeprecationWarning):
        slow = pybfs.bfs(sf, sbt, bc, gw, fl, use_jit=False)
    pd.testing.assert_frame_equal(slow, pybfs.bfs(sf, sbt, bc, gw, fl), check_exact=True)


def test_forecast_first_step_repeats_initial_state():
    """Pinned behavior (suspected issue U8)."""
    sf, params, flow, *_ = _site_2312200()
    result = simulate(sf["Streamflow"].to_numpy(), params, flow)
    state = result.final_state()
    table = base_table_from_frame(pybfs.base_table(params.basin_length, params.base_profile_scale,
                                                   params.base_width, params.base_profile_exponent,
                                                   params.base_conductivity, sf, params.drainable_porosity))
    fc = forecast(10, params, state, table)
    assert fc.baseflow[0] == state.baseflow and fc.base_storage[0] == state.base_storage
    assert np.all(np.diff(fc.base_storage[1:]) <= 0)  # draining with no impulses


def test_all_missing_record_raises():
    _, params, flow, *_ = _site_2312200()
    with pytest.raises(ValueError, match="no valid"):
        simulate(np.full(10, np.nan), params, flow)


def test_q_sim_is_component_sum():
    sf, params, flow, *_ = _site_2312200()
    result = simulate(sf["Streamflow"].to_numpy(), params, flow)
    finite = ~np.isnan(result.q_sim)
    np.testing.assert_array_equal(result.q_sim[finite], result.flow_components[finite].sum(axis=1))
    assert np.isnan(result.q_sim[0])  # direct runoff undefined on the first step
