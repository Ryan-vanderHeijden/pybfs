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


# ---------------------------------------------------------------------------
# Forward model: differential test against legacy bfs() (numba path)
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import zlib  # noqa: E402
import io  # noqa: E402

from pybfs.core.model import simulate  # noqa: E402
from pybfs.frames import legacy_bfs_frame  # noqa: E402
from pybfs.legacy.bfs import bfs as legacy_bfs  # noqa: E402
from pybfs.params import FlowMetrics, ModelParameters  # noqa: E402

N_PARAMETER_SETS = 100  # per record


def _records():
    sf_a, bc_a, gw_a, fl_a = cases.load_2312200()
    sf_b = cases.load_12167000()
    sf_c = cases.load_12167000(mask_zeros=True)
    _, _, fl_b = cases.params_12167000()
    rise = sf_a.copy()
    rise.loc[1, "Streamflow"] = rise.loc[0, "Streamflow"] * 1.5
    lead = sf_a.copy()
    lead.loc[:2, "Streamflow"] = np.nan
    return {"2312200": (sf_a, bc_a[0], fl_a), "12167000": (sf_b, 6.71e8, fl_b),
            "12167000_masked": (sf_c, 6.71e8, fl_b), "rise_day2": (rise, bc_a[0], fl_a),
            "leading_nan": (lead, bc_a[0], fl_a)}


def _random_parameters(rng, area):
    lb = 10 ** rng.uniform(2.5, 5)
    return [area, lb, 10 ** rng.uniform(0, 3), 10 ** rng.uniform(1, np.log10(area / lb) + 0.3),
            rng.uniform(0.05, 0.4)], [10 ** rng.uniform(-4, -1), rng.uniform(0.6, 10.1),
                                      10 ** rng.uniform(-2, 4), 10 ** rng.uniform(-2, 5), 10 ** rng.uniform(-4, 2)]


def _assert_frames_identical(new, old, context):
    assert list(new.columns) == list(old.columns), context
    np.testing.assert_array_equal(new.index.to_numpy(), old.index.to_numpy(), err_msg=context)
    for col in old.columns:
        a, b = new[col].to_numpy(), old[col].to_numpy()
        if col == "Date":
            np.testing.assert_array_equal(a, b, err_msg=context)
        else:
            np.testing.assert_array_equal(a.astype(float), b.astype(float), err_msg=f"{col}: {context}")


@pytest.mark.parametrize("record", list(_records()))
def test_simulate_matches_legacy_bfs(record):
    sf, area, fl = _records()[record]
    flow = FlowMetrics.from_legacy(fl)
    rng = np.random.default_rng(zlib.crc32(record.encode()))
    q = np.array(sf["Streamflow"])
    dates = pd.to_datetime(sf["Date"])
    for i in range(N_PARAMETER_SETS):
        basin_char, gw_hyd = _random_parameters(rng, area)
        error_basis = "base" if i % 2 else "total"
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sbt = legacy.base_table(basin_char[1], basin_char[2], basin_char[3], gw_hyd[1], gw_hyd[3], sf,
                                    basin_char[4])
            old = legacy_bfs(sf, sbt, basin_char, gw_hyd, fl, error_basis=error_basis)
            result = simulate(q, ModelParameters.from_legacy(basin_char, gw_hyd), flow)
        new = legacy_bfs_frame(result, dates, flow, error_basis)
        _assert_frames_identical(new, old, f"{record} set {i}: {basin_char} {gw_hyd}")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # beta = 0.5 overflows by design
def test_legacy_base_table_frame_index_type():
    """Same index *type* as legacy, not just values (RangeIndex keeps label lookups fast)."""
    from pybfs.utilities import base_table as compat_base_table
    sf = cases.load_12167000()
    for beta in (1.0, 0.5, 2.5):
        old = legacy.base_table(9072.0, 100.0, 2202.0, beta, 2850.0, sf, 0.15)
        new = compat_base_table(9072.0, 100.0, 2202.0, beta, 2850.0, sf, 0.15)
        assert type(new.index) is type(old.index)
        pd.testing.assert_frame_equal(new, old, check_exact=True)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

from pybfs.core.forecasting import forecast as core_forecast  # noqa: E402
from pybfs.frames import base_table_from_frame  # noqa: E402


def test_forecast_matches_legacy():
    """400 forecasts of 90 days from states of simulations with random parameters and lengths."""
    sf, bc0, _, fl = cases.load_2312200()
    flow = FlowMetrics.from_legacy(fl)
    rng = np.random.default_rng(3)
    dates = pd.date_range("2019-01-01", periods=90)
    for i in range(400):
        basin_char, gw_hyd = _random_parameters(rng, bc0[0])
        params = ModelParameters.from_legacy(basin_char, gw_hyd)
        sbt = legacy.base_table(basin_char[1], basin_char[2], basin_char[3], gw_hyd[1], gw_hyd[3], sf,
                                basin_char[4])
        table = base_table_from_frame(sbt)
        n_train = rng.integers(30, 365)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = simulate(np.array(sf["Streamflow"][:n_train]), params, flow, base_table=table).final_state()
        old = legacy.forecast(pd.DataFrame({"date": dates, "streamflow": np.nan}), sbt, basin_char, gw_hyd, fl,
                              state.to_legacy())
        new = core_forecast(90, params, state, table)
        for col, values in (("Baseflow", new.baseflow), ("StSur", new.surface_storage),
                            ("StBase", new.base_storage), ("Zs", new.surface_saturated_thickness),
                            ("Zb", new.base_saturated_thickness), ("Rech", new.recharge)):
            np.testing.assert_array_equal(values, old[col].to_numpy(), err_msg=f"{col}, set {i}")


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

from pybfs.frames import q_for_error  # noqa: E402
from pybfs.legacy import calibrate as legacy_calibrate  # noqa: E402
from pybfs.objectives import legacy as objectives  # noqa: E402


@pytest.mark.parametrize("error_basis", ["base", "total"])
def test_objectives_match_legacy(error_basis):
    sf, _, _, _ = cases.load_2312200()
    sf_b = cases.load_12167000()
    _, _, fl = cases.params_12167000()
    flow = FlowMetrics.from_legacy(fl)
    rng = np.random.default_rng(5)
    for i in range(100):
        record = sf if i % 2 else sf_b
        basin_char, gw_hyd = _random_parameters(rng, 6.71e8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = simulate(np.array(record["Streamflow"]), ModelParameters.from_legacy(basin_char, gw_hyd), flow)
        frame = legacy_bfs_frame(result, pd.to_datetime(record["Date"]), flow, error_basis)
        rel_error = objectives.relative_error(result.q_obs, q_for_error(result, error_basis), flow.flow_precision)
        weights = objectives.recession_weights(rel_error, result.recession_age, flow.storm_recession_rate)
        new = objectives.usgs_weighted_error(result.q_obs, result.baseflow, rel_error, weights,
                                             flow_precision=flow.flow_precision)
        assert new == legacy_calibrate.objective(frame, fl[4])
        mae = objectives.weighted_mean_abs_error(rel_error, weights)
        assert mae == legacy_calibrate.calculate_error(frame) or (np.isnan(mae) and np.isnan(
            legacy_calibrate.calculate_error(frame)))


# ---------------------------------------------------------------------------
# Calibration sub-objectives
# ---------------------------------------------------------------------------

import pybfs.calibrate as compat_calibrate  # noqa: E402


def test_calibration_subobjectives_match_legacy():
    """cal_initial/cal_base/cal_surface/cal_basetable at random points (feasible and not)."""
    sf = cases.load_12167000()
    q = sf["Streamflow"].to_numpy()
    fm = legacy.flow_metrics(q, timestep="day", fr4rise=0.05)
    flow, (rbi, rbs) = list(fm[:6]), fm[6]
    qmean = np.nanmean(q[q >= 0])
    bc, gw, _ = cases.params_12167000()
    bc, gw = [float(v) for v in bc], [float(v) for v in gw]
    rng = np.random.default_rng(9)
    logx6 = np.log10([bc[1], bc[3], gw[0], gw[2], gw[3], gw[4]])
    x_base = np.array([bc[1], bc[3], gw[3], gw[4]])
    x_table = np.array([bc[1], bc[2], bc[3], gw[3]])
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(40):
            basis = "total" if i % 4 == 0 else "base"
            lx = logx6 + rng.normal(0, 0.4, 6)
            assert (compat_calibrate.cal_initial(lx, sf, "day", basis, bc, gw, flow, qmean)
                    == legacy_calibrate.cal_initial(lx, sf, "day", basis, bc, gw, flow, qmean))
            xb = x_base * 10 ** rng.normal(0, 0.4, 4)
            assert (compat_calibrate.cal_base(xb, sf, "day", basis, bc, gw, flow)
                    == legacy_calibrate.cal_base(xb, sf, "day", basis, bc, gw, flow))
            ls = lx[:4] if i % 2 else np.array([lx[0], -1.5, -2.0, 1.0])  # second form passes the C1 check
            assert (compat_calibrate.cal_surface(ls, sf, "day", basis, bc, gw, flow)
                    == legacy_calibrate.cal_surface(ls, sf, "day", basis, bc, gw, flow))
            xt = x_table * 10 ** rng.normal(0, 0.6, 4)
            beta = rng.choice([0.6, 1.0, 2.5, 7.3, 10.1])
            params = [6.71e8, 0.15, qmean, flow[0], rbi, rbs]
            assert compat_calibrate.cal_basetable(xt, beta, params, q) == legacy_calibrate.cal_basetable(
                xt, beta, params, q)


def test_ini_params_matches_legacy():
    q = cases.load_12167000()["Streamflow"].to_numpy()
    rng = np.random.default_rng(13)
    for _ in range(30):
        area = 10 ** rng.uniform(7, 10)
        lb = 2 * (area / 2) ** 0.5
        args = (area, lb, 100.0, area / lb / 10, 0.15, 1, -10 ** rng.uniform(-3, -0.5), q * 10 ** rng.uniform(-2, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.testing.assert_frame_equal(compat_calibrate.ini_params(*args), legacy_calibrate.ini_params(*args),
                                          check_exact=True)
