"""Parameter containers and parameter-table reading."""

import numpy as np
import pandas as pd
import pytest

from pybfs.data import read_parameter_table, site_parameters
from pybfs.legacy.utilities import get_values_for_site
from pybfs.params import PARAMETERS, FlowMetrics, ModelParameters

from cases import REPO_ROOT

TABLE = REPO_ROOT / "docs/files/bfs_params_50.csv"


def test_legacy_round_trip_all_sites():
    df = pd.read_csv(TABLE)
    table = read_parameter_table(TABLE)
    assert len(table) == len(df) == 50
    for site in df["site_no"]:
        basin_char, gw_hyd, flow = get_values_for_site(df, site)
        params, metrics = site_parameters(table, site)
        assert params.to_legacy() == (basin_char, gw_hyd)
        assert metrics.to_legacy() == flow
        assert ModelParameters.from_legacy(basin_char, gw_hyd) == params
        assert FlowMetrics.from_legacy(flow) == metrics


def test_calibration_output_format_is_accepted():
    table = read_parameter_table(REPO_ROOT / "examples/bfs_params_12167000_python.csv")
    params, _ = site_parameters(table, 12167000)
    assert params.drainage_area == 6.71e8


def test_unknown_site_raises():
    table = read_parameter_table(TABLE)
    with pytest.raises(KeyError, match="not in parameter table"):
        site_parameters(table, "2312200")  # str, but keys are int


def test_vector_round_trip():
    params, _ = site_parameters(read_parameter_table(TABLE), 2312200)
    names = ["basin_length", "base_width", "surface_gradient"]
    x = params.to_vector(names, "log10")
    np.testing.assert_allclose(x, np.log10([params.basin_length, params.base_width, params.surface_gradient]))
    back = params.with_vector(names, x, "log10")
    for n in names:
        assert getattr(back, n) == pytest.approx(getattr(params, n), rel=1e-14)
    assert back.drainable_porosity == params.drainable_porosity


def test_registry_covers_every_field():
    assert set(PARAMETERS) == set(ModelParameters.__dataclass_fields__)
    assert params_symbols() == ["AREA", "Lb", "X1", "Wb", "POR", "ALPHA", "BETA", "Ks", "Kb", "Kz"]


def params_symbols():
    return [PARAMETERS[f].symbol for f in ModelParameters.__dataclass_fields__]


def test_surface_half_width():
    params, _ = site_parameters(read_parameter_table(TABLE), 2312200)
    assert params.surface_half_width == params.base_width / 2


def test_get_values_for_site_matches_legacy():
    from pybfs.utilities import get_values_for_site as compat
    df = pd.read_csv(TABLE)
    for site in list(df["site_no"]) + [123, "2312200"]:
        assert compat(df, site) == get_values_for_site(df, site)
    assert compat(df, 123) == (None, None, None)
