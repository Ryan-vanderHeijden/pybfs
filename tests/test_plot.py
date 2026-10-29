"""Plot functions: return a Figure, write files only when asked."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pybfs  # noqa: E402

import cases  # noqa: E402


def _outputs():
    sf, bc, gw, fl = cases.load_2312200()
    sbt = cases.sbt_for(sf, bc, gw)
    result = pybfs.bfs(sf, sbt, bc, gw, fl)
    initial = tuple(result.iloc[-1][["X", "Zb.L", "Zs.L", "StBase", "StSur", "SurfaceFlow", "Baseflow", "Rech"]])
    dates = pd.date_range("2019-01-01", periods=30)
    fc = pybfs.forecast(pd.DataFrame({"date": dates, "streamflow": np.nan}), sbt, bc, gw, fl, initial)
    return sf, result, fc


def test_plots_return_figures_and_write_nothing_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sf, result, fc = _outputs()
    figures = [
        pybfs.plot_baseflow_simulation(sf, result),
        pybfs.plot_forecast(sf, result, fc),
        pybfs.plot_forecast_baseflow(fc),
        pybfs.plot_forecast_baseflow_streamflow(fc, sf.iloc[:30]),
    ]
    assert all(isinstance(f, matplotlib.figure.Figure) for f in figures)
    assert list(tmp_path.iterdir()) == []
    plt.close("all")


def test_save_true_and_save_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sf, result, _ = _outputs()
    pybfs.plot_baseflow_simulation(sf, result, save=True, dpi=50)
    pybfs.plot_baseflow_simulation(sf, result, save=tmp_path / "custom.png", dpi=50)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["baseflow_simulation.png", "custom.png"]
    plt.close("all")
