"""Characterization cases that pin the current behavior of pybfs.

Each case is a function returning a flat ``dict[str, array-like]``. The same
functions generate the references (``tests/reference/generate.py``) and run the
checks (``tests/test_characterization.py``), so both sides execute identical
code. These cases describe what the code *does*, including known bugs (for
example M10, the inexact sur_z/sur_store inverse); they are not a statement of
what it should do.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pybfs
from pybfs import calibrate as cal
from pybfs import utilities as ut

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_2312200():
    """Streamflow and parameters for site 2312200, as in examples/main.py."""
    sf = pd.read_csv(REPO_ROOT / "docs/files/2312200_data.csv")
    sf["Date"] = pd.to_datetime(sf["Date"], format="%m/%d/%Y")
    params = pd.read_csv(REPO_ROOT / "docs/files/bfs_params_50.csv")
    basin_char, gw_hyd, flow = pybfs.get_values_for_site(params, 2312200)
    return sf, basin_char, gw_hyd, flow


def load_12167000(mask_zeros=False):
    """Streamflow for site 12167000, as in examples/main_calib.py.

    The record has four exact-zero days that are probably ice-affected missing
    values (suspected issue D1). ``mask_zeros=True`` sets them to NaN, as
    examples/main_skill.py does; the default keeps them, as main_calib.py does.
    """
    raw = pd.read_csv(REPO_ROOT / "bfs/12167000.csv", encoding="utf-8-sig")
    sf = pd.DataFrame({
        "Date": pd.to_datetime(raw["Date"], format="%m/%d/%Y"),
        "Streamflow": raw["mean_daily_streamflow"].astype(float),
    })
    if mask_zeros:
        sf.loc[sf["Streamflow"] == 0, "Streamflow"] = np.nan
    return sf


def params_12167000():
    """Committed Python-calibrated parameters (fixed input; see issue C16)."""
    params = pd.read_csv(REPO_ROOT / "examples/bfs_params_12167000_python.csv")
    params = params.rename(columns={"tmp.site": "site_no", "tmp.area": "AREA"})
    return pybfs.get_values_for_site(params, 12167000)


def sbt_for(sf, basin_char, gw_hyd):
    return pybfs.base_table(basin_char[1], basin_char[2], basin_char[3],
                            gw_hyd[1], gw_hyd[3], sf, basin_char[4])


# ---------------------------------------------------------------------------
# Flattening helpers
# ---------------------------------------------------------------------------

def _frame(prefix, df):
    out = {}
    for col in df.columns:
        values = df[col]
        if pd.api.types.is_datetime64_any_dtype(values):
            out[f"{prefix}.{col}"] = values.to_numpy().astype("datetime64[ns]").astype(np.int64)
        elif pd.api.types.is_bool_dtype(values):
            out[f"{prefix}.{col}"] = values.to_numpy().astype(bool)
        else:
            out[f"{prefix}.{col}"] = values.to_numpy(dtype=float)
    out[f"{prefix}.__columns__"] = np.array(list(df.columns))
    out[f"{prefix}.__index__"] = df.index.to_numpy()
    return out


def _numeric_list(values):
    return np.array([float(v) for v in values])


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_kernels():
    """Geometry kernels on a grid, plain and JIT versions, plus the sur_z round trip."""
    basin_char, gw_hyd, _ = params_12167000()
    lb, wb, por = basin_char[1], basin_char[3], basin_char[4]
    alpha, ks, kz = gw_hyd[0], gw_hyd[2], gw_hyd[4]
    ws = wb / 2
    zs = np.linspace(0.0, ws * alpha, 41)
    impulse = np.array([0.0, 1e-4, 1e-2, 0.5, ks * 2])
    xb = np.linspace(0.0, lb, 7)
    out = {}
    for suffix, mod in (("", ""), ("_jit", "_jit")):
        sur_store = getattr(ut, "sur_store" + mod)
        sur_z = getattr(ut, "sur_z" + mod)
        sur_q = getattr(ut, "sur_q" + mod)
        dir_q = getattr(ut, "dir_q" + mod)
        infiltration = getattr(ut, "infiltration" + mod)
        recharge = getattr(ut, "recharge" + mod)
        store = np.array([sur_store(lb, alpha, ws, por, z) for z in zs])
        out["sur_store" + suffix] = store
        out["sur_z_of_store" + suffix] = np.array([sur_z(lb, alpha, ws, por, s) for s in store])
        out["sur_z_large" + suffix] = np.array([sur_z(lb, alpha, ws, por, s) for s in store * 3])
        out["sur_q" + suffix] = np.array([sur_q(lb, alpha, ks, z) for z in zs])
        out["dir_q" + suffix] = np.array([[dir_q(lb, alpha, z, i) for i in impulse] for z in zs])
        out["infiltration" + suffix] = np.array(
            [[infiltration(lb, ws, ks, alpha, z, i) for i in impulse] for z in zs])
        out["recharge" + suffix] = np.array(
            [[recharge(lb, x, ws, kz, z, por) for x in xb] for z in zs])
    return out


def case_base_table():
    """base_table for both sites and both branches of the beta formula."""
    out = {}
    sf, bc, gw, _ = load_2312200()
    out.update(_frame("sbt_2312200", sbt_for(sf, bc, gw)))
    sf2 = load_12167000()
    bc2, gw2, _ = params_12167000()
    out.update(_frame("sbt_12167000", sbt_for(sf2, bc2, gw2)))
    for beta in (0.5, 2.5, 9.4):
        sbt = pybfs.base_table(bc2[1], bc2[2], bc2[3], beta, gw2[3], sf2, bc2[4])
        out.update(_frame(f"sbt_beta_{beta}", sbt))
    return out


def case_flow_metrics():
    out = {}
    inputs = {
        "12167000": load_12167000()["Streamflow"].to_numpy(),
        "12167000_masked": load_12167000(mask_zeros=True)["Streamflow"].to_numpy(),
        "2312200": load_2312200()[0]["Streamflow"].to_numpy(),
    }
    for name, q in inputs.items():
        fm = pybfs.flow_metrics(q, timestep="day", fr4rise=0.05)
        out[f"{name}.flow"] = _numeric_list(fm[:6])
        out[f"{name}.rb10"] = np.asarray(fm[6], dtype=float)
    return out


def case_bfs_2312200():
    """examples/main.py forward run, both loop implementations and error bases."""
    sf, bc, gw, fl = load_2312200()
    sbt = sbt_for(sf, bc, gw)
    out = {}
    out.update(_frame("jit_total", pybfs.bfs(sf, sbt, bc, gw, fl)))
    out.update(_frame("nojit_total", pybfs.bfs(sf, sbt, bc, gw, fl, use_jit=False)))
    out.update(_frame("jit_base", pybfs.bfs(sf, sbt, bc, gw, fl, error_basis="base")))
    return out


def case_bfs_edges_2312200():
    """Edited 2312200 inputs for paths the real records never reach.

    - ``rise_day2``: a rise on the second step, where the JIT loop (``ts > 0``) and
      the pure-Python loop (``ts > 1``) disagree (suspected issue M3). Both are pinned.
    - ``leading_nan``: the first three days missing, so the run starts at step 3.
    """
    sf, bc, gw, fl = load_2312200()
    sbt = sbt_for(sf, bc, gw)
    rise = sf.copy()
    rise.loc[1, "Streamflow"] = rise.loc[0, "Streamflow"] * 1.5
    lead = sf.copy()
    lead.loc[:2, "Streamflow"] = np.nan
    out = {}
    for label, data in (("rise_day2", rise), ("leading_nan", lead)):
        out.update(_frame(f"{label}.jit", pybfs.bfs(data, sbt, bc, gw, fl)))
        out.update(_frame(f"{label}.nojit", pybfs.bfs(data, sbt, bc, gw, fl, use_jit=False)))
    return out


def case_bfs_12167000():
    """Forward run with the committed calibrated parameters (zeros kept)."""
    sf = load_12167000()
    bc, gw, fl = params_12167000()
    sbt = sbt_for(sf, bc, gw)
    out = {}
    out.update(_frame("jit_total", pybfs.bfs(sf, sbt, bc, gw, fl)))
    base = pybfs.bfs(sf, sbt, bc, gw, fl, error_basis="base")
    out.update(_frame("jit_base", base))
    out.update(_frame("nojit_base", pybfs.bfs(sf, sbt, bc, gw, fl, error_basis="base", use_jit=False)))
    out["objective"] = np.array(cal.objective(base, prec=fl[4]))
    out["calculate_error"] = np.array(cal.calculate_error(base))
    ci_table, ci = pybfs.bf_ci(base)
    out.update(_frame("ci_table", ci_table))
    out.update(_frame("ci", ci))
    return out


def case_forecast_2312200():
    """examples/main.py forecast: train Jan-Sep 2018, forecast Oct-Nov 2018."""
    sf, bc, gw, fl = load_2312200()
    sbt = sbt_for(sf, bc, gw)
    train = sf[(sf["Date"] >= "2018-01-01") & (sf["Date"] <= "2018-09-30")]
    train_out = pybfs.bfs(train, sbt, bc, gw, fl)
    last = train_out.iloc[-1]
    initial = tuple(last[["X", "Zb.L", "Zs.L", "StBase", "StSur", "SurfaceFlow", "Baseflow", "Rech"]])
    dates = pd.date_range("2018-10-01", "2018-11-30", freq="D")
    fc = pybfs.forecast(pd.DataFrame({"date": dates, "streamflow": np.nan}), sbt, bc, gw, fl, initial)
    out = {"initial": np.array(initial, dtype=float)}
    out.update(_frame("forecast", fc))
    return out


def case_calibration_pieces():
    """Calibration building blocks at fixed inputs (fast; no optimizer)."""
    sf = load_12167000()
    q = sf["Streamflow"].to_numpy()
    area = 6.71e8
    fm = pybfs.flow_metrics(q, timestep="day", fr4rise=0.05)
    flow = list(fm[:6])
    rbi, rbs = fm[6]
    qmean = np.nanmean(q[q >= 0])
    # Same initial values as bfs_calibrate
    lb = 2 * (area / 2) ** 0.5
    wb = area / lb / 10
    por, alpha, beta = 0.15, 0.01, 1
    x1 = 1 / alpha
    ini = pybfs.ini_params(area, lb, x1, wb, por, beta, flow[2], q)
    out = {"ini_params": ini[["Lb", "Wb", "Kb"]].to_numpy()[0]}

    bc, gw, _ = params_12167000()
    bc, gw = [float(v) for v in bc], [float(v) for v in gw]
    logx6 = np.log10([bc[1], bc[3], gw[0], gw[2], gw[3], gw[4]])
    out["cal_initial"] = np.array([
        pybfs.cal_initial(logx6 + d, sf, "day", "base", bc, gw, flow, qmean)
        for d in (0.0, 0.05, -0.05)])
    out["cal_base"] = np.array([
        pybfs.cal_base(np.array([bc[1], bc[3], gw[3], gw[4]]) * f, sf, "day", "base", bc, gw, flow)
        for f in (1.0, 1.1, 0.9)])
    out["cal_surface"] = np.array([
        pybfs.cal_surface(np.log10([bc[1], bc[3], gw[0], gw[2]]), sf, "day", "base", bc, gw, flow)])
    out["cal_basetable"] = np.array([
        pybfs.cal_basetable(np.array([bc[1], bc[2], bc[3], gw[3]]), b,
                            [area, por, qmean, flow[0], rbi, rbs], q)
        for b in (0.6, 1.0, 2.5, 10.1)])
    return out


def _run_calibration(site, area, sf):
    """Run the calibration and also record every parameter set it simulates.

    The trajectory is one row per model run, ``[basin_char, gw_hyd]``. The
    outputs are the legacy ``bfs_calibrate`` tuple.
    """
    from pybfs.calibrate import legacy_outputs
    from pybfs.calibration import usgs

    calls = []

    def record(params):
        basin_char, gw_hyd = params.to_legacy()
        calls.append([float(v) for v in basin_char] + [float(v) for v in gw_hyd])

    result = usgs.calibrate(site, area, sf["Streamflow"].to_numpy(), sf["Date"].to_numpy(), on_simulate=record)
    bf_params, bff, ci_table, bfs_out = legacy_outputs(result)
    out = {"bf_params": bf_params.drop(columns=["tmp.site"]).to_numpy(dtype=float)[0],
           "bf_params.__columns__": np.array(list(bf_params.columns[1:])),
           "bff": bff.drop(columns=["tmp.site"]).to_numpy(dtype=float)[0],
           "trajectory": np.array(calls)}
    out.update(_frame("ci_table", ci_table))
    out.update(_frame("bfs_out", bfs_out))
    return out


def case_calibration_12167000():
    """Full bfs_calibrate, as in examples/main_calib.py (~26 s)."""
    return _run_calibration("12167000", 6.71e8, load_12167000())


def case_calibration_2312200():
    """Full bfs_calibrate on the second example site."""
    sf, bc, _, _ = load_2312200()
    return _run_calibration("2312200", float(bc[0]), sf)


def case_skill_12167000():
    """examples/main_skill.py: zeros masked, committed parameters."""
    sf = load_12167000(mask_zeros=True)
    bc, gw, fl = params_12167000()
    sbt = sbt_for(sf, bc, gw)
    out = {"strict_mask": pybfs.modified_strict_baseflow(sf, quantile=0.8)}
    sep_df, sep_metrics = pybfs.separation_skill(sf, sbt, bc, gw, fl, quantile=0.8)
    out.update(_frame("separation", sep_df))
    out["separation.metrics"] = _numeric_list(sep_metrics[k] for k in ("RMSE", "MAE", "n_days", "frac_strict"))
    fc_df, summary, fc_metrics = pybfs.forecast_skill(sf, sbt, bc, gw, fl)
    out.update(_frame("forecast", fc_df))
    out.update(_frame("forecast_summary", summary))
    out["forecast.metrics"] = _numeric_list([fc_metrics["overall_RMSE"], fc_metrics["overall_MAE"]])
    return out


FAST_CASES = {
    "kernels": case_kernels,
    "base_table": case_base_table,
    "flow_metrics": case_flow_metrics,
    "bfs_2312200": case_bfs_2312200,
    "bfs_edges_2312200": case_bfs_edges_2312200,
    "bfs_12167000": case_bfs_12167000,
    "forecast_2312200": case_forecast_2312200,
    "calibration_pieces": case_calibration_pieces,
    "skill_12167000": case_skill_12167000,
}

SLOW_CASES = {
    "calibration_12167000": case_calibration_12167000,
    "calibration_2312200": case_calibration_2312200,
}

ALL_CASES = {**FAST_CASES, **SLOW_CASES}
