#!/usr/bin/env python3
"""Interactive calibration explorer.

Loads saved Pareto front results and lets the user:
  - drag a BFF slider to select the Pareto solution nearest that BFF value
  - click any point in the Pareto scatter to select it directly

The selected solution is re-run through BFS and the hydrograph updates live.

Usage
-----
    python explore_calibration.py
    python explore_calibration.py --site 01234567 --basin-area 1.5e8
"""
import argparse
import math

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd

from pybfs.calibrate_ea import _run_bfs, _POR
from pybfs.utilities import flow_metrics

# ── defaults ──────────────────────────────────────────────────────────────────
SITE = "01134500"
BASIN_AREA = 195e6  # m²
DATA_PATH_TMPL = "RV_data/calibration/{site}_cal.csv"
PARETO_PATH_TMPL = "RV_data/calibration/{site}_pareto.csv"
KNEE_PATH_TMPL = "RV_data/calibration/{site}_knee.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_data(site, basin_area):
    cal_path = DATA_PATH_TMPL.format(site=site)
    pareto_path = PARETO_PATH_TMPL.format(site=site)
    knee_path = KNEE_PATH_TMPL.format(site=site)

    cal_df = pd.read_csv(cal_path, parse_dates=["date"])
    pareto_df = pd.read_csv(pareto_path)
    knee_df = pd.read_csv(knee_path)

    tmp_q = cal_df["discharge_m3d"].values
    dys = cal_df["date"].values
    streamflow_df = pd.DataFrame({"Date": pd.to_datetime(dys), "Streamflow": tmp_q})

    flow_result = flow_metrics(tmp_q, timestep="day", fr4rise=0.05)
    flow = list(flow_result[:6])

    return pareto_df, knee_df, streamflow_df, flow, basin_area


def _params_to_x(row):
    return np.array([
        math.log10(row["Lb"]),
        math.log10(row["Wb"]),
        math.log10(row["X1"]),
        math.log10(row["ALPHA"]),
        row["BETA"],
        math.log10(row["Ks"]),
        math.log10(row["Kb"]),
        math.log10(row["Kz"]),
    ])


def _run_for_row(row, streamflow_df, flow, basin_area):
    x = _params_to_x(row)
    result = _run_bfs(x, streamflow_df, flow, basin_area)
    if result is None:
        return None
    return result[0]  # bfs_out DataFrame


# ── app ───────────────────────────────────────────────────────────────────────

class CalibrationExplorer:
    def __init__(self, pareto_df, knee_df, streamflow_df, flow, basin_area, site):
        self.pareto_df = pareto_df
        self.streamflow_df = streamflow_df
        self.flow = flow
        self.basin_area = basin_area
        self.site = site

        # find knee index in pareto
        knee_bff = knee_df["BFF"].iloc[0]
        self.knee_idx = int(np.argmin(np.abs(pareto_df["BFF"].values - knee_bff)))
        self.selected_idx = self.knee_idx

        self._build_figure()
        self._initial_draw()

    # ── figure layout ─────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.suptitle(f"Calibration Explorer — Site {self.site}", fontsize=13)

        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[1, 0.07],
            hspace=0.45,
            wspace=0.35,
        )

        self.ax_pareto = self.fig.add_subplot(gs[0, 0])
        self.ax_ts = self.fig.add_subplot(gs[0, 1])
        ax_slider_area = self.fig.add_subplot(gs[1, :])

        self.ax_pareto.set_xlabel("KGE")
        self.ax_pareto.set_ylabel("Recession Error (MAE)")
        self.ax_pareto.set_title("Pareto front  (color = BFF)\nClick a point or drag the slider")

        self.ax_ts.set_ylabel("Discharge (m³/day)")
        self.ax_ts.set_yscale("log")

        bff_vals = self.pareto_df["BFF"].values
        self.slider = Slider(
            ax=ax_slider_area,
            label="Target BFF",
            valmin=float(bff_vals.min()),
            valmax=float(bff_vals.max()),
            valinit=float(bff_vals[self.selected_idx]),
            color="steelblue",
        )
        self.slider.on_changed(self._on_slider)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ── initial draw ──────────────────────────────────────────────────────────

    def _initial_draw(self):
        df = self.pareto_df
        norm = mcolors.Normalize(vmin=0, vmax=1)

        self.sc = self.ax_pareto.scatter(
            df["KGE"], df["RecessionError"],
            c=df["BFF"], cmap="viridis", s=20, alpha=0.7, norm=norm,
            picker=5,
        )
        plt.colorbar(self.sc, ax=self.ax_pareto, label="BFF")

        sel = df.iloc[self.selected_idx]
        self.sel_marker, = self.ax_pareto.plot(
            sel["KGE"], sel["RecessionError"],
            "r*", ms=14, zorder=10, label="selected",
        )
        self.ax_pareto.legend(loc="lower right", fontsize=8)

        bfs_out = _run_for_row(sel, self.streamflow_df, self.flow, self.basin_area)
        self._draw_ts(bfs_out, sel)

    # ── update helpers ────────────────────────────────────────────────────────

    def _select(self, idx):
        self.selected_idx = idx
        sel = self.pareto_df.iloc[idx]

        self.sel_marker.set_data([sel["KGE"]], [sel["RecessionError"]])

        bfs_out = _run_for_row(sel, self.streamflow_df, self.flow, self.basin_area)
        self._draw_ts(bfs_out, sel)
        self.fig.canvas.draw_idle()

    def _draw_ts(self, bfs_out, sel_row):
        ax = self.ax_ts
        ax.cla()
        ax.set_yscale("log")
        ax.set_ylabel("Discharge (m³/day)")

        bff = sel_row["BFF"]
        kge = sel_row["KGE"]
        re = sel_row["RecessionError"]
        ax.set_title(f"BFF={bff:.3f}  KGE={kge:.3f}  RecErr={re:.4f}", fontsize=9)

        if bfs_out is None:
            ax.text(0.5, 0.5, "BFS run failed", transform=ax.transAxes,
                    ha="center", va="center", color="red")
            return

        dates = bfs_out["Date"]
        ax.plot(dates, bfs_out["Qob"], label="Observed Q",
                color="steelblue", lw=0.8)
        ax.plot(dates, bfs_out["Qsim"], label="Simulated Q",
                color="seagreen", lw=0.8, linestyle="--")
        ax.plot(dates, bfs_out["Baseflow"], label="Baseflow",
                color="darkorange", lw=1.0)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_slider(self, val):
        bff_vals = self.pareto_df["BFF"].values
        nearest = int(np.argmin(np.abs(bff_vals - val)))
        if nearest != self.selected_idx:
            self._select(nearest)

    def _on_click(self, event):
        if event.inaxes is not self.ax_pareto:
            return
        if event.button != 1:
            return

        df = self.pareto_df
        kge_vals = df["KGE"].values
        re_vals = df["RecessionError"].values

        # convert axes coords to display coords for distance calculation
        ax = self.ax_pareto
        xy_disp = ax.transData.transform(np.column_stack([kge_vals, re_vals]))
        click_disp = ax.transData.transform([[event.xdata, event.ydata]])
        dists = np.linalg.norm(xy_disp - click_disp, axis=1)

        nearest = int(np.argmin(dists))
        if dists[nearest] > 20:  # pixels threshold
            return

        # sync slider without triggering redundant BFS run
        self.slider.eventson = False
        self.slider.set_val(df.iloc[nearest]["BFF"])
        self.slider.eventson = True

        self._select(nearest)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive calibration explorer")
    parser.add_argument("--site", default=SITE)
    parser.add_argument("--basin-area", type=float, default=BASIN_AREA)
    args = parser.parse_args()

    print(f"Loading data for site {args.site} …")
    pareto_df, knee_df, streamflow_df, flow, basin_area = _load_data(
        args.site, args.basin_area
    )
    print(f"  {len(pareto_df)} Pareto solutions loaded.")

    app = CalibrationExplorer(pareto_df, knee_df, streamflow_df, flow, basin_area, args.site)
    plt.show()


if __name__ == "__main__":
    main()
