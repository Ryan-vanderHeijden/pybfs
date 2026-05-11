#!/usr/bin/env python3
"""Test script for bfs_calibrate_de on site 01134500."""

import argparse
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pybfs.calibrate_de import bfs_calibrate_de
from pybfs.calibrate_ea import _run_bfs, _POR
from pybfs.utilities import flow_metrics

SITE = '04123500'
DATA_PATH = f'RV_data/calibration/{SITE}_cal.csv'

# # Basin area for 01134500 (m²) — Upper Ammonoosuc River, NH
# BASIN_AREA = 195e6  # m²
# POP_SIZE = 100
# GENERATIONS = 400

# Basin area for 04123500 (m²) — Manistee River Near Grayling, MI
# 123 sq miles × 2.58999 km²/sq mile = 318.57 km²
BASIN_AREA = 318.57e6  # m²
POP_SIZE = 200
GENERATIONS = 400


def main():
    print(f"Loading streamflow data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"  {len(df)} records ({df['date'].iloc[0].date()} – {df['date'].iloc[-1].date()})")

    tmp_q = df['discharge_m3d'].values
    dys = df['date'].values

    print(f"\nRunning DE calibration for site {SITE} ...")
    print(f"  Basin area: {BASIN_AREA / 1e6:.0f} km²")
    t0 = time.time()
    best_df, bfs_out = bfs_calibrate_de(
        tmp_site=SITE,
        tmp_area=BASIN_AREA,
        tmp_q=tmp_q,
        dys=dys,
        pop_size=POP_SIZE,
        n_gen=GENERATIONS,
        seed=3,
    )
    elapsed = time.time() - t0
    print(f"\nCalibration completed in {elapsed:.1f} s")

    if best_df is None:
        print("ERROR: calibration returned no results.")
        return

    print("\n--- Best solution ---")
    cols = ['Lb', 'Wb', 'X1', 'ALPHA', 'BETA', 'Ks', 'Kb', 'Kz',
            'BFSObjective', 'BFF', 'Error']
    print(best_df[cols].to_string(index=False))

    best_df.to_csv(f'RV_data/calibration/{SITE}_de_best.csv', index=False)
    print(f"\nBest params saved to RV_data/calibration/{SITE}_de_best.csv")

    if bfs_out is not None:
        _plot_results(bfs_out, SITE)


def _plot_results(bfs_out, site):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bfs_out['Date'], bfs_out['Qob'], label='Observed Q', color='steelblue', lw=0.8)
    ax.plot(bfs_out['Date'], bfs_out['Qsim'], label='Simulated Q', color='seagreen', lw=0.8, linestyle='--')
    ax.plot(bfs_out['Date'], bfs_out['Baseflow'], label='Baseflow', color='darkorange', lw=1.0)
    ax.set_ylabel('Discharge (m³/day)')
    ax.set_title(f'Site {site} — DE best solution')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    out_path = f'RV_data/calibration/{site}_de_plot.png'
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


def plot_only():
    best_df = pd.read_csv(f'RV_data/calibration/{SITE}_de_best.csv')
    print(f"Loaded best params for {SITE}")

    cal_df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    tmp_q = cal_df['discharge_m3d'].values
    dys = cal_df['date'].values
    streamflow_df = pd.DataFrame({'Date': pd.to_datetime(dys), 'Streamflow': tmp_q})

    flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    flow = list(flow_result[:6])

    row = best_df.iloc[0]
    x = np.array([
        math.log10(row['Lb']), math.log10(row['Wb']), math.log10(row['X1']),
        math.log10(row['ALPHA']), row['BETA'],
        math.log10(row['Ks']), math.log10(row['Kb']), math.log10(row['Kz']),
    ])
    result = _run_bfs(x, streamflow_df, flow, BASIN_AREA)
    if result is None:
        print("ERROR: BFS run failed for best params.")
        return
    bfs_out = result[0]
    _plot_results(bfs_out, SITE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Load saved CSV and re-plot without re-running calibration')
    args = parser.parse_args()
    if args.plot_only:
        plot_only()
    else:
        main()
