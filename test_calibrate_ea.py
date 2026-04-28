#!/usr/bin/env python3
"""Test script for bfs_calibrate_nsga2 on site 01134500."""

import argparse
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pybfs.calibrate_ea import bfs_calibrate_nsga2, _run_bfs, _POR
from pybfs.utilities import flow_metrics

SITE = '01134500'
DATA_PATH = f'RV_data/calibration/{SITE}_cal.csv'

# Basin area for 01134500 (m²) — Upper Ammonoosuc River, NH
# Drainage area ~404 km²
BASIN_AREA = 195e6  # m²
POP_SIZE = 200
GENERATIONS = 400

def main():
    print(f"Loading streamflow data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"  {len(df)} records ({df['date'].iloc[0].date()} – {df['date'].iloc[-1].date()})")

    tmp_q = df['discharge_m3d'].values
    dys = df['date'].values

    print(f"\nRunning NSGA-II calibration for site {SITE} ...")
    print(f"  Basin area: {BASIN_AREA / 1e6:.0f} km²")
    t0 = time.time()
    pareto_df, knee_df, bfs_out = bfs_calibrate_nsga2(
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

    if pareto_df is None:
        print("ERROR: calibration returned no results.")
        return

    print(f"\n--- Pareto front ({len(pareto_df)} solutions) ---")
    print(pareto_df[['RecessionRMSE', 'RecessionError']].describe().to_string())

    print("\n--- Knee-point solution ---")
    if knee_df is not None:
        cols = ['Lb', 'Wb', 'X1', 'ALPHA', 'BETA', 'Ks', 'Kb', 'Kz',
                'RecessionRMSE', 'RecessionError', 'Error', 'BFF']
        print(knee_df[cols].to_string(index=False))
    else:
        print("  (no knee solution returned)")

    # Save outputs
    pareto_df.to_csv(f'RV_data/calibration/{SITE}_pareto.csv', index=False)
    print(f"\nPareto front saved to RV_data/calibration/{SITE}_pareto.csv")

    if knee_df is not None:
        knee_df.to_csv(f'RV_data/calibration/{SITE}_knee.csv', index=False)
        print(f"Knee params saved to  RV_data/calibration/{SITE}_knee.csv")

    if bfs_out is not None:
        _plot_results(bfs_out, pareto_df, SITE, knee_df=knee_df)


def _plot_results(bfs_out, pareto_df, site, knee_df=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(bfs_out['Date'], bfs_out['Qob'], label='Observed Q', color='steelblue', lw=0.8)
    ax.plot(bfs_out['Date'], bfs_out['Qsim'], label='Simulated Q', color='seagreen', lw=0.8, linestyle='--')
    ax.plot(bfs_out['Date'], bfs_out['Baseflow'], label='Baseflow', color='darkorange', lw=1.0)
    ax.set_ylabel('Discharge (m³/day)')
    ax.set_title(f'Site {site} — Knee-point BFS result')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[1]
    sc = ax.scatter(pareto_df['RecessionRMSE'], pareto_df['RecessionError'],
                    s=20, alpha=0.8, color='steelblue')
    if knee_df is not None:
        ax.scatter(knee_df['RecessionRMSE'], knee_df['RecessionError'],
                   marker='*', s=200, color='red', zorder=5, label='Knee point')
        ax.legend(loc='upper right')
    ax.set_xlabel('Recession Log-RMSE (F[0])')
    ax.set_ylabel('Recession Rate MAE (F[1])')
    ax.set_title('Pareto front')

    x_margin = (pareto_df['RecessionRMSE'].max() - pareto_df['RecessionRMSE'].min()) * 0.1
    ax.set_xlim(pareto_df['RecessionRMSE'].min() - x_margin, pareto_df['RecessionRMSE'].max() + x_margin)
    re_margin = (pareto_df['RecessionError'].max() - pareto_df['RecessionError'].min()) * 0.1
    ax.set_ylim(pareto_df['RecessionError'].min() - re_margin, pareto_df['RecessionError'].max() + re_margin)

    plt.tight_layout()
    out_path = f'RV_data/calibration/{site}_cal_plot.png'
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


def plot_only():
    pareto_df = pd.read_csv(f'RV_data/calibration/{SITE}_pareto.csv')
    knee_df = pd.read_csv(f'RV_data/calibration/{SITE}_knee.csv')
    print(f"Loaded {len(pareto_df)} Pareto solutions and knee params for {SITE}")

    # Re-run BFS once with knee params to get the hydrograph
    cal_df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    tmp_q = cal_df['discharge_m3d'].values
    dys = cal_df['date'].values
    streamflow_df = pd.DataFrame({'Date': pd.to_datetime(dys), 'Streamflow': tmp_q})

    flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    flow = list(flow_result[:6])

    row = knee_df.iloc[0]
    x = np.array([
        math.log10(row['Lb']), math.log10(row['Wb']), math.log10(row['X1']),
        math.log10(row['ALPHA']), row['BETA'],
        math.log10(row['Ks']), math.log10(row['Kb']), math.log10(row['Kz']),
    ])
    result = _run_bfs(x, streamflow_df, flow, BASIN_AREA)
    if result is None:
        print("ERROR: BFS run failed for knee params.")
        return
    bfs_out = result[0]
    _plot_results(bfs_out, pareto_df, SITE, knee_df=knee_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Load saved CSVs and re-plot without re-running calibration')
    args = parser.parse_args()
    if args.plot_only:
        plot_only()
    else:
        main()
