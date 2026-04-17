#!/usr/bin/env python3
"""Test script for bfs_calibrate_nsga2 on site 01134500."""

import time
import pandas as pd
import matplotlib.pyplot as plt

from pybfs.calibrate_ea import bfs_calibrate_nsga2

SITE = '01134500'
DATA_PATH = f'RV_data/calibration/{SITE}_cal.csv'

# Basin area for 01134500 (m²) — Upper Ammonoosuc River, NH
# Drainage area ~404 km²
BASIN_AREA = 195e6  # m²


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
        pop_size=100,
        n_gen=100,
        seed=3,
    )
    elapsed = time.time() - t0
    print(f"\nCalibration completed in {elapsed:.1f} s")

    if pareto_df is None:
        print("ERROR: calibration returned no results.")
        return

    print(f"\n--- Pareto front ({len(pareto_df)} solutions) ---")
    print(pareto_df[['KGE', 'RecessionError']].describe().to_string())

    print("\n--- Knee-point solution ---")
    if knee_df is not None:
        cols = ['Lb', 'Wb', 'X1', 'ALPHA', 'BETA', 'Ks', 'Kb', 'Kz',
                'KGE', 'RecessionError', 'Error', 'BFF']
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
        _plot_results(bfs_out, pareto_df, SITE)


def _plot_results(bfs_out, pareto_df, site):
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
    ax.scatter(pareto_df['KGE'], pareto_df['RecessionError'],
               s=20, alpha=0.7, color='steelblue')
    ax.set_xlabel('KGE')
    ax.set_ylabel('Recession Error (MAE)')
    ax.set_title('Pareto front')

    plt.tight_layout()
    out_path = f'RV_data/calibration/{site}_cal_plot.png'
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
