#!/usr/bin/env python3
"""Test script for original bfs_calibrate on site 01134500 (baseline)."""

import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pybfs.calibrate import bfs_calibrate

SITE = '01134500'
DATA_PATH = f'RV_data/calibration/{SITE}_cal.csv'

# Basin area for 01134500 (m²) — Upper Ammonoosuc River, NH
BASIN_AREA = 195e6  # m²

# # Basin area for 04123500 (m²) — Manistee River Near Grayling, MI
# # 123 sq miles × 2.58999 km²/sq mile = 318.57 km²
# BASIN_AREA = 318.57e6  # m²


def main():
    print(f"Loading streamflow data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"  {len(df)} records ({df['date'].iloc[0].date()} – {df['date'].iloc[-1].date()})")

    tmp_q = df['discharge_m3d'].values
    dys = df['date'].values

    print(f"\nRunning original bfs_calibrate for site {SITE} ...")
    print(f"  Basin area: {BASIN_AREA / 1e6:.0f} km²")
    t0 = time.time()
    bf_params, bff, ci_table, bfs_out = bfs_calibrate(
        tmp_site=SITE,
        tmp_area=BASIN_AREA,
        tmp_q=tmp_q,
        dys=dys,
    )
    elapsed = time.time() - t0
    print(f"\nCalibration completed in {elapsed:.1f} s")

    if bf_params is None:
        print("ERROR: calibration returned no results.")
        return

    print("\n--- Calibrated parameters ---")
    cols = ['Lb', 'X1', 'Wb', 'ALPHA', 'BETA', 'Ks', 'Kb', 'Kz', 'Error', 'BFF']
    print(bf_params[cols].to_string(index=False))

    if bff is not None:
        print("\n--- Flow fractions ---")
        print(bff.to_string(index=False))

    out_path = f'RV_data/calibration/{SITE}_orig_params.csv'
    bf_params.to_csv(out_path, index=False)
    print(f"\nParameters saved to {out_path}")

    if bfs_out is not None:
        _plot_results(bfs_out, SITE)


def _plot_results(bfs_out, site):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bfs_out['Date'], bfs_out['Qob'], label='Observed Q', color='steelblue', lw=0.8)
    ax.plot(bfs_out['Date'], bfs_out['Qsim'], label='Simulated Q', color='seagreen', lw=0.8, linestyle='--')
    ax.plot(bfs_out['Date'], bfs_out['Baseflow'], label='Baseflow', color='darkorange', lw=1.0)
    ax.set_ylabel('Discharge (m³/day)')
    ax.set_title(f'Site {site} — original bfs_calibrate result')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    out_path = f'RV_data/calibration/{site}_orig_plot.png'
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
