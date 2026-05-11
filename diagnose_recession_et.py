#!/usr/bin/env python3
"""Diagnose the late-2003/early-2004 recession mismatch for site 01134500.

Runs BFS with both the original and NSGA-II knee-point parameters, then plots
Qob vs Qsim, ETA (residual), StBase, and Rech zoomed into the recession window
to determine the sign and magnitude of the model error and whether an ET loss
term is a plausible explanation.
"""

import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from pybfs.calibrate_ea import _run_bfs
from pybfs.utilities import flow_metrics

SITE = '01134500'
DATA_PATH = f'RV_data/calibration/{SITE}_cal.csv'
BASIN_AREA = 195e6  # m²

# Zoom window covering the prolonged recession
RECESSION_START = '2003-10-01'
RECESSION_END   = '2004-04-01'


def load_streamflow():
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    tmp_q = df['discharge_m3d'].values
    dys = df['date'].values
    streamflow_df = pd.DataFrame({'Date': pd.to_datetime(dys), 'Streamflow': tmp_q})
    flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    flow = list(flow_result[:6])
    return streamflow_df, flow


def run_orig(streamflow_df, flow):
    row = pd.read_csv(f'RV_data/calibration/{SITE}_orig_params.csv').iloc[0]
    x = np.array([
        math.log10(row['Lb']),    math.log10(row['Wb']),
        math.log10(row['X1']),    math.log10(row['ALPHA']),
        row['BETA'],
        math.log10(row['Ks']),    math.log10(row['Kb']),
        math.log10(row['Kz']),    math.log10(row['POR']),
    ])
    result = _run_bfs(x, streamflow_df, flow, BASIN_AREA)
    if result is None:
        raise RuntimeError("BFS failed for original params")
    return result[0]


def run_knee(streamflow_df, flow):
    row = pd.read_csv(f'RV_data/calibration/{SITE}_knee.csv').iloc[0]
    x = np.array([
        math.log10(row['Lb']),    math.log10(row['Wb']),
        math.log10(row['X1']),    math.log10(row['ALPHA']),
        row['BETA'],
        math.log10(row['Ks']),    math.log10(row['Kb']),
        math.log10(row['Kz']),    math.log10(row['POR']),
    ])
    result = _run_bfs(x, streamflow_df, flow, BASIN_AREA)
    if result is None:
        raise RuntimeError("BFS failed for knee params")
    return result[0]


def zoom(df, start, end):
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    return df[mask].copy()


def plot_diagnostic(orig, knee, start, end):
    o = zoom(orig, start, end)
    k = zoom(knee, start, end)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle(f'Site {SITE} — Recession diagnostic ({start} to {end})', fontsize=12)

    # --- Panel 1: Qob vs Qsim (log) ---
    ax = axes[0]
    ax.plot(o['Date'], o['Qob'],   color='steelblue',  lw=1.2, label='Qobs')
    ax.plot(o['Date'], o['Qsim'],  color='tomato',     lw=1.0, ls='--', label='Qsim (orig)')
    ax.plot(k['Date'], k['Qsim'],  color='seagreen',   lw=1.0, ls=':',  label='Qsim (knee)')
    ax.plot(o['Date'], o['Baseflow'], color='darkorange', lw=0.8, label='Baseflow (orig)')
    ax.plot(k['Date'], k['Baseflow'], color='cyan', lw=0.8, ls=':', label='Baseflow (knee)')
    ax.set_yscale('log')
    ax.set_ylabel('Q (m³/day)')
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('Observed vs simulated total flow (log scale)')

    # --- Panel 2: ETA = Qobs - Qsim (signed residual) ---
    ax = axes[1]
    ax.axhline(0, color='black', lw=0.6, ls='--')
    ax.plot(o['Date'], o['Qob'] - o['Qsim'], color='tomato',   lw=1.0, label='Qobs−Qsim (orig)')
    ax.plot(k['Date'], k['Qob'] - k['Qsim'], color='seagreen', lw=1.0, label='Qobs−Qsim (knee)')
    ax.set_ylabel('Residual (m³/day)')
    ax.legend(fontsize=8)
    ax.set_title('Residual (positive = model underproduces, negative = model overproduces)')

    # --- Panel 3: Base storage ---
    ax = axes[2]
    ax.plot(o['Date'], o['StBase'], color='tomato',   lw=1.0, label='StBase (orig)')
    ax.plot(k['Date'], k['StBase'], color='seagreen', lw=1.0, label='StBase (knee)')
    ax.set_ylabel('Base storage (m³)')
    ax.legend(fontsize=8)
    ax.set_title('Base reservoir storage')

    # --- Panel 4: Recharge (surface → base) ---
    ax = axes[3]
    ax.plot(o['Date'], o['Rech'], color='tomato',   lw=1.0, label='Recharge (orig)')
    ax.plot(k['Date'], k['Rech'], color='seagreen', lw=1.0, label='Recharge (knee)')
    ax.set_ylabel('Recharge (m³/day)')
    ax.legend(fontsize=8)
    ax.set_title('Vertical recharge (surface → base)')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    out_path = f'RV_data/calibration/{SITE}_recession_diag.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


def print_summary(orig, knee, start, end):
    o = zoom(orig, start, end)
    k = zoom(knee, start, end)

    print(f"\n=== Recession window {start} – {end} ===")
    print(f"{'Metric':<35} {'Orig':>12} {'Knee':>12}")
    print("-" * 61)

    for label, col in [
        ("Mean Qobs (m³/day)",      'Qob'),
        ("Mean Qsim (m³/day)",      'Qsim'),
        ("Mean Baseflow (m³/day)",  'Baseflow'),
        ("Mean StBase (m³)",        'StBase'),
        ("Mean Rech (m³/day)",      'Rech'),
    ]:
        ov = o[col].mean()
        kv = k[col].mean()
        print(f"  {label:<33} {ov:>12.1f} {kv:>12.1f}")

    # Residual stats
    o_res = o['Qob'] - o['Qsim']
    k_res = k['Qob'] - k['Qsim']
    print(f"\n  {'Residual mean (Qob-Qsim)':<33} {o_res.mean():>12.1f} {k_res.mean():>12.1f}")
    print(f"  {'Residual: days positive (underproduces)':<33} {(o_res > 0).sum():>12d} {(k_res > 0).sum():>12d}")
    print(f"  {'Residual: days negative (overproduces)':<33} {(o_res < 0).sum():>12d} {(k_res < 0).sum():>12d}")
    print(f"  {'Total window days':<33} {len(o):>12d} {len(k):>12d}")


if __name__ == '__main__':
    print(f"Loading data for site {SITE} ...")
    streamflow_df, flow = load_streamflow()

    print("Running BFS with original params ...")
    orig = run_orig(streamflow_df, flow)

    print("Running BFS with NSGA-II knee params ...")
    knee = run_knee(streamflow_df, flow)

    print_summary(orig, knee, RECESSION_START, RECESSION_END)
    plot_diagnostic(orig, knee, RECESSION_START, RECESSION_END)
