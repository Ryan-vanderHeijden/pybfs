#!/usr/bin/env python3
"""Compare the simplified two-bucket model (pybfs.bfs_bucket) against the
current geometry-based bfs() on a real site.

Usage:
    python compare_bucket.py            # site 01134500
    python compare_bucket.py 04123500

Reports NSE / log-NSE / baseflow fraction for the bucket model, checks
step-wise mass balance, and verifies the base bucket's zero-precip recession
matches the analytic k*S**n decay. Saves an overlay hydrograph PNG.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pybfs
from pybfs.bfs_bucket import bfs_bucket


def nse(obs, sim):
    m = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[m], sim[m]
    return 1 - np.sum((s - o) ** 2) / np.sum((o - np.mean(o)) ** 2)


def log_nse(obs, sim):
    m = ~(np.isnan(obs) | np.isnan(sim)) & (obs > 0) & (sim > 0)
    o, s = np.log(obs[m]), np.log(sim[m])
    return 1 - np.sum((s - o) ** 2) / np.sum((o - np.mean(o)) ** 2)


def bff(df):
    bf = np.minimum(df['Baseflow'], df['Qob'])
    return np.nansum(bf) / np.nansum(df['Qob'])


def main():
    site = sys.argv[1] if len(sys.argv) > 1 else '01134500'
    cal = pd.read_csv(f'RV_data/calibration/{site}_cal.csv')
    op = pd.read_csv(f'RV_data/calibration/{site}_orig_params.csv')
    area = float(op['tmp.area'][0])
    streamflow = pd.DataFrame({
        'Date': pd.to_datetime(cal['date']),
        'Streamflow': cal['discharge_m3d'].astype(float),
    })
    qin = streamflow['Streamflow'].values

    # Flow metrics (prec, rs, fr4rise) shared by both models.
    flow = pybfs.flow_metrics(qin, timestep='day', fr4rise=0.05)

    # --- bucket params: calibrated if available, else hand-picked ---
    cal_path = f'RV_data/calibration/{site}_bucket_knee.csv'
    if os.path.exists(cal_path):
        bp = pd.read_csv(cal_path)
        params = {c: float(bp[c][0]) for c in ('Ss_max', 'ks', 'kr', 'k', 'n')}
        source = 'calibrated'
    else:
        # Base k from a rough sustained-recession slope; n=1 (exponential) to start.
        qmean = np.nanmean(qin)
        params = {
            'Ss_max': 5.0 * qmean,     # surface capacity ~5 mean-flow-days
            'ks': 0.5,                 # surface drains fast
            'kr': 0.05,                # modest leakage to base
            'k': 0.02,                 # 1/day, ~35-day half-life
            'n': 1.0,
        }
        source = 'uncalibrated starting params'

    print(f'=== site {site}: {len(streamflow)} days ===')
    res = bfs_bucket(streamflow, params, flow, basin_area=area)

    # --- fit metrics ---
    print(f'\nBucket model ({source}):')
    print(f'  NSE      = {nse(res["Qob"].values, res["Qsim"].values):.3f}')
    print(f'  log-NSE  = {log_nse(res["Qob"].values, res["Qsim"].values):.3f}')
    print(f'  BFF      = {bff(res):.3f}')

    # --- verify: step-wise mass balance ---
    # dS_total = infil (in) - Qs - Qb ... but infil enters surface, Qd leaves as
    # runoff, precip W = infil + Qd. Check surface+base storage change equals
    # (infil) - Qs - Qb + (recharge cancels internally).
    dSs = res['StSur'].diff()
    dSb = res['StBase'].diff()
    # surface: dSs = infil - Qs - Rech ; base: dSb = Rech - Qb
    surf_resid = dSs - (res['Infil'] - res['SurfaceFlow'] - res['Rech'])
    base_resid = dSb - (res['Rech'] - res['Baseflow'])
    print('\nMass balance (max abs residual over series):')
    print(f'  surface bucket: {np.nanmax(np.abs(surf_resid)):.3e} m3')
    print(f'  base bucket:    {np.nanmax(np.abs(base_resid)):.3e} m3')

    # --- verify: analytic base recession on a no-precip stretch ---
    # Take the longest run of days with Impulse==0 and Rech small, compare
    # StBase decay to S0 / (1+(n-1)*k*S0^(n-1)*t) [n!=1] or S0*exp(-k t) [n=1].
    imp = res['Impulse.L'].fillna(0).values
    recess_mask = (imp == 0)
    # longest True run
    best_len, best_start = 0, 0
    run, start = 0, 0
    for i, v in enumerate(recess_mask):
        if v:
            if run == 0:
                start = i
            run += 1
            if run > best_len:
                best_len, best_start = run, start
        else:
            run = 0
    if best_len >= 10:
        seg = res.iloc[best_start:best_start + best_len]
        S0 = seg['StBase'].iloc[0]
        t = np.arange(best_len)
        k, nn = params['k'], params['n']
        if nn == 1.0:
            analytic = S0 * np.exp(-k * t)
        else:
            analytic = (S0 ** (1 - nn) + (nn - 1) * k * t) ** (1 / (1 - nn))
        rel_err = np.nanmax(np.abs(seg['StBase'].values - analytic) / (analytic + 1e-9))
        print(f'\nBase recession check (longest no-precip run = {best_len} d, '
              f'start {seg["Date"].iloc[0].date()}):')
        print(f'  max rel. deviation from analytic k*S^n decay = {rel_err:.3e}')
        # NOTE: recharge from surface still trickles in, so exact match only if
        # Rech ~ 0 over the window; residual reflects that leakage.
        print(f'  mean recharge over window = {seg["Rech"].mean():.1f} m3/day')

    # --- overlay hydrograph vs current geometry model ---
    lb, x1, wb, por = op['Lb'][0], op['X1'][0], op['Wb'][0], op['POR'][0]
    alpha, beta, ks_o, kb, kz = op['ALPHA'][0], op['BETA'][0], op['Ks'][0], op['Kb'][0], op['Kz'][0]
    basin_char = [area, lb, x1, wb, por]
    gw_hyd = [alpha, beta, ks_o, kb, kz]
    flow_o = [op['Qthresh'][0], op['Rs'][0], op['Rb1'][0], op['Rb2'][0],
              op['Prec'][0], op['Frac4Rise'][0]]
    try:
        SBT = pybfs.base_table(lb, x1, wb, beta, kb, streamflow, por)
        res_orig = pybfs.bfs(streamflow, SBT, basin_char, gw_hyd, flow_o)
        print('\nCurrent geometry model (orig calibrated params):')
        print(f'  NSE      = {nse(res_orig["Qob"].values, res_orig["Qsim"].values):.3f}')
        print(f'  log-NSE  = {log_nse(res_orig["Qob"].values, res_orig["Qsim"].values):.3f}')
        print(f'  BFF      = {bff(res_orig):.3f}')
    except Exception as e:
        res_orig = None
        print(f'\n(current model run failed: {e})')

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    d = res['Date']
    ax[0].plot(d, res['Qob'], color='0.4', lw=0.8, label='Qob')
    ax[0].plot(d, res['Qsim'], 'C0', lw=0.8, label='Qsim (bucket)')
    if res_orig is not None:
        ax[0].plot(d, res_orig['Qsim'], 'C3', lw=0.6, alpha=0.7, label='Qsim (geometry)')
    # ax[0].set_yscale('log')
    ax[0].set_ylabel('Streamflow (m3/day)')
    ax[0].legend(loc='upper right', fontsize=8)
    ax[0].set_title(f'{site}: two-bucket vs geometry model')

    ax[1].plot(d, res['Qob'], color='0.4', lw=0.8, label='Qob')
    ax[1].plot(d, res['Baseflow'], 'C0', lw=0.9, label='Baseflow (bucket)')
    if res_orig is not None:
        ax[1].plot(d, res_orig['Baseflow'], 'C3', lw=0.7, alpha=0.7, label='Baseflow (geometry)')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Baseflow (m3/day)')
    ax[1].legend(loc='upper right', fontsize=8)

    out = f'compare_bucket_{site}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
