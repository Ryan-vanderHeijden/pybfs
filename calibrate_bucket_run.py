#!/usr/bin/env python3
"""Run the two-bucket single-objective GA calibration on a site and plot it.

ks is fixed per site from the storm-flow recession coefficient, so the GA fits
the remaining 4 parameters {Ss_max, kr, Qref, n}. Writes the best solution to
RV_data/calibration/{site}_bucket_knee.csv and overlays the calibrated bucket
hydrograph and baseflow against the geometry model.

Usage:
    python calibrate_bucket_run.py [site] [pop] [gen]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pybfs
from pybfs.bfs_bucket import bfs_bucket
from pybfs.calibrate_bucket import bfs_calibrate_bucket_ga


def main():
    site = sys.argv[1] if len(sys.argv) > 1 else '01134500'
    pop = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    gen = int(sys.argv[3]) if len(sys.argv) > 3 else 150

    cal = pd.read_csv(f'RV_data/calibration/{site}_cal.csv')
    op = pd.read_csv(f'RV_data/calibration/{site}_orig_params.csv')
    area = float(op['tmp.area'][0])

    best, out = bfs_calibrate_bucket_ga(
        site, area, cal['discharge_m3d'].values, cal['date'].values,
        pop_size=pop, n_gen=gen, seed=1, n_jobs=10,
    )
    if best is None:
        print('calibration failed')
        return

    best.to_csv(f'RV_data/calibration/{site}_bucket_knee.csv', index=False)
    print('\n=== best (single-objective) solution ===')
    print(best[['Ss_max', 'ks', 'kr', 'k', 'n', 'Qref', 'tHalfSurf', 'tHalfBase',
                'RecessWeightedLogRMSE', 'BFF', 'Error']].to_string(index=False))

    # overlay calibrated bucket vs geometry model
    d = out['Date']
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(d, out['Qob'], color='0.4', lw=0.8, label='Qob')
    ax[0].plot(d, out['Qsim'], 'C0', lw=0.8, label='Qsim (bucket, calibrated)')
    ax[0].set_yscale('log'); ax[0].set_ylabel('Streamflow (m3/day)')
    ax[0].legend(loc='upper right', fontsize=8)
    ax[0].set_title(f'{site}: calibrated two-bucket (single-obj)  '
                    f'(BFF={best["BFF"][0]:.2f}, '
                    f'recWLogRMSE={best["RecessWeightedLogRMSE"][0]:.3f}, '
                    f'tHalf surf={best["tHalfSurf"][0]:.1f}d base={best["tHalfBase"][0]:.0f}d)')
    ax[1].plot(d, out['Qob'], color='0.4', lw=0.8, label='Qob')
    ax[1].plot(d, out['Baseflow'], 'C0', lw=0.9, label='Baseflow (bucket)')

    try:
        lb, x1, wb, por = op['Lb'][0], op['X1'][0], op['Wb'][0], op['POR'][0]
        alpha, beta, ks_o, kb, kz = op['ALPHA'][0], op['BETA'][0], op['Ks'][0], op['Kb'][0], op['Kz'][0]
        sf = pd.DataFrame({'Date': d, 'Streamflow': cal['discharge_m3d'].astype(float)})
        SBT = pybfs.base_table(lb, x1, wb, beta, kb, sf, por)
        ro = pybfs.bfs(sf, SBT, [area, lb, x1, wb, por], [alpha, beta, ks_o, kb, kz],
                       [op['Qthresh'][0], op['Rs'][0], op['Rb1'][0], op['Rb2'][0],
                        op['Prec'][0], op['Frac4Rise'][0]])
        ax[1].plot(d, ro['Baseflow'], 'C3', lw=0.7, alpha=0.7, label='Baseflow (geometry)')
    except Exception as e:
        print(f'(geometry overlay skipped: {e})')
    ax[1].set_yscale('log'); ax[1].set_ylabel('Baseflow (m3/day)')
    ax[1].legend(loc='upper right', fontsize=8)

    out_png = f'calibrate_bucket_{site}.png'
    fig.tight_layout(); fig.savefig(out_png, dpi=110)
    print(f'\nsaved {out_png}, {site}_bucket_knee.csv')


if __name__ == '__main__':
    main()
