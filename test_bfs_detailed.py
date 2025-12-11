#!/usr/bin/env python3
"""Compare BFS intermediate values to find where Python and R diverge"""

import pandas as pd
import numpy as np
import pybfs
from pybfs.calibrate import calculate_error

# Load R calibrated parameters
r_params = pd.read_csv('bfs/calibration_R_original/out/site_sum/bfs_params_12167000.csv')

# Load streamflow data
streamflow_data = pd.read_csv('bfs/calibration_R_original/12167000.csv', encoding='utf-8-sig')
streamflow_data['Date'] = pd.to_datetime(streamflow_data['Date'], format='%m/%d/%Y')

tmp_q = streamflow_data['mean_daily_streamflow'].values
dys = streamflow_data['Date'].values

# Get flow metrics
flow_result = pybfs.flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
flow = [flow_result[0], flow_result[1], flow_result[2], flow_result[3], flow_result[4], flow_result[5]]

# R parameters
tmp_area = r_params['tmp.area'].iloc[0]
Lb = r_params['Lb'].iloc[0]
X1 = r_params['X1'].iloc[0]
Wb = r_params['Wb'].iloc[0]
POR = r_params['POR'].iloc[0]
ALPHA = r_params['ALPHA'].iloc[0]
BETA = r_params['BETA'].iloc[0]
Ks = r_params['Ks'].iloc[0]
Kb = r_params['Kb'].iloc[0]
Kz = r_params['Kz'].iloc[0]

basin_char = [tmp_area, Lb, X1, Wb, POR]
gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]

# Create streamflow DataFrame
streamflow_df = pd.DataFrame({
    'Date': pd.to_datetime(dys),
    'Streamflow': tmp_q
})

# Generate baseflow table
SBT = pybfs.base_table(basin_char[1], basin_char[2], basin_char[3],
                       gw_hyd[1], gw_hyd[3], streamflow_df, basin_char[4])

# Run BFS
bfs_out = pybfs.bfs(streamflow_df, SBT, basin_char, gw_hyd, flow)

# Compare first 10 days of results
print("First 10 days comparison:")
print(f"{'Day':<5} {'Qob':<15} {'Qsim':<15} {'Baseflow':<15} {'SurfaceFlow':<15} {'DirectRunoff':<15}")
print("-" * 80)
for i in range(min(10, len(bfs_out))):
    qob = bfs_out['Qob'].iloc[i] if 'Qob' in bfs_out.columns else bfs_out['Qob.L3'].iloc[i]
    qsim = bfs_out['Qsim'].iloc[i] if 'Qsim' in bfs_out.columns else bfs_out['Qsim.L3'].iloc[i]
    bf = bfs_out['Baseflow'].iloc[i] if 'Baseflow' in bfs_out.columns else bfs_out['Baseflow.L3'].iloc[i]
    sf = bfs_out['SurfaceFlow'].iloc[i] if 'SurfaceFlow' in bfs_out.columns else bfs_out['SurfaceFlow.L3'].iloc[i]
    dr = bfs_out['DirectRunoff'].iloc[i] if 'DirectRunoff' in bfs_out.columns else bfs_out['DirectRunoff.L3'].iloc[i]
    print(f"{i+1:<5} {qob:<15.2f} {qsim:<15.2f} {bf:<15.2f} {sf:<15.2f} {dr:<15.2f}")

# Check for NaN values
print(f"\nNaN counts:")
print(f"  Qob: {bfs_out['Qob'].isna().sum() if 'Qob' in bfs_out.columns else bfs_out['Qob.L3'].isna().sum()}")
print(f"  Qsim: {bfs_out['Qsim'].isna().sum() if 'Qsim' in bfs_out.columns else bfs_out['Qsim.L3'].isna().sum()}")
print(f"  Baseflow: {bfs_out['Baseflow'].isna().sum() if 'Baseflow' in bfs_out.columns else bfs_out['Baseflow.L3'].isna().sum()}")

# Calculate summary stats
Error = calculate_error(bfs_out)
baseflow_col = 'Baseflow' if 'Baseflow' in bfs_out.columns else 'Baseflow.L3'
qob_col = 'Qob' if 'Qob' in bfs_out.columns else 'Qob.L3'
BFF = np.nansum(bfs_out[baseflow_col]) / np.nansum(bfs_out[qob_col])

print(f"\nSummary:")
print(f"  Error: {Error:.6f} (R: {r_params['Error'].iloc[0]:.6f}, diff: {abs(Error - r_params['Error'].iloc[0]):.6f})")
print(f"  BFF: {BFF:.6f} (R: {r_params['BFF'].iloc[0]:.6f}, diff: {abs(BFF - r_params['BFF'].iloc[0]):.6f})")

