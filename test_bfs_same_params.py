#!/usr/bin/env python3
"""Test if BFS produces same results with R's calibrated parameters"""

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

# Get flow metrics (should match R exactly now)
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

print("Testing BFS with R's calibrated parameters:")
print(f"  Lb: {Lb:.6f}, X1: {X1:.6f}, Wb: {Wb:.6f}")
print(f"  ALPHA: {ALPHA:.6f}, BETA: {BETA:.6f}")
print(f"  Ks: {Ks:.6f}, Kb: {Kb:.6f}, Kz: {Kz:.6f}")

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

# Calculate Error and BFF
Error = calculate_error(bfs_out)
baseflow_col = 'Baseflow' if 'Baseflow' in bfs_out.columns else 'Baseflow.L3'
qob_col = 'Qob' if 'Qob' in bfs_out.columns else 'Qob.L3'
BFF = np.nansum(bfs_out[baseflow_col]) / np.nansum(bfs_out[qob_col])

print(f"\nPython BFS results with R parameters:")
print(f"  Error: {Error:.6f} (R: {r_params['Error'].iloc[0]:.6f})")
print(f"  BFF: {BFF:.6f} (R: {r_params['BFF'].iloc[0]:.6f})")
print(f"  Difference Error: {abs(Error - r_params['Error'].iloc[0]):.6f}")
print(f"  Difference BFF: {abs(BFF - r_params['BFF'].iloc[0]):.6f}")

