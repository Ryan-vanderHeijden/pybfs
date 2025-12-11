#!/usr/bin/env python3
"""Test ini_params to see if it matches R"""

import pandas as pd
import numpy as np
import pybfs
import subprocess
import tempfile
import os

# Load streamflow data
streamflow_data = pd.read_csv('bfs/calibration_R_original/12167000.csv', encoding='utf-8-sig')
tmp_q = streamflow_data['mean_daily_streamflow'].values

# Get flow metrics
flow_result = pybfs.flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
Rb1 = flow_result[2]

# Initial parameters
tmp_area = 6.71e+08
Lb = 2 * (tmp_area / 2) ** 0.5
Wb = tmp_area / Lb / 10
X1 = 100.0  # 1/ALPHA where ALPHA=0.01
POR = 0.15
BETA = 1

print("Python ini_params:")
py_result = pybfs.ini_params(tmp_area, Lb, X1, Wb, POR, BETA, Rb1, tmp_q)
print(f"  Lb: {py_result['Lb'].iloc[0]:.6f}")
print(f"  Wb: {py_result['Wb'].iloc[0]:.6f}")
print(f"  Kb: {py_result['Kb'].iloc[0]:.6f}")

# Write streamflow data first
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    temp_q_file = f.name
    np.savetxt(temp_q_file, tmp_q, fmt='%.10e')

# Test with R
r_script = f"""
library(quantreg)
source('bfs/calibration_R_original/source/Rfunctions.bfs_utilities.R')
source('bfs/calibration_R_original/source/Rfunctions.bfs_calibration_sub.R')

tmp.q <- scan('{temp_q_file}', quiet=TRUE)
flow <- flow_metrics(tmp.q, timestep='day', fr4rise=0.05)
rb1 <- flow['Rb1']

tmp.area <- 6.71e+08
Lb <- 2*(tmp.area/2)^0.5
Wb <- tmp.area/Lb/10
X1 <- 100.0
POR <- 0.15
BETA <- 1

tmp <- ini_params(tmp.area, Lb, X1, Wb, POR, BETA, rb1, tmp.q)
cat('R ini_params:\\n')
cat('  Lb:', tmp$Lb, '\\n')
cat('  Wb:', tmp$Wb, '\\n')
cat('  Kb:', tmp$Kb, '\\n')
"""

try:
    # Write R script
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.R') as f:
        temp_r_file = f.name
        f.write(r_script)
    
    # Run R
    result = subprocess.run(
        ['Rscript', temp_r_file],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    
    print("\n" + result.stdout)
    if result.stderr:
        print("R errors:", result.stderr)
        
finally:
    if os.path.exists(temp_q_file):
        os.remove(temp_q_file)
    if os.path.exists(temp_r_file):
        os.remove(temp_r_file)

