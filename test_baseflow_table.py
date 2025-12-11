#!/usr/bin/env python3
"""Compare baseflow table generation between R and Python"""

import pandas as pd
import numpy as np
import pybfs
import subprocess
import tempfile
import os

# Load streamflow data
streamflow_data = pd.read_csv('bfs/calibration_R_original/12167000.csv', encoding='utf-8-sig')
tmp_q = streamflow_data['mean_daily_streamflow'].values

# Parameters from R calibration
tmp_area = 6.71e+08
Lb = 9807.36
X1 = 100.0
Wb = 2012.41
POR = 0.15
BETA = 1.0
Kb = 2984.69

print("Parameters:")
print(f"  Lb: {Lb}")
print(f"  X1: {X1}")
print(f"  Wb: {Wb}")
print(f"  POR: {POR}")
print(f"  BETA: {BETA}")
print(f"  Kb: {Kb}")
print()

# Generate Python baseflow table
streamflow_df = pd.DataFrame({
    'Date': pd.to_datetime(streamflow_data['Date'], format='%m/%d/%Y'),
    'Streamflow': tmp_q
})

py_sbt = pybfs.base_table(Lb, X1, Wb, BETA, Kb, streamflow_df, POR)

print("Python baseflow table:")
print(f"  Number of rows: {len(py_sbt)}")
print(f"  Columns: {list(py_sbt.columns)}")
print(f"  First 10 rows:")
print(py_sbt.head(10).to_string())
print()
print(f"  Last 10 rows:")
print(py_sbt.tail(10).to_string())
print()
print(f"  Statistics:")
print(f"    Xb - min: {py_sbt['Xb'].min():.6f}, max: {py_sbt['Xb'].max():.6f}, mean: {py_sbt['Xb'].mean():.6f}")
print(f"    Z - min: {py_sbt['Z'].min():.6f}, max: {py_sbt['Z'].max():.6f}, mean: {py_sbt['Z'].mean():.6f}")
print(f"    S - min: {py_sbt['S'].min():.6f}, max: {py_sbt['S'].max():.6f}, mean: {py_sbt['S'].mean():.6f}")
print(f"    Q - min: {py_sbt['Q'].min():.6f}, max: {py_sbt['Q'].max():.6f}, mean: {py_sbt['Q'].mean():.6f}")
print()

# Write streamflow data first
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    temp_q_file = f.name
    np.savetxt(temp_q_file, tmp_q, fmt='%.10e')

# Test with R
r_script = f"""
library(quantreg)
source('bfs/calibration_R_original/source/Rfunctions.bfs_utilities.R')

tmp.q <- scan('{temp_q_file}', quiet=TRUE)
tmp.area <- 6.71e+08
Lb <- 9807.36
X1 <- 100.0
Wb <- 2012.41
POR <- 0.15
BETA <- 1.0
Kb <- 2984.69

cat('R baseflow table:\\n')
SBT <- base_table(Lb, X1, Wb, BETA, Kb, tmp.q, POR)
cat('  Number of rows:', nrow(SBT), '\\n')
cat('  Columns:', paste(colnames(SBT), collapse=', '), '\\n')
cat('\\n  First 10 rows:\\n')
print(head(SBT, 10))
cat('\\n  Last 10 rows:\\n')
print(tail(SBT, 10))
cat('\\n  Statistics:\\n')
cat('    Xb - min:', min(SBT$Xb), ', max:', max(SBT$Xb), ', mean:', mean(SBT$Xb), '\\n')
cat('    Z - min:', min(SBT$Z), ', max:', max(SBT$Z), ', mean:', mean(SBT$Z), '\\n')
cat('    S - min:', min(SBT$S), ', max:', max(SBT$S), ', mean:', mean(SBT$S), '\\n')
cat('    Q - min:', min(SBT$Q), ', max:', max(SBT$Q), ', mean:', mean(SBT$Q), '\\n')

# Save to CSV for detailed comparison
write.csv(SBT, 'r_baseflow_table.csv', row.names=FALSE)
cat('\\nSaved R baseflow table to r_baseflow_table.csv\\n')
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
    
    print(result.stdout)
    if result.stderr:
        print("R errors:", result.stderr)
        
finally:
    if os.path.exists(temp_q_file):
        os.remove(temp_q_file)
    if os.path.exists(temp_r_file):
        os.remove(temp_r_file)

# Load and compare R results if available
if os.path.exists('bfs/calibration_R_original/r_baseflow_table.csv'):
    r_sbt = pd.read_csv('bfs/calibration_R_original/r_baseflow_table.csv')
    
    print("\n" + "="*80)
    print("DETAILED COMPARISON:")
    print("="*80)
    
    print(f"\nRow count - Python: {len(py_sbt)}, R: {len(r_sbt)}")
    
    min_len = min(len(py_sbt), len(r_sbt))
    
    # Compare first 20 rows
    print("\nFirst 20 rows comparison:")
    print(f"{'Row':<5} {'Xb_Py':<15} {'Xb_R':<15} {'Xb_diff':<15} {'Z_Py':<15} {'Z_R':<15} {'Z_diff':<15}")
    print("-" * 90)
    for i in range(min(20, min_len)):
        xb_py = py_sbt['Xb'].iloc[i]
        xb_r = r_sbt['Xb'].iloc[i]
        z_py = py_sbt['Z'].iloc[i]
        z_r = r_sbt['Z'].iloc[i]
        xb_diff = xb_py - xb_r
        z_diff = z_py - z_r
        print(f"{i+1:<5} {xb_py:<15.6f} {xb_r:<15.6f} {xb_diff:<15.6e} {z_py:<15.6f} {z_r:<15.6f} {z_diff:<15.6e}")
    
    # Compare S and Q
    print("\n\nS and Q comparison (first 20 rows):")
    print(f"{'Row':<5} {'S_Py':<15} {'S_R':<15} {'S_diff':<15} {'Q_Py':<15} {'Q_R':<15} {'Q_diff':<15}")
    print("-" * 90)
    for i in range(min(20, min_len)):
        s_py = py_sbt['S'].iloc[i]
        s_r = r_sbt['S'].iloc[i]
        q_py = py_sbt['Q'].iloc[i]
        q_r = r_sbt['Q'].iloc[i]
        s_diff = s_py - s_r
        q_diff = q_py - q_r
        print(f"{i+1:<5} {s_py:<15.6f} {s_r:<15.6f} {s_diff:<15.6e} {q_py:<15.6f} {q_r:<15.6f} {q_diff:<15.6e}")
    
    # Find maximum differences
    print("\n\nMaximum differences:")
    xb_diffs = np.abs(py_sbt['Xb'].iloc[:min_len].values - r_sbt['Xb'].iloc[:min_len].values)
    z_diffs = np.abs(py_sbt['Z'].iloc[:min_len].values - r_sbt['Z'].iloc[:min_len].values)
    s_diffs = np.abs(py_sbt['S'].iloc[:min_len].values - r_sbt['S'].iloc[:min_len].values)
    q_diffs = np.abs(py_sbt['Q'].iloc[:min_len].values - r_sbt['Q'].iloc[:min_len].values)
    
    print(f"  Xb - max diff: {xb_diffs.max():.6e} at row {xb_diffs.argmax()+1}")
    print(f"  Z - max diff: {z_diffs.max():.6e} at row {z_diffs.argmax()+1}")
    print(f"  S - max diff: {s_diffs.max():.6e} at row {s_diffs.argmax()+1}")
    print(f"  Q - max diff: {q_diffs.max():.6e} at row {q_diffs.argmax()+1}")
    
    # Save Python table for comparison
    py_sbt.to_csv('bfs/calibration_R_original/python_baseflow_table.csv', index=False)
    print("\nSaved Python baseflow table to bfs/calibration_R_original/python_baseflow_table.csv")

