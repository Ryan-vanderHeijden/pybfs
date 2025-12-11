#!/usr/bin/env python3
"""Compare Python and R BFS results side-by-side"""

import pandas as pd
import numpy as np

# Load both results
py_results = pd.read_csv('bfs/calibration_R_original/python_bfs_detailed.csv')
r_results = pd.read_csv('bfs/calibration_R_original/r_bfs_detailed.csv')

print("Comparison of Python vs R BFS results (first 20 time steps):")
print(f"{'ts':<5} {'Qob':<15} {'Qsim_Py':<15} {'Qsim_R':<15} {'Qsim_diff':<15} {'BF_Py':<15} {'BF_R':<15} {'BF_diff':<15}")
print("-" * 120)

for i in range(min(20, len(py_results), len(r_results))):
    qob_py = py_results['Qob'].iloc[i] if 'Qob' in py_results.columns else py_results['Qob.L3'].iloc[i]
    qsim_py = py_results['Qsim'].iloc[i] if 'Qsim' in py_results.columns else py_results['Qsim.L3'].iloc[i]
    qsim_r = r_results['Qsim.L3'].iloc[i] if not pd.isna(r_results['Qsim.L3'].iloc[i]) else np.nan
    
    bf_py = py_results['Baseflow'].iloc[i] if 'Baseflow' in py_results.columns else py_results['Baseflow.L3'].iloc[i]
    bf_r = r_results['Baseflow.L3'].iloc[i]
    
    qsim_diff = qsim_py - qsim_r if not pd.isna(qsim_r) else np.nan
    bf_diff = bf_py - bf_r
    
    qsim_py_str = f"{qsim_py:.2f}" if not pd.isna(qsim_py) else "NA"
    qsim_r_str = f"{qsim_r:.2f}" if not pd.isna(qsim_r) else "NA"
    qsim_diff_str = f"{qsim_diff:.2f}" if not pd.isna(qsim_diff) else "NA"
    
    print(f"{i+1:<5} {qob_py:<15.2f} {qsim_py_str:<15} {qsim_r_str:<15} {qsim_diff_str:<15} {bf_py:<15.2f} {bf_r:<15.2f} {bf_diff:<15.2f}")

# Calculate summary statistics
print("\n\nSummary Statistics:")
print(f"Python Qsim - mean: {py_results['Qsim'].mean():.2f}, sum: {py_results['Qsim'].sum():.2f}")
print(f"R Qsim - mean: {r_results['Qsim.L3'].mean():.2f}, sum: {r_results['Qsim.L3'].sum():.2f}")

print(f"\nPython Baseflow - mean: {py_results['Baseflow'].mean():.2f}, sum: {py_results['Baseflow'].sum():.2f}")
print(f"R Baseflow - mean: {r_results['Baseflow.L3'].mean():.2f}, sum: {r_results['Baseflow.L3'].sum():.2f}")

# Find where differences first appear
print("\n\nFirst significant differences (>1000):")
for i in range(min(20, len(py_results), len(r_results))):
    qsim_py = py_results['Qsim'].iloc[i] if 'Qsim' in py_results.columns else py_results['Qsim.L3'].iloc[i]
    qsim_r = r_results['Qsim.L3'].iloc[i]
    
    if not pd.isna(qsim_r):
        diff = abs(qsim_py - qsim_r)
        if diff > 1000:
            print(f"Time step {i+1}: Qsim difference = {diff:.2f}")

