#!/usr/bin/env python3
"""
Test script to compare R and Python flow_metrics functions

This script:
1. Loads the same streamflow data
2. Runs both R and Python flow_metrics functions
3. Compares outputs line by line
4. Identifies differences
"""

import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import pybfs

def run_r_flow_metrics(qin, timestep='day', fr4rise=0.05):
    """Run R flow_metrics function and return results"""
    
    # Write streamflow data to temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_q_file = f.name
        np.savetxt(temp_q_file, qin, fmt='%.10e')
    
    # Create R script
    r_script = f"""
library(quantreg)
source('bfs/calibration_R_original/source/Rfunctions.bfs_utilities.R')

# Read data from file
qin <- scan('{temp_q_file}', quiet=TRUE)

# Run flow_metrics
result <- flow_metrics(qin, timestep='{timestep}', fr4rise={fr4rise})

# Write results (one per line)
cat(result['Qthresh'], '\\n', sep='')
cat(result['Rs'], '\\n', sep='')
cat(result['Rb1'], '\\n', sep='')
cat(result['Rb2'], '\\n', sep='')
cat(result['Prec'], '\\n', sep='')
cat(result['Fr4Rise'], '\\n', sep='')
"""
    
    try:
        # Write R script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.R') as f:
            temp_r_file = f.name
            f.write(r_script)
        
        # Run R script
        result = subprocess.run(
            ['Rscript', temp_r_file],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            print(f"R script error:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Parse output - R outputs each value on a separate line
        lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        if len(lines) >= 6:
            return {
                'Qthresh': float(lines[0]),
                'Rs': float(lines[1]),
                'Rb1': float(lines[2]),
                'Rb2': float(lines[3]),
                'Prec': float(lines[4]),
                'Fr4Rise': float(lines[5])
            }
        else:
            print(f"Unexpected R output format:")
            print(f"Number of lines: {len(lines)}")
            print(f"Output: {result.stdout}")
            return None
            
    finally:
        # Clean up temporary files
        if os.path.exists(temp_q_file):
            os.remove(temp_q_file)
        if os.path.exists(temp_r_file):
            os.remove(temp_r_file)


def main():
    """Main test function"""
    
    print("="*60)
    print("FLOW_METRICS COMPARISON TEST")
    print("="*60)
    
    # Load streamflow data
    print("\nLoading streamflow data...")
    streamflow_data = pd.read_csv('bfs/calibration_R_original/12167000.csv', encoding='utf-8-sig')
    streamflow_data['Date'] = pd.to_datetime(streamflow_data['Date'], format='%m/%d/%Y')
    
    tmp_q = streamflow_data['mean_daily_streamflow'].values
    print(f"Loaded {len(tmp_q)} days of streamflow data")
    print(f"Date range: {streamflow_data['Date'].iloc[0]} to {streamflow_data['Date'].iloc[-1]}")
    print(f"Mean streamflow: {np.nanmean(tmp_q):.2f} m³/day")
    
    # Run Python flow_metrics
    print("\n" + "="*60)
    print("Running Python flow_metrics...")
    print("="*60)
    try:
        py_result = pybfs.flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
        print(f"Python results:")
        print(f"  Qthresh: {py_result[0]:.6f}")
        print(f"  Rs:      {py_result[1]:.6f}")
        print(f"  Rb1:     {py_result[2]:.6f}")
        print(f"  Rb2:     {py_result[3]:.6f}")
        print(f"  Prec:    {py_result[4]:.6f}")
        print(f"  Fr4Rise: {py_result[5]:.6f}")
    except Exception as e:
        print(f"ERROR in Python flow_metrics: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run R flow_metrics
    print("\n" + "="*60)
    print("Running R flow_metrics...")
    print("="*60)
    try:
        r_result = run_r_flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
        if r_result:
            print(f"R results:")
            print(f"  Qthresh: {r_result['Qthresh']:.6f}")
            print(f"  Rs:      {r_result['Rs']:.6f}")
            print(f"  Rb1:     {r_result['Rb1']:.6f}")
            print(f"  Rb2:     {r_result['Rb2']:.6f}")
            print(f"  Prec:    {r_result['Prec']:.6f}")
            print(f"  Fr4Rise: {r_result['Fr4Rise']:.6f}")
        else:
            print("ERROR: R flow_metrics returned None")
            return
    except Exception as e:
        print(f"ERROR in R flow_metrics: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON: PYTHON vs R")
    print("="*60)
    
    param_names = ['Qthresh', 'Rs', 'Rb1', 'Rb2', 'Prec', 'Fr4Rise']
    
    print(f"\n{'Parameter':<15} {'R Value':<20} {'Python Value':<20} {'Difference':<20} {'% Diff':<15}")
    print("-" * 90)
    
    max_diff = 0
    max_param = None
    
    for param in param_names:
        r_val = r_result[param]
        py_val = py_result[param_names.index(param)]
        diff = py_val - r_val
        pct_diff = (diff / abs(r_val) * 100) if r_val != 0 else 0
        
        if abs(pct_diff) > abs(max_diff):
            max_diff = pct_diff
            max_param = param
        
        print(f"{param:<15} {r_val:<20.10f} {py_val:<20.10f} {diff:<20.10f} {pct_diff:<15.2f}%")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Maximum relative difference: {max_diff:.2f}% ({max_param})")
    
    if abs(max_diff) < 0.1:
        print("✓ Python and R results are very close!")
    elif abs(max_diff) < 1:
        print("⚠ Python and R results are reasonably close")
    elif abs(max_diff) < 10:
        print("⚠ Python and R results show some differences")
    else:
        print("✗ Python and R results differ significantly")
        print("  This indicates a potential bug in the Python implementation")


if __name__ == "__main__":
    main()

