#!/usr/bin/env python3
"""Test if optimization produces same results with identical starting conditions"""

import numpy as np
import pandas as pd
from pybfs.calibrate import cal_initial, _create_scaled_minimize
from pybfs.utilities import flow_metrics

def test_optimization_path():
    """Test optimization path with identical starting conditions"""
    print("=" * 70)
    print("TEST: Optimization path comparison")
    print("=" * 70)
    
    # Load data
    data = pd.read_csv('bfs/calibration_R_original/12167000.csv', encoding='utf-8-sig')
    tmp_q = data['mean_daily_streamflow'].values
    dys = pd.to_datetime(data['Date'], format='%m/%d/%Y').values
    
    # Get flow metrics
    flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    flow = [flow_result[0], flow_result[1], flow_result[2], flow_result[3], flow_result[4], flow_result[5]]
    Qmean = np.nanmean(tmp_q[tmp_q >= 0])
    
    # R's exact initial parameters
    Lb = 7542.475350
    X1 = 100.0
    Wb = 1831.665908
    POR = 0.15
    ALPHA = 0.01
    BETA = 1.0
    Ks = 1494.574941
    Kb = 7164.319239  # Using R's value
    Kz = 3.172022
    
    basin_char = [671000000.0, Lb, X1, Wb, POR]
    gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]
    
    # Initial LOGX (matching R exactly)
    LOGX = np.log10([Lb, Wb, ALPHA, Ks, Kb, Kz])
    
    print(f"Initial LOGX: {LOGX}")
    print(f"parscale: {LOGX}")
    print()
    
    # Test objective function at initial point
    obj_initial = cal_initial(LOGX, tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow, Qmean)
    print(f"Initial objective: {obj_initial:.6f}")
    print()
    
    # Run optimization with same settings as R
    print("Running optimization (matching R: maxit=1000, parscale=LOGX, reltol=0.01)...")
    print()
    
    result = _create_scaled_minimize(
        cal_initial,
        LOGX,
        LOGX,  # parscale = LOGX
        tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow, Qmean,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'fatol': 1e-4}  # Match R: maxit=1000, reltol=0.01
    )
    
    print(f"Optimization success: {result.success}")
    print(f"Iterations: {result.nit}")
    print(f"Final objective: {result.fun:.6f}")
    print(f"Final params (log10): {result.x}")
    print()
    
    # Compare with R's results
    print("R's Step 1 cal_initial results:")
    print("  Iterations: 73")
    print("  Final objective: -266.066541")
    print("  Final params (log10): [4.024925, 3.290439, -2.024526, 3.363493, 3.557251, 0.529554]")
    print()
    
    print("Python's results:")
    print(f"  Iterations: {result.nit}")
    print(f"  Final objective: {result.fun:.6f}")
    print(f"  Final params (log10): {result.x}")
    print()
    
    # Calculate differences
    r_params = np.array([4.024925, 3.290439, -2.024526, 3.363493, 3.557251, 0.529554])
    py_params = result.x
    
    print("Parameter differences:")
    param_names = ['Lb', 'Wb', 'ALPHA', 'Ks', 'Kb', 'Kz']
    for i, name in enumerate(param_names):
        diff = py_params[i] - r_params[i]
        pct_diff = (diff / r_params[i]) * 100 if r_params[i] != 0 else 0
        print(f"  {name}: {diff:.6f} ({pct_diff:.2f}%)")
    print()
    
    obj_diff = result.fun - (-266.066541)
    obj_pct_diff = (obj_diff / abs(-266.066541)) * 100
    print(f"Objective difference: {obj_diff:.6f} ({obj_pct_diff:.2f}%)")
    print()
    
    print("Analysis:")
    print("  - If objective values match closely, the objective function is correct")
    print("  - If parameters differ but objective is similar, it's likely algorithm differences")
    print("  - If both differ significantly, there may be a bug")

if __name__ == "__main__":
    test_optimization_path()

