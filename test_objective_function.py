#!/usr/bin/env python3
"""Test if Python's objective function returns the same values as R's for identical parameters"""

import numpy as np
import pandas as pd
from pybfs.calibrate import cal_initial, cal_base, cal_surface, cal_basetable, objective
from pybfs.utilities import flow_metrics, base_table
from pybfs.bfs import bfs

def test_cal_initial_objective():
    """Test cal_initial with R's exact initial parameters"""
    print("=" * 70)
    print("TEST 1: cal_initial objective function")
    print("=" * 70)
    
    # Load data
    data = pd.read_csv('bfs/calibration_R_original/12167000.csv', encoding='utf-8-sig')
    tmp_q = data['mean_daily_streamflow'].values
    dys = pd.to_datetime(data['Date'], format='%m/%d/%Y').values
    
    # Get flow metrics
    flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    flow = [flow_result[0], flow_result[1], flow_result[2], flow_result[3], flow_result[4], flow_result[5]]
    Qmean = np.nanmean(tmp_q[tmp_q >= 0])
    
    # R's exact initial parameters (from R diagnostic output)
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
    
    # Test with initial LOGX (matching R exactly)
    LOGX = np.log10([Lb, Wb, ALPHA, Ks, Kb, Kz])
    print(f"Input LOGX: {LOGX}")
    print()
    
    # Call cal_initial
    obj = cal_initial(LOGX, tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow, Qmean)
    print(f"Python objective value: {obj:.6f}")
    print()
    print("Expected: R would return a similar value for these exact parameters")
    print("If this differs significantly from R's initial Error (0.877000), there's a bug")
    print()
    
    return obj

def test_cal_initial_with_r_optimized_params():
    """Test cal_initial with R's optimized parameters from Step 1"""
    print("=" * 70)
    print("TEST 2: cal_initial with R's Step 1 optimized parameters")
    print("=" * 70)
    
    # Load data
    data = pd.read_csv('bfs/calibration_R_original/12167000.csv', encoding='utf-8-sig')
    tmp_q = data['mean_daily_streamflow'].values
    dys = pd.to_datetime(data['Date'], format='%m/%d/%Y').values
    
    # Get flow metrics
    flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    flow = [flow_result[0], flow_result[1], flow_result[2], flow_result[3], flow_result[4], flow_result[5]]
    Qmean = np.nanmean(tmp_q[tmp_q >= 0])
    
    # R's Step 1 optimized parameters (from R diagnostic output)
    Lb = 10590.707791
    Wb = 1951.818548
    ALPHA = 0.009451
    Ks = 2309.369646
    Kb = 3607.874589
    Kz = 3.384965
    X1 = 100.0
    POR = 0.15
    BETA = 1.0
    
    basin_char = [671000000.0, Lb, X1, Wb, POR]
    gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]
    
    # Test with R's optimized LOGX
    LOGX = np.log10([Lb, Wb, ALPHA, Ks, Kb, Kz])
    print(f"Input LOGX (R's optimized): {LOGX}")
    print()
    
    # Call cal_initial
    obj = cal_initial(LOGX, tmp_q, dys, 'day', 'base', basin_char, gw_hyd, flow, Qmean)
    print(f"Python objective value: {obj:.6f}")
    print()
    print("Expected: This should match R's objective value for these parameters")
    print("R's Step 1 cal_initial final objective: -266.066541")
    print()
    
    return obj

def test_objective_function_directly():
    """Test the objective function directly with known bfs_out"""
    print("=" * 70)
    print("TEST 3: objective() function directly")
    print("=" * 70)
    
    # Load data
    data = pd.read_csv('bfs/calibration_R_original/12167000.csv', encoding='utf-8-sig')
    tmp_q = data['mean_daily_streamflow'].values
    dys = pd.to_datetime(data['Date'], format='%m/%d/%Y').values
    
    # Get flow metrics
    flow_result = flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
    flow = [flow_result[0], flow_result[1], flow_result[2], flow_result[3], flow_result[4], flow_result[5]]
    
    # R's initial parameters
    Lb = 7542.475350
    X1 = 100.0
    Wb = 1831.665908
    POR = 0.15
    ALPHA = 0.01
    BETA = 1.0
    Ks = 1494.574941
    Kb = 7164.319239
    Kz = 3.172022
    
    basin_char = [671000000.0, Lb, X1, Wb, POR]
    gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]
    
    # Create streamflow DataFrame
    streamflow_df = pd.DataFrame({
        'Date': pd.to_datetime(dys),
        'Streamflow': tmp_q
    })
    
    # Generate baseflow table
    SBT = base_table(basin_char[1], basin_char[2], basin_char[3],
                     gw_hyd[1], gw_hyd[3], streamflow_df, basin_char[4])
    
    # Run BFS
    bfs_out = bfs(streamflow_df, SBT, basin_char, gw_hyd, flow, timestep='day', error_basis='base')
    
    # Calculate objective
    prec = flow[4]  # Prec
    obj = objective(bfs_out, prec)
    
    print(f"Objective value: {obj:.6f}")
    print()
    print("This should match what R's objective() would return for the same bfs_out")
    print()
    
    return obj

def test_parscale_equivalence():
    """Test if our parscale implementation is equivalent to R's"""
    print("=" * 70)
    print("TEST 4: parscale implementation equivalence")
    print("=" * 70)
    
    # Simulate what R does with parscale
    # R: optim(LOGX, func, ..., control=list(parscale=LOGX))
    # R internally divides parameters by parscale before optimization
    
    LOGX = np.array([3.8775139, 3.26284626, -2.0, 3.1745177, 3.85517493, 0.50133619])
    parscale = LOGX.copy()
    
    print("R's behavior with parscale:")
    print(f"  Initial LOGX: {LOGX}")
    print(f"  parscale: {parscale}")
    print("  R internally: scaled_x = LOGX / parscale")
    print()
    
    # What R does internally
    scaled_x_r = LOGX / parscale
    print(f"  R's scaled_x: {scaled_x_r}")
    print()
    
    # What Python does
    print("Python's behavior with parscale:")
    scaled_x0_py = LOGX / parscale
    print(f"  Python's scaled_x0: {scaled_x0_py}")
    print()
    
    # When objective is called, R multiplies back
    print("When objective function is called:")
    print("  R: unscaled_x = scaled_x * parscale")
    unscaled_x_r = scaled_x_r * parscale
    print(f"  R's unscaled_x: {unscaled_x_r}")
    print()
    
    print("  Python: unscaled_x = scaled_x * parscale")
    unscaled_x_py = scaled_x0_py * parscale
    print(f"  Python's unscaled_x: {unscaled_x_py}")
    print()
    
    if np.allclose(unscaled_x_r, unscaled_x_py):
        print("✓ parscale implementation appears equivalent")
    else:
        print("✗ parscale implementation differs!")
    
    print()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING OBJECTIVE FUNCTION EQUIVALENCE")
    print("=" * 70 + "\n")
    
    # Test 1: Initial parameters
    test_cal_initial_objective()
    print()
    
    # Test 2: R's optimized parameters
    test_cal_initial_with_r_optimized_params()
    print()
    
    # Test 3: Direct objective function
    test_objective_function_directly()
    print()
    
    # Test 4: parscale equivalence
    test_parscale_equivalence()
    print()
    
    print("=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Compare Python objective values with R's values")
    print("2. If they differ, investigate the source of the difference")
    print("3. Verify parscale implementation matches R exactly")

