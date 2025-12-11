#!/usr/bin/env python3
"""
Calibration test script for PyBFS

This script tests the calibration process by:
1. Loading streamflow data for gage 2312200
2. Running the calibration process
3. Comparing results with pre-calibrated parameters
"""

import pandas as pd
import numpy as np
import pybfs


def main():
    """Main execution function for calibration testing"""

    # Site information
    site_id = "2312200"
    site_area = 371200000  # m² (from params file)

    # Load streamflow data
    print("Loading streamflow data...")
    streamflow_data = pd.read_csv('docs/files/2312200_data.csv')
    streamflow_data['Date'] = pd.to_datetime(streamflow_data['Date'])
    
    # Extract streamflow and dates
    tmp_q = streamflow_data['Streamflow'].values
    dys = streamflow_data['Date'].values

    print(f"Loaded {len(tmp_q)} days of streamflow data")
    print(f"Date range: {dys[0]} to {dys[-1]}")
    print(f"Mean streamflow: {np.nanmean(tmp_q):.2f} m³/day")

    # Load reference parameters for comparison
    print("\nLoading reference parameters...")
    bfs_params_ref = pd.read_csv('docs/files/bfs_params_50.csv')
    ref_params = bfs_params_ref[bfs_params_ref['site_no'] == int(site_id)].iloc[0]

    print(f"\nReference parameters for site {site_id}:")
    print(f"  AREA: {ref_params['AREA']}")
    print(f"  Lb: {ref_params['Lb']:.6f}")
    print(f"  X1: {ref_params['X1']:.6f}")
    print(f"  Wb: {ref_params['Wb']:.6f}")
    print(f"  POR: {ref_params['POR']:.6f}")
    print(f"  ALPHA: {ref_params['ALPHA']:.6f}")
    print(f"  BETA: {ref_params['BETA']:.6f}")
    print(f"  Ks: {ref_params['Ks']:.6f}")
    print(f"  Kb: {ref_params['Kb']:.6f}")
    print(f"  Kz: {ref_params['Kz']:.6f}")
    print(f"  Qthresh: {ref_params['Qthresh']:.6f}")
    print(f"  Rs: {ref_params['Rs']:.6f}")
    print(f"  Rb1: {ref_params['Rb1']:.6f}")
    print(f"  Rb2: {ref_params['Rb2']:.6f}")
    print(f"  Prec: {ref_params['Prec']:.6f}")
    print(f"  Frac4Rise: {ref_params['Frac4Rise']:.6f}")
    print(f"  Error: {ref_params['Error']:.6f}")
    print(f"  BFF: {ref_params['BFF']:.6f}")

    # Run calibration
    print("\n" + "="*60)
    print("Running calibration...")
    print("="*60)
    print("This may take several minutes...")

    try:
        bf_params, bff, ci_table, bfs_out = pybfs.bfs_calibrate(
            tmp_site=site_id,
            tmp_area=site_area,
            tmp_q=tmp_q,
            dys=dys
        )

        if bf_params is None:
            print("\nERROR: Calibration failed - flow_metrics returned invalid values")
            return

        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)

        # Display calibrated parameters
        print("\nCalibrated parameters:")
        print(f"  AREA: {bf_params['tmp.area'].iloc[0]}")
        print(f"  Lb: {bf_params['Lb'].iloc[0]:.6f}")
        print(f"  X1: {bf_params['X1'].iloc[0]:.6f}")
        print(f"  Wb: {bf_params['Wb'].iloc[0]:.6f}")
        print(f"  POR: {bf_params['POR'].iloc[0]:.6f}")
        print(f"  ALPHA: {bf_params['ALPHA'].iloc[0]:.6f}")
        print(f"  BETA: {bf_params['BETA'].iloc[0]:.6f}")
        print(f"  Ks: {bf_params['Ks'].iloc[0]:.6f}")
        print(f"  Kb: {bf_params['Kb'].iloc[0]:.6f}")
        print(f"  Kz: {bf_params['Kz'].iloc[0]:.6f}")
        print(f"  Qthresh: {bf_params['Qthresh'].iloc[0]:.6f}")
        print(f"  Rs: {bf_params['Rs'].iloc[0]:.6f}")
        print(f"  Rb1: {bf_params['Rb1'].iloc[0]:.6f}")
        print(f"  Rb2: {bf_params['Rb2'].iloc[0]:.6f}")
        print(f"  Prec: {bf_params['Prec'].iloc[0]:.6f}")
        print(f"  Frac4Rise: {bf_params['Frac4Rise'].iloc[0]:.6f}")
        print(f"  Error: {bf_params['Error'].iloc[0]:.6f}")
        print(f"  BFF: {bf_params['BFF'].iloc[0]:.6f}")

        # Display flow fractions
        print("\nFlow fractions:")
        print(f"  Qmean: {bff['Qmean'].iloc[0]:.6f}")
        print(f"  BFF (Baseflow Fraction): {bff['BFF'].iloc[0]:.6f}")
        print(f"  SFF (Surface Flow Fraction): {bff['SFF'].iloc[0]:.6f}")
        print(f"  DRF (Direct Runoff Fraction): {bff['DRF'].iloc[0]:.6f}")
        print(f"  Error: {bff['Error'].iloc[0]:.6f}")

        # Compare with reference
        print("\n" + "="*60)
        print("COMPARISON WITH REFERENCE PARAMETERS")
        print("="*60)

        param_names = ['Lb', 'X1', 'Wb', 'POR', 'ALPHA', 'BETA', 'Ks', 'Kb', 'Kz',
                      'Qthresh', 'Rs', 'Rb1', 'Rb2', 'Prec', 'Frac4Rise', 'Error', 'BFF']

        print(f"\n{'Parameter':<15} {'Reference':<15} {'Calibrated':<15} {'Difference':<15} {'% Diff':<15}")
        print("-" * 75)

        for param in param_names:
            ref_val = ref_params[param]
            cal_val = bf_params[param].iloc[0]
            diff = cal_val - ref_val
            pct_diff = (diff / ref_val * 100) if ref_val != 0 else 0
            print(f"{param:<15} {ref_val:<15.6f} {cal_val:<15.6f} {diff:<15.6f} {pct_diff:<15.2f}%")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Calculate relative differences for key parameters
        key_params = ['Lb', 'X1', 'Wb', 'ALPHA', 'BETA', 'Ks', 'Kb', 'Kz']
        max_diff = 0
        max_param = None
        
        for param in key_params:
            ref_val = ref_params[param]
            cal_val = bf_params[param].iloc[0]
            if ref_val != 0:
                pct_diff = abs((cal_val - ref_val) / ref_val * 100)
                if pct_diff > max_diff:
                    max_diff = pct_diff
                    max_param = param

        print(f"\nMaximum relative difference in key parameters: {max_diff:.2f}% ({max_param})")
        
        if max_diff < 10:
            print("✓ Calibration results are very close to reference parameters!")
        elif max_diff < 25:
            print("⚠ Calibration results are reasonably close to reference parameters")
        else:
            print("✗ Calibration results differ significantly from reference parameters")
            print("  This may be due to:")
            print("  - Different optimization convergence")
            print("  - Different random initialization")
            print("  - Numerical precision differences between R and Python")

    except Exception as e:
        print(f"\nERROR during calibration: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare BFS results with calibrated vs reference parameters
    print("\n" + "="*60)
    print("COMPARING BFS RESULTS: CALIBRATED vs REFERENCE PARAMETERS")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        
        # Prepare streamflow DataFrame
        streamflow_df = pd.DataFrame({
            'Date': pd.to_datetime(dys),
            'Streamflow': tmp_q
        })
        
        # Get flow metrics (already calculated during calibration)
        flow_metrics_result = pybfs.flow_metrics(tmp_q, timestep='day', fr4rise=0.05)
        flow = [
            flow_metrics_result[0],  # Qthresh
            flow_metrics_result[1],  # Rs
            flow_metrics_result[2],  # Rb1
            flow_metrics_result[3],  # Rb2
            flow_metrics_result[4],  # Prec
            flow_metrics_result[5]   # Fr4Rise
        ]
        
        # CALIBRATED PARAMETERS
        print("\nRunning BFS with calibrated parameters...")
        cal_basin_char = [
            site_area,
            bf_params['Lb'].iloc[0],
            bf_params['X1'].iloc[0],
            bf_params['Wb'].iloc[0],
            bf_params['POR'].iloc[0]
        ]
        cal_gw_hyd = [
            bf_params['ALPHA'].iloc[0],
            bf_params['BETA'].iloc[0],
            bf_params['Ks'].iloc[0],
            bf_params['Kb'].iloc[0],
            bf_params['Kz'].iloc[0]
        ]
        cal_SBT = pybfs.base_table(
            cal_basin_char[1], cal_basin_char[2], cal_basin_char[3],
            cal_gw_hyd[1], cal_gw_hyd[3], streamflow_df, cal_basin_char[4]
        )
        cal_bfs_out = pybfs.bfs(streamflow_df, cal_SBT, cal_basin_char, cal_gw_hyd, flow)
        
        # REFERENCE PARAMETERS
        print("Running BFS with reference parameters...")
        ref_basin_char = [
            site_area,
            ref_params['Lb'],
            ref_params['X1'],
            ref_params['Wb'],
            ref_params['POR']
        ]
        ref_gw_hyd = [
            ref_params['ALPHA'],
            ref_params['BETA'],
            ref_params['Ks'],
            ref_params['Kb'],
            ref_params['Kz']
        ]
        ref_SBT = pybfs.base_table(
            ref_basin_char[1], ref_basin_char[2], ref_basin_char[3],
            ref_gw_hyd[1], ref_gw_hyd[3], streamflow_df, ref_basin_char[4]
        )
        ref_bfs_out = pybfs.bfs(streamflow_df, ref_SBT, ref_basin_char, ref_gw_hyd, flow)
        
        # Create comparison plot
        print("Creating comparison plot...")
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        # Determine column names (handle both formats)
        qob_col = 'Qob' if 'Qob' in cal_bfs_out.columns else 'Qob.L3'
        baseflow_col = 'Baseflow' if 'Baseflow' in cal_bfs_out.columns else 'Baseflow.L3'
        surface_col = 'SurfaceFlow' if 'SurfaceFlow' in cal_bfs_out.columns else 'SurfaceFlow.L3'
        direct_col = 'DirectRunoff' if 'DirectRunoff' in cal_bfs_out.columns else 'DirectRunoff.L3'
        
        dates = cal_bfs_out['Date'].values
        
        # Plot 1: Observed Streamflow
        axes[0].plot(dates, cal_bfs_out[qob_col].values, 'k-', linewidth=1.5, label='Observed', alpha=0.7)
        axes[0].set_ylabel('Streamflow\n(m³/day)', fontsize=10)
        axes[0].set_title('Observed Streamflow', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Baseflow Comparison
        axes[1].plot(dates, cal_bfs_out[baseflow_col].values, 'b-', linewidth=1.5, 
                    label='Calibrated Baseflow', alpha=0.7)
        axes[1].plot(dates, ref_bfs_out[baseflow_col].values, 'r--', linewidth=1.5, 
                    label='Reference Baseflow', alpha=0.7)
        axes[1].set_ylabel('Baseflow\n(m³/day)', fontsize=10)
        axes[1].set_title('Baseflow Comparison', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot 3: Surface Flow Comparison
        axes[2].plot(dates, cal_bfs_out[surface_col].values, 'b-', linewidth=1.5, 
                    label='Calibrated Surface Flow', alpha=0.7)
        axes[2].plot(dates, ref_bfs_out[surface_col].values, 'r--', linewidth=1.5, 
                    label='Reference Surface Flow', alpha=0.7)
        axes[2].set_ylabel('Surface Flow\n(m³/day)', fontsize=10)
        axes[2].set_title('Surface Flow Comparison', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Plot 4: Total Simulated Flow Comparison
        cal_qsim = cal_bfs_out[baseflow_col].values + cal_bfs_out[surface_col].values + cal_bfs_out[direct_col].values
        ref_qsim = ref_bfs_out[baseflow_col].values + ref_bfs_out[surface_col].values + ref_bfs_out[direct_col].values
        axes[3].plot(dates, cal_bfs_out[qob_col].values, 'k-', linewidth=1, label='Observed', alpha=0.5)
        axes[3].plot(dates, cal_qsim, 'b-', linewidth=1.5, label='Calibrated Simulated', alpha=0.7)
        axes[3].plot(dates, ref_qsim, 'r--', linewidth=1.5, label='Reference Simulated', alpha=0.7)
        axes[3].set_ylabel('Total Flow\n(m³/day)', fontsize=10)
        axes[3].set_xlabel('Date', fontsize=10)
        axes[3].set_title('Total Simulated Flow Comparison', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        plt.tight_layout()
        plt.savefig('calibration_comparison.png', dpi=150, bbox_inches='tight')
        print("\nComparison plot saved as 'calibration_comparison.png'")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        cal_bff = np.nansum(cal_bfs_out[baseflow_col]) / np.nansum(cal_bfs_out[qob_col])
        ref_bff = np.nansum(ref_bfs_out[baseflow_col]) / np.nansum(ref_bfs_out[qob_col])
        
        print(f"\nBaseflow Fraction (BFF):")
        print(f"  Calibrated: {cal_bff:.4f}")
        print(f"  Reference:  {ref_bff:.4f}")
        print(f"  Difference:  {cal_bff - ref_bff:.4f} ({((cal_bff - ref_bff) / ref_bff * 100):.2f}%)")
        
        # Calculate mean absolute error for simulated flow
        cal_mae = np.nanmean(np.abs(cal_qsim - cal_bfs_out[qob_col].values))
        ref_mae = np.nanmean(np.abs(ref_qsim - ref_bfs_out[qob_col].values))
        
        print(f"\nMean Absolute Error (Simulated vs Observed):")
        print(f"  Calibrated: {cal_mae:.2f} m³/day")
        print(f"  Reference:  {ref_mae:.2f} m³/day")
        
        plt.show()
        
    except Exception as e:
        print(f"\nERROR during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

