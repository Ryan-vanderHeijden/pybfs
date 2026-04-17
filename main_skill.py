#!/usr/bin/env python3
"""
Test script for skill assessment functions in pybfs.

Runs all three skill functions on site 2312200 and prints results.
"""

import pandas as pd
import numpy as np
import pybfs

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
streamflow_data = pd.read_csv('docs/files/2312200_data.csv')
streamflow_data['Date'] = pd.to_datetime(streamflow_data['Date'])

bfs_params = pd.read_csv('docs/files/bfs_params_50.csv')
site_number = 2312200
basin_char, gw_hyd, flow = pybfs.get_values_for_site(bfs_params, site_number)

lb, x1, wb, por = basin_char[1], basin_char[2], basin_char[3], basin_char[4]
alpha, beta, ks, kb, kz = gw_hyd
SBT = pybfs.base_table(lb, x1, wb, beta, kb, streamflow_data, por)

print(f"Record length: {len(streamflow_data)} days "
      f"({streamflow_data['Date'].iloc[0].date()} to {streamflow_data['Date'].iloc[-1].date()})\n")

# ── Phase 1: modified_strict_baseflow ─────────────────────────────────────────
print("=" * 60)
print("PHASE 1: modified_strict_baseflow")
print("=" * 60)

bf_mask = pybfs.modified_strict_baseflow(streamflow_data, quantile=0.8)

total_days = len(bf_mask)
n_strict = bf_mask.sum()
print(f"Total days:          {total_days}")
print(f"Strict-baseflow days: {n_strict} ({n_strict / total_days:.1%})")
print(f"Mean Q on all days:       {streamflow_data['Streamflow'].mean():.2f} m³/day")
print(f"Mean Q on strict days:    {streamflow_data['Streamflow'][bf_mask].mean():.2f} m³/day")

# ── Phase 2: separation_skill ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: separation_skill")
print("=" * 60)

skill_df, metrics = pybfs.separation_skill(
    streamflow_data, SBT, basin_char, gw_hyd, flow, quantile=0.8
)

print(f"RMSE:         {metrics['RMSE']:.2f} m³/day")
print(f"MAE:          {metrics['MAE']:.2f} m³/day")
print(f"Strict days:  {metrics['n_days']} ({metrics['frac_strict']:.1%} of record)")
print("\nskill_df head:")
print(skill_df.head(10).to_string(index=False))

# ── Phase 3: forecast_skill ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: forecast_skill")
print("=" * 60)

skill_df_fc, summary_df, fc_metrics = pybfs.forecast_skill(
    streamflow_data, SBT, basin_char, gw_hyd, flow,
    min_days=30, max_days=90, min_sat=0.9, train_days=365, quantile=0.8
)

print(f"Number of qualifying sequences: {len(summary_df)}")
print(f"Overall RMSE: {fc_metrics['overall_RMSE']:.2f} m³/day")
print(f"Overall MAE:  {fc_metrics['overall_MAE']:.2f} m³/day")
print("\nSequence summary:")
print(summary_df.to_string(index=False))

if len(summary_df) > 0:
    seq1_days = skill_df_fc[skill_df_fc['SEQ'] == 1]
    print(f"\nSequence 1: {len(seq1_days)} days, "
          f"{seq1_days['Date'].iloc[0].date()} to {seq1_days['Date'].iloc[-1].date()}")
    print(seq1_days[['Date', 'Q', 'FC', 'RES']].head(10).to_string(index=False))

print("\n=== Done ===")
