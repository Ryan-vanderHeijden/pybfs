# -*- coding: utf-8 -*-
"""Prepare calibration-ready timeseries files for EA calibration.

For each site in RV_data/:
  - Drops invalid observations (valid=False -> NaN)
  - Selects the most data-rich 10-year window
  - Reindexes to a complete daily sequence and linearly interpolates gaps
  - Converts discharge from m³/s to m³/day
  - Writes to RV_data/calibration/{site_id}_cal.csv

Also writes RV_data/calibration/cal_summary.csv with per-site metadata.
"""

import os
import glob

import numpy as np
import pandas as pd

WINDOW_DAYS = 365 * 10  # 3650-day calibration window
INPUT_DIR = "RV_data"
OUTPUT_DIR = os.path.join(INPUT_DIR, "calibration")
BASIN_CHAR_FILE = os.path.join(INPUT_DIR, "selected_catchments.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load basin characteristics
basin_char = pd.read_csv(BASIN_CHAR_FILE, dtype={'gage_id': str})
basin_char['gage_id'] = basin_char['gage_id'].str.zfill(8)
area_lookup = dict(zip(basin_char['gage_id'], basin_char['area_gages2']))  # km²

q_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_daily_q.csv")))

summary_rows = []

for fpath in q_files:
    site_id = os.path.basename(fpath).replace("_daily_q.csv", "").zfill(8)

    df = pd.read_csv(fpath, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    df.index = df.index.tz_localize(None)  # strip timezone

    # Mask invalid observations
    df.loc[df['valid'] == False, 'discharge_cms'] = np.nan

    # Reindex to complete daily sequence
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_idx)

    # Find most data-rich 10-year window using rolling valid-day count
    valid = df['discharge_cms'].notna().astype(int)
    rolling_valid = valid.rolling(WINDOW_DAYS).sum()
    window_end = rolling_valid.idxmax()
    window_start = window_end - pd.Timedelta(days=WINDOW_DAYS - 1)

    subset = df.loc[window_start:window_end, 'discharge_cms'].copy()

    # Interpolate gaps within the window
    n_gaps_before = subset.isna().sum()
    subset = subset.interpolate(method='linear')
    # Forward/backward fill any remaining NaN at edges
    subset = subset.ffill().bfill()

    # Convert m³/s -> m³/day
    discharge_m3d = subset * 86400.0

    # Build output DataFrame
    out = pd.DataFrame({
        'date': discharge_m3d.index.strftime('%Y-%m-%d'),
        'discharge_m3d': discharge_m3d.values,
    })
    out_path = os.path.join(OUTPUT_DIR, f"{site_id}_cal.csv")
    out.to_csv(out_path, index=False)

    area_km2 = area_lookup.get(site_id, np.nan)
    n_valid_after = discharge_m3d.notna().sum()
    pct_interpolated = 100 * n_gaps_before / len(subset) if len(subset) > 0 else np.nan

    summary_rows.append({
        'site_id': site_id,
        'cal_start': window_start.strftime('%Y-%m-%d'),
        'cal_end': window_end.strftime('%Y-%m-%d'),
        'n_days': len(subset),
        'n_interpolated': int(n_gaps_before),
        'pct_interpolated': round(pct_interpolated, 2),
        'area_km2': area_km2,
        'area_m2': area_km2 * 1e6 if not np.isnan(area_km2) else np.nan,
        'mean_q_m3d': round(discharge_m3d.mean(), 4),
    })

    print(
        f"  {site_id}: {window_start.date()} – {window_end.date()} | "
        f"{n_gaps_before} days interpolated ({pct_interpolated:.1f}%)"
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUTPUT_DIR, "cal_summary.csv"), index=False)

print(f"\nWrote {len(q_files)} calibration files to {OUTPUT_DIR}/")
print(summary[['site_id', 'cal_start', 'cal_end', 'n_interpolated', 'pct_interpolated', 'area_km2']].to_string(index=False))
