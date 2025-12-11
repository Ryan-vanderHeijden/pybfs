#!/usr/bin/env python3
"""Profile the calibration to find performance bottlenecks"""

import cProfile
import pstats
import io
import pandas as pd
import numpy as np
from pybfs.calibrate import bfs_calibrate

# Load data
data = pd.read_csv('bfs/calibration_R_original/12167000.csv', encoding='utf-8-sig')
tmp_q = data['mean_daily_streamflow'].values
dys = pd.to_datetime(data['Date'], format='%m/%d/%Y').values

print("Profiling calibration...")
print("This will take a few minutes...")
print()

# Profile the calibration
profiler = cProfile.Profile()
profiler.enable()

bf_params, bff, ci_table, bfs_out = bfs_calibrate('12167000', 671000000.0, tmp_q, dys)

profiler.disable()

# Get profiling results
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s)
ps.sort_stats('cumulative')
ps.print_stats(30)  # Top 30 functions

print("=" * 70)
print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
print("=" * 70)
print(s.getvalue())

# Also print by total time
s2 = io.StringIO()
ps2 = pstats.Stats(profiler, stream=s2)
ps2.sort_stats('tottime')
ps2.print_stats(30)

print("=" * 70)
print("TOP 30 FUNCTIONS BY TOTAL TIME (excluding subcalls)")
print("=" * 70)
print(s2.getvalue())

