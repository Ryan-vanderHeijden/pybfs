# Phase 3 summary: refactor as implemented

Branch `refactor/core-cleanup`, commits `e1828e8` … (see `git log baseline-original..`).
Not pushed. All numbers below were measured in the `hydro` environment
(Python 3.11, numpy 2.4.5, pandas 3.0.3, numba 0.65.1, scipy 1.17.1, statsmodels 0.14.6).

## What changed

| Area | Before | After |
|---|---|---|
| Model loop | 3 drifting copies (numba, pure Python, forecast) | one numba loop in `core/model.py` (+ the forecast loop in `core/forecasting.py`) |
| Geometry kernels | 2 copies each (plain + `_jit`) | one numba copy each in `core/geometry.py`, docstrings cite manual equations and departures |
| Inputs | positional lists `basin_char`, `gw_hyd`, `flow` | `ModelParameters`, `FlowMetrics` dataclasses; `PARAMETERS` registry (symbol, units, definition) |
| Base table | pandas DataFrame, linear-scan lookups | `BaseTable` arrays, binary-search lookups (exact legacy count semantics) |
| Errors/weights | computed inside `bfs()` | `objectives/legacy.py`, outside the model |
| Calibration | one 560-line function, module globals, `print` | `calibration/` (stateless `CalibrationProblem`, `ScoreSpec`, optimizer wrapper, `usgs.calibrate` in named steps, `logging`) |
| Flow metrics | 440 lines incl. R index transcription | `streamflow_metrics.py`, direct slicing |
| Plotting | always wrote PNGs to CWD and called `plt.show()` | returns the Figure; `save`/`show` opt-in |
| Python | `requires-python >= 3.7` (not actually supported) | `>= 3.10` |

The pybfs 0.2.0 interface (`bfs`, `base_table`, `forecast`, `bfs_calibrate`,
`cal_*`, `ini_params`, `objective`, `flow_metrics`, `bf_ci`, skill functions,
geometry helpers, `get_values_for_site`) keeps its names, signatures and
results, as wrappers. The only signature changes are the plot functions
(`save` default, new `show`), as agreed (D3). `pybfs/legacy/` holds the
untouched original.

## What was verified

- **Phase 1 characterization suite** (11 cases) passes after every commit, unregenerated.
  The two full calibrations are **bit-identical**, including every one of the
  880 and 715 model runs in their recorded trajectories.
- **Differential tests against `pybfs.legacy`**, all requiring exact equality
  (`tests/test_core_equivalence.py`):
  - kernels: 100,000 random inputs each;
  - base table: 300 random parameter sets (incl. β = 0.5 and overflowing tables); lookups on random/edge queries;
  - `simulate` vs `bfs()`: 500 random parameter sets × 5 records (both sites, masked zeros, rise on day 2, leading NaN), all 18 columns; every run non-degenerate (69,266 impulse days in total);
  - `forecast`: 400 random 90-day forecasts;
  - objectives, `cal_initial`/`cal_base`/`cal_surface`/`cal_basetable` at random feasible and infeasible points, `ini_params`;
  - `flow_metrics`: 171 fuzzed records (gaps, zeros, negatives, exact ties, constant/short series, daily and hourly), including fallback branches and the inputs where legacy raises;
  - `get_values_for_site` on all 50 sites and missing sites.
- All three example scripts run headless and reproduce the same calibrated parameters.
- The README example is a test.
- 46 tests in total; `-m "not slow"` runs in ~40 s, the full suite in ~70 s.

## Performance

| | Before | After |
|---|---|---|
| One model evaluation, 12167000 (730 d) | 2.6 ms | 0.37 ms (record prepared once) |
| Full `bfs_calibrate`, 12167000 | 26 s | 6 s |
| Full `bfs_calibrate`, 2312200 | 56 s | 13 s |

## Visible behavior differences (none numerical)

1. Parameter warnings in `bfs()` use `warnings.warn` instead of `print`; calibration diagnostics use `logging` (silent by default).
2. Plot defaults (D3).
3. An all-missing record raises `ValueError` in `simulate` (legacy: `IndexError`).
4. `pybfs.calibrate` no longer has module globals `bfs_out` / `_reltol_tracker` (they had no readers).

## What remains uncertain

1. **Bit-identity is per environment.** It holds on this machine with these library versions. A different numba/LLVM or scipy version could change the last bits, but it would change legacy too.
2. **Python `pow` vs `x*x`.** The legacy plain-Python kernels round `x**2` through libm `pow`, which differs from `x*x` by 1 ulp in about 0.08% of random calls. The legacy *forecast* used the plain kernels, and the new forecast uses numba. They matched exactly on 36,000 forecast steps, but this is not guaranteed for every input. The public plain helpers (`sur_store` etc.) still use Python arithmetic.
3. **Hourly mode and `use_jit=False`** still run the legacy code (D4, D5). Neither is covered by the new implementation or tested beyond Phase 1.
4. `ModelParameters` keeps numpy scalar types as given, so its repr shows `np.float64(...)`. This is harmless, but cosmetic cleanup would mean casting, and casting has to stay bit-neutral.
5. `__version__` is still `0.2.0`. I suggest `0.3.0.dev0` once you accept the refactor.

## Waiting for your confirmation

- ~~Deleting duplicated code~~ **Done after you confirmed the match:** `pybfs/legacy/`
  and the equivalence tests are removed; `use_jit` is ignored (M3) and hourly input
  raises `NotImplementedError`. The plain-Python geometry helpers (`sur_store` etc.)
  stay, because they are public names; they run the kernels' Python source.

## Suspected issues needing a decision (`suspected_issues.md`)

All are still reproduced exactly. In order of relevance to Phase 4:

| ID | Issue | Fix size |
|---|---|---|
| M10 | `sur_z` is not the inverse of `sur_store` (pinned by your decision) | 1 line; every output changes |
| C1 | `cal_surface` never runs the model | small; calibration results change |
| C2 | R's reltol stopping rule not implemented | small; calibration results change |
| C3 | β sweep stops at 10.1 (tautological test; `b == 20` unreachable) | small |
| M13 | Manual's error rules (100-day spin-up exclusion, Qthresh gating, Rb1/Rb2 weights) not implemented | now expressible as `ScoreSpec` options |
| C11/C12 | max-BFF selection; objective ignores over-prediction | Phase 4 objective design |
| M3 | JIT vs pure-Python impulse condition (pure-Python path still on legacy) | decide, then drop one path |
| D1 | 12167000 zero days (ice) treated as real zeros in `main_calib.py` | data decision |
| U1, U8, C4, C8, M12 | flow-metric defaults, forecast one-day offset, Kb bound, X1 fallback, initial baseflow | small each |
