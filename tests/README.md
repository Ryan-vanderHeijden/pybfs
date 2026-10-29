# Characterization tests

These tests pin the behavior of `baseline-original` (`d8e2446`), so the refactor
can be checked against it. They record what the code does, **including known
bugs** (see `notes/suspected_issues.md`, e.g. M10). They are not a statement of
what the model should do.

## Running

From the repo root, with the `hydro` environment:

```bash
PYTHONPATH=. python -m pytest tests/                 # everything, ~1 min
PYTHONPATH=. python -m pytest tests/ -m "not slow"   # skip the two full calibrations, ~5 s
```

## Layout

| File | Purpose |
|---|---|
| `cases.py` | One function per case; returns a flat dict of arrays. Shared by generation and tests. |
| `reference/generate.py` | Writes `reference/<case>.npz` and `reference/meta.json` (library versions, timings). |
| `reference/*.npz` | Exact (binary, lossless) reference arrays. |
| `test_characterization.py` | Re-runs each case and compares with its reference. |
| `test_core.py`, `test_params.py`, `test_plot.py`, `test_readme.py` | Unit tests: lookups, interface behavior, parameter tables, plots, the README example (slow). |

During Phase 3 a `test_core_equivalence.py` compared the new code with the
verbatim pybfs 0.2.0 copy (`pybfs/legacy`) on random and fuzzed inputs; both
were removed after the match was confirmed (see `git log`, commit `28d6568` and earlier).

## Cases

| Case | What it runs | Data |
|---|---|---|
| `kernels` | `sur_store`, `sur_z` (incl. round trip, which pins M10), `sur_q`, `dir_q`, `infiltration`, `recharge`, plain and `_jit`, on grids | 12167000 calibrated geometry |
| `base_table` | `base_table` for both sites, and for β = 0.5 (exponential branch), 2.5, 9.4 | both |
| `flow_metrics` | `flow_metrics` (6 metrics + `rb10`) | 12167000 with zeros kept and with zeros masked; 2312200 |
| `bfs_2312200` | `bfs`: JIT/total (default), pure-Python/total, JIT/base | 2312200, `docs/files/bfs_params_50.csv` |
| `bfs_edges_2312200` | `bfs` JIT and pure-Python on edited inputs: a rise on day 2 (where the two loops diverge, issue M3) and three leading NaN days | 2312200, edited |
| `bfs_12167000` | `bfs`: JIT/total, JIT/base, pure-Python/base; `objective`, `calculate_error`, `bf_ci` | 12167000 (zeros kept), committed `examples/bfs_params_12167000_python.csv` |
| `forecast_2312200` | `examples/main.py` workflow: train Jan–Sep 2018, `forecast` Oct–Nov | 2312200 |
| `calibration_pieces` | `ini_params`, `cal_initial`, `cal_base`, `cal_surface`, `cal_basetable` at fixed points | 12167000 |
| `skill_12167000` | `modified_strict_baseflow`, `separation_skill`, `forecast_skill` | 12167000 with zeros masked (as `examples/main_skill.py`) |
| `calibration_12167000` (slow) | full `bfs_calibrate`: final params, flow fractions, CI table, final `bfs_out`, and the **trajectory** of every parameter set simulated | 12167000 (zeros kept, as `examples/main_calib.py`) |
| `calibration_2312200` (slow) | same | 2312200 |

The calibration trajectory (one row per `bfs` call, `[basin_char, gw_hyd]`) shows
*where* a refactored calibration first diverges, not just that the end result differs.

## Tolerances

- Forward computations: `rtol = 1e-12`, `atol = 0`, NaNs must match position for position.
- Full calibrations: `rtol = 1e-9`.
- Integer, boolean, date, and column-name arrays must match exactly.

On the unmodified code every case is **bit-identical** across reruns (checked with
`generate.py --check-determinism`). The tolerances only leave room for
reordered floating-point arithmetic. Anything larger is a real behavior change.
Discrete decisions (rise detection, SBT lookups) can turn tiny arithmetic
differences into large output differences; when that happens, the trajectory
will show it.

## Randomness and determinism

**pybfs uses no random numbers.** There is no RNG call anywhere in `pybfs/`, and
`scipy.optimize.minimize(method='Nelder-Mead')` is deterministic: its initial
simplex is a fixed 5% perturbation of the starting point. Repeat runs are
therefore identical, and there is no seed to control. What *can* change results:

1. **Library versions.** scipy's Nelder-Mead implementation, numpy reductions, and
   statsmodels' `QuantReg` (used for `rb10`) can change between releases. The test
   warns when versions differ from `reference/meta.json`. The committed example
   output `examples/bfs_params_12167000_python.csv` already disagrees in Kz with a
   fresh run (issue C16), probably for this reason.
2. **Input preprocessing**, e.g. whether the four zero days at 12167000 are masked
   (Rs changes by about 30%).
3. **Thread counts:** not relevant today (no parallel reductions), but it will be once calibration is parallelized.

If a later phase adds a stochastic optimizer, it should take an explicit
`seed`/`np.random.Generator` argument, never global state.

## Coverage (measured 2026-10-29 with stdlib `trace`; numba disabled for the forward cases so JIT bodies are counted)

| Module | Statements executed |
|---|---|
| `bfs.py` | 94% |
| `calibrate.py` | 89% |
| `skill.py` | 89% |
| `utilities.py` | 88% |
| `plot.py` | 0% |

A mutation check (scratch script, nothing committed) confirmed that a 1e-9 relative
change to `recharge`, to `objective`, or to a single input flow each fails the suite.

**Not covered:**
- `timestep='hour'` anywhere (`bfs`, `flow_metrics`).
- Every fallback in `flow_metrics` (default Rs/Rb1/Rb2, NaN returns; issue U1) and the
  `bfs_calibrate` early exits on invalid flow metrics. No test input triggers them.
- `bfs` bad-parameter warnings (`bfs.py:370–378`). Neither calibration ever proposed a negative or oversize parameter set.
- `ini_params` length/width adjustment branches (issues C5/C6), the step-2 X1 fallback
  (C8), and the `cal_initial` penalty return: neither calibration reached them.
- `cal_surface`'s body below its rejection test, which is unreachable (C1).
- `skill.py` empty-result branches (no strict days, no training data, empty summaries).
- `plot.py`: not tested (it writes files and opens windows).
- Correctness against R. These tests pin the *Python* baseline only. The R outputs in
  `bfs/out/` were not used, because the Python port differs from R by design in several
  places (`notes/suspected_issues.md`).
- Input variety: two sites, both perennial. No intermittent streams, hourly data, or long records.

## Regenerating

Only on purpose, for example after an *approved* behavior change, and in its own
commit, so the diff shows exactly which outputs moved:

```bash
PYTHONPATH=. python tests/reference/generate.py <case> [<case> ...]
```
