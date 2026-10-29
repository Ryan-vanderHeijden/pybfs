# pybfs: current architecture (Phase 0 reconnaissance)

Snapshot of `refactor/core-cleanup` at `d8e2446` (identical to tag
`baseline-original` and to `upstream/main`, release 0.2.0). Line numbers below
refer to this commit.

Environment used for all checks: `/home/ryan/miniforge3/envs/hydro/bin/python3`
(Python 3.11, numpy 2.4.5, pandas 3.0.3, numba 0.65.1, scipy 1.17.1,
statsmodels 0.14.6, pytest 9.1.1). The package is not installed; run with
`PYTHONPATH=.` from the repo root.

---

## 1. Repository map

| Path | What it is | Notes |
|---|---|---|
| `pybfs/__init__.py` | Public API re-exports, `__version__ = "0.2.0"` | Exports 26 names, including low-level geometry helpers |
| `pybfs/bfs.py` (577 lines) | The forward model `bfs()` plus the numba kernel `_bfs_core_loop` | Two full copies of the time-step loop (JIT and pure Python) |
| `pybfs/utilities.py` (1135) | Reservoir geometry helpers (×2: plain + `_jit`), `base_table`, `get_values_for_site`, `forecast`, `bf_ci`, `flow_metrics` | Mixed bag: physics, I/O-ish helpers, forecasting, statistics |
| `pybfs/calibrate.py` (1160) | `bfs_calibrate` (3-step Nelder-Mead procedure) and its sub-objectives | Module-level global state; heavy `print` diagnostics |
| `pybfs/skill.py` (331) | `modified_strict_baseflow`, `separation_skill`, `forecast_skill` | Added April 2026; uses `bfs` and `forecast` |
| `pybfs/plot.py` (308) | 4 matplotlib plotting functions | Each writes a PNG to CWD and calls `plt.show()` |
| `examples/main.py` | Forward run + forecast, site 2312200 | Uses `docs/files/` data |
| `examples/main_calib.py` | Full calibration of site 12167000 and comparison with the R result | Writes CSV/PNG to CWD |
| `examples/main_skill.py` | Skill functions on site 12167000 | Reads `examples/bfs_params_12167000_python.csv` |
| `examples/bfs_params_12167000_python.csv` | A committed output of `main_calib.py` | Used as input by `main_skill.py` |
| `docs/files/2312200_data.csv` | 365 days (2018) of daily streamflow, m³/day | Only data with pre-calibrated parameters |
| `docs/files/bfs_params_50.csv` | 50 sites of calibrated parameters (USGS R BFS output format) | Column names = the R names |
| `bfs/` | **The original USGS R implementation** (`bfs/source/*.R`), R driver scripts, site 12167000 input (730 days, WY2014–2015) and R outputs in `bfs/out/` | The reference the Python port was written against |
| `bfs/12167000.csv` | Same data as `12167000_streamflow.csv` + `12167000_dates.csv` combined | Contains 4 days of exactly 0 flow (ice-affected; see suspected issues) |
| `bfs/out/timeseries/bfs_12167000.csv` | Full R `bfs_out` time series for R-calibrated parameters | Useful Python↔R cross-check |
| `notebooks/pybfs_forecast.ipynb` | Colab notebook (pip install pybfs, upload CSVs, run `main.py` flow) | Not runnable offline as-is (uses `google.colab`) |
| `plans/plan_skill.md` | Design doc for `skill.py` | Historical |
| `docs/`, `mkdocs.yml`, `.readthedocs.yml` | mkdocs-material site, API docs via mkdocstrings | |
| `pyproject.toml` / `requirements.txt` | setuptools; deps numpy, pandas, matplotlib, numba, scipy, statsmodels | `requirements.txt` also pulls in docs deps; `requires-python >=3.7` is not true in practice (numba 0.56+) |

**Tests: none exist.** No `tests/` directory, no CI config.

**How the model is normally run:** a script loads a streamflow CSV into a
DataFrame with columns `Date` and `Streamflow` (m³/day). Then it either
(a) loads a calibrated parameter table and calls `get_values_for_site` →
`base_table` → `bfs`, or (b) calls `bfs_calibrate(site, area_m2, q, dates)`.
Everything runs in-process; there is no CLI.

---

## 2. Module dependency overview

```
pybfs/__init__.py
 ├── utilities.py ── numpy, pandas, math, numba.jit, statsmodels.QuantReg
 ├── bfs.py ──────── utilities (sur_z, sur_store, sur_q, dir_q, infiltration, recharge  + their *_jit twins), numba.jit
 ├── plot.py ─────── matplotlib only (no pybfs imports)
 ├── skill.py ────── bfs.bfs, utilities.forecast
 └── calibrate.py ── bfs.bfs, utilities.{base_table, bf_ci, flow_metrics}, scipy.optimize.minimize
```

There are no import cycles. `utilities.py` is the base layer, but it also holds
higher-level functions (`forecast` is a second model implementation; `flow_metrics`
is signature analysis). Importing `pybfs` imports matplotlib, statsmodels,
scipy and numba.

---

## 3. End-to-end computation path

### 3a. Forward run with known parameters (`examples/main.py`)

```
pd.read_csv → DataFrame[Date, Streamflow (m³/day)]
get_values_for_site(params_df, site_no)          utilities.py:269
   → basin_char = [AREA, Lb, X1, Wb, POR]
     gw_hyd     = [ALPHA, BETA, Ks, Kb, Kz]
     flow       = [Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise]
base_table(lb, x1, wb, beta, kb, streamflow_df, por)   utilities.py:323
   → SBT: DataFrame[Xb, Z, S, Q], ≤1001 rows, base-reservoir lookup table
bfs(streamflow, SBT, basin_char, gw_hyd, flow, timestep='day',
    error_basis='total', use_jit=True)           bfs.py:224
   → 18-column DataFrame (Qsim, SurfaceFlow, Baseflow, DirectRunoff, states, …)
plot_baseflow_simulation(streamflow, result)     plot.py:12   (PNG + show)
forecast(df[date, streamflow=NaN], SBT, basin_char, gw_hyd, flow, initial)   utilities.py:409
```

### 3b. Inside `base_table` (utilities.py:323–406)

1. `qq` = 0 followed by 1000 log-spaced discharges from `min(Q>0)/10` to `max(Q)` (lines 358–368).
   **The table depends on the streamflow series passed in**, not only on parameters.
2. Invert the analytic base-reservoir discharge relation for the water-table
   height at the stream intersection `z(q)` (β ≠ 0.5: power law; β = 0.5: exponential) (371–374).
3. `x = X1 · z^(1/β)` = where the water table meets the surface; storage
   `s = Wb·POR·(∫₀ˣ (ξ/X1)^β dξ + (Lb − x)·z)` (377–378).
4. Round to 5 significant digits (R `signif`) and keep rows with `Xb < Lb` (380–406).

The geometry: the base water table is `z(x) = (x/X1)^β`, measured from the
basin outlet. Discharge is `Q = Wb·Kb·β²/(X1(2β−1)) · z^((2β−1)/β)`. See the
derivation comments in `bfs/source/Rfunctions.bfs_utilities.R:28–62`.

### 3c. Inside `bfs()` (bfs.py:224–576)

**Preprocessing (Python, 311–357)**
1. `qin = Streamflow`; negative values → NaN; `qmean = nanmean(qin)`.
2. `dq` = day-to-day change (hourly variant: change from the 24-h max).
3. `dqfr = dq/qin` (fractional change), with zero-flow special cases.
4. `rise = (dqfr > fr4rise) & (dq > prec)`; `recess = (dqfr <= fr4rise) | (dq < prec)`.
5. `recess_day` = consecutive-recession-day counter (cumsum reset trick, line 357).
6. Allocate output arrays `X, qcomp(p×3), ETA, I, Z(p×2), ST(p×2), EXC(p×2)`.

**Parameter checks (370–378):** print-only. They set `ts = 10*p`, but `ts` is
reset to 0 on line 389, so a bad parameter set still runs (see suspected issues).

**Initial state (389–431):** skip leading NaNs. Initial baseflow
`qb_in = min(q0, qmean)`. Look up `xb, zb, sb` in the SBT by `Q`. Surface flow
`qs_in = max(0, q0 − qb_in)`, then invert `sur_q` for `zs_in` and get
`ss_in = sur_store(zs_in)`. Compute available storages `ssa`, `sba` and the
initial recharge `rech_in`.
*With `use_jit=True` (the default) all of this is computed again inside
`_bfs_core_loop`, and the Python values are discarded.*

**Time-step loop (`_bfs_core_loop`, bfs.py:24–221; pure-Python copy 447–551).** For each step `ts`:
1. *Begin-of-step state* comes from step `ts−1`: `xb_in = X[ts−1]`, `qb_in = SBT.Q[count(SBT.Xb ≤ xb_in)−1]`,
   `zs_in, ss_in`, `qs_in = sur_q(zs_in)`; available storage `ssa, sba`;
   `rech_in = min(recharge(...), sba + qb_in)`.
2. *Impulse inversion* (only if `rise[ts] or rise[ts−1]` and `etaest = q − qb_in − qs_in > 0`):
   - initial guess `I = etaest/(2·Lb·ws)` (m/day over the surface zone);
   - for each factor `x` in `ifact = [2, 1.1]`, keep multiplying `i *= x` while the
     residual `eta = q − qs − qd − qb_in` exceeds `max(prec, q/100)`. The surface state is
     re-evaluated at each trial (`infiltration → ss → sur_z → sur_q`, `dir_q`). `I[ts]` holds the last
     accepted value.
   - This is a 1-D geometric search for the rainfall that reproduces observed flow,
     so **the model is driven by the streamflow itself; no precipitation input is used.**
3. `infil_in = min(infiltration(zs_in, I), ssa)`.
4. *End-of-step state:* `ss_en = max(ss_in + infil_in − rech_in − qs_in, 0)` → `zs_en`, `qs_en`;
   `infil_en`, `rech_en`; `sb_en = max(sb_in + rech_en − qb_in, 0)` → SBT lookup by `S` → `xb_en, zb_en, qb_en`.
5. *Step fluxes* (trapezoid): `SurfaceFlow = (qs_in+qs_en)/2`, `Baseflow = (qb_in+qb_en)/2`,
   `Infil = (infil_in+infil_en)/2`, `Rech = (rech_in+rech_en)/2`.
6. *Storage update* (not on the first step): `ST_s = clip(ST_s[−1] + Infil − SurfaceFlow − Rech, 0, Ss_max)`,
   `ST_b = clip(ST_b[−1] + Rech − Baseflow, 0, max SBT.S)`. Then `Z`, followed by direct runoff
   `DirectRunoff = dir_q(zs_in, I) + dir_q(Zs − zs_in, I/2) + max(2·Lb·(ws − zs_in/α)·(I − Ks), 0)`.
7. `ETA = q − (SurfaceFlow + Baseflow + DirectRunoff)`; `X[ts]` from the SBT at the new base storage.

Note that the per-step state carried forward is `X[ts]`, `Z[ts]`, `ST[ts]`. The
end-of-step quantities `ss_en, sb_en` are *not* the carried state; `ST` is recomputed
from the trapezoid fluxes in step 6.

**Post-processing (554–576):**
`q4er` = Baseflow (`error_basis='base'`) or the component sum (`'total'`);
`APE = (q4er − q)/(q + prec)`; `Weight = 1 − exp(Rs·recess_day)`, set to 1 where `APE > 0`;
`Qsim = Σ components` (NaN if any component is NaN, so day 0 is always NaN). The output
is a DataFrame with 18 columns.

### 3d. Forecast (`forecast`, utilities.py:409–599)

This is a third copy of the time step, with `I = 0` and no impulse inversion. It
starts from an 8-tuple `initial` of the last `bfs` row
`(X, Zb.L, Zs.L, StBase, StSur, SurfaceFlow, Baseflow, Rech)`. The input
DataFrame uses **lowercase** `date`/`streamflow` (its streamflow values are ignored).
Output columns: `Date, Baseflow, StSur, StBase, Zs, Zb, Rech`.

### 3e. Skill (skill.py)

- `modified_strict_baseflow(hydrograph, quantile)` returns a boolean mask of
  "baseflow-only" days from streamflow derivatives (no model).
- `separation_skill` runs `bfs` (error_basis='total') and computes RMSE/MAE of `Q − Baseflow` on the masked days.
- `forecast_skill` finds runs of masked days ≥ `min_days` and, for each one, runs
  `bfs` on the preceding `train_days` and then `forecast` over the run. It returns per-run RMSE/MAE.

---

## 4. Calibration (`bfs_calibrate`, calibrate.py:599–1159)

Port of `bfs/source/Rfunction.bfs_calibrate.R`. Entry signature:
`bfs_calibrate(tmp_site, tmp_area [m²], tmp_q [m³/day array], dys)`. It returns
`(bf_params, bff, ci_table, bfs_out)`.

### 4a. Fixed inputs derived from data (not optimized)

`flow_metrics(tmp_q, 'day', 0.05)` (utilities.py:694) returns
`[Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise, rb10]`. `rb10 = (RbI, RbS)` are the
intercept and slope of a 10th-percentile quantile regression of 10-day recession
rate on log10 Q. These values are fixed for the whole calibration.
`Qmean = mean(q ≥ 0)`.

### 4b. Initial guesses (calibrate.py:681–699)

| Param | Initial value | Optimized? |
|---|---|---|
| Lb | `2·sqrt(A/2)`, then adjusted by `ini_params` | yes |
| Wb | `A/Lb/10`, adjusted by `ini_params` | yes |
| POR | 0.15 | **never** |
| ALPHA | 0.01 | yes (cal_initial only) |
| BETA | 1 | grid search in step 2 |
| X1 | `1/ALPHA` = 100 | only in `cal_basetable` (step 2) |
| Kb | from `ini_params` (matches Rb1) | yes |
| Ks | `(1 − e^Rs)·POR·(¾·Ws)/ALPHA` | yes (cal_initial only) |
| Kz | `10·Qmean/(Lb·Wb)` | yes |

### 4c. Steps

| Step | Sub-objective (function) | Variables (space) | Model run per eval? |
|---|---|---|---|
| 0 | initial `bfs` run → row 0 of `bf_params` | — | 1× |
| 1a | `cal_initial` (calibrate.py:275) | log10 [Lb, Wb, ALPHA, Ks, Kb, Kz] | yes |
| 1b | `cal_surface` (525) | log10 [Lb, Wb, ALPHA, Ks] | **no; always returns 100 (no-op, see issues)** |
| 1c | `bfs` → row 1 | — | 1× |
| 2a | `cal_basetable` (352), once per β ∈ {0.6, 0.7, …, 10.1} | [Lb, X1, Wb, Kb] linear | no (`base_table` only) |
| 2b | best β = argmin of the 2a objective; `cal_base` (458) | [Lb, Wb, Kb, Kz] linear | yes |
| 2c | `cal_surface` again | (no-op) | no |
| 2d | `bfs` → row 2 | — | 1× |
| 3a | choose the row with **max BFF** among rows 1..n | — | — |
| 3b | `cal_initial` again from that row | log10 [Lb, Wb, ALPHA, Ks, Kb, Kz] | yes |
| 3c | final `bfs`, `bf_ci`, flow fractions `bff` | — | 1× |

Observed on site 12167000 (730 days): **25.6 s, 880 `bfs` calls**, plus many
`base_table`-only calls in the β sweep. It is fully deterministic (no RNG anywhere).

### 4d. Bounds (implemented as penalty walls, not optimizer bounds)

| Function | Rejected (returns penalty) when | Penalty |
|---|---|---|
| `cal_initial` | Lb·Wb > AREA; ALPHA > 0.1; Ks, Kb, Kz ∉ [1e-8, 1e5] m/day | 100 |
| `cal_basetable` | non-finite x; Lb ≤ 0; X1 ≤ 0; Lb > A/Wb; Kb ∉ [1e-8, 1e5]; `max SBT.Q < Qthresh` | 1000 |
| `cal_basetable` | fewer than 10 usable SBT rows in the [Qmin/10, Qmean] range | 100 |
| `cal_base` | Lb·Wb > A; Lb < 0; Wb < 0; Kb, Kz ∉ [1e-8, 1e5] (R has Kb < **2**, a typo) | 100 |
| `cal_surface` | always (index mismatch) | 100 |

### 4e. Objective functions

- **Model-fit objective** `objective(bfs_out, prec)` (calibrate.py:149), used by
  `cal_initial` and `cal_base`, with `error_basis='base'`, so
  `APE = (Baseflow − Qob)/(Qob + prec)`:
  ```
  W = Weight  (= 1 − exp(Rs·recess_day): ~0 on rising limbs, → 1 deep in recessions; 1 where APE>0)
  W[APE > prec/(Qob+prec)] = 0      # over-prediction beyond prec is ignored
  W[Qob == 0 & Baseflow == 0] = 0
  OBJ = Σ W·(APE² − 1)
  ```
  This is a weighted *sum* (not a mean) and is usually negative. The `−1` term
  gives one unit of credit per counted day, so moving a day into the over-prediction
  class costs `W` units of credit. In effect, the objective pushes simulated
  **baseflow up toward the observed flow during recessions, from below**.
- **Recession-shape objective** `cal_basetable` (calibrate.py:352): mean
  |r(Q) + (RbI + RbS·log10 Q)| over SBT rows between Qmin/10 and Qmean, where
  r = Q/S is the table's linear-reservoir recession rate. It matches the base table's
  recession rate to the empirical 10th-percentile recession curve. No time
  stepping.
- **Selection criterion** (step 3): the max *BFF* row, not the min objective.
- **Reported `Error`** `calculate_error` (calibrate.py:125): Σ|APE·W| / ΣW (R's `bfs()` return value).

### 4f. Optimizer

scipy `minimize(method='Nelder-Mead')`, wrapped by `_create_scaled_minimize`
(calibrate.py:68). It emulates R's `parscale` by optimizing `x/x0` (so every
search starts at the all-ones vector, with scipy's default 5% initial simplex).
Options: `maxiter=1000, fatol=1e-4` (and scipy's default `xatol=1e-4`).
R used `reltol=0.01`. The Python "reltol tracker" computes a stop flag that
nothing reads, so **R's convergence criterion is not reproduced**.

### 4g. How the model is called in the loop

`cal_initial` / `cal_base`: decode x → copy `basin_char`/`gw_hyd` lists and
overwrite entries by position → `base_table(...)` → `bfs(...,
error_basis='base')` → `objective(...)`. The result is also written to the
module global `calibrate.bfs_out`. Each evaluation rebuilds the SBT and two pandas
DataFrames (input and output).

---

## 5. Global state, side effects, I/O inside computation

| Where | What |
|---|---|
| `calibrate.py:15` `bfs_out = None` | Module global overwritten by every `cal_*` call and by `bfs_calibrate`. Carried over from R's `assign('bfs_out', ..., envir=.GlobalEnv)`. Nothing outside reads it. It makes the functions non-reentrant. |
| `calibrate.py:18` `_reltol_tracker` dict | Mutable module global, reset/updated per eval; its `stop` flag is never consumed. |
| `bfs.py:372,375,378` | `print()` inside the model, so it prints inside optimizer loops |
| `calibrate.py:631–1099` | ~40 `print` diagnostic lines in `bfs_calibrate` |
| `@jit(cache=True)` (bfs.py, utilities.py) | numba writes cache files into `pybfs/__pycache__` on first call; the first call costs ~1.3 s to compile |
| `plot.py:82,199,247,306` | Every plot function defaults to `save=True`, writes a fixed filename into **CWD**, and calls `plt.show()` |
| `examples/main_calib.py:197–332` | Writes CSVs and a PNG into CWD |
| `get_values_for_site` | Returns `None`s silently for an unknown site (dtype mismatches such as str vs int fail silently too) |
| `bfs()` | Copies its input; no mutation of caller data found |

---

## 6. Duplicated logic

1. **Three copies of the model time step**: `_bfs_core_loop` (numba), the pure-Python
   branch of `bfs()` (447–551), and `forecast()` (503–594). They have already
   drifted: the impulse condition is `ts > 0` in the JIT copy and `ts > 1` in the Python copy.
2. **Two copies of each geometry helper** (`sur_z`/`sur_z_jit`, …, 6 pairs) in utilities.py.
3. **Initial-state block** computed in `bfs()` (389–431) and again in `_bfs_core_loop` (32–81).
4. **SBT lookup by linear scan**, written out 6 times in `_bfs_core_loop`, and in pandas
   form (`(SBT[...] <= v).sum()`) 8 more times across `bfs`/`forecast`.
5. **Parameter unpacking** `basin_char[0..4]` etc. repeated in `bfs` (twice, 330–337 and 380–387), `forecast`, examples, and notebook.
6. `bfs_calibrate`: the BFF computation is pasted 3×, the 19-field `bf_params` row 4×,
   `cal_initial`/`cal_base`/`cal_surface` are near-identical, and the optimizer call block is repeated 5×.
7. `forecast()`: the `ts_ini` and `not ts_ini` begin-of-step blocks are identical apart from where the state comes from.
8. Column-name fallbacks such as `'Qob' if 'Qob' in cols else 'Qob.L3'` (R names) are scattered throughout; some are tautological (`'AdjPctEr' if ... else 'AdjPctEr'`).
9. Upstream branch `calibration-beta1` adds `calibrate_beta1.py`, a 1186-line copy of `calibrate.py` with step 2 changed.
10. Data: `bfs/12167000.csv` duplicates `12167000_streamflow.csv` + `12167000_dates.csv`.

---

## 7. Upstream activity and license

- **Upstream** `BYU-Hydroinformatics/pybfs`: last push 2026-09-25 (release 0.2.0,
  published on PyPI as `pybfs` 0.2.0; 0.1.1 before). Contributors: njones61 (39
  commits incl. "Norm Jones"), EniolaWebsterEsho (9), Amin (2). 1 star, 1 fork (this one).
  **No open or closed issues or PRs.** `origin/main` = `upstream/main` = `baseline-original`.
- **Branches:** `upstream/calibration-beta1` (2026-09-15) adds the copy-paste
  calibration variant described above. Activity comes in bursts (Jan, Feb, Apr, Jun,
  Sep 2026), mostly from the maintainer, with paper-driven additions.
- **Development style:** variants are added by copying modules (`calibrate_beta1.py`).
  Upstream changes will therefore most likely arrive as edits to
  `bfs.py`/`calibrate.py`/`utilities.py` or as new sibling copies.
- **License:** MIT, © 2025 BYU Hydroinformatics. Redistribution and modification are
  permitted; the copyright and permission notice must be kept in copies or
  substantial portions. Nothing prevents a deep rewrite. If the fork is ever
  published to PyPI it cannot reuse the name `pybfs`.
- **R reference code** in `bfs/source/` is the USGS BFS model (Konrad, USGS SIR
  2022-5114). It has no license file in the repo. USGS-authored software is typically
  public domain, but **I could not verify that from the repo**; check before
  redistributing the R files outside this fork.

**Implications for the refactor**

1. A deep refactor will diverge from upstream. Upstream is low-traffic and has no
   PR workflow, so the risk of merge conflicts is small, and upstream fixes would be
   ported by hand anyway.
2. Keep a **compatibility layer** with the current public signatures (`bfs`,
   `base_table`, `get_values_for_site`, `forecast`, `bfs_calibrate`, `flow_metrics`).
   That way upstream changes can be diffed against `legacy/` and replayed, and
   existing user scripts and the paper repo keep working.
3. Decide early whether this is a research fork or intended to go back upstream. If
   upstream is the goal, raise it with the maintainer before Phase 3, because the
   refactor will touch every module.
4. Keep `bfs/` (R reference) untouched. It is the ground truth for "faithful port"
   questions, and several entries in `suspected_issues.md` depend on it.
