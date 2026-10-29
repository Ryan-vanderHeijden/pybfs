# pybfs variable glossary (Phase 0)

Current name → meaning → proposed name. Units are inferred from the code, R
comments (`bfs/source/*.R`), and the USGS report (SIR 2022-5114). Everything runs
on a daily step in SI units: lengths in **m**, flows in **m³/day**, storages in **m³**,
rates in **1/day**. "?" marks items I could not pin down. Those are flagged and left
for you rather than guessed.

The proposed names are a first draft for the Phase 2 naming convention: lowercase
snake_case, physical words first, no unit suffix in names (units go in docstrings
and in the parameter dataclass). Short math symbols are kept only inside the
geometry kernels, where they mirror equations in the report.

**R column-suffix convention** (still visible in Python output names and fallbacks):
`.L3` = volume per time step [L³], `.L` = length [L], `.T` = count of time steps.
Example: `Zs.L` = surface water level in m, `RecessCount.T` = number of days.

---

## 1. Model parameters (calibrated or fixed)

The containers today are positional lists. `basin_char = [AREA, Lb, X1, Wb, POR]`
and `gw_hyd = [ALPHA, BETA, Ks, Kb, Kz]`. The same quantity has up to four
spellings (CSV column, R/Python local variable, function argument).

| Current names | Meaning | Units | Defined | Used | Proposed |
|---|---|---|---|---|---|
| `AREA`, `area`, `tmp_area`, `tmp.area`, `basin_char[0]` | Drainage area | m² | param CSV; `bfs_calibrate` arg | Lb·Wb ≤ area checks; initial guesses | `drainage_area` |
| `Lb`, `lb`, `basin_char[1]` | Basin length along the stream (reservoir length) | m | param CSV; calibrated | everywhere in geometry | `basin_length` |
| `X1`, `x1`, `basin_char[2]` | Longitudinal scale of the base water-table profile z = (x/X1)^β, i.e. the distance where z = 1 m | m | param CSV; `1/ALPHA` initially; step 2 | `base_table`, `ini_params`, `cal_basetable` | `base_profile_scale` |
| `Wb`, `wb`, `basin_char[3]` | Width of the base (groundwater) zone | m | param CSV; calibrated | `base_table`, checks | `base_width` |
| `Ws`, `ws` (= Wb/2) | Width of the surface zone on one side of the channel (derived, not a parameter) | m | `bfs.py:331`, `calibrate.py:683` | all surface geometry | `surface_half_width` (derived property) |
| `POR`, `por`, `basin_char[4]` | Drainable porosity (specific yield), fixed at 0.15 in calibration | – | param CSV | storage ↔ level conversions, recharge | `drainable_porosity` |
| `ALPHA`, `alpha`, `a`, `gw_hyd[0]` | Surface hydraulic gradient = lateral slope of the surface zone. The max saturated thickness is `ws·α`, the saturated width is `zs/α` | – (m/m) | param CSV; calibrated (cal_initial) | all surface geometry | `surface_gradient` |
| `BETA`, `beta`, `b`, `gw_hyd[1]` | Exponent of the base water-table profile z = (x/X1)^β | – | param CSV; β sweep | `base_table`, `ini_params` | `base_profile_exponent` |
| `Ks`, `ks`, `gw_hyd[2]` | Hydraulic conductivity of the surface zone. It also caps the infiltration rate (`min(i, ks)`) | m/day | param CSV; calibrated | `sur_q`, `infiltration`, direct runoff | `surface_conductivity` |
| `Kb`, `kb`, `gw_hyd[3]` | Horizontal hydraulic conductivity of the base zone | m/day | param CSV; calibrated | `base_table` | `base_conductivity` |
| `Kz`, `kz`, `gw_hyd[4]` | Vertical conductivity, used as the **maximum recharge depth per step** in `min(zs·por, kz)` | m/day (used as m/step) | param CSV; calibrated | `recharge` | `max_recharge_rate` (or `vertical_conductivity`; see suspected issue M8) |

## 2. Flow metrics (data-derived constants; `flow = [...]`)

| Current names | Meaning | Units | Defined | Used | Proposed |
|---|---|---|---|---|---|
| `Qthresh`, `qthresh`, `flow[0]` | Lower threshold on baseflow: max of the min positive flow and the flow with the least-negative recession rate | m³/day | `flow_metrics` | sign check; `cal_basetable` viability only | `baseflow_threshold` |
| `Rs`, `rs`, `flow[1]` | Storm-flow (rapid) recession coefficient from the 95th pct of 2-day recessions after rises. Negative | 1/day | `flow_metrics` | `Weight = 1 − exp(Rs·recess_day)`; initial Ks | `storm_recession_rate` |
| `Rb1`, `rb1`, `flow[2]` | Baseflow recession coefficient at Qmean (rb10 line evaluated at log10 Qmean, capped ≤ −0.002) | 1/day | `flow_metrics` | `ini_params` only | `base_recession_rate_at_mean` |
| `Rb2`, `rb2`, `flow[3]` | ? "Low baseflow" recession coefficient: rb10 line at the 10th pct of log Q, capped ≤ −0.001. **The R header comment says 50th percentile, which contradicts the code** | 1/day | `flow_metrics` | sign check, output only | `base_recession_rate_low` (**meaning flagged**) |
| `Prec`, `prec`, `flow[4]` | Flow precision (measurement resolution): max of the smallest positive (Q − Q01) and 0.1·Q01 | m³/day | `flow_metrics` | rise test, APE denominator, objective cut-off, impulse tolerance | `flow_precision` |
| `Frac4Rise`, `fr4rise`, `Fr4Rise`, `flow[5]` | Minimum fractional increase in flow that counts as a rise (impulse allowed) | – | `flow_metrics` arg, default 0.05 | `rise`/`recess` masks | `rise_fraction` |
| `rb10` (7th element of the `flow_metrics` return), `RbI`, `RbS` | Intercept and slope of the 10th-percentile quantile regression of 10-day recession rate (log10 units per day) on log10 Q | 1/day; 1/day per log10(m³/day) | `flow_metrics:1077–1094` | `cal_basetable` target curve | `recession_curve_intercept`, `recession_curve_slope` |
| `Q01` | 1st-percentile flow | m³/day | `flow_metrics:749` | prec | `q_p01` |
| `Qmean`, `qmean` | Mean observed flow (≥ 0) | m³/day | several places | initial baseflow, Kz guess, X1 formula, cal_basetable range | `mean_flow` |

## 3. Base reservoir lookup table (`SBT`, `BT`, `sbt`)

| Current | Meaning | Units | Proposed |
|---|---|---|---|
| `SBT`, `BT`, `sbt` | Storage–discharge lookup table for the base reservoir, ≤ 1001 rows, sorted by Q | – | `base_table` (type `BaseTable`) |
| `Xb` (column), `sbt_xb`, `x` in `base_table` | Distance from the outlet where the base water table meets the surface | m | `intersect_distance` |
| `Z` (column), `sbt_z`, `z` | Base water-table height at that intersection (datum: basin outlet) | m | `water_table_height` |
| `S` (column), `sbt_s`, `s` | Base storage | m³ | `storage` |
| `Q` (column), `sbt_q`, `qq` | Base discharge | m³/day | `discharge` |
| `tmp_range`, `tmp_q` (in `base_table`) | log10 range [min Q>0 / 10, max Q] of the table | log10 m³/day | `log_q_range` |
| `signif` (nested fn) | R-style rounding to N significant digits | – | module-level `round_significant` |

## 4. `bfs()` time-series state and fluxes

Arrays have length `p` (number of steps). Column index meanings are positional today.

| Current | Meaning | Units | Defined | Proposed |
|---|---|---|---|---|
| `qin` | Observed streamflow, negatives → NaN | m³/day | `bfs.py:312` | `q_obs` |
| `p` | Number of time steps | – | `bfs.py:320` | `n_steps` |
| `ifact` | Multiplicative step sizes for the impulse search, [2, 1.1] | – | `bfs.py:318` | `IMPULSE_SEARCH_FACTORS` |
| `dq` | Change in flow from the previous step (hourly mode: from the max of the previous 24 h) | m³/day | `bfs.py:323` | `dq` |
| `dqfr` | Fractional change dq/q | – | `bfs.py:341` | `dq_frac` |
| `rise` | Step is a rise (dqfr > fr4rise and dq > prec), so an impulse is allowed on this and the next step | bool | `bfs.py:350` | `is_rise` |
| `recess` | Step is recession or flat | bool | `bfs.py:353` | `is_recession` |
| `recess_day` → output `RecessCount.T` | Consecutive recession-day counter | days | `bfs.py:357` | `recession_age` |
| `X` → `X` | Base intersect distance carried between steps | m | `bfs.py:360` | `intersect_distance` |
| `qcomp[:,0]` → `SurfaceFlow` | Lateral flow from the surface zone | m³/day | `bfs.py:361` | `surface_flow` |
| `qcomp[:,1]` → `Baseflow` | Flow from the base zone | m³/day | | `baseflow` |
| `qcomp[:,2]` → `DirectRunoff` | Saturation-excess plus infiltration-excess runoff from the impulse (NaN on step 0) | m³/day | | `direct_runoff` |
| `ETA` → `Eta` | Residual q_obs − Σ components. R comment: "state disturbances (positive = inputs)" | m³/day | `bfs.py:362` | `residual` |
| `I` → `Impulse.L` | Inferred precipitation impulse (depth per step over the surface zone); NaN on step 0 | m/day | `bfs.py:363` | `impulse` |
| `Z[:,0]` → `Zs.L` | **Zs**: saturated thickness of the surface reservoir at the section where the water surface intersects the land surface (manual eq. 9 key, fig. 1) | m | `bfs.py:364` | `surface_saturated_thickness` |
| `Z[:,1]` → `Zb.L` | Base water-table height at the intersect (datum = outlet) | m | | `base_level` |
| `ST[:,0]` → `StSur` | Surface storage | m³ | `bfs.py:365` | `surface_storage` |
| `ST[:,1]` → `StBase` | Base storage | m³ | | `base_storage` |
| `EXC[:,0]` → `Infil` | Infiltration into the surface zone (step average); step-0 value is meaningless (issue M4) | m³/day | `bfs.py:366` | `infiltration` |
| `EXC[:,1]` → `Rech` | Recharge surface → base (step average) | m³/day | | `recharge` |
| `Qob` (output) | = qin | m³/day | | `q_obs` |
| `Qsim` (output) | Σ of the three components (NaN if any is NaN) | m³/day | `bfs.py:570` | `q_sim` |
| `q4er` | Series the error is computed on: Baseflow (`'base'`) or Qsim (`'total'`) | m³/day | `bfs.py:554` | `q_for_error` |
| `APE` → `AdjPctEr` | Adjusted relative error (q4er − q)/(q + prec) | – | `bfs.py:560` | `relative_error` |
| `Weight` | Error weight: 1 − exp(Rs·recession_age), set to 1 where APE > 0 | – | `bfs.py:564` | `error_weight` |
| `ts` | Current step index | – | | `t` |
| `ts_ini` | First-step flag | bool | | `is_first_step` |
| `stts` | "Starting time step for error calculation", **never used** | – | `bfs.py:390` | (delete) |
| `timestep` | `'day'` or `'hour'` | – | arg | `timestep` |
| `error_basis` | `'base'` or `'total'` (see q4er) | – | arg | `error_basis` |
| `use_jit` | Use the numba kernel | bool | arg | (drop once there is a single implementation) |

### Per-step scratch variables (`_in` = start of step, `_en` = end of step)

| Current | Meaning | Units | Proposed |
|---|---|---|---|
| `qb_in`, `qb_en` | Base discharge at start/end of the step (SBT lookup) | m³/day | `q_base_start`, `q_base_end` |
| `xb_in`, `xb_en` | Base intersect distance | m | `x_base_start/end` |
| `zb_in`, `zb_en` | Base level; `zb_in` is assigned but **never read** | m | `z_base_start/end` |
| `sb_in`, `sb_en` | Base storage | m³ | `s_base_start/end` |
| `sba` | Available (unfilled) base storage = max S − sb_in | m³ | `base_capacity_left` |
| `qs_in`, `qs_en` | Surface-zone discharge | m³/day | `q_surface_start/end` |
| `zs_in`, `zs_en` | Surface saturated thickness | m | `z_surface_start/end` |
| `ss_in`, `ss_en` | Surface storage | m³ | `s_surface_start/end` |
| `ssa` | Available surface storage = S(ws·α) − ss_in | m³ | `surface_capacity_left` |
| `infil_in`, `infil_en` | Infiltration at start/end | m³/day | `infil_start/end` |
| `rech_in`, `rech_en` | Recharge at start/end, capped at sba + qb_in | m³/day | `recharge_start/end` |
| `etaest` | Residual before (or during) the impulse search: max(0, q − qb − qs) | m³/day | `residual_estimate` |
| `eta` | Residual for the current trial impulse | m³/day | `trial_residual` |
| `i` | Trial impulse | m/day | `trial_impulse` |
| `x` (loop var over `ifact`) | Current search factor | – | `factor` |
| `zs`, `qs`, `ss`, `infil`, `qd` | Surface state/flux and direct runoff for the trial impulse | m, m³/day, m³, m³/day, m³/day | `trial_*` |
| `idx` | SBT row index from a count-based search | – | replaced by a `lookup_*` helper |

## 5. Geometry kernels (`utilities.py:18–266`)

Argument names are short and inconsistent: `a` = α, `z` vs `zs` for the same thing,
`i` = impulse.

| Function | Formula | Returns | Proposed |
|---|---|---|---|
| `sur_store(lb, a, ws, por, zs)` | Lb·por·(2·ws·zs − zs²/α) | surface storage m³ (manual eq. 4 departs; see reconciliation §2.1) | `surface_storage_from_thickness` |
| `sur_z(lb, a, ws, por, ss)` | intended inverse of `sur_store`, **but not an exact inverse** (issue M10) | m | `surface_thickness_from_storage` |
| `sur_q(lb, a, ks, z)` | 2·Lb·zs·α·Ks (Darcy, two banks) | m³/day | `surface_discharge` |
| `dir_q(lb, a, z, i)` | 2·Lb·(zs/α)·i (rain on the saturated strip) | m³/day | `saturated_area_runoff` |
| `infiltration(lb, ws, ks, a, zs, i)` | 2·Lb·(ws − zs/α)·min(i, Ks) | m³/day | `infiltration_rate` |
| `recharge(lb, xb, ws, kz, zs, por)` | (Lb − xb)·2·ws·min(zs·por, Kz) | m³/day | `recharge_rate` |
| `*_jit` | numba copies of the above | | (merged into one set) |

## 6. Forecast

| Current | Meaning | Units | Proposed |
|---|---|---|---|
| `initial` = `(Xi, Zbi, Zsi, StBi, StSi, Surflow, Baseflow, Rech)` | Initial state from the last `bfs` row (positional tuple) | m, m, m, m³, m³, m³/day, m³/day, m³/day | `ModelState` dataclass |
| `date`, `streamflow` (lowercase input columns) | Forecast dates; streamflow is ignored | | `dates` argument |
| Output `Zs`, `Zb` | Same as `Zs.L`, `Zb.L` in `bfs` output (renamed inconsistently) | m | `surface_level`, `base_level` |

## 7. Calibration (`calibrate.py`)

| Current | Meaning | Units | Defined | Proposed |
|---|---|---|---|---|
| `tmp_site`, `tmp.site` | Site identifier | – | arg | `site_id` |
| `tmp_q` | Observed flow array | m³/day | arg | `q_obs` |
| `dys` | Dates | – | arg | `dates` |
| `bfs_out` (module global and local) | Latest `bfs` output DataFrame | – | `calibrate.py:15` | (remove global) |
| `_reltol_tracker` | Unused reltol bookkeeping (issue C2) | – | `calibrate.py:18` | (remove) |
| `X`, `LOGX`, `x`, `logx` | Decision vector (linear or log10), meaning changes per step | mixed | each step | named `ParameterVector` with a `space` field |
| `parscale` | Per-parameter scale; the optimizer works in x/parscale | – | `_create_scaled_minimize` | `scale` |
| `bad_X` | List of bound-violation booleans | bool | each `cal_*` | replaced by `ParameterSpec.bounds` |
| `b` (step 2) | Trial β in the sweep | – | `calibrate.py:866` | `beta_trial` |
| `n` (step 2) | Count of finite sweep results | – | | `n_valid` |
| `tmp_out` | Sweep results rows `[Lb, X1, Wb, β, Kb, obj]` | mixed | `calibrate.py:863` | `beta_sweep_results` (DataFrame) |
| `last_obj` | Objective of the last sweep result (tautological convergence test, issue C3) | – | | (remove) |
| `params` (cal_basetable arg) | `[AREA, POR, Qmean, Qthresh, RbI, RbS]` positional | mixed | | named args |
| `r` (cal_basetable) | Linear recession rate Q/S of each SBT row | 1/day | `calibrate.py:410` | `table_recession_rate` |
| `rbi`, `rbs` | = RbI, RbS | | | see §2 |
| `bf_params` | Parameter-set history (one row per step), then the final row | – | | `CalibrationHistory` / `CalibrationResult.params` |
| `g` | Index of the row selected in step 3 | – | `calibrate.py:1032` | `best_row` |
| `Error` | Mean absolute weighted APE (`calculate_error`) | – | | `weighted_mae` |
| `BFF` | Baseflow fraction Σ min(Baseflow, Qob)/Σ Qob | – | several | `baseflow_fraction` |
| `tmp_bf`, `tmp_ov` | Baseflow capped at Qob; over-prediction mask | m³/day; bool | | inside `baseflow_fraction()` |
| `bff` (DataFrame) | Summary: Qmean, BFF, SFF, DRF, Error | – | `calibrate.py:1137` | `FlowFractions` |
| `SFF` | Surface-flow fraction (in fact 1 − BFF; issue C14) | – | | `surface_flow_fraction` |
| `DRF` | Direct-runoff fraction ΣDirectRunoff/ΣQob | – | | `direct_runoff_fraction` |
| `ci_table`, `ci` | Credible-interval table by flow quantile and daily bands `CB0.05`, `CB0.95` | – | `bf_ci` | `uncertainty_table`, `daily_bounds` |
| `tmp_error`, `qnts`, `qnt` (bf_ci) | Fractional error (Qsim−Qob)/Qsim; flow quantiles; bin index | – | | `frac_error`, `q_quantiles`, `bin_index` |

### `ini_params` internals

| Current | Meaning | Units | Proposed |
|---|---|---|---|
| `xb` | Grid of 1001 intersect positions 0..Lb | m | `x_grid` |
| `z`, `dzdx` | Water-table height and gradient along the grid for z = (x/X1)^β | m, – | `z_grid`, `gradient_grid` |
| `sb` | Base storage on the grid (computed, **unused**) | m³ | (delete) |
| `kb_calc`, `kb` | Kb that would give recession rate −Rb1 at each x | m/day | `kb_for_rb1` |
| `qb` | Baseflow on the grid | m³/day | `q_grid` |
| `qmean_idx` | First grid index where qb > Qmean | – | `i_mean_flow` |

## 8. Flow-metrics internals (`flow_metrics`)

| Current | Meaning | Proposed |
|---|---|---|
| `xx` | Number of steps | `n` |
| `x` | Window length (2 days, then reused for 10 days) | `window` (two separate names) |
| `tmp1`, `tmp2`, `tmp3`, `tmp` | Gap-distance bookkeeping for interpolating gaps shorter than the window | `gap_length` |
| `qint_forward/backward` | Forward- and backward-filled flows | `q_ffill`, `q_bfill` |
| `rp_indices` | Matrix of lagged indices (row y = [y, y−1, …, y−x]) | (replace with slicing) |
| `rec` | Recession-window flag (redefined for the 10-day pass) | `is_recession_2d`, `is_recession_10d` |
| `rise`, `qrise`, `qmax`, `q4rise` | Rise flag, rise magnitude over the window, "step is window max", rise threshold | `is_rise_2d`, `rise_size`, `is_window_peak`, `rise_threshold` |
| `dq` (reused 3×) | Window change: daily differences; 2-day change / rise; 10-day fractional change | three distinct names |
| `dq_95` | 95th pct of 2-day post-rise recession fractions | `storm_recession_p95` |
| `logq`, `rb` | log10 Q and 10-day log recession rate on recession days | `log_q`, `log_recession_rate` |
| `tmp_qq`, `tmp_r` | Flow quantiles ≤ Qmean and their 95th-pct recession rate | `q_levels`, `recession_p95_by_level` |
| `logq_clean`, `rb_clean` | Finite subsets used for QuantReg | `log_q_fit`, `rate_fit` |
| `tmp_logq10` | 10th pct of log10 Q (for Rb2) | `log_q_p10` |

## 9. Skill (`skill.py`)

| Current | Meaning | Proposed |
|---|---|---|
| `hydrograph` | Input DataFrame (Date, Streamflow) | `streamflow` (consistent with `bfs`) |
| `Q`, `Q_filled` | Flow and NaN→0 copy | `q_obs`, `q_filled` |
| `dQ` | Centered derivative (Q[t+1] − Q[t−1])/2 | `dq_central` |
| `wet1`…`wet4`, `wet3_core`, `dry` | Exclusion masks from the strict-baseflow method; dry = baseflow day | descriptive names after S3 is resolved |
| `idx_first`, `idx_last`, `idx_before`, `idx_next`, `idx_remove`, `idx3*` | Index sets around rising segments | `rise_starts`, `rise_ends`, … |
| `q_thresh` | Upper-quantile flow threshold for criterion 3 | `high_flow_threshold` |
| `bf_strict`, `bf_mask`, `strict_mask`, `combined_indy`, `strict_indy`, `valid_indices` | Baseflow-day masks and indices (several names for the same concept) | `is_baseflow_day` |
| `BF_strict`, `BF_bfs`, `BF`, `SEQ`, `FC`, `RES` (output cols) | Mask, model baseflow, mask, sequence id, forecast, residual | `is_baseflow_day`, `baseflow_sim`, `sequence_id`, `baseflow_forecast`, `residual` |
| `min_sat`, `run_sat`, `sat` | Required/actual fraction of baseflow days in a sequence (no-op, issue S1) | `min_baseflow_share` |
| `train_days` | Spin-up length before each forecast | `spinup_days` |

## 10. Unresolved meanings (flagged, not guessed)

1. **Rb2**: physical meaning and percentile (comment vs. code; issue U5).
2. **Kz**: conductivity or max recharge depth per step. Code treats it as the latter (issue M8).
3. ~~Ws = Wb/2~~ **Resolved by the manual:** Wb is the full basin width spanning both hillslopes (fig. 1), and table 2 passes `Wb/2` to the surface kernels, so `ws` is the width of one hillslope. New ambiguity: manual table 4 calls Lb the "width of base reservoir" while eq. 9 calls it the width of the surface reservoir (see `geometry_reconciliation.md` §3).
7. **Kernel symbol definitions** now follow the manual; see `notes/geometry_reconciliation.md` §1. Proposed names for the surface state use the manual's term "saturated thickness" rather than "level" (for example `surface_saturated_thickness`), and `Xs = Zs/α` (saturated half-width of the land surface, manual eq. 3) gets its own name: `saturated_half_width`.
8. **Rb1/Rb2** (item 1): the manual (table 3, eq. 18) defines them as the median and 95th percentile of 10-day recession rates. The code computes neither (suspected issue M13).
4. **`wet4` in `modified_strict_baseflow`**: identical to `wet3_core` (issue S3).
5. **Eta sign convention**: the R comment says "positive values represent inputs", but in the code it is simply observed − simulated.
6. **X1** as "the distance where z = 1 m" follows from z = (x/X1)^β but makes X1 depend on the length unit; confirm that is the intended interpretation.
