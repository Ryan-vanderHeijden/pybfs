# Suspected issues

Logged during Phase 0. **None of these has been fixed.** Each one needs your
decision before anything changes numerical output. Line numbers refer to
`baseline-original` (`d8e2446`).

Origin key:
- **PY**: the problem exists only in the Python port (it diverges from the R reference in `bfs/source/`).
- **R**: inherited from the R reference and ported faithfully.
- **DESIGN**: not clearly a bug, but scientifically questionable. Flagged for your judgment.

Impact key: **H** changes calibrated parameters or default model output; **M**
changes output in plausible edge cases; **L** cosmetic, dead code, or rare.

## Status (updated 2026-10-29, after Phase 3)

| ID | Decision | Commit |
|---|---|---|
| M10 | **Open:** awaiting the original author on which surface-storage formulation is correct; reproduced as-is | — |
| C1 | **Fixed:** `cal_surface` searches [Wb, α, Ks, Kz] as documented | `2ea272c` |
| C9 | **Fixed** with C1: the step-2 surface search starts from the post-`cal_base` parameters | `2ea272c` |
| C2 | **Won't fix:** scipy's default Nelder-Mead tolerances (`fatol = xatol = 1e-4`) are accepted; R's reltol is not emulated | — |
| C3 | **Fixed:** β sweep 0.6–20 in exact 0.1 steps, early stop compares consecutive β | `dc6400a` |
| M13 | **Options:** `UsgsScoring` (spin-up exclusion, Qthresh gating, Rb1/Rb2 weights); default = pybfs 0.2.0, `MANUAL_SCORING` = manual. The Rb1/Rb2 *definitions* in `flow_metrics` are unchanged | `7e65322` |
| M3 | **Resolved:** single implementation; `use_jit` ignored | `bc61ba2` |
| M8, M9 | **Hourly mode removed** from the model (`NotImplementedError`); `flow_metrics` keeps hourly support | `f06a2b0` |
| others | Open, reproduced as in pybfs 0.2.0 | — |

---

## Calibration (`pybfs/calibrate.py`)

### C1. `cal_surface` never runs the model, so surface calibration is a no-op. [R, H] — FIXED
`calibrate.py:557–570`; R: `Rfunctions.bfs_calibration_sub.R:147–162` vs. the call sites at `Rfunction.bfs_calibrate.R:61,133`.
The callers pass `log10([Lb, Wb, ALPHA, Ks])`, but `cal_surface` was written for
`[Wb, ALPHA, Ks, Kz]`. Its rejection test `x[1] > -1` therefore tests
`log10(Wb) > -1` (Wb > 0.1 m), which is always true, so every evaluation returns 100.
**Verified:** Nelder-Mead on the constant function returns `x0` bit-for-bit (59 wasted evaluations per call).
The Python port copied the broken check on purpose (comment at line 557) and also
decodes the body correctly, but the body is unreachable.
Consequence: ALPHA and Ks are only tuned inside the 6-parameter `cal_initial` (steps 1a and 3b).
The documented "Step 2 … cal_surface" (RecalibrationDescription.txt) never happens.

### C2. R's `reltol=0.01` convergence criterion is not implemented. [PY, H] — WON'T FIX (scipy defaults accepted)
`calibrate.py:18–65,112–116`. `_wrap_objective_with_reltol` sets
`_reltol_tracker['stop']`, but nothing reads it. It also compares successive
*function evaluations*, not iterations. scipy stops on `fatol=1e-4` (absolute) and
the default `xatol=1e-4` in scaled coordinates, which is much tighter than R's 1%
relative tolerance. So Python calibrations run longer and can land on different optima than R.
`_create_reltol_callback` and `RelTolConvergence` (lines 20–29) are dead code.
`cal_initial`/`cal_surface` are wrapped twice (once at the call site, once inside `_create_scaled_minimize`).

### C3. The β sweep ends at the first β > 10, not at β = 20. [R, M] — FIXED
`calibrate.py:903–911`; R `Rfunction.bfs_calibrate.R:112–113`.
(a) The "convergence" test compares `last_obj` with `tmp_out[n-1][5]`, which is the
value that was just appended, so it is always 0 and the loop ends at β ≈ 10.1.
(b) `b == 20` can never be true: b accumulates 0.1 steps and reaches `20.000000000000014` (verified).
(c) If no `cal_basetable` result were ever finite (`n == 0`), the Python loop would never terminate.
Observed on 12167000: tested β = 0.6 … 10.0, "Convergence reached at beta=10.1".

### C4. `cal_base` Kb lower bound: the R typo `10-8` was silently corrected. [PY vs R, M]
`calibrate.py:492` uses `x[2] < 1e-8`; R `Rfunctions.bfs_calibration_sub.R:133` has `x[3]<(10-8)`, i.e. **Kb < 2 m/day**.
The R intent was almost certainly 1e-8, so Python is "right", but it does not match R.
Decide which one to keep.

### C5. `ini_params` differs from R. [PY vs R, M]
`calibrate.py:186–272` vs R `:24–71`.
(a) Python drops the `xb = 0` point (`kb_calc[xb == 0] = NaN`). R keeps it (finite for β = 1, where `0^0 = 1`).
(b) In R, `z`, `dzdx`, `kb` are computed once before the Wb loop, and `lb` changes inside
that loop do not update them. Python recomputes them every iteration.
Because the bounds loop only runs once from `bfs_calibrate`, with β = 1, the effect is probably small, but it is unverified.

### C6. `ini_params` Wb/Lb guard is dimensionally inconsistent. [R, M]
`calibrate.py:266` `if wb * area > lb:` compares m³ with m and is always true, so
`lb = 0.9*area/wb` on every widening step. The intent was presumably `wb * lb > area`.

### C7. `ini_params` returns a circular Kb. [R, L]
`calibrate.py:270`: `kb = qb[i]/(wb*z[i]*dzdx[i])`, where `qb = wb*z*kb*dzdx`, so this returns `kb[i]`. It is harmless but obscures the logic.

### C8. Step 2 adds an X1 fallback that R does not have. [PY, L]
`calibrate.py:879–881`: `if X1 < 1: X1 = max(1/ALPHA, 1)`. R has no lower clamp.

### C9. Step 2 can record parameters that disagree with the run they were scored on. [R, L] — FIXED (with C1)
`calibrate.py:953–1022`. After `cal_base`, `basin_char`/`gw_hyd` are only rebuilt
inside `if np.all(np.isfinite(result.x))` of the following `cal_surface`. If that
were ever non-finite, `bfs` would run with pre-`cal_base` geometry while `bf_params`
records the post-`cal_base` Lb/Wb/Kb/Kz. With C1 this cannot currently happen.

### C10. Step 3 never reads ALPHA back from the selected row. [R, L]
`calibrate.py:1032–1043`. Lb, X1, Wb, BETA, Kb, Ks, Kz come from row `g`, but ALPHA
stays at its current value. This is harmless only because C1 keeps ALPHA constant after step 1a.

### C11. Step 3 selects by max BFF, not by objective. [DESIGN, H]
`calibrate.py:1032`. This is the documented intent ("maximize baseflow"), but combined
with the `objective` asymmetry (C12) it makes the procedure strongly prefer high
baseflow fractions. It matters for Phase 4.

### C12. The objective ignores baseflow over-prediction beyond `prec`. [DESIGN, H]
`calibrate.py:176–183`. Days with `APE > prec/(Qob+prec)` get weight 0. The `−1`
per-day credit then turns "count as many weighted days as possible" into the main
signal. It is a sum, not a mean, so its scale depends on record length. Your
`cleanup/audit` work found related degeneracies; documented here for completeness.

### C13. `bf_params` final rounding: Python rounds to 6 decimals, R to 6 significant figures. [PY, M]
`calibrate.py:1101–1121` uses `np.round(x, 6)`; R `Rfunction.bfs_calibrate.R:186` uses `signif(..., 6)`.
Any parameter below about 5e-7 (for example a small Kb or Kz) is rounded to 0, and large values keep spurious digits.

### C14. SFF is defined as 1 − BFF, so BFF + SFF + DRF > 1. [R, M]
`calibrate.py:1151–1157`; R `:208–215`. `tmp_sf` is overwritten with `1 − tmp_bf` on
every day with flow, so direct runoff is counted on top. The R output for 12167000
shows BFF 0.40, SFF 0.60, DRF 0.268.

### C15. Defensive dead code and noise. [PY, L]
Tautological column fallbacks (`calibrate.py:138–139,168–169`); `_create_scaled_minimize`
re-imports `minimize`; ~40 `print` diagnostics.

### C16. The committed example output does not reproduce exactly. [observation]
`examples/bfs_params_12167000_python.csv` has `Kz = 3.34782`. Running `bfs_calibrate`
today gives `Kz = 3.338514`, while all other parameters, Error, and BFF match to the
printed digits. The CSV was probably generated with a different scipy/numpy version
or an earlier code state. Phase 1 references will be regenerated, not taken from this file.

---

## Model (`pybfs/bfs.py`)

### M1. Bad-parameter checks never stop the run. [PY (R similar), M]
`bfs.py:370–378` set `ts = 10*p` "to invalidate", but `bfs.py:389` resets `ts = 0`.
R sets `bad_params=TRUE` and never uses it either. A negative or inconsistent parameter
set is simulated anyway and only prints a warning (inside optimizer loops).

### M2. The parameter check lists `ks` twice and omits `kz`. [PY, L]
`bfs.py:370`. R checks `kz` (`Rfunction.bfs.R:100`) and also checks for `Inf` (`:102`), which Python does not.

### M3. The JIT and pure-Python loops differ in the impulse condition. [PY, M] — RESOLVED (single loop)
`bfs.py:118` (JIT) uses `ts > 0`; `bfs.py:475` (`use_jit=False`) uses `ts > 1`.
R's 1-indexed `ts>1` corresponds to Python `ts > 0`, so the JIT path matches R and the
pure-Python path skips a possible impulse on the second time step. The two agreed
bit-for-bit on 2312200 (no rise on day 2), but they will differ whenever day 1 or 2 is a rise.
**Verified** (Phase 1 case `bfs_edges_2312200`): with a 50% rise on day 2, the two loops
differ on 142 of 365 days, by up to 32% in Qsim. The single missed impulse changes the
surface state for months. Both behaviors are pinned.

### M4. Day-1 `Infil` is reported as half the available surface storage. [R, L]
`bfs.py:147–151`. On the first step `I[ts]` is NaN, so `infiltration()` returns NaN and
`infil_en = ssa`. On 2312200, `Infil[0] = 1.74e6 m³/day` while every later value is ≤ 4e4.
State is unaffected (the first-step storage uses `infil_in = 0`), but the reported flux is meaningless.

### M5. Python does not apply R's 6-significant-figure output rounding. [PY, L]
R rounds the `bfs_out` columns `Qob … AdjPctEr` to 6 sig figs (`Rfunction.bfs.R:232`)
*before* `objective()` reads `AdjPctEr`. Python uses full precision. This is a tiny,
systematic source of Python/R calibration differences, and probably not worth reproducing.

### M6. Duplicated parameter unpacking and dead code. [PY, L]
`bfs.py:330–337` repeated at `380–387`; the Python-side initial state (`389–431`) is
discarded on the JIT path; `stts` and `qb_en` (initial) are unused; `rise[np.isnan(rise)]`
is a no-op on a boolean array.

### M7. SBT lookups assume a monotone table. [PY, L]
The JIT lookups (`bfs.py:45–51`, etc.) break at the first failure, so they equal
R's `sum(...)` count only if the column is non-decreasing. That holds by construction
(log-spaced Q, monotone transforms, rounding), but it is an unstated invariant.
If the count is 0, the JIT path returns NaN where R returns a zero-length vector.

### M8. Recharge mixes a depth with a rate. [DESIGN, L]
`utilities.py:218` `min(zs*por, kz)` compares a drainable depth [m] with a conductivity [m/day].
This is consistent only with an implicit Δt = 1 day; hourly runs would be off by 24×.

### M9. Hourly mode is only partly supported. [R/PY, L]
`timestep='hour'` changes `dq` (and `flow_metrics` windows), but rate constants and
`ifact` assume daily steps. Calibration hard-codes `'day'`.

### M10. `sur_z` is not the inverse of `sur_store`. [R, H] — OPEN (asking the original author)
`utilities.py:47` (and the `_jit` copy at `:225`); R `Rfunctions.bfs_utilities.R:79`.
`a1 = 1/(2*a)` makes `sur_z` invert `Ss = Lb·POR·(2ws·Zs − Zs²/(2α))`, while `sur_store`
computes `Lb·POR·(2ws·Zs − Zs²/α)`. **Verified** (12167000 calibrated geometry):
`sur_z(sur_store(Zs))/Zs` = 0.997, 0.91, 0.84, 0.586 at 1%, 30%, 50%, 100% of α·ws.
In the calibrated run Zs reaches 0.34·α·ws (≈ 10% error). Every step converts storage → Zs with
`sur_z` and Zs → discharge with `sur_q`, so surface discharge and saturated area are biased low.
The negative-discriminant branch is unreachable. The docstring claims an inverse that doesn't hold.
The manual doesn't settle which form is right (eq. 4 matches neither; see `geometry_reconciliation.md` §2.1–2.2).

### M11. Kernel formulas depart from the manual's printed equations. [DESIGN / manual ambiguity, H if changed]
Details in `notes/geometry_reconciliation.md`. In short: `infiltration` uses Lb and Ks where
eq. 14 has Xb and Kz (the manual's own tables 2 and 5 agree with the code); `sur_store`
differs from eq. 4 (which conflicts with the manual's geometry); the ½ factor on the
saturated-expansion runoff term (`bfs.py:135,205`) is not in the manual. The code is a faithful port of R throughout.

### M12. Initial baseflow is `min(Q(1), Qmean)`; the manual says `Qthresh`. [R vs manual, M]
`bfs.py:42,401`; R `Rfunction.bfs.R:117`. Manual p. 12: "assuming baseflow is equal to the threshold streamflow: Qb(1) = Qthresh".

### M13. Manual error rules that neither R nor Python implement. [R vs manual, H for calibration] — OPTIONS (`UsgsScoring`)
- Manual p. 12: "The first 100 days … are not included in the calculation of overall model error." Neither `bfs` nor `objective` excludes them.
- Manual p. 10: "Model errors are not used for time steps when Q(t) < Qthresh." Not applied.
- Manual table 3: Rb1 weights errors when `error_basis='total'` and Rb2 when `'base'`. The code always weights by **Rs** (`bfs.py:564`).
- Manual table 3 / eq. 18: Rb1 = **median** and Rb2 = **95th percentile** of 10-day recession rates. `flow_metrics` evaluates a 10th-percentile quantile regression at log Qmean and at the 10th pct of log Q instead (also R). This partly resolves U5: the manual defines Rb2 as the slow (95th-pct) 10-day rate, but the code computes something else.
Caveat: the manual documents the older `bf_sep`. `bfs/RecalibrationDescription.txt` says the
weights and errors were revised afterward, so some of these may be intentional. Calibration
objectives in Phase 4 should decide on these explicitly.

---

## Utilities (`pybfs/utilities.py`)

### U1. `flow_metrics` invents defaults where R returns NA. [PY, M]
`utilities.py:927–940` sets Rs = −0.05 or −0.1 when no valid 2-day recessions exist;
`1096–1125` substitutes Rb1 = −0.002 and Rb2 = −0.001 when the quantile regression fails.
In R these cases yield NA, and `bfs_calibrate` aborts the site (`Rfunction.bfs_calibrate.R:8`).
Python instead calibrates with made-up recession constants.

### U2. `flow_metrics` Qthresh loop excludes `dq == 0`. [PY, M]
`utilities.py:1050` filters `dq < 0`; R (`:231`) includes zeros. Qthresh can differ from R at flat low-flow sites.

### U3. 10-day recession: R computes a criterion and immediately overwrites it. [R, M]
R `Rfunctions.bfs_utilities.R:207–208`: `rec[y] = (sum(dq)<=0) & (...)` is
overwritten on the next line by `rec[y] = (qp[1]==min(qp))`. Python ports only the
surviving line (`utilities.py:1001–1002`). The "no rise in window" buffer the comment describes is never applied.

### U4. `flow_metrics` Rs is sensitive to the last bit of tied flows. [R/PY, M]
(From the earlier bucket-model work.) At 01134500, a 1-ulp change to the input flips Rs
between −0.34443531 and −0.34372458, caused by exactly-tied consecutive flows in the
`dq < prec` / `qp[0] == max` comparisons. Rs sets `Weight`, so it reaches the objective.

### U5. Rb2 and Qthresh are almost unused. [observation]
Rb2 only appears in the sign check (M1) and in output tables. Qthresh only appears in
that check and in the `cal_basetable` viability test. R's header comment describes
Rb2 as the "50th percentile of 10-day recessions", but the code evaluates the regression
at the 10th percentile of log Q. **Meaning of Rb2 is uncertain.**

### U6. `sur_store` computes a clamp it never uses. [R, L]
`utilities.py:87` `z = min(ws*a, zs)` is unused (R has the same dead line with a
different, also unused, expression). Callers are expected to pass `zs ≤ ws·α`.

### U7. `bf_ci` "fixes" an R indexing bug, and its row alignment can drift. [PY vs R, L]
`utilities.py:634` replaces ±Inf errors with 1. R's `tmp.error[abs(tmp.error==-Inf)]=1`
indexes with 0/1 numerics, so it effectively sets only element 1. Python also skips
all-NaN quantile bins (`:654`), which shifts the `ci_table` row index used later at `:683–686`. CI output only.

### U8. `forecast()` probably has a one-day offset and a different API. [PY, M]
`utilities.py:559–575`. On the first forecast step the outputs are the *initial* values
copied from the last training day (`qcomp[0] = Surflow, Baseflow`; `ST[0], Z[0]` = initial states),
so forecast day 1 repeats the last observed day's state and the forecast lags by one step.
**Verified** on 2312200: the training run ends 2018-09-30 with Baseflow 179425.0 and StBase 3.168463e6;
the forecast row for 2018-10-01 has exactly the same values.
It also requires lowercase `date`/`streamflow` while `bfs` requires `Date`/`Streamflow`,
and the `streamflow` values are ignored. Not compared against R (the R forecast code isn't in `bfs/`). **Needs your judgment.**

### U9. `get_values_for_site` fails silently. [PY, L]
`utilities.py:307–320`: it returns `(None, None, None)` for a missing site or a
dtype mismatch (for example a `"12167000"` string vs an int), and iterates every row to build dicts.

---

## Skill (`pybfs/skill.py`)

### S1. `forecast_skill` `min_sat` has no effect. [PY, L]
`skill.py:230`: `run_sat = run_len/run_len` is always 1.0, so the documented saturation filter never filters anything.

### S2. `modified_strict_baseflow` treats NaN as zero flow. [PY, M]
`skill.py:36–38`: NaNs are filled with 0 before the derivatives and the quantile, so gaps
create artificial falling and rising limbs and lower `q_thresh`. Gap days themselves are reset to False, but their neighbors are affected.

### S3. `wet3_core` and `wet4` are the same expression. [uncertain]
`skill.py:56` and `:65` are identical. Criterion 4 is described as "followed by a
data point with a larger −dy/dt", which may need a different shift. I could not
check against `pybaseflow.strict_baseflow` offline. **Uncertain.**

---

## Data and examples

### D1. 12167000 contains four exact-zero days that are really missing ice records. [data, M]
`bfs/12167000.csv`: 2013-12-06/07/08 and 2014-02-06 are 0.0 mid-winter (flow goes
3.25e6 → 0 → 2.23e6). `main_calib.py` uses them as real zeros (they affect
`flow_metrics`: Q01, Prec, Qthresh). `main_skill.py` masks them to NaN. The two examples
preprocess the same data differently.

### D2. The plot functions write files into CWD by default. [PY, L]
`plot.py:17,82,…`: `save=True` writes fixed filenames such as `baseflow_simulation.png` into the current directory, then calls `plt.show()`.
