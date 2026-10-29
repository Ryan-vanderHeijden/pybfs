# Refactor plan (Phase 2 proposal)

Status: **proposal, awaiting approval.** No code has changed. References:
`architecture_current.md`, `variable_glossary.md`, `geometry_reconciliation.md`,
`suspected_issues.md`, and the Phase 1 safety net in `tests/` (12 tests, exact references).

## 0. Recommendation in one paragraph

**Write a new core alongside the old code, and migrate everything else
incrementally.** The model loop exists in three drifting copies, takes positional
lists, and mixes pandas with physics; refactoring it in place would mean editing all
three copies in lockstep. A new single-implementation core can be written in a few
hundred lines and checked number-for-number against the old one. The calibration
*procedure* (`bfs_calibrate`) is orchestration, not physics, so it is ported step by
step onto the new parts, with the Phase 1 trajectory test checking every model call
it makes. The old code moves verbatim into `pybfs/legacy/` and stays importable until
you confirm the new code matches. **No suspected issue gets fixed in Phase 3.** M10,
M3, C1, C2 and the rest are reproduced exactly. Fixes come afterwards, one approved
behavior change per commit.

---

## 1. Target package layout

```
pybfs/
  __init__.py            Public API: new names + compatibility wrappers with the old signatures
  params.py              ParameterSpec registry, ModelParameters, FlowMetrics (dataclasses)
  data.py                Streamflow validation/conversion helpers, parameter-table reader. Only I/O module.
  flow_metrics.py        flow_metrics() moved as-is (behavior unchanged), returns FlowMetrics
  core/
    geometry.py          The 6 kernels, one numba copy each, docstrings cite manual equations
    base_table.py        BaseTable (plain arrays) + build() + lookup helpers
    hydrograph.py        Rise / recession masks and recession age (preprocessing in bfs() today)
    simulate.py          simulate(): the forward model (one numba step loop)
    forecast.py          forecast() from a ModelState
    results.py           ModelState, SimulationResult (arrays), to_frame() / to_legacy_frame()
  objectives/
    __init__.py          Objective protocol, registry
    legacy.py            USGS weighted error (today's objective()), calculate_error, recession-rate fit (cal_basetable)
  calibration/
    problem.py           CalibrationProblem: which parameters, transform, bounds/penalties, objective, runner
    optimizers.py        Nelder-Mead wrapper with R-parscale emulation (bit-identical to today)
    usgs.py              The 3-step bfs_calibrate procedure, written with the pieces above
  diagnostics.py         bf_ci, flow fractions (BFF/SFF/DRF), skill functions (from skill.py)
  plot.py                Plot functions return the Figure; saving and showing are opt-in
  legacy/                Verbatim baseline modules (bfs.py, utilities.py, calibrate.py, skill.py, plot.py)
```

Rules:
- **`core/` and `objectives/` contain no pandas, no I/O, no printing, and no module state.**
  Arrays in, arrays/dataclasses out. They are safe to call thousands of times and from
  worker processes. Pickle-safety matters for spawn-based pools; the earlier bucket work
  hit exactly that.
- `pandas` appears only at the edges: `data.py`, `to_frame()`, the compatibility wrappers, `diagnostics.py`, `plot.py`.
- No module shares a name with a function it exports. Today `pybfs.bfs` (the module)
  is shadowed by `pybfs.bfs` (the function).
- The R reference (`bfs/`), the examples, and the notebook are untouched, except that the examples get import updates if you want them (§9).

## 2. The core

```python
def simulate(
    q_obs: NDArray[np.float64],          # observed flow [m³/day], NaN = missing
    params: ModelParameters,
    flow: FlowMetrics,
    *,
    base_table: BaseTable | None = None, # built from params if None
    hydrograph: HydrographSignals | None = None,  # rise/recession masks; computed if None
) -> SimulationResult: ...
```

- **One numba step loop** (`_step_loop`) takes plain arrays and scalars and fills
  preallocated output arrays. That replaces `_bfs_core_loop`, the pure-Python branch,
  and most of `forecast()`. Debugging without JIT uses `NUMBA_DISABLE_JIT=1` on the
  same code. The loop body follows the order the manual describes (start-of-step
  fluxes → impulse search → end-of-step estimates → trapezoid averaging → storage
  update), with each block in a small `@njit` helper named for that block.
- **Everything that depends only on the record is computed once**, outside the loop
  and outside calibration: rise/recession masks, recession age, `qmean`, and the Q range
  the base table spans. Today `bfs()` recomputes these on every call. The base table's
  dependence on the streamflow range becomes explicit:
  `BaseTable.build(params, q_min_positive, q_max)`.
- **Error and weights move out of the model.** `APE`/`Weight` (and `error_basis`)
  are objective concerns. The core returns physical quantities plus `recession_age`.
  `to_legacy_frame()` recomputes `AdjPctEr` and `Weight` exactly as today, so
  the old 18-column `bfs()` output is reproduced.
- **`SimulationResult`** fields (units in the docstring): `q_obs, q_sim, baseflow,
  surface_flow, direct_runoff, residual, impulse, surface_storage, base_storage,
  surface_saturated_thickness, base_saturated_thickness, intersect_distance,
  infiltration, recharge, recession_age`, plus `final_state() -> ModelState`, which
  replaces the positional 8-tuple handed to `forecast`.
- **SBT lookups** become one helper per comparison type. Today there are four distinct
  semantics (`Q ≤ v`, `Xb ≤ v`, `S < v`, `S ≤ v`, with different clamping on empty
  matches). Each gets a binary search that reproduces its count semantics exactly,
  verified against the linear scans.
- **Known bugs are reproduced, not fixed:** the `sur_z` inverse (M10), the day-1
  infiltration report (M4), `min(Q(1), Qmean)` initialization (M12), the ignored parameter
  checks (M1; the new core keeps "warn and run", but via `warnings.warn` instead of `print`).

## 3. Parameters

A single registry describes each parameter once:

```python
@dataclass(frozen=True)
class ParameterSpec:
    name: str              # "basin_length"
    symbol: str            # "Lb" (manual symbol; also the legacy CSV column)
    units: str             # "m"
    description: str       # manual definition, with table/equation reference
    transform: str         # "log10" | "linear" (search space default)
    bounds: tuple[float, float] | None   # physical limits used by calibration problems

@dataclass(frozen=True)
class ModelParameters:
    drainage_area: float           # AREA  [m²]
    basin_length: float            # Lb    [m]
    base_profile_scale: float      # X1    [m]
    base_width: float              # Wb    [m]
    drainable_porosity: float      # POR   [–]
    surface_gradient: float        # α     [–]
    base_profile_exponent: float   # β     [–]
    surface_conductivity: float    # Ks    [m/day]
    base_conductivity: float       # Kb    [m/day]
    vertical_conductivity: float   # Kz    [m/day]  (used as max recharge depth per step; manual eq. 16)

    surface_half_width -> float    # Wb/2 (property)
    replace(**changes) -> ModelParameters
    from_legacy(basin_char, gw_hyd) / to_legacy() -> (basin_char, gw_hyd)
    to_vector(names, transform) / with_vector(names, x, transform)

@dataclass(frozen=True)
class FlowMetrics:   # data-derived, never calibrated
    baseflow_threshold, storm_recession_rate, base_recession_rate_at_mean,
    base_recession_rate_low, flow_precision, rise_fraction,
    recession_curve: tuple[float, float] | None   # (RbI, RbS)
```

- **The calibration bounds of each step stay as data in `calibration/usgs.py`**, not
  in `ParameterSpec`, because today's steps use *different* walls (Kb ≥ 1e-8 in
  `cal_base` vs. R's 2; ALPHA ≤ 0.1 only in `cal_initial`; penalty 100 vs. 1000). Making
  them uniform would change results. `ParameterSpec.bounds` holds the physical
  limits for new problems in Phase 4.
- `data.read_parameter_table(path)` maps the legacy CSV columns (`AREA, Lb, …, Frac4Rise`)
  to `{site_id: (ModelParameters, FlowMetrics)}` and **raises** on an unknown site,
  replacing the silent `None` from `get_values_for_site` (which keeps its old behavior in the compatibility layer).

## 4. Objective functions

```python
class Objective(Protocol):
    name: str
    def __call__(self, obs: NDArray, sim: NDArray, *, weights: NDArray | None = None) -> float: ...
```

- **Metrics are pure functions** of observed and simulated series plus optional
  weights, returning a scalar to *minimize*. They know nothing about the model.
- **Which series and which days are scored is the calibration problem's job**, not the
  metric's:
  `ScoreSpec(series="baseflow" | "q_sim", weights=callable(result, flow) | None, eval_mask=bool array | None)`.
  `eval_mask` is what calibration/validation splits and spin-up exclusion need later
  (manual M13: "first 100 days excluded").
- **Phase 3 implements only today's objectives**, reproduced exactly:
  - `usgs_weighted_error` = today's `objective()` (Σ W·(APE² − 1) with the over-prediction cut-off),
  - `weighted_mean_abs_error` = `calculate_error`,
  - `recession_rate_mismatch` = the `cal_basetable` objective (scores a `BaseTable`, not a simulation).
- **Multi-objective:** a problem accepts a sequence of `(ScoreSpec, Objective)` pairs and
  returns a vector. Single-objective optimizers get a scalarization; multi-objective
  ones (Phase 4) get the vector.
- NSE, KGE, log variants, etc. are **not** written in Phase 3 (your Phase 4 list). Adding
  one is a single function in `objectives/` with no model changes.

## 5. Calibration

- `CalibrationProblem(names, transform, bounds_check, scores, base_params, flow, record)`
  exposes `evaluate(x) -> float | NDArray`. It precomputes everything record-dependent
  once and holds no mutable state between evaluations (replacing the `bfs_out` and
  `_reltol_tracker` globals).
- `optimizers.nelder_mead_parscale(f, x0, scale, maxiter=1000, fatol=1e-4)` reproduces
  `_create_scaled_minimize` bit-for-bit, **including not implementing reltol (C2)**.
  An `Optimizer` protocol (`minimize(problem, x0) -> OptimizeResult`) lets Phase 4 add
  other optimizers without touching `usgs.py`.
- `usgs.bfs_calibrate(...)` is a readable transcription of today's three steps, with each
  step a named function (`step1_initial`, `step2_beta_sweep`, `step3_reselect`). The
  quirks are preserved and commented with their issue IDs: the `cal_surface` no-op
  (C1), β sweep ending at 10.1 (C3), max-BFF selection (C11), ALPHA not re-read (C10),
  `np.round(…, 6)` output (C13). Diagnostics go through `logging`, not `print`.
- It returns a `CalibrationResult` dataclass (parameters, flow metrics, history table,
  final `SimulationResult`, flow fractions, CI table). The compatibility `bfs_calibrate`
  unpacks it into today's 4-tuple of DataFrames.

## 6. Conventions

- **Names:** snake_case physical words (glossary proposals, revised to the manual's terms:
  `surface_saturated_thickness`, not `surface_level`). Manual symbols appear in docstrings,
  and as names only inside short geometry kernels, where they mirror equations (`zs`, `lb`, `alpha`).
  Constants are UPPER_CASE (`IMPULSE_SEARCH_FACTORS = (2.0, 1.1)`).
- **Units:** never in names; always in docstrings, in brackets, with manual references:
  ```
  basin_length : float
      Effective basin length Lb [m] (manual table 4). Also the width of the
      surface reservoir perpendicular to its flow (eq. 9).
  ```
  Internal units are those of the input (the manual requires consistent units;
  the examples use m and m³/day). Docstrings state m and m³/day and note the daily step.
- **Docstrings:** NumPy style (already used; mkdocstrings renders it). Every kernel has a
  "Manual" section with its equation number and, where relevant, a "Departs from manual"
  note from `geometry_reconciliation.md`.
- **Type hints:** everywhere in the new code; `numpy.typing.NDArray[np.float64]` for arrays.
- **Python version:** 3.10+ in the new code (`X | None`, dataclass `slots`). See §9.

## 7. Execution sequence and validation (Phase 3)

Each step is a separate commit. **The full Phase 1 suite must pass after every commit**;
the references are never regenerated in Phase 3.

| # | Commit | Validation beyond Phase 1 |
|---|---|---|
| 1 | Copy baseline modules to `pybfs/legacy/` verbatim; the public API still points at the old code | Suite passes unchanged |
| 2 | `params.py`, `data.py`, `results.py` (no behavior) | Unit tests: `from_legacy` ↔ `to_legacy` round-trips on all 50 sites in `bfs_params_50.csv` |
| 3 | `core/geometry.py` | Bitwise equality with the legacy kernels on the `kernels` grid + 10⁵ random inputs |
| 4 | `core/base_table.py` + lookup helpers | Bitwise vs `legacy.base_table` for random parameters (β from 0.6 to 10.1, incl. 0.5); each lookup helper vs the legacy linear scan on random queries |
| 5 | `core/hydrograph.py`, `core/simulate.py` | **Differential test:** new `simulate` vs `legacy.bfs` (JIT path) for ~500 random parameter sets drawn inside calibration bounds × both sites × masked/unmasked 12167000. All 18 columns, `rtol=1e-12`, NaN positions identical |
| 6 | Switch public `bfs()`, `base_table()` to wrappers over the new core | Phase 1 suite now exercises the new core. `use_jit=False` routes to `legacy` until M3 is decided, so `bfs_edges` still pins the old pure-Python behavior |
| 7 | `core/forecast.py`, `diagnostics.py`; switch `forecast`, `bf_ci`, skill functions | Phase 1 forecast/skill cases; differential test over random initial states |
| 8 | `objectives/legacy.py` | Equality with `legacy.calibrate.objective`/`calculate_error`/`cal_basetable` on random simulations |
| 9 | `calibration/` (problem, optimizer, usgs); switch `bfs_calibrate` | Phase 1 calibration cases, **including the per-call trajectory**: the first diverging call pinpoints any discrepancy |
| 10 | Remove module globals and prints from the public path; `plot.py` returns figures (defaults per §9) | Suite |
| 11 | README section (layout + minimal example), docs API page | — |

Deleting the duplicated implementations (`*_jit` twins, pure-Python loop, `pybfs/legacy/`)
happens only **after** you confirm the match, as a separate step you approve.

**Rewrite risk and mitigation.** The fragile parts are the impulse search (two
nested multiplicative loops with data-dependent exits), the first-step special
cases (NaN impulse, `ts_ini` branches), and the four SBT lookup variants. These are
exactly where the differential test (step 5) concentrates. If bitwise equality isn't
achievable somewhere, for example a reordered sum, I stop and show you the size and
location of the difference before continuing.

## 8. Performance

Measured on 12167000 (730 days), one full `bfs_calibrate` = 26 s:

| Component | Calls | Share of calibration time |
|---|---|---|
| `cal_basetable` (β sweep, **no model run**) | 16,290 | **~90%**: `base_table` alone is ~35% of total (`signif` rounding, DataFrame build); pandas Series ops inside `cal_basetable` are most of the rest |
| `bfs()` | 880 | ~6% (3.2 ms/call) |
| `_bfs_core_loop` (numba) | 880 | ~1% (0.6 ms/call) |

Implications:
1. **The model is not the bottleneck; the plumbing is.** Removing pandas from
   `base_table`/`cal_basetable` (plain arrays, vectorized significant-figure rounding,
   no per-call DataFrame of the streamflow) should cut the β sweep by an estimated 10×.
   This is to be measured in step 4, with bitwise output.
2. **Per-evaluation cost of the model** falls from about 3.2 ms to about the 0.6 ms of the
   loop once DataFrame construction, `pd.to_datetime`, and the per-call preprocessing go away.
   Binary-search lookups replace 6 linear scans of the ~700–1000-row table on every step,
   so the loop itself should shrink further. That matters most for Phase 4
   (population optimizers, identifiability sampling: 10⁴–10⁶ evaluations).
3. **No vectorization over time is possible:** each step's impulse search depends on the
   previous state. The parallelism available is *across parameter sets*: the 19
   β-sweep optimizations are independent, and population-based optimizers evaluate whole
   generations. The core's statelessness makes process pools safe, and a numba
   `prange` over parameter sets is an option later.
4. The first JIT compile (~1.3 s) is cached on disk (`cache=True`) and doesn't matter for calibration.
5. Estimated full calibration after Phase 3: **a few seconds instead of 26 s**. That is an
   estimate, and each step reports measured timings. Any speed-up that changes the
   order of floating-point operations must still pass the bitwise tests or come to you.

## 9. Decisions needed before Phase 3

| # | Question | My recommendation |
|---|---|---|
| D1 | Minimum Python version for the new code (`pyproject.toml` says ≥3.7, which numba ≥0.56 doesn't support anyway) | **3.10** |
| D2 | Public API: keep the old functions (`bfs`, `base_table`, `get_values_for_site`, `forecast`, `bfs_calibrate`, `flow_metrics`, skill, plots) with **unchanged signatures and outputs** as wrappers, and add the new API alongside? | **Yes**, so existing scripts and the paper repo keep working |
| D3 | `plot.py` currently defaults to `save=True` (writes PNGs into CWD) and `plt.show()`. Changing the defaults to return the figure is an API behavior change (not numerical) | Change the defaults to `save=False` and no `show`, returning the figure; the examples pass `save=True` explicitly |
| D4 | Hourly mode (`timestep='hour'`): untested, partly inconsistent (M8, M9) | Keep the parameter in the wrappers; the new core raises `NotImplementedError` for hourly input until someone needs it |
| D5 | `use_jit=False` (M3 divergence) | Route to `legacy` for now; decide M3 together with the other fixes |
| D6 | Update `examples/` to the new API? | Leave them on the old API in Phase 3 (they double as compatibility smoke tests); add one new-API example to the README |
| D7 | Upstream intent (research fork vs. eventual PR to BYU-Hydroinformatics) | Doesn't change this plan, but it affects whether `legacy/` should ever be deleted |

## 10. After Phase 3: behavior fixes (each needs your approval)

Not part of Phase 3. Listed so the structure accommodates them, ordered by relevance to calibration:

1. **M10** `sur_z`/`sur_store` inverse (pinned by your decision; fix = one kernel, regenerate references in its own commit).
2. **C1** `cal_surface` no-op and **C2** missing reltol: these determine what the current calibration actually optimizes.
3. **M13** manual error rules (100-day spin-up exclusion, Qthresh gating, Rb1/Rb2 weights): become `ScoreSpec` options rather than hard-coded behavior.
4. **C3** β sweep range, **C11/C12** selection and objective asymmetry (Phase 4 territory).
5. **D1 data** zero days at 12167000, **M3**, **U1** fallbacks, **U8** forecast offset.

## 11. How this sets up Phase 4

| Phase 4 goal | Where it plugs in |
|---|---|
| Swap objective functions | New function in `objectives/`; choose it in the `CalibrationProblem`'s `scores` |
| Compare optimizers | Implement the `Optimizer` protocol; run against the same `CalibrationProblem` |
| Identifiability / equifinality | `ParameterSpec` transforms and bounds + fast stateless `evaluate(x)`, giving cheap sampling and profiles |
| Calibration/validation splits | `ScoreSpec.eval_mask` (simulate over the full record, score on a subset); spin-up exclusion uses the same mechanism |
