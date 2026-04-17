# Evolutionary Algorithm Calibration Design

## Overview

`calibrate_ea.py` replaces the legacy sequential Nelder-Mead calibration (`calibrate.py`) with a multi-objective NSGA-II optimization. All 8 physical parameters are optimized simultaneously, returning a Pareto front of solutions that trade off total flow performance against recession dynamics.

---

## Algorithm

**NSGA-II** (Non-dominated Sorting Genetic Algorithm II) via [pymoo](https://pymoo.org/).

| Setting | Value |
|---|---|
| Population size | 200 (default) |
| Generations | 300 (default) |
| Crossover | Simulated Binary (SBX), η=15, p=0.9 |
| Mutation | Polynomial Mutation (PM), η=20 |
| Sampling | Uniform random within bounds |

---

## Parameters

All 8 parameters are optimized in a single stage. Geometric parameters are encoded in log₁₀-space to allow uniform sampling across orders of magnitude; BETA is linear.

| # | Parameter | Description | Search space |
|---|---|---|---|
| 0 | Lb | Basin length (m) | log₁₀, ±2 orders around √area |
| 1 | Wb | Base reservoir width (m) | log₁₀, up to √area |
| 2 | X1 | Horizontal base surface scaling (m) | log₁₀, [1, √area × 100] |
| 3 | ALPHA | Surface hydraulic gradient | log₁₀, [10⁻⁴, 10⁻¹] |
| 4 | BETA | Base surface nonlinearity exponent | Linear, [0.5, 20] |
| 5 | Ks | Surface hydraulic conductivity (m/day) | log₁₀, [10⁻⁸, 10⁵] |
| 6 | Kb | Base hydraulic conductivity (m/day) | log₁₀, [10⁻⁸, 10⁵] |
| 7 | Kz | Vertical hydraulic conductivity (m/day) | log₁₀, [10⁻⁸, 10⁵] |

**Constraint:** `Lb × Wb ≤ basin_area` — enforced as a pymoo inequality constraint; NSGA-II deprioritizes infeasible solutions automatically.

Porosity (POR = 0.15) is fixed and not calibrated.

---

## Objectives

### F₁ — KGE Loss (total flow)

$$F_1 = 1 - \text{KGE} = \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}$$

where $r$ is Pearson correlation, $\alpha = \sigma_{sim}/\sigma_{obs}$, and $\beta = \mu_{sim}/\mu_{obs}$ between simulated (`Qsim`) and observed (`Qob`) streamflow. Lower is better.

### F₂ — Recession Error (recession dynamics)

Mean absolute error between the modeled baseflow log-recession rate and the empirical recession regression derived by `flow_metrics`:

$$F_2 = \text{mean}\left(\left|\log_{10}\frac{BF_t}{BF_{t-1}} - (r_{b1} + r_{b2} \cdot \log_{10} Q_t)\right|\right)$$

Evaluated only on days identified as recession periods (`RecessCount.T > 0`) where `Q > Qthresh` and consecutive baseflow values are positive. $r_{b1}$ and $r_{b2}$ are the intercept and slope of the 10th-percentile quantile regression of 10-day recession rates on log-streamflow (from `flow_metrics`). Lower is better.

---

## Output

### Pareto Front

A DataFrame of all non-dominated solutions, with columns:

`tmp.site`, `tmp.area`, `Lb`, `Wb`, `X1`, `ALPHA`, `BETA`, `Ks`, `Kb`, `Kz`, `POR`, `Qthresh`, `Rs`, `Rb1`, `Rb2`, `Prec`, `Frac4Rise`, **`KGE`**, **`RecessionError`**

### Knee Point

The balanced trade-off solution selected from the Pareto front using the **Achievement Scalarization Function (ASF)** with equal weights on both normalized objectives. Returned as a single-row DataFrame matching the `bfs_calibrate` output format, plus `KGE` and `RecessionError` columns.

### BFS Output

Full `bfs()` output DataFrame for the knee-point parameter set, identical in structure to the output from the legacy calibration.

---

## Parallelism

Each objective function evaluation is one independent `bfs()` call. Population members within a generation are evaluated in parallel using `multiprocessing.Pool.starmap` via pymoo's `StarmapParallelization`. Defaults to `cpu_count - 1` workers.

---

## Usage

```python
from pybfs import bfs_calibrate_nsga2

pareto_df, knee_params, bfs_out = bfs_calibrate_nsga2(
    tmp_site="site_01",
    tmp_area=1.5e8,        # m²
    tmp_q=streamflow,      # m³/day
    dys=dates,
    pop_size=200,
    n_gen=300,
    seed=42,               # for reproducibility
    n_jobs=8,              # worker processes (None = cpu_count - 1)
)
```

---

## Comparison with Legacy Calibration

| Aspect | Legacy (`bfs_calibrate`) | NSGA-II (`bfs_calibrate_nsga2`) |
|---|---|---|
| Algorithm | Nelder-Mead (sequential, 3 steps) | NSGA-II (single global search) |
| Objectives | Single (weighted APE) | Two (KGE + recession error) |
| Parameter stages | Step-wise with fixed BETA scan | All 8 simultaneously |
| Output | Single best-fit parameter set | Full Pareto front + knee point |
| Parallelism | None | Multiprocessing across population |
| Runtime | Fast (~seconds) | Slow (~minutes, infrequent use) |
| Local optima risk | High (gradient-free but local) | Low (population-based global search) |
