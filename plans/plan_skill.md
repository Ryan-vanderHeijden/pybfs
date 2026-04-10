# Plan: `forecast_skill` Function

## Overview

Implement `forecast_skill` in `pybfs/skill.py` to assess how well the pybfs forecast performs during naturally-occurring baseflow-dominated periods. The function identifies stretches of the observed hydrograph where baseflow dominates (high BFI), runs pybfs's `forecast()` from the start of each stretch, and computes error metrics between the forecast and the observed streamflow.

---

## Function Signature

```python
def forecast_skill(hydrograph, SBT, basin_char, gw_hyd, flow,
                   min_days=30, min_sat=0.9, train_days=365):
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `hydrograph` | `pd.DataFrame` | Observed streamflow with columns `Date` (datetime) and `Streamflow` (m³/day) |
| `SBT` | `pd.DataFrame` | Baseflow table from `base_table()` with columns `['Xb','Z','S','Q']` |
| `basin_char` | `list` | `[area, lb, x1, wb, por]` — same as `bfs()` |
| `gw_hyd` | `list` | `[alpha, beta, ks, kb, kz]` — same as `bfs()` |
| `flow` | `list` | `[qthresh, rs, rb1, rb2, prec, fr4rise]` — same as `bfs()` |
| `min_days` | `int` | Minimum number of days a sequence must span to be included (default 30) |
| `min_sat` | `float` | Minimum BFI (Baseflow / Streamflow) required to consider a day as baseflow-dominated (default 0.9) |
| `train_days` | `int` | Number of days of observed data immediately before a sequence used to initialize the forecast (default 365) |

### Returns

`(skill_df, summary_df)` — a tuple of two DataFrames described below.

---

## Output DataFrames

### `skill_df` — daily record for the gage

| Column | Description |
|--------|-------------|
| `Date` | Datetime of each record |
| `Q` | Observed streamflow (m³/day) |
| `BF` | Binary baseflow flag: `1` if `Baseflow[t] / Qob[t] >= min_sat`, else `0` |
| `SEQ` | Sequence number (1-indexed integer) for days inside a valid sequence; `NaN` otherwise |
| `FC` | Forecasted baseflow (m³/day) from `pybfs.forecast()`; `NaN` outside sequences |
| `RES` | Residual `Q - FC`; `NaN` outside sequences |

### `summary_df` — one row per sequence

| Column | Description |
|--------|-------------|
| `SEQ` | Sequence number |
| `LEN` | Length of sequence in days (≥ `min_days`) |
| `SAT` | Actual BFI for that sequence (mean Baseflow / mean Q over the sequence days) |
| `RMSE` | Root Mean Square Error of residuals within the sequence |
| `MAE` | Mean Absolute Error of residuals within the sequence |

---

## Implementation Steps

### Step 1 — Full BFS run

Run `pybfs.bfs(hydrograph, SBT, basin_char, gw_hyd, flow)` on the entire hydrograph to obtain:
- `Baseflow` — pybfs baseflow estimate
- `Qob` — observed streamflow
- `DirectRunoff` — direct runoff component

### Step 2 — Compute BFI and label BF column

```
BFI[t] = Baseflow[t] / Qob[t]
```

A day is baseflow-dominated (`BF = 1`) if `BFI[t] >= min_sat`. All other days: `BF = 0`.

### Step 3 — Identify and number sequences

1. Find contiguous runs of days where `BF == 1`.
2. Discard any run shorter than `min_days` days.
3. Number the surviving runs sequentially starting from `1`.
4. All days outside surviving sequences get `SEQ = NaN`.

### Step 4 — Forecast each sequence

For each sequence `i` with start date `d_start` and end date `d_end`:

1. **Training slice**: extract the `train_days` days of `hydrograph` immediately preceding `d_start`. If fewer than `train_days` are available, use whatever is available (do not skip the sequence).
2. **Train BFS**: run `pybfs.bfs(training_slice, SBT, basin_char, gw_hyd, flow)`.
3. **Extract initial conditions** from the last row of the training BFS result:
   ```python
   ini = (Xi, Zbi, Zsi, StBi, StSi, Surflow, Baseflow, Rech)
   ```
   Fields: `X`, `Zb.L`, `Zs.L`, `StBase`, `StSur`, `SurfaceFlow`, `Baseflow`, `Rech`
4. **Build forecast DataFrame**: a DataFrame with columns `date` (the sequence dates) and `streamflow` (all `NaN`), matching the `pybfs.forecast()` input format.
5. **Run forecast**: `pybfs.forecast(forecast_df, SBT, basin_char, gw_hyd, flow, ini)` → returns `Baseflow` column for each forecast day.
6. **Assign FC**: store the forecasted baseflow into `skill_df.FC` for those dates.

### Step 5 — Compute RES

```
RES[t] = Q[t] - FC[t]
```

For all days inside a sequence; `NaN` elsewhere.

### Step 6 — Build summary_df

For each sequence `i`:
- `LEN` = number of days in the sequence
- `SAT` = mean `Baseflow[t] / Q[t]` over the sequence days (from the full BFS run)
- `RMSE` = `sqrt(mean(RES²))`
- `MAE` = `mean(|RES|)`

Only sequences with `LEN >= min_days` and `SAT >= min_sat` are included (both filters applied during sequence identification in Step 3).

---

## File Location

- **Implementation**: `pybfs/skill.py`
- **Export**: add `forecast_skill` to `pybfs/__init__.py` imports and `__all__`

---

## Usage Example

```python
import pandas as pd
import pybfs

# Load data
hydrograph = pd.read_csv('streamflow.csv')
hydrograph['Date'] = pd.to_datetime(hydrograph['Date'])
params = pd.read_csv('bfs_params.csv')

site_no = 2312200
basin_char, gw_hyd, flow = pybfs.get_values_for_site(params, site_no)

area, lb, x1, wb, por = basin_char
alpha, beta, ks, kb, kz = gw_hyd
SBT = pybfs.base_table(lb, x1, wb, beta, kb, hydrograph, por)

skill_df, summary_df = pybfs.forecast_skill(
    hydrograph, SBT, basin_char, gw_hyd, flow,
    min_days=30, min_sat=0.9, train_days=365
)

print(summary_df)
```

---

## Notes and Edge Cases

- If a sequence starts within the first `train_days` days of the record, use all available prior data (do not skip).
- If `Qob == 0` for a day, skip BFI computation for that day (`BF = 0` by default).
- NaN streamflow values in the hydrograph are treated as non-baseflow days (`BF = 0`).
- The `forecast()` function assumes no precipitation (Impulse = 0), which is appropriate for baseflow-only forecasting during recession periods.
- `FC` represents the forecasted **baseflow** (not total streamflow), so `RES = Q - FC` captures total streamflow minus the baseflow forecast component.
