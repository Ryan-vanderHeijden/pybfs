# PyBFS

PyBFS - Python Baseflow Separation

A Python implementation of physically-based Baseflow Separation algorithms for hydrological analysis. PyBFS uses a coupled surface-subsurface reservoir model to separate total streamflow into surface flow, baseflow, and direct runoff components.

## Features

- Physically-based baseflow separation using coupled reservoir models
- Forecast baseflow using initial conditions
- Comprehensive visualization tools
- Well-documented API with NumPy-style docstrings

## Installation

### From PyPI (recommended)

```bash
pip install pybfs
```

### From source

```bash
git clone https://github.com/BYU-Hydroinformatics/pybfs.git
cd pybfs
pip install .
```

### Development installation

```bash
git clone https://github.com/BYU-Hydroinformatics/pybfs.git
cd pybfs
pip install -e .
```

## Quick Start

```python
import pandas as pd
from pybfs import get_values_for_site, base_table, PyBFS, plot_baseflow_simulation

# Load streamflow data
streamflow_data = pd.read_csv('streamflow.csv')

# Load site parameters
params_df = pd.read_csv('site_parameters.csv')

# Get parameters for specific site
site_number = 2312200
basin_char, gw_hyd, flow = get_values_for_site(params_df, site_number)

# Extract parameters
area, lb, x1, wb, por = basin_char[0], basin_char[1], basin_char[2], basin_char[3], basin_char[4]
alpha, beta, ks, kb, kz = gw_hyd[0], gw_hyd[1], gw_hyd[2], gw_hyd[3], gw_hyd[4]

# Generate baseflow table
SBT = base_table(lb, x1, wb, beta, kb, streamflow_data, por)

# Run PyBFS baseflow separation
result = PyBFS(streamflow_data, SBT, basin_char, gw_hyd, flow)

# Visualize results
plot_baseflow_simulation(streamflow_data, result)
```

## Package layout (refactored core)

This fork restructures the internals around a small, pure model core. The
pybfs 0.2.0 functions shown above still work with the same signatures and
give the same results (bit for bit; see `tests/`), but they are now thin
wrappers. Interface changes: plot functions return the figure and no longer
save or show it unless asked (`save=True, show=True`); `use_jit` is ignored;
hourly time steps raise `NotImplementedError`. The original implementation is
the git tag `baseline-original`.

```
pybfs/
  params.py              ModelParameters, FlowMetrics, PARAMETERS (names, symbols, units, manual definitions)
  data.py                Parameter tables and streamflow input (the only file/pandas-input module)
  streamflow_metrics.py  flow_metrics(): Qthresh, Rs, Rb1, Rb2, Prec from a record
  core/                  The model, on plain arrays (no pandas, no I/O, no module state)
    geometry.py            Surface-reservoir kernels, cross-referenced to the USGS manual equations
    base_table.py          Base-reservoir storage-discharge table and lookups
    hydrograph.py          Rise/recession classification of the record
    model.py               simulate(): the forward model
    forecasting.py         forecast() from a ModelState
    results.py             SimulationResult, ModelState, ForecastResult
  objectives/            Objective functions (pure); usgs.py holds the USGS procedure's measures
  calibration/           CalibrationProblem/ScoreSpec, optimizers, and the USGS 3-step procedure (usgs.calibrate)
  diagnostics.py         Flow fractions, credible intervals, separation/forecast skill
  frames.py              pandas views of results (new and pybfs 0.2.0 column layouts)
  plot.py                Plotting
  bfs.py, utilities.py, calibrate.py, skill.py   pybfs 0.2.0 interface (wrappers)
```

Units follow the input: the examples use m, m³/day, and a daily time step.
`notes/` documents the original code, the variable glossary, the comparison
with the USGS manual, and suspected issues (none of which are fixed yet).

### Minimal example

```python
import pandas as pd
from pybfs import read_parameter_table, site_parameters, simulate
from pybfs.calibration import usgs

streamflow = pd.read_csv("docs/files/2312200_data.csv", parse_dates=["Date"], date_format="%m/%d/%Y")
params, flow = site_parameters(read_parameter_table("docs/files/bfs_params_50.csv"), 2312200)

result = simulate(streamflow["Streamflow"].to_numpy(), params, flow)
print(result.baseflow[:3], result.q_sim[1:3])      # m³/day; arrays, one value per day
state = result.final_state()                      # start for pybfs.core.forecasting.forecast

calibrated = usgs.calibrate("2312200", params.drainage_area,
                            streamflow["Streamflow"].to_numpy(), streamflow["Date"])
print(calibrated.params, calibrated.baseflow_fraction)
```

The USGS procedure's error rules are options: `usgs.calibrate(..., scoring=usgs.MANUAL_SCORING)`
applies the manual's 100-day spin-up exclusion, Qthresh gating, and Rb1/Rb2
weights (`usgs.UsgsScoring` sets them individually; the default reproduces pybfs 0.2.0).

To try another objective, wrap a function `objective(q_obs, q_sim, weights) -> float`
in a `pybfs.calibration.problem.ScoreSpec` (choosing the series, weights, and
the days to score) and pass it as the `score` of a `CalibrationProblem`.

## Requirements

- Python 3.10 or higher
- numpy, pandas, matplotlib, numba, scipy, statsmodels (see `pyproject.toml`)

## Documentation

Full documentation is available at: [Documentation](https://github.com/BYU-Hydroinformatics/pybfs)

## Paper data and reproduction scripts

The data and analysis scripts for the streamflow partitioning paper
(Webster-Esho et al., manuscript under review) live in a separate repository:
https://github.com/EniolaWebsterEsho/streamflow_separation_paper_2026

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/contributing.md) for guidelines.

## Citation

If you use PyBFS in your research, please cite:

```
PyBFS - Python Baseflow Separation
BYU Hydroinformatics
https://github.com/BYU-Hydroinformatics/pybfs
```

## Support

For issues, questions, or contributions, please visit: https://github.com/BYU-Hydroinformatics/pybfs/issues
