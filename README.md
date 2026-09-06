# Code repository for "Resource supply dynamics control stability and chaos in complex ecosystems", doi: 10.64898/2026.09.01.748502

## Repository structure

### `consumer_resource_modules/`
Core consumer-resource model (CRM) classes and shared utilities, used by everything else in this repo.
- `models.py` — the CRM classes themselves (self-limiting, externally-supplied, hybrid, self-inhibiting, multi-trophic and "leached" resource supply), plus the `Consumer_Resource_Model()` wrapper used to construct them.
- `parameters.py` — generates growth/consumption rates and other model parameters (e.g. the `rho`-correlated growth-consumption coupling).
- `differential_equations.py` — shared simulation/ODE-solving logic mixed into the model classes.
- `initial_abundances.py` — generates initial species/resource abundances for simulations.
- `community_level_properties.py` — post-simulation community statistics (survival fractions, moments, the resource-sensitivity/"cusp" diagnostic, max. Lyapunov exponent via `max_le`).
- `effective_LV_models.py` — maps CRM communities onto an effective Lotka-Volterra (eLV) representation.

### `cavity_method_functions/`
- `self_consistency_equation_functions.py` — helpers (`parameter_combinations`, `variable_fixed_parameters`) for building parameter grids for the simulation sweeps below. The cavity-method self-consistency-equation solvers themselves are solved in Mathematica (see `cavity_solutions/`), not here.

### `external_resource_stability/`
Scripts specific to this paper's simulations and figures.
- `simulation_functions_new.py` — runs a CRM across a grid of parameter combinations, saves the resulting communities, and builds summary dataframes from saved simulation data.
- `stability_transitions/` — scripts that sweep model parameters (growth-consumption correlation `rho`, rate heterogeneity `sigma`, resource influx/outflux, etc.) and save the resulting simulation data:
  - `rho_vs_sigma_es_vs_sl.py` — rho vs. sigma, comparing externally-supplied, self-limiting and hybrid resource supply.
  - `es_largeM.py` — the same rho vs. sigma sweep at a larger resource pool size, to check for finite-size effects.
  - `hybrid_influx_vs_rho.py` — resource influx vs. rho for the hybrid model.
  - `external_supply_chaos.py` — sweeps chemostat turnover (influx = outflux) rate towards zero.
  - `estimating_sensitivities.py` — tracks the resource-sensitivity ("cusp") diagnostic against influx and rho.
- `figures/` — scripts that load the saved simulation data and produce each paper figure (`figure_2.py`, `figure_3.py`, `figure_s1.py`, `figure_s2.py`).
- `cavity_solutions/solve_sces_mathematica.nb` — Mathematica notebook that solves the cavity-method self-consistency equations used in the figures.

### `Data/`, `Figures/`
Empty local output directories (populated by running the scripts above). Not checked into git beyond a `.gitkeep`, since simulation output is too large to store here — regenerate it by running the scripts, or download it separately.

All scripts resolve the repository root at runtime, so they can be run from any clone of this repository without editing any paths.
