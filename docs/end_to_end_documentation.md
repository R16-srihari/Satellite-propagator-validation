# End-to-End Documentation

This document consolidates the repository's end-to-end usage and module details.

1. End-to-End Program: leo_simulator.py

1.1 What It Does

`leo_simulator.py` is the primary simulation pipeline. It:

- Loads orbit parameters (defaults or STK_input/Satellite1.opm when available)
- Converts Keplerian elements to ECI Cartesian state
- Integrates trajectory over 24 hours
- Exports Cartesian states, orbital elements, and conservation metrics to CSV
- Runs validation routines (energy, angular momentum, analytical comparison)
- Produces conservation plots using analytical-state baselines for error visualization
- Mirrors terminal logs to `output/terminal_log.txt`

1.2 Python Run CLI Modes (`leo_simulator.py`)

This script provides three execution modes:

- Script execution:

```bash
python leo_simulator.py
```

- Module execution:

```bash
python -m leo_simulator
```

- Programmatic execution:

```bash
python -c "from leo_simulator import run_simulation; run_simulation()"
```

All three modes run the same simulation workflow and write outputs to `output/`.

1.3 Generated Outputs

- `output/orbit_cartesian.csv`
- `output/orbit_elements.csv`
- `output/orbit_energy.csv`
- `output/energy_conservation.csv`
- `output/angular_momentum_conservation.csv`
- `output/analytical_comparison.csv`
- `output/energy_vs_time.png`
- `output/angular_momentum_vs_time.png`
- `output/terminal_log.txt`

2. STK Post-Processing Program: STK_output_energy_angular_momentum_errors.py

2.1 What It Does

This script reads an STK results CSV (default: `STK_input/Satellite1_Results.csv`) and extracts:

- `Delaunay_G (km^2/sec)` (specific angular momentum proxy)
- `Semi-major Axis (km)`

It computes specific orbital energy using `epsilon = -mu / (2a)` (where `mu` is Earth's gravitational parameter) and generates:

- Angular momentum error plot (`h - mean(h)`)
- Energy error plot (`epsilon - mean(epsilon)`)
- Optional CSV of time, angular momentum, and computed energy

2.2 CLI Interface

Run help:

```bash
python STK_output_energy_angular_momentum_errors.py --help
```

Supported options include positional `csv_path` and flags `--output-h`, `--output-e`, `--output-csv`, and `--show`.

3. Core Module Documentation (src/)

3.1 `src/constants.py`

Defines immutable physical and mathematical constants via a `Constants` dataclass and factory, including Earth's gravitational parameter and unit conversions.

3.2 `src/orbital_parameters.py`

Builds an `OrbitParameters` dataclass from defaults or an OPM file, parsing Keplerian elements and computing derived quantities such as mean motion, period, specific energy, and angular momentum.

3.3 `src/gravity_ode.py`

Defines the two-body state derivative for `[x,y,z,vx,vy,vz]` → `[vx,vy,vz,ax,ay,az]` with central gravity acceleration.

3.4 `src/eci_from_keplerian.py`

Converts Keplerian elements to ECI Cartesian vectors by computing perifocal position/velocity and applying rotation sequence (RAAN, inclination, argument of perigee).

3.5 `src/keplerian_from_eci.py`

Converts ECI Cartesian state back to Keplerian elements, computing semi-major axis, eccentricity, inclination, RAAN, argument of perigee, and true anomaly with numerical safeguards for edge cases.

3.6 `src/analytical_solution.py`

Computes a reference Keplerian state at a requested time. Uses a circular-orbit shortcut when applicable, or solves Kepler's equation for elliptic cases and converts to ECI.

3.7 `src/rk78_integrate.py`

Standalone adaptive Runge-Kutta 8(7) integrator (DOP853-style) that implements embedded error control and adaptive step-size selection. Accepts options such as `RelTol`, `AbsTol`, `MaxStep`, and `InternalStep`, and returns sampled states plus integration statistics.

3.8 `src/export_results.py`

Creates simulation result artifacts: Cartesian history CSV, Keplerian elements CSV (via `keplerian_from_eci`), energy and angular momentum time series CSVs, and console summaries.

4. Validation Module Documentation (validation/)

4.1 `validation/energy_check.py`

Computes and reports conservation performance:

- Specific orbital energy drift relative to the initial numerical state
- Specific angular momentum drift relative to the initial numerical state
- Prints analytical expected specific energy from orbit parameters for comparison
- Writes `energy_conservation.csv` and `angular_momentum_conservation.csv`

4.2 `validation/compare_analytical.py`

Compares numerical trajectory with the analytical two-body reference, reporting position error (max/mean/RMS), relative error, and period sanity checks. Outputs `analytical_comparison.csv`.

4.3 `validation/plot_conservation.py`

Generates diagnostics plots for energy and angular momentum vs time, including zoomed error panels using analytical references. Outputs PNGs under `output/`.

5. Data and Artifact Notes

5.1 STK input data

`STK_input/Satellite1_Results.csv` is intentionally not tracked in git. Maintain canonical STK input as a release asset using `scripts/update_release_asset.sh` or `scripts/update_release_asset.ps1`.

5.2 Runtime output

Simulation and post-processing artifacts are written to `output/` by default.

5.3 Typical Workflows (quick commands)

Run the complete simulation and validation:

```bash
python leo_simulator.py
```

Generate STK conservation error figures from default input:

```bash
python STK_output_energy_angular_momentum_errors.py
```

Use a custom STK file and show plots interactively:

```bash
python STK_output_energy_angular_momentum_errors.py /absolute/path/to/Satellite1_Results.csv --show
```
