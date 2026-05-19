# RK-7-8 Integrator Validation for LEO Orbit Propagation

## 1. Overview

This repository simulates and validates a 24-hour low Earth orbit (LEO) trajectory using a two-body gravitational model and a custom integration wrapper (`rk78_integrate`) built on SciPy's high-order adaptive solver (`DOP853`).

The project has two main executable Python programs:

1. `leo_simulator.py` — runs end-to-end orbit propagation + validation + exports.
2. `STK_output_energy_angular_momentum_errors.py` — post-processes STK CSV output to plot conservation-error trends.

Core assumptions:

- Dynamics: point-mass Earth gravity only (no drag, J2, SRP, third-body perturbations)
- Typical initial orbit: ~450 km circular LEO, inclination ~51.6°
- Main simulation horizon: 24 hours

---

## 2. Repository Layout

```text
RK-7-8-integrator-validation/
├── leo_simulator.py
├── STK_output_energy_angular_momentum_errors.py
├── requirements.txt
├── README.md
├── STK_input/
│   ├── Satellite1.opm
│   └── Satellite1_Results.csv (canonical file distributed as release asset; see section 8)
├── src/
│   ├── constants.py
│   ├── orbital_parameters.py
│   ├── gravity_ode.py
│   ├── eci_from_keplerian.py
│   ├── keplerian_from_eci.py
│   ├── analytical_solution.py
│   ├── rk78_integrate.py
│   └── export_results.py
├── validation/
│   ├── energy_check.py
│   ├── compare_analytical.py
│   └── plot_conservation.py
├── scripts/
│   ├── update_release_asset.sh
│   └── update_release_asset.ps1
└── output/
```

---

## 3. Environment Setup

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Main packages:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

---

## 4. End-to-End Program: `leo_simulator.py`

### 4.1 What It Does

`leo_simulator.py` is the primary simulation pipeline. It:

1. Loads orbit parameters (defaults or `STK_input/Satellite1.opm` when available)
2. Converts Keplerian elements to ECI Cartesian state
3. Integrates trajectory over 24 hours
4. Exports Cartesian states, orbital elements, and conservation metrics to CSV
5. Runs validation routines (energy, angular momentum, analytical comparison)
6. Produces conservation plots
7. Mirrors terminal logs to `output/terminal_log.txt`

### 4.2 Python Run CLI Modes (`leo_simulator.py`)

This script has **no argparse flags**; it runs in a single simulation mode.

#### Mode A — Script execution (standard)

```bash
python leo_simulator.py
```

#### Mode B — Module execution

```bash
python -m leo_simulator
```

#### Mode C — Programmatic execution

```bash
python -c "from leo_simulator import run_simulation; run_simulation()"
```

> All three modes execute the same simulation workflow and generate outputs under `output/`.

### 4.3 Generated Outputs

- `output/orbit_cartesian.csv`
- `output/orbit_elements.csv`
- `output/orbit_energy.csv`
- `output/energy_conservation.csv`
- `output/angular_momentum_conservation.csv`
- `output/analytical_comparison.csv`
- `output/energy_vs_time.png`
- `output/angular_momentum_vs_time.png`
- `output/terminal_log.txt`

---

## 5. STK Post-Processing Program: `STK_output_energy_angular_momentum_errors.py`

### 5.1 What It Does

This script reads an STK results CSV (default: `STK_input/Satellite1_Results.csv`), extracts:

- `Delaunay_G (km^2/sec)` (specific angular momentum proxy)
- `Semi-major Axis (km)`

Then it computes specific orbital energy:

- `epsilon = -mu / (2a)`, where `mu` is Earth's gravitational parameter

and generates:

- angular momentum error plot (`h - mean(h)`)
- energy error plot (`epsilon - mean(epsilon)`)
- optional CSV of time, angular momentum, and computed energy

### 5.2 Python Run CLI Modes (`STK_output_energy_angular_momentum_errors.py`)

Run help:

```bash
python STK_output_energy_angular_momentum_errors.py --help
```

Supported CLI interface:

- Positional argument:
  - `csv_path` (optional): path to STK results CSV
- Optional flags:
  - `--output-h`: output path for angular momentum error PNG
  - `--output-e`: output path for energy error PNG
  - `--output-csv`: output path for computed time-series CSV
  - `--show`: use interactive display for plots

#### Mode A — Default headless run

```bash
python STK_output_energy_angular_momentum_errors.py
```

Uses default input path and saves outputs under `output/`.

#### Mode B — Custom STK CSV input

```bash
python STK_output_energy_angular_momentum_errors.py /absolute/path/to/Satellite1_Results.csv
```

#### Mode C — Custom output locations

```bash
python STK_output_energy_angular_momentum_errors.py \
  --output-h /absolute/path/STK_h_error.png \
  --output-e /absolute/path/STK_e_error.png \
  --output-csv /absolute/path/STK_energy_series.csv
```

#### Mode D — Interactive plotting mode

```bash
python STK_output_energy_angular_momentum_errors.py --show
```

Without `--show`, a non-interactive backend (`Agg`) is used.

#### Mode E — Fully customized run

```bash
python STK_output_energy_angular_momentum_errors.py /absolute/path/to/Satellite1_Results.csv \
  --output-h /absolute/path/STK_h_error.png \
  --output-e /absolute/path/STK_e_error.png \
  --output-csv /absolute/path/STK_energy_series.csv \
  --show
```

---

## 6. Core Module Documentation (`src/`)

### `src/constants.py`

Defines immutable physical and mathematical constants via the `Constants` dataclass and `constants()` factory:

- Earth gravitational parameter `mu_earth`
- Earth radii variants
- angle conversion constants
- time conversion constants

### `src/orbital_parameters.py`

Builds `OrbitParameters` dataclass from defaults and optional OPM input:

- Reads `STK_input/Satellite1.opm` when available
- Parses OPM fields (semi-major axis, eccentricity, inclination, RAAN, argument of perigee, true anomaly, epoch)
- Computes derived quantities (`n`, orbital period, circular speed, specific energy, angular momentum magnitude)

### `src/gravity_ode.py`

Defines the state derivative for two-body dynamics:

- Input state: `[x, y, z, vx, vy, vz]`
- Output derivative: `[vx, vy, vz, ax, ay, az]` with central gravity acceleration

### `src/eci_from_keplerian.py`

Converts Keplerian elements to Cartesian ECI vectors:

- Computes perifocal position/velocity
- Applies rotation sequence (RAAN, inclination, argument of perigee)
- Returns `(r_vec, v_vec)`

### `src/keplerian_from_eci.py`

Converts ECI Cartesian state back to Keplerian elements:

- Computes energy-based semi-major axis
- Computes eccentricity, inclination, RAAN, argument of perigee, true anomaly
- Includes numerical safeguards for near-circular/equatorial edge cases

### `src/analytical_solution.py`

Computes reference Keplerian state at a requested time:

- Circular orbit shortcut (`nu = nu0 + nt`)
- Elliptic case via Kepler equation + Newton iteration
- Converts back to ECI using `eci_from_keplerian`

### `src/rk78_integrate.py`

Integration wrapper with output-time sampling and summary stats:

- Uses `scipy.integrate.solve_ivp` with `method="DOP853"`
- Accepts options (`RelTol`, `AbsTol`, `MaxStep`)
- Enforces strictly monotonic `t_eval`
- Returns sampled states plus `RK78Stats`

### `src/export_results.py`

Creates simulation result artifacts:

- Cartesian state history CSV
- Keplerian element history CSV (via `keplerian_from_eci`)
- Energy and angular momentum trend CSV
- Console summary statistics

---

## 7. Validation Module Documentation (`validation/`)

### `validation/energy_check.py`

Computes and reports conservation performance:

- specific orbital energy drift
- specific angular momentum drift
- pass/fail-like status based on relative error thresholds
- outputs:
  - `energy_conservation.csv`
  - `angular_momentum_conservation.csv`

### `validation/compare_analytical.py`

Compares numerical trajectory with analytical two-body reference:

- position error (max/mean/RMS)
- relative error
- period sanity check from perigee-like crossings
- output:
  - `analytical_comparison.csv`

### `validation/plot_conservation.py`

Generates visualization diagnostics:

- energy vs time + zoomed error panel
- angular momentum vs time + zoomed error panel
- outputs:
  - `energy_vs_time.png`
  - `angular_momentum_vs_time.png`

---

## 8. Data and Artifact Notes

### STK input data

`STK_input/Satellite1_Results.csv` is intentionally not tracked in git history to keep the repository lightweight.

Maintainers can upload the canonical file to release assets with:

- `scripts/update_release_asset.sh`
- `scripts/update_release_asset.ps1`

These scripts use GitHub CLI (`gh`) and configurable `OWNER`, `REPO`, `TAG` values.

### Runtime output

Simulation and post-processing artifacts are written to `output/` by default.

---

## 9. Typical Workflows

### Run complete simulation + validation

```bash
python leo_simulator.py
```

### Generate STK conservation error figures from default input

```bash
python STK_output_energy_angular_momentum_errors.py
```

### Use custom STK file and show plots interactively

```bash
python STK_output_energy_angular_momentum_errors.py /absolute/path/to/Satellite1_Results.csv --show
```

---

## 10. Limitations and Scope

- Two-body model only; no perturbations
- Single-body Earth-centered dynamics
- No uncertainty modeling or estimation framework
- Validation is focused on conservation and analytical consistency, not operational navigation performance
