# LEO Satellite Orbit Simulator

## Overview

This repository contains a Python implementation of a 24-hour LEO orbit propagation for a two-body Earth model. The code uses a custom fixed-substep RK7/8 (Dormand-Prince style) integrator and exports simulation and validation data to CSV.

- Simulation duration: 24 hours
- Integrator: custom RK7/8 stepping routine
- Orbit: circular LEO, 450 km altitude, 51.6 degree inclination
- Dynamics: central gravity only

## Features

- Two-body orbit propagation in ECI coordinates
- Keplerian <-> Cartesian conversion utilities
- CSV exports for Cartesian states, orbital elements, and conservation metrics
- Validation routines for energy/angular-momentum conservation and analytical comparison
- Terminal log output saved to output/terminal_log.txt

## Requirements

- Python 3.10+
- Packages from requirements.txt

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python leo_simulator.py
```

This run will:

1. Build initial conditions from Keplerian elements.
2. Propagate the trajectory for 24 hours.
3. Export CSV files under output/.
4. Run validation checks and print summary statistics.

## Output Files

- output/orbit_cartesian.csv: time, position, velocity
- output/orbit_elements.csv: time, a/e/i/Omega/omega/nu
- output/orbit_energy.csv: time, specific energy, relative energy drift, |h|
- output/energy_conservation.csv: detailed energy conservation metrics
- output/angular_momentum_conservation.csv: angular momentum conservation metrics
- output/analytical_comparison.csv: numerical vs analytical position comparison
- output/terminal_log.txt: terminal output for the full run

## Project Structure

```text
leo_sat_simulator/
├── leo_simulator.py
├── requirements.txt
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
│   └── compare_analytical.py
└── output/
```

## Dynamics Model

The propagated state is:

dr/dt = v
dv/dt = -(mu / r^3) r

with Earth gravitational parameter mu = 3.986004418e14 m^3/s^2.

## Notes

- The current model intentionally excludes drag, J2, SRP, and third-body perturbations.
- The integrator is implemented to match the original project structure and output flow.

## License

Use and modify for research and educational work.

## STK Input Data and Release Asset

The canonical STK input file `STK_input/Satellite1_Results.csv` is intentionally not tracked in the repository. To keep the repository history small we publish a canonical copy as a GitHub Release asset and provide a small update script for maintainers.

Maintainer workflow:

- Replace the local file `STK_input/Satellite1_Results.csv` when you rerun STK.
- Run `scripts/update_release_asset.sh` (or `scripts/update_release_asset.ps1`) to upload the file to the configured GitHub release tag. The scripts use `gh` by default; set `OWNER`, `REPO`, and `TAG` at the top of the script or export them as environment variables.

Collaborator workflow:

- Download the canonical file from the release page: https://github.com/<OWNER>/<REPO>/releases/tag/<TAG>
- Or use the provided download helper (not included by default) to fetch the asset into `STK_input/`.

Notes:

- These scripts require the GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`). The PowerShell script can also use a `GITHUB_TOKEN` environment variable as a fallback.
- For the first release, create a release in GitHub (via web UI or `gh release create`) and use the chosen `TAG` when running the update script.
