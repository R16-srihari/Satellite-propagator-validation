# Satellite-propagator-validation for LEO Orbit Propagation

## 1. Overview

This repository simulates and validates a 24-hour low Earth orbit (LEO) trajectory using a two-body gravitational model and custom integrators: an adaptive DOP853-coefficient RK7(8)-style stepper (`pd853_integrate.py`) and a fully implicit, fixed-step symplectic Gauss–Legendre integrator (`symplectic_integrate.py`).

The project has three main executable Python programs:

1. `leo_simulator.py` — runs end-to-end orbit propagation + validation + exports.
2. `validation/compare_analytical.py` — generates comparison plots and CSV summaries against the analytical and STK references.
3. `STK_output_energy_angular_momentum_errors.py` — post-processes STK CSV output to plot conservation-error trends.

Core assumptions:

- Dynamics: point-mass Earth gravity only (no drag, J2, SRP, third-body perturbations)
- Typical initial orbit: ~450 km circular LEO, inclination ~51.6°
- Main simulation horizon: 24 hours

Initialization behavior in the current `main` branch:

- The simulator prefers OPM Cartesian fields (`X`, `Y`, `Z`, `X_DOT`, `Y_DOT`, `Z_DOT`) from `STK_input/Satellite1.opm` to build the propagated initial state `y0 = [r, v]` directly.
- If Cartesian fields are unavailable, it falls back to a Keplerian reconstruction path.
- `OrbitParameters` is retained as a lightweight summary object for reporting, analytical comparison setup, and CSV summary metrics.

---

## 2. Repository Layout

```text
Satellite-propagator-validation/
├── leo_simulator.py
├── STK_output_energy_angular_momentum_errors.py
├── requirements.txt
├── README.md
├── docs/
│   └── Satellite-propagator-validation.wiki/
│       └── End‐to‐End-Documentation.md
├── STK_input/
│   ├── Satellite1.opm
│   ├── Satellite1.opm.example
│   ├── Satellite1_Results.csv
|   └── Satellite1_Results.csv.example
├── src/
│   ├── __init__.py
│   ├── analytical_solution.py
│   ├── constants.py
│   ├── eci_from_keplerian.py
│   ├── export_results.py
│   ├── gravity_ode.py
│   ├── keplerian_from_eci.py
│   ├── orbital_parameters.py
│   ├── pd853_integrate.py
│   └── symplectic_integrate.py
├── validation/
│   ├── __init__.py
│   ├── compare_analytical.py
│   └── plot_conservation.py
├── tests/
│   ├── test_analytical_solution.py
│   ├── test_comparative_plots.py
│   ├── test_comparison_plots.py
│   └── test_orbital_parameters.py
└── output/
```

> **Note:** `STK_input/Satellite1.opm` and `STK_input/Satellite1_Results.csv` are gitignored (listed in `.gitignore`). You must place them manually in `STK_input/` before running the simulation. Example files with the expected format are provided as `Satellite1.opm.example` and `Satellite1_Results.csv.example`.

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

### 3.1 STK input files

The simulation expects the STK input files in `STK_input/`:

- `Satellite1_Results.csv` — STK-generated trajectory results (position, velocity, orbital elements)
- `Satellite1.opm` — Orbit Parameter Message file with initial state

**These files must be placed manually in `STK_input/`.** You can download these files from the latest release assets (https://github.com/R16-srihari/Satellite-propagator-validation/releases/latest) or generate them from your own STK simulation.

You can also copy local working files into `STK_input/` if you are regenerating them from a simulation run.

### 3.2 Validation comparison plots

The validation plot workflow is available as a standalone module:

```bash
python -m validation.compare_analytical --show
```

By default this writes the comparison CSVs and PNGs under `output/pd853/STKcomparison/` (integrator defaults to `pd853`). The CLI flags control the output base directory, integrator subdirectory, STK reference file, and orbit parameters file:

```bash
python -m validation.compare_analytical \
  --output-dir output \
  --integrator symplectic \
  --stk-csv STK_input/Satellite1_Results.csv \
  --orbit STK_input/Satellite1.opm \
  --show
```

Generated artifacts:

- `comparison_errors.csv` — integrator-vs-analytical time-series errors (position, velocity, angular momentum, energy)
- `stk_comparison_errors.csv` — STK-vs-analytical time-series errors (written when the STK CSV is available)
- PNG error plots for position and velocity components/magnitudes, specific angular momentum, and specific orbital energy (integrator curve overlaid with the STK curve when available)

`leo_simulator.py` invokes this automatically after each run, so the plots and CSVs appear in `output/<integrator>/STKcomparison/` without any extra steps.

---

## 4. Limitations and Scope

- Two-body model only; no perturbations
- Single-body Earth-centred dynamics
- No uncertainty modelling or estimation framework
- Validation is focused on conservation and analytical consistency, not operational navigation performance
