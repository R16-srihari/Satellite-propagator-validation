# RK-7-8 Integrator Validation for LEO Orbit Propagation

## 1. Overview

This repository simulates and validates a 24-hour low Earth orbit (LEO) trajectory using a two-body gravitational model and a custom self-implemented adaptive DOP853 / RK7(8) integrator (`rk78_integrate`).

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

### 3.1 STK input files

The simulation expects the STK input files in `STK_input/`:

- `Satellite1_Results.csv`
- `Satellite1.opm`

If either file is missing, download the release assets from GitHub Releases and place them in `STK_input/`.

Recommended download workflow with GitHub CLI:

```bash
gh release download stk-latest --repo R16-srihari/RK-7-8-integrator-validation -p Satellite1_Results.csv -p Satellite1.opm -D STK_input
```

You can also download the two assets manually from the repository's Releases page and copy them into `STK_input/`.

### 3.2 Updating the release assets

Use the provided scripts to publish updated STK files to the `stk-latest` release. Both scripts upload `STK_input/Satellite1_Results.csv` and `STK_input/Satellite1.opm` when present, replacing any existing assets with the same name.

PowerShell:

```powershell
.\scripts\update_release_asset.ps1 -Owner <owner> -Repo RK-7-8-integrator-validation -Tag stk-latest
```

Bash:

```bash
OWNER=<owner> REPO=RK-7-8-integrator-validation TAG=stk-latest ./scripts/update_release_asset.sh
```

---

## 4. Limitations and Scope

- Two-body model only; no perturbations
- Single-body Earth-centered dynamics
- No uncertainty modeling or estimation framework
- Validation is focused on conservation and analytical consistency, not operational navigation performance
