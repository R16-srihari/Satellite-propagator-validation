# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.11
# ---

# %% [markdown]
# # LEO Orbit Propagation Demo
#
# This notebook walks through the repository's main propagation workflow in one place:
# - load the OPM-based initial state from `STK_input/Satellite1.opm`
# - run the adaptive RK7(8)-style integrator
# - optionally run the symplectic Velocity Verlet integrator
# - export CSV outputs and validate against the analytical two-body solution
#
# The notebook is paired with Jupytext for easier version control. Edit this `.py` source or the `.ipynb` file and keep them synchronized.

# %%
from contextlib import contextmanager
from pathlib import Path
import builtins
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

repo_root = Path.cwd()
while repo_root != repo_root.parent and not (repo_root / "leo_simulator.py").exists():
    repo_root = repo_root.parent

if not (repo_root / "leo_simulator.py").exists():
    raise FileNotFoundError("Could not locate the repository root from the current working directory.")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.constants import constants
from src.eci_from_keplerian import eci_from_keplerian
from src.export_results import export_results
from src.gravity_ode import gravity_ode
from src.orbital_parameters import orbital_parameters, read_opm_cartesian_state
from src.rk78_integrate import rk78_integrate
from src.symplectic_integrate import symplectic_integrate
from validation.compare_analytical import compare_analytical
from validation.plot_conservation import plot_conservation

plt.style.use("seaborn-v0_8-whitegrid")

# %% [markdown]
# ## 1. Load the initial orbit
#
# The repository prefers the Cartesian state embedded in the OPM file. If those fields are not available,
# it falls back to the Keplerian elements parsed from the same file.

# %%
const = constants()
orbit = orbital_parameters(verbose=True)
cartesian_state = read_opm_cartesian_state()

if cartesian_state is not None:
    r0, v0 = cartesian_state
    initial_state_source = "OPM Cartesian fields"
else:
    r0, v0 = eci_from_keplerian(
        orbit.a,
        orbit.e,
        orbit.i,
        orbit.omega_big,
        orbit.omega_small,
        orbit.nu,
    )
    initial_state_source = "Keplerian fallback"

y0 = np.concatenate((r0, v0))

print(f"Initial state source: {initial_state_source}")
print(f"Position [m]:  {r0}")
print(f"Velocity [m/s]: {v0}")
print(f"Orbital period: {orbit.period / 60:.2f} min")
print(f"Orbits in 24 h: {orbit.num_orbits_24h:.2f}")

# %% [markdown]
# ## 2. Shared notebook helpers
#
# The exporter prompts for the fixed-grid step. In a notebook, that prompt is replaced temporarily so the cell does not block.

# %%
@contextmanager
def patched_input(response: str = "10"):
    original_input = builtins.input
    builtins.input = lambda prompt="": response
    try:
        yield
    finally:
        builtins.input = original_input


def run_case(name: str, integrator, options: dict, output_subdir: str):
    output_dir = repo_root / "output" / "notebook" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    t_final = const.seconds_per_day
    output_interval = 10.0
    t_eval = np.arange(0.0, t_final + output_interval, output_interval)

    print(f"Running {name} for {t_final / 3600:.1f} hours with {t_eval.size} output samples...")
    t_out, y_out, stats = integrator(gravity_ode, t_eval, y0, options)

    with patched_input("10"):
        export_results(t_out, y_out, orbit, output_dir)

    compare_analytical(t_out, y_out, orbit, output_dir)
    plot_conservation(t_out, y_out, orbit, output_dir)

    return {
        "name": name,
        "output_dir": output_dir,
        "t": t_out,
        "y": y_out,
        "stats": stats,
    }


# %% [markdown]
# ## 3. Run the RK7(8) propagator
#
# Toggle `RUN_RK78` if you want to skip this path.

# %%
RUN_RK78 = True
RK78_OPTIONS = {
    "RelTol": 1e-10,
    "AbsTol": 1e-12,
    "MaxStep": 120.0,
    "InternalStep": 1e-3,
}

if RUN_RK78:
    rk78_run = run_case("rk78", rk78_integrate, RK78_OPTIONS, "rk78")
    print(rk78_run["stats"])

# %% [markdown]
# ## 4. Run the symplectic propagator
#
# Toggle `RUN_SYMPLECTIC` if you want to skip this path.

# %%
RUN_SYMPLECTIC = False
SYMPLECTIC_OPTIONS = {
    "SymplecticStep": 10.0,
}

if RUN_SYMPLECTIC:
    symplectic_run = run_case("symplectic", symplectic_integrate, SYMPLECTIC_OPTIONS, "symplectic")
    print(symplectic_run["stats"])

# %% [markdown]
# ## 5. Inspect generated outputs
#
# The repository writes a consistent CSV set for each run. This cell loads the latest RK78 run if available and shows a quick preview.

# %%
latest_output_dir = repo_root / "output" / "notebook" / "rk78"
cartesian_csv = latest_output_dir / "orbit_cartesian.csv"
energy_csv = latest_output_dir / "orbit_energy.csv"
angmom_csv = latest_output_dir / "orbit_angular_momentum.csv"
comparison_csv = latest_output_dir / "analytical_comparison.csv"

if cartesian_csv.exists():
    df_cartesian = pd.read_csv(cartesian_csv)
    display(df_cartesian.head())

if comparison_csv.exists():
    df_comparison = pd.read_csv(comparison_csv)
    display(df_comparison[["time_s", "r_error_norm_m", "energy_error_rel", "h_error_rel"]].head())

# %% [markdown]
# ## 6. Compare the two integrators
#
# If both runs are enabled, this cell compares the final state and the key conservation metrics.

# %%
if "rk78_run" in globals() and "symplectic_run" in globals():
    rk78_final = rk78_run["y"][-1]
    symplectic_final = symplectic_run["y"][-1]
    final_state_delta = symplectic_final - rk78_final

    print(f"Final state delta [m, m/s]: {final_state_delta}")
    print(f"Position delta norm [m]: {np.linalg.norm(final_state_delta[:3]):.6e}")
    print(f"Velocity delta norm [m/s]: {np.linalg.norm(final_state_delta[3:]):.6e}")
else:
    print("Enable both runs to compare RK78 and symplectic outputs in this cell.")

# %% [markdown]
# ## 7. Notes for extensions
#
# - Change `RUN_RK78` or `RUN_SYMPLECTIC` to choose the integration path.
# - Adjust the `options` dictionaries to explore tolerances or step sizes.
# - Use `output/notebook/<integrator>/` for any scenario-specific results.
# - For a full command-line run, use `python leo_simulator.py`.
