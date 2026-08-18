from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import constants


def plot_conservation(t_vector, y_matrix, orbit_params, output_dir):
    """Plot conserved quantities vs time with analytical baselines and zoomed error views."""
    t_vector = np.asarray(t_vector, dtype=float).reshape(-1)
    y_matrix = np.asarray(y_matrix, dtype=float)

    # Prefer the exported orbit_* CSVs as the source of truth for plotting.
    # Fall back to the validation-generated filenames only if the orbit_* files
    # are unavailable.
    energy_candidates = [
        Path(output_dir) / "orbit_energy.csv",
        Path(output_dir) / "energy_conservation.csv",
    ]
    angmom_candidates = [
        Path(output_dir) / "orbit_angular_momentum.csv",
        Path(output_dir) / "angular_momentum_conservation.csv",
    ]

    energy_file = next((p for p in energy_candidates if p.exists()), None)
    angmom_file = next((p for p in angmom_candidates if p.exists()), None)

    if energy_file is not None and angmom_file is not None:
        # Read the energy and angular momentum files directly to get precomputed errors
        df_e = pd.read_csv(energy_file)
        df_h = pd.read_csv(angmom_file)
        print(f"  Using exported energy and angular momentum from {energy_file} and {angmom_file}")
        
        # Use the absolute error columns directly from the files
        t_vector = np.asarray(df_e["time_s"], dtype=float).reshape(-1)
        
        # Energy error column is `dE_abs`
        energy_error = np.asarray(df_e["dE_abs"], dtype=float)
        
        # Angular momentum error column is `dH_abs`
        h_error = np.asarray(df_h["dH_abs"], dtype=float)
    else:
        # Fallback to computing from state vectors if files are not found
        mu_earth = constants().mu_earth
        r_vec = y_matrix[:, 0:3]
        v_vec = y_matrix[:, 3:6]
        r_mag = np.linalg.norm(r_vec, axis=1)
        v_mag = np.linalg.norm(v_vec, axis=1)
        energy = 0.5 * v_mag**2 - mu_earth / r_mag
        h_mag = np.linalg.norm(np.cross(r_vec, v_vec), axis=1)
        
        # Calculate errors directly from the analytical values
        energy_error = energy - orbit_params.energy
        h_error = h_mag - orbit_params.h_mag

    time_hours = t_vector / 3600.0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Only plot the error relative to analytical values (single-panel plots).
    time_index = np.arange(len(time_hours))
    tick_step = max(1, len(time_index) // 8)
    tick_positions = time_index[::tick_step]
    tick_labels = [f"{time_hours[int(i)]:.2f}h" for i in tick_positions]

    # Energy error plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_index, energy_error, linewidth=1.2, color="darkblue", label="Energy Error")
    ax.axhline(0.0, color="green", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Specific Energy Error (J/kg)")
    ax.set_title("Specific Orbital Energy Error")
    ax.grid(True, alpha=0.35)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    plt.tight_layout()
    energy_plot_file = output_path / "energy_error.png"
    plt.savefig(energy_plot_file, dpi=180)
    plt.close()

    # Angular momentum error plot
    _fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_index, h_error, linewidth=1.2, color="darkblue", label="Angular Momentum Error")
    ax.axhline(0.0, color="green", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Specific Angular Momentum Error (m^2/s)")
    ax.set_title("Specific Angular Momentum Error")
    ax.grid(True, alpha=0.35)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    plt.tight_layout()
    h_plot_file = output_path / "angular_momentum_error.png"
    plt.savefig(h_plot_file, dpi=180)
    plt.close()

    print(f"Saved: {energy_plot_file}")
    print(f"Saved: {h_plot_file}")
