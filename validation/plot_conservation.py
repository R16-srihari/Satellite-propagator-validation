from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_conservation(t_vector, y_matrix, orbit_params, output_dir):
    """Plot conserved quantities vs time with analytical baselines and zoomed error views."""
    t_vector = np.asarray(t_vector, dtype=float).reshape(-1)
    y_matrix = np.asarray(y_matrix, dtype=float)

    r_vec = y_matrix[:, 0:3]
    v_vec = y_matrix[:, 3:6]

    mu_earth = 3.986004418e14
    r_mag = np.linalg.norm(r_vec, axis=1)
    v_mag = np.linalg.norm(v_vec, axis=1)

    energy = v_mag**2 / 2.0 - mu_earth / r_mag
    h_mag = np.linalg.norm(np.cross(r_vec, v_vec), axis=1)

    energy_baseline = np.full_like(energy, orbit_params.energy)
    h_baseline = np.full_like(h_mag, orbit_params.h_mag)

    energy_error = energy - orbit_params.energy
    h_error = h_mag - orbit_params.h_mag

    time_hours = t_vector / 3600.0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    ax1.plot(time_hours, energy, linewidth=1.2, label="Numerical Energy")
    ax1.plot(time_hours, energy_baseline, "--", linewidth=1.5, color="green", label="Analytical Baseline")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Specific Energy (J/kg)")
    ax1.set_title("Energy Conservation vs Time")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")

    ax2.plot(time_hours, energy_error * 1e7, linewidth=1.2, color="darkblue", label="Error (×1e-7 J/kg)")
    ax2.axhline(0, color="green", linestyle="--", linewidth=1.5, label="Zero")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Error (×1e-7 J/kg)")
    ax2.set_title("Energy Error (Zoomed)")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="best")

    plt.tight_layout()
    energy_plot_file = output_path / "energy_vs_time.png"
    plt.savefig(energy_plot_file, dpi=180)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    ax1.plot(time_hours, h_mag, linewidth=1.2, label="Numerical |h|")
    ax1.plot(time_hours, h_baseline, "--", linewidth=1.5, color="green", label="Analytical Baseline")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Specific Angular Momentum (m^2/s)")
    ax1.set_title("Angular Momentum Conservation vs Time")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")

    ax2.plot(time_hours, h_error * 1e4, linewidth=1.2, color="darkblue", label="Error (×1e-4 m^2/s)")
    ax2.axhline(0, color="green", linestyle="--", linewidth=1.5, label="Zero")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Error (×1e-4 m^2/s)")
    ax2.set_title("Angular Momentum Error (Zoomed)")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="best")

    plt.tight_layout()
    h_plot_file = output_path / "angular_momentum_vs_time.png"
    plt.savefig(h_plot_file, dpi=180)
    plt.close()

    print(f"Saved: {energy_plot_file}")
    print(f"Saved: {h_plot_file}")
