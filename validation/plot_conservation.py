from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_conservation(t_vector, y_matrix, orbit_params, output_dir):
    """Plot conserved quantities vs time with analytical baselines and zoomed error views."""
    t_vector = np.asarray(t_vector, dtype=float).reshape(-1)
    y_matrix = np.asarray(y_matrix, dtype=float)

    # Prefer exported energy/angular momentum CSVs if present
    energy_file = Path(output_dir) / "orbit_energy.csv"
    angmom_file = Path(output_dir) / "orbit_angular_momentum.csv"
    mu_earth = 3.986004418e14
    if energy_file is not None and angmom_file is not None:
        df_e = pd.read_csv(energy_file)
        df_h = pd.read_csv(angmom_file)
        print(f"  Using exported energy and angular momentum from {energy_file} and {angmom_file}")
        t_vector = np.asarray(df_e["time_s"], dtype=float).reshape(-1)
        # Energy column name may vary; prefer `energy_Jkg` then `energy`.
        if "energy_Jkg" in df_e.columns:
            energy = np.asarray(df_e["energy_Jkg"], dtype=float)
        elif "energy" in df_e.columns:
            energy = np.asarray(df_e["energy"], dtype=float)
        else:
            # Fallback: pick the first non-time numeric column
            cols = [c for c in df_e.columns if c != "time_s"]
            energy = np.asarray(df_e[cols[0]], dtype=float)

        # Angular momentum magnitude column expected as `h_mag`
        if "h_mag" in df_h.columns:
            h_mag = np.asarray(df_h["h_mag"], dtype=float)
        else:
            # Fallback: try to compute from components if present
            if all(c in df_h.columns for c in ("hx", "hy", "hz")):
                h_mag = np.sqrt(
                    df_h["hx"].to_numpy() ** 2 + df_h["hy"].to_numpy() ** 2 + df_h["hz"].to_numpy() ** 2
                )
            else:
                # Last resort: take first numeric column not `time_s`
                cols = [c for c in df_h.columns if c != "time_s"]
                h_mag = np.asarray(df_h[cols[0]], dtype=float)
    else:
        r_vec = y_matrix[:, 0:3]
        v_vec = y_matrix[:, 3:6]
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
