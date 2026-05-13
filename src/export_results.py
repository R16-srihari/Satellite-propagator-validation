from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import constants
from src.keplerian_from_eci import keplerian_from_eci


def export_results(t_vector, y_matrix, orbit_params, output_dir):
    """Export propagated states, orbital elements, and conservation metrics to CSV."""
    const = constants()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t_vector = np.asarray(t_vector, dtype=float).reshape(-1)
    y_matrix = np.asarray(y_matrix, dtype=float)
    num_points = t_vector.size

    cartesian_df = pd.DataFrame(
        {
            "time_s": t_vector,
            "x_m": y_matrix[:, 0],
            "y_m": y_matrix[:, 1],
            "z_m": y_matrix[:, 2],
            "vx_ms": y_matrix[:, 3],
            "vy_ms": y_matrix[:, 4],
            "vz_ms": y_matrix[:, 5],
        }
    )
    cartesian_file = output_path / "orbit_cartesian.csv"
    cartesian_df.to_csv(cartesian_file, index=False)
    print(f"Saved: {cartesian_file}")

    a_array = np.zeros(num_points)
    e_array = np.zeros(num_points)
    i_array = np.zeros(num_points)
    omega_big_array = np.zeros(num_points)
    omega_small_array = np.zeros(num_points)
    nu_array = np.zeros(num_points)

    print("Converting to Keplerian elements...")
    report_stride = max(1, num_points // 10)

    for k in range(num_points):
        r_vec = y_matrix[k, 0:3]
        v_vec = y_matrix[k, 3:6]

        a_k, e_k, i_k, omega_big_k, omega_small_k, nu_k = keplerian_from_eci(r_vec, v_vec)

        a_array[k] = a_k
        e_array[k] = e_k
        i_array[k] = i_k
        omega_big_array[k] = omega_big_k
        omega_small_array[k] = omega_small_k
        nu_array[k] = nu_k

        if (k + 1) % report_stride == 0 or k == num_points - 1:
            print(f"  Progress: {100.0 * (k + 1) / num_points:.1f}%")

    keplerian_df = pd.DataFrame(
        {
            "time_s": t_vector,
            "a_m": a_array,
            "e_": e_array,
            "i_deg": i_array * const.rad2deg,
            "Omega_deg": omega_big_array * const.rad2deg,
            "omega_deg": omega_small_array * const.rad2deg,
            "nu_deg": nu_array * const.rad2deg,
        }
    )
    keplerian_file = output_path / "orbit_elements.csv"
    keplerian_df.to_csv(keplerian_file, index=False)
    print(f"Saved: {keplerian_file}")

    energy_array = np.zeros(num_points)
    h_mag_array = np.zeros(num_points)

    for k in range(num_points):
        r_vec = y_matrix[k, 0:3]
        v_vec = y_matrix[k, 3:6]
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        energy_array[k] = v**2 / 2.0 - const.mu_earth / r
        h_mag_array[k] = np.linalg.norm(np.cross(r_vec, v_vec))

    e_ref = energy_array[0]
    d_e_rel = (energy_array - e_ref) / abs(e_ref)

    energy_df = pd.DataFrame(
        {
            "time_s": t_vector,
            "energy_Jkg": energy_array,
            "dE_rel": d_e_rel,
            "h_mag_m2s": h_mag_array,
        }
    )
    energy_file = output_path / "orbit_energy.csv"
    energy_df.to_csv(energy_file, index=False)
    print(f"Saved: {energy_file}")

    print("\n=== EXPORT SUMMARY ===")
    print(f"Total points exported: {num_points}")
    print(
        "Time span: "
        f"{t_vector[-1] / const.seconds_per_hour:.2f} hours "
        f"({t_vector[-1] / orbit_params.period:.2f} orbits)"
    )
    print(
        "Energy variation (%): "
        f"min={np.min(d_e_rel) * 100:.2e}, "
        f"max={np.max(d_e_rel) * 100:.2e}, "
        f"mean={np.mean(np.abs(d_e_rel)) * 100:.2e}"
    )
    print(
        "Angular momentum variation (%): "
        f"{(np.max(h_mag_array) - np.min(h_mag_array)) / np.mean(h_mag_array) * 100:.2e}"
    )
    print(f"Semi-major axis variation (m): {np.max(a_array) - np.min(a_array):.2e}")
    print("======================\n")
