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
    if t_vector.size != y_matrix.shape[0]:
        raise ValueError("t_vector and y_matrix must contain the same number of samples")

    # Validate that time vector is strictly increasing (required for np.interp downstream)
    if t_vector.size < 2:
        raise ValueError("t_vector must contain at least two time points")
    if not np.all(np.diff(t_vector) > 0):
        raise ValueError("t_vector must be strictly increasing")
    if np.any(np.isnan(t_vector)):
        raise ValueError("t_vector contains NaN values")

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
    print(f"\nSaved: {cartesian_file}")

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
    h_vec = np.zeros((num_points, 3))

    for k in range(num_points):
        r_vec = y_matrix[k, 0:3]
        v_vec = y_matrix[k, 3:6]
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        energy_array[k] = v**2 / 2.0 - const.mu_earth / r # Specific orbital energy (J/kg)
        h_vec[k, :] = np.cross(r_vec, v_vec)
        h_mag_array[k] = np.linalg.norm(h_vec[k, :]) # Specific angular momentum magnitude (m^2/s)

    e_ref = energy_array[0]
    d_e_abs = energy_array - e_ref
    d_e_rel = (energy_array - e_ref) / abs(e_ref)

    energy_df = pd.DataFrame(
        {
            "time_s": t_vector,
            "energy_Jkg": energy_array,
            "dE_abs": d_e_abs,
            "dE_rel": d_e_rel,
        }
    )
    energy_file = output_path / "orbit_energy.csv"
    energy_df.to_csv(energy_file, index=False)
    print(f"Saved: {energy_file}")

    # Save angular momentum time series (vector + magnitude + relative change)
    h_ref = h_mag_array[0]
    d_h_abs = h_mag_array - h_ref
    d_h_rel = d_h_abs / abs(h_ref)

    angmom_df = pd.DataFrame(
        {
            "time_s": t_vector,
            "hx": h_vec[:, 0],
            "hy": h_vec[:, 1],
            "hz": h_vec[:, 2],
            "h_mag": h_mag_array,
            "dH_abs": d_h_abs,
            "dH_rel": d_h_rel,
        }
    )
    angmom_file = output_path / "orbit_angular_momentum.csv"
    angmom_df.to_csv(angmom_file, index=False)
    print(f"Saved: {angmom_file}")

    print("\n=== EXPORT SUMMARY ===")
    print(f"Total points exported: {num_points}")
    print(
        "Time span: "
        f"{t_vector[-1] / const.seconds_per_hour:.2f} hours "
        f"({t_vector[-1] / orbit_params.period:.2f} orbits)"
    )
    dt = np.diff(t_vector)
    print(
        f"Time step stats (s): min={np.min(dt):.3f}, median={np.median(dt):.3f}, max={np.max(dt):.3f}"
    )
    print(f"Non-uniform time grid: {np.max(dt) - np.min(dt) > 1e-9}")
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
