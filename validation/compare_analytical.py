from pathlib import Path

import numpy as np
import pandas as pd

from src.eci_from_keplerian import eci_from_keplerian


def compare_analytical(t_vector, y_matrix, orbit_params, output_dir):
    """Compare propagated numerical trajectory against analytical two-body motion."""
    t_vector = np.asarray(t_vector, dtype=float).reshape(-1)
    y_matrix = np.asarray(y_matrix, dtype=float)
    num_points = t_vector.size

    nu_init = orbit_params.nu

    r_numerical = y_matrix[:, 0:3].copy()
    r_analytical = np.zeros((num_points, 3), dtype=float)
    position_error = np.zeros(num_points, dtype=float)
    relative_error = np.zeros(num_points, dtype=float)

    print("  Computing analytical solution...")

    for k in range(num_points):
        n = orbit_params.n
        m = n * t_vector[k]
        nu = nu_init + m

        r_ana_vec, _ = eci_from_keplerian(
            orbit_params.a,
            orbit_params.e,
            orbit_params.i,
            orbit_params.omega_big,
            orbit_params.omega_small,
            nu,
        )
        r_analytical[k, :] = r_ana_vec

        dr = r_numerical[k, :] - r_analytical[k, :]
        position_error[k] = np.linalg.norm(dr)
        relative_error[k] = position_error[k] / orbit_params.a

    pos_error_max = np.max(position_error)
    pos_error_mean = np.mean(position_error)
    pos_error_rms = np.sqrt(np.mean(position_error**2))
    rel_error_max = np.max(relative_error)
    rel_error_mean = np.mean(relative_error)

    print(f"  Max position error:             {pos_error_max:.6e} m ({rel_error_max:.6e} rel)")
    print(f"  Mean position error:            {pos_error_mean:.6e} m ({rel_error_mean:.6e} rel)")
    print(f"  RMS position error:             {pos_error_rms:.6e} m")

    if pos_error_max < 1:
        print("  Status: EXCELLENT - Position error < 1 meter")
    elif pos_error_max < 10:
        print("  Status: GOOD - Position error < 10 meters")
    elif pos_error_max < 100:
        print("  Status: ACCEPTABLE - Position error < 100 meters")
    else:
        print("  Status: WARNING - Position error exceeds tolerance")

    print("\n  Orbital Period Verification:")
    print(f"  Theoretical period:             {orbit_params.period:.2f} s ({orbit_params.period_min:.2f} min)")

    r_magnitude = np.linalg.norm(y_matrix[:, 0:3], axis=1)
    r_ref = r_magnitude[0]
    perigee_threshold = 0.9999
    perigee_indices = np.where(r_magnitude < r_ref * perigee_threshold)[0]

    if perigee_indices.size > 1:
        time_diffs = np.diff(t_vector[perigee_indices])
        valid_period = np.median(time_diffs)
        period_error = abs(valid_period - orbit_params.period) / orbit_params.period * 100.0
        print(f"  Estimated period (from sim):    {valid_period:.2f} s ({valid_period / 60:.2f} min)")
        print(f"  Period error:                   {period_error:.4f} %")
    else:
        print("  Note: Period estimation requires multiple orbit crossings")

    comparison_file = Path(output_dir) / "analytical_comparison.csv"
    pd.DataFrame(
        {
            "time_s": t_vector,
            "x_num_m": r_numerical[:, 0],
            "y_num_m": r_numerical[:, 1],
            "z_num_m": r_numerical[:, 2],
            "x_ana_m": r_analytical[:, 0],
            "y_ana_m": r_analytical[:, 1],
            "z_ana_m": r_analytical[:, 2],
            "error_m": position_error,
            "rel_error": relative_error,
        }
    ).to_csv(comparison_file, index=False)
    print(f"\n  Data saved to: {comparison_file}")
