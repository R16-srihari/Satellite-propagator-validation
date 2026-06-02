from pathlib import Path

import numpy as np
import pandas as pd

from src.analytical_solution import analytical_solution
from src.constants import constants


def _load_state_csv(csv_path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required comparison file: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    return df


def _print_scalar_stats(label: str, analytical: np.ndarray, numerical: np.ndarray, unit: str) -> None:
    error = numerical - analytical
    abs_error = np.abs(error)
    relative_error = np.divide(
        abs_error,
        np.abs(analytical),
        out=np.zeros_like(abs_error),
        where=np.abs(analytical) > 0,
    )
    print(f"\n  {label} comparison:")
    print(f"  {label} analytical value:        {analytical[0]:.15e} {unit}")
    print(f"  {label} mean numerical value:    {np.mean(numerical):.15e} {unit}")
    print(f"  {label} mean abs error:          {np.mean(abs_error):.6e} {unit}")
    print(f"  {label} max abs error:           {np.max(abs_error):.6e} {unit}")
    print(f"  {label} mean rel error:          {np.mean(relative_error):.6e}")
    print(f"  {label} max rel error:           {np.max(relative_error):.6e}")


def _print_vector_stats(label: str, analytical: np.ndarray, numerical: np.ndarray, unit: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    error_vectors = numerical - analytical
    error_norm = np.linalg.norm(error_vectors, axis=1)
    analytical_norm = np.linalg.norm(analytical, axis=1)
    numerical_norm = np.linalg.norm(numerical, axis=1)
    relative_error = np.divide(
        error_norm,
        analytical_norm,
        out=np.zeros_like(error_norm),
        where=analytical_norm > 0,
    )
    print(f"\n  {label} comparison:")
    print(f"  Analytical {label.lower()} norm:  {analytical_norm[0]:.15e} {unit}")
    print(f"  Mean numerical {label.lower()} norm: {np.mean(numerical_norm):.15e} {unit}")
    print(f"  Mean error norm:                {np.mean(error_norm):.6e} {unit}")
    print(f"  Maximum error norm:             {np.max(error_norm):.6e} {unit}")
    print(f"  Mean rel error:                 {np.mean(relative_error):.6e}")
    print(f"  Maximum rel error:              {np.max(relative_error):.6e}")
    return error_norm, relative_error, numerical_norm


def compare_analytical(t_vector, y_matrix, orbit_params, output_dir):
    """Compare exported numerical trajectory against the analytical two-body reference.

    This routine treats the exported CSVs as the source of truth:
    - `orbit_cartesian.csv` for the propagated Cartesian trajectory
    - `orbit_energy.csv` for specific energy
    - `orbit_angular_momentum.csv` for specific angular momentum
    """
    output_path = Path(output_dir)

    cartesian_file = output_path / "orbit_cartesian.csv"
    energy_file = output_path / "orbit_energy.csv"
    angmom_file = output_path / "orbit_angular_momentum.csv"

    df_cart = _load_state_csv(cartesian_file, ("time_s", "x_m", "y_m", "z_m", "vx_ms", "vy_ms", "vz_ms"))
    df_energy = _load_state_csv(energy_file, ("time_s", "energy_Jkg"))
    df_h = _load_state_csv(angmom_file, ("time_s", "h_mag"))

    t_vector = np.asarray(df_cart["time_s"], dtype=float).reshape(-1)
    x_num = np.asarray(df_cart["x_m"], dtype=float)
    y_num = np.asarray(df_cart["y_m"], dtype=float)
    z_num = np.asarray(df_cart["z_m"], dtype=float)
    r_numerical = np.column_stack((x_num, y_num, z_num))

    vx_num = np.asarray(df_cart["vx_ms"], dtype=float)
    vy_num = np.asarray(df_cart["vy_ms"], dtype=float)
    vz_num = np.asarray(df_cart["vz_ms"], dtype=float)
    v_numerical = np.column_stack((vx_num, vy_num, vz_num))

    num_points = t_vector.size
    if df_energy.shape[0] != num_points or df_h.shape[0] != num_points:
        raise ValueError("Exported comparison CSVs must have the same number of rows")

    print(f"  Using exported cartesian states from {cartesian_file}")
    print(f"  Using exported energy from {energy_file}")
    print(f"  Using exported angular momentum from {angmom_file}")

    mu_earth = constants().mu_earth
    r_analytical = np.zeros((num_points, 3), dtype=float)
    v_analytical = np.zeros((num_points, 3), dtype=float)
    for k in range(num_points):
        r_analytical[k, :], v_analytical[k, :] = analytical_solution(
            t_vector[k],
            orbit_params.a,
            orbit_params.e,
            orbit_params.i,
            orbit_params.omega_big,
            orbit_params.omega_small,
            orbit_params.nu,
            mu_earth,
        )

    r_error_norm, r_relative_error, r_num_mag = _print_vector_stats("Position vector", r_analytical, r_numerical, "m")
    r_ana_mag = np.linalg.norm(r_analytical, axis=1)

    v_error_norm, v_relative_error, v_num_mag = _print_vector_stats("Velocity vector", v_analytical, v_numerical, "m/s")
    v_ana_mag = np.linalg.norm(v_analytical, axis=1)

    if "energy_Jkg" in df_energy.columns:
        energy = np.asarray(df_energy["energy_Jkg"], dtype=float)
    else:
        raise KeyError("orbit_energy.csv must contain an energy_Jkg column")
    energy_baseline = np.full_like(energy, orbit_params.energy)
    energy_error = energy - energy_baseline
    energy_relative_error = np.divide(
        np.abs(energy_error),
        np.abs(energy_baseline),
        out=np.zeros_like(energy_error),
        where=np.abs(energy_baseline) > 0,
    )
    _print_scalar_stats("Energy", energy_baseline, energy, "J/kg")

    if "h_mag" in df_h.columns:
        h_mag = np.asarray(df_h["h_mag"], dtype=float)
    else:
        raise KeyError("orbit_angular_momentum.csv must contain an h_mag column")
    h_baseline = np.full_like(h_mag, orbit_params.h_mag)
    h_error = h_mag - h_baseline
    h_relative_error = np.divide(
        np.abs(h_error),
        np.abs(h_baseline),
        out=np.zeros_like(h_error),
        where=np.abs(h_baseline) > 0,
    )
    _print_scalar_stats("Angular momentum", h_baseline, h_mag, "m^2/s")

    comparison_file = output_path / "analytical_comparison.csv"
    pd.DataFrame(
        {
            "time_s": t_vector,
            "x_num_m": r_numerical[:, 0],
            "y_num_m": r_numerical[:, 1],
            "z_num_m": r_numerical[:, 2],
            "r_num_m": r_num_mag,
            "x_ana_m": r_analytical[:, 0],
            "y_ana_m": r_analytical[:, 1],
            "z_ana_m": r_analytical[:, 2],
            "r_ana_m": r_ana_mag,
            "r_error_norm_m": r_error_norm,
            "r_error_rel": r_relative_error,
            "vx_num_ms": v_numerical[:, 0],
            "vy_num_ms": v_numerical[:, 1],
            "vz_num_ms": v_numerical[:, 2],
            "v_num_ms": v_num_mag,
            "vx_ana_ms": v_analytical[:, 0],
            "vy_ana_ms": v_analytical[:, 1],
            "vz_ana_ms": v_analytical[:, 2],
            "v_ana_ms": v_ana_mag,
            "v_error_norm_ms": v_error_norm,
            "v_error_rel": v_relative_error,
            "energy_error_Jkg": energy_error,
            "energy_error_rel": energy_relative_error,
            "h_error_m2s": h_error,
            "h_error_rel": h_relative_error,
        }
    ).to_csv(comparison_file, index=False)
    print(f"\n  Data saved to: {comparison_file}")
