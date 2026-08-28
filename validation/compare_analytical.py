import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytical_solution import analytical_solution
from src.constants import constants
from src.orbital_parameters import (
    OrbitParameters,
    _get_keplerian_state_from_opm,
    _parse_opm,
    orbital_parameters,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is required for plotting
    plt = None


def _load_state_csv(csv_path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required comparison file: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    return df


def _print_scalar_stats(label: str, analytical: np.ndarray, numerical: np.ndarray, err_abs: np.ndarray, err_rel: np.ndarray, unit: str) -> None:
    def max_abs(arr: np.ndarray) -> float:
        return float(np.max(np.abs(arr)))

    def mean_abs(arr: np.ndarray) -> float:
        return float(np.mean(np.abs(arr)))

    print(f"\n  {label} comparison:")
    print(f"  {label} analytical value:        {analytical[0]:.15e} {unit}")
    print(f"  {label} mean numerical value:    {np.mean(numerical):.15e} {unit}")
    print(f"  {label} mean abs error:          {mean_abs(err_abs):.6e} {unit}")
    print(f"  {label} max abs error:           {max_abs(err_abs):.6e} {unit}")
    print(f"  {label} mean rel error:          {mean_abs(err_rel):.6e}")
    print(f"  {label} max rel error:           {max_abs(err_rel):.6e}")


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


def _reference_state_for_times(t_values: np.ndarray, orbit_params, mu: float) -> tuple[np.ndarray, np.ndarray]:
    t_values = np.asarray(t_values, dtype=float).reshape(-1)
    r_ana = np.zeros((t_values.size, 3), dtype=float)
    v_ana = np.zeros((t_values.size, 3), dtype=float)
    for idx, t_value in enumerate(t_values):
        r_ana[idx], v_ana[idx] = analytical_solution(
            float(t_value),
            orbit_params.a,
            orbit_params.e,
            orbit_params.i,
            orbit_params.omega_big,
            orbit_params.omega_small,
            orbit_params.nu,
            mu,
        )
    return r_ana, v_ana


def _load_orbit_from_opm(opm_path: Path | None) -> OrbitParameters:
    if opm_path is None or not opm_path.exists():
        return orbital_parameters(verbose=False)

    const = constants()
    opm = _parse_opm(opm_path)
    keplerian_state = _get_keplerian_state_from_opm(opm)
    if keplerian_state is None:
        raise RuntimeError(f"OPM file {opm_path} is missing required Keplerian values")

    a, e, inc, omega_big, omega_small, nu = keplerian_state
    epoch: datetime | None = None
    if "EPOCH" in opm:
        try:
            epoch = datetime.fromisoformat(opm["EPOCH"])
        except ValueError:
            epoch = datetime.strptime(opm["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f")

    return OrbitParameters(
        altitude=a - const.r_earth,
        a=a,
        e=e,
        i=inc,
        omega_big=omega_big,
        omega_small=omega_small,
        nu=nu,
        n=np.sqrt(const.mu_earth / a**3),
        period=const.twopi / np.sqrt(const.mu_earth / a**3),
        period_min=(const.twopi / np.sqrt(const.mu_earth / a**3)) / 60.0,
        v_orbit=np.sqrt(const.mu_earth / a),
        energy=-const.mu_earth / (2.0 * a),
        h_mag=np.sqrt(const.mu_earth * a),
        num_orbits_24h=const.seconds_per_day / (const.twopi / np.sqrt(const.mu_earth / a**3)),
        epoch=epoch,
    )


def _load_stk_reference(stk_csv: Path | None, orbit_params) -> dict | None:
    """Load the STK reference CSV, converting km-unit headers to SI.

    Handles STK CSV columns:
    - Time (UTCG): calendar timestamp -> seconds from OPM epoch
    - X/Y/Z (km) -> m (x1000)
    - Vx/Vy/Vz (km/sec) -> m/s (x1000)
    - Delaunay_G (km^2/sec) -> |h| in m^2/s (x1e6)
    - Semimajor_Axis (km) -> m (x1000) -> energy = -mu/(2a)
    """
    if stk_csv is None:
        repo_root = Path(__file__).resolve().parents[1]
        stk_csv = repo_root / "STK_input" / "Satellite1_Results.csv"
    if not stk_csv.exists():
        return None

    try:
        df = pd.read_csv(stk_csv, on_bad_lines="skip")
    except Exception:
        return None

    # Build case-insensitive column name lookup
    col_lower = {c.lower().strip(): c for c in df.columns}

    def _find_col(*candidates):
        for cand in candidates:
            cl = cand.lower().strip()
            if cl in col_lower:
                return col_lower[cl]
        return None

    time_col = _find_col("Time (UTCG)", "time (utcg)", "time_utc", "time")
    x_col = _find_col("X (km)", "x (km)")
    y_col = _find_col("Y (km)", "y (km)")
    z_col = _find_col("Z (km)", "z (km)")
    vx_col = _find_col("Vx (km/sec)", "vx (km/sec)", 'Vx (km/s)', "vx")
    vy_col = _find_col("Vy (km/sec)", "vy (km/sec)", 'Vy (km/s)', "vy")
    vz_col = _find_col("Vz (km/sec)", "vz (km/sec)", 'Vz (km/s)', "vz")
    delaunay_col = _find_col("Delaunay_G (km^2/sec)", "delaunay_g (km^2/sec)",
                             "Delaunay_G (m^2/sec)", "delaunay_g (m^2/sec)")
    sma_col = _find_col("Semimajor_Axis (km)", "semimajor_axis (km)",
                        "Semimajor_Axis (m)", "semimajor_axis (m)")

    # Need at minimum time + position columns
    required_cols = [time_col, x_col, y_col, z_col]
    if any(c is None for c in required_cols):
        return None

    time_series = df[time_col].astype(str)
    valid_mask = ~time_series.str.contains("Statistics", case=False, na=False)
    df = df.loc[valid_mask].copy()
    if df.empty:
        return None

    try:
        stk_epoch = orbit_params.epoch
        if stk_epoch is None:
            first_time = str(df[time_col].iloc[0])
            try:
                stk_epoch = datetime.strptime(first_time, "%d %b %Y %H:%M:%S.%f")
            except ValueError:
                try:
                    stk_epoch = datetime.strptime(first_time, "%d %b %Y %H:%M:%S")
                except ValueError:
                    pass
        times = pd.to_datetime(df[time_col], format="%d %b %Y %H:%M:%S.%f", errors="coerce")
        if times.isna().all():
            times = pd.to_datetime(df[time_col], format="%d %b %Y %H:%M:%S", errors="coerce")
        if times.isna().all():
            times = pd.to_datetime(df[time_col], errors="coerce")
    except Exception:
        return None

    valid = times.notna()
    df = df.loc[valid].copy()
    times = times[valid]
    if stk_epoch is None or df.empty:
        return None

    seconds_from_epoch = (times - pd.Timestamp(stk_epoch)).dt.total_seconds().to_numpy(float)
    pos_m = np.column_stack((
        df[x_col].to_numpy(float) * 1000.0,
        df[y_col].to_numpy(float) * 1000.0,
        df[z_col].to_numpy(float) * 1000.0,
    ))

    result = {
        "time_s": seconds_from_epoch,
        "position_m": pos_m,
    }

    # Velocity (km/sec -> m/s), if available
    if vx_col is not None and vy_col is not None and vz_col is not None:
        vel_mps = np.column_stack((
            df[vx_col].to_numpy(float) * 1000.0,
            df[vy_col].to_numpy(float) * 1000.0,
            df[vz_col].to_numpy(float) * 1000.0,
        ))
        result["velocity_mps"] = vel_mps

    # Specific angular momentum: Delaunay_G (km^2/sec) -> m^2/s (x1e6)
    if delaunay_col is not None:
        result["h_mag"] = df[delaunay_col].to_numpy(float) * 1e6

    # Specific orbital energy: epsilon = -mu / (2a), a in meters from km
    if sma_col is not None:
        a_m = df[sma_col].to_numpy(float) * 1000.0
        result["energy_Jkg"] = -constants().mu_earth / (2.0 * a_m)

    return result


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
    df_energy = _load_state_csv(energy_file, ("time_s", "energy_Jkg", "dE_abs", "dE_rel"))
    df_h = _load_state_csv(angmom_file, ("time_s", "h_mag", "dH_abs", "dH_rel"))

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
    r_analytical, v_analytical = _reference_state_for_times(t_vector, orbit_params, mu_earth)

    r_error_norm, r_relative_error, r_num_mag = _print_vector_stats("Position vector", r_analytical, r_numerical, "m")
    r_ana_mag = np.linalg.norm(r_analytical, axis=1)

    v_error_norm, v_relative_error, v_num_mag = _print_vector_stats("Velocity vector", v_analytical, v_numerical, "m/s")
    v_ana_mag = np.linalg.norm(v_analytical, axis=1)

    if "energy_Jkg" in df_energy.columns:
        energy = np.asarray(df_energy["energy_Jkg"], dtype=float)
    else:
        raise KeyError("orbit_energy.csv must contain an energy_Jkg column")

    if "dE_abs" in df_energy.columns and "dE_rel" in df_energy.columns:
        dE_abs = np.asarray(df_energy["dE_abs"], dtype=float)
        dE_rel = np.asarray(df_energy["dE_rel"], dtype=float)
    else:
        raise KeyError("orbit_energy.csv must contain dE_abs and dE_rel columns")

    energy_baseline = np.full_like(energy, orbit_params.energy)
    _print_scalar_stats("Energy", energy_baseline, energy, dE_abs, dE_rel, "J/kg")

    if "h_mag" in df_h.columns:
        h_mag = np.asarray(df_h["h_mag"], dtype=float)
    else:
        raise KeyError("orbit_angular_momentum.csv must contain an h_mag column")

    if "dH_abs" in df_h.columns and "dH_rel" in df_h.columns:
        dH_abs = np.asarray(df_h["dH_abs"], dtype=float)
        dH_rel = np.asarray(df_h["dH_rel"], dtype=float)
    else:
        raise KeyError("orbit_angular_momentum.csv must contain dH_abs and dH_rel columns")

    h_baseline = np.full_like(h_mag, orbit_params.h_mag)
    _print_scalar_stats("Angular momentum magnitude", h_baseline, h_mag, dH_abs, dH_rel, "m^2/s")

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
            "energy_error_Jkg": dE_abs,
            "energy_error_rel": dE_rel,
            "|h|_error_m2s": dH_abs,
            "|h|_error_rel": dH_rel,
        }
    ).to_csv(comparison_file, index=False)
    print(f"\n  Data saved to: {comparison_file}")


def _save_plot(fig, path: Path) -> None:
    if plt is None:
        return
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_comparison_plots(output_dir, orbit_params, integrator="pd853", stk_csv=None, show=False, log_scale=True):
    """Create a validation plot set comparing the integrator against analytical and STK references.

    Parameters
    ----------
    output_dir : str or Path
        The integrator-specific output directory (e.g. output/pd853).
    orbit_params : OrbitParameters
        Orbital parameters from the OPM file.
    integrator : str
        Integrator display name used in legends.
    stk_csv : str or Path or None
        Path to the STK results CSV. If None, defaults to
        STK_input/Satellite1_Results.csv.
    show : bool
        If True, display plots interactively.
    """
    output_path = Path(output_dir)
    validation_dir = output_path / "STKcomparison"
    validation_dir.mkdir(parents=True, exist_ok=True)

    cartesian_file = output_path / "orbit_cartesian.csv"
    if not cartesian_file.exists():
        raise FileNotFoundError(f"Missing propagated state file: {cartesian_file}")

    df_cart = _load_state_csv(cartesian_file, ("time_s", "x_m", "y_m", "z_m", "vx_ms", "vy_ms", "vz_ms"))
    t_values = np.asarray(df_cart["time_s"], dtype=float)
    r_num = np.column_stack((df_cart["x_m"].to_numpy(float), df_cart["y_m"].to_numpy(float), df_cart["z_m"].to_numpy(float)))
    v_num = np.column_stack((df_cart["vx_ms"].to_numpy(float), df_cart["vy_ms"].to_numpy(float), df_cart["vz_ms"].to_numpy(float)))

    mu = constants().mu_earth
    r_ana, v_ana = _reference_state_for_times(t_values, orbit_params, mu)
    r_err = r_num - r_ana
    v_err = v_num - v_ana

    h_num = np.cross(r_num, v_num)
    h_ana = np.cross(r_ana, v_ana)
    h_num_mag = np.linalg.norm(h_num, axis=1)
    h_ana_mag = np.linalg.norm(h_ana, axis=1)
    h_err = np.abs(h_num_mag - h_ana_mag)

    r_mag = np.linalg.norm(r_num, axis=1)
    v_mag = np.linalg.norm(v_num, axis=1)
    energy_num = 0.5 * v_mag**2 - mu / np.clip(r_mag, 1e-12, None)
    energy_err = energy_num - orbit_params.energy

    comparison_df = pd.DataFrame(
        {
            "time_s": t_values,
            "r_error_norm_m": np.linalg.norm(r_err, axis=1),
            "v_error_norm_ms": np.linalg.norm(v_err, axis=1),
            "vx_error_ms": v_err[:, 0],
            "vy_error_ms": v_err[:, 1],
            "vz_error_ms": v_err[:, 2],
            "h_error_m2s": h_err,
            "energy_error_Jkg": energy_err,
        }
    )
    comparison_file = validation_dir / "comparison_errors.csv"
    comparison_df.to_csv(comparison_file, index=False)

    stk_data = _load_stk_reference(Path(stk_csv) if stk_csv is not None else None, orbit_params)

    # Precompute analytical values on the integrator time grid for interpolation
    h_ana_mag_arr = h_ana_mag  # already computed above
    energy_ana_arr = np.full_like(t_values, orbit_params.energy)

    # Interpolate analytical h and energy onto STK time grid
    def _interp_analytical(stk_time):
        """Interpolate analytical reference values onto STK time stamps."""
        h_interp = np.interp(stk_time, t_values, h_ana_mag_arr)
        energy_interp = np.interp(stk_time, t_values, energy_ana_arr)
        r_ana_interp = np.column_stack(
            [np.interp(stk_time, t_values, r_ana[:, i]) for i in range(3)]
        )
        v_ana_interp = np.column_stack(
            [np.interp(stk_time, t_values, v_ana[:, i]) for i in range(3)]
        )
        return r_ana_interp, v_ana_interp, h_interp, energy_interp

    def _plot_series(title: str, y_label: str, y_values: np.ndarray, save_name: str, x_values=None, stky=None, stklab="STK", log_y: bool = False):
        """Plot a single data series.

        The integrator curve is plotted *after* the STK curve so that the
        integrator line appears on top of the STK line in the final image.
        """
        if plt is None:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot STK data first (if available) to ensure it is underneath
        if x_values is not None and stky is not None:
            ax.plot(x_values, stky, label=stklab, linestyle="--",color='orange', linewidth=1.5)
        # Then plot the integrator data on top
        ax.plot(t_values, y_values, label=f"{integrator}", color='blue', linewidth=2.0)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(y_label + " (log scale)" if log_y else y_label)
        if log_y:
            ax.set_yscale("symlog", linthresh=1e-15)
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save_plot(fig, validation_dir / save_name)

    if stk_data is not None:
        stk_time = np.asarray(stk_data["time_s"], dtype=float)
        # Interpolate analytical reference onto STK time grid
        r_ana_stk, v_ana_stk, h_ana_stk, energy_ana_stk = _interp_analytical(stk_time)

        # STK position error (STK position vs interpolated analytical)
        stk_position_error = stk_data["position_m"] - r_ana_stk

        # STK velocity error (STK velocity vs interpolated analytical)
        if "velocity_mps" in stk_data:
            stk_velocity = stk_data["velocity_mps"]
            stk_velocity_error = stk_velocity - v_ana_stk
        else:
            stk_velocity_error = None

        # STK angular momentum error
        if "h_mag" in stk_data:
            stk_h_error = np.abs(stk_data["h_mag"] - h_ana_stk)
        else:
            stk_h_error = None

        # STK energy error
        if "energy_Jkg" in stk_data:
            stk_energy_error = np.abs(stk_data["energy_Jkg"] - energy_ana_stk)
        else:
            stk_energy_error = None

        # Build and write STK comparison errors CSV
        n_stk = len(stk_time)
        if stk_velocity_error is not None:
            stk_errors_df = pd.DataFrame(
                {
                    "time_s": stk_time,
                    "r_error_norm_m": np.linalg.norm(stk_position_error, axis=1),
                    "v_error_norm_ms": np.linalg.norm(stk_velocity_error, axis=1),
                    "vx_error_ms": stk_velocity_error[:, 0],
                    "vy_error_ms": stk_velocity_error[:, 1],
                    "vz_error_ms": stk_velocity_error[:, 2],
                    "h_error_m2s": stk_h_error if stk_h_error is not None else np.full(n_stk, np.nan),
                    "energy_error_Jkg": stk_energy_error if stk_energy_error is not None else np.full(n_stk, np.nan),
                }
            )
        else:
            stk_errors_df = pd.DataFrame(
                {
                    "time_s": stk_time,
                    "r_error_norm_m": np.linalg.norm(stk_position_error, axis=1),
                    "v_error_norm_ms": np.full(n_stk, np.nan),
                    "vx_error_ms": np.full(n_stk, np.nan),
                    "vy_error_ms": np.full(n_stk, np.nan),
                    "vz_error_ms": np.full(n_stk, np.nan),
                    "h_error_m2s": stk_h_error if stk_h_error is not None else np.full(n_stk, np.nan),
                    "energy_error_Jkg": stk_energy_error if stk_energy_error is not None else np.full(n_stk, np.nan),
                }
            )
        # Write to base output directory as STK_errors.csv
        base_output_dir = output_path.parent if output_path.name != "output" else output_path
        stk_errors_csv = base_output_dir / "STK_errors.csv"
        stk_errors_df.to_csv(stk_errors_csv, index=False)
        print(f"  STK comparison CSV saved to {stk_errors_csv}")
    else:
        stk_time = None
        stk_position_error = None
        stk_velocity_error = None
        stk_h_error = None
        stk_energy_error = None

    _plot_series("Position Error in x", "Position Error x [m]", r_err[:, 0], "x_position_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=stk_position_error[:, 0] if stk_position_error is not None else None,
                 log_y=log_scale)

    _plot_series("Position Error in y", "Position Error y [m]", r_err[:, 1], "y_position_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=stk_position_error[:, 1] if stk_position_error is not None else None,
                 log_y=log_scale)

    _plot_series("Position Error in z", "Position Error z [m]", r_err[:, 2], "z_position_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=stk_position_error[:, 2] if stk_position_error is not None else None,
                 log_y=log_scale)

    _plot_series("Position Error Magnitude", "Position Error ||r|| [m]", np.linalg.norm(r_err, axis=1), "position_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=np.linalg.norm(stk_position_error, axis=1) if stk_position_error is not None else None,
                 log_y=log_scale)

    # Velocity error plots include STK overlay when available
    _plot_series("Velocity Error in vx", "Velocity Error vx [m/s] (log scale)", v_err[:, 0], "vx_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=stk_velocity_error[:, 0] if stk_velocity_error is not None else None,
                 log_y=log_scale)

    _plot_series("Velocity Error in vy", "Velocity Error vy [m/s] (log scale)", v_err[:, 1], "vy_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=stk_velocity_error[:, 1] if stk_velocity_error is not None else None,
                 log_y=log_scale)

    _plot_series("Velocity Error in vz", "Velocity Error vz [m/s] (log scale)", v_err[:, 2], "vz_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=stk_velocity_error[:, 2] if stk_velocity_error is not None else None,
                 log_y=log_scale)

    _plot_series("Velocity Error Magnitude", "Velocity Error ||v|| [m/s] (log scale)", np.linalg.norm(v_err, axis=1), "velocity_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=np.linalg.norm(stk_velocity_error, axis=1) if stk_velocity_error is not None else None,
                 log_y=log_scale)

    _plot_series("Specific Angular Momentum Error", "Angular Momentum Error |h| [m^2/s] (log scale)", h_err, "angular_momentum_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=stk_h_error if stk_h_error is not None else None,
                 log_y=log_scale)

    _plot_series("Specific Orbital Energy Error", "Energy Error [J/kg] (log scale)", energy_err, "energy_error.png",
                 x_values=stk_time if stk_data is not None else None,
                 stky=stk_energy_error if stk_energy_error is not None else None,
                 log_y=log_scale)

    print(f"  Comparison CSV saved to {comparison_file}")
    print(f"  Validation plots saved to {validation_dir}")

    if show and plt is not None:
        plt.show()

    return comparison_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create validation comparison plots against analytical and STK references")
    parser.add_argument("--show", action="store_true", help="Display generated plots")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "output", help="Base output directory (integrator subdirectory will be appended)")
    parser.add_argument("--integrator", default="pd853", help="Integrator name subdirectory under --output-dir")
    parser.add_argument("--stk-csv", type=Path, default=None, help="Optional path to STK results CSV")
    parser.add_argument("--orbit", type=Path, default=repo_root / "STK_input" / "Satellite1.opm", help="Optional path to the OPM file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    integrator_dir = output_dir / args.integrator
    integrator_dir.mkdir(parents=True, exist_ok=True)
    orbit_params = _load_orbit_from_opm(args.orbit)
    create_comparison_plots(integrator_dir, orbit_params, integrator=args.integrator, stk_csv=args.stk_csv, show=args.show)


if __name__ == "__main__":
    main()
