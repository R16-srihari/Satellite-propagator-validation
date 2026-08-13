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
    if stk_csv is None:
        repo_root = Path(__file__).resolve().parents[1]
        stk_csv = repo_root / "STK_input" / "Satellite1_Results.csv"
    if not stk_csv.exists():
        return None

    try:
        df = pd.read_csv(stk_csv, on_bad_lines="skip")
    except Exception:
        return None

    if "Time (UTCG)" not in df.columns:
        return None

    time_series = df["Time (UTCG)"].astype(str)
    valid_mask = ~time_series.str.contains("Statistics", case=False, na=False)
    df = df.loc[valid_mask].copy()
    if df.empty:
        return None

    required_cols = {"x (km)", "y (km)", "z (km)", "Delaunay_G (m^2/sec)", "Semimajor_Axis (m)"}
    if not required_cols.issubset(df.columns):
        return None

    try:
        stk_epoch = orbit_params.epoch
        if stk_epoch is None:
            first_time = str(df["Time (UTCG)"].iloc[0])
            try:
                stk_epoch = datetime.strptime(first_time, "%d %b %Y %H:%M:%S.%f")
            except ValueError:
                stk_epoch = datetime.strptime(first_time, "%d %b %Y %H:%M:%S")
        times = pd.to_datetime(df["Time (UTCG)"], format="%d %b %Y %H:%M:%S.%f", errors="coerce")
        if times.isna().all():
            times = pd.to_datetime(df["Time (UTCG)"], format="%d %b %Y %H:%M:%S", errors="coerce")
    except Exception:
        return None

    valid = times.notna()
    df = df.loc[valid].copy()
    times = times[valid]
    if stk_epoch is None or df.empty:
        return None

    seconds_from_epoch = (times - pd.Timestamp(stk_epoch)).dt.total_seconds().to_numpy(float)
    pos_m = np.column_stack((
        df["x (km)"].to_numpy(float) * 1000.0,
        df["y (km)"].to_numpy(float) * 1000.0,
        df["z (km)"].to_numpy(float) * 1000.0,
    ))
    return {
        "time_s": seconds_from_epoch,
        "position_m": pos_m,
        "h_mag": df["Delaunay_G (m^2/sec)"].to_numpy(float),
        "energy_Jkg": -constants().mu_earth / (2.0 * df["Semimajor_Axis (m)"].to_numpy(float)),
    }


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


def create_comparison_plots(output_dir, orbit_params, integrator="rk78", stk_csv=None, show=False):
    """Create a validation plot set comparing the integrator against analytical and STK references."""
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

    def _plot_series(title: str, y_label: str, y_values: np.ndarray, save_name: str, x_values=None, stky=None, stklab="STK"):
        if plt is None:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_values, y_values, label=f"{integrator}", linewidth=2.0)
        if x_values is not None and stky is not None:
            ax.plot(x_values, stky, label=stklab, linestyle="--", linewidth=1.5)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save_plot(fig, validation_dir / save_name)

    stk_time = np.asarray(stk_data["time_s"], dtype=float) if stk_data is not None else None
    if stk_data is not None:
        assert stk_time is not None
        st_ana = np.column_stack([np.interp(stk_time, t_values, r_ana[:, i]) for i in range(3)])
        stk_position_error = stk_data["position_m"] - st_ana
        analytic_h_at_stk = np.interp(stk_time, t_values, h_ana_mag)
        analytic_energy_at_stk = np.interp(stk_time, t_values, orbit_params.energy * np.ones_like(t_values))
    else:
        stk_position_error = None
        analytic_h_at_stk = None
        analytic_energy_at_stk = None

    _plot_series("Position Error in x", "Δx [m]", r_err[:, 0], "x_position_error.png",
                 x_values=stk_time,
                 stky=stk_position_error[:, 0] if stk_position_error is not None else None)

    _plot_series("Position Error in y", "Δy [m]", r_err[:, 1], "y_position_error.png",
                 x_values=stk_time,
                 stky=stk_position_error[:, 1] if stk_position_error is not None else None)

    _plot_series("Position Error in z", "Δz [m]", r_err[:, 2], "z_position_error.png",
                 x_values=stk_time,
                 stky=stk_position_error[:, 2] if stk_position_error is not None else None)

    _plot_series("Position Error Magnitude", "||Δr|| [m]", np.linalg.norm(r_err, axis=1), "position_error.png",
                 x_values=stk_time,
                 stky=np.linalg.norm(stk_position_error, axis=1) if stk_position_error is not None else None)

    _plot_series("Velocity Error in vx", "Δvx [m/s]", v_err[:, 0], "vx_error.png")
    _plot_series("Velocity Error in vy", "Δvy [m/s]", v_err[:, 1], "vy_error.png")
    _plot_series("Velocity Error in vz", "Δvz [m/s]", v_err[:, 2], "vz_error.png")
    _plot_series("Velocity Error Magnitude", "||Δv|| [m/s]", np.linalg.norm(v_err, axis=1), "velocity_error.png")
    _plot_series("Specific Angular Momentum Error", "|Δh| [m^2/s]", h_err, "angular_momentum_error.png",
                 x_values=stk_time,
                 stky=np.abs(stk_data["h_mag"] - analytic_h_at_stk) if stk_data is not None else None)
    _plot_series("Specific Orbital Energy Error", "Δε [J/kg]", energy_err, "energy_error.png",
                 x_values=stk_time,
                 stky=np.abs(stk_data["energy_Jkg"] - analytic_energy_at_stk) if stk_data is not None else None)

    print(f"  Comparison CSV saved to {comparison_file}")
    print(f"  Validation plots saved to {validation_dir}")

    if show and plt is not None:
        plt.show()

    return comparison_df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create validation comparison plots against analytical and STK references")
    parser.add_argument("--show", action="store_true", help="Display generated plots")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "output" / "validation", help="Directory for generated validation outputs")
    parser.add_argument("--integrator", default="rk78", help="Integrator name for the plot label")
    parser.add_argument("--stk-csv", type=Path, default=None, help="Optional path to STK results CSV")
    parser.add_argument("--orbit", type=Path, default=repo_root / "STK_input" / "Satellite1.opm", help="Optional path to the OPM file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orbit_params = _load_orbit_from_opm(args.orbit)
    create_comparison_plots(output_dir, orbit_params, integrator=args.integrator, stk_csv=args.stk_csv, show=args.show)


if __name__ == "__main__":
    main()
