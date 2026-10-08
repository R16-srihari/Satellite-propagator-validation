"""Compute STK-vs-analytical two-body errors and write output/STK_errors.csv.

This script reads the STK results CSV (Cartesian states, Delaunay G, and
semi-major axis) and the OPM initial elements, propagates the analytical
Keplerian reference over the same time span, and writes an error CSV in the
project-standard comparison format:

    time_s, r_error_norm_m, v_error_norm_ms, vx_error_ms, vy_error_ms,
    vz_error_ms, h_error_m2s, energy_error_Jkg
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytical_solution import analytical_solution
from src.constants import constants

# Directory to write output files into (project root / output)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_STK_CSV = Path(__file__).resolve().parent / "STK_input" / "Satellite1_Results.csv"
DEFAULT_OPM = Path(__file__).resolve().parent / "STK_input" / "Satellite1.opm"

# Column layout of the output error CSV (matches STK_errors.csv format)
ERROR_COLUMNS = [
    "time_s",
    "r_error_norm_m",
    "v_error_norm_ms",
    "vx_error_ms",
    "vy_error_ms",
    "vz_error_ms",
    "h_error_m2s",
    "energy_error_Jkg",
]


def parse_opm(opm_path: Path) -> dict[str, str]:
    """Parse an OPM file into a raw key/value dictionary."""
    data: dict[str, str] = {}
    with opm_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("COMMENT"):
                continue
            if "=" not in line:
                continue
            key, val = (part.strip() for part in line.split("=", 1))
            data[key] = val
    return data


def opm_epoch(opm: dict[str, str]) -> datetime:
    """Return the EPOCH entry of a parsed OPM file."""
    raw = opm.get("EPOCH")
    if not raw:
        raise ValueError("OPM file is missing an EPOCH entry")
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Could not parse OPM EPOCH: {raw!r}") from exc


def opm_elements(opm: dict[str, str]) -> dict[str, float]:
    """Extract Keplerian elements (SI units) from a parsed OPM file."""
    const = constants()
    required = (
        "SEMI_MAJOR_AXIS",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "TRUE_ANOMALY",
    )
    missing = [key for key in required if key not in opm]
    if missing:
        raise ValueError(f"OPM file is missing required elements: {', '.join(missing)}")

    return {
        "a": float(opm["SEMI_MAJOR_AXIS"]) * 1000.0,
        "e": float(opm["ECCENTRICITY"]),
        "i": float(opm["INCLINATION"]) * const.deg2rad,
        "raan": float(opm["RA_OF_ASC_NODE"]) * const.deg2rad,
        "argp": float(opm["ARG_OF_PERICENTER"]) * const.deg2rad,
        "nu": float(opm["TRUE_ANOMALY"]) * const.deg2rad,
    }


def load_stk_series(csv_path: Path) -> pd.DataFrame:
    """Load the STK results CSV and convert km-based columns to SI units.

    Returns a DataFrame with columns: time_s (seconds from the first row),
    x_m, y_m, z_m and, when present in the file, vx_ms, vy_ms, vz_ms,
    h_mag (m^2/s) and energy_Jkg (m^2/s^2).
    """
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    col_lower = {column.lower().strip(): column for column in df.columns}

    def find_col(*candidates: str) -> str | None:
        for candidate in candidates:
            found = col_lower.get(candidate.lower().strip())
            if found is not None:
                return found
        return None

    time_col = find_col("Time (UTCG)", "time_utc", "time")
    x_col = find_col("X (km)")
    y_col = find_col("Y (km)")
    z_col = find_col("Z (km)")
    vx_col = find_col("Vx (km/sec)", "Vx (km/s)", "vx")
    vy_col = find_col("Vy (km/sec)", "Vy (km/s)", "vy")
    vz_col = find_col("Vz (km/sec)", "Vz (km/s)", "vz")
    delaunay_col = find_col("Delaunay_G (km^2/sec)", "Delaunay_G (m^2/sec)", "Delaunay_G")
    sma_col = find_col("Semimajor_Axis (km)", "Semimajor_Axis (m)", "Semi-major Axis (km)")

    if any(col is None for col in (time_col, x_col, y_col, z_col)):
        raise ValueError(f"CSV {csv_path} is missing required time/position columns")

    time_series = df[time_col].astype(str)
    df = df.loc[~time_series.str.contains("Statistics", case=False, na=False)].copy()
    if df.empty:
        raise ValueError(f"No valid time-series rows were found in {csv_path}")

    times = pd.to_datetime(df[time_col], format="%d %b %Y %H:%M:%S.%f", errors="coerce")
    if times.isna().all():
        times = pd.to_datetime(df[time_col], format="%d %b %Y %H:%M:%S", errors="coerce")
    if times.isna().all():
        raise ValueError(f"Could not parse the time stamps in {csv_path}")

    epoch = times.iloc[0]
    seconds = (times - epoch).dt.total_seconds().to_numpy(float)

    out = pd.DataFrame(
        {
            "time_s": seconds,
            "x_m": df[x_col].to_numpy(float) * 1000.0,
            "y_m": df[y_col].to_numpy(float) * 1000.0,
            "z_m": df[z_col].to_numpy(float) * 1000.0,
        }
    )

    if None not in (vx_col, vy_col, vz_col):
        out["vx_ms"] = df[vx_col].to_numpy(float) * 1000.0
        out["vy_ms"] = df[vy_col].to_numpy(float) * 1000.0
        out["vz_ms"] = df[vz_col].to_numpy(float) * 1000.0

    if delaunay_col is not None:
        scale = 1.0 if delaunay_col.lower().endswith("(m^2/sec)") else 1e6
        out["h_mag"] = df[delaunay_col].to_numpy(float) * scale

    if sma_col is not None:
        scale = 1.0 if sma_col.lower().endswith("(m)") else 1000.0
        a_m = df[sma_col].to_numpy(float) * scale
        out["energy_Jkg"] = -constants().mu_earth / (2.0 * a_m)

    time_values = out["time_s"].to_numpy(float)
    if not np.all(np.diff(time_values) > 0):
        raise ValueError("STK time column must be strictly increasing")

    return out


def analytical_reference(times: np.ndarray, elements: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the analytical two-body state at the given times [s]."""
    mu = constants().mu_earth
    r_ana = np.zeros((times.size, 3), dtype=float)
    v_ana = np.zeros((times.size, 3), dtype=float)
    for index, t_value in enumerate(times):
        r_ana[index], v_ana[index] = analytical_solution(
            float(t_value),
            elements["a"],
            elements["e"],
            elements["i"],
            elements["raan"],
            elements["argp"],
            elements["nu"],
            mu,
        )
    return r_ana, v_ana

def _print_scalar_stats(label: str, analytical: np.ndarray, numerical: np.ndarray,err_abs:np.ndarray,err_rel:np.ndarray, unit: str) -> None:
    
    def max_abs(arr):
        return np.max(np.abs(arr))
    def mean_abs(arr):
        return np.mean(np.abs(arr))
    print(f"\n  {label} comparison:")
    print(f"  {label} analytical value:        {analytical[0]:.15e} {unit}")
    print(f"  {label} mean numerical value:    {np.mean(numerical):.15e} {unit}")
    print(f"  {label} mean abs error:          {mean_abs(err_abs):.6e} {unit}")
    print(f"  {label} max abs error:           {max_abs(err_abs):.6e} {unit}")
    print(f"  {label} mean rel error:          {mean_abs(err_rel):.6e}")
    print(f"  {label} max rel error:           {max_abs(err_rel):.6e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute errors between STK results and analytical solution."
    )
    parser.add_argument(
        "--stk-csv",
        type=Path,
        default=DEFAULT_STK_CSV,
        help="Path to the STK results CSV file.",
    )
    parser.add_argument(
        "--opm",
        type=Path,
        default=DEFAULT_OPM,
        help="Path to the Satellite OPM file to extract analytical initial elements.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_DIR / "STK_errors.csv",
        help="Path to save the error CSV.",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load STK data
    stk_df = load_stk_series(args.stk_csv)

    # Load OPM data
    if not args.opm.exists():
        raise FileNotFoundError(f"OPM file not found: {args.opm}")
    opm_raw = parse_opm(args.opm)
    elements = opm_elements(opm_raw)

    # Compute analytical reference on STK time grid
    t_values = stk_df["time_s"].to_numpy(float)
    r_ana, v_ana = analytical_reference(t_values, elements)

    # Compute errors
    # Position error
    r_err = stk_df[["x_m", "y_m", "z_m"]].to_numpy() - r_ana
    r_error_norm = np.linalg.norm(r_err, axis=1)

    # Velocity error (if available)
    if {"vx_ms", "vy_ms", "vz_ms"}.issubset(stk_df.columns):
        v_err = stk_df[["vx_ms", "vy_ms", "vz_ms"]].to_numpy() - v_ana
        v_error_norm = np.linalg.norm(v_err, axis=1)
        vx_err = v_err[:, 0]
        vy_err = v_err[:, 1]
        vz_err = v_err[:, 2]
    else:
        v_error_norm = np.full_like(t_values, np.nan)
        vx_err = np.full_like(t_values, np.nan)
        vy_err = np.full_like(t_values, np.nan)
        vz_err = np.full_like(t_values, np.nan)

    # Angular momentum error
    if "h_mag" in stk_df.columns:
        # analytical h magnitude
        h_ana = np.linalg.norm(np.cross(r_ana, v_ana), axis=1)
        h_error = np.abs(stk_df["h_mag"].to_numpy() - h_ana)
    else:
        h_error = np.full_like(t_values, np.nan)

    # Energy error
    if "energy_Jkg" in stk_df.columns:
        # analytical energy
        v_mag = np.linalg.norm(v_ana, axis=1)
        r_mag = np.linalg.norm(r_ana, axis=1)
        e_ana = 0.5 * v_mag**2 - constants().mu_earth / np.clip(r_mag, 1e-12, None)
        energy_error = np.abs(stk_df["energy_Jkg"].to_numpy() - e_ana)
    else:
        energy_error = np.full_like(t_values, np.nan)

    # Print statistics (reuse helper from elsewhere? we'll inline simple prints)
    def _print_stats(label, err, unit):
        print(f"\n  {label} error:")
        print(f"  mean abs: {np.nanmean(np.abs(err)):.6e} {unit}")
        print(f"  max abs:  {np.nanmax(np.abs(err)):.6e} {unit}")

    _print_stats("Position", r_error_norm, "m")
    _print_stats("Velocity", v_error_norm, "m/s")
    _print_stats("Angular Momentum", h_error, "m^2/s")
    _print_stats("Energy", energy_error, "J/kg")

    # Save to CSV
    out_df = pd.DataFrame({
        "time_s": t_values,
        "x_error_m": r_err[:, 0],
        "y_error_m": r_err[:, 1],
        "z_error_m": r_err[:, 2],
        "r_error_norm_m": r_error_norm,
        "vx_error_ms": vx_err,
        "vy_error_ms": vy_err,
        "vz_error_ms": vz_err,
        "v_error_norm_ms": v_error_norm,
        "h_error_m2s": h_error,
        "energy_error_Jkg": energy_error,
    })
    out_df.to_csv(args.output_csv, index=False, float_format="%.12e")
    print(f"\nSTK errors saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
