"""Plot specific angular momentum and specific orbital energy errors.

This script reads an STK results CSV containing the Delaunay G variable and
semi-major axis, computes the specific orbital energy at each time step, and
plots the deviation of both quantities from their mean values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence, TYPE_CHECKING, Any

import matplotlib
# Backend selection and pyplot import are deferred until `main()` to avoid
# parsing CLI args or changing matplotlib state at import time. Plotting
# functions import `matplotlib.pyplot` lazily so importing this module is
# side-effect free.
import numpy as np


MU_EARTH = 398600.4418  # km^3 / s^2

# Directory to write output files into (project root / output)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

# Placeholder for pyplot for the type-checker; assigned at runtime if --show
plt: Any = None


def load_time_series(csv_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load the time, Delaunay G, and semi-major axis series from the CSV.
    
    Stops reading when encountering a row with "Statistics" in the first column.
    """

    times = []
    delaunay_g = []
    semi_major_axis = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            time_value = row.get("Time (UTCG)", "")
            
            # Stop reading if we hit the Statistics section
            if time_value and "Statistics" in time_value:
                break
            
            g_value = row.get("Delaunay_G (km^2/sec)", "")
            a_value = row.get("Semi-major Axis (km)", "")

            if not time_value or not g_value or not a_value:
                continue

            try:
                delaunay_g_value = float(g_value)
                semi_major_axis_value = float(a_value)
            except ValueError:
                continue

            times.append(time_value)
            delaunay_g.append(delaunay_g_value)
            semi_major_axis.append(semi_major_axis_value)

    if not times:
        raise ValueError(f"No valid time-series rows were found in {csv_path}")

    return times, np.asarray(delaunay_g, dtype=float), np.asarray(semi_major_axis, dtype=float)


def compute_energy(semi_major_axis: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """Return the specific orbital energy for each semi-major axis value."""

    return -mu / (2.0 * semi_major_axis)


def parse_opm(opm_path: Path) -> dict:
    """Parse a minimal set of orbital elements from an OPM file.

    Returns a dict with keys: 'a', 'e', 'i', 'raan', 'argp', 'nu', 'gm'.
    Values are in the same units as the file (typically km and degrees).
    """
    fields: dict[str, float | None] = {
        "SEMI_MAJOR_AXIS": None,
        "ECCENTRICITY": None,
        "INCLINATION": None,
        "RA_OF_ASC_NODE": None,
        "ARG_OF_PERICENTER": None,
        "TRUE_ANOMALY": None,
        "GM": None,
    }

    with opm_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("COMMENT"):
                continue
            if "=" not in line:
                continue
            key, val = [p.strip() for p in line.split("=", 1)]
            if key in fields:
                try:
                    fields[key] = float(val)
                except ValueError:
                    # ignore unparsable values
                    pass

    if fields["SEMI_MAJOR_AXIS"] is None or fields["ECCENTRICITY"] is None:
        raise ValueError(f"OPM {opm_path} is missing required elements")

    return {
        "a": fields["SEMI_MAJOR_AXIS"],
        "e": fields["ECCENTRICITY"],
        "i": fields["INCLINATION"],
        "raan": fields["RA_OF_ASC_NODE"],
        "argp": fields["ARG_OF_PERICENTER"],
        "nu": fields["TRUE_ANOMALY"],
        "gm": fields["GM"],
    }


def save_energies_to_csv(
    times: Sequence[str], angular_momentum: np.ndarray, energy: np.ndarray, csv_path: Path
) -> None:
    """Write the time, angular momentum, and orbital energy to a CSV file."""

    # Make sure destination directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time (UTCG)", "Delaunay_G (km^2/sec)", "Specific_Orbital_Energy (km^2/s^2)"])
        for time, h, e in zip(times, angular_momentum, energy):
            writer.writerow([time, f"{h:.12f}", f"{e:.12f}"])


def save_errors_to_csv(
    times: Sequence[str], h_error: np.ndarray, e_error: np.ndarray, csv_path: Path
) -> None:
    """Write the time, angular momentum error, and energy error to a CSV file."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time (UTCG)", "Delaunay_G_error (km^2/sec)", "Specific_Orbital_Energy_error (km^2/s^2)"])
        for time, he, ee in zip(times, h_error, e_error):
            writer.writerow([time, f"{he:.12e}", f"{ee:.12e}"])


def plot_angular_momentum_error(
    times: Sequence[str],
    angular_momentum: np.ndarray,
    output_path: Path | None = None,
    reference_h: float | None = None,
) -> tuple["Figure", "Axes", float]:
    """Plot deviations from the mean for specific angular momentum."""
    import matplotlib.pyplot as plt

    h_mean = float(np.mean(angular_momentum))
    if reference_h is not None:
        ref_val = float(reference_h)
    else:
        ref_val = h_mean
    time_index = np.arange(len(times))
    h_error = angular_momentum - ref_val

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_index, h_error, color="tab:blue", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("h - mean(h) [km^2/s]", fontsize=12)
    ax.set_title("Specific Angular Momentum Error", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Sample index", fontsize=12)

    tick_step = max(1, len(times) // 8)
    tick_positions = time_index[::tick_step]
    # Convert numpy indices to ints and ensure labels are plain strings
    tick_labels = [str(times[int(index)]) for index in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, ax, ref_val


def plot_energy_error(
    times: Sequence[str], energy: np.ndarray, output_path: Path | None = None, reference_e: float | None = None
) -> tuple["Figure", "Axes", float]:
    """Plot deviations from the mean for specific orbital energy."""
    import matplotlib.pyplot as plt

    e_mean = float(np.mean(energy))
    if reference_e is not None:
        ref_val = float(reference_e)
    else:
        ref_val = e_mean
    time_index = np.arange(len(times))
    e_error = energy - ref_val

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_index, e_error, color="tab:orange", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("epsilon - mean(epsilon) [km^2/s^2]", fontsize=12)
    ax.set_title("Specific Orbital Energy Error", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Sample index", fontsize=12)

    tick_step = max(1, len(times) // 8)
    tick_positions = time_index[::tick_step]
    # Convert numpy indices to ints and ensure labels are plain strings
    tick_labels = [str(times[int(index)]) for index in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, ax, ref_val


def main():
    parser = argparse.ArgumentParser(
        description="Compute specific orbital energy from semi-major axis and plot error graphs."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=Path(__file__).resolve().parent / "STK_input" / "Satellite1_Results.csv",
        type=Path,
        help="Path to the STK results CSV file.",
    )
    parser.add_argument(
        "--output-h",
        type=Path,
        default=OUTPUT_DIR / "STK_angular_momentum_error.png",
        help="Path to save the angular momentum error figure.",
    )
    parser.add_argument(
        "--output-e",
        type=Path,
        default=OUTPUT_DIR / "STK_energy_error.png",
        help="Path to save the energy error figure.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_DIR / "STK_energy_timeseries.csv",
        help="Path to save the computed energies as CSV.",
    )
    parser.add_argument(
        "--opm",
        type=Path,
        default=Path(__file__).resolve().parent / "STK_input" / "Satellite1.opm",
        help="Path to the Satellite OPM file to extract analytical initial elements.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot windows after creating the figures.",
    )
    args = parser.parse_args()

    # Select backend after parsing arguments so importing this module has no
    # side-effects. Use non-interactive Agg when not showing windows.
    if not args.show:
        matplotlib.use("Agg")

    # Import pyplot only when needed for interactive display and assign
    # it to the module-level name so `plt.show()` is always valid.
    if args.show:
        global plt
        import matplotlib.pyplot as _plt
        plt = _plt

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    times, angular_momentum, semi_major_axis = load_time_series(args.csv_path)
    energy = compute_energy(semi_major_axis)

    # If an OPM is provided and exists, compute analytical reference h and energy
    reference_h = None
    reference_e = None
    try:
        if args.opm and args.opm.exists():
            opm = parse_opm(args.opm)
            print(f"Using OPM file for analytical reference: {args.opm.resolve()}")
            a0 = float(opm["a"])  # km
            e0 = float(opm["e"])  # unitless
            gm = float(opm.get("gm", MU_EARTH))
            # specific angular momentum magnitude for Keplerian orbit: h = sqrt(mu * a * (1 - e^2))
            reference_h = float(np.sqrt(gm * a0 * (1.0 - e0 * e0)))
            reference_e = -float(gm / (2.0 * a0))
    except Exception as exc:  # keep running even if OPM parse fails
        print(f"Warning: failed to parse OPM {args.opm}: {exc}")

    # Compute error time series against the chosen references (analytical or mean)
    h_ref_for_errors = reference_h if reference_h is not None else float(np.mean(angular_momentum))
    e_ref_for_errors = reference_e if reference_e is not None else float(np.mean(energy))

    h_error_series = angular_momentum - float(h_ref_for_errors)
    e_error_series = energy - float(e_ref_for_errors)

    errors_csv = OUTPUT_DIR / "STK_errors.csv"
    save_errors_to_csv(times, h_error_series, e_error_series, errors_csv)

    fig_h, ax_h, h_ref = plot_angular_momentum_error(
        times, angular_momentum, args.output_h, reference_h=reference_h
    )
    fig_e, ax_e, e_ref = plot_energy_error(times, energy, args.output_e, reference_e=reference_e)
    
    save_energies_to_csv(times, angular_momentum, energy, args.output_csv)

    print(f"Loaded {len(times)} samples from {args.csv_path}")
    print(f"Reference specific angular momentum: {h_ref:.12f} km^2/s")
    print(f"Reference specific orbital energy:    {e_ref:.12f} km^2/s^2")
    print(f"Angular momentum error figure saved to: {args.output_h}")
    print(f"Orbital energy error figure saved to: {args.output_e}")
    print(f"Energy time-series CSV saved to: {args.output_csv}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()