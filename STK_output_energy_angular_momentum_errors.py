"""Plot specific angular momentum and specific orbital energy errors.

This script reads an STK results CSV containing the Delaunay G variable and
semi-major axis, computes the specific orbital energy at each time step, and
plots the deviation of both quantities from their mean values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


MU_EARTH = 398600.4418  # km^3 / s^2

# Directory to write output files into (project root / output)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_time_series(csv_path: Path):
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


def save_energies_to_csv(times, angular_momentum, energy, csv_path: Path):
    """Write the time, angular momentum, and orbital energy to a CSV file."""

    # Make sure destination directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time (UTCG)", "Delaunay_G (km^2/sec)", "Specific_Orbital_Energy (km^2/s^2)"])
        for time, h, e in zip(times, angular_momentum, energy):
            writer.writerow([time, f"{h:.12f}", f"{e:.12f}"])


def plot_angular_momentum_error(times, angular_momentum, output_path: Path | None = None):
    """Plot deviations from the mean for specific angular momentum."""

    h_mean = float(np.mean(angular_momentum))
    time_index = np.arange(len(times))
    h_error = angular_momentum - h_mean

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_index, h_error, color="tab:blue", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("h - mean(h) [km^2/s]", fontsize=12)
    ax.set_title("Specific Angular Momentum Error", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Sample index", fontsize=12)

    tick_step = max(1, len(times) // 8)
    tick_positions = time_index[::tick_step]
    tick_labels = [times[index] for index in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, ax, h_mean


def plot_energy_error(times, energy, output_path: Path | None = None):
    """Plot deviations from the mean for specific orbital energy."""

    e_mean = float(np.mean(energy))
    time_index = np.arange(len(times))
    e_error = energy - e_mean

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_index, e_error, color="tab:orange", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("epsilon - mean(epsilon) [km^2/s^2]", fontsize=12)
    ax.set_title("Specific Orbital Energy Error", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Sample index", fontsize=12)

    tick_step = max(1, len(times) // 8)
    tick_positions = time_index[::tick_step]
    tick_labels = [times[index] for index in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, ax, e_mean


def main():
    parser = argparse.ArgumentParser(
        description="Compute specific orbital energy from semi-major axis and plot error graphs."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=Path(__file__).resolve().parent / "STK_input" / "ISS_ZARYA_25544_STK_results.csv",
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
        "--show",
        action="store_true",
        help="Display the plot windows after creating the figures.",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    times, angular_momentum, semi_major_axis = load_time_series(args.csv_path)
    energy = compute_energy(semi_major_axis)

    fig_h, ax_h, h_mean = plot_angular_momentum_error(times, angular_momentum, args.output_h)
    fig_e, ax_e, e_mean = plot_energy_error(times, energy, args.output_e)
    
    save_energies_to_csv(times, angular_momentum, energy, args.output_csv)

    print(f"Loaded {len(times)} samples from {args.csv_path}")
    print(f"Mean specific angular momentum: {h_mean:.12f} km^2/s")
    print(f"Mean specific orbital energy:    {e_mean:.12f} km^2/s^2")
    print(f"Angular momentum error figure saved to: {args.output_h}")
    print(f"Orbital energy error figure saved to: {args.output_e}")
    print(f"Energy time-series CSV saved to: {args.output_csv}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()