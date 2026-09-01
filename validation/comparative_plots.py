"""Comparative plotting between the pd853 and symplectic integrators.

This module reads the ``analytical_comparison.csv`` files produced by
:func:`validation.compare_analytical.compare_analytical` for both integrators
and generates overlay plots that directly compare their conservation errors
(position, velocity, angular momentum and energy) against the analytical
two-body reference.

All results are written to ``output/comparison/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import constants

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is required for plotting
    plt = None


# ---------------------------------------------------------------------------
# Colour / style helpers
# ---------------------------------------------------------------------------
_INTEGRATOR_STYLES: dict[str, dict] = {
    "pd853": {"color": "tab:blue", "linewidth": 2.0},
    "symplectic": {"color": "tab:orange", "linewidth": 2.0, "linestyle": "--"},
}


# ---------------------------------------------------------------------------
# Metric specifications: (display_name, column_in_csv, unit)
# ---------------------------------------------------------------------------
_METRIC_SPECS: list[tuple[str, str, str]] = [
    ("Position error ||r||", "r_error_norm_m", "m"),
    ("Position relative error", "r_error_rel", "1"),
    ("Velocity error ||v||", "v_error_norm_ms", "m/s"),
    ("Velocity relative error", "v_error_rel", "1"),
    ("Angular momentum error |h|", "|h|_error_m2s", "m^2/s"),
    ("Angular momentum relative error", "|h|_error_rel", "1"),
    ("Energy error", "energy_error_Jkg", "J/kg"),
    ("Energy relative error", "energy_error_rel", "1"),
]


# ---------------------------------------------------------------------------
# Loading / data helpers
# ---------------------------------------------------------------------------
def _load_comparison_df(path: Path) -> pd.DataFrame:
    """Load an ``analytical_comparison.csv`` file, raising on missing files."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing analytical comparison file: {path}. "
            "Run compare_analytical() for this integrator first."
        )
    return pd.read_csv(path)


def _component_errors(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return position / velocity component-wise errors for a comparison df.

    The ``analytical_comparison.csv`` stores numerical and analytical states
    separately, so component errors are reconstructed as ``num - ana``.
    """
    r_err = np.column_stack(
        (
            df["x_num_m"].to_numpy(float) - df["x_ana_m"].to_numpy(float),
            df["y_num_m"].to_numpy(float) - df["y_ana_m"].to_numpy(float),
            df["z_num_m"].to_numpy(float) - df["z_ana_m"].to_numpy(float),
        )
    )
    v_err = np.column_stack(
        (
            df["vx_num_ms"].to_numpy(float) - df["vx_ana_ms"].to_numpy(float),
            df["vy_num_ms"].to_numpy(float) - df["vy_ana_ms"].to_numpy(float),
            df["vz_num_ms"].to_numpy(float) - df["vz_ana_ms"].to_numpy(float),
        )
    )
    return {"r_err": r_err, "v_err": v_err}


def _interpolate_df(df: pd.DataFrame, target_times: np.ndarray) -> pd.DataFrame:
    """Interpolate every numeric column of *df* onto *target_times*."""
    result = pd.DataFrame({"time_s": target_times})
    for col in df.columns:
        if col == "time_s":
            continue
        result[col] = np.interp(
            target_times, df["time_s"].to_numpy(float), df[col].to_numpy(float)
        )
    return result


def _save_plot(fig, path: Path) -> None:
    """Apply tight layout, save to disk and close the figure."""
    if plt is None:
        return
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_overlay(
    title: str,
    y_label: str,
    series: dict[str, np.ndarray],
    t_values: np.ndarray,
    save_name: str,
    out_dir: Path,
    log_y: bool = False,
) -> None:
    """Create an overlay line plot of several integrator series vs. time.

    Parameters
    ----------
    series
        Mapping ``{integrator_name: y_values}`` to overlay.
    """
    if plt is None:
        return
    # Pop first element
    t_values = t_values[1:]
    series = {k: v[1:] for k, v in series.items()}
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, y_values in series.items():
        style = _INTEGRATOR_STYLES.get(name, {})
        ax.plot(
            t_values,
            y_values,
            label=name,
            linewidth=style.get("linewidth", 2.0),
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
        )
    if log_y:
        ax.set_yscale("symlog", linthresh=1e-15)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Integrator")
    _save_plot(fig, out_dir / save_name)


def _plot_component_overlay(
    title: str,
    y_label_unit: str,
    components: list[tuple[str, np.ndarray, np.ndarray]],
    t_values: np.ndarray,
    save_name: str,
    out_dir: Path,
    log_y: bool = False,
) -> None:
    """Create a 3-panel overlay plot for x/y/z or vx/vy/vz component errors.

    Each tuple in *components* is ``(label, pd853_values, symplectic_values)``.
    """
    if plt is None:
        return
    # Pop first element
    t_values = t_values[1:]
    components = [(label, pd_vals[1:], sym_vals[1:]) for label, pd_vals, sym_vals in components]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, (label, pd_vals, sym_vals) in zip(axes, components):
        ax.plot(
            t_values,
            pd_vals,
            label="pd853",
            linewidth=1.8,
            color=_INTEGRATOR_STYLES["pd853"]["color"],
        )
        ax.plot(
            t_values,
            sym_vals,
            label="symplectic",
            linewidth=1.8,
            color=_INTEGRATOR_STYLES["symplectic"]["color"],
            linestyle=_INTEGRATOR_STYLES["symplectic"]["linestyle"],
        )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        if log_y:
            ax.set_yscale("symlog", linthresh=1e-15)
        ax.set_ylabel(f"{label} [{y_label_unit}]" + (" (log scale)" if log_y else ""))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=14)
    _save_plot(fig, out_dir / save_name)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def _compute_summary_statistics(df: pd.DataFrame, integrator: str) -> pd.DataFrame:
    """Compute max / mean / RMS absolute errors for one integrator.

    Returns a DataFrame with one row per metric.
    """
    rows = []
    for name, col, unit in _METRIC_SPECS:
        if col not in df.columns:
            continue
        values = np.abs(df[col].to_numpy(float))
        record = {
            "integrator": integrator,
            "metric": name,
            "unit": unit,
            "max_abs": float(np.max(values)),
            "mean_abs": float(np.mean(values)),
            "rms": float(np.sqrt(np.mean(values**2))),
        }
        rows.append(record)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _generate_comparison_report(
    pd853_df: pd.DataFrame,
    symplectic_df: pd.DataFrame,
    orbit_params,
    out_dir: Path,
) -> Path:
    """Write a Markdown report comparing integrator conservation performance."""
    pd853_stats = _compute_summary_statistics(pd853_df, "pd853")
    sym_stats = _compute_summary_statistics(symplectic_df, "symplectic")
    all_stats = pd.concat([pd853_stats, sym_stats], ignore_index=True)

    # Build dict lookups keyed by metric name for clean float comparisons
    pd853_lookup = dict(zip(pd853_stats["metric"], pd853_stats["rms"]))
    sym_lookup = dict(zip(sym_stats["metric"], sym_stats["rms"]))
    units_lookup = dict(zip(pd853_stats["metric"], pd853_stats["unit"]))

    const = constants()
    lines: list[str] = []
    lines.append("# Integrator Comparison Report")
    lines.append("")
    lines.append("Comparison of **pd853** vs **symplectic** integrator conservation")
    lines.append("errors against the analytical two-body reference.")
    lines.append("")
    lines.append("## Orbit Parameters")
    lines.append("")
    lines.append(f"- Semi-major axis: **{orbit_params.a:.6e} m**")
    if orbit_params.e > 1e-9:
        lines.append(f"- Eccentricity: **{orbit_params.e:.6f}**")
    else:
        lines.append("- Eccentricity: **0.0** (circular)")
    lines.append(f"- Inclination: **{orbit_params.i * const.rad2deg:.1f} deg**")
    lines.append(
        f"- Orbital period: **{orbit_params.period:.2f} s** "
        f"({orbit_params.period_min:.2f} min)"
    )
    lines.append(f"- Specific orbital energy: **{orbit_params.energy:.6e} J/kg**")
    lines.append(f"- Angular momentum magnitude: **{orbit_params.h_mag:.6e} m^2/s**")
    if orbit_params.epoch is not None:
        lines.append(f"- Epoch (UTC): **{orbit_params.epoch.isoformat(sep=' ')}**")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("Both integrators share the same time grid (0-86400 s, 10 s steps).")
    lines.append("Errors are computed against the closed-form Keplerian two-body")
    lines.append("solution using `compare_analytical()`. This report overlays the")
    lines.append("per-time-step errors for each metric.")
    lines.append("")
    lines.append("## Summary Statistics (RMS error)")
    lines.append("")
    lines.append("| Metric | Unit | pd853 RMS | symplectic RMS | Ratio (sym/pd853) |")
    lines.append("|--------|------|-----------|----------------|--------------------|")
    for metric in pd853_lookup:
        unit = units_lookup[metric]
        pd_val = float(pd853_lookup[metric])
        sym_val = float(sym_lookup[metric])
        ratio = sym_val / pd_val if (pd_val != 0.0 and not np.isnan(pd_val)) else float("nan")
        lines.append(
            f"| {metric} | {unit} | {pd_val:.6e} | {sym_val:.6e} | {ratio:.4f} |"
        )
    lines.append("")
    lines.append("## Detailed Statistics")
    lines.append("")
    lines.append("| Integrator | Metric | Max | Mean | RMS |")
    lines.append("|------------|--------|-----|------|-----|")
    for _, row in all_stats.sort_values(["metric", "integrator"]).iterrows():
        lines.append(
            f"| {row['integrator']} | {row['metric']} | "
            f"{row['max_abs']:.6e} | {row['mean_abs']:.6e} | {row['rms']:.6e} |"
        )
    lines.append("")
    lines.append("## Per-Metric Winner")
    lines.append("")
    lines.append("| Metric | Better Integrator |")
    lines.append("|--------|-------------------|")
    for metric in pd853_lookup:
        pd_val = float(pd853_lookup[metric])
        sym_val = float(sym_lookup[metric])
        if np.isnan(pd_val) or np.isnan(sym_val):
            winner = "N/A"
        elif pd_val <= sym_val:
            winner = "pd853"
        else:
            winner = "symplectic"
        lines.append(f"| {metric} | {winner} |")
    lines.append("")
    lines.append("## Generated Plots")
    lines.append("")
    for p in sorted(out_dir.glob("*.png")):
        lines.append(f"- `{p.name}`")
    lines.append("")

    report_path = out_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Comparison report saved to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def create_comparative_plots(
    pd853_dir: str | Path,
    symplectic_dir: str | Path,
    output_dir: str | Path | None = None,
    orbit_params=None,
    show: bool = False,
    log_scale: bool = True,
) -> Path:
    """Create overlay comparison plots between pd853 and symplectic integrators.

    Reads ``analytical_comparison.csv`` from each integrator directory and
    writes comparative PNG plots, a ``summary_statistics.csv`` and a
    ``comparison_report.md`` to ``output/comparison/``.

    Parameters
    ----------
    pd853_dir
        Directory containing pd853's ``analytical_comparison.csv``.
    symplectic_dir
        Directory containing symplectic's ``analytical_comparison.csv``.
    output_dir
        Output directory for comparative results.  Defaults to
        ``<parent of pd853_dir>/comparison``.
    orbit_params
        :class:`~src.orbital_parameters.OrbitParameters` instance used in the
        report header.  Optional.
    show
        If ``True``, display plots interactively (requires a display backend).

    Returns
    -------
    Path
        The comparative output directory.
    """
    pd853_path = Path(pd853_dir)
    symplectic_path = Path(symplectic_dir)

    if output_dir is None:
        output_dir = pd853_path.parent / "comparison"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading analytical comparison data ...")
    pd853_df = _load_comparison_df(pd853_path / "analytical_comparison.csv")
    symplectic_df = _load_comparison_df(symplectic_path / "analytical_comparison.csv")

    # Verify compatible time grids; interpolate if they differ.
    t_pd = pd853_df["time_s"].to_numpy(float)
    t_sym = symplectic_df["time_s"].to_numpy(float)
    if t_pd.size != t_sym.size or not np.allclose(t_pd, t_sym, rtol=0, atol=1e-6):
        print("  Warning: time grids differ between integrators; interpolating ...")
        common_t = np.union1d(t_pd, t_sym)
        pd853_df = _interpolate_df(pd853_df, common_t)
        symplectic_df = _interpolate_df(symplectic_df, common_t)
    else:
        common_t = t_pd

    # Component-wise errors
    pd853_comp = _component_errors(pd853_df)
    sym_comp = _component_errors(symplectic_df)

    # --- Magnitude overlay plots ----------------------------------------
    _plot_overlay(
        "Position Error Magnitude (||r||) - pd853 vs symplectic",
        "Position Error ||r|| [m] (log scale)",
        {
            "pd853": pd853_df["r_error_norm_m"].to_numpy(float),
            "symplectic": symplectic_df["r_error_norm_m"].to_numpy(float),
        },
        common_t,
        "comparative_position_error.png",
        out_dir,
        log_y=log_scale,
    )

    _plot_overlay(
        "Velocity Error Magnitude (||v||) - pd853 vs symplectic",
        "Velocity Error ||v|| [m/s] (log scale)",
        {
            "pd853": pd853_df["v_error_norm_ms"].to_numpy(float),
            "symplectic": symplectic_df["v_error_norm_ms"].to_numpy(float),
        },
        common_t,
        "comparative_velocity_error.png",
        out_dir,
        log_y=log_scale,
    )

    _plot_overlay(
        "Specific Angular Momentum Error (|h|) - pd853 vs symplectic",
        "Angular Momentum Error |h| [m^2/s] (log scale)",
        {
            "pd853": pd853_df["|h|_error_m2s"].to_numpy(float),
            "symplectic": symplectic_df["|h|_error_m2s"].to_numpy(float),
        },
        common_t,
        "comparative_angular_momentum_error.png",
        out_dir,
        log_y=log_scale,
    )

    _plot_overlay(
        "Specific Orbital Energy Error (epsilon) - pd853 vs symplectic",
        "Energy Error [J/kg] (log scale)",
        {
            "pd853": pd853_df["energy_error_Jkg"].to_numpy(float),
            "symplectic": symplectic_df["energy_error_Jkg"].to_numpy(float),
        },
        common_t,
        "comparative_energy_error.png",
        out_dir,
        log_y=log_scale,
    )

    # --- Component overlay plots ----------------------------------------
    _plot_component_overlay(
        "Position Error Components - pd853 vs symplectic",
        "m",
        [
            ("x", pd853_comp["r_err"][:, 0], sym_comp["r_err"][:, 0]),
            ("y", pd853_comp["r_err"][:, 1], sym_comp["r_err"][:, 1]),
            ("z", pd853_comp["r_err"][:, 2], sym_comp["r_err"][:, 2]),
        ],
        common_t,
        "comparative_position_components.png",
        out_dir,
        log_y=log_scale,
    )

    _plot_component_overlay(
        "Velocity Error Components - pd853 vs symplectic",
        "m/s",
        [
            ("vx", pd853_comp["v_err"][:, 0], sym_comp["v_err"][:, 0]),
            ("vy", pd853_comp["v_err"][:, 1], sym_comp["v_err"][:, 1]),
            ("vz", pd853_comp["v_err"][:, 2], sym_comp["v_err"][:, 2]),
        ],
        common_t,
        "comparative_velocity_components.png",
        out_dir,
        log_y=log_scale,
    )

    # --- Summary statistics ---------------------------------------------
    pd853_stats = _compute_summary_statistics(pd853_df, "pd853")
    sym_stats = _compute_summary_statistics(symplectic_df, "symplectic")
    all_stats = pd.concat([pd853_stats, sym_stats], ignore_index=True)
    stats_file = out_dir / "summary_statistics.csv"
    all_stats.to_csv(stats_file, index=False)
    print(f"  Summary statistics saved to {stats_file}")

    # --- Report ---------------------------------------------------------
    if orbit_params is not None:
        _generate_comparison_report(pd853_df, symplectic_df, orbit_params, out_dir)

    if show and plt is not None:
        plt.show()

    print(f"  Comparative plots saved to {out_dir}")
    return out_dir
