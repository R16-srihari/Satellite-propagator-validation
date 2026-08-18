"""Tests for the comparative plotting module (``validation.comparative_plots``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import constants
from validation.comparative_plots import create_comparative_plots


def _make_orbit_params():
    """Return a minimal OrbitParameters-like object for report headers."""
    const = constants()
    a = 7000000.0
    return type(
        "OrbitParams",
        (),
        {
            "a": a,
            "e": 0.0,
            "i": 51.6 * const.deg2rad,
            "omega_big": 0.0,
            "omega_small": 0.0,
            "nu": 0.0,
            "period": const.twopi * np.sqrt(a**3 / const.mu_earth),
            "period_min": const.twopi * np.sqrt(a**3 / const.mu_earth) / 60.0,
            "energy": -const.mu_earth / (2.0 * a),
            "h_mag": np.sqrt(const.mu_earth * a),
            "epoch": None,
        },
    )()


def _write_comparison_csv(path: Path, n: int = 20) -> None:
    """Write a synthetic ``analytical_comparison.csv`` with all required columns."""
    t = np.linspace(0, 100, n)
    # Analytical position is circular in the xy-plane.
    omega = np.sqrt(constants().mu_earth / 7000000.0**3)
    r_ana = 7000000.0
    x_ana = r_ana * np.cos(omega * t)
    y_ana = r_ana * np.sin(omega * t)
    z_ana = np.zeros_like(t)
    # Numerical state has small additive drift so errors are non-zero.
    drift = np.linspace(0, 1.0, n)
    x_num = x_ana + drift * 1e-3
    y_num = y_ana + drift * 1e-3
    z_num = z_ana + drift * 1e-4

    vx_ana = -r_ana * omega * np.sin(omega * t)
    vy_ana = r_ana * omega * np.cos(omega * t)
    vz_ana = np.zeros_like(t)
    vx_num = vx_ana + drift * 1e-6
    vy_num = vy_ana + drift * 1e-6
    vz_num = vz_ana + drift * 1e-7

    r_error = np.linalg.norm(
        np.column_stack([x_num - x_ana, y_num - y_ana, z_num - z_ana]), axis=1
    )
    v_error = np.linalg.norm(
        np.column_stack([vx_num - vx_ana, vy_num - vy_ana, vz_num - vz_ana]), axis=1
    )

    energy_err = drift * 1e-9
    h_err = drift * 1e-6

    df = pd.DataFrame(
        {
            "time_s": t,
            "x_num_m": x_num,
            "y_num_m": y_num,
            "z_num_m": z_num,
            "r_num_m": np.linalg.norm(np.column_stack([x_num, y_num, z_num]), axis=1),
            "x_ana_m": x_ana,
            "y_ana_m": y_ana,
            "z_ana_m": z_ana,
            "r_ana_m": np.full_like(t, r_ana),
            "r_error_norm_m": r_error,
            "r_error_rel": r_error / r_ana,
            "vx_num_ms": vx_num,
            "vy_num_ms": vy_num,
            "vz_num_ms": vz_num,
            "v_num_ms": np.linalg.norm(np.column_stack([vx_num, vy_num, vz_num]), axis=1),
            "vx_ana_ms": vx_ana,
            "vy_ana_ms": vy_ana,
            "vz_ana_ms": vz_ana,
            "v_ana_ms": np.linalg.norm(np.column_stack([vx_ana, vy_ana, vz_ana]), axis=1),
            "v_error_norm_ms": v_error,
            "v_error_rel": v_error / np.linalg.norm(np.column_stack([vx_ana, vy_ana, vz_ana]), axis=1),
            "energy_error_Jkg": energy_err,
            "energy_error_rel": energy_err / np.abs(-constants().mu_earth / (2.0 * 7000000.0)),
            "|h|_error_m2s": h_err,
            "|h|_error_rel": h_err / np.sqrt(constants().mu_earth * 7000000.0),
        }
    )
    df.to_csv(path, index=False)


@pytest.fixture()
def fake_comparison_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create pd853/ and symplectic/ dirs each with an analytical_comparison.csv."""
    pd853_dir = tmp_path / "pd853"
    sym_dir = tmp_path / "symplectic"
    pd853_dir.mkdir()
    sym_dir.mkdir()
    _write_comparison_csv(pd853_dir / "analytical_comparison.csv", n=20)
    _write_comparison_csv(sym_dir / "analytical_comparison.csv", n=20)
    comp_dir = tmp_path / "comparison"
    return pd853_dir, sym_dir, comp_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_create_comparative_plots_generates_all_outputs(fake_comparison_dirs) -> None:
    pd853_dir, sym_dir, comp_dir = fake_comparison_dirs
    orbit_params = _make_orbit_params()

    result = create_comparative_plots(
        pd853_dir, sym_dir, output_dir=comp_dir, orbit_params=orbit_params
    )

    assert result == comp_dir

    expected_pngs = [
        "comparative_position_error.png",
        "comparative_velocity_error.png",
        "comparative_angular_momentum_error.png",
        "comparative_energy_error.png",
        "comparative_position_components.png",
        "comparative_velocity_components.png",
    ]
    for name in expected_pngs:
        assert (comp_dir / name).exists(), f"Missing plot: {name}"

    assert (comp_dir / "summary_statistics.csv").exists()
    assert (comp_dir / "comparison_report.md").exists()


def test_summary_statistics_has_both_integrators(fake_comparison_dirs) -> None:
    pd853_dir, sym_dir, comp_dir = fake_comparison_dirs

    create_comparative_plots(pd853_dir, sym_dir, output_dir=comp_dir)

    stats = pd.read_csv(comp_dir / "summary_statistics.csv")
    integrators = set(stats["integrator"].unique())
    assert integrators == {"pd853", "symplectic"}

    # max_abs, mean_abs and rms are finite numbers
    for col in ("max_abs", "mean_abs", "rms"):
        assert np.all(np.isfinite(stats[col].to_numpy(float)))

    # At least the four core metric types are present
    metric_names = set(stats["metric"].unique())
    assert "Position error ||r||" in metric_names
    assert "Velocity error ||v||" in metric_names
    assert "Angular momentum error |h|" in metric_names
    assert "Energy error" in metric_names


def test_report_contains_comparative_tables(fake_comparison_dirs) -> None:
    pd853_dir, sym_dir, comp_dir = fake_comparison_dirs
    orbit_params = _make_orbit_params()

    create_comparative_plots(
        pd853_dir, sym_dir, output_dir=comp_dir, orbit_params=orbit_params
    )

    report = (comp_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "# Integrator Comparison Report" in report
    assert "## Summary Statistics (RMS error)" in report
    assert "## Detailed Statistics" in report
    assert "## Per-Metric Winner" in report
    # Orbit parameters rendered
    assert "Semi-major axis" in report


def test_missing_comparison_file_raises(tmp_path: Path) -> None:
    """A missing analytical_comparison.csv should raise FileNotFoundError."""
    pd853_dir = tmp_path / "pd853"
    sym_dir = tmp_path / "symplectic"
    pd853_dir.mkdir()
    sym_dir.mkdir()
    # No CSV files written.
    with pytest.raises(FileNotFoundError):
        create_comparative_plots(pd853_dir, sym_dir, output_dir=tmp_path / "out")
