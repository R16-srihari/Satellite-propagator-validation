from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import constants
from validation.compare_analytical import create_comparison_plots


def test_create_comparison_plots_generates_expected_error_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "run_rk78"
    output_dir.mkdir()

    t = np.array([0.0, 10.0, 20.0], dtype=float)
    mu = constants().mu_earth
    a = 7000000.0
    e = 0.01
    i = 51.6 * constants().deg2rad
    omega_big = 0.2
    omega_small = 0.3
    nu = 0.1

    r0 = np.array([7000000.0, 0.0, 0.0], dtype=float)
    v0 = np.array([0.0, 7800.0, 0.0], dtype=float)

    rs = np.vstack([r0 + np.array([0.0, 0.0, 0.0]), r0 + np.array([10.0, -5.0, 2.0]), r0 + np.array([20.0, 10.0, -2.0])])
    vs = np.vstack([v0, v0 + np.array([0.1, -0.2, 0.3]), v0 + np.array([-0.2, 0.1, -0.1])])

    cartesian = pd.DataFrame(
        {
            "time_s": t,
            "x_m": rs[:, 0],
            "y_m": rs[:, 1],
            "z_m": rs[:, 2],
            "vx_ms": vs[:, 0],
            "vy_ms": vs[:, 1],
            "vz_ms": vs[:, 2],
        }
    )
    cartesian.to_csv(output_dir / "orbit_cartesian.csv", index=False)

    energy = pd.DataFrame(
        {
            "time_s": t,
            "energy_Jkg": np.full_like(t, -mu / (2.0 * a), dtype=float),
            "dE_abs": np.zeros_like(t, dtype=float),
            "dE_rel": np.zeros_like(t, dtype=float),
        }
    )
    energy.to_csv(output_dir / "orbit_energy.csv", index=False)

    h_mag = np.linalg.norm(np.cross(rs, vs), axis=1)
    angmom = pd.DataFrame(
        {
            "time_s": t,
            "hx": np.cross(rs, vs)[:, 0],
            "hy": np.cross(rs, vs)[:, 1],
            "hz": np.cross(rs, vs)[:, 2],
            "h_mag": h_mag,
            "dH_abs": np.zeros_like(t, dtype=float),
            "dH_rel": np.zeros_like(t, dtype=float),
        }
    )
    angmom.to_csv(output_dir / "orbit_angular_momentum.csv", index=False)

    orbit_params = type(
        "OrbitParams",
        (),
        {
            "a": a,
            "e": e,
            "i": i,
            "omega_big": omega_big,
            "omega_small": omega_small,
            "nu": nu,
            "period": 2.0 * np.pi * np.sqrt(a**3 / mu),
            "energy": -mu / (2.0 * a),
            "h_mag": np.sqrt(mu * a),
            "epoch": None,
        },
    )()

    create_comparison_plots(output_dir, orbit_params, integrator="rk78", show=False)

    comparison_csv = output_dir / "validation" / "comparison_errors.csv"
    assert comparison_csv.exists()

    df = pd.read_csv(comparison_csv)
    assert len(df) == len(t)
    expected_cols = {
        "time_s",
        "r_error_norm_m",
        "v_error_norm_ms",
        "vx_error_ms",
        "vy_error_ms",
        "vz_error_ms",
        "h_error_m2s",
        "energy_error_Jkg",
    }
    assert expected_cols.issubset(df.columns)
