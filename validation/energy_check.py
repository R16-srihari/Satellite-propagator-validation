from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import constants


def energy_check(t_vector, y_matrix, orbit_params, output_dir):
    """Verify energy and angular momentum conservation over propagation."""
    const = constants()

    t_vector = np.asarray(t_vector, dtype=float).reshape(-1)
    y_matrix = np.asarray(y_matrix, dtype=float)
    num_points = t_vector.size
    # Prefer exported energy/angular-momentum CSVs if available
    energy_file = Path(output_dir) / "orbit_energy.csv"
    angmom_file = Path(output_dir) / "orbit_angular_momentum.csv"
    try:
        df_e = pd.read_csv(energy_file)
        print(f"  Using exported energy from {energy_file}")
        t_vector = np.asarray(df_e["time_s"], dtype=float).reshape(-1)
        energy = np.asarray(df_e["energy_Jkg"], dtype=float)
        num_points = t_vector.size
        e_init = energy[0]
        try:
            d_e_abs = np.asarray(df_e["dE_abs"], dtype=float)
        except KeyError:
            d_e_abs = energy - e_init
        try:
            d_e_rel = np.asarray(df_e["dE_rel"], dtype=float)
        except KeyError:
            d_e_rel = d_e_abs / abs(e_init)
    except (FileNotFoundError, KeyError) as e:
        print(f"  ERROR: Could not load exported energy table: {e}")
        print("  Falling back to computing energy from state vectors.")
        energy = np.zeros(num_points)
        for k in range(num_points):
            r_vec = y_matrix[k, 0:3]
            v_vec = y_matrix[k, 3:6]
            r = np.linalg.norm(r_vec)
            v = np.linalg.norm(v_vec)
            energy[k] = v**2 / 2.0 - const.mu_earth / r
        e_init = energy[0]
        d_e_abs = energy - e_init
        d_e_rel = d_e_abs / abs(e_init)

    d_e_abs_max = np.max(np.abs(d_e_abs))
    d_e_rel_max = np.max(np.abs(d_e_rel))
    d_e_rel_mean = np.mean(np.abs(d_e_rel))

    print(f"  Initial energy:                 {e_init:.15e} J/kg")
    print(f"  Expected energy (analytical):   {orbit_params.energy:.15e} J/kg")
    print(f"  Max absolute energy error:      {d_e_abs_max:.6e} J/kg")
    print(f"  Max relative energy error:      {d_e_rel_max:.6e} ({d_e_rel_max * 100:.2e} %)")
    print(f"  Mean relative energy error:     {d_e_rel_mean:.6e} ({d_e_rel_mean * 100:.2e} %)")

    if d_e_rel_max < 1e-9:
        print("  Status: EXCELLENT - Energy conserved to < 1e-9 relative error")
    elif d_e_rel_max < 1e-8:
        print("  Status: GOOD - Energy conserved to < 1e-8 relative error")
    elif d_e_rel_max < 1e-6:
        print("  Status: ACCEPTABLE - Energy conserved to < 1e-6 relative error")
    else:
        print("  Status: WARNING - Energy error exceeds acceptable tolerance")

    output_path = Path(output_dir)
    energy_check_file = output_path / "energy_conservation.csv"
    pd.DataFrame(
        {
            "time_s": t_vector,
            "energy_Jkg": energy,
            "dE_rel": d_e_rel,
        }
    ).to_csv(energy_check_file, index=False)
    print(f"  Data saved to: {energy_check_file}")
    # Use exported angular momentum if present, otherwise compute from states
    if angmom_file.exists():
        df_h = pd.read_csv(angmom_file)
        print(f"  Using exported angular momentum from {angmom_file}")
        h_vec = np.column_stack((df_h["hx"].to_numpy(), df_h["hy"].to_numpy(), df_h["hz"].to_numpy()))
        h_mag = np.asarray(df_h["h_mag"], dtype=float)
        h_init = h_mag[0]
        d_h_abs = h_mag - h_init
        d_h_rel = d_h_abs / abs(h_init)
    else:
        h_vec = np.cross(y_matrix[:, 0:3], y_matrix[:, 3:6])
        h_mag = np.linalg.norm(h_vec, axis=1)
        h_init = h_mag[0]
        d_h_abs = h_mag - h_init
        d_h_rel = d_h_abs / abs(h_init)

    d_h_abs_max = np.max(np.abs(d_h_abs))
    d_h_rel_max = np.max(np.abs(d_h_rel))
    d_h_rel_mean = np.mean(np.abs(d_h_rel))

    print("\n  Angular momentum (specific) conservation:")
    print(f"  Initial |h|:                   {h_init:.15e} m^2/s")
    print(f"  Max absolute |h| error:        {d_h_abs_max:.6e} m^2/s")
    print(f"  Max relative |h| error:        {d_h_rel_max:.6e} ({d_h_rel_max * 100:.2e} %)")
    print(f"  Mean relative |h| error:       {d_h_rel_mean:.6e} ({d_h_rel_mean * 100:.2e} %)")

    if d_h_rel_max < 1e-9:
        print("  Status: EXCELLENT - Angular momentum conserved to < 1e-9 relative error")
    elif d_h_rel_max < 1e-8:
        print("  Status: GOOD - Angular momentum conserved to < 1e-8 relative error")
    elif d_h_rel_max < 1e-6:
        print("  Status: ACCEPTABLE - Angular momentum conserved to < 1e-6 relative error")
    else:
        print("  Status: WARNING - Angular momentum error exceeds acceptable tolerance")

    angmom_file = output_path / "angular_momentum_conservation.csv"
    pd.DataFrame(
        {
            "time_s": t_vector,
            "hx": h_vec[:, 0],
            "hy": h_vec[:, 1],
            "hz": h_vec[:, 2],
            "h_mag": h_mag,
            "dH_abs": d_h_abs,
            "dH_rel": d_h_rel,
        }
    ).to_csv(angmom_file, index=False)
    print(f"  Data saved to: {angmom_file}")
