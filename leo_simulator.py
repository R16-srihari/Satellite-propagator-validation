from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from src.constants import constants
from src.gravity_ode import gravity_ode
from src.eci_from_keplerian import eci_from_keplerian
from src.orbital_parameters import orbital_parameters, read_opm_cartesian_state
from src.rk78_integrate import rk78_integrate
from src.symplectic_integrate import symplectic_integrate
from src.export_results import export_results
from validation.compare_analytical import compare_analytical
from validation.energy_check import energy_check
from validation.plot_conservation import plot_conservation
import pandas as pd


class Tee:
    """Write stdout to terminal and log file simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        """Write text to all streams and return number of characters written.

        The return type matches the text IO `write` signature expected by
        `contextlib.redirect_stdout` (which requires a `write(str) -> int`).
        """
        for stream in self.streams:
            stream.write(data)
        try:
            return len(data)
        except Exception:
            return 0

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


INTEGRATOR_DISPLAY_NAMES = {
    "rk78": "Custom RK7(8)",
    "symplectic": "Velocity Verlet (Symplectic)",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEO satellite orbit simulation and validation")
    parser.add_argument(
        "--integrator",
        choices=sorted(INTEGRATOR_DISPLAY_NAMES.keys()),
        default="rk78",
        help="Integrator scheme used for propagation",
    )
    return parser.parse_args(argv)


def run_simulation(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    root = Path(__file__).resolve().parent
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_output_dir = output_dir / args.integrator
    run_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_output_dir / "terminal_log.txt"
    root_log_file = output_dir / "terminal_log.txt"

    root_log_file.write_text(
        (
            "This run writes logs under integrator-specific folders.\n"
            f"Latest run integrator: {args.integrator}\n"
            f"Latest run log: {log_file}\n"
        ),
        encoding="utf-8",
    )

    with log_file.open("w", encoding="utf-8") as handle:
        tee = Tee(sys.stdout, handle)
        with redirect_stdout(tee):
            _run_simulation_core(run_output_dir, log_file, args.integrator)


def _run_selected_integrator(
    integrator: str,
    t_output: np.ndarray,
    y0: np.ndarray,
    options: dict,
):
    if integrator == "rk78":
        return rk78_integrate(gravity_ode, t_output, y0, options)

    return symplectic_integrate(gravity_ode, t_output, y0, options)


def _run_simulation_core(output_dir: Path, log_file: Path, integrator: str) -> None:
    integrator_name = INTEGRATOR_DISPLAY_NAMES[integrator]

    print("\n========== LEO SATELLITE ORBIT SIMULATOR ==========")
    print(f"Integrator: {integrator_name}")
    print("=====================================================\n")

    const = constants()
    orbit = orbital_parameters(verbose=True)

    print("=== INITIAL CONDITIONS ===")

    cartesian_state = read_opm_cartesian_state()
    if cartesian_state is not None:
        r_init, v_init = cartesian_state
        print("Initial state source: OPM Cartesian fields")
    else:
        r_init, v_init = eci_from_keplerian(
            orbit.a,
            orbit.e,
            orbit.i,
            orbit.omega_big,
            orbit.omega_small,
            orbit.nu,
        )
        print("Initial state source: OrbitParameters fallback")

    y0 = np.concatenate((r_init, v_init))

    print(f"Position [m]:       [{r_init[0]:.6e}, {r_init[1]:.6e}, {r_init[2]:.6e}]")
    print(f"Velocity [m/s]:     [{v_init[0]:.6e}, {v_init[1]:.6e}, {v_init[2]:.6e}]")
    print(f"Initial radius:     {np.linalg.norm(r_init):.6e} m")
    print(f"Initial velocity:   {np.linalg.norm(v_init):.6e} m/s")
    epoch = getattr(orbit, "epoch", None)
    if epoch is not None:
        print(f"Epoch (UTC):         {epoch.isoformat(sep=' ')}")
    print("==========================\n")

    print("=== INTEGRATION SETUP ===")

    t_final = const.seconds_per_day
    output_interval = 10.0
    t_output = np.arange(0.0, t_final + output_interval, output_interval)

    print(f"Simulation duration:    24 hours ({t_final:.0f} seconds)")
    print(f"Output interval:        {int(output_interval)} seconds")
    print(f"Expected orbits:        {t_final / orbit.period:.2f}")
    print(f"Output points:          {t_output.size}")

    options = {
        "RelTol": 1e-10,
        "AbsTol": 1e-12,
        "MaxStep": 120.0,
        "InternalStep": 1e-3,
    }

    print("RelTol:"                 f"                {options['RelTol']:.0e}")
    print("AbsTol:"                 f"                {options['AbsTol']:.0e}")
    print("MaxStep:"                f"                {options['MaxStep']:.0e} s")
    print("InternalStep:"           f"                {options['InternalStep']:.1e} s")
    print(f"Output directory:       {output_dir}")
    print("=========================\n")

    print(f"=== RUNNING {integrator_name.upper()} ===")
    print("Integration in progress...\n")

    t_adapt, y_adapt, stats = _run_selected_integrator(integrator, t_output, y0, options)

    print("\n=== INTEGRATION COMPLETE ===")
    print(f"Solver steps:           {stats.accepted_steps}")
    print(f"Rejected steps:         {stats.rejected_steps}")
    print(f"Function evaluations:   {stats.function_evaluations}")
    print(f"Output points:          {t_adapt.size}")
    print("===========================\n")

    print("=== SAVING RESULTS ===")
    export_results(t_adapt, y_adapt, orbit, output_dir)

    # Load exported fixed-grid cartesian states and use them for validation
    cartesian_file = output_dir / "orbit_cartesian.csv"
    df_cart = pd.read_csv(cartesian_file)
    t_export = df_cart["time_s"].to_numpy()
    y_export = np.column_stack(
        (
            df_cart["x_m"].to_numpy(),
            df_cart["y_m"].to_numpy(),
            df_cart["z_m"].to_numpy(),
            df_cart["vx_ms"].to_numpy(),
            df_cart["vy_ms"].to_numpy(),
            df_cart["vz_ms"].to_numpy(),
        )
    )

    print("=== VALIDATION ===")
    print("\nEnergy Conservation Test:")
    energy_check(t_export, y_export, orbit, output_dir)

    print("\nAnalytical Comparison Test:")
    compare_analytical(t_export, y_export, orbit, output_dir)

    print("\nConservation Plots:")
    plot_conservation(t_export, y_export, orbit, output_dir)

    print("\n===== SIMULATION COMPLETED SUCCESSFULLY =====\n")
    print(f"Terminal log saved to: {log_file}")


if __name__ == "__main__":
    run_simulation()
