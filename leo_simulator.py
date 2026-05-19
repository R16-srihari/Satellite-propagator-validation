from __future__ import annotations

import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from src.eci_from_keplerian import eci_from_keplerian
from src.constants import constants
from src.gravity_ode import gravity_ode
from src.orbital_parameters import orbital_parameters
from src.rk78_integrate import rk78_integrate
from src.export_results import export_results
from validation.compare_analytical import compare_analytical
from validation.energy_check import energy_check
from validation.plot_conservation import plot_conservation


class Tee:
    """Write stdout to terminal and log file simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_simulation() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "terminal_log.txt"

    with log_file.open("w", encoding="utf-8") as handle:
        tee = Tee(sys.stdout, handle)
        with redirect_stdout(tee):
            _run_simulation_core(output_dir, log_file)


def _run_simulation_core(output_dir: Path, log_file: Path) -> None:
    print("\n========== LEO SATELLITE ORBIT SIMULATOR ==========")
    print("RK7(8) Integrator (custom implementation)")
    print("=====================================================\n")

    const = constants()
    orbit = orbital_parameters(verbose=True)

    print("=== INITIAL CONDITIONS ===")

    r_init, v_init = eci_from_keplerian(
        orbit.a,
        orbit.e,
        orbit.i,
        orbit.omega_big,
        orbit.omega_small,
        orbit.nu,
    )
    y0 = np.concatenate((r_init, v_init))

    print(f"Position [m]:       [{r_init[0]:.6e}, {r_init[1]:.6e}, {r_init[2]:.6e}]")
    print(f"Velocity [m/s]:     [{v_init[0]:.6e}, {v_init[1]:.6e}, {v_init[2]:.6e}]")
    print(f"Initial radius:     {np.linalg.norm(r_init):.6e} m")
    print(f"Initial velocity:   {np.linalg.norm(v_init):.6e} m/s")
    if getattr(orbit, "epoch", None) is not None:
        print(f"Epoch (UTC):         {orbit.epoch.isoformat(sep=' ')}")
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
        "RelTol": 1e-12,
        "AbsTol": 1e-14,
        "MaxStep": 60.0,
        "InternalStep": 0.2,
    }

    print("RelTol:                 1e-12")
    print("AbsTol:                 1e-14")
    print("MaxStep:                60 s")
    print("InternalStep:           0.2 s")
    print("=========================\n")

    print("=== RUNNING CUSTOM RK7(8) INTEGRATOR ===")
    print("Integration in progress...\n")

    t, y_output, stats = rk78_integrate(gravity_ode, t_output, y0, options)

    print("\n=== INTEGRATION COMPLETE ===")
    print(f"Solver steps:           {stats.accepted_steps}")
    print(f"Rejected steps:         {stats.rejected_steps}")
    print(f"Function evaluations:   {stats.function_evaluations}")
    print(f"Output points:          {y_output.shape[0]}")
    print("===========================\n")

    print("=== SAVING RESULTS ===")
    export_results(t, y_output, orbit, output_dir)

    print("=== VALIDATION ===")
    print("\nEnergy Conservation Test:")
    energy_check(t, y_output, orbit, output_dir)

    print("\nAnalytical Comparison Test:")
    compare_analytical(t, y_output, orbit, output_dir)

    print("\nConservation Plots:")
    plot_conservation(t, y_output, orbit, output_dir, num_samples=20)

    print("\n===== SIMULATION COMPLETED SUCCESSFULLY =====\n")
    print(f"Terminal log saved to: {log_file}")


if __name__ == "__main__":
    run_simulation()
