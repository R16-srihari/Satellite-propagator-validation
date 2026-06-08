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
from validation.plot_conservation import plot_conservation
import pandas as pd

from typing import TypedDict, Literal, Union


IntegratorType = Literal["rk78", "symplectic"]


class RK78Options(TypedDict, total=False):
    RelTol: float
    AbsTol: float
    MaxStep: float
    InternalStep: float


class SymplecticOptions(TypedDict, total=False):
    SymplecticStep: float
    GaussLegendreTol: float
    GaussLegendreMaxFEV: int
    GaussLegendreXtol: float


OptionsType = Union[RK78Options, SymplecticOptions]


def _build_rk78_options() -> RK78Options:
    return {
        "RelTol": 1e-12,
        "AbsTol": 1e-14,
        "MaxStep": 60.0,
        "InternalStep": 1e-3,
    }


def _build_symplectic_options() -> SymplecticOptions:
    # symplectic_integrate uses SymplecticStep (with fallback to InternalStep/InitialStep)
    return {"SymplecticStep": 1,
            "GaussLegendreTol": 1e-14,
            "GaussLegendreMaxFEV": 200,
            "GaussLegendreXtol": 1e-14}


def _build_integrator_options(integrator: str) -> OptionsType:
    if integrator == "rk78":
        return _build_rk78_options()
    if integrator == "symplectic":
        return _build_symplectic_options()
    raise ValueError(f"Unsupported integrator: {integrator!r}")


def _print_integrator_options(integrator: str, options: OptionsType) -> None:
    if integrator == "rk78":
        print("RelTol:" f"                {options.get('RelTol', 1e-10):.0e}")
        print("AbsTol:" f"                {options.get('AbsTol', 1e-12):.0e}")
        print("MaxStep:" f"                {options.get('MaxStep', 120.0):.0e} s")
        print(
            "InternalStep:" f"                {options.get('InternalStep', 1e-3):.1e} s"
        )
        return

    # symplectic
    symp_step = float(options.get("SymplecticStep", 1e-3))
    print("SymplecticStep:" f"        {symp_step:.1e} s")
    print("GaussLegendreTol:" f"     {options.get('GaussLegendreTol', 1e-14):.0e}")
    print("GaussLegendreMaxFEV:" f"  {options.get('GaussLegendreMaxFEV', 200)}")
    print("GaussLegendreXtol:" f"    {options.get('GaussLegendreXtol', 1e-14):.0e}")


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
    "symplectic": "Gauss-Legendre (Symplectic)",
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
    options: OptionsType,
):
    # Integrator implementations expect `options: dict | None`.
    # Cast here to keep type-checkers happy without changing runtime behavior.
    options_dict: dict = dict(options)

    if integrator == "rk78":
        return rk78_integrate(gravity_ode, t_output, y0, options_dict)
    elif integrator == "symplectic":
        return symplectic_integrate(gravity_ode, t_output, y0, options_dict)
    else:
        raise ValueError(f"Unsupported integrator: {integrator!r}")


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
    output_interval = input("Enter output grid step in seconds [default value is 10]: ").strip()
    if output_interval:
        time_step_s = float(output_interval)
        if time_step_s <= 0:
            raise ValueError("Output grid step must be positive")
    else:
        time_step_s = 10.0

    t_output = np.arange(0.0, t_final + time_step_s * 0.5, time_step_s)

    print(f"Simulation duration:    24 hours ({t_final:.0f} seconds)")
    print(f"Output interval:        {int(time_step_s)} seconds")
    print(f"Expected orbits:        {t_final / orbit.period:.2f}")
    print(f"Output points:          {t_output.size}")

    options = _build_integrator_options(integrator)

    _print_integrator_options(integrator, options)

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
    export_results(t_adapt, y_adapt, orbit, output_dir,time_step_s)
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

    print("\n=== VALIDATION ===\n")
    print("\nAnalytical Comparison Test:")
    compare_analytical(t_export, y_export, orbit, output_dir)

    print("\nConservation Plots:")
    plot_conservation(t_export, y_export, orbit, output_dir)
    
    print("\n===== SIMULATION COMPLETED SUCCESSFULLY =====\n")
    print(f"Terminal log saved to: {log_file}")


if __name__ == "__main__":
    run_simulation()
