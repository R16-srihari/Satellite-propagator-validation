#!/usr/bin/env python3
"""Standalone script to compare the pd853 and symplectic integrators.

This orchestrator runs the **full** validation pipeline for both integrators:

1.  Runs :func:`compare_analytical` so that an ``analytical_comparison.csv`` is
    generated in each integrator's output directory.
2.  Runs :func:`create_comparison_plots` so that the individual STK-comparison
    plots are (re)generated in each ``STKcomparison/`` subdirectory.
3.  Runs :func:`create_comparative_plots` which overlays the two integrators'
    error curves in ``output/comparison/`` and produces a summary report.

Usage (from the repository root)::

    python -m validation.compare_integrators
    python -m validation.compare_integrators --show
    python validation/compare_integrators.py --output-dir output

The script reuses the existing ``compare_analytical()`` and
``create_comparison_plots()`` functions from ``validation.compare_analytical``
and the new ``create_comparative_plots()`` from ``validation.comparative_plots``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repository root is importable when the script is executed
# directly (e.g. ``python validation/compare_integrators.py``) as well as
# when run as a module (``python -m validation.compare_integrators``).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.comparative_plots import create_comparative_plots  # noqa: E402
from validation.compare_analytical import (  # noqa: E402
    _load_orbit_from_opm,
    compare_analytical,
    create_comparison_plots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_single_integrator(
    output_dir: Path, orbit_params, integrator_name: str
) -> None:
    """Run the analytical comparison and individual plots for one integrator.

    Parameters
    ----------
    output_dir
        Integrator output directory (e.g. ``output/pd853``).
    orbit_params
        Orbital parameters loaded from the OPM file.
    integrator_name
        Display name used in plot legends.
    """
    print(f"\n=== Processing {integrator_name} ({output_dir}) ===")

    # Phase 1a - generate analytical_comparison.csv (recomputes errors vs
    # the closed-form two-body reference).
    compare_analytical(
        t_vector=None,
        y_matrix=None,
        orbit_params=orbit_params,
        output_dir=output_dir,
    )

    # Phase 1b - generate individual STK-comparison plots.
    create_comparison_plots(
        output_dir, orbit_params, integrator=integrator_name, show=False
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _resolve_integrator_dirs(
    output_dir: Path, comparison_dir: Path | None
) -> tuple[Path, Path, Path]:
    """Return (pd853_dir, symplectic_dir, final_comparison_dir)."""
    pd853_dir = output_dir / "pd853"
    symplectic_dir = output_dir / "symplectic"
    if comparison_dir is not None:
        final_comparison_dir = comparison_dir
    else:
        final_comparison_dir = output_dir / "comparison"
    return pd853_dir, symplectic_dir, final_comparison_dir


def run_comparison(
    orbit_path: Path,
    output_dir: Path,
    comparison_dir: Path | None = None,
    show: bool = False,
) -> Path:
    """Execute the full comparison pipeline for both integrators.

    Parameters
    ----------
    orbit_path
        Path to the OPM file describing the reference orbit.
    output_dir
        Base output directory containing ``pd853/`` and ``symplectic/``
        sub-directories.
    comparison_dir
        Directory for the comparative overlay plots.  Defaults to
        ``<output_dir>/comparison``.
    show
        Forwarded to the comparative plotting routine to optionally display
        figures interactively.

    Returns
    -------
    Path
        The comparative output directory.
    """
    pd853_dir, symplectic_dir, comp_dir = _resolve_integrator_dirs(
        output_dir, comparison_dir
    )

    # Load orbit parameters from the OPM file (Phase 0).
    print("Loading orbit parameters from OPM ...")
    orbit_params = _load_orbit_from_opm(orbit_path)

    # Phase 1 - individual analytical comparisons + plots.
    _run_single_integrator(pd853_dir, orbit_params, "pd853")
    _run_single_integrator(symplectic_dir, orbit_params, "symplectic")

    # Phase 2 - comparative overlay plots + summary report.
    print("\n=== Creating comparative overlay plots ===")
    create_comparative_plots(
        pd853_dir,
        symplectic_dir,
        output_dir=comp_dir,
        orbit_params=orbit_params,
        show=show,
    )

    # Phase 3 - list the generated outputs.
    print("\nComparison complete. Outputs written to:")
    for p in sorted(comp_dir.iterdir()):
        print(f"  {p}")

    return comp_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare pd853 and symplectic integrators against the analytical solution"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display generated plots interactively",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "output",
        help="Base output directory containing pd853/ and symplectic/ sub-dirs",
    )
    parser.add_argument(
        "--orbit",
        type=Path,
        default=repo_root / "STK_input" / "Satellite1.opm",
        help="Path to the OPM file",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Output directory for comparative plots (defaults to <output>/comparison)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_comparison(
        orbit_path=args.orbit,
        output_dir=args.output_dir,
        comparison_dir=args.comparison_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
