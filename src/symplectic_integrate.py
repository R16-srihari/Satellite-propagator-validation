from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.constants import constants


@dataclass
class SymplecticStats:
    accepted_steps: int
    rejected_steps: int
    function_evaluations: int
    final_time: float
    final_state: np.ndarray
    first_attempt_error_norm: float
    first_accepted_step: float


def _as_1d_float_array(values: np.ndarray) -> np.ndarray:
    
    return np.asarray(values, dtype=float).reshape(-1)


def _acceleration(r_vec: np.ndarray, mu: float) -> np.ndarray:
    r_mag = float(np.linalg.norm(r_vec))
    if r_mag == 0.0:
        raise ValueError("Position magnitude must be nonzero for gravitational acceleration.")

    return -mu * r_vec / (r_mag**3)


def symplectic_integrate(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    y0: np.ndarray,
    options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, SymplecticStats]:
    """Integrate to requested output times using fixed-step Velocity Verlet."""
    if options is None:
        options = {}

    requested_step = options.get("SymplecticStep", options.get("InternalStep", options.get("InitialStep", 1.0)))
    step_size = abs(float(requested_step))
    if step_size <= 0.0:
        raise ValueError("Symplectic step size must be positive.")

    mu = float(options.get("Mu", constants().mu_earth))

    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    if t_eval.size < 2:
        raise ValueError("t_eval must contain at least two time points.")

    direction = float(np.sign(t_eval[-1] - t_eval[0]))
    if direction == 0.0:
        raise ValueError("t_eval must span a nonzero time interval.")

    if np.any(np.diff(t_eval) * direction <= 0):
        raise ValueError("t_eval must be strictly monotonic.")

    y_current = _as_1d_float_array(y0)
    if y_current.size == 0:
        raise ValueError("y0 must contain at least one state component.")

    if y_current.size % 2 != 0:
        raise ValueError("Symplectic integrator expects even-sized [r, v] state vectors.")

    t_current = float(t_eval[0])

    y_out = np.empty((t_eval.size, y_current.size), dtype=float)
    y_out[0] = y_current

    half = y_current.size // 2

    accepted_steps = 0
    function_evaluations = 0
    first_accepted_step = float("nan")

    output_index = 1
    while output_index < t_eval.size:
        next_output_time = float(t_eval[output_index])

        while direction * (next_output_time - t_current) > 0.0:
            remaining = next_output_time - t_current
            h = direction * min(step_size, abs(remaining))
            if h == 0.0:
                break

            r_current = y_current[:half]
            v_current = y_current[half:]

            a_current = _acceleration(r_current, mu)
            function_evaluations += 1

            v_half = v_current + 0.5 * h * a_current
            r_new = r_current + h * v_half

            a_new = _acceleration(r_new, mu)
            function_evaluations += 1

            v_new = v_half + 0.5 * h * a_new
            y_current = np.concatenate((r_new, v_new))
            t_current += h

            accepted_steps += 1
            if np.isnan(first_accepted_step):
                first_accepted_step = abs(h)

        y_out[output_index] = y_current
        output_index += 1

    stats = SymplecticStats(
        accepted_steps=accepted_steps,
        rejected_steps=0,
        function_evaluations=function_evaluations,
        final_time=t_current,
        final_state=y_current.copy(),
        first_attempt_error_norm=float("nan"),
        first_accepted_step=first_accepted_step,
    )

    return t_eval, y_out, stats
