from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate._ivp import dop853_coefficients as coeffs

SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 2.0


@dataclass
class RK78Stats:
    accepted_steps: int
    rejected_steps: int
    function_evaluations: int
    final_time: float
    final_state: np.ndarray
    first_attempt_error_norm: float
    first_accepted_step: float


def _as_1d_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _build_stage(
    y_base: np.ndarray,
    h: float,
    stage_weights: np.ndarray,
    stage_values: np.ndarray,
    work: np.ndarray,
) -> np.ndarray:
    work[:] = y_base
    if stage_weights.size:
        work += h * np.dot(stage_weights, stage_values)
    return work


def _compute_error_norm(stage_values: np.ndarray, h: float, scale: np.ndarray) -> float:
    err5 = np.dot(coeffs.E5, stage_values) / scale
    err3 = np.dot(coeffs.E3, stage_values) / scale

    err5_norm_sq = float(np.dot(err5, err5))
    err3_norm_sq = float(np.dot(err3, err3))
    if err5_norm_sq == 0.0 and err3_norm_sq == 0.0:
        return 0.0

    return abs(h) * err5_norm_sq / np.sqrt((err5_norm_sq + 0.01 * err3_norm_sq) * scale.size)


def _initial_step_size(
    y0: np.ndarray,
    f0: np.ndarray,
    rel_tol: float,
    abs_tol: float,
    max_step: float,
    remaining: float,
    requested_step: float | None,
) -> float:
    if requested_step is not None:
        step = abs(float(requested_step))
        if step <= 0.0:
            raise ValueError("InternalStep/InitialStep must be positive when provided.")
        return min(step, max_step, remaining)

    scale = np.full_like(y0, abs_tol, dtype=float) + rel_tol * np.abs(y0)
    d0 = float(np.linalg.norm(y0 / scale) / np.sqrt(y0.size))
    d1 = float(np.linalg.norm(f0 / scale) / np.sqrt(y0.size))

    step = float(np.where((d0 < 1e-8) or (d1 < 1e-8), 1e-6, 0.01 * d0 / d1))

    return min(step, max_step, remaining)


def rk78_integrate(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    y0: np.ndarray,
    options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, RK78Stats]:
    """Integrate to requested output times using a standalone adaptive DOP853 stepper."""
    if options is None:
        options = {}

    rel_tol = float(options.get("RelTol", 1e-12))
    abs_tol = float(options.get("AbsTol", 1e-14))
    max_step = float(options.get("MaxStep", 60.0))
    requested_step = options.get("InternalStep", options.get("InitialStep", options.get("FirstStep")))

    if max_step <= 0.0:
        raise ValueError("MaxStep must be positive.")

    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    if t_eval.size < 2:
        raise ValueError("t_eval must contain at least two time points.")

    direction = np.sign(t_eval[-1] - t_eval[0])
    if direction == 0:
        raise ValueError("t_eval must span a nonzero time interval.")

    if np.any(np.diff(t_eval) * direction <= 0):
        raise ValueError("t_eval must be strictly monotonic.")

    y_current = _as_1d_float_array(y0)
    if y_current.size == 0:
        raise ValueError("y0 must contain at least one state component.")

    t_current = float(t_eval[0])
    t_final = float(t_eval[-1])

    y_out = np.empty((t_eval.size, y_current.size), dtype=float)
    y_out[0] = y_current

    f_current = _as_1d_float_array(fun(t_current, y_current))
    if f_current.size != y_current.size:
        raise ValueError("fun must return a derivative vector with the same size as y0.")

    function_evaluations = 1
    accepted_steps = 0
    rejected_steps = 0
    first_attempt_error_norm = float("nan")
    first_accepted_step = float("nan")

    stage_values = np.empty((coeffs.N_STAGES + 1, y_current.size), dtype=float)
    y_work = np.empty_like(y_current)

    h_abs = _initial_step_size(
        y_current,
        f_current,
        rel_tol,
        abs_tol,
        max_step,
        abs(t_final - t_current),
        None if requested_step is None else float(requested_step),
    )

    output_index = 1
    while output_index < t_eval.size:
        remaining = t_final - t_current
        if direction * remaining <= 0.0:
            break

        next_output_time = float(t_eval[output_index])
        h_abs = min(h_abs, max_step, abs(remaining), abs(next_output_time - t_current))
        if h_abs == 0.0:
            y_out[output_index] = y_current
            output_index += 1
            continue

        h = direction * h_abs
        t_new = t_current + h
        if direction * (t_new - t_final) > 0.0:
            t_new = t_final
            h = t_new - t_current
            h_abs = abs(h)

        stage_values[0] = f_current

        for stage_index in range(1, coeffs.N_STAGES):
            y_stage = _build_stage(
                y_current,
                h,
                coeffs.A[stage_index, :stage_index],
                stage_values[:stage_index],
                y_work,
            )
            stage_values[stage_index] = _as_1d_float_array(fun(t_current + coeffs.C[stage_index] * h, y_stage))
        function_evaluations += coeffs.N_STAGES - 1

        y_new = y_current + h * np.dot(coeffs.B, stage_values[: coeffs.N_STAGES])
        f_new = _as_1d_float_array(fun(t_new, y_new))
        stage_values[coeffs.N_STAGES] = f_new
        function_evaluations += 1

        scale = np.full_like(y_current, abs_tol, dtype=float) + rel_tol * np.maximum(np.abs(y_current), np.abs(y_new))
        error_norm = _compute_error_norm(stage_values, h, scale)
        if np.isnan(first_attempt_error_norm):
            first_attempt_error_norm = error_norm

        if error_norm <= 1.0:
            accepted_steps += 1
            if np.isnan(first_accepted_step):
                first_accepted_step = abs(h)

            t_current = t_new
            y_current = y_new
            f_current = f_new

            while output_index < t_eval.size:
                target_time = float(t_eval[output_index])
                output_ready = (direction > 0.0 and target_time <= t_current) or (
                    direction < 0.0 and target_time >= t_current
                )
                if output_ready:
                    y_out[output_index] = y_current
                    output_index += 1
                    continue
                break

            if error_norm == 0.0:
                factor = MAX_FACTOR
            else:
                factor = SAFETY * error_norm ** (-1.0 / 8.0)
                factor = min(MAX_FACTOR, max(MIN_FACTOR, factor))
                if rejected_steps > 0:
                    factor = min(1.0, factor)

            h_abs = min(max_step, abs(h) * factor)
        else:
            rejected_steps += 1
            factor = SAFETY * error_norm ** (-1.0 / 8.0)
            h_abs = min(max_step, abs(h) * max(MIN_FACTOR, factor))

    if output_index < t_eval.size:
        raise RuntimeError("RK78 integration failed to reach the final output time.")

    stats = RK78Stats(
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        function_evaluations=function_evaluations,
        final_time=float(t_current),
        final_state=y_current.copy(),
        first_attempt_error_norm=first_attempt_error_norm,
        first_accepted_step=first_accepted_step,
    )

    return t_eval, y_out, stats
