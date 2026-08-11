from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root


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



def _gauss_legendre_tableau(s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return A, b, c for the s-stage Gauss–Legendre collocation RK method.
    c are Legendre nodes shifted to [0, 1], and b are quadrature weights.
    """
    x, w = np.polynomial.legendre.leggauss(s)
    c = 0.5 * (x + 1.0)
    b = 0.5 * w

    A = np.zeros((s, s), dtype=float)
    for j in range(s):
        coeff = np.poly1d([1.0])
        denom = 1.0
        for m in range(s):
            if m != j:
                coeff *= np.poly1d([1.0, -c[m]])
                denom *= (c[j] - c[m])
        lj = coeff / denom
        int_lj = np.polyint(lj)
        for i in range(s):
            A[i, j] = float(int_lj(c[i]) - int_lj(0.0))
    return A, b, c


def symplectic_integrate(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    y0: np.ndarray,
    options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, SymplecticStats]:
    """
    Integrate to requested output times using the s=4 (order-8) Gauss–Legendre
    implicit Runge–Kutta method (symplectic RK for Kepler-like systems).

    Notes:
    - The provided `fun` parameter is used directly for the right-hand side
      evaluations. For two-body dynamics, pass `src.gravity_ode.gravity_ode`.
    - Fixed internal step size comes from options:
      SymplecticStep / InternalStep / InitialStep.
    """
    if options is None:
        options = {}

    requested_step = options.get(
        "SymplecticStep", options.get("InternalStep", options.get("InitialStep", 1.0))
    )
    step_size = abs(float(requested_step))
    if step_size <= 0.0:
        raise ValueError("Symplectic step size must be positive.")

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

    d = y_current.size // 2
    if d == 0:
        raise ValueError("State vector must contain non-empty position and velocity parts.")

    # Gauss–Legendre tableau for s=4 (order 8)
    s = 4
    A, b, c = _gauss_legendre_tableau(s)

    t_current = float(t_eval[0])
    y_out = np.empty((t_eval.size, y_current.size), dtype=float)
    y_out[0] = y_current

    accepted_steps = 0
    function_evaluations = 0
    first_accepted_step = float("nan")

    K_prev: np.ndarray | None = None  # warm start: shape (s, 2d)

    def rhs_counted(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal function_evaluations
        function_evaluations += 1
        return fun(t, y)

    output_index = 1
    while output_index < t_eval.size:
        next_output_time = float(t_eval[output_index])

        while direction * (next_output_time - t_current) > 0.0:
            remaining = next_output_time - t_current
            h = direction * min(step_size, abs(remaining))
            if h == 0.0:
                break

            y_n = y_current.copy()
            t_n = t_current

            # Initialize K for nonlinear solve
            if K_prev is None:
                f0 = rhs_counted(t_n, y_n)
                K_prev = np.tile(f0, (s, 1))

            def residual(K_flat: np.ndarray) -> np.ndarray:
                K = K_flat.reshape(s, 2 * d)
                R = np.empty_like(K)
                for i in range(s):
                    stage_state = y_n + h * np.sum(A[i, :, None] * K, axis=0)  # noqa: B023
                    t_i = t_n + c[i] * h  # noqa: B023
                    R[i] = K[i] - rhs_counted(t_i, stage_state)
                return R.ravel()

            solver_tol = float(options.get("GaussLegendreTol", 1e-10))
            maxfev = options.get("GaussLegendreMaxFEV", None)
            xtol = options.get("GaussLegendreXtol", None)

            root_kwargs: dict = {}
            root_options: dict = {}
            if maxfev is not None:
                root_options["maxfev"] = int(maxfev)
            if xtol is not None:
                # SciPy's root(hybr) supports xtol
                root_options["xtol"] = float(xtol)

            if root_options:
                root_kwargs["options"] = root_options

            sol = root(residual, K_prev.ravel(), method="hybr", tol=solver_tol, **root_kwargs)
            if not sol.success:
                # Fallback initial guess
                f_guess = rhs_counted(t_n, y_n)
                K_guess = np.tile(f_guess, (s, 1))
                sol = root(residual, K_guess.ravel(), method="hybr", tol=solver_tol, **root_kwargs)

            if not sol.success:
                raise RuntimeError(
                    f"Gauss–Legendre nonlinear solve failed at t={t_n}: {sol.message}"
                )

            K = sol.x.reshape(s, 2 * d)
            y_np1 = y_n + h * np.sum(b[:, None] * K, axis=0)

            y_current = y_np1
            t_current += h

            accepted_steps += 1
            if np.isnan(first_accepted_step):
                first_accepted_step = abs(h)

            K_prev = K

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
