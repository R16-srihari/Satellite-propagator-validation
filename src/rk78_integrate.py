from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class RK78Stats:
    accepted_steps: int
    rejected_steps: int
    function_evaluations: int
    final_time: float
    final_state: np.ndarray
    first_attempt_error_norm: float
    first_accepted_step: float


def rk78_integrate(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    y0: np.ndarray,
    options: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, RK78Stats]:
    """Integrate to requested output times using adaptive DOP853 (RK8) stepping."""
    if options is None:
        options = {}

    rel_tol = float(options.get("RelTol", 1e-12))
    abs_tol = float(options.get("AbsTol", 1e-13))
    max_step = float(options.get("MaxStep", 60.0))

    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    if t_eval.size < 2:
        raise ValueError("t_eval must contain at least two time points.")

    direction = np.sign(t_eval[-1] - t_eval[0])
    if direction == 0:
        raise ValueError("t_eval must span a nonzero time interval.")

    if np.any(np.diff(t_eval) * direction <= 0):
        raise ValueError("t_eval must be strictly monotonic.")

    y_current = np.asarray(y0, dtype=float).reshape(-1)

    solution = solve_ivp(
        fun,
        (float(t_eval[0]), float(t_eval[-1])),
        y_current,
        method="DOP853",
        t_eval=t_eval,
        rtol=rel_tol,
        atol=abs_tol,
        max_step=max_step,
    )

    if not solution.success:
        raise RuntimeError(f"RK78 integration failed: {solution.message}")

    y_out = solution.y.T

    stats = RK78Stats(
        accepted_steps=max(solution.t.size - 1, 0),
        rejected_steps=0,
        function_evaluations=int(solution.nfev),
        final_time=float(solution.t[-1]),
        final_state=solution.y[:, -1].copy(),
        first_attempt_error_norm=float("nan"),
        first_accepted_step=float("nan"),
    )

    return t_eval, y_out, stats
