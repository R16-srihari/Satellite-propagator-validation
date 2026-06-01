import numpy as np

from src.analytical_solution import analytical_solution
from src.constants import constants
from src.eci_from_keplerian import eci_from_keplerian
from src.orbital_parameters import orbital_parameters, read_opm_cartesian_state


def test_analytical_solution_returns_to_initial_state_each_orbit() -> None:
    orbit = orbital_parameters(verbose=False)
    const = constants()

    cartesian_state = read_opm_cartesian_state()
    if cartesian_state is not None:
        r0, v0 = cartesian_state
    else:
        r0, v0 = eci_from_keplerian(
            orbit.a,
            orbit.e,
            orbit.i,
            orbit.omega_big,
            orbit.omega_small,
            orbit.nu,
        )

    y0 = np.concatenate((r0, v0))

    for multiplier in (1, 2, 3):
        r, v = analytical_solution(
            multiplier * orbit.period,
            orbit.a,
            orbit.e,
            orbit.i,
            orbit.omega_big,
            orbit.omega_small,
            orbit.nu,
            const.mu_earth,
        )
        y = np.concatenate((r, v))
        error = y - y0
        pos_err_norm = float(np.linalg.norm(error[:3]))
        vel_err_norm = float(np.linalg.norm(error[3:]))
        print(f"multiplier={multiplier} position_error_norm_m={pos_err_norm:.12e} velocity_error_norm_ms={vel_err_norm:.12e}")
        print(f"  error_vector_pos_m= {error[:3]}")
        print(f"  error_vector_vel_ms= {error[3:]}")

        assert pos_err_norm < 1e-6
        assert vel_err_norm < 1e-5