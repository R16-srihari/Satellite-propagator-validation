import numpy as np

from src.constants import constants


def eci_from_keplerian(
    a: float,
    e: float,
    i: float,
    omega_big: float,
    omega_small: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Keplerian elements to ECI position and velocity vectors."""
    const = constants()

    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(nu))

    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0], dtype=float)
    v_pqw = np.sqrt(const.mu_earth / p) * np.array(
        [-np.sin(nu), e + np.cos(nu), 0.0], dtype=float
    )

    rz_omega_big = np.array(
        [
            [np.cos(omega_big), -np.sin(omega_big), 0.0],
            [np.sin(omega_big), np.cos(omega_big), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rx_i = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(i), -np.sin(i)],
            [0.0, np.sin(i), np.cos(i)],
        ],
        dtype=float,
    )
    rz_omega_small = np.array(
        [
            [np.cos(omega_small), -np.sin(omega_small), 0.0],
            [np.sin(omega_small), np.cos(omega_small), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    rotation = rz_omega_big @ rx_i @ rz_omega_small

    r_vec = rotation @ r_pqw
    v_vec = rotation @ v_pqw

    return r_vec, v_vec
