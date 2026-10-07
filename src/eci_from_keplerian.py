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
    """Convert Keplerian elements to ECI position and velocity vectors.

    Handles singularities:
    - Circular orbits (e ≈ 0): uses argument of latitude (u = ω + ν) instead of ω and ν separately
    - Equatorial orbits (i ≈ 0): uses longitude of perigee (ϖ = Ω + ω) instead of Ω and ω separately
    - Circular equatorial (e ≈ 0, i ≈ 0): uses true longitude (l = Ω + ω + ν)
    """
    const = constants()

    # Input validation
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive, got a={a}")
    if e < 0 or e >= 1:
        raise ValueError(f"Eccentricity must be in [0, 1) for elliptical orbits, got e={e}")
    if i < 0 or i > np.pi:
        raise ValueError(f"Inclination must be in [0, π], got i={i}")

    # Tolerances for singularity detection
    EPS_E = 1e-12  # eccentricity tolerance
    EPS_I = 1e-12  # inclination tolerance

    # Handle singularities by computing effective angles
    # Circular orbit (e ≈ 0): argument of perigee undefined, use argument of latitude u = ω + ν
    # Equatorial orbit (i ≈ 0): RAAN undefined, use longitude of perigee ϖ = Ω + ω
    # Circular equatorial (e ≈ 0, i ≈ 0): use true longitude l = Ω + ω + ν

    if e < EPS_E and i < EPS_I:
        # Circular equatorial: true longitude
        true_longitude = omega_big + omega_small + nu
        omega_big_eff = 0.0
        omega_small_eff = 0.0
        nu_eff = true_longitude
    elif e < EPS_E:
        # Circular inclined: argument of latitude
        arg_latitude = omega_small + nu
        omega_big_eff = omega_big
        omega_small_eff = 0.0
        nu_eff = arg_latitude
    elif i < EPS_I:
        # Equatorial elliptical: longitude of perigee
        long_perigee = omega_big + omega_small
        omega_big_eff = long_perigee
        omega_small_eff = 0.0
        nu_eff = nu
    else:
        # General case: no singularity
        omega_big_eff = omega_big
        omega_small_eff = omega_small
        nu_eff = nu

    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(nu_eff))

    r_pqw = np.array([r * np.cos(nu_eff), r * np.sin(nu_eff), 0.0], dtype=float)
    v_pqw = np.sqrt(const.mu_earth / p) * np.array(
        [-np.sin(nu_eff), e + np.cos(nu_eff), 0.0], dtype=float
    )

    rz_omega_big = np.array(
        [
            [np.cos(omega_big_eff), -np.sin(omega_big_eff), 0.0],
            [np.sin(omega_big_eff), np.cos(omega_big_eff), 0.0],
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
            [np.cos(omega_small_eff), -np.sin(omega_small_eff), 0.0],
            [np.sin(omega_small_eff), np.cos(omega_small_eff), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    rotation = rz_omega_big @ rx_i @ rz_omega_small

    r_vec = rotation @ r_pqw
    v_vec = rotation @ v_pqw

    return r_vec, v_vec
