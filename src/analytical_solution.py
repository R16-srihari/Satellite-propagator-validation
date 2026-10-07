import math

from src.eci_from_keplerian import eci_from_keplerian


def analytical_solution(
    t_eval: float,
    a: float,
    e: float,
    i: float,
    omega_big: float,
    omega_small: float,
    nu0: float,
    mu: float,
):
    """Compute analytical Keplerian state at a given time.

    Parameters
    ----------
    t_eval : float
        Time at which to evaluate the solution [seconds].
    a : float
        Semi-major axis [meters]. Must be positive.
    e : float
        Eccentricity. Must satisfy 0 <= e < 1 (elliptical orbits only).
    i : float
        Inclination [radians]. Must satisfy 0 <= i <= pi.
    omega_big : float
        Right ascension of ascending node (RAAN) [radians].
    omega_small : float
        Argument of perigee [radians].
    nu0 : float
        Initial true anomaly at epoch [radians].
    mu : float
        Gravitational parameter [m^3/s^2].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Position and velocity vectors in ECI frame [m, m/s].

    Raises
    ------
    ValueError
        If a <= 0, e < 0, e >= 1, or i outside [0, pi].
    """
    # Input validation
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive, got a={a}")
    if e < 0:
        raise ValueError(f"Eccentricity must be non-negative, got e={e}")
    if e >= 1:
        raise ValueError(f"Eccentricity must be < 1 for elliptical orbits, got e={e}")
    if not (0 <= i <= math.pi):
        raise ValueError(f"Inclination must be in [0, pi], got i={i}")

    n = math.sqrt(mu / a**3)

    if e < 1e-10:
        nu = nu0 + n * t_eval
    else:
        e0 = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(nu0 / 2.0))
        m0 = e0 - e * math.sin(e0)
        m = m0 + n * t_eval

        ecc_anomaly = m
        tol = 1e-12
        max_iter = 50
        for _ in range(max_iter):
            f = ecc_anomaly - e * math.sin(ecc_anomaly) - m
            df = 1.0 - e * math.cos(ecc_anomaly)
            d_e = -f / df
            ecc_anomaly += d_e
            if abs(d_e) < tol:
                break

        nu = 2.0 * math.atan2(
            math.sqrt(1.0 + e) * math.sin(ecc_anomaly / 2.0),
            math.sqrt(1.0 - e) * math.cos(ecc_anomaly / 2.0),
        )

    nu = nu % (2.0 * math.pi)
    return eci_from_keplerian(a, e, i, omega_big, omega_small, nu)
