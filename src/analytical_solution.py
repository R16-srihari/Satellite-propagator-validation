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
    """Compute analytical Keplerian state at a given time."""
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
