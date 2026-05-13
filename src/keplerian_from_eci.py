import numpy as np

from src.constants import constants


def keplerian_from_eci(r_vec: np.ndarray, v_vec: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Convert ECI Cartesian state vectors into Keplerian elements."""
    const = constants()

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    xi = v**2 / 2.0 - const.mu_earth / r
    a = -const.mu_earth / (2.0 * xi)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    e_vec = ((v**2 - const.mu_earth / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / const.mu_earth
    e = np.linalg.norm(e_vec)

    i = np.arccos(h_vec[2] / h)

    n_vec = np.array([-h_vec[1], h_vec[0], 0.0], dtype=float)
    n = np.linalg.norm(n_vec)

    if n > 1e-10:
        omega_big = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0:
            omega_big = const.twopi - omega_big
    else:
        omega_big = 0.0

    if e > 1e-10 and n > 1e-10:
        omega_small = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            omega_small = const.twopi - omega_small
    else:
        omega_small = 0.0

    if e > 1e-10:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            nu = const.twopi - nu
    elif n > 1e-10:
        nu = np.arccos(np.clip(np.dot(r_vec, n_vec) / (r * n), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            nu = const.twopi - nu
    else:
        nu = 0.0

    if np.isnan(nu):
        nu = 0.0
    if np.isnan(omega_small):
        omega_small = 0.0
    if np.isnan(omega_big):
        omega_big = 0.0

    return a, e, i, omega_big, omega_small, nu
