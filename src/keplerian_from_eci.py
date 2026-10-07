import numpy as np

from src.constants import constants


def keplerian_from_eci(r_vec: np.ndarray, v_vec: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Convert ECI Cartesian state vectors into Keplerian elements.

    Handles singularities:
    - Circular equatorial (e≈0, i≈0): RAAN, ArgPerigee, TrueAnomaly undefined.
      Returns true longitude λ = atan2(r_y, r_x) as nu, with Ω=0, ω=0.
    - Circular inclined (e≈0, i>0): ArgPerigee undefined.
      Returns argument of latitude u = Ω + ν as nu, with ω=0.
    - Equatorial elliptical (e>0, i≈0): RAAN undefined.
      Returns longitude of perigee ϖ = Ω + ω as omega_big, with ω=0.
    """
    const = constants()

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    xi = v**2 / 2.0 - const.mu_earth / r
    a = -const.mu_earth / (2.0 * xi)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    e_vec = ((v**2 - const.mu_earth / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / const.mu_earth
    e = np.linalg.norm(e_vec)

    i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    n_vec = np.array([-h_vec[1], h_vec[0], 0.0], dtype=float)
    n = np.linalg.norm(n_vec)

    # Tolerance for singularity detection
    SINGULAR_TOL = 1e-10

    # Case 1: Circular equatorial orbit (e≈0, i≈0)
    # All three angles (Ω, ω, ν) are undefined. Use true longitude λ.
    if e < SINGULAR_TOL and i < SINGULAR_TOL:
        omega_big = 0.0
        omega_small = 0.0
        nu = np.arctan2(r_vec[1], r_vec[0])
        if nu < 0:
            nu += const.twopi

    # Case 2: Circular inclined orbit (e≈0, i>0)
    # Arg of perigee (ω) undefined. Use argument of latitude u = Ω + ν.
    elif e < SINGULAR_TOL:
        # RAAN (Ω) is well-defined
        omega_big = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0:
            omega_big = const.twopi - omega_big

        omega_small = 0.0  # ω undefined, set to 0 by convention

        # Argument of latitude u = atan2(r·(n×h), r·n) in orbital plane
        # n×h points along the line of nodes in the orbital plane
        n_cross_h = np.cross(n_vec, h_vec)
        u = np.arctan2(np.dot(r_vec, n_cross_h), np.dot(r_vec, n_vec))
        if u < 0:
            u += const.twopi
        nu = u  # For circular orbits, u = Ω + ν, and with ω=0, ν = u

    # Case 3: Equatorial elliptical orbit (e>0, i≈0)
    # RAAN (Ω) undefined. Use longitude of perigee ϖ = Ω + ω.
    elif n < SINGULAR_TOL:
        omega_big = np.arctan2(e_vec[1], e_vec[0])  # longitude of perigee ϖ
        if omega_big < 0:
            omega_big += const.twopi
        omega_small = 0.0  # ω undefined, set to 0 by convention

        # True anomaly from eccentricity vector
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            nu = const.twopi - nu

    # Case 4: General elliptical inclined orbit
    else:
        omega_big = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
        if n_vec[1] < 0:
            omega_big = const.twopi - omega_big

        omega_small = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            omega_small = const.twopi - omega_small

        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            nu = const.twopi - nu

    return float(a), float(e), float(i), float(omega_big), float(omega_small), float(nu)
