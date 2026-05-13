import numpy as np

from src.constants import constants


def gravity_ode(t: float, y: np.ndarray) -> np.ndarray:
    """Two-body orbital dynamics: d/dt[r,v] = [v, -mu r / |r|^3]."""
    del t
    const = constants()

    r_vec = y[0:3]
    v_vec = y[3:6]

    r_mag = np.linalg.norm(r_vec)
    a_vec = -(const.mu_earth / r_mag**3) * r_vec

    return np.concatenate((v_vec, a_vec))
