from dataclasses import dataclass
import math

from src.constants import constants


@dataclass(frozen=True)
class OrbitParameters:
    altitude: float
    a: float
    e: float
    i: float
    omega_big: float
    omega_small: float
    nu: float
    n: float
    period: float
    period_min: float
    v_orbit: float
    energy: float
    h_mag: float
    num_orbits_24h: float


def orbital_parameters(verbose: bool = True) -> OrbitParameters:
    const = constants()

    altitude = 450e3
    a = const.r_earth + altitude
    e = 0.0
    inc = 51.6 * const.deg2rad
    omega_big = 0.0
    omega_small = 0.0
    nu = 0.0

    n = math.sqrt(const.mu_earth / a**3)
    period = const.twopi / n
    period_min = period / 60.0
    v_orbit = math.sqrt(const.mu_earth / a)
    energy = -const.mu_earth / (2.0 * a)
    h_mag = v_orbit * a
    num_orbits_24h = const.seconds_per_day / period

    orbit = OrbitParameters(
        altitude=altitude,
        a=a,
        e=e,
        i=inc,
        omega_big=omega_big,
        omega_small=omega_small,
        nu=nu,
        n=n,
        period=period,
        period_min=period_min,
        v_orbit=v_orbit,
        energy=energy,
        h_mag=h_mag,
        num_orbits_24h=num_orbits_24h,
    )

    if verbose:
        print("\n=== LEO ORBIT PARAMETERS ===")
        print(f"Altitude:                {orbit.altitude / 1e3:.1f} km")
        print(f"Semi-major axis:         {orbit.a:.6e} m")
        print(f"Eccentricity:            {orbit.e:.6f}")
        print(f"Inclination:             {orbit.i * const.rad2deg:.1f} degrees")
        print(f"Orbital period:          {orbit.period_min:.2f} min ({orbit.period:.0f} sec)")
        print(f"Orbital velocity:        {orbit.v_orbit:.2f} m/s")
        print(f"Specific energy:         {orbit.energy:.6e} J/kg")
        print(f"Angular momentum mag:    {orbit.h_mag:.6e} m^2/s")
        print(f"Orbits in 24 hours:      {orbit.num_orbits_24h:.2f}")
        print("=============================\n")

    return orbit
