import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

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
    epoch: datetime | None = None


def _parse_opm(path: Path) -> dict:
    data: dict = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("COMMENT"):
                continue
            parts = line.split("=")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip()
                data[key] = val
            else:
                # Fallback for whitespace-separated OPM style
                tokens = line.split()
                if not tokens:
                    continue
                key = tokens[0]
                val = " ".join(tokens[1:])
                data[key] = val
    return data


def _resolve_opm_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    opm_path = repo_root / "STK_input" / "Satellite1.opm"
    if not opm_path.exists():
        opm_path = repo_root / "Satellite1.opm"
    return opm_path


def _get_cartesian_state_from_opm(opm: dict) -> tuple[np.ndarray, np.ndarray] | None:
    required_keys = ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
    if not all(key in opm for key in required_keys):
        return None

    r_vec = np.array([float(opm["X"]), float(opm["Y"]), float(opm["Z"])] , dtype=float) * 1000.0
    v_vec = np.array(
        [float(opm["X_DOT"]), float(opm["Y_DOT"]), float(opm["Z_DOT"])],
        dtype=float,
    ) * 1000.0
    return r_vec, v_vec

def _get_keplerian_state_from_opm(opm: dict) -> tuple[float, float, float, float, float, float] | None:
    required_keys = ("SEMI_MAJOR_AXIS", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "TRUE_ANOMALY")
    if not all(key in opm for key in required_keys):
        return None

    a = float(opm["SEMI_MAJOR_AXIS"]) * 1000.0
    e = float(opm["ECCENTRICITY"])
    inc = float(opm["INCLINATION"]) * constants().deg2rad
    omega_big = float(opm["RA_OF_ASC_NODE"]) * constants().deg2rad
    omega_small = float(opm["ARG_OF_PERICENTER"]) * constants().deg2rad
    nu = float(opm["TRUE_ANOMALY"]) * constants().deg2rad
    return a, e, inc, omega_big, omega_small, nu


def read_opm_cartesian_state() -> tuple[np.ndarray, np.ndarray] | None:
    opm_path = _resolve_opm_path()
    if not opm_path.exists():
        return None

    try:
        opm = _parse_opm(opm_path)
        cart = _get_cartesian_state_from_opm(opm)
        if cart is None:
            raise RuntimeError(
                f"OPM file found at {opm_path} but missing required cartesian fields: X, Y, Z, X_DOT, Y_DOT, Z_DOT"
            )
        return cart
    except Exception:  # noqa: TRY203
        # Re-raise RuntimeError to halt execution when caller explicitly requested
        # cartesian state but the OPM is malformed. For other exceptions, wrap
        # to provide context.
        raise



def orbital_parameters(verbose: bool = True) -> OrbitParameters:
    const = constants()
    orbit: OrbitParameters | None = None

    # Default values (meters, radians)
    altitude = 450e3
    a = const.r_earth + altitude
    e = 0.0
    inc = 51.6 * const.deg2rad
    omega_big = 0.0
    omega_small = 0.0
    nu = 0.0
    epoch: datetime | None = None

    opm_path = _resolve_opm_path()
    if opm_path.exists():
        try:
            opm = _parse_opm(opm_path)
            keplerian_state = _get_keplerian_state_from_opm(opm)
            if keplerian_state is None:
                # Requested keplerian state is missing — stop the run.
                raise RuntimeError(
                    f"OPM file found at {opm_path} but missing required keplerian fields: "
                    "SEMI_MAJOR_AXIS, ECCENTRICITY, INCLINATION, RA_OF_ASC_NODE, ARG_OF_PERICENTER, TRUE_ANOMALY"
                )

            a, e, inc, omega_big, omega_small, nu = keplerian_state
            altitude = a - const.r_earth
            orbit = OrbitParameters(
                altitude=altitude,
                a=a,
                e=e,
                i=inc,
                omega_big=omega_big,
                omega_small=omega_small,
                nu=nu,
                n=math.sqrt(const.mu_earth / a**3),
                period=const.twopi / math.sqrt(const.mu_earth / a**3),
                period_min=(const.twopi / math.sqrt(const.mu_earth / a**3)) / 60.0,
                v_orbit=math.sqrt(const.mu_earth / a),
                energy=-const.mu_earth / (2.0 * a),
                h_mag=math.sqrt(const.mu_earth * a),
                num_orbits_24h=const.seconds_per_day / (const.twopi / math.sqrt(const.mu_earth / a**3)),
            )

            if "EPOCH" in opm:
                # OPM EPOCH expected in ISO format
                try:
                    epoch = datetime.fromisoformat(opm["EPOCH"])  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    # Try alternative parsing
                    epoch = datetime.strptime(opm["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f")

            if epoch is not None:
                orbit = OrbitParameters(
                    altitude=orbit.altitude,
                    a=orbit.a,
                    e=orbit.e,
                    i=orbit.i,
                    omega_big=orbit.omega_big,
                    omega_small=orbit.omega_small,
                    nu=orbit.nu,
                    n=orbit.n,
                    period=orbit.period,
                    period_min=orbit.period_min,
                    v_orbit=orbit.v_orbit,
                    energy=orbit.energy,
                    h_mag=orbit.h_mag,
                    num_orbits_24h=orbit.num_orbits_24h,
                    epoch=epoch,
                )
        except Exception as exc:
            # Re-raise runtime errors so the caller stops the run; otherwise
            # print a friendly message and fall back to defaults only when
            # the OPM file isn't present or another unexpected error occurs.
            if isinstance(exc, RuntimeError):
                raise
            if verbose:
                print(f"Failed to parse {opm_path}: {exc}. Falling back to defaults.")

    if orbit is None:
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
            epoch=epoch,
        )

    assert orbit is not None

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
        if orbit.epoch is not None:
            print(f"Epoch (UTC):             {orbit.epoch.isoformat(sep=' ')}")
        print("=============================\n")

    return orbit
