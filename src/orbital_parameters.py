from dataclasses import dataclass
import math
from pathlib import Path
from datetime import datetime
from typing import Optional

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
    epoch: Optional[datetime] = None


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


def orbital_parameters(verbose: bool = True) -> OrbitParameters:
    const = constants()

    # Default values (meters, radians)
    altitude = 450e3
    a = const.r_earth + altitude
    e = 0.0
    inc = 51.6 * const.deg2rad
    omega_big = 0.0
    omega_small = 0.0
    nu = 0.0
    epoch: Optional[datetime] = None

    # Attempt to read Satellite1.opm from STK_input (release asset) then root
    repo_root = Path(__file__).resolve().parents[1]
    opm_path = repo_root / "STK_input" / "Satellite1.opm"
    if not opm_path.exists():
        # fallback to repository root for backward compatibility
        opm_path = repo_root / "Satellite1.opm"
    if opm_path.exists():
        try:
            opm = _parse_opm(opm_path)
            # OPM convention: SEMI_MAJOR_AXIS in km
            if "SEMI_MAJOR_AXIS" in opm:
                a_km = float(opm["SEMI_MAJOR_AXIS"])
                a = a_km * 1000.0
                altitude = a - const.r_earth
            if "ECCENTRICITY" in opm:
                e = float(opm["ECCENTRICITY"])
            if "INCLINATION" in opm:
                inc = float(opm["INCLINATION"]) * const.deg2rad
            if "RA_OF_ASC_NODE" in opm:
                omega_big = float(opm["RA_OF_ASC_NODE"]) * const.deg2rad
            if "ARG_OF_PERICENTER" in opm:
                omega_small = float(opm["ARG_OF_PERICENTER"]) * const.deg2rad
            if "TRUE_ANOMALY" in opm:
                nu = float(opm["TRUE_ANOMALY"]) * const.deg2rad
            if "EPOCH" in opm:
                # OPM EPOCH expected in ISO format
                try:
                    epoch = datetime.fromisoformat(opm["EPOCH"])  # type: ignore[arg-type]
                except Exception:
                    # Try alternative parsing
                    epoch = datetime.strptime(opm["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f")
            if verbose:
                print(f"Using OPM file for initial orbit: {opm_path}")
        except Exception as exc:
            if verbose:
                print(f"Failed to parse {opm_path}: {exc}. Falling back to defaults.")

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
