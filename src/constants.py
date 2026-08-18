import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    """Physical and numerical constants used by the simulator."""

    mu_earth: float = 3.986004415e14
    r_earth: float = 6378137.0
    r_earth_eq: float = 6378137.0
    r_earth_polar: float = 6356752.314245

    deg2rad: float = math.pi / 180.0
    rad2deg: float = 180.0 / math.pi

    pi: float = math.pi
    twopi: float = 2.0 * math.pi

    seconds_per_hour: float = 3600.0
    seconds_per_day: float = 86400.0


def constants() -> Constants:
    return Constants()
