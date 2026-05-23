from __future__ import annotations

from src.orbital_parameters import orbital_parameters


def main() -> None:
    # verbose=True prints human-readable orbit summary
    orb = orbital_parameters(verbose=True)
    # Print the dataclass repr for machine-readable output
    print(orb)


if __name__ == "__main__":
    main()
