from src.orbital_parameters import orbital_parameters
import json
import dataclasses

if __name__ == '__main__':
    o = orbital_parameters(verbose=False)
    print(json.dumps(dataclasses.asdict(o), default=str, indent=2))
