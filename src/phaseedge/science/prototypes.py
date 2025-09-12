from typing import Literal
import numpy as np
from ase.build import bulk
from ase.atoms import Atoms

PrototypeName = Literal["rocksalt"]  # extend later

def make_prototype(
    name: PrototypeName,
    *,
    a: float = 4.3,
    cubic: bool = True,
) -> Atoms:
    """
    Build a primitive prototype cell as an ASE Atoms.
    """
    if name != "rocksalt":
        raise ValueError(f"Unknown prototype: {name}")

    atoms = bulk("MgO", crystalstructure="rocksalt", a=a, cubic=cubic)

    # Identify cation sites (Mg in this prototype)
    syms = np.array(atoms.get_chemical_symbols())
    cation_mask = (syms == "Mg")

    # Set cation sites to placeholder X (atomic number 0)
    nums = np.array(atoms.get_atomic_numbers(), dtype=int)
    nums[cation_mask] = 99 # Es
    atoms.set_atomic_numbers(nums.tolist())
    return atoms
