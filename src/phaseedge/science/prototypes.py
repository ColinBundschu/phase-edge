import numpy as np
from ase.build import bulk
from ase.atoms import Atoms
from ase.spacegroup import crystal
from enum import Enum


class PrototypeName(str, Enum):
    ROCKSALT = "rocksalt"
    SPINEL = "spinel"
    DOUBLE_PEROVSKITE = "double_perovskite"


def make_prototype(
    name: PrototypeName,
    *,
    a: float,
    inactive_cation: str | None = None,
) -> Atoms:
    """
    Build a primitive prototype cell as an ASE Atoms.

    - ROCKSALT: starts from MgO rocksalt; cation (Mg) sites are replaced by Z=99 (Es).
    - SPINEL (Fd-3m, #227): primitive spinel AB2O4 with two cation sublattices:
        * A-site (8a)  -> element 99 ("Es")  -- placeholder for A sublattice
        * B-site (16d) -> element 100 ("Fm") -- placeholder for B sublattice
        * O-site (32e) -> oxygen at (u, u, u) with u=0.2625

    Parameters
    ----------
    name
        Prototype name: "rocksalt" or "spinel".
    a
        Lattice constant 'a' in Å. For spinel, typical values are ~8.08 Å for MgAl2O4,
        but this is left to the caller.
    """
    if name == PrototypeName.ROCKSALT:
        atoms = bulk("MgO", crystalstructure="rocksalt", a=a, cubic=True)

        # Identify cation sites (Mg in this prototype)
        syms = np.array(atoms.get_chemical_symbols())
        cation_mask = (syms == "Mg")

        # Replace cation sites with placeholder element Z=99 ("Es")
        nums = np.array(atoms.get_atomic_numbers(), dtype=int)
        nums[cation_mask] = 99  # Es
        atoms.set_atomic_numbers(nums.tolist())
        return atoms

    if name == PrototypeName.SPINEL:
        # Spinel: Fd-3m (227) with A on 8a, B on 16d, O on 32e(u,u,u).
        # Use a standard oxygen parameter u. Caller can vary 'a'.
        u = 0.361279

        # Build primitive cell directly.
        atoms = crystal(
            symbols=["Es", "Fm", "O"],                     # A=Es(99), B=Fm(100), O
            basis=[(1/4, 3/4, 3/4), (5/8, 3/8, 3/8), (u + 1/2, u, u)],  # 8a, 16d, 32e(u)
            spacegroup=227, # Fd-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=True,
        )
        return atoms
    
    if name == PrototypeName.DOUBLE_PEROVSKITE:
        # Use a standard oxygen parameter u. Caller can vary 'a'.
        u = 0.254328
        
        if inactive_cation is None:
            raise ValueError("inactive_cation must be specified for double_perovskite prototype.")

        # Build primitive cell directly.
        atoms = crystal(
            symbols=["Es", "Fm", inactive_cation, "O"],
            basis=[(0, 0, 0), (0, 0, 1/2), (1/4, 3/4, 3/4), (u, 0, 0)],
            spacegroup=225, # Fm-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=True,
        )
        return atoms

    raise ValueError(f"Unknown prototype: {name}")
