from typing import Literal
import numpy as np
from ase.atoms import Atoms
from ase.build import bulk
from ase.spacegroup import crystal

PrototypeName = Literal["rocksalt", "spinel"]

def _tag_sublattices_spinel(at: Atoms, A_placeholder: str, B_placeholder: str) -> None:
    """
    Add an integer 'sublattice' array:
      0 = Td (A-site), 1 = Oh (B-site), 2 = O (anions)
    """
    syms = np.array(at.get_chemical_symbols())
    subl = np.full(len(at), 2, dtype=np.int8)     # default O
    subl[syms == A_placeholder] = 0               # Td
    subl[syms == B_placeholder] = 1               # Oh
    # Persist on Atoms; will be replicated by .repeat(...)
    at.new_array("sublattice", subl)

def _tag_sublattices_rocksalt(at: Atoms, cation: str, anion: str) -> None:
    """
    Add an integer 'sublattice' array:
      0 = cation fcc sublattice, 2 = anion fcc sublattice
    (keeps '2' for anions to be consistent with spinel's O tag)
    """
    syms = np.array(at.get_chemical_symbols())
    subl = np.full(len(at), 2, dtype=np.int8)
    subl[syms == cation] = 0
    at.new_array("sublattice", subl)

def make_prototype(
    name: PrototypeName,
    *,
    a: float | None = None,
    cubic: bool = True,
    # spinel specifics
    u: float = 0.262,                 # oxygen parameter (typical for many spinels)
    A_placeholder: str = "Mg",        # Td cation placeholder (to be replaced later)
    B_placeholder: str = "Al",        # Oh cation placeholder (to be replaced later)
    primitive_cell: bool = True,      # True -> 14-atom primitive (2 f.u.); False -> 56-atom conventional (8 f.u.)
) -> Atoms:
    """
    Build a prototype Atoms object.

    - 'rocksalt': uses ase.build.bulk
    - 'spinel'  : uses ase.spacegroup.crystal with Fd-3m (227),
                  Wyckoff A:8a (1/8,1/8,1/8), B:16d (1/2,1/2,1/2), O:32e (u,u,u)

    Returns an Atoms with a 'sublattice' array:
      spinel: 0=Td(A), 1=Oh(B), 2=O
      rocksalt: 0=cation, 2=anion
    """
    if name == "rocksalt":
        # default a for MgO-like, if not supplied
        a = 4.3 if a is None else a
        at = bulk(f"{A_placeholder}{B_placeholder}",  # e.g., "MgO"
                  crystalstructure="rocksalt", a=a, cubic=cubic)
        _tag_sublattices_rocksalt(at, cation=A_placeholder, anion=B_placeholder)
        return at

    if name == "spinel":
        # default a for MgAl2O4-like, if not supplied
        a = 8.08 if a is None else a
        at = crystal(
            symbols=[A_placeholder, B_placeholder, "O"],
            basis=[(1/8, 1/8, 1/8), (1/2, 1/2, 1/2), (u, u, u)],
            spacegroup=227,  # Fd-3m
            cellpar=[a, a, a, 90, 90, 90],
            primitive_cell=primitive_cell,
        )
        _tag_sublattices_spinel(at, A_placeholder=A_placeholder, B_placeholder=B_placeholder)
        return at

    raise ValueError(f"Unknown prototype: {name}")
