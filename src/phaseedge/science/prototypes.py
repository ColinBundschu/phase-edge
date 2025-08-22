from typing import Literal
from ase.build import bulk
from ase.atoms import Atoms


PrototypeName = Literal["rocksalt"]  # extend later: "spinel", etc.


def make_prototype(
    name: PrototypeName,
    *,
    a: float = 4.3,
    cubic: bool = True,
) -> Atoms:
    """
    Build a primitive prototype cell as an ASE Atoms.
    MVP: rocksalt MgO. Extend with spinel etc. later.
    """
    if name == "rocksalt":
        # Mg is the cation we'll replace on the cation sublattice
        return bulk("MgO", crystalstructure="rocksalt", a=a, cubic=cubic)
    raise ValueError(f"Unknown prototype: {name}")
