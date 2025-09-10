from dataclasses import dataclass
from typing import Any, TypedDict

from monty.json import MSONable
from jobflow.core.job import job
from ase.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.utils.keys import (
    rng_for_index,
    occ_key_for_atoms,
    compute_set_id,
)

class RandomConfigResult(TypedDict):
    structure: Structure
    set_id: str
    occ_key: str

@dataclass
class RandomConfigSpec(MSONable):
    # inputs that define the snapshot set + which index to generate
    prototype: PrototypeName
    prototype_params: dict[str, Any]
    supercell_diag: tuple[int, int, int]
    replace_element: str
    counts: dict[str, int]          # integers, not fractions
    seed: int
    index: int                      # which snapshot index in the set to generate
    attempt: int = 0                # bump only if collision forces a retry

    # monty JSON hooks
    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "prototype": self.prototype,
            "prototype_params": self.prototype_params,
            "supercell_diag": list(self.supercell_diag),
            "replace_element": self.replace_element,
            "counts": self.counts,
            "seed": self.seed,
            "index": self.index,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RandomConfigSpec":
        return cls(
            prototype=d["prototype"],
            prototype_params=d["prototype_params"],
            supercell_diag=tuple(d["supercell_diag"]),
            replace_element=d["replace_element"],
            counts={k: int(v) for k, v in (d.get("counts") or {}).items()},
            seed=int(d["seed"]),
            index=int(d["index"]),
            attempt=int(d.get("attempt", 0)),
        )

@job
def make_random_config(spec: RandomConfigSpec) -> RandomConfigResult:
    """
    Deterministically generate ONE random configuration + metadata.
    Uses exact integer counts (no fractions).
    """
    conv_cell: Atoms = make_prototype(spec.prototype, **(spec.prototype_params or {}))
    set_id = compute_set_id(
        prototype=spec.prototype,
        prototype_params=spec.prototype_params if spec.prototype_params else None,
        supercell_diag=spec.supercell_diag,
        replace_element=spec.replace_element,
        counts=spec.counts,
        seed=spec.seed,
    )

    rng = rng_for_index(set_id, spec.index, spec.attempt)

    snapshot = make_one_snapshot(
        conv_cell=conv_cell,
        supercell_diag=spec.supercell_diag,
        replace_element=spec.replace_element,
        counts=spec.counts,
        rng=rng,
    )

    occ_key = occ_key_for_atoms(snapshot)
    structure = AseAtomsAdaptor.get_structure(snapshot) # pyright: ignore[reportArgumentType]

    return {"structure": structure, "set_id": set_id, "occ_key": occ_key}
