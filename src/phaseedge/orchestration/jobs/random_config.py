from dataclasses import dataclass
from typing import Any

from monty.json import MSONable
from jobflow.core.job import job
from ase.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.utils.keys import (
    fingerprint_conv_cell,
    rng_for_index,
    occ_key_for_atoms,
    compute_set_id_counts
)


@dataclass
class RandomConfigSpec(MSONable):
    # inputs that define the snapshot set + which index to generate
    conv_cell: Atoms | None
    prototype: PrototypeName | None
    prototype_params: dict[str, Any] | None
    supercell_diag: tuple[int, int, int]
    replace_element: str
    counts: dict[str, int]          # <-- integers, not fractions
    seed: int
    index: int                      # which snapshot index in the set to generate
    attempt: int = 0                # bump only if collision forces a retry

    # monty JSON hooks
    def as_dict(self) -> dict[str, Any]:
        conv_struct_dict = None
        if self.conv_cell is not None:
            conv_struct = AseAtomsAdaptor.get_structure(self.conv_cell) # pyright: ignore[reportArgumentType]
            conv_struct_dict = conv_struct.as_dict()
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "conv_cell_structure": conv_struct_dict,
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
        conv_cell = None
        if d.get("conv_cell_structure") is not None:
            struct = Structure.from_dict(d["conv_cell_structure"])
            conv_cell = AseAtomsAdaptor.get_atoms(struct)
        return cls(
            conv_cell=conv_cell, # pyright: ignore[reportArgumentType]
            prototype=d.get("prototype"),
            prototype_params=d.get("prototype_params"),
            supercell_diag=tuple(d["supercell_diag"]),
            replace_element=d["replace_element"],
            counts={k: int(v) for k, v in (d.get("counts") or {}).items()},
            seed=int(d["seed"]),
            index=int(d["index"]),
            attempt=int(d.get("attempt", 0)),
        )

@job
def make_random_config(spec: RandomConfigSpec) -> dict[str, Any]:
    """
    Deterministically generate ONE random configuration + metadata.
    Uses exact integer counts (no fractions).
    """

    if spec.prototype is not None:
        if spec.conv_cell is not None:
            raise ValueError("Provide exactly one of conv_cell OR prototype(+params).")
        conv_cell = make_prototype(spec.prototype, **(spec.prototype_params or {}))
    elif spec.conv_cell is not None:
        conv_cell = spec.conv_cell
    else:
        raise ValueError("Provide exactly one of conv_cell OR prototype(+params).")

    set_id = compute_set_id_counts(
        conv_fingerprint=None if spec.prototype else fingerprint_conv_cell(conv_cell),
        prototype=spec.prototype if spec.prototype else None,
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
