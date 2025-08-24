from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
import hashlib

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
)

# ---- helpers (counts-based set_id) -------------------------------------------------

def _hash_dict_stable(d: dict[str, Any]) -> str:
    # stable JSON and sha256
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def compute_set_id_counts(
    *,
    conv_fingerprint: str | None,
    prototype: PrototypeName | None,
    prototype_params: dict[str, Any] | None,
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    counts: dict[str, int],
    seed: int,
    algo_version: str = "randgen-2-counts-1",
) -> str:
    payload = {
        "algo": algo_version,
        "conv_fingerprint": conv_fingerprint,
        "prototype": prototype,
        "prototype_params": prototype_params,
        "supercell_diag": list(supercell_diag),
        "replace_element": replace_element,
        "counts": counts,  # exact integers only
        "seed": seed,      # set-level seed; index/attempt handled downstream
    }
    return _hash_dict_stable(payload)

# ---- spec + job --------------------------------------------------------------------

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
            conv_struct = AseAtomsAdaptor.get_structure(self.conv_cell)
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
            conv_cell=conv_cell,
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
    if (spec.conv_cell is None) == (spec.prototype is None):
        raise ValueError("Provide exactly one of conv_cell OR prototype(+params).")

    conv_cell = spec.conv_cell or make_prototype(spec.prototype, **(spec.prototype_params or {}))

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
    structure = AseAtomsAdaptor.get_structure(snapshot)  # pmg Structure

    return {"structure": structure, "set_id": set_id, "occ_key": occ_key}
