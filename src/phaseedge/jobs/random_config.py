from dataclasses import dataclass
from typing import Any, TypedDict

from monty.json import MSONable
from jobflow.core.job import job
from ase.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import make_one_snapshot, SublatticeSpec
from phaseedge.utils.keys import (
    rng_for_index,
    occ_key_for_atoms,
    compute_set_id_counts,
)


class RandomConfigResult(TypedDict):
    structure: Structure
    set_id: str
    occ_key: str


@dataclass
class RandomConfigSpec(MSONable):
    """
    Specification for generating ONE random configuration on possibly multiple sublattices.
    Uses exact integer counts (no fractions). Deterministic under (seed, index, attempt).
    """
    # prototype definition
    prototype: PrototypeName
    prototype_params: dict[str, Any]

    # supercell
    supercell_diag: tuple[int, int, int]

    # per-sublattice integer counts (multi-lattice)
    sublattices: list[SublatticeSpec]

    # RNG & indexing
    seed: int
    index: int           # which snapshot index in the set to generate
    attempt: int = 0     # bump only if collision forces a retry

    # ---- MSON hooks ----
    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "prototype": self.prototype,
            "prototype_params": self.prototype_params,
            "supercell_diag": list(self.supercell_diag),
            "sublattices": [sl.as_dict() if isinstance(sl, MSONable) else {
                "replace_element": sl.replace_element,
                "counts": dict(sl.counts),  # best-effort fallback
            } for sl in self.sublattices],
            "seed": self.seed,
            "index": self.index,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RandomConfigSpec":
        # Expect SublatticeSpec to be MSONable; fall back to plain dict if needed.
        subls: list[SublatticeSpec] = []
        for entry in d.get("sublattices", []):
            if isinstance(entry, dict) and "@class" in entry and entry["@class"] == "SublatticeSpec":
                subls.append(SublatticeSpec.from_dict(entry))  # type: ignore[attr-defined]
            else:
                subls.append(SublatticeSpec(  # type: ignore[call-arg]
                    replace_element=entry["replace_element"],
                    counts={k: int(v) for k, v in (entry.get("counts") or {}).items()},
                ))

        return cls(
            prototype=d["prototype"],
            prototype_params=d.get("prototype_params", {}) or {},
            supercell_diag=tuple(d["supercell_diag"]),
            sublattices=subls,
            seed=int(d["seed"]),
            index=int(d["index"]),
            attempt=int(d.get("attempt", 0)),
        )


@job
def make_random_config(spec: RandomConfigSpec) -> RandomConfigResult:
    """
    Deterministically generate ONE random configuration + metadata.
    Supports multiple sublattices via exact integer counts per sublattice.
    """
    conv_cell: Atoms = make_prototype(spec.prototype, **(spec.prototype_params or {}))

    # Deterministic "set id" that uniquely identifies the snapshot set (excluding index/attempt)
    # IMPORTANT: update compute_set_id_counts to accept 'sublattices' (list of dicts) deterministically
    set_id = compute_set_id_counts(
        prototype=spec.prototype,
        prototype_params=spec.prototype_params if spec.prototype_params else None,
        supercell_diag=spec.supercell_diag,
        sublattices=spec.sublattices,
        seed=spec.seed,
    )

    rng = rng_for_index(set_id, spec.index, spec.attempt)

    # Multi-sublattice replacement in one shot
    snapshot = make_one_snapshot(
        conv_cell=conv_cell,
        supercell_diag=spec.supercell_diag,
        sublattices=spec.sublattices,
        rng=rng,
    )

    occ_key = occ_key_for_atoms(snapshot)
    structure = AseAtomsAdaptor.get_structure(snapshot)  # pyright: ignore[reportArgumentType]

    return {"structure": structure, "set_id": set_id, "occ_key": occ_key}
