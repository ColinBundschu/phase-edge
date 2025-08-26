from typing import Any, Mapping, Sequence, TypedDict, cast
import hashlib
import json

from jobflow.core.job import job
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms

from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import make_one_snapshot, validate_counts_for_sublattice
from phaseedge.orchestration.jobs.store_mace_result import lookup_mace_result
from phaseedge.utils.keys import rng_for_index, occ_key_for_atoms


class CETrainRef(TypedDict):
    set_id: str
    occ_key: str
    model: str
    relax_cell: bool
    dtype: str


def _dataset_hash(records: Sequence[tuple[str, str, float]]) -> str:
    """
    Canonical hash over sorted (set_id, occ_key, energy) triplets.
    Guards against occ_key collisions across different set_ids.
    """
    payload = [
        {"set_id": sid, "occ_key": ok, "energy": float(e)}
        for (sid, ok, e) in sorted(records, key=lambda t: (t[0], t[1]))
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@job
def fetch_training_set_multi(
    *,
    # groups are produced by ensure_snapshots_multi's final gather
    groups: Sequence[Mapping[str, Any]],
    # prototype-only system identity (needed to rebuild snapshots)
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    # engine identity (for energy lookup)
    model: str,
    relax_cell: bool,
    dtype: str,
) -> dict[str, Any]:
    """
    Reconstruct EXACT snapshots for each group (composition), verify occ_keys match,
    and fetch relaxed energies from the cache. Fails loudly if anything is missing
    or inconsistent.

    Input group schema (validated here):
      {
        "set_id": str,
        "occ_keys": list[str],         # ordered; length defines K for this group
        "counts": dict[str, int],      # exact integers for this composition
        "seed": int,                   # effective seed used for this group's generation
      }

    Returns:
      {
        "structures": list[Structure],
        "energies": list[float],
        "train_refs": list[CETrainRef],
        "dataset_hash": str,  # hash over (set_id, occ_key, energy)
      }
    """
    if not groups:
        raise ValueError("fetch_training_set_multi: 'groups' must be a non-empty sequence.")

    # Build prototype conventional cell once; validate each group's counts against it.
    conv: Atoms = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))

    structures: list[Structure] = []
    energies: list[float] = []
    train_refs: list[CETrainRef] = []
    problems: list[str] = []
    hash_records: list[tuple[str, str, float]] = []

    for gi, g in enumerate(groups):
        if not isinstance(g, Mapping):
            raise TypeError(f"group[{gi}] is not a mapping: {type(g)!r}")
        set_id = cast(str, g.get("set_id"))
        occ_keys = cast(Sequence[str], g.get("occ_keys"))
        counts = {str(k): int(v) for k, v in dict(g.get("counts", {})).items()}
        seed = int(g.get("seed", 0))

        if not set_id or not isinstance(set_id, str):
            raise ValueError(f"group[{gi}] missing valid 'set_id'.")
        if not isinstance(occ_keys, Sequence) or not all(isinstance(x, str) for x in occ_keys):
            raise ValueError(f"group[{gi}] 'occ_keys' must be a list[str].")
        if not counts:
            raise ValueError(f"group[{gi}] missing or empty 'counts'.")
        # Validate counts vs prototype/supercell (arity-agnostic)
        validate_counts_for_sublattice(
            conv_cell=conv,
            supercell_diag=tuple(map(int, supercell_diag)),
            replace_element=replace_element,
            counts=counts,
        )

        # Deterministic rebuild and energy fetch
        for i, ok in enumerate(occ_keys):
            rng = rng_for_index(set_id, int(i), 0)  # index is 0..len(occ_keys)-1
            snap = make_one_snapshot(
                conv_cell=conv,
                supercell_diag=tuple(map(int, supercell_diag)),
                replace_element=replace_element,
                counts=counts,
                rng=rng,
            )
            ok2 = occ_key_for_atoms(snap)
            if ok2 != ok:
                raise ValueError(
                    f"occ_key mismatch in group[{gi}] at index={i}: expected {ok}, rebuilt {ok2}. "
                    "This indicates a change in snapshot generation or inputs."
                )

            # Convert to pymatgen Structure
            pmg = AseAtomsAdaptor.get_structure(snap)  # pyright: ignore[reportArgumentType]
            structures.append(pmg)

            # Energy lookup (must exist and be successful)
            doc = lookup_mace_result(set_id, ok, model=model, relax_cell=relax_cell, dtype=dtype)
            if not doc:
                problems.append(f"group[{gi}] occ_key={ok}: not_found")
                continue
            success = bool(doc.get("success", False))  # type: ignore[arg-type]
            e_val = cast(float | None, doc.get("energy"))
            if not success:
                problems.append(f"group[{gi}] occ_key={ok}: found_but_failed")
                continue
            if e_val is None:
                problems.append(f"group[{gi}] occ_key={ok}: found_but_no_energy")
                continue

            e = float(e_val)
            energies.append(e)
            train_refs.append(
                {
                    "set_id": set_id,
                    "occ_key": ok,
                    "model": model,
                    "relax_cell": relax_cell,
                    "dtype": dtype,
                }
            )
            hash_records.append((set_id, ok, e))

    if problems:
        raise RuntimeError(
            "Training set incomplete or invalid. Missing/failed items:\n  - "
            + "\n  - ".join(problems)
        )

    ds_hash = _dataset_hash(hash_records)

    return {
        "structures": structures,
        "energies": energies,
        "train_refs": train_refs,
        "dataset_hash": ds_hash,
    }
