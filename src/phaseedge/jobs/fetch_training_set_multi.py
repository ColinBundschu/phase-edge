from typing import Any, Mapping, Sequence, TypedDict, cast
import hashlib
import json

from jobflow.core.job import job
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms

from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import (
    make_one_snapshot,
    validate_counts_for_sublattice,
)
from phaseedge.utils.keys import rng_for_index, occ_key_for_atoms
from phaseedge.storage import store


class CETrainRef(TypedDict):
    set_id: str
    occ_key: str
    model: str
    relax_cell: bool
    dtype: str


def _lookup_energy_and_check_converged(
    *, set_id: str, occ_key: str, model: str, relax_cell: bool, dtype: str
) -> float:
    """
    Find the FF TaskDocument in `outputs` via job metadata and return total energy,
    but *only* if the relaxation is converged. Raises with a clear message otherwise.
    """
    coll = store.db_rw()["outputs"]
    doc: Mapping[str, Any] | None = coll.find_one(
        {
            "metadata.set_id": set_id,
            "metadata.occ_key": occ_key,
            "metadata.model": model,
            "metadata.relax_cell": relax_cell,
            "metadata.dtype": dtype,
        },
        projection={
            "output.is_force_converged": 1,
            "output.energy_downhill": 1,
            "output.output.energy": 1,
        },
    )

    if not doc:
        raise RuntimeError(
            f"Missing FF document for set_id={set_id} occ_key={occ_key} "
            f"(model={model}, relax_cell={relax_cell}, dtype={dtype})."
        )

    out = doc.get("output", {}) if isinstance(doc, Mapping) else {}
    is_conv = bool(out.get("is_force_converged"))
    if not is_conv:
        raise RuntimeError(
            f"Force-field relaxation not converged for set_id={set_id} occ_key={occ_key} "
            f"(model={model}, relax_cell={relax_cell}, dtype={dtype})."
        )

    # Optional extra guard: ensure optimization went downhill (can be relaxed if desired)
    downhill = out.get("energy_downhill")
    if downhill is False:
        raise RuntimeError(
            f"Relaxation ended uphill in energy for set_id={set_id} occ_key={occ_key} "
            f"(model={model}, relax_cell={relax_cell}, dtype={dtype})."
        )

    energy = out.get("output", {}).get("energy")
    if energy is None:
        raise RuntimeError(
            f"No total energy stored for set_id={set_id} occ_key={occ_key} "
            f"(model={model}, relax_cell={relax_cell}, dtype={dtype})."
        )

    return float(energy)


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
    # groups are produced by ensure_snapshots_compositions' final gather
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
    and fetch relaxed energies from the Atomate2 cache. Fails loudly if anything is missing
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

    # Make a strictly-typed supercell diag
    sx, sy, sz = map(int, supercell_diag)
    sc_diag: tuple[int, int, int] = (sx, sy, sz)

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

        if not set_id or not isinstance(set_id, str):
            raise ValueError(f"group[{gi}] missing valid 'set_id'.")
        if not isinstance(occ_keys, Sequence) or not all(isinstance(x, str) for x in occ_keys):
            raise ValueError(f"group[{gi}] 'occ_keys' must be a list[str].")
        if not counts:
            raise ValueError(f"group[{gi}] missing or empty 'counts'.")

        # Validate counts vs prototype/supercell (arity-agnostic)
        validate_counts_for_sublattice(
            conv_cell=conv,
            supercell_diag=sc_diag,
            replace_element=replace_element,
            counts=counts,
        )

        # Deterministic rebuild and energy fetch
        for i, ok in enumerate(occ_keys):
            rng = rng_for_index(set_id, int(i), 0)  # index is 0..len(occ_keys)-1
            snap = make_one_snapshot(
                conv_cell=conv,
                supercell_diag=sc_diag,
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

            # Convert to pymatgen Structure (training features are built from this)
            pmg = AseAtomsAdaptor.get_structure(snap)  # pyright: ignore[reportArgumentType]
            structures.append(pmg)

            # Energy lookup from ff_tasks (must exist)
            try:
                e = _lookup_energy_and_check_converged(
                    set_id=set_id, occ_key=ok, model=model, relax_cell=relax_cell, dtype=dtype
                )
            except Exception as exc:
                problems.append(f"group[{gi}] occ_key={ok}: {exc}")
                continue

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
