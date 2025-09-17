from typing import Any, Mapping, Sequence, TypedDict, cast, Optional
import hashlib
import json
import numpy as np

from jobflow.core.job import job
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms

from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.utils.keys import rng_for_index, occ_key_for_atoms
from phaseedge.storage import store
from smol.moca.ensemble import Ensemble
from phaseedge.utils.rehydrators import rehydrate_ensemble_by_ce_key


class CETrainRef(TypedDict):
    set_id: str
    occ_key: str
    model: str
    relax_cell: bool
    dtype: str
    structure: Structure


def _lookup_energy_and_check_converged(
    *, set_id: str, occ_key: str, model: str, relax_cell: bool, dtype: str
) -> float:
    coll = store.db_ro()["outputs"]
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
    if not bool(out.get("is_force_converged")):
        raise RuntimeError(
            f"Force-field relaxation not converged for set_id={set_id} occ_key={occ_key} "
            f"(model={model}, relax_cell={relax_cell}, dtype={dtype})."
        )
    if out.get("energy_downhill") is False:
        raise RuntimeError(
            f"Relaxation ended uphill for set_id={set_id} occ_key={occ_key} "
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
    payload = [
        {"set_id": sid, "occ_key": ok, "energy": float(e)}
        for (sid, ok, e) in sorted(records, key=lambda t: (t[0], t[1]))
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@job
def fetch_training_set_multi(
    *,
    # groups can come from RNG (composition) or WL (refined path)
    groups: Sequence[Mapping[str, Any]],
    # prototype identity (needed only for RNG composition)
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    # engine identity (for energy lookup)
    model: str,
    relax_cell: bool,
    dtype: str,
    # When groups carry "occs" (WL path), this is REQUIRED to decode occ -> structure.
    # For composition RNG groups (no "occs"), this can be None.
    ce_key_for_rebuild: str | None = None,
) -> dict[str, Any]:
    """
    Reconstruct EXACT snapshots and fetch relaxed energies.

    Supported input group schemas:
      RNG groups (composition):
        {
          "set_id": str,
          "occ_keys": list[str],         # ordered
          "composition_map": dict[str, dict[str, int]],
        }

      WL groups (refined):
        {
          "set_id": str,
          "occ_keys": list[str],         # ordered, structure-based keys
          "occs": list[list[int]],       # raw occupancies, length matches occ_keys
        }

    Returns:
      {
        "structures": list[Structure],
        "energies": list[float],
        "train_refs": list[CETrainRef],
        "dataset_hash": str,
      }
    """
    if not groups:
        raise ValueError("fetch_training_set_multi: 'groups' must be a non-empty sequence.")

    conv = make_prototype(prototype, **dict(prototype_params))
    sx, sy, sz = map(int, supercell_diag)
    sc_diag: tuple[int, int, int] = (sx, sy, sz)

    # Ensemble only needed when decoding WL occupancies
    ensemble: Ensemble | None = None

    structures: list[Structure] = []
    energies: list[float] = []
    train_refs: list[CETrainRef] = []
    problems: list[str] = []
    hash_records: list[tuple[str, str, float]] = []

    for gi, g in enumerate(groups):
        if not isinstance(g, Mapping):
            raise TypeError(f"group[{gi}] is not a mapping: {type(g)!r}")
        set_id = str(g["set_id"])
        occ_keys = list(g["occ_keys"])

        is_wl_group = "occs" in g
        if is_wl_group:
            if ce_key_for_rebuild is None:
                raise ValueError("fetch_training_set_multi: WL groups require ce_key_for_rebuild.")
            if ensemble is None:
                ensemble = rehydrate_ensemble_by_ce_key(ce_key_for_rebuild)

            occs = cast(Sequence[Sequence[int]], g["occs"])
            if len(occs) != len(occ_keys):
                raise ValueError(f"group[{gi}] length mismatch: len(occs) != len(occ_keys).")

            for i, (occ_raw, ok_expected) in enumerate(zip(occs, occ_keys)):
                if not isinstance(occ_raw, list) or not occ_raw:
                    raise ValueError(f"group[{gi}] occs[{i}] is empty or not a list")
                occ_arr = np.asarray([int(x) for x in occ_raw], dtype=np.int32)

                pmg_struct = ensemble.processor.structure_from_occupancy(occ_arr)
                atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
                ok2 = occ_key_for_atoms(cast(Atoms, atoms))
                if ok2 != ok_expected:
                    raise ValueError(
                        f"WL group occ_key mismatch in group[{gi}] at index={i}: "
                        f"expected {ok_expected}, rebuilt {ok2}."
                    )

                e = _lookup_energy_and_check_converged(
                    set_id=set_id, occ_key=ok_expected, model=model, relax_cell=relax_cell, dtype=dtype
                )

                structures.append(pmg_struct)
                energies.append(e)
                train_refs.append(
                    CETrainRef(
                        set_id=set_id,
                        occ_key=ok_expected,
                        model=model,
                        relax_cell=relax_cell,
                        dtype=dtype,
                        structure=pmg_struct,
                    )
                )
                hash_records.append((set_id, ok_expected, e))

        else:
            # RNG path: deterministic regeneration via seed/index (no CE needed)
            comp_map = cast(Mapping[str, Mapping[str, int]], g["composition_map"])
            for i, ok in enumerate(occ_keys):
                rng = rng_for_index(set_id, int(i))
                snap = make_one_snapshot(
                    conv_cell=conv,
                    supercell_diag=sc_diag,
                    composition_map=comp_map,
                    rng=rng,
                )
                ok2 = occ_key_for_atoms(snap)
                if ok2 != ok:
                    raise ValueError(
                        f"occ_key mismatch in group[{gi}] at index={i}: expected {ok}, rebuilt {ok2}."
                    )
                pmg = cast(Structure, AseAtomsAdaptor.get_structure(snap)) # pyright: ignore[reportArgumentType]

                e = _lookup_energy_and_check_converged(
                    set_id=set_id, occ_key=ok, model=model, relax_cell=relax_cell, dtype=dtype
                )

                structures.append(pmg)
                energies.append(e)
                train_refs.append(
                    CETrainRef(
                        set_id=set_id,
                        occ_key=ok,
                        model=model,
                        relax_cell=relax_cell,
                        dtype=dtype,
                        structure=pmg,
                    )
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
