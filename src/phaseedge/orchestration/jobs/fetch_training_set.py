from typing import Any, Mapping, Sequence, TypedDict, cast
import hashlib
import json

from jobflow.core.job import job
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.orchestration.jobs.store_mace_result import lookup_mace_result
from phaseedge.utils.keys import rng_for_index, occ_key_for_atoms


class CETrainRef(TypedDict):
    set_id: str
    occ_key: str
    model: str
    relax_cell: bool
    dtype: str


def _dataset_hash(pairs: list[tuple[str, float]]) -> str:
    """
    Hash over sorted (occ_key, energy) pairs for immutability checks.
    """
    payload = [{"occ_key": k, "energy": float(e)} for k, e in sorted(pairs, key=lambda x: x[0])]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@job
def fetch_training_set(
    *,
    # identity of the snapshot set (prototype-only design)
    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    replace_element: str,
    counts: Mapping[str, int],
    seed: int,
    # exact membership and ordering
    set_id: str,
    indices: Sequence[int],
    occ_keys: Sequence[str],
    # engine identity (how to look up energies)
    model: str,
    relax_cell: bool,
    dtype: str,
) -> dict[str, Any]:
    """
    Reconstruct the EXACT snapshots (by index), verify occ_keys match,
    and fetch their relaxed energies from the cache. Fail loudly if anything
    is missing or inconsistent.
    """
    if len(indices) != len(occ_keys):
        raise ValueError(f"Length mismatch: indices={len(indices)} vs occ_keys={len(occ_keys)}")

    # 1) Deterministically rebuild snapshots + verify occ_keys
    conv = make_prototype(cast(PrototypeName, prototype), **dict(prototype_params))  # <- cast fixes Pylance
    structures: list[Structure] = []
    occ_keys_rebuilt: list[str] = []
    for i, ok in zip(indices, occ_keys):
        rng = rng_for_index(set_id, int(i), 0)
        snapshot = make_one_snapshot(
            conv_cell=conv,  # ASE Atoms
            supercell_diag=supercell_diag,
            replace_element=replace_element,
            counts={k: int(v) for k, v in counts.items()},
            rng=rng,
        )
        # compute occ_key and convert structure
        ok2 = occ_key_for_atoms(snapshot)
        if ok2 != ok:
            raise ValueError(
                f"occ_key mismatch at index={i}: expected {ok}, rebuilt {ok2}. "
                "This indicates a change in snapshot generation or inputs."
            )
        pmg = AseAtomsAdaptor.get_structure(snapshot) # pyright: ignore[reportArgumentType]
        structures.append(pmg)
        occ_keys_rebuilt.append(ok2)

    # 2) Fetch relaxed energies for each occ_key; fail with actionable report
    energies: list[float] = []
    train_refs: list[CETrainRef] = []
    problems: list[dict[str, Any]] = []

    for ok in occ_keys_rebuilt:
        doc = lookup_mace_result(set_id, ok, model=model, relax_cell=relax_cell, dtype=dtype)
        status = "ok"
        energy_val: float | None = None

        if doc is None:
            status = "not_found"
        else:
            # enforce success True and presence of energy
            success = bool(doc.get("success", False))  # type: ignore[arg-type]
            energy_val = cast(float | None, doc.get("energy"))
            if not success:
                status = "found_but_failed"
            elif energy_val is None:
                status = "found_but_no_energy"

        if status != "ok":
            problems.append({"occ_key": ok, "status": status})
        else:
            energies.append(float(energy_val))  # type: ignore[arg-type]
            train_refs.append(
                {
                    "set_id": set_id,
                    "occ_key": ok,
                    "model": model,
                    "relax_cell": relax_cell,
                    "dtype": dtype,
                }
            )

    if problems:
        details = "; ".join(f"{p['occ_key']}={p['status']}" for p in problems)
        raise RuntimeError("Training set incomplete or invalid. Missing/failed items: " + details)

    # 3) Canonical dataset hash (sorted by occ_key)
    ds_hash = _dataset_hash(list(zip(occ_keys_rebuilt, energies)))

    return {
        "structures": structures,
        "energies": energies,
        "train_refs": train_refs,
        "dataset_hash": ds_hash,
    }
