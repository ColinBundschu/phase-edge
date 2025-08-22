from typing import Dict, Tuple, Any
from datetime import datetime
from pathlib import Path
from jobflow import job  # type: ignore[reportPrivateImportUsage]
from ase.atoms import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.utils.keys import (
    compute_set_id,
    fingerprint_conv_cell,
    rng_for_index,
    occ_key_for_atoms,
)
from phaseedge.storage.store import (
    upsert_snapshot_set,
    insert_snapshot_unique,
    count_by_set,
)


@job
def ensure_snapshots_job(
    *,
    # Option A: supply an explicit conv_cell (e.g., from POSCAR)
    conv_cell: Atoms | None = None,

    # Option B: build prototype on the fly (MVP: rocksalt MgO)
    prototype: PrototypeName | None = None,
    prototype_params: Dict[str, Any] | None = None,   # e.g., {"a": 4.3, "cubic": True}

    supercell_diag: Tuple[int, int, int],
    replace_element: str,
    composition: Dict[str, float],   # one composition per set (simplest)
    seed: int,
    target_count: int,
    outdir: str | None = None,
    algo_version: str = "randgen-2",
    max_attempts_per_index: int = 10,
) -> Dict:
    """
    Ensure the snapshot set has at least `target_count` items. The set identity
    includes either an explicit input cell fingerprint OR the prototype+params.
    """
    # build or validate conv_cell
    proto_used: Dict[str, Any] | None = None
    if conv_cell is not None and prototype is not None:
        raise ValueError("Provide either conv_cell OR prototype(+params), not both.")

    if conv_cell is None:
        if prototype is None:
            raise ValueError("You must provide conv_cell or prototype.")
        params = prototype_params or {}
        conv_cell = make_prototype(prototype, **params)
        proto_used = {"name": prototype, "params": params}

    conv_fp = fingerprint_conv_cell(conv_cell)

    # identity
    set_id = compute_set_id(
        conv_fingerprint=None if proto_used else conv_fp,
        prototype=(proto_used["name"] if proto_used else None),
        prototype_params=(proto_used["params"] if proto_used else None),
        supercell_diag=supercell_diag,
        replace_element=replace_element,
        compositions=[composition],
        seed=seed,
        algo_version=algo_version,
    )

    # header upsert
    header = {
        "set_id": set_id,
        "seed": seed,
        "compositions": [composition],
        "replace_element": replace_element,
        "supercell_diag": supercell_diag,
        "algo_version": algo_version,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "conv_fingerprint": conv_fp if proto_used is None else None,
        "prototype": (proto_used["name"] if proto_used else None),
        "prototype_params": (proto_used["params"] if proto_used else None),
    }
    upsert_snapshot_set(header)

    have = count_by_set(set_id)
    if have >= target_count:
        return {"set_id": set_id, "count": have, "added": 0}

    outp = Path(outdir) if outdir else None
    if outp:
        outp.mkdir(parents=True, exist_ok=True)

    added = 0
    for idx in range(have, target_count):
        for attempt in range(max_attempts_per_index):
            rng = rng_for_index(set_id, idx, attempt)
            snapshot = make_one_snapshot(
                conv_cell=conv_cell,
                supercell_diag=supercell_diag,
                replace_element=replace_element,
                composition=composition,
                rng=rng,
            )
            ok = occ_key_for_atoms(snapshot)

            path = None
            if outp:
                pmg = AseAtomsAdaptor.get_structure(snapshot)  # type: ignore[arg-type]
                path = str((outp / f"{ok}.poscar").resolve())
                pmg.to(fmt="poscar", filename=path)

            doc = {
                "set_id": set_id,
                "index": idx,
                "occ_key": ok,
                "composition": composition,
                "path": path,
                # helpful provenance:
                "prototype": header["prototype"],
                "prototype_params": header["prototype_params"],
            }
            if insert_snapshot_unique(doc):
                added += 1
                break
        else:
            raise RuntimeError(f"Could not create unique snapshot for index {idx} after {max_attempts_per_index} attempts")

    new_total = count_by_set(set_id)
    return {"set_id": set_id, "count": new_total, "added": added}
