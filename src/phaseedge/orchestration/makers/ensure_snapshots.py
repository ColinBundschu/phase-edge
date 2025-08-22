from typing import TypedDict, Any
import datetime
from pathlib import Path

from jobflow import job  # type: ignore[reportPrivateImportUsage]
from ase.atoms import Atoms
from ase.io import write as ase_write

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


class EnsureSnapshotsResult(TypedDict):
    set_id: str
    count: int
    added: int


@job
def ensure_snapshots_job(
    *,
    # Option A: supply an explicit conv_cell (e.g., from POSCAR)
    conv_cell: Atoms | None = None,

    # Option B: build prototype on the fly (MVP: rocksalt MgO)
    prototype: PrototypeName | None = None,
    prototype_params: dict[str, Any] | None = None,  # e.g., {"a": 4.3, "cubic": True}

    supercell_diag: tuple[int, int, int],
    replace_element: str,
    composition: dict[str, float],  # one composition per set (simplest)
    seed: int,
    target_count: int,
    outdir: str | None = None,
    algo_version: str = "randgen-2",
    max_attempts_per_index: int = 10,
) -> EnsureSnapshotsResult:
    """
    Ensure the snapshot set has at least `target_count` items.

    Identity includes either an explicit input cell fingerprint OR the
    (prototype, prototype_params). Deterministic per-index RNG makes the set
    expandable and reproducible.
    """
    # Build or validate conv_cell
    proto_used: dict[str, Any] | None = None
    if conv_cell is not None and prototype is not None:
        raise ValueError("Provide either conv_cell OR prototype(+params), not both.")

    if conv_cell is None:
        if prototype is None:
            raise ValueError("You must provide conv_cell or prototype.")
        params = prototype_params or {}
        conv_cell = make_prototype(prototype, **params)
        proto_used = {"name": prototype, "params": params}

    conv_fp = fingerprint_conv_cell(conv_cell)

    # Compute identity (set_id)
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

    # Header upsert (idempotent)
    header = {
        "set_id": set_id,
        "seed": seed,
        "compositions": [composition],
        "replace_element": replace_element,
        "supercell_diag": supercell_diag,
        "algo_version": algo_version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
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
                path = str((outp / f"{ok}.poscar").resolve())
                ase_write(path, snapshot, format="vasp", direct=True, vasp5=True, sort=True)

            doc = {
                "set_id": set_id,
                "index": idx,
                "occ_key": ok,
                "composition": composition,
                "path": path,
                # helpful provenance:
                "prototype": header["prototype"],
                "prototype_params": header["prototype_params"],
                # denormalized convenience fields for ad-hoc queries:
                "supercell_diag": supercell_diag,
                "replace_element": replace_element,
            }
            if insert_snapshot_unique(doc):
                added += 1
                break
        else:
            raise RuntimeError(
                f"Could not create unique snapshot for index {idx} after {max_attempts_per_index} attempts"
            )

    new_total = count_by_set(set_id)
    return {"set_id": set_id, "count": new_total, "added": added}
