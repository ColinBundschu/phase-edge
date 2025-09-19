from typing import Any, Mapping, Sequence, cast

import numpy as np
from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms

from phaseedge.jobs.decide_relax import relax_structure
from phaseedge.jobs.train_ce import CETrainRef
from phaseedge.schemas.mixture import counts_sig
from phaseedge.storage.store import lookup_total_energy_eV
from phaseedge.utils.keys import compute_dataset_key, occ_key_for_structure
from phaseedge.jobs.store_ce_model import rehydrate_ensemble_by_ce_key


@job
def relax_selected_from_wl(
    *,
    ce_key: str,
    # From select_d_optimal_basis: list of {"source","wl_key","checkpoint_hash","bin","occ","occ_hash", ["counts" for endpoints]}
    selected: Sequence[Mapping[str, Any]],
    # Deterministic composition for each WL chain
    wl_counts_map: Mapping[str, Mapping[str, int]],
    model: str,
    relax_cell: bool,
    dtype: str,
    category: str = "gpu",
    set_id_prefix: str = "wlref",
) -> Mapping[str, Any] | Response:
    """
    Group selected occupancies by composition (WL & endpoints), map to structures, schedule
    relaxes in parallel, and return groups compatible with fetch_training_set_multi:
        groups = [{"set_id": str, "counts": dict[str,int], "occ_keys": [str, ...], "occs": [[int,...], ...]}, ...]
    """

    ensemble = rehydrate_ensemble_by_ce_key(ce_key)

    sub_jobs: list[Job | Flow] = []
    groups_map: dict[str, dict[str, Any]] = {} # counts_sig -> {"set_id","counts","occ_keys","occs"}
    train_refs: list[CETrainRef] = []
    for rec in selected:
        src = rec["source"]
        if src == "wl":
            wl_key = str(rec["wl_key"])
            counts = wl_counts_map[wl_key]
        elif src == "endpoint":
            counts = rec["counts"]
        else:
            raise RuntimeError(f"relax_selected_from_wl: unrecognized source='{src}' in record.")

        sig = counts_sig(counts)
        set_id = f"{set_id_prefix}::{ce_key}::{sig}"
        if sig not in groups_map:
            groups_map[sig] = {"set_id": set_id, "occ_keys": [], "occs": []}

        # Build structure from occupancy via CE processor
        occ_seq = cast(Sequence[int], rec["occ"])
        occ_arr = np.asarray([int(x) for x in occ_seq], dtype=np.int32)
        pmg_struct = ensemble.processor.structure_from_occupancy(occ_arr)
        occ_key = occ_key_for_structure(pmg_struct)

        # Compute a structure-based key (matches fetch_training_set_multi's scheme)
        atoms = AseAtomsAdaptor.get_atoms(pmg_struct)

        # Schedule relax
        energy = lookup_total_energy_eV(
            set_id=set_id, occ_key=occ_key, model=model,
            relax_cell=relax_cell, dtype=dtype, require_converged=True
        )
        if energy is None:
            j_relax: Job = relax_structure(
                set_id=set_id,
                occ_key=occ_key,
                structure=pmg_struct,
                model=model,
                relax_cell=relax_cell,
                dtype=dtype,
                category=category,
            )
            j_relax.name = f"relax[{sig}::{occ_key[:12]}]"
            j_relax.update_metadata({"_category": category})
            sub_jobs.append(j_relax)
            energy = j_relax.output

        train_refs.append(
            CETrainRef(
                set_id=set_id,
                occ_key=occ_key,
                model=model,
                relax_cell=relax_cell,
                dtype=dtype,
                structure=pmg_struct,
            )
        )

    train_refs_out = sorted(train_refs, key=lambda train_ref: (train_ref["set_id"], train_ref["occ_key"]))
    dataset_key = compute_dataset_key([{k:v for k,v in train_ref.items() if k != "structure"} for train_ref in train_refs_out])
    output={"train_refs": train_refs_out, "dataset_key": dataset_key, "kind": "CETrainRef_dataset"}
    if not sub_jobs:
        # All references were already relaxed; just return the train_refs
        return output


    subflow = Flow(sub_jobs, name="Relax selected (parallel)")
    return Response(replace=subflow, output=output)