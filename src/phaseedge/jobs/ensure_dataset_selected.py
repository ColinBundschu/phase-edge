from typing import Any, Mapping, Sequence, cast

import numpy as np
from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow

from phaseedge.jobs.relax_structure import relax_structure
from phaseedge.jobs.train_ce import CETrainRef, train_refs_exist
from phaseedge.schemas.mixture import composition_map_sig, counts_sig
from phaseedge.storage.store import lookup_total_energy_eV
from phaseedge.utils.keys import compute_dataset_key, occ_key_for_structure
from phaseedge.jobs.store_ce_model import rehydrate_ensemble_by_ce_key


@job(data="train_refs")
def ensure_dataset_selected(
    *,
    ce_key: str,
    selected: Sequence[Mapping[str, Any]],
    model: str,
    relax_cell: bool,
    category: str = "gpu",
    set_id_prefix: str = "wlref",
) -> Mapping[str, Any] | Response:
    ensemble = rehydrate_ensemble_by_ce_key(ce_key)

    sub_jobs: list[Job | Flow] = []
    train_refs: list[CETrainRef] = []
    for rec in selected:
        sig = composition_map_sig(rec["composition_map"])
        set_id = f"{set_id_prefix}::{ce_key}::{sig}"

        # Build structure from occupancy via CE processor
        occ_seq = cast(Sequence[int], rec["occ"])
        occ_arr = np.asarray([int(x) for x in occ_seq], dtype=np.int32)
        pmg_struct = ensemble.processor.structure_from_occupancy(occ_arr)
        occ_key = occ_key_for_structure(pmg_struct)

        # Schedule relax
        energy = lookup_total_energy_eV(set_id=set_id, occ_key=occ_key, model=model, relax_cell=relax_cell)
        if energy is None:
            j_relax: Job = relax_structure(
                set_id=set_id,
                occ_key=occ_key,
                structure=pmg_struct,
                model=model,
                relax_cell=relax_cell,
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
                structure=pmg_struct,
            )
        )

    train_refs_out = sorted(train_refs, key=lambda r: (r["set_id"], r["occ_key"]))
    dataset_key = compute_dataset_key([{k:v for k,v in train_ref.items() if k != "structure"} for train_ref in train_refs_out])
    output = {"dataset_key": dataset_key}
    if not train_refs_exist(dataset_key):
        output = output | {"train_refs": train_refs_out, "kind": "CETrainRef_dataset"}
    
    if not sub_jobs:
        # All references were already relaxed; just return the train_refs
        return output

    subflow = Flow(sub_jobs, name="Relax selected (parallel)")
    return Response(replace=subflow, output=output)