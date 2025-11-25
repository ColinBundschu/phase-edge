from typing import Any, Mapping, Sequence, cast

import numpy as np
from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow

from phaseedge.jobs.evaluate_structure import evaluate_structure
from phaseedge.schemas.calc_spec import CalcSpec
from phaseedge.schemas.mixture import composition_map_sig
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.storage.cetrainref_dataset import CETrainRef, Dataset
from phaseedge.storage.store import lookup_total_energy_eV
from phaseedge.utils.keys import occ_key_for_structure
from phaseedge.jobs.store_ce_model import rehydrate_ensemble_by_ce_key


@job(data="train_refs")
def ensure_dataset_selected(
    *,
    ce_key: str,
    selected: Sequence[Mapping[str, Any]],
    calc_spec: CalcSpec,
    prototype_spec: PrototypeSpec,
    supercell_diag: tuple[int, int, int],
    category: str = "gpu",
) -> Mapping[str, Any] | Response:
    ensemble = rehydrate_ensemble_by_ce_key(ce_key)

    sub_jobs: list[Job | Flow] = []
    train_refs: list[CETrainRef] = []
    for rec in selected:
        composition_map = rec["composition_map"]
        sig = composition_map_sig(composition_map)

        # Build structure from occupancy via CE processor
        occ_seq = cast(Sequence[int], rec["occ"])
        occ_arr = np.asarray([int(x) for x in occ_seq], dtype=np.int32)
        pmg_struct = ensemble.processor.structure_from_occupancy(occ_arr)
        occ_key = occ_key_for_structure(pmg_struct)

        # Schedule relax
        energy_result = lookup_total_energy_eV(occ_key=occ_key, calc_spec=calc_spec)
        if energy_result is None or energy_result.max_force_eV_per_A > calc_spec.max_force_eV_per_A:
            j_relax = evaluate_structure(
                occ_key=occ_key,
                structure=pmg_struct,
                calc_spec=calc_spec,
                category=category,
                prototype_spec=prototype_spec,
                supercell_diag=supercell_diag,
                comp_map_sig=sig,
            )
            j_relax.name = f"relax_selected::{sig}::{occ_key[:12]}"
            j_relax.update_metadata({"_category": category})
            sub_jobs.append(j_relax)

        train_refs.append(
            CETrainRef(
                composition_map=composition_map,
                occ_key=occ_key,
                calc_spec=calc_spec,
                structure=pmg_struct,
            )
        )

    output = Dataset(train_refs).jobflow_output
    if not sub_jobs:
        # All references were already relaxed; just return the train_refs
        return output

    subflow = Flow(sub_jobs, name="Relax selected (parallel)")
    return Response(replace=subflow, output=output)