from typing import Any, Mapping, Sequence, cast

import numpy as np
from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow
from pymatgen.io.ase import AseAtomsAdaptor
from ase.atoms import Atoms

from phaseedge.jobs.decide_relax import check_or_schedule_relax, extract_relax_energy
from phaseedge.jobs.train_ce import CETrainRef
from phaseedge.schemas.mixture import counts_sig
from phaseedge.utils.keys import occ_key_for_atoms
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
    for r_ix, rec in enumerate(selected):
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

        # Compute a structure-based key (matches fetch_training_set_multi's scheme)
        atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
        occ_key = occ_key_for_atoms(cast(Atoms, atoms))

        # Schedule relax
        j_relax: Job = check_or_schedule_relax(
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

        # 2) extract scalar energy (depends on j_relax)
        j_energy = extract_relax_energy(doc=j_relax.output)
        j_energy.name = f"energy[{r_ix}:{occ_key[:12]}]"
        j_energy.update_metadata({"_category": category})
        sub_jobs.append(j_energy)

        train_refs.append(
            CETrainRef(
                set_id=set_id,
                occ_key=occ_key,
                model=model,
                relax_cell=relax_cell,
                dtype=dtype,
                energy=cast(float, j_energy.output),
                structure=pmg_struct,
            )
        )

    train_refs_out = sorted(train_refs, key=lambda train_ref: (train_ref["set_id"], train_ref["occ_key"]))
    subflow = Flow(sub_jobs, name="Relax selected (parallel)")
    return Response(replace=subflow, output={"train_refs": train_refs_out})
