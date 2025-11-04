from typing import Any, Mapping, Sequence
from jobflow.core.flow import Flow
from jobflow.core.job import job, Response, Job

from phaseedge.schemas.calc_spec import CalcSpec
from phaseedge.schemas.mixture import Mixture, composition_map_sig
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.jobs.evaluate_structure import evaluate_structure
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.storage.cetrainref_dataset import CETrainRef, Dataset
from phaseedge.storage.store import lookup_total_energy_eV
from phaseedge.utils.keys import compute_set_id, occ_key_for_structure, rng_for_index
from pymatgen.io.ase import AseAtomsAdaptor


@job(data="train_refs")
def ensure_dataset_compositions(
    *,
    prototype_spec: PrototypeSpec,
    supercell_diag: tuple[int, int, int],
    mixtures: Sequence[Mixture],
    calc_spec: CalcSpec,
    category: str,
) -> Mapping[str, Any] | Response:
    if not mixtures:
        raise ValueError("mixtures must be a non-empty sequence.")

    train_refs: list[CETrainRef] = []
    sub_jobs: list[Job | Flow] = []
    for mixture in mixtures:
        set_id = compute_set_id(
            prototype_spec=prototype_spec,
            supercell_diag=supercell_diag,
            composition_map=mixture.composition_map,
            seed=mixture.seed,
        )
        duplicate_map = set()
        for idx in range(mixture.K):
            rng = rng_for_index(set_id, idx)
            structure = make_one_snapshot(
                primitive_cell=prototype_spec.primitive_cell,
                supercell_diag=supercell_diag,
                composition_map=mixture.composition_map,
                rng=rng,
            )
            occ_key = occ_key_for_structure(structure)
            energy = lookup_total_energy_eV(set_id=set_id, occ_key=occ_key, calc_spec=calc_spec)
            if energy is None and occ_key not in duplicate_map:
                j_relax = evaluate_structure(
                    set_id=set_id,
                    occ_key=occ_key,
                    structure=structure,
                    calc_spec=calc_spec,
                    category=category,
                    prototype_spec=prototype_spec,
                    supercell_diag=supercell_diag,
                )
                j_relax.name = f"relax_composition::{composition_map_sig(mixture.composition_map)}"
                j_relax.update_metadata({"_category": category})
                sub_jobs.append(j_relax)
                duplicate_map.add(occ_key)

            train_refs.append(
                CETrainRef(
                    set_id=set_id,
                    occ_key=occ_key,
                    calc_spec=calc_spec,
                    structure=structure,
                )
            )

    output = Dataset(train_refs).jobflow_output
    if not sub_jobs:
        # All references were already relaxed; just return the train_refs
        return output

    subflow = Flow(sub_jobs, name="Relax + extract energies (parallel)")
    return Response(replace=subflow, output=output)
