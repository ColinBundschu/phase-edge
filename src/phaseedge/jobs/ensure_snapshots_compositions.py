from typing import Any, Mapping, Sequence
from jobflow.core.flow import Flow
from jobflow.core.job import job, Response, Job

from phaseedge.jobs.train_ce import CETrainRef
from phaseedge.schemas.mixture import Mixture
from phaseedge.science.prototypes import PrototypeName, make_prototype
from phaseedge.jobs.decide_relax import relax_structure
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.storage.store import lookup_total_energy_eV
from phaseedge.utils.keys import compute_dataset_key, compute_set_id, occ_key_for_atoms, rng_for_index
from pymatgen.io.ase import AseAtomsAdaptor


@job
def ensure_snapshots_compositions(
    *,
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],
    mixtures: Sequence[Mixture],
    model: str,
    relax_cell: bool,
    dtype: str,
    category: str = "gpu",
) -> Mapping[str, Any] | Response:
    if not mixtures:
        raise ValueError("mixtures must be a non-empty sequence.")

    # Build prototype once
    conv_cell = make_prototype(prototype, **(prototype_params or {}))

    train_refs: list[CETrainRef] = []
    sub_jobs: list[Job | Flow] = []
    for m_ix, mixture in enumerate(mixtures):
        set_id = compute_set_id(
            prototype=prototype,
            prototype_params=prototype_params if prototype_params else None,
            supercell_diag=supercell_diag,
            composition_map=mixture.composition_map,
            seed=mixture.seed,
        )
        for idx in range(mixture.K):
            rng = rng_for_index(set_id, idx)
            snapshot = make_one_snapshot(
                conv_cell=conv_cell,
                supercell_diag=supercell_diag,
                composition_map=mixture.composition_map,
                rng=rng,
            )
            occ_key = occ_key_for_atoms(snapshot)
            structure = AseAtomsAdaptor.get_structure(snapshot)  # pyright: ignore[reportArgumentType]

            # 1) schedule relax
            energy = lookup_total_energy_eV(
                set_id=set_id, occ_key=occ_key, model=model,
                relax_cell=relax_cell, dtype=dtype, require_converged=True
            )
            if energy is None:
                j_relax = relax_structure(
                    set_id=set_id,
                    occ_key=occ_key,
                    structure=structure,
                    model=model,
                    relax_cell=relax_cell,
                    dtype=dtype,
                    category=category,
                )
                j_relax.name = f"relax[{m_ix}:{idx}:{occ_key[:12]}]"
                j_relax.update_metadata({"_category": category})
                sub_jobs.append(j_relax)
                energy = j_relax.output

            # 3) reference the scalar energy output (clean OutputReference[float])
            train_refs.append(
                CETrainRef(
                    set_id=set_id,
                    occ_key=occ_key,
                    model=model,
                    relax_cell=relax_cell,
                    dtype=dtype,
                    structure=structure,
                )
            )

    train_refs_out = sorted(train_refs, key=lambda r: (r["set_id"], r["occ_key"]))
    dataset_key = compute_dataset_key([{k:v for k,v in train_ref.items() if k != "structure"} for train_ref in train_refs_out])
    output={"train_refs": train_refs_out, "dataset_key": dataset_key, "kind": "CETrainRef_dataset"}
    if not sub_jobs:
        # All references were already relaxed; just return the train_refs
        return output


    subflow = Flow(sub_jobs, name="Relax + extract energies (parallel)")
    return Response(replace=subflow, output=output)
