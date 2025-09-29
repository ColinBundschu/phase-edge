from typing import Any, Mapping

from jobflow.core.flow import Flow, JobOrder
from jobflow.core.job import job, Job, Response

from phaseedge.jobs.add_wl_block import add_wl_block
from phaseedge.jobs.ensure_ce_from_mixtures import ensure_ce_from_mixtures
from phaseedge.jobs.refine_wl_block_samples import RefineWLSpec, refine_wl_block_samples
from phaseedge.jobs.select_d_optimal_basis import select_d_optimal_basis
from phaseedge.jobs.ensure_dataset_selected import ensure_dataset_selected
from phaseedge.jobs.train_ce import train_ce
from phaseedge.jobs.store_ce_model import store_ce_model
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.schemas.ensure_ce_from_refined_wl_spec import EnsureCEFromRefinedWLSpec
from phaseedge.schemas.mixture import sublattices_from_mixtures
from phaseedge.storage.wang_landau import get_first_matching_wl_block


@job
def ensure_ce_from_refined_wl(*, spec: EnsureCEFromRefinedWLSpec) -> Mapping[str, Any] | Response:
    """Ensure a CE using refined WL data, idempotently."""
    if lookup_ce_by_key(spec.final_ce_key):
        raise RuntimeError(f"Final CE already exists for ce_key: {spec.final_ce_key}")
    
    # 1) Cache check: if CE exists, short-circuit
    wl_jobs: list[Job | Flow] = []
    wl_blocks = []
    

    for sampler_spec in spec.wl_sampler_specs:
        tip = get_first_matching_wl_block(sampler_spec)
        if tip:
            wl_blocks.append(tip["wl_block_key"])
        else:
            j_wl = add_wl_block(sampler_spec)
            j_wl.update_metadata({"_category": spec.category})
            wl_jobs.append(j_wl)
            wl_blocks.append(j_wl.output["wl_block_key"])

    # WL jobs run in parallel after CE completes (linear barrier at outer level)
    wl_flow_inner = Flow(wl_jobs, name="WL jobs (parallel)")
    if lookup_ce_by_key(spec.ce_spec.ce_key):
        ensure_wl_from_ce_flow = wl_flow_inner
    else:
        j_ce: Job = ensure_ce_from_mixtures(spec.ce_spec)
        j_ce.name = "ensure_ce"
        j_ce.update_metadata({"_category": spec.category})
        ensure_wl_from_ce_flow = Flow([j_ce, wl_flow_inner], name="Ensure WL samples from CE", order=JobOrder.LINEAR)

    # -------------------------------------------------------------------------
    # 2) Refine per WL block
    refine_jobs: list[Job | Flow] = []
    for wl_block_key in wl_blocks:
        r_spec = RefineWLSpec(
            mode=spec.refine_mode,
            n_total=spec.refine_n_total,
            per_bin_cap=spec.refine_per_bin_cap,
            strategy=spec.refine_strategy,
        )
        j_r = refine_wl_block_samples(spec=r_spec, wl_block_key=wl_block_key)
        j_r.update_metadata({"_category": spec.category})
        refine_jobs.append(j_r)
    refine_flow = Flow(refine_jobs, name="stage2: refine (parallel)")

    # -------------------------------------------------------------------------
    # 3) Select D‑optimal basis
    chains_payload = [
        {
            "wl_key": r.output["wl_key"],
            "wl_block_key": r.output["wl_block_key"],
            "samples": r.output["selected"],
        }
        for r in refine_jobs
    ]
    j_select = select_d_optimal_basis(
        ce_key=spec.ce_spec.ce_key,
        prototype=spec.ce_spec.prototype,
        prototype_params=spec.ce_spec.prototype_params,
        supercell_diag=spec.ce_spec.supercell_diag,
        endpoints=spec.endpoints,
        chains=chains_payload,
        budget=spec.budget,
        ridge=1e-10,
        wl_compoisition_maps={sampler_spec.wl_key: sampler_spec.initial_comp_map for sampler_spec in spec.wl_sampler_specs},
    )
    j_select.name = "select_d_optimal_basis"
    j_select.update_metadata({"_category": spec.category})

    # -------------------------------------------------------------------------
    # 4) Relax (parallel), grouped per composition
    j_relax = ensure_dataset_selected(
        ce_key=spec.ce_spec.ce_key,
        selected=j_select.output["chosen"],
        model=spec.train_model,
        relax_cell=spec.train_relax_cell,
        dtype=spec.train_dtype,
        category=spec.category,
    )
    j_relax.name = "ensure_dataset_selected"
    j_relax.update_metadata({"_category": spec.category})

    # -------------------------------------------------------------------------
    # 5) Fetch → Train → Store
    j_train: Job = train_ce(
        dataset_key=j_relax.output["dataset_key"],
        prototype=spec.ce_spec.prototype,
        prototype_params=spec.ce_spec.prototype_params,
        supercell_diag=spec.ce_spec.supercell_diag,
        sublattices=sublattices_from_mixtures(spec.ce_spec.mixtures),
        basis_spec=spec.ce_spec.basis_spec,
        regularization=spec.ce_spec.regularization,
        cv_seed=spec.ce_spec.seed,
        weighting=spec.ce_spec.weighting,
    )
    j_train.name = "train_ce"
    j_train.update_metadata({"_category": spec.category})

    j_store: Job = store_ce_model(
        ce_key=spec.final_ce_key,
        prototype=spec.ce_spec.prototype,
        prototype_params=spec.ce_spec.prototype_params,
        supercell_diag=spec.ce_spec.supercell_diag,
        algo_version="refined-wl-dopt-v2",
        sources=[spec.source],
        model=spec.train_model,
        relax_cell=spec.train_relax_cell,
        dtype=spec.train_dtype,
        basis_spec=spec.ce_spec.basis_spec,
        regularization=spec.ce_spec.regularization,
        weighting=spec.ce_spec.weighting,
        dataset_key=j_relax.output["dataset_key"],
        payload=j_train.output["payload"],
        stats=j_train.output["stats"],
        design_metrics=j_train.output["design_metrics"],
    )
    j_store.name = "store_ce_model"
    j_store.update_metadata({"_category": spec.category})

    # -------------------------------------------------------------------------
    base_flow = [ensure_wl_from_ce_flow] if ensure_wl_from_ce_flow else []
    flow = Flow(
        [*base_flow, refine_flow, j_select, j_relax, j_train, j_store],
        name="Ensure CE from refined WL",
    )

    out = {
        "initial_ce_key": spec.ce_spec.ce_key,
        "final_ce_key": spec.final_ce_key,
        "wl_blocks": wl_blocks,
        "refines": [
            {
                "wl_key": r.output["wl_key"],
                "wl_block_key": r.output["wl_block_key"],
                "refine_key": r.output["refine_key"],
            }
            for r in refine_jobs
        ],
        "selection_seed_size": j_select.output["seed_size"],
        "selection_budget": spec.budget,
    }
    return Response(replace=flow, output=out)
