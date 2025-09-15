from typing import Any, Mapping

from jobflow.core.flow import Flow, JobOrder
from jobflow.core.job import job, Job, Response

from phaseedge.jobs.ensure_ce_from_mixtures import ensure_ce_from_mixtures
from phaseedge.jobs.ensure_wl_samples import ensure_wl_samples
from phaseedge.jobs.refine_wl_block import RefineWLSpec, refine_wl_block
from phaseedge.jobs.select_d_optimal_basis import select_d_optimal_basis
from phaseedge.jobs.relax_selected_from_wl import relax_selected_from_wl
from phaseedge.jobs.fetch_training_set_multi import fetch_training_set_multi
from phaseedge.jobs.train_ce import train_ce
from phaseedge.jobs.store_ce_model import store_ce_model
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.schemas.ensure_ce_from_refined_wl_spec import EnsureCEFromRefinedWLSpec
from phaseedge.schemas.mixture import counts_sig, sublattices_from_mixtures
from phaseedge.schemas.wl_sampler_spec import WLSamplerSpec


@job
def ensure_ce_from_refined_wl(*, spec: EnsureCEFromRefinedWLSpec) -> Mapping[str, Any] | Response:
    """Ensure a CE using refined WL data, idempotently."""
    # -------------------------------------------------------------------------
    # Early exit keying
    final_ce_key = spec.final_ce_key
    existing_ce = lookup_ce_by_key(final_ce_key)
    if existing_ce is not None:
        return existing_ce

    # -------------------------------------------------------------------------
    # 1) Ensure CE + WL
    ce_key = spec.ce_spec.ce_key

    j_ce: Job = ensure_ce_from_mixtures(spec.ce_spec)
    j_ce.name = "ensure_ce"
    j_ce.update_metadata({"_category": spec.category})

    wl_jobs: list[Job | Flow] = []
    wl_chunks: list[Mapping[str, Any]] = []
    
    wl_composition_map = {wl_key: comp for wl_key, comp in spec.wl_key_composition_pairs}
    for wl_key, composition_counts in wl_composition_map.items():
        sig = counts_sig(composition_counts)
        run_spec = WLSamplerSpec(
            wl_key=wl_key,
            ce_key=ce_key,
            bin_width=spec.wl_bin_width,
            steps=spec.wl_steps_to_run,
            sublattice_labels=spec.sublattice_labels,
            composition_counts=composition_counts,
            step_type=spec.wl_step_type,
            check_period=spec.wl_check_period,
            update_period=spec.wl_update_period,
            seed=spec.wl_seed,
            samples_per_bin=spec.wl_samples_per_bin,
        )

        j_wl: Job = ensure_wl_samples(run_spec)
        j_wl.name = f"ensure_wl_samples::{wl_key[:12]}::{sig}"
        j_wl.update_metadata({"_category": spec.category, "wl_key": wl_key})
        wl_jobs.append(j_wl)
        wl_chunks.append({
            "counts": sig,
            "wl_key": wl_key,
            "hash": j_wl.output["hash"],
        })

    # WL jobs run in parallel after CE completes (linear barrier at outer level)
    wl_flow_inner = Flow(wl_jobs, name="WL jobs (parallel)")
    ensure_wl_from_ce_flow = Flow([j_ce, wl_flow_inner], name="Ensure WL samples from CE", order=JobOrder.LINEAR)

    # -------------------------------------------------------------------------
    # 2) Refine per WL checkpoint
    refine_jobs: list[Job | Flow] = []
    for chunk in wl_chunks:
        wl_key_ref = chunk["wl_key"]
        ck_hash_ref = chunk["hash"]
        r_spec = RefineWLSpec(
            wl_key=wl_key_ref,
            mode=spec.refine_mode,
            n_total=spec.refine_total,
            per_bin_cap=spec.refine_per_bin_cap,
            strategy=spec.refine_strategy,
        )
        j_r = refine_wl_block(r_spec, checkpoint_hash=ck_hash_ref)
        j_r.name = f"refine_wl_block[{wl_key_ref[:12]}]"
        j_r.update_metadata({"_category": spec.category})
        refine_jobs.append(j_r)
    refine_flow = Flow(refine_jobs, name="stage2: refine (parallel)")

    # -------------------------------------------------------------------------
    # 3) Select D‑optimal basis
    chains_payload = [
        {
            "wl_key": r.output["wl_key"],
            "checkpoint_hash": r.output["checkpoint_hash"],
            "samples": r.output["selected"],
        }
        for r in refine_jobs
    ]
    j_select = select_d_optimal_basis(
        ce_key=ce_key,
        prototype=spec.ce_spec.prototype,
        prototype_params=spec.ce_spec.prototype_params,
        supercell_diag=spec.ce_spec.supercell_diag,
        endpoints=spec.endpoints,
        chains=chains_payload,
        budget=spec.budget,
        ridge=1e-10,
        wl_counts_map=wl_composition_map,
    )
    j_select.name = "select_d_optimal_basis"
    j_select.update_metadata({"_category": spec.category})

    # -------------------------------------------------------------------------
    # 4) Relax (parallel), grouped per composition
    j_relax = relax_selected_from_wl(
        ce_key=ce_key,
        selected=j_select.output["chosen"],
        wl_counts_map=wl_composition_map,
        model=spec.train_model,
        relax_cell=spec.train_relax_cell,
        dtype=spec.train_dtype,
        category=spec.category,
    )
    j_relax.name = "relax_selected_from_wl"
    j_relax.update_metadata({"_category": spec.category})

    # -------------------------------------------------------------------------
    # 5) Fetch → Train → Store
    j_fetch: Job = fetch_training_set_multi(
        groups=j_relax.output["groups"],
        prototype=spec.ce_spec.prototype,
        prototype_params=dict(spec.ce_spec.prototype_params),
        supercell_diag=tuple(spec.ce_spec.supercell_diag),
        model=spec.train_model,
        relax_cell=spec.train_relax_cell,
        dtype=spec.train_dtype,
        ce_key_for_rebuild=ce_key,
    )
    j_fetch.name = "fetch_training_set_multi"
    j_fetch.update_metadata({"_category": spec.category})

    j_train: Job = train_ce(
        structures=j_fetch.output["structures"],
        energies=j_fetch.output["energies"],
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
        ce_key=final_ce_key,
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
        train_refs=j_fetch.output["train_refs"],
        dataset_hash=j_fetch.output["dataset_hash"],
        payload=j_train.output["payload"],
        stats=j_train.output["stats"],
        design_metrics=j_train.output["design_metrics"],
    )
    j_store.name = "store_ce_model"
    j_store.update_metadata({"_category": spec.category})

    # -------------------------------------------------------------------------
    flow = Flow(
        [ensure_wl_from_ce_flow, refine_flow, j_select, j_relax, j_fetch, j_train, j_store],
        name="Ensure CE from refined WL",
    )

    out = {
        "initial_ce_key": ce_key,
        "final_ce_key": final_ce_key,
        "wl_chunks": wl_chunks,
        "refines": [
            {
                "wl_key": r.output["wl_key"],
                "checkpoint_hash": r.output["checkpoint_hash"],
                "refine_key": r.output["refine_key"],
            }
            for r in refine_jobs
        ],
        "selection_seed_size": j_select.output["seed_size"],
        "selection_budget": spec.budget,
    }
    return Response(replace=flow, output=out)
