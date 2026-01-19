from typing import Any, Mapping

import hashlib

from jobflow.core.flow import Flow, JobOrder
from jobflow.core.job import job, Job, Response

from phaseedge.jobs.add_wl_block import add_wl_block
from phaseedge.jobs.ensure_ce_from_mixtures import ensure_ce_from_mixtures
from phaseedge.jobs.select_d_optimal_basis import select_d_optimal_basis
from phaseedge.jobs.ensure_dataset_selected import ensure_dataset_selected
from phaseedge.jobs.train_ce import train_ce
from phaseedge.jobs.store_ce_model import store_ce_model
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.schemas.ensure_dopt_ce_spec import EnsureDoptCESpec
from phaseedge.schemas.mixture import sublattices_from_mixtures
from phaseedge.storage.wang_landau import get_first_matching_wl_block, lookup_wl_block_by_key


def _occ_hash(occ: list[int]) -> str:
    return hashlib.sha256(bytes(int(x) & 0xFF for x in occ)).hexdigest()


@job
def aggregate_wl_block_samples(*, wl_block_key: str) -> Mapping[str, Any]:
    """
    Lightweight pass-through aggregator that returns ALL samples from a WL block,
    deterministically sorted by (bin asc, occ_hash asc).
    """
    block = lookup_wl_block_by_key(wl_block_key)
    if not block:
        raise RuntimeError(f"WL block not found for wl_block_key={wl_block_key}")

    bin_samples = block.get("bin_samples")
    if not isinstance(bin_samples, list):
        raise RuntimeError("WL block document lacks 'bin_samples' list.")

    samples = []
    for rec in bin_samples:
        b = int(rec["bin"])
        occ = [int(x) for x in rec["occ"]]
        samples.append({"bin": b, "occ": occ})

    # Deterministic ordering for reproducibility
    samples.sort(key=lambda s: (int(s["bin"]), _occ_hash(s["occ"])))

    return {
        "wl_key": block["wl_key"],
        "wl_block_key": wl_block_key,
        "n_selected": len(samples),
        "selected": samples,
    }


@job
def ensure_dopt_ce(*, spec: EnsureDoptCESpec) -> Mapping[str, Any] | Response:
    """Ensure a CE using WL data, idempotently (aggregate all WL samples)."""
    if lookup_ce_by_key(spec.final_ce_key):
        raise RuntimeError(f"Final CE already exists for ce_key: {spec.final_ce_key}")

    # 1) Ensure WL blocks exist (or extend); build list of wl_block_keys
    wl_jobs: list[Job | Flow] = []
    wl_blocks: list[str] = []

    for sampler_spec in spec.wl_sampler_specs:
        tip = get_first_matching_wl_block(sampler_spec)
        if tip:
            wl_blocks.append(tip["wl_block_key"])
        else:
            j_wl = add_wl_block(sampler_spec, name=f"extend_wl::{sampler_spec.wl_key}", category=spec.category)
            wl_jobs.append(j_wl)
            wl_blocks.append(j_wl.output["wl_block_key"])

    # WL jobs run in parallel after CE completes (linear barrier at outer level)
    wl_flow_inner = Flow(wl_jobs, name="WL jobs (parallel)")
    if lookup_ce_by_key(spec.ce_spec.ce_key):
        ensure_wl_from_ce_flow = wl_flow_inner
    else:
        j_ce: Job = ensure_ce_from_mixtures(spec.ce_spec)
        j_ce.name = f"ensure_ce::{spec.ce_spec.ce_key}"
        j_ce.update_metadata({"_category": spec.category})
        ensure_wl_from_ce_flow = Flow([j_ce, wl_flow_inner], name="Ensure WL samples from CE", order=JobOrder.LINEAR)

    # -------------------------------------------------------------------------
    # 2) Aggregate all samples per WL block
    agg_jobs: list[Flow | Job] = []
    for wl_block_key in wl_blocks:
        j_agg = aggregate_wl_block_samples(wl_block_key=wl_block_key)
        j_agg.update_metadata({"_category": spec.category})
        agg_jobs.append(j_agg)
    agg_flow = Flow(agg_jobs, name="stage2: aggregate WL samples (parallel)")

    # -------------------------------------------------------------------------
    # 3) Select D-optimal basis directly from aggregated samples
    chains_payload = [
        {
            "wl_key": a.output["wl_key"],
            "wl_block_key": a.output["wl_block_key"],
            "samples": a.output["selected"],
        }
        for a in agg_jobs
    ]
    j_select = select_d_optimal_basis(
        ce_key=spec.ce_spec.ce_key,
        prototype_spec=spec.ce_spec.prototype_spec,
        supercell_diag=spec.ce_spec.supercell_diag,
        endpoints=spec.endpoints,
        chains=chains_payload,
        budget=spec.budget,
        ridge=1e-10,
        wl_composition_maps={sampler_spec.wl_key: sampler_spec.initial_comp_map for sampler_spec in spec.wl_sampler_specs},
    )
    j_select.name = "select_d_optimal_basis"
    j_select.update_metadata({"_category": spec.category})

    # -------------------------------------------------------------------------
    # 4) Relax (parallel), grouped per composition
    j_relax = ensure_dataset_selected(
        ce_key=spec.ce_spec.ce_key,
        selected=j_select.output["chosen"],
        calc_spec=spec.calc_spec,
        category=spec.category,
        prototype_spec=spec.ce_spec.prototype_spec,
        supercell_diag=spec.ce_spec.supercell_diag,
        skip_unrelaxed=spec.allow_partial,
    )
    j_relax.name = "ensure_dataset_selected"
    j_relax.update_metadata({"_category": spec.category})

    # -------------------------------------------------------------------------
    # 5) Fetch → Train → Store
    j_train: Job = train_ce(
        dataset_key=j_relax.output["dataset_key"],
        prototype_spec=spec.ce_spec.prototype_spec,
        supercell_diag=spec.ce_spec.supercell_diag,
        sublattices=sublattices_from_mixtures(spec.ce_spec.mixtures),
        basis_spec=spec.ce_spec.basis_spec,
        regularization=spec.ce_spec.regularization,
        cv_seed=spec.ce_spec.seed,
        weighting=spec.ce_spec.weighting,
    )
    j_train.name = f"train_ce::{spec.final_ce_key}"
    j_train.update_metadata({"_category": spec.category})

    j_store: Job = store_ce_model(
        ce_key=spec.final_ce_key,
        prototype_spec=spec.ce_spec.prototype_spec,
        supercell_diag=spec.ce_spec.supercell_diag,
        algo_version=spec.algo_version,
        sources=[spec.source],
        calc_spec=spec.calc_spec,
        basis_spec=spec.ce_spec.basis_spec,
        regularization=spec.ce_spec.regularization,
        weighting=spec.ce_spec.weighting,
        dataset_key=j_relax.output["dataset_key"],
        payload=j_train.output["payload"],
        stats=j_train.output["stats"],
        design_metrics=j_train.output["design_metrics"],
    )
    j_store.name = f"store_ce_model::{spec.final_ce_key}"
    j_store.update_metadata({"_category": spec.category})

    # -------------------------------------------------------------------------
    base_flow = [ensure_wl_from_ce_flow] if ensure_wl_from_ce_flow else []
    flow = Flow(
        [*base_flow, agg_flow, j_select, j_relax, j_train, j_store],
        name="Ensure CE from WL (no refine)",
    )

    out = {
        "initial_ce_key": spec.ce_spec.ce_key,
        "final_ce_key": spec.final_ce_key,
        "wl_blocks": wl_blocks,
        # Preserve a similar shape to previous outputs for consumers that might read it
        "refines": [
            {
                "wl_key": a.output["wl_key"],
                "wl_block_key": a.output["wl_block_key"],
            }
            for a in agg_jobs
        ],
        "selection_seed_size": j_select.output["seed_size"],
        "selection_budget": spec.budget,
    }
    return Response(replace=flow, output=out)
