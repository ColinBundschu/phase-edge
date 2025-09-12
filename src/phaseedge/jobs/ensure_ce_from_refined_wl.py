"""
Modified ensure_ce_from_refined_wl job to support idempotent shortcutting.

This version computes the deterministic base CE key and the final refined intent
CE key up-front using the same logic as the original code. It then checks
whether a CE model with the final key already exists in the backing store via
``lookup_ce_by_key``. If such a model is found, the job returns the existing
document immediately instead of launching the full workflow. This makes the
workflow idempotent and avoids recomputing an identical CE.
"""

from typing import Any, Literal, Mapping, Final

from jobflow.core.flow import Flow, JobOrder
from jobflow.core.job import job, Job, Response

from phaseedge.jobs.ensure_wl_samples_from_ce import (
    EnsureWLSamplesFromCESpec,
    ensure_wl_samples_from_ce,
)
from phaseedge.jobs.refine_wl_block import RefineWLSpec, refine_wl_block
from phaseedge.jobs.select_d_optimal_basis import select_d_optimal_basis
from phaseedge.jobs.relax_selected_from_wl import relax_selected_from_wl
from phaseedge.jobs.fetch_training_set_multi import fetch_training_set_multi
from phaseedge.jobs.train_ce import train_ce
from phaseedge.jobs.store_ce_model import store_ce_model
from phaseedge.jobs.prepare_refined_wl_sources import prepare_refined_wl_sources
from phaseedge.schemas.mixture import canonical_counts, compute_ce_key, compute_wl_key
from phaseedge.storage.ce_store import lookup_ce_by_key


def _counts_sig(counts: Mapping[str, int]) -> str:
    cc = canonical_counts(counts)
    return ",".join(f"{k}:{int(v)}" for k, v in cc.items())


@job
def ensure_ce_from_refined_wl(
    *,
    ensure_wl_spec: EnsureWLSamplesFromCESpec,
    refine_n_total: int = 25,
    refine_per_bin_cap: int = 5,
    refine_strategy: Literal["energy_spread", "energy_stratified", "hash_round_robin"] = "energy_spread",
    train_model: str = "MACE-MPA-0",
    train_relax_cell: bool = False,
    train_dtype: str = "float64",
    budget: int = 64,
    category: str = "gpu",
) -> Mapping[str, Any] | Response:
    """Ensure a CE using refined WL data, idempotently."""
    # -------------------------------------------------------------------------
    # Early exit keying
    ce_spec = ensure_wl_spec.ce_spec
    _canon_mix_for_key: list[dict[str, Any]] = []
    for elem in ce_spec.mixture:
        counts = canonical_counts(elem.get("counts", {}))
        K = int(elem.get("K", 0))
        seed = int(elem.get("seed", ce_spec.default_seed))
        if counts and K > 0:
            _canon_mix_for_key.append({"counts": counts, "K": K, "seed": seed})
    _endpoints_canon_for_key: list[dict[str, int]] = [canonical_counts(e) for e in ensure_wl_spec.endpoints]
    for e in _endpoints_canon_for_key:
        _canon_mix_for_key.append({"counts": e, "K": 1, "seed": 0})
    _sources_base_key = [{"type": "composition", "elements": _canon_mix_for_key}]
    _base_ce_key_for_key = compute_ce_key(
        prototype=ce_spec.prototype,
        prototype_params=dict(ce_spec.prototype_params),
        supercell_diag=ce_spec.supercell_diag,
        replace_element=ce_spec.replace_element,
        sources=_sources_base_key,
        model=ce_spec.model,
        relax_cell=ce_spec.relax_cell,
        dtype=ce_spec.dtype,
        basis_spec=dict(ce_spec.basis_spec),
        regularization=dict(ce_spec.regularization or {}),
        algo_version="randgen-3-comp-1",
        weighting=dict(ce_spec.weighting or {}),
    )
    _wl_policy_for_key = {
        "bin_width": float(ensure_wl_spec.wl_bin_width),
        "step_type": str(ensure_wl_spec.wl_step_type),
        "check_period": int(ensure_wl_spec.wl_check_period),
        "update_period": int(ensure_wl_spec.wl_update_period),
        "seed": int(ensure_wl_spec.wl_seed),
    }
    _ensure_policy_for_key = {
        "steps_to_run": int(ensure_wl_spec.wl_steps_to_run),
        "samples_per_bin": int(ensure_wl_spec.wl_samples_per_bin),
    }
    _refine_options_for_key = {
        "mode": ("all" if int(refine_n_total) == 0 else "refine"),
        "n_total": (None if int(refine_n_total) == 0 else int(refine_n_total)),
        "per_bin_cap": int(refine_per_bin_cap),
        "strategy": str(refine_strategy),
    }
    _dopt_options_for_key = {
        "budget": int(budget),
        "ridge": float(1e-10),
        "tie_breaker": "bin_then_hash",
    }
    _src_intent_for_key = {
        "type": "wl_refined_intent",
        "base_ce_key": str(_base_ce_key_for_key),
        "endpoints": _endpoints_canon_for_key,
        "wl_policy": _wl_policy_for_key,
        "ensure": _ensure_policy_for_key,
        "refine": _refine_options_for_key,
        "dopt": _dopt_options_for_key,
        "versions": {
            "refine": "refine-wl-v1",
            # *** changed to round-robin version tag ***
            "dopt": "dopt-rr-sm-v1",
            "sampler": "wl-grid-v1",
        },
    }
    _final_ce_key_for_key = compute_ce_key(
        prototype=ce_spec.prototype,
        prototype_params=dict(ce_spec.prototype_params),
        supercell_diag=ce_spec.supercell_diag,
        replace_element=ce_spec.replace_element,
        sources=[_src_intent_for_key],
        model=str(train_model),
        relax_cell=bool(train_relax_cell),
        dtype=str(train_dtype),
        basis_spec=dict(ce_spec.basis_spec),
        regularization=dict(ce_spec.regularization or {}),
        algo_version="refined-wl-dopt-v2",
        weighting=dict(ce_spec.weighting or {}),
    )
    existing_ce = lookup_ce_by_key(_final_ce_key_for_key)
    if existing_ce is not None:
        return existing_ce

    # -------------------------------------------------------------------------
    # 1) Ensure CE + WL
    j_wl = ensure_wl_samples_from_ce(ensure_wl_spec)  # type: ignore[assignment]
    j_wl.name = "ensure_wl_samples_from_ce"
    j_wl.update_metadata({"_category": category})

    # -------------------------------------------------------------------------
    # Plan WL chains
    ce_spec = ensure_wl_spec.ce_spec

    canon_mix: list[dict[str, Any]] = []
    for elem in ce_spec.mixture:
        counts = canonical_counts(elem.get("counts", {}))
        K = int(elem.get("K", 0))
        seed = int(elem.get("seed", ce_spec.default_seed))
        if not counts or K <= 0:
            raise ValueError(f"Invalid mixture element: counts={counts}, K={K}")
        canon_mix.append({"counts": counts, "K": K, "seed": seed})
    endpoints_canon: list[dict[str, int]] = [canonical_counts(e) for e in ensure_wl_spec.endpoints]
    for e in endpoints_canon:
        canon_mix.append({"counts": e, "K": 1, "seed": 0})

    algo_base: Final = "randgen-3-comp-1"
    sources_base = [{"type": "composition", "elements": canon_mix}]
    ce_key_planned = compute_ce_key(
        prototype=ce_spec.prototype,
        prototype_params=dict(ce_spec.prototype_params),
        supercell_diag=ce_spec.supercell_diag,
        replace_element=ce_spec.replace_element,
        sources=sources_base,
        model=ce_spec.model,
        relax_cell=ce_spec.relax_cell,
        dtype=ce_spec.dtype,
        basis_spec=dict(ce_spec.basis_spec),
        regularization=dict(ce_spec.regularization or {}),
        algo_version=algo_base,
        weighting=dict(ce_spec.weighting or {}),
    )

    endpoint_fps = {_counts_sig(e) for e in endpoints_canon}
    planned: list[dict[str, str]] = []
    seen: set[str] = set()
    wl_counts_map: dict[str, Mapping[str, int]] = {}

    for elem in ce_spec.mixture:
        counts = canonical_counts(elem.get("counts", {}))
        if not counts:
            continue
        sig = _counts_sig(counts)
        if sig in endpoint_fps or sig in seen:
            seen.add(sig)
            continue
        seen.add(sig)
        wl_key = compute_wl_key(
            ce_key=ce_key_planned,
            bin_width=float(ensure_wl_spec.wl_bin_width),
            step_type=str(ensure_wl_spec.wl_step_type),
            composition_counts=counts,
            check_period=int(ensure_wl_spec.wl_check_period),
            update_period=int(ensure_wl_spec.wl_update_period),
            seed=int(ensure_wl_spec.wl_seed),
            algo_version="wl-grid-v1",
        )
        planned.append({"counts_sig": sig, "wl_key": wl_key})
        wl_counts_map[wl_key] = counts

    # -------------------------------------------------------------------------
    # 2) Refine per WL checkpoint
    refine_jobs: list[Job | Flow] = []
    for i, rec in enumerate(planned):
        wl_key = rec["wl_key"]
        ck_hash_ref = j_wl.output["wl_chunks"][i]["hash"]  # index‑based; no iteration
        mode = "all" if int(refine_n_total) == 0 else "refine"
        r_spec = RefineWLSpec(
            wl_key=str(wl_key),
            mode=mode,
            n_total=None if int(refine_n_total) == 0 else int(refine_n_total),
            per_bin_cap=int(refine_per_bin_cap),
            strategy=refine_strategy,
        )
        j_r = refine_wl_block(r_spec, checkpoint_hash=ck_hash_ref)  # type: ignore[assignment]
        j_r.name = f"refine_wl_block[{str(wl_key)[:12]}]"
        j_r.update_metadata({"_category": category})
        refine_jobs.append(j_r)

    # -------------------------------------------------------------------------
    # 3) Select D‑optimal basis (round‑robin)
    chains_payload = [
        {
            "wl_key": r.output["wl_key"],
            "checkpoint_hash": r.output["checkpoint_hash"],
            "samples": r.output["selected"],
        }
        for r in refine_jobs
    ]
    j_select = select_d_optimal_basis(
        ce_key=j_wl.output["ce_key"],
        prototype=ce_spec.prototype,
        prototype_params=dict(ce_spec.prototype_params),
        supercell_diag=tuple(ce_spec.supercell_diag),
        replace_element=ce_spec.replace_element,
        endpoints=[canonical_counts(e) for e in ensure_wl_spec.endpoints],
        chains=chains_payload,
        budget=int(budget),
        ridge=1e-10,
        wl_counts_map=wl_counts_map,
    )
    j_select.name = "select_d_optimal_basis"
    j_select.update_metadata({"_category": category})

    # -------------------------------------------------------------------------
    # 3b) Prepare INTENT sources + final_ce_key
    wl_policy = {
        "bin_width": float(ensure_wl_spec.wl_bin_width),
        "step_type": str(ensure_wl_spec.wl_step_type),
        "check_period": int(ensure_wl_spec.wl_check_period),
        "update_period": int(ensure_wl_spec.wl_update_period),
        "seed": int(ensure_wl_spec.wl_seed),
    }
    ensure_policy = {
        "steps_to_run": int(ensure_wl_spec.wl_steps_to_run),
        "samples_per_bin": int(ensure_wl_spec.wl_samples_per_bin),
    }
    refine_options = {
        "mode": ("all" if int(refine_n_total) == 0 else "refine"),
        "n_total": (None if int(refine_n_total) == 0 else int(refine_n_total)),
        "per_bin_cap": int(refine_per_bin_cap),
        "strategy": str(refine_strategy),
    }
    dopt_options = {
        "budget": int(budget),
        "ridge": float(1e-10),
        "tie_breaker": "bin_then_hash",
    }

    j_prep = prepare_refined_wl_sources(
        prototype=ce_spec.prototype,
        prototype_params=dict(ce_spec.prototype_params),
        supercell_diag=tuple(ce_spec.supercell_diag),
        replace_element=ce_spec.replace_element,
        basis_spec=dict(ce_spec.basis_spec),
        regularization=dict(ce_spec.regularization or {}),
        weighting=dict(ce_spec.weighting or {}),
        train_model=str(train_model),
        train_relax_cell=bool(train_relax_cell),
        train_dtype=str(train_dtype),
        base_ce_key=str(ce_key_planned),
        endpoints=[canonical_counts(e) for e in ensure_wl_spec.endpoints],
        wl_policy=wl_policy,
        ensure=ensure_policy,
        refine=refine_options,
        dopt=dopt_options,
        chosen=j_select.output["chosen"],
        refine_results=[r.output for r in refine_jobs],
        algo_version="refined-wl-dopt-v2",
    )
    j_prep.name = "prepare_refined_wl_sources"
    j_prep.update_metadata({"_category": category})

    # -------------------------------------------------------------------------
    # 4) Relax (parallel), grouped per composition
    j_relax = relax_selected_from_wl(
        ce_key=j_wl.output["ce_key"],
        selected=j_select.output["chosen"],
        wl_counts_map=wl_counts_map,
        model=train_model,
        relax_cell=train_relax_cell,
        dtype=train_dtype,
        category=category,
    )
    j_relax.name = "relax_selected_from_wl"
    j_relax.update_metadata({"_category": category})

    # -------------------------------------------------------------------------
    # 5) Fetch → Train → Store
    j_fetch: Job = fetch_training_set_multi(
        groups=j_relax.output["groups"],
        prototype=ce_spec.prototype,
        prototype_params=dict(ce_spec.prototype_params),
        supercell_diag=tuple(ce_spec.supercell_diag),
        replace_element=ce_spec.replace_element,
        model=train_model,
        relax_cell=train_relax_cell,
        dtype=train_dtype,
        ce_key_for_rebuild=j_wl.output["ce_key"],
    )  # type: ignore[assignment]
    j_fetch.name = "fetch_training_set_multi"
    j_fetch.update_metadata({"_category": category})

    j_train: Job = train_ce(
        structures=j_fetch.output["structures"],
        energies=j_fetch.output["energies"],
        prototype=ce_spec.prototype,
        prototype_params=dict(ce_spec.prototype_params),
        supercell_diag=tuple(ce_spec.supercell_diag),
        replace_element=ce_spec.replace_element,
        basis_spec=dict(ce_spec.basis_spec),
        regularization=dict(ce_spec.regularization or {}),
        cv_seed=int(ce_spec.default_seed),
        weighting=dict(ce_spec.weighting or {}),
    )  # type: ignore[assignment]
    j_train.name = "train_ce"
    j_train.update_metadata({"_category": category})

    j_store: Job = store_ce_model(
        ce_key=j_prep.output["final_ce_key"],
        system={
            "prototype": ce_spec.prototype,
            "prototype_params": dict(ce_spec.prototype_params),
            "supercell_diag": list(ce_spec.supercell_diag),
            "replace_element": ce_spec.replace_element,
        },
        sampling={
            "algo_version": "refined-wl-dopt-v2",
            "sources": j_prep.output["sources_intent"],
            "provenance": j_prep.output["provenance"],
        },
        engine={"model": train_model, "relax_cell": train_relax_cell, "dtype": train_dtype},
        hyperparams={
            "basis_spec": dict(ce_spec.basis_spec),
            "regularization": dict(ce_spec.regularization or {}),
            "weighting": dict(ce_spec.weighting or {}),
        },
        train_refs=j_fetch.output["train_refs"],
        dataset_hash=j_fetch.output["dataset_hash"],
        payload=j_train.output["payload"],
        stats=j_train.output["stats"],
        design_metrics=j_train.output["design_metrics"],
    )  # type: ignore[assignment]
    j_store.name = "store_ce_model"
    j_store.update_metadata({"_category": category})

    # -------------------------------------------------------------------------
    stage1 = Flow([j_wl], name="stage1: ensure_wl")
    stage2 = Flow(refine_jobs, name="stage2: refine (parallel)") if refine_jobs else Flow([], name="stage2: refine")
    stage3 = Flow([j_select], name="stage3: select basis")
    stage3b = Flow([j_prep], name="stage3b: prepare sources (intent)")
    stage4 = Flow([j_relax], name="stage4: relax (parallel subflow)")
    stage5 = Flow([j_fetch, j_train, j_store], name="stage5: final CE")

    flow = Flow(
        [stage1, stage2, stage3, stage3b, stage4, stage5],
        name="Ensure CE from refined WL",
        order=JobOrder.AUTO,
    )

    out = {
        "initial_ce_key": j_wl.output["ce_key"],
        "final_ce_key": j_prep.output["final_ce_key"],
        "wl_chunks": j_wl.output["wl_chunks"],
        "refines": [
            {
                "wl_key": r.output["wl_key"],
                "checkpoint_hash": r.output["checkpoint_hash"],
                "refine_key": r.output["refine_key"],
            }
            for r in refine_jobs
        ],
        "selection_seed_size": j_select.output["seed_size"],
        "selection_budget": budget,
    }
    return Response(replace=flow, output=out)
