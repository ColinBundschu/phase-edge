"""
Modified ensure_ce_from_refined_wl job to support idempotent shortcutting.

This version computes the deterministic base CE key and the final refined intent
CE key up-front using the same logic as the original code. It then checks
whether a CE model with the final key already exists in the backing store via
``lookup_ce_by_key``. If such a model is found, the job returns the existing
document immediately instead of launching the full workflow. This makes the
workflow idempotent and avoids recomputing an identical CE.
"""

from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

from jobflow.core.flow import Flow, JobOrder
from jobflow.core.job import job, Job, Response
from monty.json import MSONable

from phaseedge.jobs.ensure_ce import CEEnsureMixturesSpec, ensure_ce
from phaseedge.jobs.ensure_wl_samples import ensure_wl_samples
from phaseedge.jobs.refine_wl_block import RefineWLSpec, refine_wl_block
from phaseedge.jobs.select_d_optimal_basis import select_d_optimal_basis
from phaseedge.jobs.relax_selected_from_wl import relax_selected_from_wl
from phaseedge.jobs.fetch_training_set_multi import fetch_training_set_multi
from phaseedge.jobs.train_ce import train_ce
from phaseedge.jobs.store_ce_model import store_ce_model
from phaseedge.jobs.prepare_refined_wl_sources import prepare_refined_wl_sources
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.schemas.mixture import composition_counts_from_map, counts_sig, sorted_composition_maps, sublattices_from_mixtures
from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.utils.keys import compute_ce_key, compute_wl_key


@dataclass(frozen=True, slots=True)
class EnsureCEFromRefinedWLSpec(MSONable):
    ce_spec: CEEnsureMixturesSpec
    endpoints: tuple[dict[str, dict[str, int]], ...]

    wl_bin_width: float
    wl_steps_to_run: int
    wl_samples_per_bin: int

    wl_step_type: str = "swap"
    wl_check_period: int = 5_000
    wl_update_period: int = 1
    wl_seed: int = 0

    refine_n_total: int = 25
    refine_per_bin_cap: int = 5
    refine_strategy: Literal["energy_spread", "energy_stratified", "hash_round_robin"] = "energy_spread"
    train_model: str = "MACE-MPA-0"
    train_relax_cell: bool = False
    train_dtype: str = "float64"
    budget: int = 64

    # Single category for *everything* (wrapper, CE subflow, WL jobs)
    category: str = "gpu"

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "ce_spec": self.ce_spec.as_dict(),
            "endpoints": self.endpoints,
            "wl_bin_width": self.wl_bin_width,
            "wl_steps_to_run": self.wl_steps_to_run,
            "wl_samples_per_bin": self.wl_samples_per_bin,
            "wl_step_type": self.wl_step_type,
            "wl_check_period": self.wl_check_period,
            "wl_update_period": self.wl_update_period,
            "wl_seed": self.wl_seed,
            "refine_n_total": self.refine_n_total,
            "refine_per_bin_cap": self.refine_per_bin_cap,
            "refine_strategy": self.refine_strategy,
            "train_model": self.train_model,
            "train_relax_cell": self.train_relax_cell,
            "train_dtype": self.train_dtype,
            "budget": self.budget,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EnsureCEFromRefinedWLSpec":
        ce_spec = d["ce_spec"]
        if not isinstance(ce_spec, CEEnsureMixturesSpec):
            ce_spec = CEEnsureMixturesSpec.from_dict(ce_spec)
        return cls(
            ce_spec=ce_spec,
            endpoints=sorted_composition_maps([{
                str(sublat): {str(k): int(v) for k, v in counts.items()}
                for sublat, counts in e.items()
            } for e in d["endpoints"]]),
            wl_bin_width=float(d["wl_bin_width"]),
            wl_steps_to_run=int(d["wl_steps_to_run"]),
            wl_samples_per_bin=int(d["wl_samples_per_bin"]),
            wl_step_type=str(d["wl_step_type"]),
            wl_check_period=int(d["wl_check_period"]),
            wl_update_period=int(d["wl_update_period"]),
            wl_seed=int(d["wl_seed"]),
            refine_n_total=int(d["refine_n_total"]),
            refine_per_bin_cap=int(d["refine_per_bin_cap"]),
            refine_strategy=cast(Literal["energy_spread", "energy_stratified", "hash_round_robin"], d["refine_strategy"]),
            train_model=str(d["train_model"]),
            train_relax_cell=bool(d["train_relax_cell"]),
            train_dtype=str(d["train_dtype"]),
            budget=int(d["budget"]),
            category=str(d.get("category", "gpu")),
        )
    
    @property
    def sublattice_labels(self) -> tuple[str, ...]:
        all_labels = [tuple(sorted(mixture.composition_map.keys())) for mixture in self.ce_spec.mixtures]
        # If all the labels are not identical, raise an error
        first = all_labels[0]
        for labels in all_labels[1:]:
            if labels != first:
                raise ValueError("All mixtures must have the same sublattice labels.")
        return first
    
    @property
    def refine_mode(self) -> Literal["all", "refine"]:
        return "all" if self.refine_n_total == 0 else "refine"
    
    @property
    def refine_total(self) -> int | None:
        return None if self.refine_n_total == 0 else self.refine_n_total

    @property
    def wl_key_composition_pairs(self) -> tuple[tuple[str, dict[str, int]], ...]:
        ce_key = self.ce_spec.ce_key
        pairs = []
        seen_sigs = {counts_sig(composition_counts_from_map(ep)) for ep in self.endpoints}
        for mixture in self.ce_spec.mixtures:
            composition_counts = composition_counts_from_map(mixture.composition_map)
            sig = counts_sig(composition_counts)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)

            wl_key = compute_wl_key(
                ce_key=ce_key,
                bin_width=self.wl_bin_width,
                step_type=self.wl_step_type,
                composition_counts=composition_counts,
                check_period=self.wl_check_period,
                update_period=self.wl_update_period,
                seed=self.wl_seed,
                algo_version="wl-grid-v1",
            )
            pairs.append((wl_key, composition_counts))
        return tuple(sorted(pairs))
    
    @property
    def final_ce_key(self) -> str:
        return compute_ce_key(
            prototype=self.ce_spec.prototype,
            prototype_params=dict(self.ce_spec.prototype_params),
            supercell_diag=self.ce_spec.supercell_diag,
            sources=[self.source],
            model=self.train_model,
            relax_cell=self.train_relax_cell,
            dtype=self.train_dtype,
            basis_spec=self.ce_spec.basis_spec,
            regularization=self.ce_spec.regularization,
            algo_version="refined-wl-dopt-v2",
            weighting=self.ce_spec.weighting,
        )

    @property
    def source(self):
        wl_policy_for_key = {
            "bin_width": self.wl_bin_width,
            "step_type": self.wl_step_type,
            "check_period": self.wl_check_period,
            "update_period": self.wl_update_period,
            "seed": self.wl_seed,
        }
        ensure_policy_for_key = {
            "steps_to_run": self.wl_steps_to_run,
            "samples_per_bin": self.wl_samples_per_bin,
        }
        refine_options_for_key = {
            "mode": self.refine_mode,
            "n_total": self.refine_total,
            "per_bin_cap": self.refine_per_bin_cap,
            "strategy": self.refine_strategy,
        }
        dopt_options_for_key = {
            "budget": self.budget,
            "ridge": float(1e-10),
            "tie_breaker": "bin_then_hash",
        }
        source = {
            "type": "wl_refined_intent",
            "base_ce_key": self.ce_spec.ce_key,
            "endpoints": self.endpoints,
            "wl_policy": wl_policy_for_key,
            "ensure": ensure_policy_for_key,
            "refine": refine_options_for_key,
            "dopt": dopt_options_for_key,
            "versions": {
                "refine": "refine-wl-v1",
                "dopt": "dopt-rr-sm-v1",
                "sampler": "wl-grid-v1",
            },
        }
        
        return source


@job
def ensure_ce_from_refined_wl(
    *,
    ensure_wl_spec: EnsureCEFromRefinedWLSpec,
) -> Mapping[str, Any] | Response:
    """Ensure a CE using refined WL data, idempotently."""
    # -------------------------------------------------------------------------
    # Early exit keying
    final_ce_key = ensure_wl_spec.final_ce_key
    existing_ce = lookup_ce_by_key(final_ce_key)
    if existing_ce is not None:
        return existing_ce

    # -------------------------------------------------------------------------
    # 1) Ensure CE + WL
    ce_key = ensure_wl_spec.ce_spec.ce_key

    j_ce: Job = ensure_ce(ensure_wl_spec.ce_spec)
    j_ce.name = "ensure_ce"
    j_ce.update_metadata({"_category": ensure_wl_spec.category})

    wl_jobs: list[Job | Flow] = []
    wl_chunks: list[Mapping[str, Any]] = []  # minimal, safe fields only
    
    wl_composition_map = {wl_key: comp for wl_key, comp in ensure_wl_spec.wl_key_composition_pairs}
    for wl_key, composition_counts in wl_composition_map.items():
        sig = counts_sig(composition_counts)
        run_spec = WLSamplerSpec(
            wl_key=wl_key,
            ce_key=ce_key,  # pass the resolved string; barrier enforced by linear flow below
            bin_width=ensure_wl_spec.wl_bin_width,
            steps=ensure_wl_spec.wl_steps_to_run,
            sublattice_labels=ensure_wl_spec.sublattice_labels,
            composition_counts=composition_counts,
            step_type=ensure_wl_spec.wl_step_type,
            check_period=ensure_wl_spec.wl_check_period,
            update_period=ensure_wl_spec.wl_update_period,
            seed=ensure_wl_spec.wl_seed,
            samples_per_bin=ensure_wl_spec.wl_samples_per_bin,
        )

        j_wl: Job = ensure_wl_samples(run_spec)
        j_wl.name = f"ensure_wl_samples::{wl_key[:12]}::{sig}"
        j_wl.update_metadata({"_category": ensure_wl_spec.category, "wl_key": wl_key})
        wl_jobs.append(j_wl)

        # IMPORTANT: expose only the fields we NEED as references.
        # Avoid referencing child outputs like "samples_per_bin" that may not be present
        # in add_wl_chunk job output.
        wl_chunks.append({
            "counts": sig,       # static
            "wl_key": wl_key,    # static
            "hash": j_wl.output["hash"],        # reference (exists for both found+new)
            # if you ever want to include samples_per_bin, set it to the POLICY value:
            # "samples_per_bin": int(spec.wl_samples_per_bin),
            # likewise for chunk_size:
            # "chunk_size": int(spec.wl_steps_to_run),
        })

    # WL jobs run in parallel after CE completes (linear barrier at outer level)
    wl_flow_inner = Flow(wl_jobs, name="WL jobs (parallel)")
    ensure_wl_from_ce_flow = Flow([j_ce, wl_flow_inner], name="Ensure WL samples from CE", order=JobOrder.LINEAR)

    # -------------------------------------------------------------------------
    # Plan WL chains
    

    # -------------------------------------------------------------------------
    # 2) Refine per WL checkpoint
    # refine_jobs: list[Job | Flow] = []
    # for i in range(len(wl_composition_map)):
    #     chunk = j_wl.output["wl_chunks"][i]
    #     wl_key = str(chunk["wl_key"])
    #     mode = "all" if int(refine_n_total) == 0 else "refine"
    #     r_spec = RefineWLSpec(
    #         wl_key=wl_key,
    #         mode=mode,
    #         n_total=None if int(refine_n_total) == 0 else int(refine_n_total),
    #         per_bin_cap=int(refine_per_bin_cap),
    #         strategy=refine_strategy,
    #     )
    #     j_r = refine_wl_block(r_spec, checkpoint_hash=str(chunk["hash"]))
    #     j_r.name = f"refine_wl_block[{wl_key[:12]}]"
    #     j_r.update_metadata({"_category": category})
    #     refine_jobs.append(j_r)

    refine_jobs: list[Job | Flow] = []
    for chunk in wl_chunks:
        wl_key_ref = chunk["wl_key"]      # OutputReference -> will resolve to str at runtime
        ck_hash_ref = chunk["hash"]       # OutputReference -> resolves to str at runtime

        # Note: wl_counts_map keys must be plain strings; wl_key_ref resolves to str at runtime,
        # but we need a stable key *now*. Use the same key the downstream code will use:
        # pass wl_counts_map later together with wl_key values coming from refine outputs.
        # If downstream expects wl_counts_map[resolved_wl_key], it’s fine to populate after refine.
        # Otherwise, if you want it now, compute planned wl_keys deterministically and join there.

        r_spec = RefineWLSpec(
            wl_key=wl_key_ref,                   # <-- DO NOT str() — keep the reference
            mode=ensure_wl_spec.refine_mode,
            n_total=ensure_wl_spec.refine_total,
            per_bin_cap=ensure_wl_spec.refine_per_bin_cap,
            strategy=ensure_wl_spec.refine_strategy,
        )
        j_r = refine_wl_block(r_spec, checkpoint_hash=ck_hash_ref)  # keep reference
        j_r.name = f"refine_wl_block[{wl_key_ref[:12]}]"            # this is okay; Jobflow formats refs
        j_r.update_metadata({"_category": ensure_wl_spec.category})
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
        ce_key=ce_key,
        prototype=ensure_wl_spec.ce_spec.prototype,
        prototype_params=ensure_wl_spec.ce_spec.prototype_params,
        supercell_diag=ensure_wl_spec.ce_spec.supercell_diag,
        endpoints=ensure_wl_spec.endpoints,
        chains=chains_payload,
        budget=ensure_wl_spec.budget,
        ridge=1e-10,
        wl_counts_map=wl_composition_map,
    )
    j_select.name = "select_d_optimal_basis"
    j_select.update_metadata({"_category": ensure_wl_spec.category})

    # -------------------------------------------------------------------------
    # 3b) Prepare INTENT sources + final_ce_key
    # wl_policy = {
    #     "bin_width": float(ensure_wl_spec.wl_bin_width),
    #     "step_type": str(ensure_wl_spec.wl_step_type),
    #     "check_period": int(ensure_wl_spec.wl_check_period),
    #     "update_period": int(ensure_wl_spec.wl_update_period),
    #     "seed": int(ensure_wl_spec.wl_seed),
    # }
    # ensure_policy = {
    #     "steps_to_run": int(ensure_wl_spec.wl_steps_to_run),
    #     "samples_per_bin": int(ensure_wl_spec.wl_samples_per_bin),
    # }
    # refine_options = {
    #     "mode": ("all" if int(refine_n_total) == 0 else "refine"),
    #     "n_total": (None if int(refine_n_total) == 0 else int(refine_n_total)),
    #     "per_bin_cap": int(refine_per_bin_cap),
    #     "strategy": str(refine_strategy),
    # }
    # dopt_options = {
    #     "budget": int(budget),
    #     "ridge": float(1e-10),
    #     "tie_breaker": "bin_then_hash",
    # }

    # j_prep = prepare_refined_wl_sources(
    #     prototype=ce_spec.prototype,
    #     prototype_params=dict(ce_spec.prototype_params),
    #     supercell_diag=tuple(ce_spec.supercell_diag),
    #     basis_spec=dict(ce_spec.basis_spec),
    #     regularization=dict(ce_spec.regularization or {}),
    #     weighting=dict(ce_spec.weighting or {}),
    #     train_model=str(train_model),
    #     train_relax_cell=bool(train_relax_cell),
    #     train_dtype=str(train_dtype),
    #     base_ce_key=ce_spec.ce_key,
    #     endpoints=ensure_wl_spec.endpoints,
    #     wl_policy=wl_policy,
    #     ensure=ensure_policy,
    #     refine=refine_options,
    #     dopt=dopt_options,
    #     chosen=j_select.output["chosen"],
    #     refine_results=[r.output for r in refine_jobs],
    #     algo_version="refined-wl-dopt-v2",
    # )
    # j_prep.name = "prepare_refined_wl_sources"
    # j_prep.update_metadata({"_category": category})

    # -------------------------------------------------------------------------
    # 4) Relax (parallel), grouped per composition
    j_relax = relax_selected_from_wl(
        ce_key=ce_key,
        selected=j_select.output["chosen"],
        wl_counts_map=wl_composition_map,
        model=ensure_wl_spec.train_model,
        relax_cell=ensure_wl_spec.train_relax_cell,
        dtype=ensure_wl_spec.train_dtype,
        category=ensure_wl_spec.category,
    )
    j_relax.name = "relax_selected_from_wl"
    j_relax.update_metadata({"_category": ensure_wl_spec.category})

    # -------------------------------------------------------------------------
    # 5) Fetch → Train → Store
    j_fetch: Job = fetch_training_set_multi(
        groups=j_relax.output["groups"],
        prototype=ensure_wl_spec.ce_spec.prototype,
        prototype_params=dict(ensure_wl_spec.ce_spec.prototype_params),
        supercell_diag=tuple(ensure_wl_spec.ce_spec.supercell_diag),
        model=ensure_wl_spec.train_model,
        relax_cell=ensure_wl_spec.train_relax_cell,
        dtype=ensure_wl_spec.train_dtype,
        ce_key_for_rebuild=ce_key,
    )
    j_fetch.name = "fetch_training_set_multi"
    j_fetch.update_metadata({"_category": ensure_wl_spec.category})

    j_train: Job = train_ce(
        structures=j_fetch.output["structures"],
        energies=j_fetch.output["energies"],
        prototype=ensure_wl_spec.ce_spec.prototype,
        prototype_params=ensure_wl_spec.ce_spec.prototype_params,
        supercell_diag=ensure_wl_spec.ce_spec.supercell_diag,
        sublattices=sublattices_from_mixtures(ensure_wl_spec.ce_spec.mixtures),
        basis_spec=ensure_wl_spec.ce_spec.basis_spec,
        regularization=ensure_wl_spec.ce_spec.regularization,
        cv_seed=ensure_wl_spec.ce_spec.seed,
        weighting=ensure_wl_spec.ce_spec.weighting,
    )
    j_train.name = "train_ce"
    j_train.update_metadata({"_category": ensure_wl_spec.category})

    j_store: Job = store_ce_model(
        ce_key=final_ce_key,
        prototype=ensure_wl_spec.ce_spec.prototype,
        prototype_params=ensure_wl_spec.ce_spec.prototype_params,
        supercell_diag=ensure_wl_spec.ce_spec.supercell_diag,
        algo_version="refined-wl-dopt-v2",
        sources=[ensure_wl_spec.source],
        model=ensure_wl_spec.train_model,
        relax_cell=ensure_wl_spec.train_relax_cell,
        dtype=ensure_wl_spec.train_dtype,
        basis_spec=ensure_wl_spec.ce_spec.basis_spec,
        regularization=ensure_wl_spec.ce_spec.regularization,
        weighting=ensure_wl_spec.ce_spec.weighting,
        train_refs=j_fetch.output["train_refs"],
        dataset_hash=j_fetch.output["dataset_hash"],
        payload=j_train.output["payload"],
        stats=j_train.output["stats"],
        design_metrics=j_train.output["design_metrics"],
    )  # type: ignore[assignment]
    j_store.name = "store_ce_model"
    j_store.update_metadata({"_category": ensure_wl_spec.category})

    # -------------------------------------------------------------------------
    stage1 = Flow([ensure_wl_from_ce_flow], name="stage1: ensure_wl")
    stage2 = Flow(refine_jobs, name="stage2: refine (parallel)") if refine_jobs else Flow([], name="stage2: refine")
    stage3 = Flow([j_select], name="stage3: select basis")
    # stage3b = Flow([j_prep], name="stage3b: prepare sources (intent)")
    stage4 = Flow([j_relax], name="stage4: relax (parallel subflow)")
    stage5 = Flow([j_fetch, j_train, j_store], name="stage5: final CE")

    flow = Flow(
        [stage1, stage2, stage3, stage4, stage5],
        name="Ensure CE from refined WL",
        order=JobOrder.AUTO,
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
        "selection_budget": ensure_wl_spec.budget,
    }
    return Response(replace=flow, output=out)
