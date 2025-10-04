from typing import Any

from jobflow.core.job import Response, job
from jobflow.core.flow import Flow

from phaseedge.schemas.ensure_ce_from_mixtures_spec import EnsureCEFromMixturesSpec
from phaseedge.schemas.mixture import sublattices_from_mixtures
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.jobs.ensure_dataset_compositions import ensure_dataset_compositions
from phaseedge.jobs.train_ce import train_ce
from phaseedge.jobs.store_ce_model import store_ce_model


@job
def ensure_ce_from_mixtures(spec: EnsureCEFromMixturesSpec) -> Any:
    """
    Idempotent CE training over compositions:
      - Canonicalize mixtures and compute ce_key (sources-aware).
      - If a CE already exists for ce_key, return it.
      - Otherwise replace this job with a Flow:
          ensure_dataset_compositions -> fetch_training_set_multi -> train_ce -> store_ce_model
    """

    if lookup_ce_by_key(spec.ce_key):
        raise RuntimeError(f"CE already exists for ce_key: {spec.ce_key}")


    # 2) Ensure snapshots for all mixtures (each subflow barriers via emit job)
    f_ensure_all = ensure_dataset_compositions(
        prototype=spec.prototype,
        prototype_params=spec.prototype_params,
        supercell_diag=spec.supercell_diag,
        mixtures=spec.mixtures,
        model=spec.model,
        relax_cell=spec.relax_cell,
        category=spec.category,
    )
    f_ensure_all.name = "ensure_dataset_compositions"
    f_ensure_all.update_metadata({"_category": spec.category})

    # 4) Train CE (pooled); pass cv_seed for deterministic folds
    sublattices = sublattices_from_mixtures(spec.mixtures)
    j_train = train_ce(
        dataset_key=f_ensure_all.output["dataset_key"],
        prototype=spec.prototype,
        prototype_params=spec.prototype_params,
        supercell_diag=spec.supercell_diag,
        sublattices=sublattices,
        basis_spec=spec.basis_spec,
        regularization=spec.regularization,
        cv_seed=spec.seed,
        weighting=spec.weighting,
    )
    j_train.name = "train_ce"
    j_train.update_metadata({"_category": spec.category})

    # 5) Store CE model (returns stored doc)
    j_store = store_ce_model(
        ce_key=spec.ce_key,
        prototype=spec.prototype,
        prototype_params=spec.prototype_params,
        supercell_diag=spec.supercell_diag,
        algo_version=spec.algo,
        sources=[spec.source],
        model=spec.model,
        relax_cell=spec.relax_cell,
        basis_spec=spec.basis_spec,
        regularization=spec.regularization,
        weighting=spec.weighting,
        dataset_key=f_ensure_all.output["dataset_key"],
        payload=j_train.output["payload"],
        stats=j_train.output["stats"],
        design_metrics=j_train.output["design_metrics"],
    )
    j_store.name = "store_ce_model"
    j_store.update_metadata({"_category": spec.category})

    # Compose a single Flow so Response.replace is well-typed
    end_to_end = Flow([f_ensure_all, j_train, j_store], name="Ensure CE (compositions, barriered)")

    # Replace this decision job with the full chain; alias our output to the stored doc
    return Response(replace=end_to_end, output=j_store.output)
