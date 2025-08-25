from dataclasses import dataclass
from typing import Any, Mapping, cast, Final
from monty.json import MSONable

from jobflow.core.job import Response, job, Job

from phaseedge.utils.keys import compute_ce_key, compute_set_id_counts
from phaseedge.storage.ce_store import lookup_ce_by_key
from phaseedge.orchestration.flows.ensure_snapshots import make_ensure_snapshots_flow
from phaseedge.orchestration.jobs.fetch_training_set import fetch_training_set
from phaseedge.orchestration.jobs.train_ce import train_ce
from phaseedge.orchestration.jobs.store_ce_model import store_ce_model


# --------------------------------------------------------------------------------------
# Spec for a single-composition CE ensure (now MSON-serializable)
# --------------------------------------------------------------------------------------

@dataclass(slots=True)
class CEEnsureSpec(MSONable):
    prototype: str
    prototype_params: Mapping[str, Any]
    supercell_diag: tuple[int, int, int]
    replace_element: str

    # --- composition / sampling (EXACT COUNTS)
    counts: Mapping[str, int]
    seed: int
    K: int  # exact number of snapshots; indices are 0..K-1

    # --- relax/engine identity (for training energies)
    model: str
    relax_cell: bool
    dtype: str

    # --- CE hyperparameters (all knobs that distinguish models)
    basis_spec: Mapping[str, Any]
    regularization: Mapping[str, Any] | None = None
    extra_hyperparams: Mapping[str, Any] | None = None

    # --- scheduling
    category: str = "gpu"  # FireWorks category tag

    # --- MSON hooks ---
    def as_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "prototype": self.prototype,
            "prototype_params": dict(self.prototype_params),
            "supercell_diag": list(self.supercell_diag),
            "replace_element": self.replace_element,
            "counts": {str(k): int(v) for k, v in dict(self.counts).items()},
            "seed": int(self.seed),
            "K": int(self.K),
            "model": self.model,
            "relax_cell": bool(self.relax_cell),
            "dtype": self.dtype,
            "basis_spec": dict(self.basis_spec),
            # store empty dicts instead of None for stability
            "regularization": dict(self.regularization or {}),
            "extra_hyperparams": dict(self.extra_hyperparams or {}),
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CEEnsureSpec":  # type: ignore[override]
        return cls(
            prototype=str(d["prototype"]),
            prototype_params=dict(d.get("prototype_params", {})),
            supercell_diag=tuple(d["supercell_diag"]),  # type: ignore[arg-type]
            replace_element=str(d["replace_element"]),
            counts={str(k): int(v) for k, v in dict(d.get("counts", {})).items()},
            seed=int(d["seed"]),
            K=int(d["K"]),
            model=str(d["model"]),
            relax_cell=bool(d["relax_cell"]),
            dtype=str(d["dtype"]),
            basis_spec=dict(d.get("basis_spec", {})),
            regularization=dict(d.get("regularization", {})),
            extra_hyperparams=dict(d.get("extra_hyperparams", {})),
            category=str(d.get("category", "gpu")),
        )


# --------------------------------------------------------------------------------------
# Decision job: ensure CE (idempotent)
# --------------------------------------------------------------------------------------

@job
def check_or_schedule_ce(spec: CEEnsureSpec) -> Mapping[str, Any]:
    """
    Idempotent CE training:
      - Compute ce_key (counts-based, exact membership).
      - If a CE already exists for ce_key, return it.
      - Otherwise REPLACE this job with:
         [ensure_snapshots(K)] -> fetch_training_set -> train_ce -> store_ce_model
    """
    proto_name = cast(str, spec.prototype)
    proto_params = spec.prototype_params

    # 1) exact membership indices
    indices: list[int] = [int(i) for i in range(int(spec.K))]
    algo: Final[str] = "randgen-2-counts-1"

    # 2) Deterministic set_id (same definition used by random configs)
    set_id: str = compute_set_id_counts(
        prototype=proto_name,
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
        replace_element=spec.replace_element,
        counts=dict(spec.counts),
        seed=int(spec.seed),
        algo_version=algo,
    )

    # 3) Compute CE key (single source of truth for idempotency)
    ce_key: str = compute_ce_key(
        prototype=proto_name,
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
        replace_element=spec.replace_element,
        counts=dict(spec.counts),
        seed=int(spec.seed),
        indices=indices,
        algo_version=algo,
        model=spec.model,
        relax_cell=spec.relax_cell,
        dtype=spec.dtype,
        basis_spec=spec.basis_spec,
        regularization=spec.regularization or {},
        extra_hyperparams=spec.extra_hyperparams or {},
    )

    # 4) Cache check: if CE exists, short-circuit
    existing = lookup_ce_by_key(ce_key)
    if existing:
        return existing

    # 5) Not found: ensure exact snapshots (barriered) and wire remaining jobs.
    f_ensure, j_gather = make_ensure_snapshots_flow(
        prototype=proto_name,
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
        replace_element=spec.replace_element,
        counts=dict(spec.counts),
        seed=int(spec.seed),
        indices=indices,
        model=spec.model,
        relax_cell=spec.relax_cell,
        dtype=spec.dtype,
        category=spec.category,
    )

    # Job B: fetch training set for EXACT occ_keys; fail hard if any are missing.
    j_fetch: Job = cast(
        Job,
        fetch_training_set(
            set_id=j_gather.output["set_id"],
            occ_keys=j_gather.output["occ_keys"],
            prototype=proto_name,
            prototype_params=proto_params,
            supercell_diag=spec.supercell_diag,
            replace_element=spec.replace_element,
            counts=dict(spec.counts),
            seed=int(spec.seed),
            indices=indices,
            model=spec.model,
            relax_cell=spec.relax_cell,
            dtype=spec.dtype,
        ),
    )
    j_fetch.name = "fetch_training_set"
    j_fetch.metadata = {**(j_fetch.metadata or {}), "_category": spec.category}

    # Job C: train CE (pure job)
    j_train: Job = cast(
        Job,
        train_ce(
            structures=j_fetch.output["structures"],
            energies=j_fetch.output["energies"],
            prototype=proto_name,
            prototype_params=proto_params,
            supercell_diag=spec.supercell_diag,
            replace_element=spec.replace_element,
            basis_spec=spec.basis_spec,
            regularization=spec.regularization or {},
            extra_hyperparams=spec.extra_hyperparams or {},
        ),
    )
    j_train.name = "train_ce"
    j_train.metadata = {**(j_train.metadata or {}), "_category": spec.category}

    # Job D: store CE model (parallel-safe insert; returns the stored doc)
    j_store: Job = cast(
        Job,
        store_ce_model(
            ce_key=ce_key,
            system={
                "prototype": proto_name,
                "prototype_params": proto_params,
                "supercell_diag": list(spec.supercell_diag),
                "replace_element": spec.replace_element,
            },
            sampling={
                "counts": dict(spec.counts),
                "seed": int(spec.seed),
                "algo_version": algo,
                "indices": indices,
            },
            engine={
                "model": spec.model,
                "relax_cell": spec.relax_cell,
                "dtype": spec.dtype,
            },
            hyperparams={
                "basis_spec": spec.basis_spec,
                "regularization": spec.regularization or {},
                "extra": spec.extra_hyperparams or {},
            },
            train_refs=j_fetch.output["train_refs"],
            dataset_hash=j_fetch.output["dataset_hash"],
            payload=j_train.output["payload"],
            stats=j_train.output["stats"],
        ),
    )
    j_store.name = "store_ce_model"
    j_store.metadata = {**(j_store.metadata or {}), "_category": spec.category}

    # Replace this decision job with the full chain.
    return Response(replace=[f_ensure, j_fetch, j_train, j_store])
