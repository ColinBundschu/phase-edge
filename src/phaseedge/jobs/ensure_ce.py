from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast, Final

from monty.json import MSONable
from jobflow.core.job import Response, job, Job
from jobflow.core.flow import Flow

from phaseedge.science.prototypes import PrototypeName
from phaseedge.storage.ce_store import lookup_ce_by_key
from phaseedge.jobs.ensure_snapshots_compositions import make_ensure_snapshots_compositions
from phaseedge.jobs.fetch_training_set_multi import fetch_training_set_multi
from phaseedge.jobs.train_ce import train_ce
from phaseedge.jobs.store_ce_model import store_ce_model
from phaseedge.utils.keys import compute_ce_key


# --------------------------------------------------------------------------------------
# Spec for a compositions-based CE ensure (MSON-serializable)
# (Retains existing class name for minimal churn; field names unchanged)
# --------------------------------------------------------------------------------------

@dataclass(slots=True)
class CEEnsureMixtureSpec(MSONable):
    prototype: str
    prototype_params: Mapping[str, Any]
    supercell_diag: tuple[int, int, int]
    replace_element: str

    # compositions list: each element has counts, K, and optional seed
    mixture: Sequence[Mapping[str, Any]]
    default_seed: int  # used when an element doesn't specify its own seed

    # relax/engine identity (for training energies)
    model: str
    relax_cell: bool
    dtype: str

    # CE hyperparameters (all knobs that distinguish models)
    basis_spec: Mapping[str, Any]
    regularization: Mapping[str, Any] | None = None
    weighting: Mapping[str, Any] | None = None

    # scheduling
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
            "mixture": [
                {
                    "counts": {str(k): int(v) for k, v in dict(elem.get("counts", {})).items()},
                    "K": int(elem.get("K", 0)),
                    **({"seed": int(elem["seed"])} if "seed" in elem else {}),
                }
                for elem in self.mixture
            ],
            "default_seed": int(self.default_seed),
            "model": self.model,
            "relax_cell": bool(self.relax_cell),
            "dtype": self.dtype,
            "basis_spec": dict(self.basis_spec),
            "regularization": dict(self.regularization or {}),
            "weighting": dict(self.weighting or {}),
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CEEnsureMixtureSpec":  # type: ignore[override]
        raw_mix = list(d.get("mixture", []))
        mixture: list[dict[str, Any]] = []
        for elem in raw_mix:
            counts = {str(k): int(v) for k, v in dict(elem.get("counts", {})).items()}
            k_val = int(elem.get("K", 0))
            out = {"counts": counts, "K": k_val}
            if "seed" in elem:
                out["seed"] = int(elem["seed"])
            mixture.append(out)

        return cls(
            prototype=str(d["prototype"]),
            prototype_params=dict(d.get("prototype_params", {})),
            supercell_diag=tuple(d["supercell_diag"]),  # type: ignore[arg-type]
            replace_element=str(d["replace_element"]),
            mixture=mixture,
            default_seed=int(d["default_seed"]),
            model=str(d["model"]),
            relax_cell=bool(d["relax_cell"]),
            dtype=str(d["dtype"]),
            basis_spec=dict(d.get("basis_spec", {})),
            regularization=dict(d.get("regularization", {})),
            weighting=dict(d.get("weighting", {})) or None,
            category=str(d.get("category", "gpu")),
        )


# --------------------------------------------------------------------------------------
# Decision job: ensure CE over compositions (idempotent)
# --------------------------------------------------------------------------------------

@job
def ensure_ce(spec: CEEnsureMixtureSpec) -> Any:
    """
    Idempotent CE training over compositions:
      - Canonicalize the composition list and compute ce_key (sources-aware).
      - If a CE already exists for ce_key, return it.
      - Otherwise replace this job with a single Flow:
          ensure_snapshots_compositions -> fetch_training_set_multi -> train_ce -> store_ce_model
    """
    proto_name = cast(str, spec.prototype)
    proto_params = dict(spec.prototype_params)  # Mapping -> dict for typed helpers
    algo: Final[str] = "randgen-3-comp-1"

    # Canonicalize composition elements: counts dict[str,int], K:int >= 1, seed:int (fallback to default)
    canon_elems: list[dict[str, Any]] = []
    for elem in spec.mixture:
        counts = {str(k): int(v) for k, v in dict(elem.get("counts", {})).items()}
        K = int(elem.get("K", 0))
        if not counts or K <= 0:
            raise ValueError(f"Invalid composition element: counts={counts}, K={K}")
        seed_eff = int(elem.get("seed", spec.default_seed))
        canon_elems.append({"counts": counts, "K": K, "seed": seed_eff})

    # Unified sources block (only composition source for now)
    sources = [
        {
            "type": "composition",
            "elements": canon_elems,
        }
    ]

    # 1) Compute CE key (single source of truth for idempotency)
    ce_key: str = compute_ce_key(
        prototype=proto_name,
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
        replace_element=spec.replace_element,
        sources=sources,
        model=spec.model,
        relax_cell=spec.relax_cell,
        dtype=spec.dtype,
        basis_spec=dict(spec.basis_spec),
        regularization=dict(spec.regularization or {}),
        algo_version=algo,
        weighting=dict(spec.weighting or {}),
    )

    # 2) Cache check: if CE exists, short-circuit
    existing = lookup_ce_by_key(ce_key)
    if existing:
        return existing

    # 3) Ensure snapshots for all composition elements (barriered per element)
    f_ensure_all, j_groups = make_ensure_snapshots_compositions(
        prototype=cast(PrototypeName, proto_name),
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
        replace_element=spec.replace_element,
        mixture=canon_elems,  # function still uses 'mixture' param name internally
        model=spec.model,
        relax_cell=spec.relax_cell,
        dtype=spec.dtype,
        default_seed=int(spec.default_seed),
        category=spec.category,
    )

    # 4) Fetch training set across all groups (hard fail if any missing)
    j_fetch: Job = cast(
        Job,
        fetch_training_set_multi(
            groups=j_groups.output["groups"],
            prototype=proto_name,
            prototype_params=proto_params,
            supercell_diag=spec.supercell_diag,
            replace_element=spec.replace_element,
            model=spec.model,
            relax_cell=spec.relax_cell,
            dtype=spec.dtype,
            ce_key_for_rebuild=ce_key,
        ),
    )
    j_fetch.name = "fetch_training_set_multi"
    j_fetch.update_metadata({"_category": spec.category})

    # 5) Train CE (pooled); pass cv_seed for deterministic folds
    j_train: Job = cast(
        Job,
        train_ce(
            structures=j_fetch.output["structures"],
            energies=j_fetch.output["energies"],
            prototype=proto_name,
            prototype_params=proto_params,
            supercell_diag=spec.supercell_diag,
            replace_element=spec.replace_element,
            basis_spec=dict(spec.basis_spec),
            regularization=dict(spec.regularization or {}),
            cv_seed=int(spec.default_seed),
            weighting=dict(spec.weighting or {}),
        ),
    )
    j_train.name = "train_ce"
    j_train.update_metadata({"_category": spec.category})

    # 6) Store CE model (returns stored doc)
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
                "algo_version": algo,
                "sources": [
                    {
                        "type": "composition",
                        "elements": canon_elems,
                    }
                ],
            },
            engine={
                "model": spec.model,
                "relax_cell": spec.relax_cell,
                "dtype": spec.dtype,
            },
            hyperparams={
                "basis_spec": dict(spec.basis_spec),
                "regularization": dict(spec.regularization or {}),
                "weighting": dict(spec.weighting or {}),
            },
            train_refs=j_fetch.output["train_refs"],
            dataset_hash=j_fetch.output["dataset_hash"],
            payload=j_train.output["payload"],
            stats=j_train.output["stats"],
            design_metrics=j_train.output["design_metrics"],
        ),
    )
    j_store.name = "store_ce_model"
    j_store.update_metadata({"_category": spec.category})

    # Compose a single Flow so Response.replace is well-typed
    end_to_end = Flow([f_ensure_all, j_fetch, j_train, j_store], name="Ensure CE (compositions, barriered)")

    # Replace this decision job with the full chain; alias our output to the stored doc
    return Response(replace=end_to_end, output=j_store.output)
