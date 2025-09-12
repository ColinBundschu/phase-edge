from dataclasses import dataclass
from typing import Any, Mapping, cast, Final

from monty.json import MSONable
from jobflow.core.job import Response, job, Job
from jobflow.core.flow import Flow

from phaseedge.schemas.mixture import Mixture, sublattices_from_mixtures
from phaseedge.science.prototypes import PrototypeName
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.jobs.ensure_snapshots_compositions import make_ensure_snapshots_compositions
from phaseedge.jobs.fetch_training_set_multi import fetch_training_set_multi
from phaseedge.jobs.train_ce import train_ce
from phaseedge.jobs.store_ce_model import store_ce_model
from phaseedge.utils.keys import compute_ce_key


# --------------------------------------------------------------------------------------
# Spec for a compositions-based CE ensure (MSON-serializable)
# --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CEEnsureMixturesSpec(MSONable):
    prototype: str
    prototype_params: Mapping[str, Any]
    supercell_diag: tuple[int, int, int]
    mixtures: tuple[Mixture, ...]
    seed: int

    model: str
    relax_cell: bool
    dtype: str

    basis_spec: Mapping[str, Any]
    regularization: Mapping[str, Any] | None = None
    weighting: Mapping[str, Any] | None = None

    category: str = "gpu"

    def __post_init__(self) -> None:
        # canonicalize/cast
        sc = tuple(int(x) for x in self.supercell_diag)
        if len(sc) != 3:
            raise ValueError("supercell_diag must be length-3 (a, b, c).")

        mix_tuple = tuple(sorted(self.mixtures, key=lambda m: m.sort_key()))

        object.__setattr__(self, "supercell_diag", sc)
        object.__setattr__(self, "mixtures", mix_tuple)
        object.__setattr__(self, "relax_cell", bool(self.relax_cell))

    # Monty expects plain "dict" here; using it avoids override warnings.
    def as_dict(self) -> dict:
        d: dict[str, Any] = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "prototype": self.prototype,
            "prototype_params": dict(self.prototype_params),
            "supercell_diag": list(self.supercell_diag),
            "mixtures": [m.as_dict() for m in self.mixtures],
            "seed": int(self.seed),
            "model": self.model,
            "relax_cell": self.relax_cell,
            "dtype": self.dtype,
            "basis_spec": dict(self.basis_spec),
            "category": self.category,
        }
        if self.regularization is not None:
            d["regularization"] = dict(self.regularization)
        if self.weighting is not None:
            d["weighting"] = dict(self.weighting)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CEEnsureMixturesSpec":
        sx, sy, sz = (int(x) for x in d["supercell_diag"])
        return cls(
            prototype=str(d["prototype"]),
            prototype_params=dict(d.get("prototype_params", {})),
            supercell_diag=(sx, sy, sz),
            mixtures=tuple(Mixture.from_dict(m) for m in d.get("mixtures", [])),
            seed=int(d["seed"]),
            model=str(d["model"]),
            relax_cell=bool(d["relax_cell"]),
            dtype=str(d["dtype"]),
            basis_spec=dict(d.get("basis_spec", {})),
            regularization=(dict(d["regularization"]) if "regularization" in d else None),
            weighting=(dict(d["weighting"]) if "weighting" in d else None),
            category=str(d.get("category", "gpu")),
        )

# --------------------------------------------------------------------------------------
# Decision job: ensure CE over compositions (idempotent)
# --------------------------------------------------------------------------------------

@job
def ensure_ce(spec: CEEnsureMixturesSpec) -> Any:
    """
    Idempotent CE training over compositions:
      - Canonicalize mixtures and compute ce_key (sources-aware).
      - If a CE already exists for ce_key, return it.
      - Otherwise replace this job with a Flow:
          ensure_snapshots_compositions -> fetch_training_set_multi -> train_ce -> store_ce_model
    """
    proto_name = cast(str, spec.prototype)
    proto_params = dict(spec.prototype_params)  # Mapping -> dict for typed helpers
    algo: Final[str] = "randgen-3-comp-1"

    # Unified sources block (only composition source for now)
    sources = [
        {
            "type": "composition",
            "mixtures": spec.mixtures,  # Mixture objects; normalized downstream
        }
    ]

    # 1) Compute CE key (single source of truth for idempotency)
    ce_key: str = compute_ce_key(
        prototype=proto_name,
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
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

    # 3) Ensure snapshots for all mixtures (each subflow barriers via emit job)
    f_ensure_all: Flow = make_ensure_snapshots_compositions(
        prototype=cast(PrototypeName, proto_name),
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
        mixtures=spec.mixtures,
        model=spec.model,
        relax_cell=spec.relax_cell,
        dtype=spec.dtype,
        category=spec.category,
    )

    # 4) Fetch training set across all groups (hard fail if any missing)
    j_fetch: Job = cast(
        Job,
        fetch_training_set_multi(
            groups=f_ensure_all.output["groups"],
            prototype=proto_name,
            prototype_params=proto_params,
            supercell_diag=spec.supercell_diag,
            model=spec.model,
            relax_cell=spec.relax_cell,
            dtype=spec.dtype,
            ce_key_for_rebuild=ce_key,
        ),
    )
    j_fetch.name = "fetch_training_set_multi"
    j_fetch.update_metadata({"_category": spec.category})

    # 5) Train CE (pooled); pass cv_seed for deterministic folds
    sublattices = sublattices_from_mixtures(spec.mixtures)
    j_train: Job = cast(
        Job,
        train_ce(
            structures=j_fetch.output["structures"],
            energies=j_fetch.output["energies"],
            prototype=proto_name,
            prototype_params=proto_params,
            supercell_diag=spec.supercell_diag,
            sublattices=sublattices,
            basis_spec=dict(spec.basis_spec),
            regularization=dict(spec.regularization or {}),
            cv_seed=int(spec.seed),
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
            prototype=proto_name,
            prototype_params=proto_params,
            supercell_diag=list(spec.supercell_diag),
            algo_version=algo,
            sources = [
                {
                    "type": "composition",
                    "mixtures": [
                        {"composition_map": m.composition_map, "K": m.K, "seed": m.seed}
                        for m in spec.mixtures
                    ],
                }
            ],
            model=spec.model,
            relax_cell=spec.relax_cell,
            dtype=spec.dtype,
            basis_spec=dict(spec.basis_spec),
            regularization=dict(spec.regularization or {}),
            weighting=dict(spec.weighting or {}),
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
