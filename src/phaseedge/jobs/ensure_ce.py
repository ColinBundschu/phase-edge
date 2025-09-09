from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast, Final

from monty.json import MSONable
from jobflow.core.job import Response, job, Job
from jobflow.core.flow import Flow

from phaseedge.science.prototypes import PrototypeName
from phaseedge.storage.ce_store import lookup_ce_by_key
from phaseedge.jobs.ensure_snapshots_compositions import (
    make_ensure_snapshots_compositions,
    MixtureElement,
)
from phaseedge.jobs.fetch_training_set_multi import fetch_training_set_multi
from phaseedge.jobs.train_ce import train_ce
from phaseedge.jobs.store_ce_model import store_ce_model
from phaseedge.schemas.sublattice import SublatticeSpec
from phaseedge.utils.keys import (
    compute_ce_key, CEKeySpec, SublatticeMixtureElement, _canon_sublattice_specs
)


@dataclass(slots=True)
class CEEnsureMixtureSpec(MSONable):
    prototype: str
    prototype_params: Mapping[str, Any]
    supercell_diag: tuple[int, int, int]
    replace_elements: Sequence[str]           # explicit for downstream validation/logging

    mixture: Sequence[MixtureElement]         # jobflow payload
    default_seed: int

    model: str
    relax_cell: bool
    dtype: str

    basis_spec: Mapping[str, Any]
    regularization: Mapping[str, Any] | None = None
    extra_hyperparams: Mapping[str, Any] | None = None
    weighting: Mapping[str, Any] | None = None

    category: str = "gpu"

    def as_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "prototype": self.prototype,
            "prototype_params": dict(self.prototype_params),
            "supercell_diag": list(self.supercell_diag),
            "replace_elements": list(self.replace_elements),
            "mixture": [
                {
                    "sublattices": [sl.as_dict() for sl in elem["sublattices"]],
                    "K": int(elem["K"]),
                    "seed": int(elem["seed"]),
                }
                for elem in self.mixture
            ],
            "default_seed": int(self.default_seed),
            "model": self.model,
            "relax_cell": bool(self.relax_cell),
            "dtype": self.dtype,
            "basis_spec": dict(self.basis_spec),
            "regularization": dict(self.regularization or {}),
            "extra_hyperparams": dict(self.extra_hyperparams or {}),
            "weighting": dict(self.weighting or {}),
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CEEnsureMixtureSpec":
        raw_mix = list(d.get("mixture", []))
        mixture: list[MixtureElement] = []
        for elem in raw_mix:
            subls = [SublatticeSpec.from_dict(sd) for sd in elem["sublattices"]]
            mixture.append({"sublattices": subls, "K": int(elem["K"]), "seed": int(elem["seed"])})
        return cls(
            prototype=str(d["prototype"]),
            prototype_params=dict(d.get("prototype_params", {})),
            supercell_diag=tuple(d["supercell_diag"]),
            replace_elements=list(d.get("replace_elements", [])),
            mixture=mixture,
            default_seed=int(d["default_seed"]),
            model=str(d["model"]),
            relax_cell=bool(d["relax_cell"]),
            dtype=str(d["dtype"]),
            basis_spec=dict(d.get("basis_spec", {})),
            regularization=dict(d.get("regularization", {})) or None,
            extra_hyperparams=dict(d.get("extra_hyperparams", {})) or None,
            weighting=dict(d.get("weighting", {})) or None,
            category=str(d.get("category", "gpu")),
        )


@job
def ensure_ce(spec: CEEnsureMixtureSpec) -> Any:
    """
    Idempotent CE training over multi-sublattice compositions with a single source of truth:
    compute_ce_key(spec=CEKeySpec(...)).
    """
    proto_name = cast(str, spec.prototype)
    proto_params = dict(spec.prototype_params)
    algo: Final[str] = "randgen-4-sublcomp-1"

    # Build the strongly-typed mixtures for keying
    mix_for_key: list[SublatticeMixtureElement] = [
        SublatticeMixtureElement(
            sublattices=elem["sublattices"],
            K=int(elem["K"]),
            seed=int(elem["seed"]),
        )
        for elem in spec.mixture
    ]

    # CE identity (single interface)
    ce_key: str = compute_ce_key(
        spec=CEKeySpec(
            prototype=proto_name,
            prototype_params=proto_params,
            supercell_diag=spec.supercell_diag,
            mixtures=mix_for_key,
            model=spec.model,
            relax_cell=spec.relax_cell,
            dtype=spec.dtype,
            basis_spec=dict(spec.basis_spec),
            regularization=dict(spec.regularization or {}),
            extra_hyperparams=dict(spec.extra_hyperparams or {}),
            weighting=dict(spec.weighting or {}),
            algo_version=algo,
        )
    )

    # Cache check
    existing = lookup_ce_by_key(ce_key)
    if existing:
        return existing

    # Generate & relax snapshots (barriered per mixture)
    f_ensure_all, j_groups = make_ensure_snapshots_compositions(
        prototype=cast(PrototypeName, proto_name),
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
        mixture=spec.mixture,
        model=spec.model,
        relax_cell=spec.relax_cell,
        dtype=spec.dtype,
        default_seed=int(spec.default_seed),
        category=spec.category,
    )

    # Fetch training set
    j_fetch = fetch_training_set_multi(
        groups=j_groups.output["groups"],
        prototype=proto_name,
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
        # replace_elements=list(spec.replace_elements),
        model=spec.model,
        relax_cell=spec.relax_cell,
        dtype=spec.dtype,
        ce_key_for_rebuild=ce_key,
    )
    j_fetch.name = "fetch_training_set_multi"
    j_fetch.update_metadata({"_category": spec.category})

    # Train CE
    j_train = train_ce(
        structures=j_fetch.output["structures"],
        energies=j_fetch.output["energies"],
        prototype=proto_name,
        prototype_params=proto_params,
        supercell_diag=spec.supercell_diag,
        replace_elements=list(spec.replace_elements),
        basis_spec=dict(spec.basis_spec),
        regularization=dict(spec.regularization or {}),
        extra_hyperparams=dict(spec.extra_hyperparams or {}),
        cv_seed=int(spec.default_seed),
        weighting=dict(spec.weighting or {}),
    )
    j_train.name = "train_ce"
    j_train.update_metadata({"_category": spec.category})

    # Persist (store canonical mixtures right next to the key)
    canon_elements = [
        {"sublattices": _canon_sublattice_specs(e["sublattices"]), "K": int(e["K"]), "seed": int(e["seed"])}
        for e in spec.mixture
    ]

    j_store: Job = cast(
        Job,
        store_ce_model(
            ce_key=ce_key,
            system={
                "prototype": proto_name,
                "prototype_params": proto_params,
                "supercell_diag": list(spec.supercell_diag),
                "replace_elements": list(spec.replace_elements),
            },
            sampling={
                "algo_version": algo,
                "type": "sublattice_composition",
                "elements": canon_elements,
            },
            engine={
                "model": spec.model,
                "relax_cell": spec.relax_cell,
                "dtype": spec.dtype,
            },
            hyperparams={
                "basis_spec": dict(spec.basis_spec),
                "regularization": dict(spec.regularization or {}),
                "extra": dict(spec.extra_hyperparams or {}),
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

    end_to_end = Flow([f_ensure_all, j_fetch, j_train, j_store], name="Ensure CE (sublattice compositions)")
    return Response(replace=end_to_end, output=j_store.output)
