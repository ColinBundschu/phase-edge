from dataclasses import dataclass
from typing import Any, Mapping

from monty.json import MSONable

from phaseedge.schemas.mixture import Mixture
from phaseedge.science.prototypes import PrototypeName
from phaseedge.utils.keys import compute_ce_key


@dataclass(frozen=True, slots=True)
class EnsureCEFromMixturesSpec(MSONable):
    prototype: PrototypeName
    prototype_params: Mapping[str, Any]
    supercell_diag: tuple[int, int, int]
    mixtures: tuple[Mixture, ...]
    seed: int

    model: str
    relax_cell: bool

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
            "basis_spec": dict(self.basis_spec),
            "category": self.category,
        }
        if self.regularization is not None:
            d["regularization"] = dict(self.regularization)
        if self.weighting is not None:
            d["weighting"] = dict(self.weighting)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EnsureCEFromMixturesSpec":
        sx, sy, sz = (int(x) for x in d["supercell_diag"])
        return cls(
            prototype=PrototypeName(d["prototype"]),
            prototype_params=d["prototype_params"],
            supercell_diag=(sx, sy, sz),
            mixtures=tuple(Mixture.from_dict(m) for m in d["mixtures"]),
            seed=int(d["seed"]),
            model=d["model"],
            relax_cell=bool(d["relax_cell"]),
            basis_spec=d["basis_spec"],
            regularization=d.get("regularization"),
            weighting=d.get("weighting"),
            category=d["category"],
        )

    @property
    def source(self) -> dict[str, Any]:
        return {"type": "composition", "mixtures": self.mixtures}

    @property
    def algo(self) -> str:
        return "randgen-3-comp-1"

    @property
    def ce_key(self) -> str:
        return compute_ce_key(
            prototype=self.prototype,
            prototype_params=self.prototype_params,
            supercell_diag=self.supercell_diag,
            sources=[self.source],
            model=self.model,
            relax_cell=self.relax_cell,
            basis_spec=self.basis_spec,
            regularization=self.regularization,
            algo_version=self.algo,
            weighting=self.weighting,
        )
