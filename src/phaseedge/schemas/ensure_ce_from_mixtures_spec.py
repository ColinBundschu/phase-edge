from dataclasses import dataclass
import dataclasses
from typing import Any, Mapping

from monty.json import MSONable

from phaseedge.schemas.calc_spec import CalcSpec
from phaseedge.schemas.mixture import Mixture
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.utils.keys import compute_ce_key


@dataclass(frozen=True, slots=True)
class EnsureCEFromMixturesSpec(MSONable):
    _: dataclasses.KW_ONLY
    prototype_spec: PrototypeSpec
    supercell_diag: tuple[int, int, int]
    mixtures: tuple[Mixture, ...]
    seed: int

    calc_spec: CalcSpec
    category: str

    basis_spec: Mapping[str, Any]
    regularization: Mapping[str, Any] | None = None
    weighting: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # canonicalize/cast
        sc = tuple(int(x) for x in self.supercell_diag)
        if len(sc) != 3:
            raise ValueError("supercell_diag must be length-3 (a, b, c).")

        mix_tuple = tuple(sorted(self.mixtures, key=lambda m: m.sort_key()))

        object.__setattr__(self, "supercell_diag", sc)
        object.__setattr__(self, "mixtures", mix_tuple)

    # Monty expects plain "dict" here; using it avoids override warnings.
    def as_dict(self) -> dict:
        d: dict[str, Any] = {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "prototype_spec": self.prototype_spec.as_dict(),
            "supercell_diag": list(self.supercell_diag),
            "mixtures": [m.as_dict() for m in self.mixtures],
            "seed": int(self.seed),
            "calc_spec": self.calc_spec.as_dict(),
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
            prototype_spec=PrototypeSpec.from_dict(d["prototype_spec"]),
            supercell_diag=(sx, sy, sz),
            mixtures=tuple(Mixture.from_dict(m) for m in d["mixtures"]),
            seed=int(d["seed"]),
            calc_spec=CalcSpec.from_dict(d["calc_spec"]),
            basis_spec=d["basis_spec"],
            regularization=d.get("regularization"),
            weighting=d.get("weighting"),
            category=d["category"],
        )

    @property
    def source(self) -> dict[str, Any]:
        return {
            "type": "composition",
            "mixtures": self.mixtures,
            "algo_version": self.algo_version,
        }

    @property
    def algo_version(self) -> str:
        return "mixture-ce-v1"

    @property
    def ce_key(self) -> str:
        return compute_ce_key(
            prototype_spec=self.prototype_spec,
            supercell_diag=self.supercell_diag,
            sources=[self.source],
            calc_spec=self.calc_spec,
            basis_spec=self.basis_spec,
            regularization=self.regularization,
            algo_version=self.algo_version,
            weighting=self.weighting,
            partial=False,
        )
