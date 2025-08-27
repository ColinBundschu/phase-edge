from dataclasses import dataclass
from typing import Any, Mapping, TypedDict

from jobflow import job
from monty.json import MSONable

from phaseedge.storage.ce_store import lookup_ce_by_key
from phaseedge.science.ce import validate_ce_composition
from phaseedge.utils.keys import compute_wl_run_key


class WangLandauResult(TypedDict):
    run_key: str
    ce_key: str
    composition: Mapping[str, float]
    dos: list[Any]
    samples: list[Any]


@dataclass
class WangLandauSpec(MSONable):
    ce_key: str
    composition: dict[str, float]
    steps: int
    bin_size: float
    n_samples: int
    seed: int | None = None
    algo_version: str = "wl-1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "ce_key": self.ce_key,
            "composition": self.composition,
            "steps": self.steps,
            "bin_size": self.bin_size,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "algo_version": self.algo_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WangLandauSpec":
        return cls(
            ce_key=str(d["ce_key"]),
            composition={k: float(v) for k, v in (d.get("composition") or {}).items()},
            steps=int(d["steps"]),
            bin_size=float(d["bin_size"]),
            n_samples=int(d["n_samples"]),
            seed=d.get("seed"),
            algo_version=str(d.get("algo_version", "wl-1")),
        )

    def run_key(self) -> str:
        return compute_wl_run_key(
            ce_key=self.ce_key,
            composition=self.composition,
            steps=self.steps,
            bin_size=self.bin_size,
            n_samples=self.n_samples,
            seed=self.seed,
            algo_version=self.algo_version,
        )


@job
def make_wang_landau_run(spec: WangLandauSpec) -> WangLandauResult:
    """Load CE, validate composition, and return placeholder WL results."""
    ce_doc = lookup_ce_by_key(spec.ce_key)
    if ce_doc is None:
        raise ValueError(f"Unknown CE key: {spec.ce_key}")

    validate_ce_composition(ce_doc, spec.composition)
    run_key = spec.run_key()

    # Placeholder: real WL algorithm will produce DOS and samples.
    return {
        "run_key": run_key,
        "ce_key": spec.ce_key,
        "composition": spec.composition,
        "dos": [],
        "samples": [],
    }
