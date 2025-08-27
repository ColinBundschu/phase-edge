from dataclasses import dataclass
from typing import Any, Mapping, TypedDict
import math
import random

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


def _basic_wl_algorithm(
    steps: int, bin_size: float, n_samples: int, seed: int | None
) -> tuple[list[float], list[dict[str, float]]]:
    """Toy Wang-Landau walk over synthetic energy bins."""

    rng = random.Random(seed)
    num_bins = 10
    g = [1.0] * num_bins
    hist = [0] * num_bins
    f = math.e
    current = rng.randrange(num_bins)
    samples: list[dict[str, float]] = []

    for step in range(steps):
        proposal = rng.randrange(num_bins)
        accept_prob = min(1.0, g[current] / g[proposal])
        if rng.random() < accept_prob:
            current = proposal

        g[current] *= f
        hist[current] += 1

        if len(samples) < n_samples:
            energy = (current + 0.5) * bin_size
            samples.append({"energy": energy, "bin": float(current)})

        if (step + 1) % 100 == 0:
            avg = sum(hist) / num_bins
            if min(hist) >= 0.8 * avg:
                hist = [0] * num_bins
                f = math.sqrt(f)
                if f < 1 + 1e-8:
                    break

    min_g = min(g)
    dos = [math.log(x / min_g) for x in g]
    return dos, samples


def run_wang_landau(spec: WangLandauSpec, ce_doc: Mapping[str, Any]) -> WangLandauResult:
    """Validate CE, run a toy WL walk, and return results."""

    validate_ce_composition(ce_doc, spec.composition)
    dos, samples = _basic_wl_algorithm(
        spec.steps, spec.bin_size, spec.n_samples, spec.seed
    )

    return {
        "run_key": spec.run_key(),
        "ce_key": spec.ce_key,
        "composition": spec.composition,
        "dos": dos,
        "samples": samples,
    }


@job
def make_wang_landau_run(spec: WangLandauSpec) -> WangLandauResult:
    """Load CE from storage and execute a Wang–Landau run."""
    ce_doc = lookup_ce_by_key(spec.ce_key)
    if ce_doc is None:
        raise ValueError(f"Unknown CE key: {spec.ce_key}")
    return run_wang_landau(spec, ce_doc)
