from dataclasses import dataclass
from typing import Any, Mapping, Literal, TypedDict, cast

import hashlib
import json

from jobflow.core.job import job
from monty.json import MSONable

from phaseedge.storage.wang_landau import lookup_wl_block_by_key
from phaseedge.science.refine_wl import RefineStrategy, refine_wl_samples


class RefinedSample(TypedDict):
    bin: int
    occ: list[int]


@dataclass(frozen=True, slots=True)
class RefineWLSpec(MSONable):
    """
    Idempotent refinement spec for a single WL block.

    Note: The block hash is passed as a TOP-LEVEL job kwarg (not inside
    this dataclass) so Jobflow will resolve any OutputReference properly.
    """

    # behavior: "refine" uses options; "all" returns every stored sample
    mode: Literal["refine", "all"]

    # refinement options (ignored when mode == "all")
    n_total: int
    per_bin_cap: int | None
    strategy: RefineStrategy

    def as_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "mode": self.mode,
            "n_total": self.n_total,
            "per_bin_cap": self.per_bin_cap,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RefineWLSpec":  # type: ignore[override]
        return cls(
            mode=cast(Literal["refine", "all"], d["mode"]),
            n_total=int(d["n_total"]),
            per_bin_cap=cast(int | None, d["per_bin_cap"]),
            strategy=RefineStrategy(d["strategy"]),
        )


def _occ_hash(occ: list[int]) -> str:
    return hashlib.sha256(bytes(int(x) & 0xFF for x in occ)).hexdigest()


def _compute_refine_key(
    *, wl_key: str, wl_block_key: str, mode: str, n_total: int | None, per_bin_cap: int | None, strategy: str
) -> str:
    payload = {
        "wl_key": wl_key,
        "wl_block_key": wl_block_key,
        "mode": mode,
        "n_total": n_total,
        "per_bin_cap": per_bin_cap,
        "strategy": strategy,
        "algo": "refine-v1" if mode == "refine" else "all-v1",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@job
def refine_wl_block_samples(*, spec: RefineWLSpec, wl_block_key: str) -> Mapping[str, Any]:
    """
    Deterministically refine (or pass-through) samples from a single WL block.

    Parameters
    ----------
    spec
        Static refinement spec (wl_key + policy).
    wl_block_key
        The hash of the WL block to refine. This can be a Jobflow
        OutputReference and will be resolved before execution.

    Output schema:
        {
          "refine_key": <sha256 identity>,
          "wl_key": "...",
          "wl_block_key": "...",
          "n_selected": int,
          "selected": [{"bin": int, "occ": [int, ...]}, ...],
          "policy": {
            "mode": "refine"|"all",
            "n_total": int|null,
            "per_bin_cap": int|null,
            "strategy": "energy_spread"|"energy_stratified"|"hash_round_robin"
          }
        }
    """
    block = lookup_wl_block_by_key(wl_block_key)
    if not block:
        raise RuntimeError(f"WL block not found for wl_block_key={wl_block_key}")

    bin_samples = block.get("bin_samples")
    if not isinstance(bin_samples, list):
        raise RuntimeError("WL block document lacks 'bin_samples' list.")

    if spec.mode == "all":
        # Return all samples deterministically sorted by (bin asc, occ_hash asc)
        samples: list[RefinedSample] = []
        for rec in bin_samples:
            b = int(rec["bin"])
            occ = [int(x) for x in rec["occ"]]
            samples.append(RefinedSample(bin=b, occ=occ))
        samples.sort(key=lambda s: (int(s["bin"]), _occ_hash(s["occ"])))
        selected = samples
    else:
        selected = refine_wl_samples(
            bin_samples,
            n_total=spec.n_total,
            per_bin_cap=spec.per_bin_cap,
            strategy=spec.strategy,
        )

    refine_key = _compute_refine_key(
        wl_key=block["wl_key"],
        wl_block_key=wl_block_key,
        mode=spec.mode,
        n_total=spec.n_total,
        per_bin_cap=spec.per_bin_cap,
        strategy=spec.strategy,
    )

    return {
        "refine_key": refine_key,
        "wl_key": block["wl_key"],
        "wl_block_key": wl_block_key,
        "n_selected": len(selected),
        "selected": selected,
        "policy": {
            "mode": spec.mode,
            "n_total": spec.n_total,
            "per_bin_cap": spec.per_bin_cap,
            "strategy": spec.strategy,
        },
    }
