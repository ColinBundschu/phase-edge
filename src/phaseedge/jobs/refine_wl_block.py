from dataclasses import dataclass
from typing import Any, Mapping, Literal, TypedDict, cast

import hashlib
import json

from jobflow.core.job import job
from monty.json import MSONable

from phaseedge.storage import store
# Uses your existing selector implementation
from phaseedge.science.refine_wl import (
    RefineOptions,
    RefineStrategy,
    refine_wl_samples as _refine_wl_samples,
)


class RefinedSample(TypedDict):
    bin: int
    occ: list[int]


@dataclass(frozen=True, slots=True)
class RefineWLSpec(MSONable):
    """
    Idempotent refinement spec for a single WL checkpoint.

    Note: The checkpoint hash is passed as a TOP-LEVEL job kwarg (not inside
    this dataclass) so Jobflow will resolve any OutputReference properly.
    """

    wl_key: str

    # behavior: "refine" uses options; "all" returns every stored sample
    mode: Literal["refine", "all"] = "refine"

    # refinement options (ignored when mode == "all")
    n_total: int | None = 25
    per_bin_cap: int | None = 5
    strategy: RefineStrategy = RefineStrategy.ENERGY_SPREAD

    def as_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "wl_key": self.wl_key,
            "mode": self.mode,
            "n_total": self.n_total,
            "per_bin_cap": self.per_bin_cap,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RefineWLSpec":  # type: ignore[override]
        return cls(
            wl_key=str(d["wl_key"]),
            mode=cast(Literal["refine", "all"], d.get("mode", "refine")),
            n_total=cast(int | None, d.get("n_total", 25)),
            per_bin_cap=cast(int | None, d.get("per_bin_cap", 5)),
            strategy=RefineStrategy(d["strategy"]),
        )


def _occ_hash(occ: list[int]) -> str:
    return hashlib.sha256(bytes(int(x) & 0xFF for x in occ)).hexdigest()


def _compute_refine_key(
    *, wl_key: str, checkpoint_hash: str, mode: str, n_total: int | None, per_bin_cap: int | None, strategy: str
) -> str:
    payload = {
        "wl_key": wl_key,
        "hash": checkpoint_hash,
        "mode": mode,
        "n_total": n_total,
        "per_bin_cap": per_bin_cap,
        "strategy": strategy,
        "algo": "refine-v1" if mode == "refine" else "all-v1",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@job
def refine_wl_block(spec: RefineWLSpec, *, checkpoint_hash: str) -> Mapping[str, Any]:
    """
    Deterministically refine (or pass-through) samples from a single WL checkpoint.

    Parameters
    ----------
    spec
        Static refinement spec (wl_key + policy).
    checkpoint_hash
        The hash of the WL checkpoint to refine. This can be a Jobflow
        OutputReference and will be resolved before execution.

    Output schema:
        {
          "refine_key": <sha256 identity>,
          "wl_key": "...",
          "checkpoint_hash": "...",
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
    coll = store.db_ro()["wang_landau_ckpt"]
    ckpt = coll.find_one({"wl_key": spec.wl_key, "hash": checkpoint_hash})
    if not ckpt:
        raise RuntimeError(f"WL checkpoint not found for wl_key={spec.wl_key}, hash={checkpoint_hash}")

    bin_samples = ckpt.get("bin_samples")
    if not isinstance(bin_samples, list):
        raise RuntimeError("Checkpoint document lacks 'bin_samples' list.")

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
        options = RefineOptions(
            n_total=None if spec.n_total is None else int(spec.n_total),
            per_bin_cap=None if spec.per_bin_cap is None else int(spec.per_bin_cap),
            strategy=cast(Any, spec.strategy),
        )
        block = {"wl_key": spec.wl_key, "hash": checkpoint_hash, "bin_samples": bin_samples}
        refined = _refine_wl_samples(block, options=options)
        selected = cast(list[RefinedSample], refined["selected"])

    refine_key = _compute_refine_key(
        wl_key=spec.wl_key,
        checkpoint_hash=checkpoint_hash,
        mode=spec.mode,
        n_total=spec.n_total,
        per_bin_cap=spec.per_bin_cap,
        strategy=spec.strategy,
    )

    return {
        "refine_key": refine_key,
        "wl_key": spec.wl_key,
        "checkpoint_hash": checkpoint_hash,
        "n_selected": len(selected),
        "selected": selected,
        "policy": {
            "mode": spec.mode,
            "n_total": spec.n_total,
            "per_bin_cap": spec.per_bin_cap,
            "strategy": spec.strategy,
        },
    }
