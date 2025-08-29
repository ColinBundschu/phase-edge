from dataclasses import dataclass, asdict
from typing import Any, Literal, Mapping, TypedDict, Union, Dict

from jobflow.core.job import job, Job
from monty.json import MSONable

from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.sampling.wang_landau import run_wl
from phaseedge.storage.wl_store import insert_wl_result, to_doc
from phaseedge.storage import store
from phaseedge.utils.keys import compute_wl_key


class WLEnsureOutcome(TypedDict, total=False):
    _id: str
    wl_key: str
    status: Literal["cached", "inserted"]


def _wl_coll():
    return store.db_rw()["wang_landau"]


@dataclass(frozen=True)
class WLEnsureSpec(MSONable):
    # Identity / MC contract
    ce_key: str
    bin_width: float
    steps: int
    composition_counts: Mapping[str, int]
    step_type: str = "swap"
    check_period: int = 5_000
    update_period: int = 1
    seed: int = 0

    # Routing (FireWorks category, queue tags, etc.)
    category: str = "gpu"

    # ---- MSONable ----
    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Standard MSON header so Monty can round-trip
        d.update({
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        })
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WLEnsureSpec":
        # Accept dicts with or without MSON header
        # Remove MSON header keys if present
        d = {k: v for k, v in d.items() if not k.startswith("@")}
        return cls(**d)  # type: ignore[arg-type]


@job
def ensure_wl(spec: Union[WLEnsureSpec, Mapping[str, Any]]) -> WLEnsureOutcome:
    """
    Idempotent WL ensure job. If wl_key exists -> 'cached', else run and insert.
    Accepts either a WLEnsureSpec or a plain dict (post-deserialization).
    """
    if not isinstance(spec, WLEnsureSpec):
        spec = WLEnsureSpec.from_dict(spec)  # type: ignore[arg-type]

    wl_key = compute_wl_key(
        ce_key=spec.ce_key,
        bin_width=spec.bin_width,
        steps=spec.steps,
        step_type=spec.step_type,
        composition_counts=spec.composition_counts,
        check_period=spec.check_period,
        update_period=spec.update_period,
        seed=spec.seed,
        grid_anchor=0.0,
        algo_version="wl-grid-v1",
    )

    existing = _wl_coll().find_one({"keys.wl": wl_key}, {"_id": 1})
    if existing:
        return WLEnsureOutcome(_id=str(existing["_id"]), wl_key=wl_key, status="cached")

    # Build private runner spec
    run_spec = WLSamplerSpec(
        ce_key=spec.ce_key,
        bin_width=spec.bin_width,
        steps=spec.steps,
        step_type=spec.step_type,
        composition_counts=spec.composition_counts,
        check_period=spec.check_period,
        update_period=spec.update_period,
        seed=spec.seed,
    )

    result = run_wl(run_spec)
    doc = to_doc(result)
    doc["keys"] = {"wl": wl_key, "ce": spec.ce_key}
    oid = insert_wl_result(doc)

    return WLEnsureOutcome(_id=oid, wl_key=wl_key, status="inserted")


def check_or_schedule_wl(spec: WLEnsureSpec) -> Job:
    """
    Return a JobFlow Job to run on a FireWorker (mirrors CE's pattern).
    Caller is responsible for wrapping in a Flow and submitting to FireWorks.
    """
    j = ensure_wl(spec)
    j.name = "ensure_wl"
    # Belt & suspenders: stash routing/category in metadata for FireWorks filters
    j.metadata = {**(j.metadata or {}), "_category": spec.category}
    return j
