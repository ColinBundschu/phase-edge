from typing import Any, Mapping, Optional, Literal, TypedDict

from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.sampling.wang_landau import run_wl
from phaseedge.storage.wl_store import insert_wl_result, to_doc
from phaseedge.storage import store
from phaseedge.utils.keys import compute_wl_key


__all__ = ["ensure_wang_landau"]


class EnsureWLResult(TypedDict, total=False):
    _id: str
    wl_key: str
    status: Literal["cached", "inserted"]
    grid: dict[str, Any]
    meta: dict[str, Any]


def _wl_coll():
    return store.db_rw()["wang_landau"]


def _derive_ensemble_kind(
    composition: Optional[Mapping[str, float]],
    chemical_potentials: Optional[Mapping[str, float]],
) -> Literal["canonical", "semi_grand"]:
    if composition is not None and chemical_potentials is not None:
        raise ValueError("Provide either composition (canonical) OR chemical_potentials (semi_grand), not both.")
    if chemical_potentials is not None:
        return "semi_grand"
    return "canonical"


def ensure_wang_landau(
    *,
    ce_key: str,
    bin_width: float,
    steps: int,
    composition: Optional[Mapping[str, float]] = None,
    chemical_potentials: Optional[Mapping[str, float]] = None,
    step_type: str = "swap",
    check_period: int = 5_000,
    update_period: int = 1,
    seed: int = 0,
    # Internal knobs (not part of the public key / contract):
    _pilot_samples: int = 256,
    _sigma_multiplier: float = 50.0,
) -> EnsureWLResult:
    """
    Idempotent entrypoint to compute (or fetch) a Wang–Landau density of states.

    Public contract for idempotency (hashed into wl_key):
      - ce_key
      - ensemble spec (composition OR chemical_potentials)
      - grid contract: (anchor=0.0, bin_width)
      - MC contract: (step_type, steps, check_period, update_period, seed)

    Implementation details (V1 "hack"): a small CE-based pilot derives a fixed window
    centered at median(H) with radius 50*std(H), snapped to a zero-anchored grid.
    These details are *not* part of the key and can be changed later without breaking
    idempotency, as long as the public contract above remains the same.
    """
    ensemble_kind = _derive_ensemble_kind(composition, chemical_potentials)

    wl_key = compute_wl_key(
        ce_key=ce_key,
        bin_width=bin_width,
        steps=steps,
        step_type=step_type,
        ensemble=ensemble_kind,
        composition=composition,
        chemical_potentials=chemical_potentials,
        check_period=check_period,
        update_period=update_period,
        seed=seed,
        grid_anchor=0.0,
        algo_version="wl-grid-v1",
    )

    # Short-circuit if present
    existing = _wl_coll().find_one({"keys.wl": wl_key}, {"_id": 1, "grid": 1, "meta": 1})
    if existing:
        return EnsureWLResult(
            _id=str(existing["_id"]),
            wl_key=wl_key,
            status="cached",
            grid=dict(existing.get("grid", {})),
            meta=dict(existing.get("meta", {})),
        )

    # Build a private spec for the runner (internal knobs deliberately excluded from wl_key)
    spec = WLSamplerSpec(
        ce_key=ce_key,
        ensemble=ensemble_kind,
        composition=composition,
        chemical_potentials=chemical_potentials,
        bin_width=bin_width,
        steps=steps,
        check_period=check_period,
        update_period=update_period,
        seed=seed,
        pilot_samples=_pilot_samples,
        sigma_multiplier=_sigma_multiplier,
    )

    # Execute the sampler (V1 uses smol kernel + fixed derived window under the hood)
    result = run_wl(spec)

    # Persist with wl_key attached; keep CE key for convenience
    doc = to_doc(result)
    doc["keys"] = {"wl": wl_key, "ce": ce_key}
    oid = insert_wl_result(doc)

    return EnsureWLResult(
        _id=oid,
        wl_key=wl_key,
        status="inserted",
        grid=dict(doc.get("grid", {})),
        meta=dict(doc.get("meta", {})),
    )
