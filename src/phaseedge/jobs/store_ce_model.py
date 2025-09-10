from typing import Any, Mapping, Sequence, TypedDict, cast
import json

from jobflow.core.job import job
from phaseedge.storage import store
from monty.json import MontyEncoder


class _StoredCE(TypedDict, total=False):
    ce_key: str
    system: Mapping[str, Any]
    sampling: Mapping[str, Any]
    engine: Mapping[str, Any]
    hyperparams: Mapping[str, Any]
    train_refs: Sequence[Mapping[str, Any]]
    dataset_hash: str
    payload: Mapping[str, Any]
    stats: Mapping[str, Any]           # in_sample / five_fold_cv / by_composition
    design_metrics: Mapping[str, Any]  # design diagnostics for X
    success: bool


def _payload_to_dict(payload: Any) -> Mapping[str, Any]:
    """
    Accept either a Mapping, a smol ClusterExpansion, or anything else.
    Prefer a proper dict via .as_dict(); fall back to repr().
    """
    if isinstance(payload, Mapping):
        return payload
    # smol ClusterExpansion and most monty objects implement as_dict()
    as_dict = getattr(payload, "as_dict", None)
    if callable(as_dict):
        try:
            d = as_dict()
            if isinstance(d, dict):
                return cast(Mapping[str, Any], d)
        except Exception:
            pass
    # last resort: store a readable representation
    return {"repr": repr(payload)}


def _ce_coll():
    return store.db_rw()["ce_models"]


@job
def store_ce_model(
    *,
    ce_key: str,
    system: Mapping[str, Any],
    sampling: Mapping[str, Any],
    engine: Mapping[str, Any],
    hyperparams: Mapping[str, Any],
    train_refs: Sequence[Mapping[str, Any]],
    dataset_hash: str,
    payload: Any,          # may be a dict or a ClusterExpansion object
    stats: Mapping[str, Any],
    design_metrics: Mapping[str, Any],
) -> _StoredCE:
    """
    Idempotently persist a trained CE (mixture-friendly).
    If a doc with ce_key exists, we overwrite fields (upsert semantics).
    Returns the stored document from DB.
    """
    doc: _StoredCE = {
        "ce_key": ce_key,
        "system": dict(system),
        "sampling": dict(sampling),
        "engine": dict(engine),
        "hyperparams": dict(hyperparams),
        "train_refs": list(train_refs),
        "dataset_hash": str(dataset_hash),
        "payload": _payload_to_dict(payload),
        "stats": dict(stats),
        "design_metrics": dict(design_metrics),
        "success": True,
    }

    # Clean, canonical conversion of *all* nested MSONable/NumPy/etc to pure Python
    doc = cast(_StoredCE, json.loads(json.dumps(doc, cls=MontyEncoder)))

    coll = _ce_coll()
    coll.update_one({"ce_key": ce_key}, {"$set": doc}, upsert=True)
    stored = coll.find_one({"ce_key": ce_key}) or doc
    return cast(_StoredCE, stored)
