from typing import Any, Mapping, Sequence, TypedDict, cast

from jobflow.core.job import job
from phaseedge.science.prototypes import PrototypeName
from phaseedge.storage import store
from monty.json import jsanitize

from phaseedge.utils.keys import normalize_sources

class CEModelDoc(TypedDict, total=True):
    ce_key: str
    prototype: PrototypeName
    prototype_params: Mapping[str, Any]
    supercell_diag: tuple[int, int, int]
    algo_version: str
    sources: Sequence[Mapping[str, Any]]  # e.g., training set fetch params
    model: str
    relax_cell: bool
    dtype: str
    basis_spec: Mapping[str, Any]
    regularization: Mapping[str, Any]
    weighting: Mapping[str, Any]
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
    if isinstance(payload, dict):
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
    coll = store.db_rw()["ce_models"]
    coll.create_index("ce_key", unique=True, background=True)
    return coll


def lookup_ce_by_key(ce_key: str) -> CEModelDoc | None:
    """Fetch a CE model by its unique key."""
    doc = _ce_coll().find_one({"ce_key": ce_key})
    return cast(CEModelDoc | None, doc)


@job
def store_ce_model(
    *,
    ce_key: str,
    
    prototype: PrototypeName,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],

    algo_version: str,
    sources: Sequence[Mapping[str, Any]],

    model: str,
    relax_cell: bool,
    dtype: str,

    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any],
    weighting: Mapping[str, Any],

    train_refs: Sequence[Mapping[str, Any]],
    dataset_hash: str,
    payload: Any,          # may be a dict or a ClusterExpansion object
    stats: Mapping[str, Any],
    design_metrics: Mapping[str, Any],
) -> CEModelDoc:
    """
    Idempotently persist a trained CE (mixture-friendly).
    If a doc with ce_key exists, we overwrite fields (upsert semantics).
    Returns the stored document from DB.
    """
    doc: CEModelDoc = {
        "ce_key": ce_key,
        "prototype": prototype,
        "prototype_params": dict(prototype_params),
        "supercell_diag": supercell_diag,
        "algo_version": algo_version,
        "sources": normalize_sources(sources),
        "model": model,
        "relax_cell": relax_cell,
        "dtype": dtype,
        "basis_spec": dict(basis_spec),
        "regularization": dict(regularization),
        "weighting": dict(weighting),
        "train_refs": [dict(r) for r in train_refs],
        "dataset_hash": str(dataset_hash),
        "payload": _payload_to_dict(payload),
        "stats": dict(stats),
        "design_metrics": dict(design_metrics),
        "success": True,
    }

    # ---- Sanitize everything to Mongo-safe primitives ----
    doc_sanitized = jsanitize(doc, strict=True)

    coll = _ce_coll()
    coll.update_one({"ce_key": ce_key}, {"$set": doc_sanitized}, upsert=True)
    stored = coll.find_one({"ce_key": ce_key}) or doc_sanitized
    return cast(CEModelDoc, stored)
