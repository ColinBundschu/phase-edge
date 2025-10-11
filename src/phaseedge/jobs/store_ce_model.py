from typing import Any, Mapping, Sequence, TypedDict, cast
import numpy as np

from jobflow.core.job import job

from smol.cofe import ClusterExpansion
from smol.moca.ensemble import Ensemble

from phaseedge.storage.store import lookup_unique
from phaseedge.utils.keys import normalize_sources


# -------------------------
# Types
# -------------------------

class CEModelDoc(TypedDict, total=True):
    kind: str  # "CEModelDoc"
    ce_key: str
    prototype: str
    prototype_params: Mapping[str, Any]
    supercell_diag: tuple[int, int, int]
    algo_version: str
    sources: Sequence[Mapping[str, Any]]  # e.g., training set fetch params
    model: str
    relax_cell: bool
    basis_spec: Mapping[str, Any]
    regularization: Mapping[str, Any]
    weighting: Mapping[str, Any]
    payload: Mapping[str, Any]
    dataset_key: str
    stats: Mapping[str, Any]           # in_sample / five_fold_cv / by_composition
    design_metrics: Mapping[str, Any]  # design diagnostics for X
    success: bool


# -------------------------
# Helpers
# -------------------------

def _payload_to_dict(payload: Any) -> Mapping[str, Any]:
    """
    Accept either a Mapping, a smol ClusterExpansion, or anything else.
    Prefer a proper dict via .as_dict(); fall back to repr().
    """
    if isinstance(payload, dict):
        return payload
    as_dict = getattr(payload, "as_dict", None)
    if callable(as_dict):
        try:
            d = as_dict()
            if isinstance(d, dict):
                return cast(Mapping[str, Any], d)
        except Exception:
            pass
    return {"repr": repr(payload)}


# -------------------------
# Lookups (outputs store)
# -------------------------

def lookup_ce_by_key(ce_key: str) -> CEModelDoc | None:
    """
    Fetch a CEModelDoc by ce_key from Jobflow's outputs store (rehydrated).
    """
    criteria={"output.kind": "CEModelDoc", "output.ce_key": ce_key}
    result = lookup_unique(criteria=criteria)
    return cast(CEModelDoc, result) if result is not None else None


def rehydrate_ensemble_by_ce_key(ce_key: str) -> Ensemble:
    """
    Build a SMOL Ensemble from a stored CEModelDoc in outputs.
    """
    doc = lookup_ce_by_key(ce_key)
    if not doc:
        raise RuntimeError(f"No CE found for ce_key={ce_key}")

    payload = doc["payload"]
    ce = ClusterExpansion.from_dict(payload)

    sx, sy, sz = (int(x) for x in doc["supercell_diag"])
    sc_matrix = np.diag([sx, sy, sz])
    return Ensemble.from_cluster_expansion(ce, supercell_matrix=sc_matrix)


# -------------------------
# Writer (job return)
# -------------------------

# Offload large fields to the additional store named "data" (as configured in jobflow.yaml).
# You already identified train_refs as large; add "design_metrics" here if it grows big.
@job
def store_ce_model(
    *,
    ce_key: str,

    prototype: str,
    prototype_params: Mapping[str, Any],
    supercell_diag: tuple[int, int, int],

    algo_version: str,
    sources: Sequence[Mapping[str, Any]],

    model: str,
    relax_cell: bool,

    basis_spec: Mapping[str, Any],
    regularization: Mapping[str, Any],
    weighting: Mapping[str, Any],

    dataset_key: str,
    payload: Any,          # may be a dict or a ClusterExpansion object
    stats: Mapping[str, Any],
    design_metrics: Mapping[str, Any],
) -> CEModelDoc:
    """
    Idempotently persist a trained CE (mixture-friendly) by returning a CEModelDoc.
    Jobflow + FireWorks will store it in the "outputs" collection (docs_store),
    and offload configured fields to GridFS (additional store) automatically.
    """
    doc: CEModelDoc = {
        "kind": "CEModelDoc",
        "ce_key": ce_key,
        "prototype": prototype,
        "prototype_params": dict(prototype_params),
        "supercell_diag": supercell_diag,
        "algo_version": algo_version,
        "sources": normalize_sources(sources),
        "model": model,
        "relax_cell": relax_cell,
        "basis_spec": dict(basis_spec),
        "regularization": dict(regularization),
        "weighting": dict(weighting),

        # Deterministic hash of the dataset you trained on
        "dataset_key": dataset_key,

        # CE payload and metrics (small in your current runs; leave in docs_store)
        "payload": _payload_to_dict(payload),
        "stats": dict(stats),
        "design_metrics": dict(design_metrics),

        "success": True,
    }
    return doc
