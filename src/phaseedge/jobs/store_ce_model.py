from typing import Any, Mapping, Sequence, TypedDict, cast
import os
import numpy as np

from monty.serialization import loadfn
from maggma.stores import MongoStore, GridFSStore
from jobflow.core.store import JobStore
from jobflow.core.job import job

from smol.cofe import ClusterExpansion
from smol.moca.ensemble import Ensemble

from phaseedge.jobs.train_ce import CETrainRef, dataset_hash
from phaseedge.science.prototypes import PrototypeName
from phaseedge.utils.keys import normalize_sources


# -------------------------
# Types
# -------------------------

class CEModelDoc(TypedDict, total=True):
    kind: str  # "CEModelDoc"
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


def _build_jobstore() -> JobStore:
    """
    Build and connect a JobStore from a jobflow.yaml file.
    """
    path = os.environ.get("JOBFLOW_CONFIG_FILE")
    if not path:
        raise RuntimeError("JOBFLOW_CONFIG_FILE env var is not set.")
    cfg = loadfn(path)["JOB_STORE"]

    def mk(conf: Mapping[str, Any]):
        t = conf.get("type", "MongoStore")
        params = {k: v for k, v in conf.items() if k != "type"}
        if t == "MongoStore":
            return MongoStore(**params)
        if t == "GridFSStore":
            return GridFSStore(**params)
        raise ValueError(f"Unsupported store type: {t!r}")

    js = JobStore(
        docs_store=mk(cfg["docs_store"]),
        additional_stores={name: mk(s) for name, s in cfg.get("additional_stores", {}).items()},
    )
    js.docs_store.connect()
    for st in js.additional_stores.values():
        st.connect()
    return js


# -------------------------
# Lookups (outputs store)
# -------------------------

def lookup_ce_by_key(ce_key: str) -> CEModelDoc | None:
    """
    Fetch a CEModelDoc by ce_key from Jobflow's outputs store (rehydrated).

    Returns the inner payload dict (what your job returned), not the wrapper doc.
    """
    js = _build_jobstore()
    rows = list(js.query(
        criteria={"output.kind": "CEModelDoc", "output.ce_key": ce_key},
        load=True,
    ))
    if not rows:
        return None
    # There should only ever be one document per ce_key
    [doc] = rows
    # Your deployment stores the job return under "output"
    return cast(CEModelDoc, doc["output"])


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
@job(data=["train_refs"])
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

    train_refs: Sequence[CETrainRef],
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
        "dtype": dtype,
        "basis_spec": dict(basis_spec),
        "regularization": dict(regularization),
        "weighting": dict(weighting),

        # BIG field (offloaded via @job(data=[...]))
        "train_refs": [dict(r) for r in train_refs],

        # Deterministic hash of the dataset you trained on
        "dataset_hash": dataset_hash(train_refs),

        # CE payload and metrics (small in your current runs; leave in docs_store)
        "payload": _payload_to_dict(payload),
        "stats": dict(stats),
        "design_metrics": dict(design_metrics),

        "success": True,
    }
    return doc
