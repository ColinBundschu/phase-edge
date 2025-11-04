import os
from threading import Lock
from urllib.parse import quote_plus
from typing import Any, Mapping, cast

from pymongo import MongoClient
from pymongo.database import Database
from monty.serialization import loadfn
from jobflow.core.store import JobStore
from maggma.stores import MongoStore, GridFSStore

from phaseedge.schemas.calc_spec import CalcSpec, CalcType


# -------------------------
# Environment → connection
# -------------------------

def _mongo_uri(user_env: str, pass_env: str) -> str:
    """
    Compose a mongodb:// URI from env vars.

    Required env:
      - MONGO_HOST
      - MONGO_DB
      - user_env, pass_env

    Optional env:
      - MONGO_PORT (default: 27017)
    """
    host = os.environ["MONGO_HOST"]
    db = os.environ["MONGO_DB"]
    user = os.environ[user_env]
    pwd = os.environ[pass_env]

    port = os.environ.get("MONGO_PORT", "27017")

    user_q = quote_plus(user)
    pwd_q = quote_plus(pwd)

    return f"mongodb://{user_q}:{pwd_q}@{host}:{port}/{db}"


_rw_client: MongoClient | None = None
_js_lock: Lock = Lock()
_shared_jobstore: JobStore | None = None


def db_rw() -> Database:
    """Read/Write DB using admin creds (for inserts + index creation)."""
    global _rw_client
    if _rw_client is None:
        _rw_client = MongoClient(_mongo_uri("MONGO_ADMIN_USER", "MONGO_ADMIN_PASS"), tz_aware=True)
    return _rw_client[os.environ["MONGO_DB"]]


def _mk_store(conf: Mapping[str, Any]):
    """
    Construct a Maggma store from YAML config without creating per-call clients.
    We rely on Maggma's internals to manage a client per store, but since we
    cache the JobStore, those clients are created once per process.
    """
    t = conf.get("type", "MongoStore")
    params: dict[str, Any] = {k: v for k, v in conf.items() if k != "type"}

    if t == "MongoStore":
        return MongoStore(**params)
    if t == "GridFSStore":
        return GridFSStore(**params)
    raise ValueError(f"Unsupported store type: {t!r}")


def _build_jobstore_once() -> JobStore:
    """
    Build and connect a JobStore from a jobflow.yaml file exactly once
    per process, preventing connection churn.
    """
    path = os.environ.get("JOBFLOW_CONFIG_FILE")
    if not path:
        raise RuntimeError("JOBFLOW_CONFIG_FILE env var is not set.")
    cfg = loadfn(path)["JOB_STORE"]

    js = JobStore(
        docs_store=_mk_store(cfg["docs_store"]),
        additional_stores={name: _mk_store(s) for name, s in cfg.get("additional_stores", {}).items()},
    )
    js.docs_store.connect()
    for st in js.additional_stores.values():
        st.connect()
    return js


def get_jobstore() -> JobStore:
    """
    Process-wide JobStore accessor. This avoids creating new MongoClients
    on every query helper call.
    """
    global _shared_jobstore
    if _shared_jobstore is None:
        with _js_lock:
            if _shared_jobstore is None:
                _shared_jobstore = _build_jobstore_once()
    return _shared_jobstore


# -------------------------
# Query helpers
# -------------------------

def lookup_total_energy_eV(
    *,
    set_id: str,
    occ_key: str,
    calc_spec: CalcSpec,
) -> float | None:
    """
    Efficiently fetch just the total energy (eV) for a ForceFieldTaskDocument
    from Jobflow's docs store (outputs), using a projected query. This does not
    load any blobs or full documents.
    """
    js: JobStore = get_jobstore()

    criteria: dict[str, Any] = {
        "metadata.set_id": set_id,
        "metadata.occ_key": occ_key,
        "metadata.calculator": calc_spec.calculator,
        "metadata.relax_type": calc_spec.relax_type,
        "metadata.frozen_sublattices": calc_spec.frozen_sublattices,
        "output.output.energy": {"$exists": True},
    }
    if calc_spec.calculator_info["calc_type"] == CalcType.MACE_MPA_0:
        criteria["output.is_force_converged"] = True

    projection = {"output.output.energy": 1}
    docs = list(js.docs_store.query(criteria=criteria, properties=projection))

    if not docs:
        return None
    if len(docs) > 1:
        raise RuntimeError(
            f"Expected exactly one FF task, found {len(docs)} for set_id={set_id} "
            f"occ_key={occ_key} calculator={calc_spec.calculator} relax_type={calc_spec.relax_type}"
        )

    energy = docs[0]["output"]["output"]["energy"]
    return float(energy)


def lookup_unique(criteria: dict[str, Any]) -> dict[str, Any] | None:
    """
    Fails if more than one document matches.
    """
    js: JobStore = get_jobstore()

    rows = list(js.query(criteria=criteria, load=True))
    if not rows:
        return None
    if len(rows) > 1:
        raise RuntimeError(f"Expected exactly one document, found {len(rows)} for criteria={criteria!r}")

    return cast(dict[str, Any], rows[0]["output"])


def exists_unique(criteria: dict[str, Any]) -> bool:
    """
    Returns True if exactly one document matches.
    """
    js: JobStore = get_jobstore()

    count = js.count(criteria=criteria)
    if count > 1:
        raise RuntimeError(f"Expected at most one document, found {count} for criteria={criteria!r}")

    return count == 1
