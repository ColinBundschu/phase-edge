import os
from urllib.parse import quote_plus
from typing import Literal, Any, Mapping, cast

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from monty.serialization import loadfn
from jobflow.core.store import JobStore
from maggma.stores import MongoStore, GridFSStore


# -------------------------
# Environment → connection
# -------------------------

def _mongo_uri(user_env: str, pass_env: str) -> str:
    """
    Compose a mongodb:// URI from env vars.

    Required env:
      - MONGO_HOST
      - MONGO_DB
      - user_env, pass_env (e.g., MONGO_ADMIN_USER / MONGO_ADMIN_PASS
        or MONGO_RO_USER / MONGO_RO_PASS)

    Optional env:
      - MONGO_PORT (default: 27017)
      - MONGO_OPTIONS (e.g., "tls=true&retryWrites=true")
      - MONGO_AUTHSOURCE (default: MONGO_DB)
    """
    host = os.environ["MONGO_HOST"]
    db = os.environ["MONGO_DB"]
    user = os.environ[user_env]
    pwd = os.environ[pass_env]

    port = os.environ.get("MONGO_PORT", "27017")
    authsource = os.environ.get("MONGO_AUTHSOURCE", db)
    options = os.environ.get("MONGO_OPTIONS", "")

    user_q = quote_plus(user)
    pwd_q = quote_plus(pwd)

    base = f"mongodb://{user_q}:{pwd_q}@{host}:{port}/{db}?authSource={authsource}"
    if options:
        base = f"{base}&{options}"
    return base


_rw_client: MongoClient | None = None


def db_rw() -> Database:
    """Read/Write DB using admin creds (for inserts + index creation)."""
    global _rw_client
    if _rw_client is None:
        _rw_client = MongoClient(_mongo_uri("MONGO_ADMIN_USER", "MONGO_ADMIN_PASS"), tz_aware=True)
    return _rw_client[os.environ["MONGO_DB"]]


def build_jobstore() -> JobStore:
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


def lookup_total_energy_eV(
    *,
    set_id: str,
    occ_key: str,
    model: str,
    relax_cell: bool,
) -> float | None:
    """
    Efficiently fetch just the total energy (eV) for a ForceFieldTaskDocument
    from Jobflow's docs store (outputs), using a projected query. This does not
    load any blobs or full documents.
    """
    js: JobStore = build_jobstore()

    # Minimal criteria to pinpoint the task and ensure energy is present.
    criteria: dict[str, Any] = {
        "metadata.set_id": set_id,
        "metadata.occ_key": occ_key,
        "metadata.model": model,
        "metadata.relax_cell": relax_cell,
        "output.output.energy": {"$exists": True},
    }
    if model != "vasp":
        criteria["output.is_force_converged"] = True

    projection = {
        "output.output.energy": 1,
    }

    docs_iter = js.docs_store.query(criteria=criteria, properties=projection)
    docs = list(docs_iter)

    if not docs:
        return None
    if len(docs) > 1:
        raise RuntimeError(
            f"Expected exactly one FF task, found {len(docs)} for set_id={set_id} occ_key={occ_key} "
            f"model={model} relax_cell={relax_cell}"
        )
    
    energy = docs[0]["output"]["output"]["energy"]
    return float(energy)


def lookup_unique(criteria: dict[str, Any]) -> dict[str, Any] | None:
    """
    Fails if more than one document matches.
    """
    js: JobStore = build_jobstore()

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
    js: JobStore = build_jobstore()

    count = js.count(criteria=criteria)
    if count > 1:
        raise RuntimeError(f"Expected at most one document, found {count} for criteria={criteria!r}")

    return count == 1

