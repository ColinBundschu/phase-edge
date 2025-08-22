import os
from urllib.parse import quote_plus

from pymongo import MongoClient, ASCENDING
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError


# -------------------------
# Environment → connection
# -------------------------

def _mongo_uri(user_env: str, pass_env: str) -> str:
    """
    Compose a mongodb:// URI from your NERSC-style env vars.

    Required env:
      - MONGO_HOST
      - MONGO_DB
      - user_env, pass_env (e.g., MONGO_ADMIN_USER / MONGO_ADMIN_PASS or MONGO_RO_USER / MONGO_RO_PASS)

    Optional env:
      - MONGO_PORT (defaults to 27017)
      - MONGO_OPTIONS (extra URI query string, e.g., "tls=true&retryWrites=true")
      - MONGO_AUTHSOURCE (defaults to MONGO_DB)
    """
    host = os.environ["MONGO_HOST"]
    db = os.environ["MONGO_DB"]
    user = os.environ[user_env]
    pwd = os.environ[pass_env]

    port = os.environ.get("MONGO_PORT", "27017")
    authsource = os.environ.get("MONGO_AUTHSOURCE", db)
    options = os.environ.get("MONGO_OPTIONS", "")

    # URL-escape password in case it has special chars
    user_q = quote_plus(user)
    pwd_q = quote_plus(pwd)

    base = f"mongodb://{user_q}:{pwd_q}@{host}:{port}/{db}?authSource={authsource}"
    if options:
        sep = "&" if "?" in base else "?"
        base = f"{base}{sep}{options}"
    return base


_rw_client: MongoClient | None = None
_ro_client: MongoClient | None = None


def db_rw() -> Database:
    """Read/Write DB using admin credentials (for inserts and index creation)."""
    global _rw_client
    if _rw_client is None:
        _rw_client = MongoClient(_mongo_uri("MONGO_ADMIN_USER", "MONGO_ADMIN_PASS"), tz_aware=True)
        _ensure_indexes(_rw_client[os.environ["MONGO_DB"]])
    return _rw_client[os.environ["MONGO_DB"]]


def db_ro() -> Database:
    """Read-only DB using RO credentials (for queries)."""
    global _ro_client
    if _ro_client is None:
        _ro_client = MongoClient(_mongo_uri("MONGO_RO_USER", "MONGO_RO_PASS"), tz_aware=True)
        # No index creation here.
    return _ro_client[os.environ["MONGO_DB"]]


# -------------------------
# One-time index creation
# -------------------------

def _ensure_indexes(db: Database) -> None:
    """
    Idempotent: create unique indexes we rely on.
    Call once on RW connection.
    """
    db.snapshot_sets.create_index([("set_id", ASCENDING)], unique=True, name="uniq_set_id")
    db.snapshots.create_index([("set_id", ASCENDING), ("index", ASCENDING)], unique=True, name="uniq_set_index")
    db.snapshots.create_index([("occ_key", ASCENDING)], unique=True, name="uniq_occ_key")


# -------------------------
# Public helpers (unchanged API)
# -------------------------

def upsert_snapshot_set(doc: dict) -> None:
    db = db_rw()
    db.snapshot_sets.update_one({"set_id": doc["set_id"]}, {"$setOnInsert": doc}, upsert=True)


def insert_snapshot_unique(doc: dict) -> bool:
    """
    Insert a snapshot with unique (set_id, index) and unique occ_key.
    Returns True if inserted, False if duplicate prevented insert.
    """
    db = db_rw()
    try:
        db.snapshots.insert_one(doc)
        return True
    except DuplicateKeyError:
        return False


def count_by_set(set_id: str) -> int:
    db = db_ro()
    return db.snapshots.count_documents({"set_id": set_id})


def list_by_set(set_id: str, fields: tuple[str, ...] = ("occ_key", "index", "path")) -> list[dict]:
    db = db_ro()
    proj = {f: 1 for f in fields}
    proj["_id"] = 0
    return list(db.snapshots.find({"set_id": set_id}, proj))
