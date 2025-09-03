import os
from urllib.parse import quote_plus
from typing import Literal

from pymongo import MongoClient, ASCENDING
from pymongo.database import Database
from pymongo.collection import Collection

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
_ro_client: MongoClient | None = None


def db_rw() -> Database:
    """Read/Write DB using admin creds (for inserts + index creation)."""
    global _rw_client
    if _rw_client is None:
        _rw_client = MongoClient(_mongo_uri("MONGO_ADMIN_USER", "MONGO_ADMIN_PASS"), tz_aware=True)
    return _rw_client[os.environ["MONGO_DB"]]


def db_ro() -> Database:
    """Read-only DB using RO creds (for queries)."""
    global _ro_client
    if _ro_client is None:
        _ro_client = MongoClient(_mongo_uri("MONGO_RO_USER", "MONGO_RO_PASS"), tz_aware=True)
    return _ro_client[os.environ["MONGO_DB"]]


def coll(name: str, *, mode: Literal["rw", "ro"] = "ro") -> Collection:
    """Convenience accessor for a collection in RO/RW mode."""
    return (db_rw() if mode == "rw" else db_ro())[name]
