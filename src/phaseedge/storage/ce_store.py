from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, TypedDict, cast

from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from phaseedge.storage import store

__all__ = [
    "CETrainRef",
    "CEStats",
    "CESystem",
    "CESampling",
    "CEEngine",
    "CEHyperparams",
    "CEModelDoc",
    "lookup_ce_by_key",
    "insert_ce_model",
    "ensure_ce_indexes",
]

# --------------------------- Typed Schemas -----------------------------------------


class CETrainRef(TypedDict):
    """Stable reference to a relax datum used for CE training."""
    set_id: str
    occ_key: str
    model: str
    relax_cell: bool
    dtype: str


class CEStats(TypedDict):
    """
    Canonical fit statistics (per configurational site, i.e., per cation site).
    All fields are required; keep storage deterministic and branch-free.
    """
    n: int
    mae_per_site: float
    rmse_per_site: float
    max_abs_per_site: float


class CESystem(TypedDict):
    """
    System identity.
    """
    prototype: str
    prototype_params: Mapping[str, Any]
    supercell_diag: Sequence[int]
    # MULTI-SUBLATTICE: all prototype placeholders that are replaceable (e.g., ["Mg", "Al"] for spinel)
    replace_elements: Sequence[str]


class CESampling(TypedDict):
    """
    Exact sampling identity.

    For multi-sublattice CE we persist a normalized 'sources' block
    (e.g., [{"type": "sublattice_composition", "elements": [...]}])
    plus an explicit algo_version tag.
    """
    sources: Sequence[Mapping[str, Any]]   # canonicalized at keying time
    algo_version: str


class CEEngine(TypedDict):
    """Relaxation/engine identity for the training data."""
    model: str
    relax_cell: bool
    dtype: str


class CEHyperparams(TypedDict):
    """All knobs that distinguish CE models (baked into ce_key)."""
    basis_spec: Mapping[str, Any]
    regularization: Mapping[str, Any]
    extra: Mapping[str, Any]      # reserved for future knobs (e.g., weights) when added


class CEModelDoc(TypedDict):
    """
    Stored CE model document. All fields required to keep persistence exact and immutable.
    """
    ce_key: str

    system: CESystem
    sampling: CESampling
    engine: CEEngine
    hyperparams: CEHyperparams

    # Provenance and payload
    train_refs: Sequence[CETrainRef]
    dataset_hash: str                        # hash over sorted (occ_key, energy) pairs
    payload: Mapping[str, Any]               # ECIs, basis metadata, etc.

    stats: CEStats

    # Code/version provenance (git SHAs, tags, etc.)
    phaseedge_git: str
    disorder_git: str

    created_at: str                          # ISO8601 UTC
    updated_at: str                          # ISO8601 UTC


# --------------------------- Mongo helpers -----------------------------------------

_INDEX_ENSURED: bool = False


def _ce_coll() -> Collection:
    """RW handle to the CE models collection (ensures unique index on first access)."""
    global _INDEX_ENSURED
    coll = store.db_rw()["ce_models"]
    if not _INDEX_ENSURED:
        coll.create_index("ce_key", unique=True)
        _INDEX_ENSURED = True
    return coll


def ensure_ce_indexes() -> None:
    """Idempotently (re)create helpful indexes. Safe to call many times."""
    coll = _ce_coll()
    coll.create_index("ce_key", unique=True)
    # Optional: coll.create_index([("system.replace_elements", 1)])

def lookup_ce_by_key(ce_key: str) -> CEModelDoc | None:
    """Fetch a CE model by its unique key."""
    doc = _ce_coll().find_one({"ce_key": ce_key})
    return cast(CEModelDoc | None, doc)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def insert_ce_model(doc: CEModelDoc) -> CEModelDoc:
    """
    Parallel-safe insert of a CE model.
    - On first writer: inserts and returns the stored document.
    - On duplicate-key (another writer won the race): fetches and returns existing.
    Never overwrites an existing CE.
    """
    # Do not mutate caller's object
    to_write: CEModelDoc = {**doc}
    now = _utc_now_iso()
    to_write.setdefault("created_at", now)
    to_write["updated_at"] = now

    coll = _ce_coll()
    try:
        coll.insert_one(to_write)  # may raise DuplicateKeyError
        stored = coll.find_one({"ce_key": to_write["ce_key"]})
        return cast(CEModelDoc, stored) if stored is not None else to_write
    except DuplicateKeyError:
        existing = coll.find_one({"ce_key": to_write["ce_key"]})
        if existing is None:
            # Extremely rare race; fall back to returning caller's document with timestamps
            return to_write
        return cast(CEModelDoc, existing)
