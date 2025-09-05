from typing import Any, Mapping, Sequence, Tuple
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from phaseedge.storage import store


def _coll():
    return store.db_rw()["wang_landau_ckpt"]

def ensure_indexes() -> None:
    coll = _coll()
    coll.create_index([("wl_key", 1), ("parent_hash", 1)], name="uniq_child_per_parent", unique=True)
    coll.create_index([("wl_key", 1), ("step_end", 1)],     name="uniq_step_end",        unique=True)
    coll.create_index([("wl_key", 1), ("hash", 1)],         name="uniq_hash",            unique=True)
    coll.create_index([("wl_key", 1), ("step_end", -1)],    name="latest_by_steps")


def _to_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    return x

def canonical_payload(wl_key: str, step_end: int, chunk_size: int,
                      state: Mapping[str, Any], occupancy: np.ndarray) -> bytes:
    payload = {
        "version": 1,
        "wl_key": wl_key,
        "step_end": int(step_end),
        "chunk_size": int(chunk_size),
        "state": _to_list(state),
        "occupancy": _to_list(np.asarray(occupancy, dtype=int)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_tip(wl_key: str) -> Mapping[str, Any] | None:
    return _coll().find_one({"wl_key": wl_key}, sort=[("step_end", -1)])


def insert_checkpoint(
    *,
    wl_key: str,
    step_end: int,
    chunk_size: int,
    parent_hash: str,
    state: Mapping[str, Any],
    occupancy: np.ndarray,
    # --- explicit top-level metadata ---
    mod_updates: Sequence[Mapping[str, Any]],
    bin_samples: Sequence[Mapping[str, Any]],
    samples_per_bin: int,
) -> Tuple[str, Mapping[str, Any]]:
    """
    Insert an immutable checkpoint. Returns (inserted_id, doc).

    Note: the checkpoint `hash` remains a function ONLY of
    (wl_key, step_end, chunk_size, state, occupancy).
    """
    coll = _coll()
    payload_bytes = canonical_payload(wl_key, step_end, chunk_size, state, occupancy)
    this_hash = sha256_hex(payload_bytes)

    doc: dict[str, Any] = {
        "schema_version": 2,  # optional but helpful
        "created_at": datetime.now(timezone.utc).isoformat(),  # optional but helpful
        "wl_key": wl_key,
        "step_end": int(step_end),
        "chunk_size": int(chunk_size),
        "parent_hash": parent_hash,
        "hash": this_hash,
        "state": _to_list(state),
        "occupancy": _to_list(np.asarray(occupancy, dtype=int)),
        "mod_updates": _to_list(list(mod_updates)),
        "bin_samples": _to_list(list(bin_samples)),
        "samples_per_bin": int(samples_per_bin),
    }
    res = coll.insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    return doc["_id"], doc
