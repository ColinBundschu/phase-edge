from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from phaseedge.storage import store

def _coll():
    return store.db_rw()["wang_landau"]

def insert_wl_result(doc: Mapping[str, Any]) -> str:
    res = _coll().insert_one(dict(doc))
    return str(res.inserted_id)

def to_doc(result) -> dict[str, Any]:
    return {
        "levels": result.levels.tolist(),
        "entropy": result.entropy.tolist(),
        "histogram": result.histogram.tolist(),
        "visited_mask": result.visited_mask.tolist(),
        "grid": {"anchor": result.grid_anchor, "bin_width": result.bin_width, "window": list(result.window_used)},
        "meta": dict(result.meta),
    }
