from typing import Any, Mapping
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
        "bin_indices": result.bin_indices.tolist(),
        "grid": {"anchor": result.grid_anchor, "bin_width": result.bin_width},
        "meta": dict(result.meta),
    }
