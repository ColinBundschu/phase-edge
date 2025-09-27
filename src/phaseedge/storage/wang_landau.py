from typing import Any, TypedDict, cast
from phaseedge.storage.store import build_jobstore, lookup_unique
from pymongo.collection import Collection


class WLBlockDoc(TypedDict, total=True):
    kind: str  # "WLBlockDoc"
    wl_key: str
    wl_block_key: str
    parent_wl_block_key: str  # "GENESIS" for the first chunk
    mod_updates: list[dict[str, Any]]  # list of {"step": int, "m": float}
    bin_samples: list[dict[str, Any]]  # list of {"bin": int, "occ": list[int]}
    samples_per_bin: int
    block_steps: int
    step_end: int
    cation_counts: list[dict[str, Any]]  # list of {"bin": int, "sublattice": str, "element": str, "n_sites": int, "count": int}
    production_mode: bool
    collect_cation_stats: bool
    state: dict[str, Any]
    occupancy: list[int]
    

def fetch_wl_tip(wl_key: str) -> WLBlockDoc | None:
    js = build_jobstore()
    # fetch the last block for this wl_key
    rows = list(js.docs_store.query(
        criteria={"output.kind": "WLBlockDoc", "output.wl_key": wl_key},
        properties={"_id": 0, "output": 1},
        sort={"output.step_end": -1},  # latest first
        limit=1,
    ))
    return rows[0]["output"] if rows else None


def lookup_wl_block_by_key(wl_block_key: str) -> WLBlockDoc | None:
    """
    Fetch a WLBlockDoc by wl_block_key from Jobflow's outputs store (rehydrated).
    """
    criteria = {"output.kind": "WLBlockDoc", "output.wl_block_key": wl_block_key}
    result = lookup_unique(criteria=criteria)
    return cast(WLBlockDoc, result) if result is not None else None


def get_first_matching_wl_block(run_spec) -> WLBlockDoc | None:
    js = build_jobstore()
    rows = list(js.docs_store.query(
            criteria={
                "output.kind": "WLBlockDoc",
                "output.wl_key": run_spec.wl_key,
                "output.block_steps": {"$gte": int(run_spec.steps)},
                "output.samples_per_bin": {"$gte": int(run_spec.samples_per_bin)},
            },
            properties={"_id": 0, "output.wl_block_key": 1, "output.step_end": 1},
            sort={"output.step_end": 1},
            limit=1,
        ))
    
    if not rows:
        return None
    [tip] = rows
    return tip["output"]


def ensure_wl_output_indexes() -> None:
    coll = cast(Collection, build_jobstore().docs_store._collection)

    coll.create_index(
        [("output.wl_block_key", 1)],
        name="wl_out_uniq_ckpt_key",
        unique=True,
        partialFilterExpression={"output.kind": "WLBlockDoc"},
    )
    coll.create_index(
        [("output.wl_key", 1), ("output.parent_wl_block_key", 1)],
        name="wl_out_uniq_parent_per_wl",
        unique=True,
        partialFilterExpression={"output.kind": "WLBlockDoc"},
    )
    coll.create_index(
        [("output.wl_key", 1), ("output.step_end", -1)],
        name="wl_out_tip",
        unique=False,
        partialFilterExpression={"output.kind": "WLBlockDoc"},
    )
