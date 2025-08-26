from typing import Any, TypedDict, cast

from jobflow.core.job import job, Response, Job
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from pymatgen.core import Structure

# pull both the lookup helper and the store job from the same module
from phaseedge.orchestration.jobs.store_mace_result import (
    lookup_mace_result,
    store_mace_result,
)

class RelaxResult(TypedDict):
    occ_key: str
    status: str            # "existing" | "stored" | "scheduled"
    relax_doc_id: str | None


@job
def _finish_relax_record(
    *,
    set_id: str,
    occ_key: str,
    model: str,
    relax_cell: bool,
    dtype: str,
    # purely to enforce dependency on the store job; value is unused
    wait_for: Any | None = None,
) -> RelaxResult:
    """
    Finalize a consistent output payload after store_mace_result has completed.
    Looks up the record to return a stable RelaxResult for gather().
    """
    doc = lookup_mace_result(set_id, occ_key, model=model, relax_cell=relax_cell, dtype=dtype)
    relax_id = str(doc["_id"]) if doc and "_id" in doc else None
    status = "stored" if relax_id is not None else "scheduled"
    return {"occ_key": occ_key, "status": status, "relax_doc_id": relax_id}


@job
def check_or_schedule_relax(
    *,
    set_id: str,
    occ_key: str,
    structure: Structure,
    model: str,
    relax_cell: bool,
    dtype: str,
    category: str,
) -> RelaxResult | Response:
    """
    If result exists for (set_id, occ_key, model, relax_cell, dtype): return it.
    Otherwise replace this job with [ForceFieldRelax -> store_mace_result -> _finish_relax_record]
    and alias this job’s output to the finisher’s output, so downstream gather()
    sees a stable dict with 'occ_key'.
    """
    existing = lookup_mace_result(
        set_id, occ_key, model=model, relax_cell=relax_cell, dtype=dtype
    )
    if existing:
        relax_id = str(existing["_id"]) if "_id" in existing else None
        return {"occ_key": occ_key, "status": "existing", "relax_doc_id": relax_id}

    # Build relax job
    maker = ForceFieldRelaxMaker(
        force_field_name=model,
        relax_cell=relax_cell,
        calculator_kwargs={"default_dtype": dtype},
    )
    j_relax = cast(Job, maker.make(structure))
    j_relax.name = f"mace_relax[{model}]"
    j_relax.metadata = {**(j_relax.metadata or {}), "_category": category}

    # Store job (writes the result to Mongo)
    j_store = store_mace_result(
        set_id=set_id,
        occ_key=occ_key,
        model=model,
        relax_cell=relax_cell,
        dtype=dtype,
        result=j_relax.output,
    )
    j_store.name = "store_mace_result"
    j_store.metadata = {**(j_store.metadata or {}), "_category": category}

    # Finisher ensures a consistent dict output and enforces dependency on j_store
    j_finish = _finish_relax_record(
        set_id=set_id,
        occ_key=occ_key,
        model=model,
        relax_cell=relax_cell,
        dtype=dtype,
        wait_for=j_store.output,  # ensures we run after store
    )
    j_finish.name = "finish_relax_record"
    j_finish.metadata = {**(j_finish.metadata or {}), "_category": category}

    # IMPORTANT: alias this job's output to the finisher's output
    return Response(replace=[j_relax, j_store, j_finish], output=j_finish.output)
