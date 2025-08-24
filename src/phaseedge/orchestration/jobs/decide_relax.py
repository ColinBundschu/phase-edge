from typing import cast

from jobflow.core.job import job, Response, Job
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from pymatgen.core import Structure

# pull both the lookup helper and the store job from the same module
from phaseedge.orchestration.jobs.store_mace_result import (
    lookup_mace_result,
    store_mace_result,
)

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
):
    """
    If result exists for (set_id, occ_key, model, relax_cell, dtype): return it.
    Otherwise schedule [ForceFieldRelax -> store_mace_result] and tag with _category.
    """
    existing = lookup_mace_result(
        set_id, occ_key, model=model, relax_cell=relax_cell, dtype=dtype
    )
    if existing:
        return Response(output=existing)

    maker = ForceFieldRelaxMaker(
        force_field_name=model,
        relax_cell=relax_cell,
        calculator_kwargs={"default_dtype": dtype},
    )
    j_relax = cast(Job, maker.make(structure))
    j_relax.name = f"mace_relax[{model}]"
    j_relax.metadata = {**(j_relax.metadata or {}), "_category": category}

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

    return Response(replace=[j_relax, j_store])
