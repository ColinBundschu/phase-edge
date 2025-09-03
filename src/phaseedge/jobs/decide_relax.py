from typing import cast

from jobflow.core.job import job, Response, Job
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from pymatgen.core import Structure

from phaseedge.storage import store


def lookup_ff_task(
    *, set_id: str, occ_key: str, model: str, relax_cell: bool, dtype: str
) -> ForceFieldTaskDocument | None:
    """
    Find an Atomate2 ForceFieldTaskDocument by metadata we inject at job submission.
    We only treat a doc as 'existing' if it has a final structure (finished run).
    """
    return store.db_rw()["outputs"].find_one(  # type: ignore[return-value]
        {
            "metadata.set_id": set_id,
            "metadata.occ_key": occ_key,
            "metadata.model": model,
            "metadata.relax_cell": relax_cell,
            "metadata.dtype": dtype,
            # Ensure it actually finished
            "structure": {"$exists": True},
            "output.energy": {"$exists": True},
        }
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
) -> ForceFieldTaskDocument | Response:
    """
    If a ForceFieldTaskDocument already exists for (set_id, occ_key, model, relax_cell, dtype),
    return that doc. Otherwise schedule a new relax job and alias our output to its TaskDocument.
    """
    existing = lookup_ff_task(
        set_id=set_id, occ_key=occ_key, model=model, relax_cell=relax_cell, dtype=dtype
    )
    if existing:
        return existing

    maker = ForceFieldRelaxMaker(
        force_field_name=model,
        relax_cell=relax_cell,
        calculator_kwargs={"default_dtype": dtype},
    )
    j_relax = cast(Job, maker.make(structure))

    # Queue routing only; assumed to be persisted onto the stored TaskDocument as 'metadata'
    j_relax.update_metadata(
        {
            "_category": category,
            "set_id": set_id,
            "occ_key": occ_key,
            "model": model,
            "relax_cell": relax_cell,
            "dtype": dtype,
        }
    )

    # Alias our output to the TaskDocument produced by Atomate2
    return Response(replace=j_relax, output=j_relax.output)
