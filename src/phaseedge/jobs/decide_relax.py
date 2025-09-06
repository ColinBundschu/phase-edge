from dataclasses import dataclass
from typing import Any, cast

from jobflow.core.job import job, Response, Job
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from pymatgen.core import Structure

from phaseedge.storage import store


def lookup_ff_task(
    *,
    set_id: str,
    occ_key: str,
    model: str,
    relax_cell: bool,
    dtype: str,
    require_converged: bool = True,
) -> ForceFieldTaskDocument | None:
    """
    Assumes the Atomate2 ForceFieldTaskDocument is embedded under 'output'.
    Returns that embedded document if it exists (and is finished/converged per flags).
    """
    q: dict[str, object] = {
        "metadata.set_id": set_id,
        "metadata.occ_key": occ_key,
        "metadata.model": model,          # keep the full unsplit string
        "metadata.relax_cell": relax_cell,
        "metadata.dtype": dtype,
        # FINISHED (embedded)
        "output.structure": {"$exists": True},
        "output.output.energy": {"$exists": True},
    }
    if require_converged:
        q["output.is_force_converged"] = True

    # Only pull the embedded TD
    doc = store.db_rw()["outputs"].find_one(q, {"_id": 0, "output": 1})
    if not doc or "output" not in doc:
        return None
    return cast(ForceFieldTaskDocument, doc["output"])


@dataclass(frozen=True)
class _FFSpec:
    force_field_name: str
    calculator_kwargs: dict[str, Any]


def _parse_model_spec(model: str, *, dtype: str) -> _FFSpec:
    """
    Interpret the user 'model' string.

    Semantics:
      - 'NAME'                       -> force_field_name='NAME'
      - 'NAME;EXTRA'                 -> force_field_name='NAME', calculator_kwargs['model']='EXTRA'
    Everything before the first ';' is the force-field name. Everything after (if non-empty)
    is passed through as the 'model' kwarg to the calculator. Whitespace is stripped.

    We always add {'default_dtype': dtype}.
    """
    head, sep, tail = model.partition(";")
    ff_name = head.strip()
    calc_kwargs: dict[str, Any] = {"default_dtype": dtype}
    if sep and tail.strip():
        calc_kwargs["model"] = tail.strip()
    return _FFSpec(force_field_name=ff_name, calculator_kwargs=calc_kwargs)


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
    existing = lookup_ff_task(
        set_id=set_id, occ_key=occ_key, model=model,
        relax_cell=relax_cell, dtype=dtype, require_converged=True
    )
    if existing:
        return existing

    ff_spec = _parse_model_spec(model, dtype=dtype)
    maker = ForceFieldRelaxMaker(
        force_field_name=ff_spec.force_field_name,
        relax_cell=relax_cell,
        calculator_kwargs=ff_spec.calculator_kwargs,
    )
    j_relax = cast(Job, maker.make(structure))
    j_relax.update_metadata(
        {
            "_category": category,
            "set_id": set_id,
            "occ_key": occ_key,
            "model": model,       # keep full string
            "relax_cell": relax_cell,
            "dtype": dtype,
        }
    )
    return Response(replace=j_relax, output=j_relax.output)
