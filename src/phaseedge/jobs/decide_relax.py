from dataclasses import dataclass
from typing import Any, Mapping, cast

from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow
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
    q: dict[str, object] = {
        "metadata.set_id": set_id,
        "metadata.occ_key": occ_key,
        "metadata.model": model,
        "metadata.relax_cell": relax_cell,
        "metadata.dtype": dtype,
        "output.structure": {"$exists": True},
        "output.output.energy": {"$exists": True},
    }
    if require_converged:
        q["output.is_force_converged"] = True

    doc = store.db_rw()["outputs"].find_one(q, {"_id": 0, "output": 1})
    if not doc or "output" not in doc:
        return None
    return cast(ForceFieldTaskDocument, doc["output"])


@dataclass(frozen=True)
class _FFSpec:
    force_field_name: str
    calculator_kwargs: dict[str, Any]


def _parse_model_spec(model: str, *, dtype: str, calc_overrides: Mapping[str, Any] | None = None) -> _FFSpec:
    head, sep, tail = model.partition(";")
    ff_name = head.strip()
    calc_kwargs: dict[str, Any] = {"default_dtype": dtype}
    if sep and tail.strip():
        calc_kwargs["model"] = tail.strip()
    if calc_overrides:
        calc_kwargs.update({str(k): v for k, v in calc_overrides.items()})
    return _FFSpec(force_field_name=ff_name, calculator_kwargs=calc_kwargs)


@job
def _require_converged(doc: ForceFieldTaskDocument) -> None:
    if not doc.is_force_converged:
        raise RuntimeError(f"Force-field relaxation did not converge (is_force_converged=False). n_steps={doc.output.n_steps}")


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
    # Strict reuse: only return an existing doc if converged
    existing = lookup_ff_task(
        set_id=set_id, occ_key=occ_key, model=model,
        relax_cell=relax_cell, dtype=dtype, require_converged=True
    )
    if existing:
        return existing

    ff_spec = _parse_model_spec(model, dtype=dtype, calc_overrides=None)
    maker = ForceFieldRelaxMaker(
        force_field_name=ff_spec.force_field_name,
        relax_cell=relax_cell,
        steps=5000,  # large default so you don't have to pass flags
        calculator_kwargs=ff_spec.calculator_kwargs,
    )

    j_relax = cast(Job, maker.make(structure))
    j_relax.name = "ff_relax"
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

    j_assert = _require_converged(j_relax.output)
    j_assert.name = "ff_require_converged"
    j_assert.update_metadata(j_relax.metadata or {})

    subflow = Flow([j_relax, j_assert], name="ff_relax_then_assert")
    # Expose the relax TaskDoc as the flow's output
    return Response(replace=subflow, output=j_relax.output)
