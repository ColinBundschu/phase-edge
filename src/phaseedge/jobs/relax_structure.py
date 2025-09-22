from dataclasses import dataclass
from typing import Any, cast

from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from pymatgen.core import Structure

from phaseedge.storage.store import lookup_total_energy_eV


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
def _require_converged(is_force_converged: bool) -> None:
    if not is_force_converged:
        raise RuntimeError("Force-field relaxation did not converge (is_force_converged=False).")


@job
def relax_structure(
    *,
    set_id: str,
    occ_key: str,
    structure: Structure,
    model: str,
    relax_cell: bool,
    dtype: str,
    category: str,
) -> float | Response:
    # Strict reuse: only return an existing doc if converged
    existing_energy = lookup_total_energy_eV(
        set_id=set_id, occ_key=occ_key, model=model,
        relax_cell=relax_cell, dtype=dtype, require_converged=True
    )
    if existing_energy is not None:
        raise RuntimeError(
            "A converged relaxation already exists for this set_id/occ_key/model/relax_cell/dtype. "
            "No new relaxation scheduled."
        )

    ff_spec = _parse_model_spec(model, dtype=dtype)
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

    j_assert = _require_converged(j_relax.output.is_force_converged)
    j_assert.name = "ff_require_converged"
    j_assert.update_metadata(j_relax.metadata or {})

    subflow = Flow([j_relax, j_assert], name="ff_relax_then_assert")
    # Expose the relax TaskDoc as the flow's output
    return Response(replace=subflow, output=j_relax.output.output.energy)
