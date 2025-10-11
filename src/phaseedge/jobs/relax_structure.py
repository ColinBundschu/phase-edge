from dataclasses import dataclass
from typing import Any, cast
from enum import Enum

from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
from pymatgen.core import Structure
from atomate2.vasp.jobs.mp import MPGGARelaxMaker, MP24RelaxMaker
from atomate2.vasp.powerups import update_user_incar_settings

from phaseedge.storage.store import lookup_total_energy_eV


class RelaxType(str, Enum):
    VASP_MP_GGA = "vasp-mp-gga"
    VASP_MP_24 = "vasp-mp-24"
    MACE_MPA_0 = "MACE-MPA-0"


@dataclass(frozen=True)
class RelaxSpec:
    relax_type: RelaxType
    calculator_kwargs: dict[str, Any]


def _parse_relax_spec(model: str) -> RelaxSpec:
    """
    Interpret the user 'model' string.

    Semantics:
      - 'NAME'                       -> force_field_name='NAME'
      - 'NAME;EXTRA'                 -> force_field_name='NAME', calculator_kwargs['model']='EXTRA'
    Everything before the first ';' is the force-field name. Everything after (if non-empty)
    is passed through as the 'model' kwarg to the calculator. Whitespace is stripped.
    """
    head, sep, tail = model.partition(";")
    ff_name = head.strip()
    calc_kwargs: dict[str, Any] = {}
    if sep and tail.strip():
        calc_kwargs["model"] = tail.strip()
    return RelaxSpec(relax_type=RelaxType(ff_name), calculator_kwargs=calc_kwargs)


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
    category: str,
) -> float | Response:
    existing_energy = lookup_total_energy_eV(set_id=set_id, occ_key=occ_key, model=model, relax_cell=relax_cell)
    if existing_energy is not None:
        raise RuntimeError(
            "A converged relaxation already exists for this set_id/occ_key/model/relax_cell. "
            "No new relaxation scheduled."
        )

    relax_spec = _parse_relax_spec(model)

    if relax_spec.relax_type == RelaxType.VASP_MP_GGA:
        if not relax_cell:
            raise NotImplementedError("TODO: Implement fixed-cell VASP calculation.")
        maker = MPGGARelaxMaker()
        j_relax = cast(Job, maker.make(structure))
    elif relax_spec.relax_type == RelaxType.VASP_MP_24:
        if not relax_cell:
            raise NotImplementedError("TODO: Implement fixed-cell VASP calculation.")
        maker1 = ForceFieldRelaxMaker(
            force_field_name="MATPES_R2SCAN",
            relax_cell=True,
        )
        relax1 = cast(Job, maker1.make(structure))
        maker2 = MP24RelaxMaker()
        relax2 = cast(Job, maker2.make(cast(Structure, relax1.output.structure)))
        j_relax = Flow([relax1, relax2], output=relax2.output, name="DoubleRelax")

    elif relax_spec.relax_type == RelaxType.MACE_MPA_0:
        maker = ForceFieldRelaxMaker(
            force_field_name=relax_spec.relax_type,
            relax_cell=relax_cell,
            steps=5000,
            calculator_kwargs=relax_spec.calculator_kwargs | {"default_dtype": "float64"},
        )
        j_relax = cast(Job, maker.make(structure))
    else:
        raise ValueError(f"Unrecognized relax_type: {relax_spec.relax_type}")

    j_relax.name = f"{relax_spec.relax_type}_relax"
    j_relax.update_metadata(
        {
            "_category": category,
            "set_id": set_id,
            "occ_key": occ_key,
            "model": model,
            "relax_cell": relax_cell,
        }
    )

    if relax_spec.relax_type == RelaxType.VASP_MP_GGA or relax_spec.relax_type == RelaxType.VASP_MP_24:
        j_relax = cast(Job, update_user_incar_settings(j_relax, incar_updates={"NCORE": 1, "NSIM": 8, "KPAR": 4, "EDIFF": 1e-6, "EDIFFG": -0.02}))
        subflow = Flow([j_relax], name=f"{relax_spec.relax_type}_relax")
    elif relax_spec.relax_type == RelaxType.MACE_MPA_0:
        j_assert = _require_converged(j_relax.output.is_force_converged)
        j_assert.name = "ff_require_converged"
        j_assert.update_metadata(j_relax.metadata)
        subflow = Flow([j_relax, j_assert], name=f"{relax_spec.relax_type}_relax_then_assert")
    else:
        raise ValueError(f"Unrecognized relax_type: {relax_spec.relax_type}")

    # Expose the relax TaskDoc as the flow's output
    return Response(replace=subflow, output=j_relax.output.output.energy)
