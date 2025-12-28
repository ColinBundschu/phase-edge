import dataclasses
from typing import Iterable, cast, Mapping

from jobflow.core.job import job, Job
from jobflow.core.flow import Flow
from atomate2.forcefields.jobs import ForceFieldRelaxMaker
import numpy as np
from pymatgen.core import Structure
from atomate2.vasp.jobs.mp import MPGGARelaxMaker, MP24RelaxMaker
from atomate2.vasp.powerups import update_user_incar_settings

from phaseedge.schemas.calc_spec import CalcSpec, CalcType, RelaxType, SpinType
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.science.random_configs import build_sublattice_positions_for_struct
from phaseedge.storage.store import lookup_total_energy_eV


def strip_placeholder_species(
    structure: Structure,
    *,
    placeholder: str = "X",
) -> Structure:
    """
    Return a copy of `structure` with any sites occupied by a placeholder species removed.

    Raises:
      ValueError if no non-placeholder atoms remain or if mixed sites are present.
    """
    if any(not site.is_ordered for site in structure.sites):
        raise ValueError("Disordered structures are not supported; enumerate to a concrete occupancy first.")

    rm_indices: list[int] = []
    new_struct = structure.copy()
    for i, site in enumerate(new_struct.sites):
        if placeholder == site.specie.symbol:
            rm_indices.append(i)

    if rm_indices:
        new_struct.remove_sites(rm_indices)

    if len(new_struct) == 0:
        raise ValueError("All atoms were placeholders—nothing left to evaluate.")

    return new_struct


def freeze_sublattices_on_structure(
    structure: Structure,
    frozen_sublattices: set[str],
    sublattice_positions: Mapping[str, Iterable[tuple[float, float, float]]],
) -> Structure:
    if not frozen_sublattices:
        return structure
    
    frozen_indices = []
    for label in frozen_sublattices:
        frozen_indices.extend(sublattice_positions[label])

    sd_array = []
    for site in structure.sites:
        frac_coords = tuple(site.frac_coords)
        is_frozen = any(
            np.allclose(frac_coords, pos, atol=1e-5) for pos in frozen_indices
        )
        sd_array.append([False, False, False] if is_frozen else [True, True, True])
    new_struct = structure.copy()
    new_struct.add_site_property("selective_dynamics", sd_array)
    return new_struct


def randomly_kick_unfrozen_atoms(
    structure: Structure,
    *,
    amplitude_A: float = 0.01,
) -> Structure:
    """
    Return a copy of `structure` where unfrozen atoms have been given a small
    random Cartesian displacement of magnitude ~amplitude_A (Å).

    "Frozen" is interpreted via the 'selective_dynamics' site property:
      - If present: a component is movable iff the corresponding boolean is True.
      - If absent: the site is treated as fully movable.

    Parameters
    ----------
    structure : Structure
        Input structure, possibly with a 'selective_dynamics' site property.
    amplitude_A : float
        Typical scale of the random displacement in Å. If <= 0, no changes
        are applied and the structure is returned unchanged.
    rng : np.random.Generator, optional
        RNG to use. If None, a new default_rng() is created.

    Returns
    -------
    Structure
        New Structure with kicked coordinates.
    """
    if amplitude_A <= 0:
        return structure

    rng = np.random.default_rng(12345)

    new_struct = structure.copy()
    lattice = new_struct.lattice

    # Work in Cartesian, then convert back to fractional for translation.
    for i, site in enumerate(new_struct.sites):
        sd = site.properties.get("selective_dynamics")

        if sd is None:
            # No selective dynamics: treat as fully movable
            movable_mask = np.array([True, True, True], dtype=bool)
        else:
            movable_mask = np.array(sd, dtype=bool)
            if movable_mask.shape != (3,):
                raise ValueError(
                    f"selective_dynamics for site {i} must have length 3, "
                    f"got {movable_mask}."
                )

        if not movable_mask.any():
            # Fully frozen site: skip
            continue

        # Random direction and magnitude in Cartesian
        disp_cart = rng.normal(loc=0.0, scale=amplitude_A, size=3)
        # Zero-out frozen components (if any)
        disp_cart = disp_cart * movable_mask

        # Convert Cartesian displacement to fractional
        disp_frac = lattice.get_fractional_coords(disp_cart)
        if np.allclose(disp_frac, 0.0):
            continue

        new_struct.translate_sites(
            indices=[i],
            vector=disp_frac,
            frac_coords=True,
        )

    return new_struct


def zero_out_initial_magmoms(structure: Structure) -> Structure:
    """
    Return a copy of `structure` with site-wise initial magnetic moments set to 0.
    """
    new_struct = structure.copy()

    if "magmom" not in new_struct.site_properties:
        raise ValueError("Structure has no 'magmom' site property to zero out.")

    new_struct.remove_site_property("magmom")
    new_struct.add_site_property("magmom", [0.0] * len(new_struct))
    return new_struct


@job
def _require_converged(is_force_converged: bool) -> None:
    if not is_force_converged:
        raise RuntimeError("Force-field relaxation did not converge (is_force_converged=False).")


def evaluate_structure(
    *,
    occ_key: str,
    structure: Structure,
    calc_spec: CalcSpec,
    category: str,
    prototype_spec: PrototypeSpec,
    supercell_diag: tuple[int, int, int],
    comp_map_sig: str,
) -> Flow:
    energy_result = lookup_total_energy_eV(occ_key=occ_key, calc_spec=calc_spec)
    if energy_result is not None:
        if energy_result.max_force_eV_per_A <= calc_spec.max_force_eV_per_A:
            raise RuntimeError(
                "A converged relaxation already exists for this occ_key/calculator/relax_type/spin_type. "
                "No new relaxation scheduled."
            )
        # Frozen-ness of atoms is preserved in stored structure
        structure = energy_result.structure
    elif calc_spec.spin_type == SpinType.ZERO_INIT_MAGNETIC:
        fm_calc_spec = dataclasses.replace(calc_spec, spin_type=SpinType.FERROMAGNETIC)
        energy_result = lookup_total_energy_eV(occ_key=occ_key, calc_spec=fm_calc_spec)
        if energy_result is None:
            raise RuntimeError(
                "No existing ferromagnetic calculation found to initialize zero-init-magnetic structure."
            )
        structure = energy_result.structure
        structure = zero_out_initial_magmoms(structure)
    else:
        structure = strip_placeholder_species(structure)
        sublattice_positions = build_sublattice_positions_for_struct(prototype_spec=prototype_spec, supercell_diag=supercell_diag)
        structure = freeze_sublattices_on_structure(
            structure,
            frozen_sublattices=calc_spec.frozen_sublattices_set,
            sublattice_positions=sublattice_positions,
        )
        structure = randomly_kick_unfrozen_atoms(structure, amplitude_A=0.01)

    metadata = {
        "occ_key": occ_key,
        "calculator": calc_spec.calculator,
        "relax_type": calc_spec.relax_type,
        "spin_type": SpinType.NONMAGNETIC.value if calc_spec.calc_type == CalcType.MACE_MPA_0 else calc_spec.spin_type,
        "max_force_eV_per_A": calc_spec.max_force_eV_per_A,
        "frozen_sublattices": calc_spec.frozen_sublattices,
        "prototype": prototype_spec.prototype,
        "comp_map_sig": comp_map_sig,
        "algo_version": "eval-struct-v1",
    }

    if calc_spec.spin_type not in [SpinType.NONMAGNETIC, SpinType.FERROMAGNETIC, SpinType.ZERO_INIT_MAGNETIC]:
        raise NotImplementedError("Only nonmagnetic, ferromagnetic, and zero-init-magnetic spin types are supported.")

    if calc_spec.calc_type == CalcType.VASP_MP_GGA:
        if calc_spec.relax_type != RelaxType.FULL:
            raise NotImplementedError("TODO: Implement fixed-cell VASP calculation.")
        maker = MPGGARelaxMaker()
        j_evaluate = cast(Job, maker.make(structure))
        j_evaluate.update_metadata(metadata)

    elif calc_spec.calc_type == CalcType.VASP_MP_24:
        if calc_spec.relax_type != RelaxType.FULL:
            raise NotImplementedError("TODO: Implement fixed-cell VASP calculation.")
        if calc_spec.frozen_sublattices_set or energy_result is not None:
            maker = MP24RelaxMaker()
            j_evaluate = cast(Job, maker.make(structure))
            j_evaluate.update_metadata(metadata)
        else:
            maker1 = ForceFieldRelaxMaker(
                force_field_name="MATPES_R2SCAN",
                relax_cell=True,
                steps=2500,
                relax_kwargs={'fmax': 0.02},
            )
            relax1 = cast(Job, maker1.make(structure))
            maker2 = MP24RelaxMaker()
            relax2 = cast(Job, maker2.make(cast(Structure, relax1.output.structure)))
            relax2.update_metadata(metadata)
            j_evaluate = Flow([relax1, relax2], output=relax2.output, name="DoubleRelax")

    elif calc_spec.calc_type == CalcType.MACE_MPA_0:
        maker = ForceFieldRelaxMaker(
            force_field_name=calc_spec.calc_type,
            relax_cell=(calc_spec.relax_type == RelaxType.FULL),
            steps=5000,
            calculator_kwargs=calc_spec.calc_kwargs | {"default_dtype": "float64"},
            relax_kwargs={'fmax': calc_spec.max_force_eV_per_A},
        )
        j_evaluate = cast(Job, maker.make(structure))
        j_evaluate.update_metadata(metadata)
    else:
        raise ValueError(f"Unrecognized calc_type: {calc_spec.calc_type}")

    j_evaluate.name = f"evaluate_{calc_spec.calc_type}"
    j_evaluate.update_metadata({"_category": category})

    if calc_spec.calc_type == CalcType.VASP_MP_GGA or calc_spec.calc_type == CalcType.VASP_MP_24:
        ispin = 1 if calc_spec.spin_type == SpinType.NONMAGNETIC else 2
        j_evaluate = cast(Job, update_user_incar_settings(j_evaluate, incar_updates={
            "NCORE": 1, "NSIM": 64, "KPAR": 1, "EDIFF": 1e-6, "EDIFFG": -calc_spec.max_force_eV_per_A, "SYMPREC": 1E-7, "ISPIN": ispin, "NSW": 500,
            "IBRION": 1, "ISYM": 0, "LWAVE": False, "LVTOT": False, "LEFLF": False, "POTIM": 0.7, "NELMIN": 8, "NFREE": 10, "NELM": 60,
            }))
        subflow = Flow([j_evaluate], name=f"evaluate_{calc_spec.calc_type}", output=j_evaluate.output.output.energy)
    elif calc_spec.calc_type == CalcType.MACE_MPA_0:
        j_assert = _require_converged(j_evaluate.output.is_force_converged)
        j_assert.name = "ff_require_converged"
        j_assert.update_metadata(j_evaluate.metadata)
        subflow = Flow([j_evaluate, j_assert], name=f"evaluate_{calc_spec.calc_type}_then_assert", output=j_evaluate.output.output.energy)
    else:
        raise ValueError(f"Unrecognized calc_type: {calc_spec.calc_type}")

    return subflow
