from typing import Any

from phaseedge.cli.common import parse_cutoffs_arg
from phaseedge.export._common import build_swap_mixtures, resolve_or_launch
from phaseedge.jobs.ensure_ce_from_mixtures import EnsureCEFromMixturesSpec
from phaseedge.schemas.calc_spec import CalcSpec, CalcType, RelaxType, SpinType
from phaseedge.schemas.ensure_dopt_ce_spec import EnsureDoptCESpec
from phaseedge.schemas.mixture import Mixture, sorted_composition_maps
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.science.random_configs import validate_counts_for_sublattices


def spinel_ce(
    a_cation: str,
    b_cation: str,
    convergence: float,
    *,
    category: str,
    spin_type: str,
    budget: int = 500,
    base_convergence: float = 0.05,
    launch: bool = False,
    min_partial_frac: float = 1.0,
) -> str | None:
    """
    Look up (and optionally launch) the final CE for a spinel A(B)2O4 system
    with the hard-coded D-opt CE settings used in the CLI example.

    Parameters
    ----------
    a_cation : str
        Element symbol on the tetrahedral A sublattice (Es).
    b_cation : str
        Element symbol on the octahedral B sublattice (Fm).
    convergence : float
        Max force convergence criterion in eV/Å for the final CalcSpec.
    base_convergence : float
        Max force convergence criterion in eV/Å for the MACE base CalcSpec.
    launch : bool, keyword-only
        If False (default), only check whether the final CE already exists and
        return its key if present, else None.
        If True, launch ensure_dopt_ce if the CE does not yet exist. In this
        mode, the function always returns the CE key (spec.final_ce_key).

    Returns
    -------
    str | None
        If launch is False:
            - existing final CE key if present in the store
            - None if no CE exists yet
        If launch is True:
            - the final CE key (even if it did not exist prior to launching)
    """
    # --- Prototype and lattice parameters (hard-coded) ---
    prototype = "spinel_Q0O"
    proto_params: dict[str, float] = {"a": 8.08, "u": 0.36}
    supercell_diag: tuple[int, int, int] = (2, 2, 2)
    seed = 42

    prototype_spec = PrototypeSpec(
        prototype=prototype,
        params=proto_params,
    )

    # --- Endpoint composition ---
    endpoint_cm: dict[str, dict[str, int]] = {
        "Es": {a_cation: 16},
        "Fm": {b_cation: 32},
    }
    endpoints = sorted_composition_maps([endpoint_cm])

    # --- Mixtures: sweep k = 1..16 (Es size), plus endpoints as K=1 ---
    mixtures: list[Mixture] = build_swap_mixtures(
        cation_a=a_cation, cation_b=b_cation,
        es_size=16, fm_size=32, K=30, seed=seed,
    )
    for ep in endpoints:
        mixtures.append(Mixture(composition_map=ep, K=1, seed=0))

    # --- Validation ---
    primitive_cell = prototype_spec.primitive_cell
    for mixture in mixtures:
        validate_counts_for_sublattices(
            primitive_cell=primitive_cell,
            supercell_diag=supercell_diag,
            composition_map=mixture.composition_map,
        )

    # --- CE hyperparameters ---
    cutoffs = parse_cutoffs_arg("2:10,3:8,4:5")

    base_calc_spec = CalcSpec(
        calculator=CalcType("MACE-MPA-0"),
        relax_type=RelaxType("full"),
        spin_type=SpinType("nonmagnetic"),
        max_force_eV_per_A=base_convergence,
        frozen_sublattices="",
    )

    final_calc_spec = CalcSpec(
        calculator=CalcType("vasp-mp-24"),
        relax_type=RelaxType("full"),
        spin_type=SpinType(spin_type),
        max_force_eV_per_A=convergence,
        frozen_sublattices="",
    )

    weighting: dict[str, Any] | None = {
        "scheme": "balance_by_comp",
        "alpha": 1.0,
    }

    ce_spec = EnsureCEFromMixturesSpec(
        prototype_spec=prototype_spec,
        supercell_diag=supercell_diag,
        mixtures=tuple(mixtures),
        seed=seed,
        calc_spec=base_calc_spec,
        basis_spec={"basis": "sinusoid", "cutoffs": cutoffs},
        regularization={"type": "ridge", "alpha": 1e-3, "l1_ratio": 0.5},
        category=category,
        weighting=weighting,
    )

    spec = EnsureDoptCESpec(
        ce_spec=ce_spec,
        endpoints=endpoints,
        wl_bin_width=0.1,
        wl_steps_to_run=100_000,
        wl_samples_per_bin=50,
        wl_step_type="swap",
        wl_check_period=5_000,
        wl_update_period=1,
        wl_seed=0,
        reject_cross_sublattice_swaps=True,
        calc_spec=final_calc_spec,
        budget=budget,
        category=category,
        min_partial_frac=1.0,
    )

    label = f"A={a_cation} B={b_cation}"
    return resolve_or_launch(
        spec, label, prototype, supercell_diag,
        launch=launch, min_partial_frac=min_partial_frac,
    )
