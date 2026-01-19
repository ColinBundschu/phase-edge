import dataclasses
from typing import Any
from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.cli.common import parse_cutoffs_arg
from phaseedge.jobs.ensure_ce_from_mixtures import EnsureCEFromMixturesSpec
from phaseedge.jobs.ensure_dopt_ce import ensure_dopt_ce
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.schemas.calc_spec import CalcSpec, CalcType, RelaxType, SpinType
from phaseedge.schemas.ensure_dopt_ce_spec import EnsureDoptCESpec
from phaseedge.schemas.mixture import Mixture, sorted_composition_maps
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.science.random_configs import validate_counts_for_sublattices


def double_perovskite_ce(
    a_cation: str,
    bprime_cation: str,
    bdblprime_cation: str,
    convergence: float,
    *,
    category: str,
    spin_type: str,
    base_convergence: float = 0.05,
    launch: bool = False,
    partial: bool = False,
) -> str | None:
    """
    Look up (and optionally launch) the final CE for a double perovskite
    A2B'B''O6 system with the hard-coded D-opt CE settings used in the provided CLI example.

    Mapping to prototype/spec:
      - A-site (inactive)       -> J0 in the prototype tag
      - B'-site (active)        -> Es sublattice
      - B''-site (active)       -> Fm sublattice

    Parameters
    ----------
    a_cation : str
        Element on the A site (J0).
    bprime_cation : str
        Element on the B' active sublattice (Es).
    bdblprime_cation : str
        Element on the B'' active sublattice (Fm).
    convergence : float
        Max force convergence criterion in eV/Å for the final CalcSpec.
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
    prototype = f"doubleperovskite_J0{a_cation}_Q0O"
    proto_params: dict[str, float] = {"a": 7.86}
    supercell_diag: tuple[int, int, int] = (2, 2, 2)
    seed = 42

    prototype_spec = PrototypeSpec(
        prototype=prototype,
        params=proto_params,
    )

    # --- Endpoints (two, as in your CLI) ---
    endpoint_1: dict[str, dict[str, int]] = {
        "Es": {bprime_cation: 8},
        "Fm": {bdblprime_cation: 8},
    }
    endpoint_2: dict[str, dict[str, int]] = {
        "Es": {bdblprime_cation: 8},
        "Fm": {bprime_cation: 8},
    }
    endpoints = sorted_composition_maps([endpoint_1, endpoint_2])

    # --- Mixtures: k = 1..7 (K=50), plus endpoints as K=1 ---
    # Pattern from your CLI (Al/Ta example):
    #   Es: Al^{8-k} Ta^k
    #   Fm: Al^k   Ta^{8-k}
    mixtures: list[Mixture] = []
    for k in range(1, 8):
        es_counts: dict[str, int] = {}
        fm_counts: dict[str, int] = {}

        # Es: B'^(8-k) + B''^k
        if 8 - k > 0:
            es_counts[bprime_cation] = 8 - k
        if k > 0:
            es_counts[bdblprime_cation] = es_counts.get(bdblprime_cation, 0) + k

        # Fm: B'^k + B''^(8-k)
        if k > 0:
            fm_counts[bprime_cation] = fm_counts.get(bprime_cation, 0) + k
        if 8 - k > 0:
            fm_counts[bdblprime_cation] = fm_counts.get(bdblprime_cation, 0) + (8 - k)

        cm = {"Es": es_counts, "Fm": fm_counts}
        mixtures.append(Mixture(composition_map=cm, K=50, seed=seed))

    for ep in endpoints:
        mixtures.append(Mixture(composition_map=ep, K=1, seed=0))

    # --- Optional early validation (same as CLI) ---
    primitive_cell = prototype_spec.primitive_cell
    for mixture in mixtures:
        validate_counts_for_sublattices(
            primitive_cell=primitive_cell,
            supercell_diag=supercell_diag,
            composition_map=mixture.composition_map,
        )

    # --- CE hyperparameters (hard-coded from your command) ---
    cutoffs = parse_cutoffs_arg("2:12,3:12,4:9")

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
        wl_steps_to_run=1_000_000,
        wl_samples_per_bin=20,
        wl_step_type="swap",
        wl_check_period=5_000,
        wl_update_period=1,
        wl_seed=0,
        reject_cross_sublattice_swaps=True,
        calc_spec=final_calc_spec,
        budget=200,
        category=category,
        allow_partial=False,
    )

    existing_ce = lookup_ce_by_key(spec.final_ce_key)
    partial_spec = dataclasses.replace(spec, allow_partial=True)
    existing_partial_ce = lookup_ce_by_key(partial_spec.final_ce_key)

    if not launch:
        if existing_ce:
            print(f"A={a_cation} Bp={bprime_cation} Bpp={bdblprime_cation} ce_key={spec.final_ce_key}")
            return spec.final_ce_key
        if partial and existing_partial_ce:
            print(f"A={a_cation} Bp={bprime_cation} Bpp={bdblprime_cation} partial CE key={partial_spec.final_ce_key}")
            return partial_spec.final_ce_key
        return None

    if existing_ce:
        print("Final complete CE already exists, no workflow submitted.")
        print(f"A={a_cation} Bp={bprime_cation} Bpp={bdblprime_cation} ce_key={spec.final_ce_key}")
        return spec.final_ce_key
    
    if partial and existing_partial_ce:
        print("Final partial CE already exists, no workflow submitted.")
        print(f"A={a_cation} Bp={bprime_cation} Bpp={bdblprime_cation} partial CE key={partial_spec.final_ce_key}")
        return partial_spec.final_ce_key
    
    if partial:
        spec = partial_spec
    launchpad_path = "/home/cbu/fw_config/my_launchpad.yaml"

    j = ensure_dopt_ce(spec=spec)
    j.name = (
        f"ensure_dopt_ce::{prototype}::{supercell_diag}::{final_calc_spec.calculator}"
    )
    j.update_metadata({"_category": spec.category})

    wf = flow_to_workflow(j)
    lp = LaunchPad.from_file(launchpad_path)
    _wf_id = lp.add_wf(wf)
    print(f"A={a_cation} Bp={bprime_cation} Bpp={bdblprime_cation} ce_key={spec.final_ce_key}")
    return spec.final_ce_key
