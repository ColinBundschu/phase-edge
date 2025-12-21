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


def garnet_ce(
    a_cation: str,
    b_cation: str,
    c_cation: str,
    convergence: float,
    *,
    category: str,
    launch: bool = False,
) -> str | None:
    """
    Look up (and optionally launch) the final CE for a garnet A3B2C3O12 system
    with the hard-coded D-opt CE settings used in the provided CLI example.

    Mapping to prototype/spec:
      - A-site (dodecahedral)  -> J0 in the prototype tag
      - B-site (octahedral)    -> Es sublattice
      - C-site (tetrahedral)   -> Fm sublattice

    Parameters
    ----------
    a_cation : str
        Element on the A (dodecahedral) sublattice (J0).
    b_cation : str
        Element on the B (octahedral) sublattice (Es).
    c_cation : str
        Element on the C (tetrahedral) sublattice (Fm).
    convergence : float
        Max force convergence criterion in eV/Å for both base and final CalcSpec.
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
    # Example CLI: --prototype garnet_J0Y_Q0O --a 12.0
    # Here we generalize J0 to the requested A-site element.
    prototype = f"garnet_J0{a_cation}_Q0O"
    proto_params: dict[str, float] = {"a": 12.0}
    supercell_diag: tuple[int, int, int] = (1, 1, 1)
    seed = 42

    prototype_spec = PrototypeSpec(
        prototype=prototype,
        params=proto_params,
    )

    # --- Endpoint composition (canonicalized the same way as CLI) ---
    # Endpoint in your example:
    #   Es:{Al:8}, Fm:{Ga:12}
    # Generalized:
    endpoint_cm: dict[str, dict[str, int]] = {
        "Es": {b_cation: 8},
        "Fm": {c_cation: 12},
    }
    endpoints = sorted_composition_maps([endpoint_cm])

    # --- Mixture line: hard coded logic from your CLI path ---
    # Example pattern (Al= B, Ga= C):
    #   Es: Al^{8-k} Ga^k
    #   Fm: Al^k   Ga^{12-k}   for k = 1..8
    mixtures: list[Mixture] = []
    for k in range(1, 9):
        es_counts: dict[str, int] = {}
        fm_counts: dict[str, int] = {}

        # Es counts
        if 8 - k > 0:
            es_counts[b_cation] = 8 - k
        if k > 0:
            es_counts[c_cation] = es_counts.get(c_cation, 0) + k

        # Fm counts
        if k > 0:
            fm_counts[b_cation] = fm_counts.get(b_cation, 0) + k
        if 12 - k > 0:
            fm_counts[c_cation] = fm_counts.get(c_cation, 0) + (12 - k)

        cm = {"Es": es_counts, "Fm": fm_counts}
        mixtures.append(Mixture(composition_map=cm, K=30, seed=seed))

    # Add endpoint(s) as K=1 mixtures, just like the CLI helper does.
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

    # --- CE hyperparameters (hard-coded from your garnet command) ---
    cutoffs = parse_cutoffs_arg("2:8,3:7,4:6")

    base_calc_spec = CalcSpec(
        calculator=CalcType("MACE-MPA-0"),
        relax_type=RelaxType("full"),
        spin_type=SpinType("nonmagnetic"),
        max_force_eV_per_A=convergence,
        frozen_sublattices="",
    )

    final_calc_spec = CalcSpec(
        calculator=CalcType("vasp-mp-24"),
        relax_type=RelaxType("full"),
        spin_type=SpinType("nonmagnetic"),
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
        budget=250,
        category=category,
    )

    ce_key = spec.final_ce_key
    existing_ce = lookup_ce_by_key(ce_key)

    if not launch:
        # Pure “check” mode: only return an existing key, else None.
        result_key: str | None = ce_key if existing_ce is not None else None
        print(f"A={a_cation} B={b_cation} C={c_cation} ce_key={result_key}")
        return result_key

    # --- launch=True mode ---
    if existing_ce is None:
        # Hard-coded LaunchPad config path from your CLI.
        launchpad_path = "/home/cbu/fw_config/my_launchpad.yaml"

        j = ensure_dopt_ce(spec=spec)
        j.name = (
            f"ensure_dopt_ce::{prototype}::{supercell_diag}::{final_calc_spec.calculator}"
        )
        j.update_metadata({"_category": spec.category})

        wf = flow_to_workflow(j)
        lp = LaunchPad.from_file(launchpad_path)
        _wf_id = lp.add_wf(wf)
        # Intentionally not printing wf_id; you only asked for A/B/C and ce_key.

    # After launch (or if it already existed), we know the CE key.
    print(f"A={a_cation} B={b_cation} C={c_cation} ce_key={ce_key}")
    return ce_key
