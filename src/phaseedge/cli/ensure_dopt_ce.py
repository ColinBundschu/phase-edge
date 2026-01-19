import argparse
import dataclasses
from typing import Any
import json

from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.cli.common import parse_composition_map, parse_cutoffs_arg, parse_mix_item
from phaseedge.jobs.ensure_ce_from_mixtures import EnsureCEFromMixturesSpec
from phaseedge.jobs.ensure_dopt_ce import ensure_dopt_ce
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.schemas.calc_spec import CalcSpec, CalcType, RelaxType, SpinType
from phaseedge.schemas.ensure_dopt_ce_spec import EnsureDoptCESpec
from phaseedge.schemas.mixture import Mixture, composition_map_sig, sorted_composition_maps
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.science.random_configs import validate_counts_for_sublattices


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-ensure-dopt-ce",
        description=(
            "Ensure a CE, run WL on non-endpoint compositions, "
            "select a D-optimal basis to a target budget, relax, and train a final CE."
        ),
    )
    p.add_argument("--launchpad", required=True)

    # System / snapshot identity (prototype-only)
    p.add_argument("--prototype", required=True)
    p.add_argument("--a", required=True, type=float, help="Prototype lattice parameter (e.g., 4.3).")
    p.add_argument("--c", required=False, type=float, help="Prototype lattice parameter c (e.g., 12.3).")
    p.add_argument("--u", required=False, type=float, help="Prototype-specific parameter (e.g., oxygen u for spinels).")
    p.add_argument("--x", required=False, type=float, help="Prototype-specific parameter (e.g., anion position for double perovskites).")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))

    # Composition input
    p.add_argument("--mix", action="append", required=True, help="Composition mixture: 'composition_map=...;K=...;seed=...'")
    p.add_argument("--endpoint", action="append", default=[], help="Endpoint composition: 'composition_map=...'. No K/seed allowed. (repeatable)")
    p.add_argument("--seed", type=int, default=0, help="Default seed for CE mixture elements missing 'seed'.")
    p.add_argument("--reject-cross-sublattice-swaps", action="store_true", help="Reject WL swap moves that cross sublattices.")

    # Relax/engine for training energies
    p.add_argument("--base-calculator", required=True, choices=[r.value for r in CalcType])
    p.add_argument("--final-calculator", required=True, choices=[r.value for r in CalcType])
    p.add_argument("--relax-type", required=True, choices=[r.value for r in RelaxType])
    p.add_argument("--spin-type", required=True, choices=[r.value for r in SpinType])
    p.add_argument("--base-max-force-eV-per-A", type=float, required=True, help="Maximum force convergence criterion in eV/Å.")
    p.add_argument("--final-max-force-eV-per-A", type=float, required=True, help="Maximum force convergence criterion in eV/Å.")
    p.add_argument("--frozen-sublattices", default="")

    # CE hyperparameters
    p.add_argument("--basis", default="sinusoid")
    p.add_argument("--cutoffs", default="1:100,2:10,3:8,4:6")
    p.add_argument("--reg-type", choices=["ols", "ridge", "lasso", "elasticnet"], default="ols")
    p.add_argument("--alpha", type=float, default=1e-6)
    p.add_argument("--l1-ratio", type=float, default=0.5)

    # Weighting controls
    p.add_argument("--balance-by-comp", action="store_true")
    p.add_argument("--weight-alpha", type=float, default=1.0)

    # Unified routing
    p.add_argument("--category", help="FireWorks category for ALL jobs.")

    # WL policy / schedule
    p.add_argument("--wl-bin-width", required=True, type=float)
    p.add_argument("--steps-to-run", required=True, type=int, dest="steps_to_run")
    p.add_argument("--samples-per-bin", type=int, default=0)
    p.add_argument("--step-type", default="swap", choices=["swap"])
    p.add_argument("--check-period", type=int, default=5_000)
    p.add_argument("--update-period", type=int, default=1)
    p.add_argument("--wl-seed", type=int, default=0)

    # D-optimal budget
    p.add_argument("--budget", required=True, type=int)

    # Output
    p.add_argument("--json", action="store_true")

    p.add_argument("--partial", action="store_true", help="Allow partial D-optimal basis if some calculations are unrelaxed.")

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    cutoffs = parse_cutoffs_arg(args.cutoffs)

    # Parse + canonicalize inputs
    proper_mixtures = [parse_mix_item(s) for s in args.mix]
    endpoints = sorted_composition_maps([parse_composition_map(s) for s in args.endpoint])
    mixtures = (*proper_mixtures, *(Mixture(composition_map=ep, K=1, seed=0) for ep in endpoints))

    
    # Optional early validation
    proto_params: dict[str, Any] = {"a": float(args.a)}
    if args.c is not None:
        proto_params["c"] = float(args.c)
    if args.u is not None:
        proto_params["u"] = float(args.u)
    if args.x is not None:
        proto_params["x"] = float(args.x)
    prototype_spec = PrototypeSpec(prototype=args.prototype, params=proto_params)
    for mixture in mixtures:
        validate_counts_for_sublattices(
            primitive_cell=prototype_spec.primitive_cell,
            supercell_diag=tuple(args.supercell),
            composition_map=mixture.composition_map,
        )

    # Weighting config payload
    weighting: dict[str, Any] | None = (
        {"scheme": "balance_by_comp", "alpha": float(args.weight_alpha)} if args.balance_by_comp else None
    )

    supercell_x, supercell_y, supercell_z = tuple(int(x) for x in args.supercell)
    supercell_diag = (supercell_x, supercell_y, supercell_z)

    base_calc_spec = CalcSpec(
        calculator=CalcType(args.base_calculator),
        relax_type=RelaxType(args.relax_type),
        spin_type=SpinType(args.spin_type),
        max_force_eV_per_A=args.base_max_force_eV_per_A,
        frozen_sublattices=args.frozen_sublattices,
    )

    final_calc_spec = CalcSpec(
        calculator=CalcType(args.final_calculator),
        relax_type=RelaxType(args.relax_type),
        spin_type=SpinType(args.spin_type),
        max_force_eV_per_A=args.final_max_force_eV_per_A,
        frozen_sublattices=args.frozen_sublattices,
    )

    # Build CE spec
    ce_spec = EnsureCEFromMixturesSpec(
        prototype_spec=prototype_spec,
        supercell_diag=supercell_diag,
        mixtures=mixtures,
        seed=int(args.seed),
        calc_spec=base_calc_spec,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        category=args.category,
        weighting=weighting,
    )

    # Compose the master ensure job
    spec = EnsureDoptCESpec(
        ce_spec=ce_spec,
        endpoints=endpoints,
        wl_bin_width=float(args.wl_bin_width),
        wl_steps_to_run=int(args.steps_to_run),
        wl_samples_per_bin=int(args.samples_per_bin),
        wl_step_type=str(args.step_type),
        wl_check_period=int(args.check_period),
        wl_update_period=int(args.update_period),
        wl_seed=int(args.wl_seed),
        reject_cross_sublattice_swaps=bool(args.reject_cross_sublattice_swaps),
        calc_spec=final_calc_spec,
        budget=int(args.budget),
        category=str(args.category),
        allow_partial=False,
    )

    planned_wl_runs: list[dict[str, Any]] = [{
        "initial_comp_map": composition_map_sig(sampler_spec.initial_comp_map),
        "wl_key": sampler_spec.wl_key,
    } for sampler_spec in spec.wl_sampler_specs]

    # Build + submit workflow
    # Early exit keying
    payload: dict[str, Any] = {
        "seed_size_estimate": len(endpoints) + len(planned_wl_runs),
    } | spec.as_dict()

    partial_spec = dataclasses.replace(spec, allow_partial=True)
    if lookup_ce_by_key(spec.final_ce_key):
        print("Final complete CE already exists, no workflow submitted.")
    elif args.partial and lookup_ce_by_key(partial_spec.final_ce_key):
        print("Final partial CE already exists, no workflow submitted.")
    else:
        if args.partial:
            spec = partial_spec
        j = ensure_dopt_ce(spec=spec)
        j.name = f"ensure_dopt_ce::{args.prototype}::{tuple(args.supercell)}::{args.final_calculator}"
        j.update_metadata({"_category": spec.category})

        wf = flow_to_workflow(j)
        lp = LaunchPad.from_file(args.launchpad)
        wf_id = lp.add_wf(wf)

        payload: dict[str, Any] = payload | {
            "submitted_workflow_id": wf_id,
            "planned_wl_runs": planned_wl_runs,
        }

        print("Submitted workflow:", wf_id)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print({
            "base_ce_key": spec.ce_spec.ce_key,
            "final_ce_key": spec.final_ce_key,
            } | {
            k: payload["ce_spec"][k] for k in (
                "prototype_spec", "supercell_diag",
                "regularization", "weighting", "category", "basis_spec",
            )
        })
        print("Planned WL chains:")
        for rec in planned_wl_runs:
            print(f"  {rec['initial_comp_map']}  wl_key={rec['wl_key']}")
        print({
            "initial_calc_spec": base_calc_spec.as_dict(),
        })
        print({
            "final_calc_spec": final_calc_spec.as_dict(),
            "budget": spec.budget,
            "seed_size_estimate": payload["seed_size_estimate"],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
