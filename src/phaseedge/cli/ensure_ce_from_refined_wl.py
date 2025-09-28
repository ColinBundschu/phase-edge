import argparse
from typing import Any
import json

from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.cli.common import parse_composition_map, parse_cutoffs_arg, parse_mix_item
from phaseedge.jobs.ensure_ce_from_mixtures import EnsureCEFromMixturesSpec
from phaseedge.jobs.ensure_ce_from_refined_wl import ensure_ce_from_refined_wl
from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.schemas.ensure_ce_from_refined_wl_spec import EnsureCEFromRefinedWLSpec
from phaseedge.schemas.mixture import Mixture, counts_sig, sorted_composition_maps
from phaseedge.science.prototypes import PrototypeName, make_prototype
from phaseedge.science.random_configs import validate_counts_for_sublattices
from phaseedge.science.refine_wl import RefineStrategy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-ensure-ce-from-refined-wl",
        description=(
            "Ensure a CE, run WL on non-endpoint compositions, optionally refine WL samples, "
            "select a D-optimal basis to a target budget, relax, and train a final CE."
        ),
    )
    p.add_argument("--launchpad", required=True)

    # System / snapshot identity (prototype-only)
    p.add_argument("--prototype", required=True, choices=[p.value for p in PrototypeName])
    p.add_argument("--a", required=True, type=float)
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))

    # Composition input
    p.add_argument("--mix", action="append", required=True, help="Composition mixture: 'composition_map=...;K=...;seed=...'")
    p.add_argument("--endpoint", action="append", default=[], help="Endpoint composition: 'composition_map=...'. No K/seed allowed. (repeatable)")
    p.add_argument("--seed", type=int, default=0, help="Default seed for CE mixture elements missing 'seed'.")

    p.add_argument(
        "--sl-comp-map",
        required=True,
        help='Canonical map to identify the sublattices (e.g., Es:{Mg:16},Fm:{Al:32}).',
    )

    # Initial CE (engine for training energies of the initial dataset)
    p.add_argument("--model", default="MACE-MPA-0")
    p.add_argument("--relax-cell", action="store_true")
    p.add_argument("--dtype", default="float64")

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
    p.add_argument("--category", default="gpu", help="FireWorks category for ALL jobs.")

    # WL policy / schedule
    p.add_argument("--wl-bin-width", required=True, type=float)
    p.add_argument("--steps-to-run", required=True, type=int, dest="steps_to_run")
    p.add_argument("--samples-per-bin", type=int, default=0)
    p.add_argument("--step-type", default="swap", choices=["swap"])
    p.add_argument("--check-period", type=int, default=5_000)
    p.add_argument("--update-period", type=int, default=1)
    p.add_argument("--wl-seed", type=int, default=0)

    # Refinement policy (per WL chain). n_total == 0 means "use all".
    p.add_argument("--refine-n-total", type=int, default=25)
    p.add_argument("--refine-per-bin-cap", type=int, default=5)
    p.add_argument("--refine-strategy",
                   choices=["energy_spread", "energy_stratified", "hash_round_robin"],
                   default="energy_spread")

    # Final CE training (for relaxations + final ce_key)
    p.add_argument("--train-model", required=True)
    p.add_argument("--train-relax-cell", action="store_true")
    p.add_argument("--train-dtype", default="float64")

    # D-optimal budget
    p.add_argument("--budget", required=True, type=int)

    # Output
    p.add_argument("--json", action="store_true")
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    cutoffs = parse_cutoffs_arg(args.cutoffs)

    # Parse + canonicalize inputs
    proper_mixtures = [parse_mix_item(s) for s in args.mix]
    endpoints = sorted_composition_maps([parse_composition_map(s) for s in args.endpoint])
    mixtures = (*proper_mixtures, *(Mixture(composition_map=ep, K=1, seed=0) for ep in endpoints))
    sl_comp_map = parse_composition_map(args.sl_comp_map)

    # Optional early validation
    conv = make_prototype(args.prototype, a=args.a)
    for mixture in mixtures:
        validate_counts_for_sublattices(
            conv_cell=conv,
            supercell_diag=tuple(args.supercell),
            composition_map=mixture.composition_map,
        )

    # Weighting config payload
    weighting: dict[str, Any] | None = (
        {"scheme": "balance_by_comp", "alpha": float(args.weight_alpha)} if args.balance_by_comp else None
    )

    supercell_x, supercell_y, supercell_z = tuple(int(x) for x in args.supercell)
    supercell_diag = (supercell_x, supercell_y, supercell_z)

    # Build CE spec
    ce_spec = EnsureCEFromMixturesSpec(
        prototype=PrototypeName(args.prototype),
        prototype_params={"a": args.a},
        supercell_diag=supercell_diag,
        mixtures=mixtures,
        seed=int(args.seed),
        model=args.model,
        relax_cell=bool(args.relax_cell),
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        category=args.category,
        weighting=weighting,
    )

    # Compose the master ensure job
    spec = EnsureCEFromRefinedWLSpec(
        ce_spec=ce_spec,
        endpoints=endpoints,
        wl_bin_width=float(args.wl_bin_width),
        wl_steps_to_run=int(args.steps_to_run),
        wl_samples_per_bin=int(args.samples_per_bin),
        wl_step_type=str(args.step_type),
        wl_check_period=int(args.check_period),
        wl_update_period=int(args.update_period),
        wl_seed=int(args.wl_seed),
        sl_comp_map=sl_comp_map,
        refine_n_total=int(args.refine_n_total),
        refine_per_bin_cap=int(args.refine_per_bin_cap),
        refine_strategy=RefineStrategy(args.refine_strategy),
        train_model=str(args.train_model),
        train_relax_cell=bool(args.train_relax_cell),
        train_dtype=str(args.train_dtype),
        budget=int(args.budget),
        category=str(args.category),
    )

    planned_wl_runs: list[dict[str, Any]] = [{
        "counts_sig": counts_sig(composition_counts),
        "wl_key": wl_key,
    } for wl_key, composition_counts in spec.wl_key_composition_pairs]

    # Build + submit workflow
    # Early exit keying
    payload: dict[str, Any] = {
        "seed_size_estimate": len(endpoints) + len(planned_wl_runs),
    } | spec.as_dict()

    existing_ce = lookup_ce_by_key(spec.final_ce_key)
    if existing_ce is None:
        j = ensure_ce_from_refined_wl(spec=spec)
        j.name = "ensure_ce_from_refined_wl"
        j.update_metadata({"_category": spec.category})

        wf = flow_to_workflow(j)
        lp = LaunchPad.from_file(args.launchpad)
        wf_id = lp.add_wf(wf)

        payload: dict[str, Any] = payload | {
            "submitted_workflow_id": wf_id,
            "planned_wl_runs": planned_wl_runs,
        }

        print("Submitted workflow:", wf_id)
    else:
        print("Final CE already exists, no workflow submitted.")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print({
            "base_ce_key": spec.ce_spec.ce_key,
            "final_ce_key": spec.final_ce_key,
            } | {
            k: payload["ce_spec"][k] for k in (
                "prototype", "prototype_params", "supercell_diag",
                "regularization", "weighting", "category", "basis_spec",
            )
        })
        print("Planned WL chains:")
        for rec in planned_wl_runs:
            print(f"  {rec['counts_sig']:>18}  wl_key={rec['wl_key']}")
        print({
            "initial_training": {
                "model": spec.ce_spec.model,
                "relax_cell": spec.ce_spec.relax_cell,
                "dtype": spec.ce_spec.dtype,
            },
        })
        print({
            "final_training": {
                "model": spec.train_model,
                "relax_cell": spec.train_relax_cell,
                "dtype": spec.train_dtype,
            },
            "budget": spec.budget,
            "seed_size_estimate": payload["seed_size_estimate"],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
