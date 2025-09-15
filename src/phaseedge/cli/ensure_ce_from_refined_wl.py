import argparse
from typing import Any, Literal, cast
import json

from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.cli.common import parse_composition_map, parse_cutoffs_arg, parse_mix_item
from phaseedge.jobs.ensure_ce import CEEnsureMixturesSpec
from phaseedge.jobs.ensure_ce_from_refined_wl import EnsureCEFromRefinedWLSpec, ensure_ce_from_refined_wl
from phaseedge.schemas.mixture import Mixture, counts_sig, sorted_composition_maps
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import validate_counts_for_sublattices


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
    p.add_argument("--prototype", required=True, choices=["rocksalt"])
    p.add_argument("--a", required=True, type=float)
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))

    # Composition input
    p.add_argument("--mix", action="append", required=True, help="Composition mixture: 'composition_map=...;K=...;seed=...'")
    p.add_argument("--endpoint", action="append", default=[], help="Endpoint composition: 'composition_map=...'. No K/seed allowed. (repeatable)")
    p.add_argument("--seed", type=int, default=0, help="Default seed for CE mixture elements missing 'seed'.")

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
    ce_spec = CEEnsureMixturesSpec(
        prototype=args.prototype,
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
    wl_spec = EnsureCEFromRefinedWLSpec(
        ce_spec=ce_spec,
        endpoints=endpoints,
        wl_bin_width=float(args.wl_bin_width),
        wl_steps_to_run=int(args.steps_to_run),
        wl_samples_per_bin=int(args.samples_per_bin),
        wl_step_type=str(args.step_type),
        wl_check_period=int(args.check_period),
        wl_update_period=int(args.update_period),
        wl_seed=int(args.wl_seed),
        refine_n_total=int(args.refine_n_total),
        refine_per_bin_cap=int(args.refine_per_bin_cap),
        refine_strategy=cast(Literal["energy_spread", "energy_stratified", "hash_round_robin"], args.refine_strategy),
        train_model=str(args.train_model),
        train_relax_cell=bool(args.train_relax_cell),
        train_dtype=str(args.train_dtype),
        budget=int(args.budget),
        category=str(args.category),
    )

    planned_wl_runs: list[dict[str, Any]] = [{
        "counts_sig": counts_sig(composition_counts),
        "wl_key": wl_key,
    } for wl_key, composition_counts in wl_spec.wl_key_composition_pairs]

    # INTENT block for refined CE (submit-time determinism)
    # wl_policy = {
    #     "bin_width": float(args.wl_bin_width),
    #     "step_type": str(args.step_type),
    #     "check_period": int(args.check_period),
    #     "update_period": int(args.update_period),
    #     "seed": int(args.wl_seed),
    # }
    # ensure_policy = {
    #     "steps_to_run": int(args.steps_to_run),
    #     "samples_per_bin": int(args.samples_per_bin),
    # }
    # refine_mode = "all" if int(args.refine_n_total) == 0 else "refine"
    # refine_block = {
    #     "mode": refine_mode,
    #     "n_total": (None if refine_mode == "all" else int(args.refine_n_total)),
    #     "per_bin_cap": int(args.refine_per_bin_cap),
    #     "strategy": str(args.refine_strategy),
    # }
    # dopt_block = {
    #     "budget": int(args.budget),
    #     "ridge": float(1e-10),
    #     "tie_breaker": "bin_then_hash",
    # }
    # intent_source = {
    #     "type": "wl_refined_intent",
    #     "base_ce_key": ce_spec.ce_key,
    #     "endpoints": endpoints,  # already canonicalized
    #     "wl_policy": wl_policy,
    #     "ensure": ensure_policy,
    #     "refine": refine_block,
    #     "dopt": dopt_block,
    #     "versions": {
    #         "refine": "refine-wl-v1",
    #         "dopt": "dopt-greedy-sm-v1",
    #         "sampler": "wl-grid-v1",
    #     },
    # }

    # # Compute refined (intent-based) CE key at submit time
    # refined_intent_ce_key = compute_ce_key(
    #     prototype=args.prototype,
    #     prototype_params={"a": float(args.a)},
    #     supercell_diag=supercell_diag,
    #     sources=[intent_source],
    #     algo_version="refined-wl-dopt-v2",
    #     model=str(args.train_model),
    #     relax_cell=bool(args.train_relax_cell),
    #     dtype=str(args.train_dtype),
    #     basis_spec={"basis": args.basis, "cutoffs": cutoffs},
    #     regularization={"type": args.reg_type, "alpha": float(args.alpha), "l1_ratio": float(args.l1_ratio)},
    #     weighting=weighting,
    # )

    # Build + submit workflow
    j = ensure_ce_from_refined_wl(ensure_wl_spec=wl_spec)
    j.name = "ensure_ce_from_refined_wl"
    j.metadata = {**(j.metadata or {}), "_category": args.category}

    flow = Flow([j], name="Ensure CE from refined WL")
    wf = flow_to_workflow(flow)
    for fw in wf.fws:
        fw.spec = {**(fw.spec or {}), "_category": args.category}

    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)

    payload: dict[str, Any] = {
        "submitted_workflow_id": wf_id,
        "planned_wl_runs": planned_wl_runs,
        "seed_size_estimate": len(endpoints) + len(planned_wl_runs),
    } | wl_spec.as_dict()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print("Submitted workflow:", wf_id)
        print({
            "base_ce_key": wl_spec.ce_spec.ce_key,
            "final_ce_key": wl_spec.final_ce_key,
            } | {
            k: payload["ce_spec"][k] for k in (
                "prototype", "prototype_params", "supercell_diag",
                "regularization", "weighting", "category", "basis_spec",
            )
        })
        if planned_wl_runs:
            print("Planned WL chains:")
            for rec in planned_wl_runs:
                print(f"  {rec['counts_sig']:>18}  wl_key={rec['wl_key']}")
        else:
            print("Planned WL chains: []")
        print({
            "initial_training": {
                "model": wl_spec.ce_spec.model,
                "relax_cell": wl_spec.ce_spec.relax_cell,
                "dtype": wl_spec.ce_spec.dtype,
            },
        })
        print({
            "final_training": {
                "model": wl_spec.train_model,
                "relax_cell": wl_spec.train_relax_cell,
                "dtype": wl_spec.train_dtype,
            },
            "budget": wl_spec.budget,
            "seed_size_estimate": payload["seed_size_estimate"],
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
