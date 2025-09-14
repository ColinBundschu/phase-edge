import argparse
from typing import Any

from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.cli.common import parse_composition_map, parse_cutoffs_arg, parse_mix_item
from phaseedge.jobs.ensure_ce import CEEnsureMixturesSpec
from phaseedge.jobs.ensure_wl_samples_from_ce import (
    EnsureWLSamplesFromCESpec,
    ensure_wl_samples_from_ce,
)
from phaseedge.schemas.mixture import Mixture, composition_counts_from_map, counts_sig
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import validate_counts_for_sublattices
from phaseedge.utils.keys import compute_wl_key


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-ensure-wl-from-ce",
        description="Ensure a CE and WL samples (WL for all non-endpoint compositions).",
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

    # Relax/engine for CE training energies
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

    # Output
    p.add_argument("--json", action="store_true")
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    cutoffs = parse_cutoffs_arg(args.cutoffs)

    # Parse + canonicalize inputs
    proper_mixtures = [parse_mix_item(s) for s in args.mix]
    endpoints = tuple(parse_composition_map(s) for s in args.endpoint)
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

    # Build CE spec
    ce_spec = CEEnsureMixturesSpec(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
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

    ce_key = ce_spec.ce_key

    # Precompute the WL keys for all unique NON-endpoint compositions (once per composition)
    seen_sigs: set[str] = {counts_sig(composition_counts_from_map(ep)) for ep in endpoints}
    planned_wl_runs: list[dict[str, Any]] = []
    for mixture in proper_mixtures:
        composition_counts = composition_counts_from_map(mixture.composition_map)
        sig = counts_sig(composition_counts)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        wl_key = compute_wl_key(
            ce_key=ce_key,
            bin_width=float(args.wl_bin_width),
            step_type=str(args.step_type),
            composition_counts=composition_counts,
            check_period=int(args.check_period),
            update_period=int(args.update_period),
            seed=int(args.wl_seed),
            algo_version="wl-grid-v1",
        )
        planned_wl_runs.append({
            "counts_sig": sig,
            "wl_key": wl_key,
        })

    # Compose the master ensure job
    master_spec = EnsureWLSamplesFromCESpec(
        ce_spec=ce_spec,
        endpoints=endpoints,
        wl_bin_width=float(args.wl_bin_width),
        wl_steps_to_run=int(args.steps_to_run),
        wl_samples_per_bin=int(args.samples_per_bin),
        wl_step_type=str(args.step_type),
        wl_check_period=int(args.check_period),
        wl_update_period=int(args.update_period),
        wl_seed=int(args.wl_seed),
        category=str(args.category),
    )

    j = ensure_wl_samples_from_ce(master_spec)
    j.name = "ensure_wl_samples_from_ce"
    j.metadata = {**(j.metadata or {}), "_category": args.category}  # wrapper FW

    flow = Flow([j], name="Ensure WL samples from CE")
    wf = flow_to_workflow(flow)
    for fw in wf.fws:
        fw.spec = {**(fw.spec or {}), "_category": args.category}  # wrapper FW gets same category

    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)

    payload: dict[str, Any] = {
        "submitted_workflow_id": wf_id,
        "ce_key": ce_key,
        "prototype": args.prototype,
        "a": float(args.a),
        "supercell": tuple(int(x) for x in args.supercell),
        "model": args.model,
        "relax_cell": bool(args.relax_cell),
        "dtype": args.dtype,
        "basis": args.basis,
        "cutoffs": cutoffs,
        "reg_type": args.reg_type,
        "alpha": float(args.alpha),
        "l1_ratio": float(args.l1_ratio),
        "weighting": weighting,
        "category": str(args.category),
        "endpoints": endpoints,
        "wl": {
            "bin_width": float(args.wl_bin_width),
            "steps_to_run": int(args.steps_to_run),
            "samples_per_bin": int(args.samples_per_bin),
            "step_type": str(args.step_type),
            "check_period": int(args.check_period),
            "update_period": int(args.update_period),
            "seed": int(args.wl_seed),
        },
        "planned_wl_runs": planned_wl_runs,
    }

    if args.json:
        import json as _json
        print(_json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print("Submitted workflow:", wf_id)
        print({
            k: payload[k] for k in (
                "ce_key", "prototype", "a", "supercell",
                "model", "relax_cell", "dtype", "basis", "cutoffs",
                "reg_type", "alpha", "l1_ratio", "weighting",
                "category", "endpoints",
            )
        })
        if planned_wl_runs:
            print("Planned WL chains:")
            for rec in planned_wl_runs:
                print(f"  {rec['counts_sig']:>18}  wl_key={rec['wl_key']}")
        else:
            print("Planned WL chains: []")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
