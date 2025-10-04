import argparse
from typing import Any

from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad

from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.science.prototypes import PrototypeName, make_prototype
from phaseedge.science.random_configs import validate_counts_for_sublattices
from phaseedge.utils.keys import compute_ce_key
from phaseedge.cli.common import parse_cutoffs_arg, parse_mix_item
from phaseedge.jobs.ensure_ce_from_mixtures import ensure_ce_from_mixtures
from phaseedge.schemas.ensure_ce_from_mixtures_spec import EnsureCEFromMixturesSpec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-ensure-ce",
        description="Ensure a CE over compositions (deterministic snapshots → MACE relax cache → train CE → store).",
    )
    p.add_argument("--launchpad", required=True)

    # System / snapshot identity (prototype-only)
    p.add_argument("--prototype", required=True, choices=[p.value for p in PrototypeName])
    p.add_argument("--a", required=True, type=float, help="Prototype lattice parameter (e.g., 4.3).")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    p.add_argument("--inactive-cation", dest="inactive_cation", default=None, help="Fixed A-site for double_perovskite (e.g., 'Sr').")

    # Composition input (repeatable)
    p.add_argument("--mix", action="append", required=True, help="Composition mixture: 'composition_map=...;K=...;seed=...'")
    p.add_argument("--seed", type=int, default=0, help="Global/default seed")

    # Relax/engine for training energies
    p.add_argument("--model", default="MACE-MPA-0")
    p.add_argument("--relax-cell", action="store_true")

    # CE hyperparameters
    p.add_argument("--basis", default="sinusoid", help="Basis family understood by smol (default: sinusoid)")
    p.add_argument("--cutoffs", default="1:100,2:10,3:8,4:6",
                   help="Comma sep. per-body cutoffs, e.g. '1:100,2:10,3:8,4:6'")
    p.add_argument("--reg-type", choices=["ols", "ridge", "lasso", "elasticnet"], default="ols")
    p.add_argument("--alpha", type=float, default=1e-6, help="Regularization strength (ignored for OLS)")
    p.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet l1_ratio (ignored otherwise)")

    # Weighting controls
    p.add_argument("--balance-by-comp", action="store_true",
                   help="Reweight samples inversely by composition count (mean weight normalized to 1).")
    p.add_argument("--weight-alpha", type=float, default=1.0,
                   help="Exponent alpha for inverse-count weighting (w ~ n_g^(-alpha)). Default: 1.0")

    # Routing
    p.add_argument("--category", default="gpu")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    cutoffs = parse_cutoffs_arg(args.cutoffs)

    # Build composition list
    mixtures = tuple([parse_mix_item(s) for s in args.mix])

    proto_params: dict[str, Any] = {"a": float(args.a)}
    if args.prototype == PrototypeName.DOUBLE_PEROVSKITE.value:
        if not args.inactive_cation:
            raise SystemExit("--inactive-cation is required when --prototype=double_perovskite")
        proto_params["inactive_cation"] = str(args.inactive_cation)
    elif args.inactive_cation:
        raise SystemExit("--inactive-cation is only valid when --prototype=double_perovskite")
    
    # Optional early validation
    conv = make_prototype(PrototypeName(args.prototype), **proto_params)
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

    # Build the idempotent ensure job (mixture spec retained) and submit
    spec = EnsureCEFromMixturesSpec(
        prototype=PrototypeName(args.prototype),
        prototype_params=proto_params,
        supercell_diag=tuple(args.supercell),
        mixtures=mixtures,
        seed=int(args.seed),
        model=args.model,
        relax_cell=bool(args.relax_cell),
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        category=args.category,
        weighting=weighting,
    )

    existing_ce = lookup_ce_by_key(spec.ce_key)
    if existing_ce:
        print("CE already exists for ce_key:", spec.ce_key)
    else:
        j = ensure_ce_from_mixtures(spec)
        j.name = "ensure_ce"
        j.metadata = {**(j.metadata or {}), "_category": args.category}

        flow = Flow([j], name="Ensure CE (compositions)")
        wf = flow_to_workflow(flow)
        for fw in wf.fws:
            fw.spec = {**(fw.spec or {}), "_category": args.category}

        lp = LaunchPad.from_file(args.launchpad)
        wf_id = lp.add_wf(wf)

        print("Submitted workflow:", wf_id)
    print(
        {
            "ce_key": spec.ce_key,
            "prototype": args.prototype,
            "a": args.a,
            "supercell": tuple(args.supercell),
            "source": spec.source,
            "model": args.model,
            "relax_cell": bool(args.relax_cell),
            "basis": args.basis,
            "cutoffs": cutoffs,
            "reg_type": args.reg_type,
            "alpha": args.alpha,
            "l1_ratio": args.l1_ratio,
            "weighting": weighting,
            "category": args.category,
        }
    )
