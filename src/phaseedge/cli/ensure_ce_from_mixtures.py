import argparse
from typing import Any

from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad

from phaseedge.jobs.store_ce_model import lookup_ce_by_key
from phaseedge.schemas.calc_spec import CalcSpec, CalcType, RelaxType, SpinType
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.science.random_configs import validate_counts_for_sublattices
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
    p.add_argument("--prototype", required=True)
    p.add_argument("--a", required=True, type=float, help="Prototype lattice parameter (e.g., 4.3).")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))

    # Composition input (repeatable)
    p.add_argument("--mix", action="append", required=True, help="Composition mixture: 'composition_map=...;K=...;seed=...'")
    p.add_argument("--seed", type=int, default=0, help="Global/default seed")

    # Relax/engine for training energies
    p.add_argument("--calculator", required=True, choices=[r.value for r in CalcType])
    p.add_argument("--relax-type", required=True, choices=[r.value for r in RelaxType])
    p.add_argument("--spin-type", required=True, choices=[r.value for r in SpinType])
    p.add_argument("--max-force-eV-per-A", type=float, required=True, help="Maximum force convergence criterion in eV/Å.")
    p.add_argument("--frozen-sublattices", default="")

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
    
    # Optional early validation
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

    calc_spec = CalcSpec(
        calculator=CalcType(args.calculator),
        relax_type=RelaxType(args.relax_type),
        spin_type=SpinType(args.spin_type),
        max_force_eV_per_A=args.max_force_eV_per_A,
        frozen_sublattices=args.frozen_sublattices,
    )

    # Build the idempotent ensure job (mixture spec retained) and submit
    spec = EnsureCEFromMixturesSpec(
        prototype_spec=prototype_spec,
        supercell_diag=tuple(args.supercell),
        mixtures=mixtures,
        seed=int(args.seed),
        calc_spec=calc_spec,
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
        j.name = f"ensure_ce::{args.prototype}::{tuple(args.supercell)}::{args.calculator}"
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
            "calc_spec": calc_spec,
            "basis": args.basis,
            "cutoffs": cutoffs,
            "reg_type": args.reg_type,
            "alpha": args.alpha,
            "l1_ratio": args.l1_ratio,
            "weighting": weighting,
            "category": args.category,
        }
    )
