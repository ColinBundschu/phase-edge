import argparse
import re
from typing import Any, Set

from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad

from phaseedge.jobs.ensure_ce import (
    CEEnsureMixtureSpec,
    ensure_ce,
    MixtureElement,
)
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import validate_counts_for_sublattice
from phaseedge.utils.keys import compute_ce_key, CEKeySpec, SublatticeMixtureElement
from phaseedge.cli.common import parse_cutoffs_arg, parse_mix_item


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-ensure-ce",
        description="Ensure a CE over sublattice-formatted compositions "
                    "(deterministic snapshots → MACE relax cache → train CE → store).",
    )
    p.add_argument("--launchpad", required=True)

    # System / snapshot identity (prototype-only)
    p.add_argument("--prototype", required=True, choices=["rocksalt", "spinel"])
    p.add_argument("--a", required=True, type=float, help="Prototype lattice parameter (e.g., 4.3).")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))

    # Composition input (repeatable; sublattice-aware)
    p.add_argument(
        "--mix",
        action="append",
        required=True,
        help=("Mixture element with per-sublattice counts. "
              "Repeat this flag for multiple mixture elements. "
              "Example: --mix \"K=50;seed=123;"
              "subl=replace=Mg,counts=Fe:64,Mg:192;"
              "subl=replace=Al,counts=Al:256\""),
    )

    # Seeds
    p.add_argument("--seed", type=int, default=0, help="Default CV seed (does not fill mixture seeds).")

    # Relax/engine for training energies
    p.add_argument("--model", default="MACE-MPA-0")
    p.add_argument("--relax-cell", action="store_true")
    p.add_argument("--dtype", default="float64")

    # CE hyperparameters
    p.add_argument("--basis", default="sinusoid", help="Basis family understood by smol (default: sinusoid)")
    p.add_argument("--cutoffs", default="1:100,2:10,3:8,4:6",
                   help="Comma sep. per-body cutoffs, e.g. '1:100,2:10,3:8,4:6'")
    p.add_argument("--reg-type", choices=["ols", "ridge", "lasso", "elasticnet"], default="ols")
    p.add_argument("--alpha", type=float, default=1e-6, help="Regularization strength (ignored for OLS)")
    p.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet l1_ratio (ignored otherwise)")

    # Weighting controls
    p.add_argument("--balance-by-comp", action="store_true",
                   help="Reweight samples inversely by total-composition group count (mean weight → 1).")
    p.add_argument("--weight-alpha", type=float, default=1.0,
                   help="Exponent alpha for inverse-count weighting (w ~ n_g^(-alpha)). Default: 1.0")

    # Routing
    p.add_argument("--category", default="gpu")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    cutoffs = parse_cutoffs_arg(args.cutoffs)

    # Parse sublattice-aware mixture
    mixture: list[MixtureElement] = [parse_mix_item(s) for s in args.mix]

    # Derive global replace_elements (must match across all mixture entries)
    replace_sets: list[Set[str]] = [set(sl.replace_element for sl in m["sublattices"]) for m in mixture]
    ref_replace: Set[str] = replace_sets[0]
    for i, rs in enumerate(replace_sets[1:], start=1):
        if rs != ref_replace:
            raise ValueError(
                f"Mismatch in replace placeholders between mixture[0]={sorted(ref_replace)} "
                f"and mixture[{i}]={sorted(rs)}"
            )
    replace_elements: list[str] = sorted(ref_replace)

    # Optional early validation: ensure counts match sublattice sizes in this prototype+supercell
    conv = make_prototype(args.prototype, a=args.a)
    for mi, elem in enumerate(mixture):
        for sl in elem["sublattices"]:
            validate_counts_for_sublattice(
                conv_cell=conv,
                supercell_diag=tuple(args.supercell),
                replace_element=sl.replace_element,
                counts=dict(sl.counts),
            )

    # Weighting config payload
    weighting: dict[str, Any] | None = (
        {"scheme": "inv_count", "alpha": float(args.weight_alpha)} if args.balance_by_comp else None
    )

    # Build typed mixtures for keying (callers keep using MixtureElement for jobs)
    mixtures_for_key = [
        SublatticeMixtureElement(sublattices=m["sublattices"], K=int(m["K"]), seed=int(m["seed"]))
        for m in mixture
    ]

    algo = "randgen-4-sublcomp-1"
    ce_key = compute_ce_key(
        spec=CEKeySpec(
            prototype=args.prototype,
            prototype_params={"a": args.a},
            supercell_diag=tuple(args.supercell),
            mixtures=mixtures_for_key,
            model=args.model,
            relax_cell=bool(args.relax_cell),
            dtype=args.dtype,
            basis_spec={"basis": args.basis, "cutoffs": cutoffs},
            regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
            extra_hyperparams={},
            weighting=weighting,
            algo_version=algo,
        )
    )

    # Build the idempotent ensure job (typed, sublattice-aware) and submit
    spec = CEEnsureMixtureSpec(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
        replace_elements=replace_elements,     # list[str]
        mixture=mixture,
        default_seed=args.seed,                # CV/control seed (not the per-mixture seeds)
        model=args.model,
        relax_cell=bool(args.relax_cell),
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        extra_hyperparams={},
        category=args.category,
        weighting=weighting,
    )

    j = ensure_ce(spec)
    j.name = "ensure_ce"
    j.metadata = {**(j.metadata or {}), "_category": args.category}

    flow = Flow([j], name="Ensure CE (sublattice compositions)")
    wf = flow_to_workflow(flow)
    for fw in wf.fws:
        fw.spec = {**(fw.spec or {}), "_category": args.category}

    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)

    print("Submitted workflow:", wf_id)
    print(
        {
            "ce_key": ce_key,
            "prototype": args.prototype,
            "a": args.a,
            "supercell": tuple(args.supercell),
            "replace_elements": replace_elements,
            "sources": [{"type": "sublattice_composition", "elements": [
                {"sublattices": [sl.as_dict() for sl in m["sublattices"]], "K": m["K"], "seed": m["seed"]}
                for m in mixture
            ]}],
            "model": args.model,
            "relax_cell": bool(args.relax_cell),
            "dtype": args.dtype,
            "basis": args.basis,
            "cutoffs": cutoffs,
            "reg_type": args.reg_type,
            "alpha": args.alpha,
            "l1_ratio": args.l1_ratio,
            "weighting": weighting,
            "category": args.category,
        }
    )
