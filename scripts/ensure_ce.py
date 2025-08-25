#!/usr/bin/env python3
import argparse

from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.orchestration.jobs.check_or_schedule_ce import (
    CEEnsureSpec,
    check_or_schedule_ce,
)
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import validate_counts_for_sublattice
from phaseedge.utils.keys import compute_ce_key


def _parse_counts_arg(s: str) -> dict[str, int]:
    """Parse --counts 'Co:76,Fe:32' -> {'Co': 76, 'Fe': 32} (whitespace-tolerant)."""
    out: dict[str, int] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if ":" not in kv:
            raise ValueError(f"Bad counts token '{kv}' (expected 'El:INT').")
        k, v = kv.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Empty element in counts token '{kv}'.")
        if k in out:
            raise ValueError(f"Duplicate element '{k}' in --counts.")
        out[k] = int(v)
    if not out:
        raise ValueError("Empty --counts.")
    return out


def _parse_cutoffs_arg(s: str) -> dict[int, float]:
    """Parse --cutoffs '1:100,2:10,3:8,4:6' -> {1:100.0,2:10.0,3:8.0,4:6.0}."""
    out: dict[int, float] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if ":" not in kv:
            raise ValueError(f"Bad cutoffs token '{kv}' (expected 'ORDER:VALUE').")
        k, v = kv.split(":", 1)
        out[int(k.strip())] = float(v.strip())
    if not out:
        raise ValueError("Empty --cutoffs.")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ensure a CE (deterministic snapshots → MACE relax cache → train CE → store)."
    )
    p.add_argument("--launchpad", required=True)

    # System / snapshot identity (prototype-only)
    p.add_argument("--prototype", required=True, choices=["rocksalt"])
    p.add_argument("--a", required=True, type=float, help="Prototype lattice parameter (e.g., 4.3).")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    p.add_argument("--replace-element", required=True)
    p.add_argument("--counts", required=True, help="Exact counts per species on target sublattice, e.g. 'Co:76,Fe:32'")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--K", type=int, required=True, help="Exact number of snapshots (indices 0..K-1)")

    # Relax/engine for training energies
    p.add_argument("--model", default="MACE-MPA-0")
    p.add_argument("--relax-cell", action="store_true")
    p.add_argument("--dtype", default="float64")

    # CE hyperparameters
    p.add_argument("--basis", default="sinusoid", help="Basis family name understood by smol (default: sinusoid)")
    p.add_argument(
        "--cutoffs",
        default="1:100,2:10,3:8,4:6",
        help="Comma sep. per-body cutoffs, e.g. '1:100,2:10,3:8,4:6'",
    )
    p.add_argument("--reg-type", choices=["ols", "ridge", "lasso", "elasticnet"], default="ols")
    p.add_argument("--alpha", type=float, default=1e-6, help="Regularization strength (ignored for OLS)")
    p.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet l1_ratio (ignored otherwise)")

    # Routing
    p.add_argument("--category", default="gpu")

    args = p.parse_args()
    counts = _parse_counts_arg(args.counts)
    cutoffs = _parse_cutoffs_arg(args.cutoffs)
    indices = list(range(int(args.K)))

    # Optional early validation: counts match replaceable sites
    conv = make_prototype(args.prototype, a=args.a)
    _ = validate_counts_for_sublattice(
        conv_cell=conv,
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        counts=counts,
    )

    # Compute CE key now so you can grep/track this run
    ce_key = compute_ce_key(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        counts=counts,
        seed=int(args.seed),
        indices=indices,
        algo_version="randgen-2-counts-1",
        model=args.model,
        relax_cell=args.relax_cell,
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        extra_hyperparams={},
    )

    # Build the idempotent ensure job
    spec = CEEnsureSpec(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        counts=counts,
        seed=int(args.seed),
        K=int(args.K),
        model=args.model,
        relax_cell=args.relax_cell,
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        extra_hyperparams={},
        category=args.category,
    )

    j = check_or_schedule_ce(spec)
    j.name = "ensure_ce"
    j.metadata = {**(j.metadata or {}), "_category": args.category}

    flow = Flow([j], name="Ensure CE")
    wf = flow_to_workflow(flow)
    # belt & suspenders category tagging
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
            "replace_element": args.replace_element,
            "K": args.K,
            "model": args.model,
            "relax_cell": args.relax_cell,
            "dtype": args.dtype,
            "basis": args.basis,
            "cutoffs": cutoffs,
            "reg_type": args.reg_type,
            "alpha": args.alpha,
            "l1_ratio": args.l1_ratio,
            "category": args.category,
        }
    )


if __name__ == "__main__":
    main()
