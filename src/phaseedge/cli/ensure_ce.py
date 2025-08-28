import argparse
from typing import Any, Mapping, Sequence, Optional, TypedDict

from phaseedge.jobs.check_or_schedule_ce import (
    CEEnsureMixtureSpec,
    check_or_schedule_ce,
)

from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad

from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import validate_counts_for_sublattice
from phaseedge.utils.keys import compute_ce_key_mixture


def _parse_counts_arg(s: str) -> dict[str, int]:
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


def _parse_mix_item(s: str) -> dict[str, Any]:
    """
    Parse one --mix item like:
      "counts=Fe:54,Mn:46;K=50;seed=123"
    Keys: counts (required), K (required), seed (optional).
    """
    item: dict[str, Any] = {}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad --mix token '{p}' (expected key=value)")
        k, v = p.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "counts":
            item["counts"] = _parse_counts_arg(v)
        elif k == "k":
            item["K"] = int(v)
        elif k == "seed":
            item["seed"] = int(v)
        else:
            raise ValueError(f"Unknown key '{k}' in --mix item")
    if "counts" not in item or "K" not in item:
        raise ValueError("Each --mix item must include counts=... and K=...")
    return item


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-ensure-ce",
        description="Ensure a CE over a mixture of compositions (deterministic snapshots → MACE relax cache → train CE → store).",
    )
    p.add_argument("--launchpad", required=True)

    # System / snapshot identity (prototype-only)
    p.add_argument("--prototype", required=True, choices=["rocksalt"])
    p.add_argument("--a", required=True, type=float, help="Prototype lattice parameter (e.g., 4.3).")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    p.add_argument("--replace-element", required=True)

    # Mixture input (repeatable)
    p.add_argument("--mix", action="append", default=[], help="Mixture element: 'counts=...;K=...;seed=...'")

    # Back-compat single-composition flags (used only if --mix not supplied)
    p.add_argument("--counts", help="Exact counts per species, e.g. 'Co:76,Fe:24'")
    p.add_argument("--seed", type=int, help="Global/default seed (and seed for single-composition mode)")
    p.add_argument("--K", type=int, help="Exact number of snapshots for single-composition mode")

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
                   help="Reweight samples inversely by composition count (mean weight normalized to 1).")
    p.add_argument("--weight-alpha", type=float, default=1.0,
                   help="Exponent alpha for inverse-count weighting (w ~ n_g^(-alpha)). Default: 1.0")

    # Routing
    p.add_argument("--category", default="gpu")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    cutoffs = _parse_cutoffs_arg(args.cutoffs)

    # Build mixture (either from --mix or from legacy single-composition flags)
    if args.mix:
        mixture = [_parse_mix_item(s) for s in args.mix]
        default_seed = 0 if args.seed is None else int(args.seed)
    else:
        if args.counts is None or args.seed is None or args.K is None:
            raise SystemExit("When --mix is not used, you must provide --counts, --seed, and --K.")
        mixture = [{"counts": _parse_counts_arg(args.counts), "K": int(args.K), "seed": int(args.seed)}]
        default_seed = int(args.seed)

    # Optional early validation
    conv = make_prototype(args.prototype, a=args.a)
    for elem in mixture:
        cts = dict(elem["counts"])
        validate_counts_for_sublattice(
            conv_cell=conv,
            supercell_diag=tuple(args.supercell),
            replace_element=args.replace_element,
            counts=cts,
        )

    # Weighting config payload
    weighting: dict[str, Any] | None = (
        {"scheme": "inv_count", "alpha": float(args.weight_alpha)} if args.balance_by_comp else None
    )

    # Compute CE key (mixture-aware)
    ce_key = compute_ce_key_mixture(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        mixture=mixture,
        algo_version="randgen-3-mix-1",
        model=args.model,
        relax_cell=bool(args.relax_cell),
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        extra_hyperparams={},
        weighting=weighting,
    )

    # Build the idempotent ensure job (mixture spec) and submit
    spec = CEEnsureMixtureSpec(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        mixture=mixture,
        default_seed=default_seed,
        model=args.model,
        relax_cell=bool(args.relax_cell),
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        extra_hyperparams={},
        category=args.category,
        weighting=weighting,
    )

    j = check_or_schedule_ce(spec)
    j.name = "ensure_ce"
    j.metadata = {**(j.metadata or {}), "_category": args.category}

    flow = Flow([j], name="Ensure CE (mixture)")
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
            "replace_element": args.replace_element,
            "mixture": mixture,
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
