import argparse
from typing import Any

from phaseedge.jobs.ensure_ce import CEEnsureMixtureSpec, ensure_ce  # spec name kept for now

from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad

from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import validate_counts_for_sublattices
from phaseedge.utils.keys import compute_ce_key
from phaseedge.cli.common import parse_counts_arg


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
            item["counts"] = parse_counts_arg(v)
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
        description="Ensure a CE over compositions (deterministic snapshots → MACE relax cache → train CE → store).",
    )
    p.add_argument("--launchpad", required=True)

    # System / snapshot identity (prototype-only)
    p.add_argument("--prototype", required=True, choices=["rocksalt"])
    p.add_argument("--a", required=True, type=float, help="Prototype lattice parameter (e.g., 4.3).")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    p.add_argument("--replace-element", required=True)

    # Composition input (repeatable)
    p.add_argument("--mix", action="append", required=True, help="Composition element: 'counts=...;K=...;seed=...'")
    p.add_argument("--seed", type=int, default=0, help="Global/default seed")

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

    # Build composition list
    compositions = [_parse_mix_item(s) for s in args.mix]

    # Optional early validation
    conv = make_prototype(args.prototype, a=args.a)
    for elem in compositions:
        counts_clean = {str(k): int(v) for k, v in elem["counts"].items()}
        validate_counts_for_sublattices(
            conv_cell=conv,
            supercell_diag=tuple(args.supercell),
            composition_map={args.replace_element: counts_clean},
        )

    # Weighting config payload
    weighting: dict[str, Any] | None = (
        {"scheme": "inv_count", "alpha": float(args.weight_alpha)} if args.balance_by_comp else None
    )

    # Compute CE key (unified, sources-based)
    algo = "randgen-3-comp-1"
    sources = [
        {"type": "composition", "elements": compositions}
    ]
    ce_key = compute_ce_key(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        sources=sources,
        algo_version=algo,
        model=args.model,
        relax_cell=bool(args.relax_cell),
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        weighting=weighting,
    )

    # Build the idempotent ensure job (mixture spec retained) and submit
    spec = CEEnsureMixtureSpec(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        mixture=compositions,                # name retained in spec for now
        default_seed=args.seed,
        model=args.model,
        relax_cell=bool(args.relax_cell),
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": args.alpha, "l1_ratio": args.l1_ratio},
        category=args.category,
        weighting=weighting,
    )

    j = ensure_ce(spec)
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
            "ce_key": ce_key,
            "prototype": args.prototype,
            "a": args.a,
            "supercell": tuple(args.supercell),
            "replace_element": args.replace_element,
            "sources": sources,
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
