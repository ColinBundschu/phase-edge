import argparse
from typing import Any, Mapping

from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.cli.common import parse_counts_arg
from phaseedge.jobs.ensure_ce import CEEnsureMixtureSpec
from phaseedge.jobs.ensure_wl_samples_from_ce import (
    EnsureWLSamplesFromCESpec,
    ensure_wl_samples_from_ce,
)
from phaseedge.utils.keys import compute_ce_key, compute_wl_key, canonical_counts


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


def _parse_endpoint_item(s: str) -> dict[str, int]:
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if len(parts) != 1 or not parts[0].lower().startswith("counts="):
        raise ValueError("Endpoint must be 'counts=El:INT[,El2:INT...]' with no other keys.")
    _, v = parts[0].split("=", 1)
    return parse_counts_arg(v)


def _counts_sig(counts: Mapping[str, int]) -> str:
    """Stable 'El:cnt,El2:cnt2' signature with canonical ordering."""
    cc = canonical_counts(counts)
    return ",".join(f"{k}:{int(v)}" for k, v in cc.items())


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
    p.add_argument("--replace-element", required=True)

    # Composition input
    p.add_argument("--mix", action="append", required=True,
                   help="Composition element for CE: 'counts=...;K=...;seed=...' (repeatable).")
    p.add_argument("--endpoint", action="append", default=[],
                   help="Endpoint composition: 'counts=...'. No K/seed allowed. (repeatable)")
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

    # Parse + canonicalize inputs
    compositions: list[dict[str, Any]] = [_parse_mix_item(s) for s in args.mix]
    endpoints_raw: list[dict[str, int]] = [_parse_endpoint_item(s) for s in (args.endpoint or [])]
    endpoints = [canonical_counts(ep) for ep in endpoints_raw]

    # Reject illegal duplicates (mix comp equals an endpoint)
    endpoint_sigs = {_counts_sig(ep) for ep in endpoints}
    for elem in compositions:
        sig = _counts_sig(elem["counts"])
        if sig in endpoint_sigs:
            raise SystemExit(f"--mix contains a composition that is also specified as an --endpoint: {sig}")

    weighting: dict[str, Any] | None = (
        {"scheme": "inv_count", "alpha": float(args.weight_alpha)}
        if args.balance_by_comp else None
    )
    cutoffs = _parse_cutoffs_arg(args.cutoffs)

    supercell_x, supercell_y, supercell_z = tuple(int(x) for x in args.supercell)
    supercell_diag = (supercell_x, supercell_y, supercell_z)

    # Build CE spec (endpoints injected later by the job as K=1, seed=0)
    ce_spec = CEEnsureMixtureSpec(
        prototype=args.prototype,
        prototype_params={"a": float(args.a)},
        supercell_diag=supercell_diag,
        replace_element=args.replace_element,
        mixture=compositions,
        default_seed=int(args.seed),
        model=args.model,
        relax_cell=bool(args.relax_cell),
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": float(args.alpha), "l1_ratio": float(args.l1_ratio)},
        category=str(args.category),
        weighting=weighting,
    )

    # Deterministic CE key, including endpoints (as K=1, seed=0)
    algo = "randgen-3-comp-1"
    sources = [{
        "type": "composition",
        "elements": [
            *compositions,
            *({"counts": ep, "K": 1, "seed": 0} for ep in endpoints)
        ],
    }]
    ce_key = compute_ce_key(
        prototype=args.prototype,
        prototype_params={"a": float(args.a)},
        supercell_diag=supercell_diag,
        replace_element=args.replace_element,
        sources=sources,
        algo_version=algo,
        model=args.model,
        relax_cell=bool(args.relax_cell),
        dtype=args.dtype,
        basis_spec={"basis": args.basis, "cutoffs": cutoffs},
        regularization={"type": args.reg_type, "alpha": float(args.alpha), "l1_ratio": float(args.l1_ratio)},
        weighting=weighting or {},
    )

    # Precompute the WL keys for all unique NON-endpoint compositions (once per composition)
    seen_sigs: set[str] = set()
    planned_wl_runs: list[dict[str, Any]] = []
    for elem in compositions:
        counts_canon = canonical_counts(elem["counts"])
        sig = _counts_sig(counts_canon)
        if sig in endpoint_sigs or sig in seen_sigs:
            seen_sigs.add(sig)
            continue
        seen_sigs.add(sig)

        wl_key = compute_wl_key(
            ce_key=ce_key,
            bin_width=float(args.wl_bin_width),
            step_type=str(args.step_type),
            composition_counts=counts_canon,
            check_period=int(args.check_period),
            update_period=int(args.update_period),
            seed=int(args.wl_seed),
            algo_version="wl-grid-v1",
        )
        planned_wl_runs.append({
            "counts_sig": sig,
            "wl_key": wl_key,
            "short": wl_key[:12],
        })

    # Compose the master ensure job
    master_spec = EnsureWLSamplesFromCESpec(
        ce_spec=ce_spec,
        endpoints=endpoints,  # already canonicalized
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
        "replace_element": args.replace_element,
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
        "endpoints": sorted(endpoint_sigs),
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
                "ce_key", "prototype", "a", "supercell", "replace_element",
                "model", "relax_cell", "dtype", "basis", "cutoffs",
                "reg_type", "alpha", "l1_ratio", "weighting",
                "category", "endpoints",
            )
        })
        if planned_wl_runs:
            print("Planned WL chains:")
            for rec in planned_wl_runs:
                print(f"  {rec['counts_sig']:>18}  wl_key={rec['wl_key']}  (short={rec['short']})")
        else:
            print("Planned WL chains: []")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
