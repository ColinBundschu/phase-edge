import argparse
import json
from typing import Dict

from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.jobs.check_or_schedule_wl import WLEnsureSpec, check_or_schedule_wl
from phaseedge.utils.keys import compute_wl_key


def _parse_counts_arg(s: str) -> dict[str, int]:
    """Parse 'Fe:54,Mn:54' -> {'Fe': 54, 'Mn': 54} (whitespace-tolerant)."""
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
            raise ValueError(f"Duplicate element '{k}' in --composition-counts.")
        iv = int(v)
        if iv < 0:
            raise ValueError(f"Negative count for '{k}': {iv}")
        out[k] = iv
    if not out:
        raise ValueError("Empty --composition-counts.")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-wl-run",
        description="Submit an idempotent Wang-Landau job to FireWorks (counts-only canonical).",
    )
    p.add_argument("--launchpad", required=True, help="Path to LaunchPad YAML.")
    p.add_argument("--ce-key", required=True)
    p.add_argument("--bin-width", required=True, type=float, help="Uniform enthalpy bin width (eV).")
    p.add_argument("--steps", required=True, type=int, help="Number of WL steps to perform.")
    p.add_argument(
        "--composition-counts",
        required=True,
        help="Exact counts per species, e.g. 'Fe:54,Mn:54'.",
    )
    p.add_argument("--step-type", default="swap", choices=["swap"], help="MC move type (canonical).")
    p.add_argument("--check-period", type=int, default=5_000)
    p.add_argument("--update-period", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--category", default="gpu")

    # Internal dev knobs (do NOT affect wl_key)
    p.add_argument("--pilot-samples", type=int, default=256)
    p.add_argument("--sigma-multiplier", type=float, default=50.0)

    # Output mode
    p.add_argument("--json", action="store_true", help="Print a machine-readable submission summary.")
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    composition_counts: Dict[str, int] = _parse_counts_arg(args.composition_counts)

    # Compute the wl_key up-front (public contract only) so we can print and brand the workflow.
    wl_key: str = compute_wl_key(
        ce_key=str(args.ce_key),
        bin_width=float(args.bin_width),
        steps=int(args.steps),
        step_type=str(args.step_type),
        composition_counts=composition_counts,
        check_period=int(args.check_period),
        update_period=int(args.update_period),
        seed=int(args.seed),
        grid_anchor=0.0,
        algo_version="wl-grid-v1",
    )
    short = wl_key[:12]

    # Build the idempotent ensure job (FireWorker will short-circuit if wl_key already exists)
    spec = WLEnsureSpec(
        ce_key=str(args.ce_key),
        bin_width=float(args.bin_width),
        steps=int(args.steps),
        composition_counts=composition_counts,
        step_type=str(args.step_type),
        check_period=int(args.check_period),
        update_period=int(args.update_period),
        seed=int(args.seed),
        category=str(args.category),
    )

    j = check_or_schedule_wl(spec)
    j.name = f"ensure_wl::{short}"
    j.metadata = {**(j.metadata or {}), "_category": args.category, "wl_key": wl_key}

    flow = Flow([j], name=f"Ensure WL :: {short}")
    wf = flow_to_workflow(flow)
    # Tag category & wl_key at the FireWorks layer too; brand names for greppability.
    for fw in wf.fws:
        fw.name = f"{fw.name}::{short}"
        fw.spec = {**(fw.spec or {}), "_category": args.category, "wl_key": wl_key}

    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)

    payload = {
        "submitted_workflow_id": wf_id,
        "wl_key": wl_key,
        "ce_key": args.ce_key,
        "bin_width": float(args.bin_width),
        "steps": int(args.steps),
        "composition_counts": composition_counts,
        "step_type": str(args.step_type),
        "check_period": int(args.check_period),
        "update_period": int(args.update_period),
        "seed": int(args.seed),
        "category": str(args.category),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print("Submitted WL workflow:", wf_id)
        print({k: payload[k] for k in (
            "wl_key", "ce_key", "bin_width", "steps", "composition_counts", "step_type",
            "check_period", "update_period", "seed", "category"
        )})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
