import argparse
import json
from typing import Any, Dict

from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.jobs.add_wl_chunk import add_wl_chunk
from phaseedge.utils.keys import compute_wl_key
from phaseedge.cli.common import parse_counts_arg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-extend-wl",
        description="Extend a Wang-Landau chain by N steps (idempotent checkpoint block).",
    )
    p.add_argument("--launchpad", required=True, help="Path to LaunchPad YAML.")
    p.add_argument("--ce-key", required=True)
    p.add_argument("--bin-width", required=True, type=float, help="Uniform enthalpy bin width (eV).")
    p.add_argument(
        "--composition-counts",
        required=True,
        help="Exact counts per species, e.g. 'Fe:54,Mg:54'.",
    )
    p.add_argument("--step-type", default="swap", choices=["swap"], help="MC move type (canonical).")
    p.add_argument("--check-period", type=int, default=5_000)
    p.add_argument("--update-period", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    # Capture up to K samples per visited enthalpy bin (0 = disabled)
    p.add_argument("--samples-per-bin", type=int, default=0,
                   help="Max snapshots to capture per enthalpy bin (0 disables).")

    p.add_argument("--category", default="cpu")

    # Extension length for THIS block
    p.add_argument(
        "--steps-to-run",
        dest="steps_to_run",
        required=True,
        type=int,
        help="Number of WL steps to add in this checkpoint block.",
    )

    # Output mode
    p.add_argument("--json", action="store_true", help="Print a machine-readable submission summary.")
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    composition_counts: Dict[str, int] = parse_counts_arg(args.composition_counts)

    # wl_key encodes chain identity only (no steps, no samples_per_bin).
    wl_key: str = compute_wl_key(
        ce_key=str(args.ce_key),
        bin_width=float(args.bin_width),
        step_type=str(args.step_type),
        composition_counts=composition_counts,
        check_period=int(args.check_period),
        update_period=int(args.update_period),
        seed=int(args.seed),
        algo_version="wl-grid-v1",
    )
    short = wl_key[:12]

    # Build run_spec. steps & samples_per_bin are runtime policies (non-key).
    run_spec = WLSamplerSpec(
        wl_key=wl_key,
        ce_key=str(args.ce_key),
        bin_width=float(args.bin_width),
        steps=int(args.steps_to_run),
        composition_counts=composition_counts,
        step_type=str(args.step_type),
        check_period=int(args.check_period),
        update_period=int(args.update_period),
        seed=int(args.seed),
        samples_per_bin=int(args.samples_per_bin),
    )

    j = add_wl_chunk(run_spec)
    j.name = f"extend_wl::{short}::+{int(args.steps_to_run):,}"
    j.metadata = {**(j.metadata or {}), "_category": args.category, "wl_key": wl_key}

    flow = Flow([j], name=f"Extend WL :: {short} :: +{int(args.steps_to_run):,}")
    wf = flow_to_workflow(flow)
    for fw in wf.fws:
        fw.name = f"{fw.name}::{short}"
        fw.spec = {**(fw.spec or {}), "_category": args.category, "wl_key": wl_key}

    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)

    payload: Dict[str, Any] = {
        "submitted_workflow_id": wf_id,
        "wl_key": wl_key,
        "ce_key": args.ce_key,
        "bin_width": float(args.bin_width),
        "composition_counts": composition_counts,
        "step_type": str(args.step_type),
        "check_period": int(args.check_period),
        "update_period": int(args.update_period),
        "seed": int(args.seed),
        "samples_per_bin": int(args.samples_per_bin),  # transparency
        "category": str(args.category),
        "steps_to_run": int(args.steps_to_run),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print("Submitted WL extension workflow:", wf_id)
        print({k: payload[k] for k in (
            "wl_key", "ce_key", "bin_width", "composition_counts", "step_type",
            "check_period", "update_period", "seed", "samples_per_bin", "category", "steps_to_run"
        )})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
