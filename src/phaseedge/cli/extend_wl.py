import argparse
import json
from typing import Any

from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.schemas.wl_sampler_spec import WLSamplerSpec
from phaseedge.jobs.add_wl_block import add_wl_block, add_wl_chain
from phaseedge.cli.common import parse_composition_map


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-extend-wl",
        description="Extend a Wang-Landau chain by N steps (idempotent block).",
    )
    p.add_argument("--launchpad", required=True, help="Path to LaunchPad YAML.")
    p.add_argument("--ce-key", required=True)
    p.add_argument("--bin-width", required=True, type=float, help="Uniform enthalpy bin width (eV).")
    p.add_argument("--sl-comp-map", required=True, help='Canonical map to identify the sublattices (e.g., Es:{Mg:16},Fm:{Al:32}).')
    p.add_argument("--initial-comp-map", required=True, help='Initial composition map for the entire WL supercell (e.g., Es:{Mg:8,Al:8},Fm:{Mg:8,Al:24}).')
    p.add_argument("--reject-cross-sublattice-swaps", action="store_true", help="Reject WL swap moves that cross sublattices.")

    p.add_argument("--step-type", default="swap", choices=["swap"], help="MC move type (canonical).")
    p.add_argument("--check-period", type=int, default=5_000)
    p.add_argument("--update-period", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    # Capture up to K samples per visited enthalpy bin (0 = disabled)
    p.add_argument(
        "--samples-per-bin",
        type=int,
        default=0,
        help="Max snapshots to capture per enthalpy bin (0 disables).",
    )
    p.add_argument(
        "--collect-cation-stats",
        action="store_true",
        help="If set, collect per-bin cation-count histograms per sublattice.",
    )
    p.add_argument(
        "--production-mode",
        action="store_true",
        help="If set, freeze WL updates (entropy/histogram/mod-factor) and only collect statistics.",
    )

    p.add_argument("--category", default="cpu")

    # Extension length for THIS block
    p.add_argument(
        "--steps-to-run",
        dest="steps_to_run",
        required=True,
        type=int,
        help="Number of WL steps to add in this block.",
    )

    # Optional chaining: repeat the same chunk sequentially
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="If >1, create a linear chain of this many WL chunks run back-to-back.",
    )

    # Output mode
    p.add_argument("--json", action="store_true", help="Print a machine-readable submission summary.")
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    initial_comp_map = parse_composition_map(args.initial_comp_map)
    sl_comp_map = parse_composition_map(args.sl_comp_map)

    # Build run_spec. steps/samples_per_bin/flags are runtime policies (non-key).
    run_spec = WLSamplerSpec(
        ce_key=str(args.ce_key),
        bin_width=float(args.bin_width),
        steps=int(args.steps_to_run),
        sl_comp_map=sl_comp_map,
        initial_comp_map=initial_comp_map,
        step_type=str(args.step_type),
        check_period=int(args.check_period),
        update_period=int(args.update_period),
        seed=int(args.seed),
        samples_per_bin=int(args.samples_per_bin),
        collect_cation_stats=bool(args.collect_cation_stats),
        production_mode=bool(args.production_mode),
        reject_cross_sublattice_swaps=bool(args.reject_cross_sublattice_swaps),
    )

    short = run_spec.wl_key[:12]

    # Build either a single-chunk flow or a linear chain of chunks.
    repeats = int(args.repeats)
    if repeats > 1:
        flow = add_wl_chain(run_spec, repeats=repeats)
        # Decorate each job with metadata and a clearer name; CLI has access to wl_key/category.
        for idx, j in enumerate(flow):
            j.metadata = {**(j.metadata or {}), "_category": args.category, "wl_key": run_spec.wl_key}
            j.name = f"extend_wl::{short}::chunk{idx + 1}/{repeats}::+{int(args.steps_to_run):,}"
        flow.name = f"Extend WL :: {short} :: +{int(args.steps_to_run):,} × {repeats}"
    else:
        j = add_wl_block(run_spec)
        j.name = f"extend_wl::{short}::+{int(args.steps_to_run):,}"
        j.metadata = {**(j.metadata or {}), "_category": args.category, "wl_key": run_spec.wl_key}
        flow = Flow([j], name=f"Extend WL :: {short} :: +{int(args.steps_to_run):,}")

    wf = flow_to_workflow(flow)
    for fw in wf.fws:
        # Append short wl_key to each FireWork name and propagate routing metadata to the manager.
        fw.name = f"{fw.name}::{short}"
        fw.spec = {**(fw.spec or {}), "_category": args.category, "wl_key": run_spec.wl_key}

    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)

    payload: dict[str, Any] = {
        "submitted_workflow_id": wf_id,
        "wl_key": run_spec.wl_key,
        "ce_key": args.ce_key,
        "bin_width": float(args.bin_width),
        "sl_comp_map": sl_comp_map,
        "initial_comp_map": initial_comp_map,
        "step_type": str(args.step_type),
        "check_period": int(args.check_period),
        "update_period": int(args.update_period),
        "seed": int(args.seed),
        "samples_per_bin": int(args.samples_per_bin),  # transparency
        "collect_cation_stats": bool(args.collect_cation_stats),
        "production_mode": bool(args.production_mode),
        "category": str(args.category),
        "steps_to_run": int(args.steps_to_run),
        "repeats": repeats,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print("Submitted WL extension workflow:", wf_id)
        print(
            {
                k: payload[k]
                for k in (
                    "wl_key",
                    "ce_key",
                    "bin_width",
                    "sl_comp_map",
                    "initial_comp_map",
                    "step_type",
                    "check_period",
                    "update_period",
                    "seed",
                    "samples_per_bin",
                    "collect_cation_stats",
                    "production_mode",
                    "category",
                    "steps_to_run",
                    "repeats",
                )
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
