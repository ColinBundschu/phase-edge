#!/usr/bin/env python3
import argparse
from pathlib import Path
from fireworks import LaunchPad
from jobflow import Flow  # type: ignore[reportPrivateImportUsage]
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.orchestration.makers.ensure_snapshots import ensure_snapshots_job
from phaseedge.utils.keys import compute_set_id


def main() -> None:
    p = argparse.ArgumentParser(description="Add expandable snapshot-set workflow (prototype-only) to FireWorks")
    # FireWorks connection
    p.add_argument("--launchpad", type=str, required=True, help="Path to my_launchpad.yaml")

    # Prototype (MVP supports 'rocksalt'; extend later)
    p.add_argument("--prototype", type=str, default="rocksalt", help='Prototype name (default: "rocksalt")')
    p.add_argument("--a", type=float, default=4.3, help="Lattice parameter a for prototype (default: 4.3 Å)")
    p.add_argument("--cubic", action="store_true", help="Use cubic prototype (default True)")
    # Supercell / chemistry
    p.add_argument("--diag", type=int, nargs=3, default=[3, 3, 3], help="Supercell diag, e.g. 3 3 3")
    p.add_argument("--replace", type=str, required=True, help="Element in prototype to replace (e.g., Mg)")
    p.add_argument("--comp", type=str, required=True, help='Composition like "Fe:0.3,Co:0.7"')
    # Sequence identity
    p.add_argument("--seed", type=int, default=42, help="Base seed for the set")
    p.add_argument("--target", type=int, default=100, help="Ensure at least this many snapshots exist")
    # Output artifacts
    p.add_argument("--outdir", type=str, default="artifacts/seeds", help="Directory for POSCAR outputs")
    args = p.parse_args()

    # parse composition string
    comp_pairs = [kv.strip() for kv in args.comp.split(",") if kv.strip()]
    composition: dict[str, float] = {}
    for kv in comp_pairs:
        el, val = kv.split(":")
        composition[el.strip()] = float(val)

    supercell_diag = (args.diag[0], args.diag[1], args.diag[2])

    # prototype params (MVP)
    proto_params = {"a": args.a, "cubic": True if args.cubic or True else True}

    # compute set_id up front (for logging / later querying)
    set_id = compute_set_id(
        conv_fingerprint=None,
        prototype=args.prototype,
        prototype_params=proto_params,
        supercell_diag=supercell_diag,
        replace_element=args.replace,
        compositions=[composition],
        seed=args.seed,
        algo_version="randgen-2",
    )
    print(f"set_id: {set_id}")

    # build single-job Flow
    job = ensure_snapshots_job(
        prototype=args.prototype,            # type: ignore[arg-type]
        prototype_params=proto_params,
        supercell_diag=supercell_diag,
        replace_element=args.replace,
        composition=composition,
        seed=args.seed,
        target_count=args.target,
        outdir=str(Path(args.outdir) / f"{set_id[:8]}"),
        algo_version="randgen-2",
    )
    flow = Flow([job], name=f"ensure-{set_id[:8]}")

    # submit to FireWorks
    lp = LaunchPad.from_file(args.launchpad)
    lp.add_wf(flow_to_workflow(flow))
    print("Workflow added. Use qlaunch/rlaunch to execute.")


if __name__ == "__main__":
    main()
