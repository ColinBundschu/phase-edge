#!/usr/bin/env python3
from __future__ import annotations
import argparse
from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow

from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor

from phaseedge.orchestration.flows.mace_relax import (
    RandomConfigSpec,
    make_mace_relax_flow,
    lookup_mace_result,
)
from phaseedge.science.prototypes import make_prototype, PrototypeName
from phaseedge.utils.keys import compute_set_id, fingerprint_conv_cell, occ_key_for_atoms, rng_for_index
from phaseedge.science.random_configs import make_one_snapshot

def main():
    p = argparse.ArgumentParser(description="Generate one random config → MACE relax (GPU).")
    p.add_argument("--launchpad", required=True)
    # Snapshot identity
    p.add_argument("--prototype", choices=["rocksalt"])
    p.add_argument("--a", type=float, help="Prototype lattice param if using --prototype (e.g., 4.3).")
    p.add_argument("--conv-poscar", help="Alternative: path to a primitive POSCAR for explicit conv_cell.")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX","NY","NZ"))
    p.add_argument("--replace-element", required=True)
    p.add_argument("--composition", required=True, help="e.g. 'Co:0.7,Fe:0.3'")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--index", type=int, default=0)
    # MACE settings
    p.add_argument("--model", default="MACE-MPA-0")
    p.add_argument("--relax-cell", action="store_true")
    p.add_argument("--dtype", default="float64")  # we’ll enforce via sitecustomize as discussed
    # Routing
    p.add_argument("--category", default="gpu")
    args = p.parse_args()

    # Parse composition
    comp = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in args.composition.split(",")}

    # Build conv_cell deterministically for pre-check
    if (args.conv_poscar is None) == (args.prototype is None):
        raise SystemExit("Provide exactly one of --conv-poscar or --prototype")
    if args.conv_poscar:
        conv = ase_read(args.conv_poscar)
    else:
        if not args.a:
            raise SystemExit("--a is required with --prototype")
        conv = make_prototype(args.prototype, a=args.a)

    # Compute set_id and occ_key for the intended snapshot
    set_id = compute_set_id(
        conv_fingerprint=None if args.prototype else fingerprint_conv_cell(conv),
        prototype=args.prototype,
        prototype_params=(None if args.prototype is None else {"a": args.a}),
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        compositions=[comp],
        seed=args.seed,
        algo_version="randgen-2",
    )
    rng = rng_for_index(set_id, args.index, 0)
    snapshot = make_one_snapshot(
        conv_cell=conv, supercell_diag=tuple(args.supercell), replace_element=args.replace_element, composition=comp, rng=rng
    )
    occ_key = occ_key_for_atoms(snapshot)

    # Idempotency pre-check
    cached = lookup_mace_result(set_id, occ_key, model=args.model, relax_cell=args.relax_cell, dtype=args.dtype)
    if cached and cached.get("success"):
        print("MACE relax already completed for this config:")
        print({"set_id": set_id, "occ_key": occ_key, "model": args.model, "relax_cell": args.relax_cell, "dtype": args.dtype})
        return

    # Build the flow
    spec = RandomConfigSpec(
        conv_cell=None if args.prototype else conv,
        prototype=args.prototype,
        prototype_params=(None if args.prototype is None else {"a": args.a}),
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        composition=comp,
        seed=args.seed,
        index=args.index,
    )
    flow = make_mace_relax_flow(
        snapshot=spec,
        model=args.model,
        relax_cell=args.relax_cell,
        dtype=args.dtype,
        category=args.category,
    )

    # Submit
    lp = LaunchPad.from_file(args.launchpad)
    wf = flow_to_workflow(flow)  # category injection already handled inside flow builder
    wf_id = lp.add_wf(wf)
    print("Submitted workflow:", wf_id)
    print({"set_id": set_id, "occ_key": occ_key, "model": args.model, "relax_cell": args.relax_cell, "dtype": args.dtype})

if __name__ == "__main__":
    main()
