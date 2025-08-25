#!/usr/bin/env python3
from __future__ import annotations
import argparse

from fireworks import LaunchPad
from ase.io import read as ase_read
import numpy as np

from phaseedge.orchestration.flows.mace_relax import (
    make_mace_relax_workflow,
)
from phaseedge.orchestration.jobs.random_config import RandomConfigSpec
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.utils.keys import (
    occ_key_for_atoms,
    rng_for_index,
)
from phaseedge.utils.keys import compute_set_id_counts  # counts-based

# ---- CLI --------------------------------------------------------------------------

def _parse_counts_arg(s: str) -> dict[str, int]:
    """
    Parse "--counts 'Co:76,Fe:32'" -> {"Co": 76, "Fe": 32}
    """
    out: dict[str, int] = {}
    for kv in s.split(","):
        k, v = kv.split(":")
        out[k.strip()] = int(v)
    return out

def main():
    p = argparse.ArgumentParser(description="Generate one random config (exact counts) → MACE relax (GPU).")
    p.add_argument("--launchpad", required=True)

    # Snapshot identity
    p.add_argument("--prototype", choices=["rocksalt"])
    p.add_argument("--a", type=float, help="Prototype lattice param if using --prototype (e.g., 4.3).")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    p.add_argument("--replace-element", required=True)
    p.add_argument("--counts", required=True, help="Exact counts per species on target sublattice, e.g. 'Co:76,Fe:32'")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--index", type=int, default=0)

    # MACE settings
    p.add_argument("--model", default="MACE-MPA-0")
    p.add_argument("--relax-cell", action="store_true")
    p.add_argument("--dtype", default="float64")

    # Routing
    p.add_argument("--category", default="gpu")
    args = p.parse_args()

    counts = _parse_counts_arg(args.counts)

    if not args.a:
        raise SystemExit("--a is required")
    conv = make_prototype(args.prototype, a=args.a)
    proto = args.prototype
    proto_params = {"a": args.a}

    # Validate counts against replacement sublattice size
    sc = conv.repeat(tuple(args.supercell))
    n_sites = int(np.sum(np.array(sc.get_chemical_symbols()) == args.replace_element))
    total = sum(int(v) for v in counts.values())
    if total != n_sites:
        raise SystemExit(f"--counts must sum to replacement site count {n_sites}; got {total}")

    # set_id (counts-based) + precompute occ_key for explicit preview
    set_id = compute_set_id_counts(
        prototype=proto,
        prototype_params=proto_params,
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        counts=counts,
        seed=args.seed,
    )
    rng = rng_for_index(set_id, args.index, 0)
    snapshot = make_one_snapshot(
        conv_cell=conv, # pyright: ignore[reportArgumentType]
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        counts=counts,
        rng=rng,
    )
    occ_key = occ_key_for_atoms(snapshot)

    # Build spec
    spec = RandomConfigSpec(
        prototype=proto,
        prototype_params=proto_params,
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        counts=counts,
        seed=args.seed,
        index=args.index,
    )

    # Build Workflow with category injected
    wf = make_mace_relax_workflow(
        snapshot=spec,
        model=args.model,
        relax_cell=args.relax_cell,
        dtype=args.dtype,
        category=args.category,
    )

    # Submit
    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)
    print("Submitted workflow:", wf_id)
    print({
        "set_id": set_id,
        "occ_key": occ_key,
        "model": args.model,
        "relax_cell": args.relax_cell,
        "dtype": args.dtype,
    })

if __name__ == "__main__":
    main()
