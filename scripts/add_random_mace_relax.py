#!/usr/bin/env python3
import argparse

from fireworks import LaunchPad

from phaseedge.orchestration.flows.mace_relax import make_mace_relax_workflow
from phaseedge.orchestration.jobs.random_config import RandomConfigSpec
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import validate_counts_for_sublattice
from phaseedge.utils.keys import compute_set_id_counts


def _parse_counts_arg(s: str) -> dict[str, int]:
    """Parse --counts 'Co:76,Fe:32' -> {'Co': 76, 'Fe': 32}."""
    out: dict[str, int] = {}
    for kv in s.split(","):
        k, v = kv.split(":")
        out[k.strip()] = int(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate one random config (exact counts) → MACE relax (GPU)."
    )
    p.add_argument("--launchpad", required=True)

    # Snapshot identity (prototype-only)
    p.add_argument("--prototype", required=True, choices=["rocksalt"])
    p.add_argument("--a", required=True, type=float,
                   help="Prototype lattice parameter (e.g., 4.3).")
    p.add_argument("--supercell", type=int, nargs=3, required=True,
                   metavar=("NX", "NY", "NZ"))
    p.add_argument("--replace-element", required=True)
    p.add_argument(
        "--counts",
        required=True,
        help="Exact counts per species on target sublattice, e.g. 'Co:76,Fe:32'",
    )
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

    # Build the conventional prototype for validation only
    conv = make_prototype(args.prototype, a=args.a)

    # Validate counts match the number of replaceable sites (raises on mismatch)
    n_sites = validate_counts_for_sublattice(
        conv_cell=conv,
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        counts=counts,
    )
    # (n_sites is returned for convenience/logging; not otherwise used here.)

    # Deterministic set identity (counts-based; no ratios)
    set_id = compute_set_id_counts(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        counts=counts,
        seed=args.seed,
    )

    # Build spec (prototype-only)
    spec = RandomConfigSpec(
        prototype=args.prototype,
        prototype_params={"a": args.a},
        supercell_diag=tuple(args.supercell),
        replace_element=args.replace_element,
        counts=counts,
        seed=args.seed,
        index=args.index,
    )

    # Build and submit workflow
    wf = make_mace_relax_workflow(
        snapshot=spec,
        model=args.model,
        relax_cell=args.relax_cell,
        dtype=args.dtype,
        category=args.category,
    )

    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)
    print("Submitted workflow:", wf_id)
    print(
        {
            "set_id": set_id,
            "model": args.model,
            "relax_cell": args.relax_cell,
            "dtype": args.dtype,
            "index": args.index,
            "n_sites": n_sites,
        }
    )


if __name__ == "__main__":
    main()
