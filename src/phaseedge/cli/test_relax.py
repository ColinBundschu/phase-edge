import argparse
import json
from typing import Any

from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow

from phaseedge.cli.common import parse_mix_item
from phaseedge.jobs.decide_relax import check_or_schedule_relax
from phaseedge.jobs.random_config import RandomConfigSpec, make_random_config
from phaseedge.science.prototypes import make_prototype
from phaseedge.science.random_configs import make_one_snapshot
from phaseedge.utils.keys import compute_set_id_counts, rng_for_index, occ_key_for_atoms
from ase.atoms import Atoms


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pe-test-relax",
        description="Generate ONE random snapshot (sublattice format) and run check_or_schedule_relax on it.",
    )
    p.add_argument("--launchpad", required=True)

    # System / prototype identity
    p.add_argument("--prototype", required=True, choices=["rocksalt", "spinel"])
    p.add_argument("--a", required=True, type=float, help="Prototype lattice parameter (e.g., 4.3).")
    p.add_argument("--supercell", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))

    # Composition input (same format as pe-ensure-ce, but this CLI expects exactly one --mix)
    p.add_argument(
        "--mix",
        action="append",
        required=True,
        help=(
            "Mixture element with per-sublattice counts. Example:\n"
            "--mix \"K=10;seed=42;subl=replace=Mg,counts=Fe:23,Mg:233\""
        ),
    )

    # Which snapshot inside the --mix (0 <= index < K)
    p.add_argument("--index", type=int, default=0, help="Which snapshot index inside the --mix to generate.")

    # Relaxation / engine identity
    p.add_argument("--model", default="MACE-MPA-0")
    p.add_argument("--relax-cell", action="store_true")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--category", default="gpu")

    # Output mode
    p.add_argument("--json", action="store_true", help="Print a machine-readable submission summary.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # Parse the first --mix item (this CLI exercises one snapshot from one mixture element)
    mix = parse_mix_item(args.mix[0])
    sublats = mix["sublattices"]
    K = int(mix["K"])
    seed = int(mix["seed"])

    proto_params: dict[str, Any] = {"a": float(args.a)}
    supercell_diag = (int(args.supercell[0]), int(args.supercell[1]), int(args.supercell[2]))

    # Compute set_id deterministically from inputs
    set_id: str = compute_set_id_counts(
        prototype=args.prototype,
        prototype_params=proto_params,
        supercell_diag=supercell_diag,
        sublattices=sublats,
        seed=seed,
    )

    # Rebuild the exact snapshot locally to compute occ_key
    conv_cell: Atoms = make_prototype(args.prototype, **proto_params)
    rng = rng_for_index(set_id, int(args.index), 0)
    snapshot = make_one_snapshot(
        conv_cell=conv_cell,
        supercell_diag=supercell_diag,
        sublattices=sublats,
        rng=rng,
    )
    occ_key: str = occ_key_for_atoms(snapshot)

    if not (0 <= int(args.index) < K):
        raise ValueError(f"--index must be in [0, {K}), got {args.index}")

    # Build the generator job
    spec = RandomConfigSpec(
        prototype=args.prototype,
        prototype_params=proto_params,
        supercell_diag=supercell_diag,
        sublattices=sublats,
        seed=seed,
        index=int(args.index),
    )
    j_gen = make_random_config(spec)
    j_gen.name = f"generate_random_config[{args.index}]"
    j_gen.update_metadata({"_category": args.category})

    # Decide/schedule relax
    j_decide = check_or_schedule_relax(
        set_id=j_gen.output["set_id"],
        occ_key=j_gen.output["occ_key"],
        structure=j_gen.output["structure"],
        model=str(args.model),
        relax_cell=bool(args.relax_cell),
        dtype=str(args.dtype),
        category=str(args.category),
    )
    j_decide.name = f"check_or_schedule_relax[{args.index}]"
    j_decide.update_metadata({"_category": args.category})

    flow = Flow([j_gen, j_decide], name="Test relax (sublattice)")
    wf = flow_to_workflow(flow)
    for fw in wf.fws:
        fw.spec = {**(fw.spec or {}), "_category": args.category}

    lp = LaunchPad.from_file(args.launchpad)
    wf_id = lp.add_wf(wf)

    payload = {
        "submitted_workflow_id": wf_id,
        "set_id": set_id,
        "occ_key": occ_key,
        "prototype": args.prototype,
        "a": float(args.a),
        "supercell": supercell_diag,
        "mix_K": K,
        "mix_seed": seed,
        "sublattices": [sl.as_dict() for sl in sublats],
        "index": int(args.index),
        "model": str(args.model),
        "relax_cell": bool(args.relax_cell),
        "dtype": str(args.dtype),
        "category": str(args.category),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
