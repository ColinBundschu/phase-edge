import argparse
import sys
from typing import Literal

from phaseedge.storage import store


def _drop_phaseedge_collections() -> None:
    db = store.db_rw()
    names = set(db.list_collection_names())
    for col in ("mace_relax",):
        if col in names:
            db[col].drop()
            print(f"[drop] {db.name}.{col}")
        else:
            print(f"[skip] {db.name}.{col} (missing)")


def _reset_fireworks(launchpad_yaml: str) -> None:
    from fireworks import LaunchPad  # local import to be safe
    lp = LaunchPad.from_file(launchpad_yaml)
    lp.reset(password=None, require_password=False)
    dbname = getattr(lp, "name", getattr(lp, "fw_name", "<unknown>"))
    print(f"[fw] reset FireWorks collections in DB='{dbname}'")


def _drop_entire_db() -> None:
    db = store.db_rw()
    name = db.name
    db.client.drop_database(name)
    print(f"[nuke] dropped database: {name}")


def _print_counts() -> None:
    """Print existence + doc counts for PhaseEdge + FireWorks collections."""
    db = store.db_rw()
    names = set(db.list_collection_names())
    cols = [
        "mace_relax",
        "fireworks",
        "workflows",
        "launches",
        "outputs",
        "fw_id_assigner",
    ]
    print("\n[status] Collections present:", sorted(names))
    for c in cols:
        if c in names:
            try:
                n = db[c].count_documents({})
            except Exception as e:  # pragma: no cover
                print(f"[status] {c:15s} exists, count: <error: {e}>")
            else:
                print(f"[status] {c:15s} exists, count: {n}")
        else:
            print(f"[status] {c:15s} missing")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Reset PhaseEdge data and/or FireWorks state."
    )
    p.add_argument(
        "--mode",
        choices=["phaseedge", "fw", "both", "all"],
        default="both",
        help=(
            "phaseedge: drop only {mace_relax}; "
            "fw: FireWorks reset; "
            "both: do both (default); "
            "all: DROP ENTIRE DATABASE (no FW state survives)."
        ),
    )
    p.add_argument(
        "--launchpad",
        type=str,
        help="Path to my_launchpad.yaml (required for --mode fw or both).",
    )
    p.add_argument(
        "-y", "--yes", action="store_true", help="Do not prompt for confirmation."
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Show what would happen and exit."
    )
    args = p.parse_args()

    if args.mode in ("fw", "both") and not args.launchpad and args.mode != "all":
        p.error("--launchpad is required for --mode fw or --mode both")

    if not args.yes:
        print(f"About to run mode='{args.mode}'. This cannot be undone.")
        inp = input("Type 'yes' to continue: ").strip().lower()
        if inp != "yes":
            print("Aborted.")
            sys.exit(1)

    if args.dry_run:
        print("[dry-run] No changes made.")
        _print_counts()
        sys.exit(0)

    if args.mode == "phaseedge":
        _drop_phaseedge_collections()
    elif args.mode == "fw":
        _reset_fireworks(args.launchpad)  # type: ignore[arg-type]
    elif args.mode == "both":
        _drop_phaseedge_collections()
        _reset_fireworks(args.launchpad)  # type: ignore[arg-type]
    elif args.mode == "all":
        # all = drop entire DB (covers FW + PhaseEdge). No FW reset needed.
        _drop_entire_db()
    else:  # pragma: no cover
        p.error(f"unknown mode: {args.mode}")

    _print_counts()


if __name__ == "__main__":
    main()
