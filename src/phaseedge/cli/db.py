import argparse
from phaseedge.storage import store


def main() -> None:
    p = argparse.ArgumentParser(description="Ping the PhaseEdge MongoDB")
    p.add_argument(
        "--mode",
        choices=["rw", "ro"],
        default="ro",
        help="Which connection to test: rw=read/write, ro=read-only (default)",
    )
    args = p.parse_args()

    if args.mode == "rw":
        db = store.db_rw()
        who = "RW"
    else:
        db = store.db_ro()
        who = "RO"

    try:
        info = db.command("ping")
        print(f"[{who}] Ping successful:", info)
        print("Database name:", db.name)
        print("Collections:", db.list_collection_names())
    except Exception as exc:
        print(f"[{who}] Ping failed:", exc)
        raise SystemExit(1)
