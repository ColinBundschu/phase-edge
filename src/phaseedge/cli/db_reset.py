#!/usr/bin/env python3
from __future__ import annotations

import sys
from typing import Iterable

from phaseedge.storage import store
from pymongo.errors import PyMongoError


def _list_collections() -> list[str]:
    db = store.db_rw()
    try:
        return sorted(db.list_collection_names())
    except PyMongoError as e:
        print(f"[status] error listing collections: {e}")
        return []


def _print_status(prefix: str) -> None:
    db = store.db_rw()
    cols = _list_collections()
    print(f"\n[status] {prefix} DB='{db.name}'")
    if cols:
        print(f"[status] Collections present ({len(cols)}): {', '.join(cols)}")
    else:
        print("[status] No collections present.")


def _drop_entire_db() -> None:
    db = store.db_rw()
    name = db.name
    client = db.client
    client.drop_database(name)
    print(f"[nuke] Dropped database: {name} (all collections + indexes)")


def main() -> None:
    # Final safety net. Remove this block if you want zero interaction.
    print("WARNING: This will DROP your entire PhaseEdge MongoDB database "
          "(including FireWorks and Jobflow collections) and ALL INDEXES.")
    confirm = input("Type 'YES' to proceed: ").strip()
    if confirm != "YES":
        print("Aborted.")
        sys.exit(1)

    _print_status("Before")
    _drop_entire_db()

    # Verify drop using the same client (fresh handle)
    db = store.db_rw()
    client = db.client
    remaining = [n for n in client.list_database_names() if n == db.name]
    if remaining:
        print(f"[warn] Database '{db.name}' still listed by server (may be cached).")
    else:
        print(f"[ok] Database '{db.name}' no longer listed by server.")

    _print_status("After")
    print("\n[done] Clean reset complete.")


if __name__ == "__main__":
    main()
