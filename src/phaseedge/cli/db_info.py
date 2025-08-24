#!/usr/bin/env python3
from __future__ import annotations

import argparse
from statistics import mean
from collections import Counter, defaultdict
from typing import Any

from phaseedge.storage import store


def _fmt_counts(d: dict[str, int] | None) -> str:
    if not d:
        return "<none>"
    return ",".join(f"{k}:{d[k]}" for k in sorted(d))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Show info about PhaseEdge MACE relax results for a set_id"
    )
    p.add_argument("--set-id", required=True, help="set_id to inspect")
    p.add_argument("--limit", type=int, default=5, help="show up to N example docs")
    args = p.parse_args()

    db = store.db_ro()
    coll = db["mace_relax"]

    # Fetch all docs for this set_id (projection keeps output readable)
    cursor = coll.find(
        {"set_id": args.set_id},
        {
            "_id": 0,
            "set_id": 1,
            "occ_key": 1,
            "model": 1,
            "relax_cell": 1,
            "dtype": 1,
            "energy": 1,
            "final_formula": 1,
            "details": 1,
            "success": 1,
        },
    )

    docs = list(cursor)
    if not docs:
        print("No MACE relax docs found for set_id:", args.set_id)
        raise SystemExit(1)

    print("== MACE Relax Results ==")
    print(" set_id         :", args.set_id)
    print(" total docs     :", len(docs))

    # Simple group-by summary on calc-key fields
    by_key = Counter((d.get("model"), d.get("relax_cell"), d.get("dtype")) for d in docs)
    print("\n by (model, relax_cell, dtype):")
    for k, n in by_key.most_common():
        model, relax_cell, dtype = k
        print(f"  ({model}, {relax_cell}, {dtype}) -> {n}")

    # Energies and sizes (when available)
    energies = [d["energy"] for d in docs if isinstance(d.get("energy"), (int, float))]
    sizes = []
    for d in docs:
        det = d.get("details") or {}
        ns = det.get("n_sites")
        if isinstance(ns, int):
            sizes.append(ns)

    if energies:
        print("\n energy stats   :",
              f"min={min(energies):.6f}, max={max(energies):.6f}, mean={mean(energies):.6f}")
    else:
        print("\n energy stats   : <none>")

    if sizes:
        by_n = Counter(sizes)
        print(" n_sites dist   :", ", ".join(f"{k}:{v}" for k, v in sorted(by_n.items())))
    else:
        print(" n_sites dist   : <none>")

    # Show a few examples
    print(f"\n first {min(args.limit, len(docs))} example(s):")
    for d in docs[: args.limit]:
        print(
            "  - occ_key:", d.get("occ_key"),
            "model:", d.get("model"),
            "relax_cell:", d.get("relax_cell"),
            "dtype:", d.get("dtype"),
            "energy:", d.get("energy"),
            "formula:", d.get("final_formula"),
        )


if __name__ == "__main__":
    main()
