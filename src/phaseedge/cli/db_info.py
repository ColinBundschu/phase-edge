import argparse
from collections import Counter
from typing import Iterable

from phaseedge.storage import store


def _fmt_comp(d: dict[str, float]) -> str:
    # stable key order for readability
    return ",".join(f"{k}:{d[k]:.3f}" for k in sorted(d))


def _missing_indices(existing: Iterable[int], total: int) -> list[int]:
    have = set(existing)
    return [i for i in range(total) if i not in have]


def main() -> None:
    p = argparse.ArgumentParser(description="Show info about a PhaseEdge snapshot set")
    p.add_argument("--set-id", required=True, help="set_id to inspect")
    p.add_argument("--limit", type=int, default=5, help="show up to N example docs")
    args = p.parse_args()

    db = store.db_ro()

    # header (if present)
    header = db.snapshot_sets.find_one({"set_id": args.set_id}, {"_id": 0})
    if not header:
        print("No snapshot_set found for set_id:", args.set_id)
        raise SystemExit(1)

    print("== Snapshot Set ==")
    print(" set_id         :", args.set_id)
    print(" seed           :", header.get("seed"))
    print(" replace_element:", header.get("replace_element"))
    print(" supercell_diag :", header.get("supercell_diag"))
    print(" algo_version   :", header.get("algo_version"))
    comps = header.get("compositions") or []
    if comps:
        print(" compositions   :", "; ".join(_fmt_comp(c) for c in comps))
    print(" created_at     :", header.get("created_at"))

    # count + composition summary
    total = store.count_by_set(args.set_id)
    print("\n== Snapshots ==")
    print(" count          :", total)

    cursor = db.snapshots.find({"set_id": args.set_id}, {"_id": 0, "index": 1, "occ_key": 1, "composition": 1})
    idxs, examples, comp_counter = [], [], Counter()
    for i, doc in enumerate(cursor):
        idxs.append(doc["index"])
        # normalize comp keys order before counting
        comp_counter[_fmt_comp(doc["composition"])] += 1
        if i < args.limit:
            examples.append(doc)

    if examples:
        print(f"\n first {len(examples)} example(s):")
        for d in examples:
            print("  - index:", d["index"], "occ_key:", d["occ_key"], "comp:", _fmt_comp(d["composition"]))

    # gaps (if any)
    if total > 0:
        miss = _missing_indices(idxs, total)
        if miss:
            print("\n missing indices:", miss[:20], ("…" if len(miss) > 20 else ""))
        else:
            print("\n missing indices: none")

    # composition distribution
    if comp_counter:
        print("\n composition distribution:")
        for comp, n in comp_counter.most_common():
            print(f"  {comp} -> {n}")
