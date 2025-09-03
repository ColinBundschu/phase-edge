# ... existing imports ...
import argparse
from phaseedge.storage import store
from phaseedge.science.refine_wl import refine_wl_samples, RefineOptions

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wl-key", required=True)
    p.add_argument("--n-total", type=int, default=25)
    p.add_argument("--per-bin-cap", type=int, default=5)
    p.add_argument(
        "--strategy",
        choices=["energy_spread", "energy_stratified", "hash_round_robin"],
        default="energy_spread",            # <- ensure the default is spread here
    )
    p.add_argument("--debug", action="store_true")  # optional, see below
    args = p.parse_args()

    ckpt = store.db_rw()["wang_landau_ckpt"].find_one(
        {"wl_key": args.wl_key}, sort=[("step_end", -1)]
    )
    if not ckpt:
        raise SystemExit("No checkpoint found.")

    # optional: normalize _id for cleanliness
    if "_id" in ckpt:
        ckpt["_id"] = str(ckpt["_id"])

    opts = RefineOptions(
        n_total=args.n_total,
        per_bin_cap=args.per_bin_cap,
        strategy=args.strategy,             # <- forward user’s choice
    )

    out = refine_wl_samples(ckpt, options=opts)

    if args.debug:
        bins = [s["bin"] for s in out["selected"]]
        uniq_bins = sorted({s["bin"] for s in ckpt.get("bin_samples", [])})
        print(
            {
                "nbins_available": len(uniq_bins),
                "nbins_selected": len(set(bins)),
                "first_bin": bins[0] if bins else None,
                "last_bin": bins[-1] if bins else None,
                "selected_bins": bins,
            }
        )

    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
