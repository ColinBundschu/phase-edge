import argparse
from phaseedge.orchestration.jobs.wang_landau import WangLandauSpec, run_wang_landau
from phaseedge.storage.ce_store import lookup_ce_by_key


def _parse_composition(s: str) -> dict[str, float]:
    comp: dict[str, float] = {}
    for pair in s.split(','):
        el, frac = pair.split(':')
        comp[el] = float(frac)                                         
    return comp


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run a basic Wang-Landau sampling")
    p.add_argument("--ce-key", required=True)
    p.add_argument("--composition", required=True, help="Comma-separated fractions like A:0.5,B:0.5")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--bin-size", type=float, default=1.0)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    comp = _parse_composition(args.composition)
    spec = WangLandauSpec(
        ce_key=args.ce_key,
        composition=comp,
        steps=args.steps,
        bin_size=args.bin_size,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    ce_doc = lookup_ce_by_key(args.ce_key)
    if ce_doc is None:
        raise SystemExit(f"Unknown CE key: {args.ce_key}")

    result = run_wang_landau(spec, ce_doc)
    print("run_key:", result["run_key"])
    print("dos_bins:", len(result["dos"]))
    print("n_samples:", len(result["samples"]))


if __name__ == "__main__":
    main()
