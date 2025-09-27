from dataclasses import dataclass
from typing import Any, Mapping, TypedDict
import hashlib
from enum import Enum


# Current WL checkpoint bin_samples schema:
#   { "bin": int, "occ": list[int] }
class RefinedSample(TypedDict):
    bin: int
    occ: list[int]


class RefinedWLSamples(TypedDict):
    wl_key: str
    wl_checkpoint_key: str
    n_selected: int
    selected: list[RefinedSample]


class RefineStrategy(str, Enum):
    ENERGY_SPREAD = "energy_spread"
    ENERGY_STRATIFIED = "energy_stratified"
    HASH_ROUND_ROBIN = "hash_round_robin"


@dataclass(frozen=True)
class RefineOptions:
    """
    Options controlling how we down-select WL samples from a single checkpoint.

    n_total:
        Target number of samples to return. If None, returns all available.
    per_bin_cap:
        Max samples to take from any single bin. If None, unbounded.
    strategy:
        "energy_spread"      = choose bins evenly spaced across range (includes endpoints).
        "energy_stratified"  = round-robin across bins.
        "hash_round_robin"   = global order by occ-hash (ignores bins).
    """
    n_total: int | None = 25
    per_bin_cap: int | None = 5
    strategy: RefineStrategy = RefineStrategy.ENERGY_SPREAD


def _occ_hash(occ: list[int]) -> str:
    return hashlib.sha256(bytes(int(x) & 0xFF for x in occ)).hexdigest()


def _evenly_spaced_indices(nbins: int, k: int) -> list[int]:
    """
    Return k indices in [0, nbins-1], evenly spaced and deterministic, always including
    both endpoints when k >= 2. Uses integer ("Bresenham-style") remainder distribution
    to avoid any floating point rounding or platform-dependent behavior.

    Preconditions:
      - nbins > 0
      - k > 0
    Postconditions:
      - If k == 1 or nbins == 1 -> [0]
      - If 2 <= k <= nbins -> strictly increasing list covering [0, nbins-1]
    """
    if k <= 0 or nbins <= 0:
        return []
    if k == 1 or nbins == 1:
        return [0]

    # We only ever request at most one index per bin here.
    k = min(k, nbins)

    span = nbins - 1
    steps = k - 1
    # Integer division + remainder: we will have exactly `r` gaps of size (d+1); others size d.
    d, r = divmod(span, steps)

    idxs: list[int] = [0]
    x = 0
    err = 0
    for _ in range(steps):
        inc = d
        err += r
        if err >= steps:
            inc += 1
            err -= steps
        x += inc
        idxs.append(x)

    # Sanity: endpoints and monotonicity
    if idxs[0] != 0 or idxs[-1] != nbins - 1:
        raise AssertionError("Fencepost construction failed to include endpoints.")
    return idxs


def refine_wl_samples(
    block: Mapping[str, Any],
    *,
    options: RefineOptions = RefineOptions(),
) -> RefinedWLSamples:
    """
    Deterministically refine a single WL checkpoint (block) down to a subset of samples.

    Required checkpoint fields:
      - block["wl_key"]: str
      - block["hash"]: str
      - block["bin_samples"]: list[dict] with { "bin": int, "occ": list[int] }

    Behavior:
      - "energy_spread": pick bins evenly across min..max (fenceposts included), 1 sample per chosen bin.
                         If n_total > #bins (post-cap), fill remainder round-robin.
      - "energy_stratified": round-robin across bins from lowest bin upward.
      - "hash_round_robin": ignore bins; global order by occ-hash.
    """
    wl_key = str(block["wl_key"])
    ckpt_hash = str(block["hash"])

    raw = block.get("bin_samples")
    if not isinstance(raw, list):
        raise ValueError("block['bin_samples'] must be a list of {bin:int, occ:list[int]}")

    # Strict schema
    samples: list[RefinedSample] = []
    for rec in raw:
        if not isinstance(rec, dict) or "bin" not in rec or "occ" not in rec:
            raise ValueError("Each bin_samples entry must have keys 'bin' and 'occ'.")
        b = int(rec["bin"])
        occ_raw = rec["occ"]
        if not isinstance(occ_raw, list) or len(occ_raw) == 0:
            raise ValueError("Field 'occ' must be a non-empty list[int].")
        occ = [int(x) for x in occ_raw]
        samples.append(RefinedSample(bin=b, occ=occ))

    total_available = len(samples)
    if options.n_total is not None and total_available < int(options.n_total):
        raise ValueError(
            f"Checkpoint has only {total_available} samples; need n_total={options.n_total}."
        )

    # Group by bin
    by_bin: dict[int, list[RefinedSample]] = {}
    for s in samples:
        by_bin.setdefault(s["bin"], []).append(s)

    # Deterministic order within each bin: by occ-hash
    for b in by_bin:
        by_bin[b].sort(key=lambda s: _occ_hash(s["occ"]))

    # Optional per-bin cap
    if options.per_bin_cap is not None:
        cap = int(options.per_bin_cap)
        for b in list(by_bin.keys()):
            if len(by_bin[b]) > cap:
                by_bin[b] = by_bin[b][:cap]

    selected: list[RefinedSample] = []
    bins_sorted = sorted(by_bin.keys())

    if options.strategy == "energy_spread":
        # Choose k bins evenly spaced across the sorted bin list (include endpoints)
        k = (
            sum(len(v) for v in by_bin.values())
            if options.n_total is None
            else int(options.n_total)
        )
        # We can only choose at most one from each bin at this stage
        k_bins = min(k, len(bins_sorted))
        idxs = _evenly_spaced_indices(len(bins_sorted), k_bins)
        # take first (deterministic) sample from each chosen bin
        for idx in idxs:
            b = bins_sorted[idx]
            if by_bin[b]:
                selected.append(by_bin[b][0])

        # If we still need more, fill round-robin from remaining entries
        # NOTE: when n_total is None, k == total_available, so we continue until we've selected all.
        need = k - len(selected)
        if need > 0:
            chosen_bins = {bins_sorted[i] for i in idxs}
            cursors: dict[int, int] = {b: (1 if b in chosen_bins else 0) for b in bins_sorted}
            while len(selected) < k:
                progressed = False
                for b in bins_sorted:
                    i = cursors[b]
                    if i < len(by_bin[b]):
                        selected.append(by_bin[b][i])
                        cursors[b] = i + 1
                        progressed = True
                        if len(selected) == k:
                            break
                if not progressed:
                    break

    elif options.strategy == "energy_stratified":
        target = options.n_total if options.n_total is not None else sum(len(v) for v in by_bin.values())
        cursors: dict[int, int] = {b: 0 for b in bins_sorted}
        while len(selected) < int(target):
            progressed = False
            for b in bins_sorted:
                i = cursors[b]
                if i < len(by_bin[b]):
                    selected.append(by_bin[b][i])
                    cursors[b] = i + 1
                    progressed = True
                    if len(selected) == int(target):
                        break
            if not progressed:
                break

    else:  # "hash_round_robin"
        flat: list[RefinedSample] = []
        for b in bins_sorted:
            flat.extend(by_bin[b])
        flat.sort(key=lambda s: _occ_hash(s["occ"]))
        if options.n_total is None:
            selected = flat
        else:
            selected = flat[: options.n_total]

    if options.n_total is not None and len(selected) < options.n_total:
        raise ValueError(
            f"Could only select {len(selected)} of requested {options.n_total} samples."
        )

    return RefinedWLSamples(
        wl_key=wl_key,
        wl_checkpoint_key=ckpt_hash,
        n_selected=len(selected),
        selected=selected,
    )
