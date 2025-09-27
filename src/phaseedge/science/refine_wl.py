from dataclasses import dataclass
from typing import Any, TypedDict
import hashlib
from enum import Enum


class RefinedSample(TypedDict):
    bin: int
    occ: list[int]


class RefineStrategy(str, Enum):
    ENERGY_SPREAD = "energy_spread"
    ENERGY_STRATIFIED = "energy_stratified"
    HASH_ROUND_ROBIN = "hash_round_robin"


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
    bin_samples: list[dict[str, Any]], # list[dict] with { "bin": int, "occ": list[int] }
    *,
    n_total: int,
    per_bin_cap: int | None,
    strategy: RefineStrategy,
) -> list[RefinedSample]:
    """
    Deterministically refine a single WL block down to a subset of samples.

    Behavior:
      - "energy_spread": pick bins evenly across min..max (fenceposts included), 1 sample per chosen bin.
                         If n_total > #bins (post-cap), fill remainder round-robin.
      - "energy_stratified": round-robin across bins from lowest bin upward.
      - "hash_round_robin": ignore bins; global order by occ-hash.
    """
    # Strict schema
    samples: list[RefinedSample] = []
    for rec in bin_samples:
        if not isinstance(rec, dict) or "bin" not in rec or "occ" not in rec:
            raise ValueError("Each bin_samples entry must have keys 'bin' and 'occ'.")
        b = int(rec["bin"])
        occ_raw = rec["occ"]
        if not isinstance(occ_raw, list) or len(occ_raw) == 0:
            raise ValueError("Field 'occ' must be a non-empty list[int].")
        occ = [int(x) for x in occ_raw]
        samples.append(RefinedSample(bin=b, occ=occ))

    total_available = len(samples)
    if total_available < n_total:
        raise ValueError(f"Block has only {total_available} samples; need n_total={n_total}.")

    # Group by bin
    by_bin: dict[int, list[RefinedSample]] = {}
    for s in samples:
        by_bin.setdefault(s["bin"], []).append(s)

    # Deterministic order within each bin: by occ-hash
    for b in by_bin:
        by_bin[b].sort(key=lambda s: _occ_hash(s["occ"]))

    # Optional per-bin cap
    if per_bin_cap is not None:
        for b in list(by_bin.keys()):
            if len(by_bin[b]) > per_bin_cap:
                by_bin[b] = by_bin[b][:per_bin_cap]

    selected: list[RefinedSample] = []
    bins_sorted = sorted(by_bin.keys())

    if strategy == RefineStrategy.ENERGY_SPREAD:
        # We can only choose at most one from each bin at this stage
        k_bins = min(n_total, len(bins_sorted))
        idxs = _evenly_spaced_indices(len(bins_sorted), k_bins)
        # take first (deterministic) sample from each chosen bin
        for idx in idxs:
            b = bins_sorted[idx]
            if by_bin[b]:
                selected.append(by_bin[b][0])

        # If we still need more, fill round-robin from remaining entries
        need = n_total - len(selected)
        if need > 0:
            chosen_bins = {bins_sorted[i] for i in idxs}
            cursors: dict[int, int] = {b: (1 if b in chosen_bins else 0) for b in bins_sorted}
            while len(selected) < n_total:
                progressed = False
                for b in bins_sorted:
                    i = cursors[b]
                    if i < len(by_bin[b]):
                        selected.append(by_bin[b][i])
                        cursors[b] = i + 1
                        progressed = True
                        if len(selected) == n_total:
                            break
                if not progressed:
                    break

    elif strategy == RefineStrategy.ENERGY_STRATIFIED:
        cursors: dict[int, int] = {b: 0 for b in bins_sorted}
        while len(selected) < n_total:
            progressed = False
            for b in bins_sorted:
                i = cursors[b]
                if i < len(by_bin[b]):
                    selected.append(by_bin[b][i])
                    cursors[b] = i + 1
                    progressed = True
                    if len(selected) == n_total:
                        break
            if not progressed:
                break

    elif strategy == RefineStrategy.HASH_ROUND_ROBIN:
        flat = [sample for b in bins_sorted for sample in by_bin[b][:per_bin_cap or len(by_bin[b])]]
        flat.sort(key=lambda s: _occ_hash(s["occ"]))
        selected = flat[:n_total]

    else:
        raise ValueError(f"Unrecognized RefineStrategy: {strategy!r}")

    if len(selected) != n_total:
        raise ValueError(f"Could only select {len(selected)} of requested {n_total} samples.")

    return selected
