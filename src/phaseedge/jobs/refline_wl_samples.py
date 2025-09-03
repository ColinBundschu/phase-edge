from __future__ import annotations

from typing import Any, Mapping
from jobflow.core.job import job

from phaseedge.science.refine_wl import RefineOptions, RefinedWLSamples, refine_wl_samples


def _normalize_id(doc: Mapping[str, Any]) -> dict[str, Any]:
    d = dict(doc)
    if "_id" in d:
        d["_id"] = str(d["_id"])
    return d


@job
def refine_wl_samples_job(
    block: Mapping[str, Any],
    *,
    n_total: int | None = 25,
    per_bin_cap: int | None = 5,
    strategy: str = "energy_spread",   # NEW default
) -> RefinedWLSamples:
    """
    Thin Jobflow wrapper around the pure refine_wl_samples function.
    """
    norm_block = _normalize_id(block)
    if strategy not in {"energy_spread", "energy_stratified", "hash_round_robin"}:
        strategy = "energy_spread"
    opts = RefineOptions(
        n_total=n_total,
        per_bin_cap=per_bin_cap,
        strategy=strategy,  # type: ignore[arg-type]
    )
    return refine_wl_samples(norm_block, options=opts)
