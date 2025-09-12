from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast, Final

from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow, JobOrder
from monty.json import MSONable

from phaseedge.jobs.ensure_ce import CEEnsureMixturesSpec, ensure_ce
from phaseedge.jobs.ensure_wl_samples import ensure_wl_samples
from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.utils.keys import compute_ce_key, compute_wl_key, canonical_counts


def _counts_sig(counts: Mapping[str, int]) -> str:
    c = canonical_counts(counts)
    return ",".join(f"{k}:{int(v)}" for k, v in c.items())


@dataclass(frozen=True, slots=True)
class EnsureWLSamplesFromCESpec(MSONable):
    ce_spec: CEEnsureMixturesSpec
    endpoints: Sequence[Mapping[str, int]]

    wl_bin_width: float
    wl_steps_to_run: int
    wl_samples_per_bin: int

    wl_step_type: str = "swap"
    wl_check_period: int = 5_000
    wl_update_period: int = 1
    wl_seed: int = 0

    # Single category for *everything* (wrapper, CE subflow, WL jobs)
    category: str = "gpu"

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "ce_spec": self.ce_spec.as_dict(),
            "endpoints": [canonical_counts(e) for e in self.endpoints],
            "wl_bin_width": float(self.wl_bin_width),
            "wl_steps_to_run": int(self.wl_steps_to_run),
            "wl_samples_per_bin": int(self.wl_samples_per_bin),
            "wl_step_type": str(self.wl_step_type),
            "wl_check_period": int(self.wl_check_period),
            "wl_update_period": int(self.wl_update_period),
            "wl_seed": int(self.wl_seed),
            "category": str(self.category),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EnsureWLSamplesFromCESpec":
        ce_spec = d.get("ce_spec")
        if not isinstance(ce_spec, CEEnsureMixturesSpec):
            ce_spec = CEEnsureMixturesSpec.from_dict(cast(Mapping[str, Any], ce_spec))
        return cls(
            ce_spec=cast(CEEnsureMixturesSpec, ce_spec),
            endpoints=cast(Sequence[Mapping[str, int]], d.get("endpoints", [])),
            wl_bin_width=float(d["wl_bin_width"]),
            wl_steps_to_run=int(d["wl_steps_to_run"]),
            wl_samples_per_bin=int(d["wl_samples_per_bin"]),
            wl_step_type=str(d.get("wl_step_type", "swap")),
            wl_check_period=int(d.get("wl_check_period", 5_000)),
            wl_update_period=int(d.get("wl_update_period", 1)),
            wl_seed=int(d.get("wl_seed", 0)),
            category=str(d.get("category", "gpu")),
        )


@job
def ensure_wl_samples_from_ce(spec: EnsureWLSamplesFromCESpec) -> Mapping[str, Any] | Response:
    # 1) Canonicalize mixture and inject endpoints (K=1, seed=0)
    ce_spec = spec.ce_spec
    canon_mix: list[dict[str, Any]] = []
    for elem in ce_spec.mixture:
        counts = canonical_counts(elem.get("counts", {}))
        K = int(elem.get("K", 0))
        seed = int(elem.get("seed", ce_spec.default_seed))
        if not counts or K <= 0:
            raise ValueError(f"Invalid mixture element: counts={counts}, K={K}")
        canon_mix.append({"counts": counts, "K": K, "seed": seed})

    endpoints_canon: list[dict[str, int]] = [canonical_counts(e) for e in spec.endpoints]
    for e in endpoints_canon:
        canon_mix.append({"counts": e, "K": 1, "seed": 0})

    # 2) Compute ce_key (deterministic, for transparency + wl_key computation)
    algo: Final = "randgen-3-comp-1"
    sources = [{"type": "composition", "elements": canon_mix}]
    ce_key = compute_ce_key(
        prototype=ce_spec.prototype,
        prototype_params=dict(ce_spec.prototype_params),
        supercell_diag=ce_spec.supercell_diag,
        replace_element=ce_spec.replace_element,
        sources=sources,
        model=ce_spec.model,
        relax_cell=ce_spec.relax_cell,
        dtype=ce_spec.dtype,
        basis_spec=dict(ce_spec.basis_spec),
        regularization=dict(ce_spec.regularization or {}),
        algo_version=algo,
        weighting=dict(ce_spec.weighting or {}),
    )

    # 3) Ensure CE; pass unified category into the CE spec so inner jobs adopt it
    ce_spec_for_run = CEEnsureMixturesSpec(
        prototype=ce_spec.prototype,
        prototype_params=dict(ce_spec.prototype_params),
        supercell_diag=ce_spec.supercell_diag,
        replace_element=ce_spec.replace_element,
        mixture=canon_mix,
        default_seed=int(ce_spec.default_seed),
        model=ce_spec.model,
        relax_cell=bool(ce_spec.relax_cell),
        dtype=ce_spec.dtype,
        basis_spec=dict(ce_spec.basis_spec),
        regularization=dict(ce_spec.regularization or {}),
        weighting=dict(ce_spec.weighting or {}) if ce_spec.weighting else None,
        category=str(spec.category),
    )
    j_ce: Job = ensure_ce(ce_spec_for_run)  # type: ignore[assignment]
    j_ce.name = "ensure_ce"
    j_ce.update_metadata({"_category": spec.category})

    # 4) Ensure WL once per unique, non-endpoint composition
    endpoint_fps = {_counts_sig(e) for e in endpoints_canon}
    seen: set[str] = set()

    wl_jobs: list[Job | Flow] = []
    wl_manifest: list[Mapping[str, Any]] = []
    wl_chunks: list[Mapping[str, Any]] = []  # minimal, safe fields only

    for elem in ce_spec.mixture:
        counts = canonical_counts(elem.get("counts", {}))
        if not counts:
            continue
        sig = _counts_sig(counts)
        if sig in endpoint_fps or sig in seen:
            seen.add(sig)
            continue
        seen.add(sig)

        wl_key = compute_wl_key(
            ce_key=ce_key,
            bin_width=float(spec.wl_bin_width),
            step_type=str(spec.wl_step_type),
            composition_counts=counts,
            check_period=int(spec.wl_check_period),
            update_period=int(spec.wl_update_period),
            seed=int(spec.wl_seed),
            algo_version="wl-grid-v1",
        )
        short = wl_key[:12]

        run_spec = WLSamplerSpec(
            wl_key=wl_key,
            ce_key=ce_key,  # pass the resolved string; barrier enforced by linear flow below
            bin_width=float(spec.wl_bin_width),
            steps=int(spec.wl_steps_to_run),
            composition_counts=counts,
            step_type=str(spec.wl_step_type),
            check_period=int(spec.wl_check_period),
            update_period=int(spec.wl_update_period),
            seed=int(spec.wl_seed),
            samples_per_bin=int(spec.wl_samples_per_bin),
        )

        j_wl: Job = ensure_wl_samples(run_spec)  # type: ignore[assignment]
        j_wl.name = f"ensure_wl_samples::{short}::{sig}"
        j_wl.update_metadata({"_category": spec.category, "wl_key": wl_key})
        wl_jobs.append(j_wl)

        wl_manifest.append({"counts": sig, "wl_key": wl_key})

        # IMPORTANT: expose only the fields we NEED as references.
        # Avoid referencing child outputs like "samples_per_bin" that may not be present
        # in add_wl_chunk job output.
        wl_chunks.append({
            "counts": sig,       # static
            "wl_key": wl_key,    # static
            "hash": j_wl.output["hash"],        # reference (exists for both found+new)
            # if you ever want to include samples_per_bin, set it to the POLICY value:
            # "samples_per_bin": int(spec.wl_samples_per_bin),
            # likewise for chunk_size:
            # "chunk_size": int(spec.wl_steps_to_run),
        })

    # WL jobs run in parallel after CE completes (linear barrier at outer level)
    flow_inner = Flow(wl_jobs, name="WL jobs (parallel)")
    flow = Flow([j_ce, flow_inner], name="Ensure WL samples from CE", order=JobOrder.LINEAR)

    out = {
        "ce_key": ce_key,
        "wl": wl_manifest,
        "wl_chunks": wl_chunks,          # <-- orchestrators use wl_chunks[i]['hash']
        "endpoints": sorted(endpoint_fps),
    }
    return Response(replace=flow, output=out)
