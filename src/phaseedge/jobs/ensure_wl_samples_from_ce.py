from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast, Final

from jobflow.core.job import job, Response, Job
from jobflow.core.flow import Flow, JobOrder
from monty.json import MSONable

from phaseedge.jobs.ensure_ce import CEEnsureMixturesSpec, ensure_ce
from phaseedge.jobs.ensure_wl_samples import ensure_wl_samples
from phaseedge.schemas.mixture import composition_counts_from_map, counts_sig, sorted_composition_maps
from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.utils.keys import compute_wl_key, canonical_counts


@dataclass(frozen=True, slots=True)
class EnsureWLSamplesFromCESpec(MSONable):
    ce_spec: CEEnsureMixturesSpec
    endpoints: tuple[dict[str, dict[str, int]], ...]

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
            "endpoints": list(self.endpoints),
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
        ce_spec = d["ce_spec"]
        if not isinstance(ce_spec, CEEnsureMixturesSpec):
            ce_spec = CEEnsureMixturesSpec.from_dict(ce_spec)
        return cls(
            ce_spec=cast(CEEnsureMixturesSpec, ce_spec),
            endpoints=sorted_composition_maps([{
                str(sublat): {str(k): int(v) for k, v in counts.items()}
                for sublat, counts in e.items()
            } for e in d["endpoints"]]),
            wl_bin_width=float(d["wl_bin_width"]),
            wl_steps_to_run=int(d["wl_steps_to_run"]),
            wl_samples_per_bin=int(d["wl_samples_per_bin"]),
            wl_step_type=str(d.get("wl_step_type", "swap")),
            wl_check_period=int(d.get("wl_check_period", 5_000)),
            wl_update_period=int(d.get("wl_update_period", 1)),
            wl_seed=int(d.get("wl_seed", 0)),
            category=str(d.get("category", "gpu")),
        )
    
    @property
    def sublattice_labels(self) -> tuple[str, ...]:
        all_labels = [tuple(sorted(mixture.composition_map.keys())) for mixture in self.ce_spec.mixtures]
        # If all the labels are not identical, raise an error
        first = all_labels[0]
        for labels in all_labels[1:]:
            if labels != first:
                raise ValueError("All mixtures must have the same sublattice labels.")
        return first
    
    @property
    def wl_key_composition_pairs(self) -> tuple[tuple[str, dict[str, int]], ...]:
        ce_key = self.ce_spec.ce_key
        pairs = []
        seen_sigs = {counts_sig(composition_counts_from_map(ep)) for ep in self.endpoints}
        for mixture in self.ce_spec.mixtures:
            composition_counts = composition_counts_from_map(mixture.composition_map)
            sig = counts_sig(composition_counts)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)

            wl_key = compute_wl_key(
                ce_key=ce_key,
                bin_width=self.wl_bin_width,
                step_type=self.wl_step_type,
                composition_counts=composition_counts,
                check_period=self.wl_check_period,
                update_period=self.wl_update_period,
                seed=self.wl_seed,
                algo_version="wl-grid-v1",
            )
            pairs.append((wl_key, composition_counts))
        return tuple(sorted(pairs))


@job
def ensure_wl_samples_from_ce(spec: EnsureWLSamplesFromCESpec) -> Mapping[str, Any] | Response:
    ce_key = spec.ce_spec.ce_key

    j_ce: Job = ensure_ce(spec.ce_spec)
    j_ce.name = "ensure_ce"
    j_ce.update_metadata({"_category": spec.category})

    wl_jobs: list[Job | Flow] = []
    wl_chunks: list[Mapping[str, Any]] = []  # minimal, safe fields only
    
    for wl_key, composition_counts in spec.wl_key_composition_pairs:
        sig = counts_sig(composition_counts)
        run_spec = WLSamplerSpec(
            wl_key=wl_key,
            ce_key=ce_key,  # pass the resolved string; barrier enforced by linear flow below
            bin_width=spec.wl_bin_width,
            steps=spec.wl_steps_to_run,
            sublattice_labels=spec.sublattice_labels,
            composition_counts=composition_counts,
            step_type=spec.wl_step_type,
            check_period=spec.wl_check_period,
            update_period=spec.wl_update_period,
            seed=spec.wl_seed,
            samples_per_bin=spec.wl_samples_per_bin,
        )

        j_wl: Job = ensure_wl_samples(run_spec)  # type: ignore[assignment]
        j_wl.name = f"ensure_wl_samples::{wl_key[:12]}::{sig}"
        j_wl.update_metadata({"_category": spec.category, "wl_key": wl_key})
        wl_jobs.append(j_wl)

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
        "wl_chunks": wl_chunks,          # <-- orchestrators use wl_chunks[i]['hash']
        "endpoints": sorted({counts_sig(composition_counts_from_map(ep)) for ep in spec.endpoints}),
    }
    return Response(replace=flow, output=out)
