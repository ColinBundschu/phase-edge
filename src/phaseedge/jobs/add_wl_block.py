from jobflow.core.job import job, Job
from jobflow.core.flow import Flow, JobOrder

from phaseedge.jobs.store_ce_model import lookup_ce_by_key, rehydrate_ensemble_by_ce_key
from phaseedge.science.prototype_spec import PrototypeSpec
from phaseedge.storage.wang_landau import WLBlockDoc, fetch_wl_tip, verify_wl_output_indexes
from phaseedge.schemas.wl_sampler_spec import WLSamplerSpec
from phaseedge.sampling.wl_block_driver import run_wl_block


@job(data=["bin_samples", "cation_counts"])
def _add_wl_block_job(spec: WLSamplerSpec) -> WLBlockDoc:
    """Extend the WL chain by run_spec.steps, idempotently. Fails if not on tip."""
    verify_wl_output_indexes()
    tip = fetch_wl_tip(spec.wl_key)
    ensemble = rehydrate_ensemble_by_ce_key(spec.ce_key)
    ce_doc = lookup_ce_by_key(spec.ce_key)
    if not ce_doc:
        raise RuntimeError(f"No CE found for ce_key={spec.ce_key}")

    prototype_spec = PrototypeSpec.from_dict(ce_doc["prototype_spec"])
    sx, sy, sz = (int(x) for x in ce_doc["supercell_diag"])
    supercell_diag=(sx, sy, sz)
    return run_wl_block(spec=spec, ensemble=ensemble, tip=tip, prototype_spec=prototype_spec, supercell_diag=supercell_diag)


def add_wl_block(
    spec: WLSamplerSpec,
    *,
    name: str,
    category: str,
) -> Job:
    """
    Create a Job that extends the WL chain and automatically sets Job.name and metadata.
    """
    j: Job = _add_wl_block_job(spec)
    j.name = name
    j.update_metadata({"_category": category, "wl_key": spec.wl_key})
    return j


def add_wl_chain(*, spec: WLSamplerSpec, base_name: str, repeats: int, category: str) -> Flow:
    """
    Build a *sequential* (linear-ordered) Flow of WL chunks.

    Parameters
    ----------
    spec
        The WLSamplerSpec to use for each chunk.
    repeats
        Number of sequential chunks to append to the chain. Must be > 0.

    Notes
    -----
    The individual block jobs do not share explicit data dependencies, but the
    WL checkpointing requires that each subsequent chunk starts from the
    previous chunk's end. We enforce strict sequencing via JobOrder.LINEAR.

    Returns
    -------
    Flow
        A Flow with `repeats` add_wl_block jobs, executed strictly in order.
    """
    if repeats <= 0:
        raise ValueError("repeats must be a positive integer")

    jobs: list = []
    for i in range(repeats):
        j = add_wl_block(spec, name=base_name, category=category)
        j.name += f"::chunk[{i + 1}/{repeats}]"
        jobs.append(j)

    # Force linear execution order irrespective of data refs.
    return Flow(jobs=jobs, order=JobOrder.LINEAR, name=f"add_wl_chain x{repeats}")
