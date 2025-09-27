from jobflow.core.job import job
from jobflow.core.flow import Flow, JobOrder

from phaseedge.storage.wang_landau import WLBlockDoc
from phaseedge.schemas.wl_sampler_spec import WLSamplerSpec
from phaseedge.sampling.wl_block_driver import run_wl_block


@job(data=["bin_samples", "cation_counts"])
def add_wl_block(spec: WLSamplerSpec) -> WLBlockDoc:
    """Extend the WL chain by run_spec.steps, idempotently. Fails if not on tip."""
    return run_wl_block(spec=spec)


def add_wl_chain(spec: WLSamplerSpec, repeats: int) -> Flow:
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
        j = add_wl_block(spec)
        # Give each chunk a stable, distinguishable name; the caller may further
        # decorate names/metadata at submission time.
        j.name = f"add_wl_block[{i + 1}/{repeats}]"
        jobs.append(j)

    # Force linear execution order irrespective of data refs.
    return Flow(jobs=jobs, order=JobOrder.LINEAR, name=f"add_wl_chain x{repeats}")
