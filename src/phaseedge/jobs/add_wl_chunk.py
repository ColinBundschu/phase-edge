from typing import Any

from jobflow.core.job import job

from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.sampling.wl_chunk_driver import run_wl_chunk

@job
def add_wl_chunk(spec: WLSamplerSpec) -> dict[str, Any]:
    """Extend the WL chain by run_spec.steps, idempotently. Fails if not on tip."""
    return run_wl_chunk(spec=spec)
