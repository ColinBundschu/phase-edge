"""
Job for ensuring that a Wang-Landau (WL) sampling chain contains at
least one chunk with a minimum number of steps and a minimum number of
samples captured per enthalpy bin.  If no existing checkpoint meets
the criteria, this job schedules a new WL chunk via the existing
``add_wl_chunk`` job.  Whether an existing checkpoint is found or a new
chunk is scheduled, the job output is the checkpoint document
containing the required samples.

This decision job is idempotent.  It never mutates state directly
(aside from optionally creating a new WL chunk), and if multiple
ensure jobs are submitted concurrently they will either find the same
qualifying checkpoint or schedule a single new chunk on the tip of
the chain.
"""

from typing import Any, Mapping

from jobflow.core.job import job, Response, Job

from phaseedge.schemas.wl import WLSamplerSpec
from phaseedge.jobs.add_wl_chunk import add_wl_chunk
from phaseedge.storage import store


@job
def ensure_wl_samples(
    spec: WLSamplerSpec,
) -> Mapping[str, Any] | Response:
    """Ensure a WL chain contains a chunk meeting the sampling criteria.

    This job searches the ``wang_landau_ckpt`` collection for the first
    checkpoint (by ascending ``step_end``) belonging to ``spec.wl_key``
    whose chunk size is at least ``spec.run_spec.steps`` and whose
    ``samples_per_bin`` runtime policy is at least
    ``spec.run_spec.samples_per_bin``.  If such a checkpoint is found,
    the checkpoint document is returned directly.  Otherwise, a new
    chunk is scheduled via the :func:`add_wl_chunk` job using the
    provided run specification.  In the latter case this job aliases
    its output to the result of the scheduled chunk.

    Parameters
    ----------
    spec
        The specification describing which WL chain to inspect and
        providing the parameters to schedule a new chunk if needed.

    Returns
    -------
    Mapping[str, Any] or Response
        Either the checkpoint document mapping when a qualifying
        checkpoint exists, or a :class:`~jobflow.core.job.Response`
        instructing Jobflow to replace this job with the scheduled
        :func:`add_wl_chunk` job.  In the latter case, this job's
        output is aliased to the output of the scheduled job.
    """
    # Build a read-only connection to the WL checkpoint collection
    coll = store.db_ro()["wang_landau_ckpt"]

    # Minimum requirements derived from the run spec
    min_steps: int = int(spec.steps)
    min_samples_per_bin: int = int(spec.samples_per_bin)

    # Compose a query: same wl_key, chunk length >= min_steps, samples >= min_samples_per_bin
    query: dict[str, Any] = {
        "wl_key": spec.wl_key,
        "chunk_size": {"$gte": min_steps},
        "samples_per_bin": {"$gte": min_samples_per_bin},
    }

    # Find the earliest qualifying checkpoint (ascending by step_end)
    doc = coll.find_one(query, sort=[("step_end", 1)])
    if doc:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        # Return the stored checkpoint as-is; Jobflow will serialize this mapping
        return dict(doc)

    # Instantiate the add_wl_chunk job; cast to Job for type hinting
    new_job: Job = add_wl_chunk(spec)

    # Forward the return value of the scheduled job as our own output
    return Response(replace=new_job, output=new_job.output)
